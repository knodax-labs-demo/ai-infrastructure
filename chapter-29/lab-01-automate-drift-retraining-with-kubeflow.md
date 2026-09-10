# Hands-On Lab: Automate Drift Retraining with Kubeflow

Production machine learning models can degrade as real-world data changes over time. In this lab, you will build an automated **drift-driven retraining workflow with Kubeflow Pipelines (KFP)**.

The workflow detects changes between reference and current production data, determines whether retraining is required, trains a new candidate model, evaluates the candidate against a promotion threshold, and deploys it only when the required quality gate is satisfied.

The lab focuses on the orchestration pattern rather than model complexity. A lightweight scikit-learn classifier is used so that the infrastructure concepts remain clear.

The complete workflow is:

```text
Reference Data
      │
      ├──────────────┐
      │              │
      ▼              ▼
Current Data ──► Drift Detection
                       │
                 Drift Detected?
                    │     │
                   No    Yes
                    │     │
                  Stop    ▼
                     Import Training Data
                            ↓
                         Retrain
                            ↓
                     Candidate Model
                            ↓
                         Evaluate
                            ↓
                      Quality Gate
                       │         │
                     Fail       Pass
                       │         │
                     Stop        ▼
                              Deploy
```

---

## Lab Objectives

By completing this lab, you will learn how to:

* Detect data drift between reference and production datasets.
* Build reusable Kubeflow Pipeline components.
* Return drift decisions as pipeline outputs.
* Trigger retraining conditionally.
* Pass datasets and models as KFP artifacts.
* Evaluate a candidate model against a quality gate.
* Deploy only models that satisfy promotion criteria.
* Use conditional control flow with `dsl.If`.
* Track model and dataset lineage.
* Understand the architecture of closed-loop retraining systems.

---

## Estimated Time

**Approximately 120–180 minutes**

---

## Tools

This lab uses:

* Kubernetes
* Kubeflow Pipelines
* KFP SDK
* Python
* pandas
* SciPy
* scikit-learn
* joblib
* KFP `Dataset` and `Model` artifacts

A GPU is not required.

---

# Step 1: Verify the Prerequisites

You need access to a **Kubeflow Pipelines v2-compatible environment** with permission to create pipeline runs.

The environment should also provide artifact storage such as:

```text
MinIO
Amazon S3
Google Cloud Storage
Other KFP-Compatible Object Storage
```

Install the KFP SDK:

```bash
pip install --upgrade kfp
```

Verify:

```bash
python -c "import kfp; print(kfp.__version__)"
```

---

# Step 2: Understand the Closed-Loop MLOps Pattern

Traditional training is often linear:

```text
Data
 ↓
Train
 ↓
Evaluate
 ↓
Deploy
```

A production retraining system adds feedback:

```text
Production Model
       ↓
Production Data
       ↓
Monitoring
       ↓
Drift Detection
       ↓
Retraining Decision
       ↓
New Candidate
       ↓
Evaluation
       ↓
Promotion Decision
       ↓
Deployment
       ↓
Production Model
```

This creates a **closed-loop MLOps system**.

---

# Step 3: Prepare the Example Data

Create three datasets:

```text
ref.csv
train.csv
test.csv
```

Their roles are:

| Dataset     | Purpose                        |
| ----------- | ------------------------------ |
| `ref.csv`   | Reference distribution         |
| `train.csv` | Latest labeled retraining data |
| `test.csv`  | Candidate evaluation data      |

Each should contain numeric feature columns plus:

```text
label
```

for binary classification.

In production, these datasets might come from:

```text
Object Storage
Data Warehouse
Feature Store
Streaming Pipeline
Lakehouse
```

---

# Step 4: Understand Drift Detection

The drift detector compares:

```text
Reference Distribution
        vs.
Current Distribution
```

The lab uses the **Kolmogorov-Smirnov test** for numeric features.

Conceptually:

```text
Feature
   ↓
Reference Samples
        vs.
Current Samples
   ↓
KS Test
   ↓
p-value
   ↓
Drifted?
```

For this lab:

```text
p-value < 0.05
```

means the feature is counted as drifted.

---

# Step 5: Understand the Pipeline Decision Output

A key architectural principle is:

```text
Drift Detector
      ↓
Boolean Output
      ↓
Pipeline Condition
```

Do **not** use:

```text
Exit Code 0 / 1
```

to represent whether drift occurred.

Process failure and business decision are different concepts.

Instead:

```text
Successful Component
      ↓
drift_detected = true / false
```

---

# Step 6: Create the Pipeline File

Create:

```text
lab-07-drift-retrain-pipeline.py
```

Start with:

```python
from typing import NamedTuple

from kfp import compiler, dsl
from kfp.dsl import (
    Dataset,
    Input,
    Model,
    Output,
)
```

---

# Step 7: Create the Drift Detection Component

Add:

```python
@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "scipy",
    ],
)
def detect_drift(
    reference_path: str,
    current_path: str,
    threshold: float = 0.1,
) -> bool:
    import pandas as pd
    from scipy.stats import ks_2samp

    reference = pd.read_csv(
        reference_path
    )

    current = pd.read_csv(
        current_path
    )

    feature_columns = [
        column
        for column in reference.columns
        if column != "label"
        and column in current.columns
    ]

    drifted_features = 0
    evaluated_features = 0

    for feature in feature_columns:

        if (
            pd.api.types.is_numeric_dtype(
                reference[feature]
            )
            and
            pd.api.types.is_numeric_dtype(
                current[feature]
            )
        ):
            reference_values = (
                reference[feature]
                .dropna()
            )

            current_values = (
                current[feature]
                .dropna()
            )

            if (
                len(reference_values) == 0
                or
                len(current_values) == 0
            ):
                continue

            _, p_value = ks_2samp(
                reference_values,
                current_values,
            )

            evaluated_features += 1

            if p_value < 0.05:
                drifted_features += 1

    if evaluated_features == 0:
        raise ValueError(
            "No compatible numeric features "
            "were available for drift detection."
        )

    drift_share = (
        drifted_features
        / evaluated_features
    )

    print(
        f"Drifted features: "
        f"{drifted_features}/"
        f"{evaluated_features}"
    )

    print(
        f"Drift share: "
        f"{drift_share:.3f}"
    )

    return bool(
        drift_share > threshold
    )
```

---

# Step 8: Understand the Drift Metric

The detector calculates:

```text
Drift Share
=
Number of Drifted Features
──────────────────────────
Number of Evaluated Features
```

For example:

```text
10 numeric features
3 detected as drifted

Drift Share = 3 / 10 = 0.30
```

If:

```text
threshold = 0.10
```

then:

```text
0.30 > 0.10
```

and retraining is triggered.

---

# Step 9: Understand Statistical vs. Operational Drift

A statistical difference does not automatically mean the model should be retrained.

Production systems should consider:

```text
Statistical Drift
+
Feature Importance
+
Model Performance
+
Sample Size
+
Historical Behavior
+
Business Impact
```

The KS test in this lab is intentionally simplified.

---

# Step 10: Create a Dataset Import Component

Add:

```python
@dsl.component(
    base_image="python:3.11",
)
def import_dataset(
    source_path: str,
    dataset: Output[Dataset],
):
    import shutil

    shutil.copyfile(
        source_path,
        dataset.path,
    )
```

This converts an external file path into a KFP-managed `Dataset` artifact.

---

# Step 11: Understand Why Dataset Import Is Separate

The architecture becomes:

```text
External Dataset
       ↓
Import Component
       ↓
KFP Dataset Artifact
       ↓
Training / Evaluation
```

This is preferable to assuming that all pipeline pods share the same local filesystem.

---

# Step 12: Create the Training Component

Add:

```python
@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
def train_model(
    training_data: Input[Dataset],
    model: Output[Model],
):
    import joblib
    import pandas as pd

    from sklearn.linear_model import (
        LogisticRegression
    )

    df = pd.read_csv(
        training_data.path
    )

    X = df.drop(
        columns=["label"]
    )

    y = df["label"]

    classifier = (
        LogisticRegression(
            max_iter=500
        )
    )

    classifier.fit(
        X,
        y,
    )

    joblib.dump(
        classifier,
        model.path,
    )

    model.metadata[
        "framework"
    ] = "scikit-learn"

    model.metadata[
        "model_type"
    ] = "LogisticRegression"
```

---

# Step 13: Understand the Training Artifact

The component produces:

```text
Training Dataset
      ↓
Logistic Regression
      ↓
Serialized Candidate
      ↓
KFP Model Artifact
```

The artifact can then move into:

```text
Evaluation
Registration
Deployment
```

without relying on temporary container storage.

---

# Step 14: Add Model Metadata

The lab records:

```text
framework = scikit-learn
model_type = LogisticRegression
```

Production metadata might also contain:

```text
Dataset Version
Git Commit
Feature Version
Training Timestamp
Hyperparameters
Metrics
Model Owner
```

This improves lineage.

---

# Step 15: Define Named Evaluation Outputs

Use explicit output names rather than relying on generated names such as:

```text
Output-0
Output-1
```

Add:

```python
class EvaluationOutputs(
    NamedTuple
):
    accuracy: float
    passed: bool
```

This makes downstream conditions significantly easier to understand.

---

# Step 16: Create the Evaluation Component

Add:

```python
@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
def evaluate_model(
    model: Input[Model],
    test_data: Input[Dataset],
    accuracy_threshold: float,
) -> EvaluationOutputs:

    import joblib
    import pandas as pd

    from sklearn.metrics import (
        accuracy_score
    )

    classifier = joblib.load(
        model.path
    )

    test = pd.read_csv(
        test_data.path
    )

    X = test.drop(
        columns=["label"]
    )

    y = test["label"]

    predictions = (
        classifier.predict(
            X
        )
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    passed = (
        accuracy
        >= accuracy_threshold
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Promotion gate passed: "
        f"{passed}"
    )

    return EvaluationOutputs(
        float(accuracy),
        bool(passed),
    )
```

---

# Step 17: Understand the Promotion Gate

The evaluation component produces:

```text
accuracy
passed
```

The flow is:

```text
Candidate Model
      +
Test Dataset
      ↓
Evaluation
      ↓
Accuracy
      ↓
Threshold Check
      ↓
passed = true / false
```

This separates:

```text
Model Quality
```

from:

```text
Pipeline Execution Success
```

---

# Step 18: Understand Why an Absolute Threshold Is Simplified

This lab uses:

```text
Candidate Accuracy >= Threshold
```

A stronger production rule is:

```text
Candidate Model
      vs.
Current Production Model
```

For example:

```text
Candidate Accuracy
    >=
Champion Accuracy

AND

Latency <= Limit

AND

Fairness >= Requirement

AND

Robustness >= Requirement
```

A candidate should not automatically replace the current model simply because it exceeds a static number.

---

# Step 19: Create the Deployment Component

Add:

```python
@dsl.component(
    base_image="python:3.11",
)
def deploy_model(
    model: Input[Model],
):
    print(
        "Candidate model passed validation. "
        "Beginning deployment."
    )

    print(
        f"Model artifact: "
        f"{model.path}"
    )
```

This is intentionally a placeholder.

A production implementation might instead call:

```text
KServe
Model Registry
Deployment API
GitOps Pipeline
Internal Serving Platform
```

---

# Step 20: Understand the Two-Level Conditional Flow

Deployment should occur only when:

```text
Condition 1:
Drift Detected

AND

Condition 2:
Candidate Passed Evaluation
```

Conceptually:

```text
Drift?
 │
 ├── No → Stop
 │
 └── Yes
       ↓
     Train
       ↓
    Evaluate
       ↓
     Pass?
      │
      ├── No → Stop
      │
      └── Yes → Deploy
```

---

# Step 21: Define the Kubeflow Pipeline

Add:

```python
@dsl.pipeline(
    name="drift-retrain-pipeline",
    description=(
        "Detect drift, retrain a model, "
        "evaluate it, and conditionally "
        "deploy the candidate."
    ),
)
def drift_retrain_pipeline(
    reference_path: str,
    current_path: str,
    training_path: str,
    test_path: str,
    drift_threshold: float = 0.1,
    accuracy_threshold: float = 0.85,
):

    drift_task = detect_drift(
        reference_path=reference_path,
        current_path=current_path,
        threshold=drift_threshold,
    )

    with dsl.If(
        drift_task.output == True,
        name="drift-detected",
    ):

        train_data_task = (
            import_dataset(
                source_path=(
                    training_path
                )
            )
        )

        test_data_task = (
            import_dataset(
                source_path=(
                    test_path
                )
            )
        )

        train_task = train_model(
            training_data=(
                train_data_task
                .outputs[
                    "dataset"
                ]
            )
        )

        evaluation_task = (
            evaluate_model(
                model=(
                    train_task
                    .outputs[
                        "model"
                    ]
                ),

                test_data=(
                    test_data_task
                    .outputs[
                        "dataset"
                    ]
                ),

                accuracy_threshold=(
                    accuracy_threshold
                ),
            )
        )

        with dsl.If(
            evaluation_task
            .outputs["passed"]
            == True,

            name="model-approved",
        ):
            deploy_model(
                model=(
                    train_task
                    .outputs[
                        "model"
                    ]
                )
            )
```

---

# Step 22: Understand the Pipeline DAG

The resulting workflow is:

```text
                    ┌──────────────────┐
                    │   Detect Drift   │
                    └────────┬─────────┘
                             │
                        Drift > Limit?
                          │       │
                         No      Yes
                          │       │
                        Stop      ▼
                    ┌──────────────────┐
                    │ Import Train Data│
                    └────────┬─────────┘
                             │
                             ▼
                         ┌───────┐
                         │ Train │
                         └───┬───┘
                             │
                           Model
                             │
                 ┌───────────┴───────────┐
                 │                       │
            Import Test Data             │
                 │                       │
                 └───────────┬───────────┘
                             ▼
                        ┌──────────┐
                        │ Evaluate │
                        └────┬─────┘
                             │
                        Quality Gate
                          │       │
                        Fail     Pass
                          │       │
                        Stop      ▼
                             ┌────────┐
                             │ Deploy │
                             └────────┘
```

---

# Step 23: Understand Conditional Execution

Without conditional execution:

```text
Monitor
 ↓
Train
 ↓
Evaluate
 ↓
Deploy
```

would run every time.

With conditions:

```text
Only Retrain When Needed
```

and:

```text
Only Deploy When Approved
```

This reduces:

* Unnecessary compute
* Unnecessary model churn
* Deployment risk
* Infrastructure cost

---

# Step 24: Compile the Pipeline

Add:

```python
if __name__ == "__main__":

    compiler.Compiler().compile(
        pipeline_func=(
            drift_retrain_pipeline
        ),

        package_path=(
            "drift_retrain_pipeline.yaml"
        ),
    )
```

Run:

```bash
python \
  lab-07-drift-retrain-pipeline.py
```

Expected:

```text
drift_retrain_pipeline.yaml
```

---

# Step 25: Understand Compilation

The pipeline source defines:

```text
Components
+
Parameters
+
Artifacts
+
Dependencies
+
Conditions
```

The compiler produces:

```text
Python Pipeline
      ↓
KFP Compiler
      ↓
Pipeline YAML
```

The YAML can then be submitted to the Kubeflow Pipelines backend.

---

# Step 26: Run the Pipeline

Upload:

```text
drift_retrain_pipeline.yaml
```

to Kubeflow Pipelines.

Provide parameters such as:

```text
drift_threshold = 0.10
accuracy_threshold = 0.85
```

and valid dataset locations.

---

# Step 27: Test the No-Drift Path

First provide a current dataset with a distribution similar to the reference dataset.

Expected flow:

```text
Detect Drift
     ↓
False
     ↓
Retraining Branch Skipped
```

Verify that:

```text
Training = Skipped
Evaluation = Skipped
Deployment = Skipped
```

This is an important test of control-flow correctness.

---

# Step 28: Test the Drift-Detected Path

Next, provide current data with intentionally shifted feature distributions.

Expected:

```text
Detect Drift
     ↓
True
     ↓
Import Data
     ↓
Train
     ↓
Evaluate
```

Whether deployment runs depends on the evaluation gate.

---

# Step 29: Test Evaluation Failure

Set the accuracy threshold high enough that the model fails.

For example:

```text
accuracy_threshold = 0.99
```

Expected:

```text
Drift Detected
      ↓
Retrain
      ↓
Evaluate
      ↓
passed = false
      ↓
Deployment Skipped
```

---

# Step 30: Test Evaluation Success

Use an achievable threshold.

Expected:

```text
Drift Detected
      ↓
Retrain
      ↓
Evaluate
      ↓
passed = true
      ↓
Deploy
```

This validates both conditional gates.

---

# Step 31: Understand Threshold Calibration

The values:

```text
0.10
0.85
```

are demonstration parameters.

Production drift thresholds should consider:

```text
Historical Data
Feature Importance
Sample Size
Natural Seasonality
Model Sensitivity
Business Impact
```

Production promotion thresholds should consider:

```text
Champion Performance
Candidate Performance
Fairness
Robustness
Latency
Cost
Operational Risk
```

---

# Step 32: Examine Pipeline Lineage

Open the completed run in the Kubeflow Pipelines UI.

Inspect:

* Drift component execution
* Input datasets
* Training dataset artifact
* Candidate model artifact
* Evaluation outputs
* Conditional branches
* Deployment execution

The lineage becomes:

```text
Reference Data
     +
Current Data
     ↓
Drift Decision
     ↓
Training Dataset
     ↓
Candidate Model
     ↓
Evaluation
     ↓
Promotion Decision
     ↓
Deployment
```

---

# Step 33: Understand Why Lineage Matters

A production system should be able to answer:

```text
Why was this model retrained?
Which drift event triggered it?
Which dataset trained it?
Which code version produced it?
What metrics approved it?
Who or what authorized deployment?
```

Without lineage, automated retraining becomes difficult to audit.

---

# Step 34: Save Drift Metrics as Artifacts

A production drift component should ideally produce more than:

```text
true / false
```

It should also record:

```text
Drift Score
Drifted Features
Feature-Level Statistics
Sample Counts
Reference Dataset Version
Current Dataset Version
Timestamp
```

This provides evidence for the retraining decision.

---

# Step 35: Add Candidate-vs-Champion Evaluation

Extend the architecture:

```text
Candidate Model
      +
Champion Model
      +
Evaluation Dataset
      ↓
Compare
      ↓
Candidate Better?
```

The gate might become:

```text
Candidate Accuracy >= Champion Accuracy
```

rather than:

```text
Candidate Accuracy >= Static Threshold
```

---

# Step 36: Add Multiple Evaluation Gates

A stronger promotion architecture might require:

```text
Accuracy Gate
     AND
Fairness Gate
     AND
Latency Gate
     AND
Robustness Gate
     AND
Cost Gate
```

Only then:

```text
Promote Candidate
```

---

# Step 37: Add a Model Registry

A production flow can insert registration before deployment:

```text
Train
 ↓
Evaluate
 ↓
Approved
 ↓
Register Candidate
 ↓
Approve Version
 ↓
Deploy
```

The registry can track:

* Version
* Metrics
* Dataset
* Model metadata
* Approval status
* Deployment history

---

# Step 38: Replace Deployment with KServe

The placeholder:

```text
deploy_model()
```

can later become:

```text
KServe InferenceService
```

The architecture becomes:

```text
Approved Model
      ↓
KServe Deployment
      ↓
Inference Endpoint
      ↓
Production Traffic
```

---

# Step 39: Add a Human Approval Gate

High-risk production systems may require:

```text
Automated Evaluation
       ↓
Technical Approval
       ↓
Human Review
       ↓
Deployment
```

This prevents fully autonomous deployment where governance requirements demand human oversight.

---

# Step 40: Add Canary Deployment

Instead of moving immediately to full production traffic:

```text
Candidate
    ↓
Canary Deployment
    ↓
Small Traffic %
    ↓
Observe Metrics
    ↓
Healthy?
   │    │
  No   Yes
   │    │
Rollback  Increase Traffic
```

This reduces deployment risk.

---

# Step 41: Add Automated Rollback

The closed loop can extend beyond deployment:

```text
Deploy Candidate
      ↓
Monitor Production
      ↓
Performance Regression?
      │
     Yes
      ↓
Rollback
```

The result is a more complete production lifecycle.

---

# Step 42: Trigger the Pipeline Automatically

The current lab runs the pipeline manually.

Production triggers may include:

```text
Schedule
Kafka Event
Monitoring Alert
New Dataset
Feature Store Update
External Workflow
```

The architecture becomes:

```text
Monitoring System
      ↓
Trigger
      ↓
Kubeflow Retraining Pipeline
```

---

# Step 43: Avoid Retraining on Every Drift Event

A production system should usually avoid:

```text
Any Drift
   ↓
Immediately Retrain
```

A stronger decision combines:

```text
Drift
+
Model Degradation
+
Sufficient New Data
+
Business Impact
+
Minimum Retraining Interval
```

Only then:

```text
Retraining Candidate
```

This reduces unnecessary retraining.

---

# Step 44: Add Model Performance Monitoring

Data drift does not always cause performance degradation.

Therefore monitor:

```text
Data Drift
+
Prediction Drift
+
Model Performance
```

A stronger retraining decision can be:

```text
Data Drift
     AND/OR
Performance Degradation
        ↓
Retraining Decision
```

---

# Step 45: Add Feature-Level Drift Importance

Not every feature matters equally.

For example:

```text
Critical Feature Drift
       >
Low-Importance Feature Drift
```

A production drift system may weight:

```text
Drift Magnitude
×
Feature Importance
```

to prioritize impactful changes.

---

# Step 46: Understand Closed-Loop Governance

Automation does not eliminate governance.

The complete system may include:

```text
Monitoring
   ↓
Drift Detection
   ↓
Retraining
   ↓
Evaluation
   ↓
Governance Gate
   ↓
Registration
   ↓
Deployment
   ↓
Production Monitoring
   ↓
Rollback
```

This connects MLOps automation with operational control.

---

# Step 47: Production Architecture

A mature enterprise design may look like:

```text
                   Production Data
                         ↓
                  Monitoring System
                         ↓
                   Drift Detector
                         ↓
                 Retraining Trigger
                         ↓
                  Kubeflow Pipeline
                         ↓
                    Data Snapshot
                         ↓
                      Training
                         ↓
                  Candidate Model
                         ↓
                     Evaluation
                         ↓
                   Quality Gates
                         ↓
                  Model Registry
                         ↓
                      Approval
                         ↓
                       KServe
                         ↓
                 Production Traffic
                         ↓
                 Runtime Monitoring
                         ↓
                Rollback / Retraining
```

---

# Step 48: Additional Challenges

Extend the lab by:

* Replacing the KS-based detector with Evidently or another monitoring framework.
* Adding categorical-feature drift.
* Logging feature-level drift metrics.
* Saving drift reports as artifacts.
* Adding candidate-vs-champion comparison.
* Adding precision, recall, and F1.
* Adding fairness evaluation.
* Adding latency and cost gates.
* Registering models before deployment.
* Replacing the deployment placeholder with KServe.
* Adding human approval.
* Implementing canary deployment.
* Adding automatic rollback.
* Triggering the pipeline from Kafka.
* Scheduling periodic monitoring.
* Integrating Katib for hyperparameter tuning.
* Recording Git commit and dataset-version metadata.

---

# Step 49: Recommended Repository Structure

For a standalone lab:

```text
drift-retraining-lab/
├── drift_retrain_pipeline.py
├── drift_retrain_pipeline.yaml
├── ref.csv
├── train.csv
├── test.csv
└── current.csv
```

For your book companion repository:

```text
chapter-30/
├── lab-07-automate-drift-retraining-with-kubeflow.md
├── lab-07-drift-retrain-pipeline.py
└── lab-07-generate-sample-data.py
```

I would normally treat:

```text
drift_retrain_pipeline.yaml
```

as a generated artifact because it can be regenerated from the Python pipeline source.

The Python file should remain the authoritative source.

---

# Step 50: Clean Up

Remove local generated YAML if desired:

```bash
rm -f \
  drift_retrain_pipeline.yaml
```

Delete test runs from the Kubeflow Pipelines UI when they are no longer required.

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Accessed a KFP v2-compatible environment.
* [ ] Installed the KFP SDK.
* [ ] Prepared reference data.
* [ ] Prepared current production-style data.
* [ ] Prepared labeled training data.
* [ ] Prepared separate test data.
* [ ] Created the drift detector.
* [ ] Returned the drift decision as a Boolean output.
* [ ] Calculated the share of drifted numeric features.
* [ ] Created the dataset import component.
* [ ] Created KFP dataset artifacts.
* [ ] Created the training component.
* [ ] Produced a KFP model artifact.
* [ ] Added model metadata.
* [ ] Created named evaluation outputs.
* [ ] Evaluated candidate accuracy.
* [ ] Implemented the promotion gate.
* [ ] Created the deployment component.
* [ ] Added the drift conditional.
* [ ] Added the model-approval conditional.
* [ ] Compiled the pipeline.
* [ ] Generated the YAML specification.
* [ ] Tested the no-drift path.
* [ ] Tested the drift-detected path.
* [ ] Tested evaluation failure.
* [ ] Tested evaluation success.
* [ ] Inspected pipeline lineage.
* [ ] Reviewed candidate-vs-champion evaluation.
* [ ] Reviewed model registration.
* [ ] Reviewed canary deployment and rollback.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Explain drift-driven retraining architecture.
* Distinguish monitoring signals from pipeline failures.
* Build reusable KFP components.
* Use Boolean component outputs for pipeline control flow.
* Use `dsl.If` for conditional execution.
* Pass datasets and models through KFP artifacts.
* Train a candidate model conditionally.
* Implement a promotion gate.
* Deploy only approved candidates.
* Explain why absolute thresholds are insufficient for many production systems.
* Describe candidate-versus-champion evaluation.
* Explain the importance of lineage in automated retraining.
* Describe how retraining integrates with model registries, KServe, monitoring, governance, and rollback.

---

# Key Takeaway

**A production retraining pipeline should not retrain or deploy models simply because new data arrived. Monitoring signals must be converted into explicit decisions, and every candidate should pass evaluation and governance gates before promotion.**

The core architecture is:

```text
Production Data
      ↓
Drift Detection
      ↓
Retrain?
   │      │
  No     Yes
   │      ↓
 Stop   Training
          ↓
      Candidate
          ↓
       Evaluate
          ↓
       Approve?
       │      │
      No     Yes
       │      ↓
     Stop   Deploy
```

The stronger production pattern is:

```text
Monitoring
+
Drift
+
Performance
+
New Data
+
Business Impact
      ↓
Retraining Decision
      ↓
Candidate Model
      ↓
Quality + Governance Gates
      ↓
Controlled Deployment
      ↓
Production Monitoring
```

The broader principle is:

```text
Monitoring
    ↓
Decision
    ↓
Automation
    ↓
Validation
    ↓
Governance
    ↓
Deployment
    ↓
Feedback
```

This is what turns a basic training pipeline into a **closed-loop enterprise MLOps system**.
