# Hands-On Lab: Build a Machine Learning Pipeline with Kubeflow Pipelines

In this lab, you will build a complete machine learning workflow using **Kubeflow Pipelines (KFP)**.

The pipeline will contain three independent stages: **data preprocessing, model training, and model evaluation**. Each stage will execute as a separate pipeline component, and data will move between components through explicit Kubeflow artifacts rather than a shared local filesystem.

You will use the Iris dataset, train a logistic-regression classifier, record model accuracy as a structured pipeline metric, compile the workflow into a YAML specification, and execute it through a Kubeflow Pipelines environment.

The complete workflow is:

```text
                ┌───────────────┐
                │  Preprocess   │
                └───────┬───────┘
                        │
             ┌──────────┴──────────┐
             │                     │
     Training Dataset         Test Dataset
             │                     │
             ▼                     │
        ┌─────────┐                │
        │  Train  │                │
        └────┬────┘                │
             │                     │
           Model                   │
             │                     │
             └──────────┬──────────┘
                        ▼
                 ┌────────────┐
                 │  Evaluate  │
                 └─────┬──────┘
                       │
                       ▼
                    Metrics
```

---

## Lab Objectives

By completing this lab, you will learn how to:

* Define a machine learning pipeline with Kubeflow Pipelines.
* Build reusable preprocessing, training, and evaluation components.
* Run each pipeline stage in an isolated container.
* Pass datasets and model artifacts between components.
* Record structured evaluation metrics.
* Define dependencies through artifact relationships.
* Compile Python pipeline code into a YAML specification.
* Upload and execute a pipeline through Kubeflow.
* Inspect task logs, artifacts, metrics, and execution history.
* Understand how pipeline orchestration supports reproducible MLOps.

---

## Estimated Time

**Approximately 90–150 minutes**

---

## Tools

This lab uses:

* Kubernetes
* Kubeflow Pipelines
* Python
* KFP Python SDK
* pandas
* scikit-learn
* joblib
* Iris dataset
* Kubeflow Pipelines UI

---

# Step 1: Verify the Prerequisites

You need access to a Kubernetes environment with **Kubeflow Pipelines** installed and configured.

The environment may run:

```text
Public Cloud
On-Premises
Local Kubernetes Lab
Managed Kubernetes
```

You also need Python in your local development environment or Kubeflow Notebook.

Install the KFP SDK:

```bash
pip install kfp
```

Verify the installation:

```bash
python -c "import kfp; print(kfp.__version__)"
```

---

# Step 2: Understand the Pipeline Architecture

A conventional Python script might perform all operations sequentially:

```text
Load Data
   ↓
Preprocess
   ↓
Train
   ↓
Evaluate
```

Kubeflow converts this logic into independent components:

```text
Component 1
Preprocess
    ↓
Artifacts
    ↓
Component 2
Train
    ↓
Model Artifact
    ↓
Component 3
Evaluate
```

Each component can run in its own container.

This provides:

* Isolation
* Reproducibility
* Explicit dependencies
* Artifact tracking
* Independent execution
* Easier debugging

---

# Step 3: Understand KFP Artifacts

Kubeflow components should not assume that files written locally by one container will automatically appear in another container.

Instead:

```text
Component A
    ↓
Output Artifact
    ↓
Kubeflow Artifact Handling
    ↓
Input Artifact
    ↓
Component B
```

This lab uses:

```text
Dataset
Model
Metrics
```

artifacts.

---

# Step 4: Create the Pipeline File

Create:

```text
lab-07-iris-pipeline.py
```

Start with:

```python
from kfp import dsl
from kfp.dsl import (
    Dataset,
    Input,
    Metrics,
    Model,
    Output,
)
```

---

# Step 5: Create the Preprocessing Component

Add:

```python
@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "scikit-learn",
    ],
)
def preprocess(
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    import pandas as pd

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris(
        as_frame=True
    )

    df = iris.frame

    train_df, test_df = (
        train_test_split(
            df,
            test_size=0.2,
            random_state=42,
        )
    )

    train_df.to_csv(
        train_dataset.path,
        index=False,
    )

    test_df.to_csv(
        test_dataset.path,
        index=False,
    )
```

---

# Step 6: Understand the Preprocessing Component

The component performs:

```text
Iris Dataset
     ↓
Pandas DataFrame
     ↓
80 / 20 Split
     ↓
Training Dataset Artifact
+
Test Dataset Artifact
```

The fixed:

```text
random_state = 42
```

makes the split reproducible.

---

# Step 7: Understand Output Artifacts

The function signature declares:

```python
train_dataset: Output[Dataset]
test_dataset: Output[Dataset]
```

Kubeflow provides artifact paths for these outputs.

The component writes directly to:

```python
train_dataset.path
```

and:

```python
test_dataset.path
```

This makes the outputs visible to downstream components through Kubeflow's artifact mechanisms.

---

# Step 8: Create the Training Component

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
def train(
    train_dataset: Input[Dataset],
    model: Output[Model],
):
    import joblib
    import pandas as pd

    from sklearn.linear_model import LogisticRegression

    train_df = pd.read_csv(
        train_dataset.path
    )

    X = train_df.drop(
        "target",
        axis=1,
    )

    y = train_df["target"]

    classifier = LogisticRegression(
        max_iter=200
    )

    classifier.fit(
        X,
        y,
    )

    joblib.dump(
        classifier,
        model.path,
    )
```

---

# Step 9: Understand the Training Component

The training flow is:

```text
Training Dataset Artifact
          ↓
       Load CSV
          ↓
    Separate X and y
          ↓
Logistic Regression
          ↓
       Train Model
          ↓
    Serialize Model
          ↓
      Model Artifact
```

The training component receives:

```python
Input[Dataset]
```

and produces:

```python
Output[Model]
```

This gives the pipeline an explicit model artifact.

---

# Step 10: Understand Component Isolation

The training component installs its own dependencies:

```text
pandas
scikit-learn
joblib
```

inside its runtime environment.

Conceptually:

```text
Preprocess Container
      ≠
Training Container
      ≠
Evaluation Container
```

Each component should therefore declare everything it requires.

---

# Step 11: Create the Evaluation Component

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
def evaluate(
    test_dataset: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics],
):
    import joblib
    import pandas as pd

    from sklearn.metrics import accuracy_score

    test_df = pd.read_csv(
        test_dataset.path
    )

    X = test_df.drop(
        "target",
        axis=1,
    )

    y = test_df["target"]

    classifier = joblib.load(
        model.path
    )

    predictions = classifier.predict(
        X
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    metrics.log_metric(
        "accuracy",
        float(accuracy),
    )

    print(
        f"Model accuracy: {accuracy:.4f}"
    )
```

---

# Step 12: Understand the Evaluation Flow

The evaluation stage receives:

```text
Test Dataset
+
Trained Model
```

and produces:

```text
Accuracy Metric
```

The complete path is:

```text
Test Dataset Artifact ──┐
                        │
Model Artifact ─────────┼──► Evaluate
                        │
                        ▼
                     Accuracy
```

---

# Step 13: Understand Structured Metrics

Instead of only:

```python
print(accuracy)
```

the lab uses:

```python
metrics.log_metric(
    "accuracy",
    float(accuracy),
)
```

This makes model quality part of the pipeline execution metadata.

Structured metrics can later support:

```text
Evaluation
    ↓
Threshold Check
    ↓
Promote Model?
```

---

# Step 14: Define the Pipeline

Add:

```python
@dsl.pipeline(
    name="iris-classifier-pipeline",
    description=(
        "Iris classification pipeline with "
        "preprocessing, training, and evaluation"
    ),
)
def iris_pipeline():

    preprocess_task = preprocess()

    train_task = train(
        train_dataset=(
            preprocess_task
            .outputs[
                "train_dataset"
            ]
        )
    )

    evaluate(
        test_dataset=(
            preprocess_task
            .outputs[
                "test_dataset"
            ]
        ),
        model=(
            train_task
            .outputs[
                "model"
            ]
        ),
    )
```

---

# Step 15: Understand the Dependency Graph

Kubeflow infers dependencies from data flow.

You do not manually say:

```text
Run A
Then B
Then C
```

Instead:

```text
B requires output from A
C requires outputs from A and B
```

Therefore:

```text
Artifact Dependencies
        ↓
Execution Dependencies
```

This is a key workflow-orchestration principle.

---

# Step 16: Visualize the DAG

The pipeline graph is:

```text
                   ┌──────────────┐
                   │  Preprocess  │
                   └──────┬───────┘
                          │
               ┌──────────┴──────────┐
               │                     │
        Training Dataset        Test Dataset
               │                     │
               ▼                     │
          ┌─────────┐                │
          │  Train  │                │
          └────┬────┘                │
               │                     │
             Model                   │
               │                     │
               └──────────┬──────────┘
                          ▼
                   ┌────────────┐
                   │  Evaluate  │
                   └─────┬──────┘
                         │
                         ▼
                      Metrics
```

This is a directed acyclic graph, or **DAG**.

---

# Step 17: Compile the Pipeline

Add to the bottom of the file:

```python
from kfp import compiler


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path=(
            "iris_pipeline.yaml"
        ),
    )
```

Run:

```bash
python lab-07-iris-pipeline.py
```

Expected output artifact:

```text
iris_pipeline.yaml
```

---

# Step 18: Understand Pipeline Compilation

The Python source describes:

```text
Components
Artifacts
Dependencies
Configuration
```

The compiler converts it into:

```text
Python Pipeline Definition
           ↓
       KFP Compiler
           ↓
      Pipeline YAML
```

The YAML becomes the executable workflow specification for Kubeflow Pipelines.

---

# Step 19: Keep Source and Generated Artifacts Separate

The recommended pattern is:

```text
Source of Truth
    ↓
Python Pipeline Code
```

while:

```text
Generated Artifact
    ↓
Compiled YAML
```

can be regenerated when the source changes.

This supports normal software-engineering workflows:

* Version control
* Code review
* Testing
* CI/CD
* Reproducibility

---

# Step 20: Upload the Pipeline

Open the **Kubeflow Pipelines UI**.

Upload:

```text
iris_pipeline.yaml
```

Create a pipeline run.

Kubeflow should schedule:

```text
Preprocess
    ↓
Train
    ↓
Evaluate
```

according to artifact dependencies.

---

# Step 21: Observe Component Execution

Each stage runs independently.

Conceptually:

```text
Kubernetes Cluster
│
├── Preprocess Pod
│      ↓
│   Dataset Artifacts
│
├── Train Pod
│      ↓
│   Model Artifact
│
└── Evaluate Pod
       ↓
     Metrics
```

The Kubeflow UI allows you to inspect individual tasks.

---

# Step 22: Inspect Logs

For each component, review:

* Task status
* Start time
* Completion time
* Logs
* Inputs
* Outputs
* Artifacts
* Metrics

If a component fails, inspect that component directly rather than treating the pipeline as a single opaque application.

---

# Step 23: Inspect the Dataset Artifacts

The preprocessing component creates:

```text
Training Dataset
Test Dataset
```

Verify that both artifacts are visible in the run.

The lineage is:

```text
Preprocess
   ├──► Training Dataset
   └──► Test Dataset
```

---

# Step 24: Inspect the Model Artifact

The training component produces:

```text
Model Artifact
```

The model is serialized using:

```text
joblib
```

Its lineage is:

```text
Training Dataset
      ↓
Train
      ↓
Model Artifact
```

---

# Step 25: Inspect the Evaluation Metric

The evaluation stage records:

```text
accuracy
```

through the `Metrics` artifact.

The complete lineage is:

```text
Training Dataset
      ↓
Train
      ↓
Model
      ↓
Evaluate
      ↓
Accuracy
```

with the test dataset feeding evaluation separately.

---

# Step 26: Understand Pipeline Lineage

Artifact relationships allow the platform to answer questions such as:

```text
Which run created this model?
Which dataset was used?
Which component generated it?
Which evaluation metric belongs to it?
```

This establishes the foundation for **ML lineage and reproducibility**.

---

# Step 27: Add a Model Quality Threshold

A production workflow should not necessarily promote every trained model.

A better architecture is:

```text
Train
  ↓
Evaluate
  ↓
Accuracy
  ↓
Threshold
  │
  ├── Pass → Register / Deploy
  │
  └── Fail → Stop
```

For example:

```text
accuracy >= 0.90
```

could become a promotion condition.

---

# Step 28: Extend Evaluation Metrics

Accuracy alone may not be sufficient.

Add metrics such as:

```text
Precision
Recall
F1 Score
ROC-AUC
```

depending on the application.

A production evaluation stage may produce:

```text
Model
  ↓
Evaluate
  ├── Accuracy
  ├── Precision
  ├── Recall
  └── F1
```

---

# Step 29: Parameterize the Pipeline

Instead of hard-coding values such as:

```text
test_size = 0.2
max_iter = 200
```

turn them into pipeline parameters.

For example:

```text
Pipeline Parameters
├── Test Size
├── Random Seed
├── Max Iterations
└── Accuracy Threshold
```

This allows the same pipeline definition to support multiple experiments.

---

# Step 30: Understand Reusable Components

The current components use:

```python
@dsl.component
```

with runtime package installation.

This is convenient for learning.

For repeated production workloads, consider:

```text
Reusable Component
      ↓
Prebuilt Container Image
      ↓
Pinned Dependencies
```

instead of installing packages every time a component starts.

---

# Step 31: Understand Startup Overhead

The lab uses:

```python
packages_to_install=[
    ...
]
```

which may cause each component to install dependencies during startup.

Conceptually:

```text
Start Pod
   ↓
Install Packages
   ↓
Run Component
```

For production:

```text
Build Container
   ↓
Dependencies Already Installed
   ↓
Start Component Faster
```

---

# Step 32: Add Hyperparameter Tuning

A natural extension is:

```text
Training Data
     ↓
Multiple Training Configurations
     ↓
Evaluation
     ↓
Best Model
```

Kubeflow environments may integrate with tools such as:

```text
Katib
```

for hyperparameter optimization.

---

# Step 33: Understand the Hyperparameter-Tuning Flow

Conceptually:

```text
                 ┌── Train Config A ──► Metric A
Training Data ───┼── Train Config B ──► Metric B
                 └── Train Config C ──► Metric C
                              ↓
                         Select Best
```

This extends the pipeline beyond a single fixed model run.

---

# Step 34: Add Model Registration

After successful evaluation:

```text
Train
   ↓
Evaluate
   ↓
Threshold Passed
   ↓
Register Model
```

A model registry can track:

* Model versions
* Evaluation metrics
* Training metadata
* Deployment status
* Approval state

---

# Step 35: Add Model Serving

A later production stage might use:

```text
KServe
```

or another serving platform.

The pipeline becomes:

```text
Preprocess
    ↓
Train
    ↓
Evaluate
    ↓
Register
    ↓
Deploy
    ↓
Serve
```

---

# Step 36: Integrate MLflow

Another extension is:

```text
Kubeflow Pipeline
       ↓
Training
       ↓
MLflow Tracking / Registry
```

This can add experiment tracking and model-management capabilities alongside pipeline orchestration.

---

# Step 37: Understand Orchestration vs. Experiment Tracking

Kubeflow Pipelines primarily orchestrates:

```text
What runs
When it runs
What depends on what
What artifacts move between stages
```

Experiment-tracking systems focus more strongly on:

```text
Parameters
Metrics
Runs
Artifacts
Model Versions
```

These capabilities can complement one another.

---

# Step 38: Add Durable Artifact Storage

Production model and dataset artifacts should use durable storage appropriate to the Kubeflow deployment.

Conceptually:

```text
Component
    ↓
Artifact
    ↓
Durable Object / Artifact Storage
    ↓
Downstream Component
```

This prevents workflow state from depending on the lifecycle of individual pods.

---

# Step 39: Understand Why Local Files Are Not Enough

A local file path such as:

```text
/tmp/model.pkl
```

inside one container does not inherently exist inside another container.

Therefore:

```text
Container A Local Filesystem
           ≠
Container B Local Filesystem
```

KFP artifacts solve this by making data transfer explicit.

---

# Step 40: Add Conditional Execution

A production pipeline may include conditional logic:

```text
Evaluate
   ↓
accuracy >= threshold?
   │
   ├── Yes → Register
   │           ↓
   │         Deploy
   │
   └── No → Stop
```

This turns evaluation into an operational control.

---

# Step 41: Add Pipeline Observability

Monitor:

* Pipeline duration
* Component duration
* Failure rate
* Retry count
* Artifact sizes
* Training metrics
* Resource utilization
* Scheduling delays

A useful architecture is:

```text
Kubeflow Pipeline
       ↓
Logs + Metrics + Events
       ↓
Observability Platform
```

---

# Step 42: Understand Reproducibility

A reproducible pipeline requires more than code.

Consider versioning:

```text
Pipeline Code
Container Images
Dependencies
Dataset
Model Configuration
Random Seeds
Environment
```

The goal is:

```text
Same Inputs
+
Same Configuration
+
Compatible Environment
        ↓
Reproducible Workflow
```

---

# Step 43: Production MLOps Architecture

A mature pipeline may evolve into:

```text
Raw Data
   ↓
Validation
   ↓
Feature Engineering
   ↓
Training
   ↓
Evaluation
   ↓
Quality Gate
   ↓
Model Registry
   ↓
Deployment
   ↓
Monitoring
   ↓
Retraining Trigger
```

Kubeflow Pipelines can orchestrate many of these stages.

---

# Step 44: Additional Challenges

Extend the lab by:

* Adding precision, recall, and F1.
* Parameterizing the train/test split.
* Parameterizing logistic-regression settings.
* Adding a model-quality threshold.
* Adding conditional deployment.
* Creating reusable container images.
* Adding Katib hyperparameter tuning.
* Integrating MLflow.
* Integrating KServe.
* Persisting artifacts in durable object storage.
* Adding pipeline-level monitoring.
* Adding scheduled execution.
* Triggering retraining from new data.

---

# Step 45: Recommended Repository Structure

For a standalone project:

```text
kubeflow-pipeline-lab/
├── iris_pipeline.py
└── iris_pipeline.yaml
```

For your book companion repository:

```text
chapter-28/
├── lab-07-build-a-machine-learning-pipeline-with-kubeflow-pipelines.md
└── lab-07-iris-pipeline.py
```

The compiled:

```text
iris_pipeline.yaml
```

can either be committed for convenience or regenerated from the Python source.

For a learning repository, I would generally keep the **Python pipeline source as the authoritative artifact** and treat the YAML as generated output.

---

# Step 46: Clean Up

Delete test pipeline runs from the Kubeflow UI if they are no longer required.

Remove local generated YAML if desired:

```bash
rm -f iris_pipeline.yaml
```

The Python source can regenerate it at any time.

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Accessed a working Kubeflow Pipelines environment.
* [ ] Installed the KFP SDK.
* [ ] Created the pipeline source file.
* [ ] Created the preprocessing component.
* [ ] Loaded the Iris dataset.
* [ ] Created training and test dataset artifacts.
* [ ] Created the training component.
* [ ] Trained logistic regression.
* [ ] Produced a model artifact.
* [ ] Created the evaluation component.
* [ ] Calculated model accuracy.
* [ ] Logged accuracy through a `Metrics` artifact.
* [ ] Defined the pipeline DAG.
* [ ] Connected components through artifact outputs and inputs.
* [ ] Compiled the pipeline.
* [ ] Generated `iris_pipeline.yaml`.
* [ ] Uploaded the pipeline to Kubeflow.
* [ ] Started a pipeline run.
* [ ] Inspected the execution graph.
* [ ] Inspected component logs.
* [ ] Inspected dataset artifacts.
* [ ] Inspected the model artifact.
* [ ] Inspected evaluation metrics.
* [ ] Reviewed pipeline lineage.
* [ ] Reviewed conditional model promotion.
* [ ] Reviewed production artifact-storage considerations.
* [ ] Reviewed reusable-container strategies.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Explain Kubeflow Pipelines architecture.
* Build reusable KFP components.
* Use typed KFP artifacts.
* Pass datasets and models between isolated components.
* Record structured pipeline metrics.
* Define dependencies through artifact relationships.
* Build an ML workflow as a DAG.
* Compile pipeline source into YAML.
* Execute pipelines through Kubeflow.
* Inspect execution history, logs, metrics, and artifacts.
* Explain why artifact lineage improves reproducibility.
* Distinguish pipeline orchestration from experiment tracking.
* Describe how training pipelines can evolve into production MLOps systems.

---

# Key Takeaway

**Kubeflow Pipelines turns machine learning code into modular, reproducible, containerized workflows by separating pipeline stages and connecting them through explicit data and model artifacts.**

The core architecture is:

```text
Data
 ↓
Preprocess
 ↓
Dataset Artifact
 ↓
Train
 ↓
Model Artifact
 ↓
Evaluate
 ↓
Metrics
```

The most important orchestration principle is:

```text
Data Dependencies
      ↓
Execution Dependencies
```

Rather than manually controlling execution order, Kubeflow derives the workflow graph from the relationships between component inputs and outputs.

This enables a progression from:

```text
Python ML Script
      ↓
Reusable Components
      ↓
Pipeline DAG
      ↓
Tracked Artifacts
      ↓
Evaluation Gates
      ↓
Model Registration
      ↓
Deployment
      ↓
Production MLOps
```

The broader infrastructure principle is that machine learning workflows become easier to scale, reproduce, troubleshoot, and govern when **data movement, model artifacts, metrics, dependencies, and execution environments are explicitly represented rather than hidden inside a monolithic training script**.
