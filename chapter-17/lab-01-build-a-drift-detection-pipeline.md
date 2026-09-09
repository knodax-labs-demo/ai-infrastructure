# Hands-On Lab: Build a Drift Detection Pipeline

In this lab, you will build a practical **drift detection and monitoring pipeline** for production machine learning systems.

The pipeline compares incoming production data with a trusted reference dataset using **Evidently**. Streaming data can arrive through **Apache Kafka** or be simulated locally when Kafka is unavailable. Drift results are exposed as **Prometheus metrics**, making it possible to visualize model-related changes in **Grafana** and generate alerts when drift exceeds defined thresholds.

The complete monitoring path is:

```text
Reference Data
      ↓
Production / Streaming Data
      ↓
Drift Detection
      ↓
Prometheus Metrics
      ↓
Grafana
      ↓
Alerts
```

This architecture demonstrates how model observability can become part of the broader AI infrastructure monitoring stack rather than remaining an isolated offline data-science activity.

---

## Lab Objective

Build a drift monitoring pipeline that can:

* Establish a reference dataset representing the expected production distribution.
* Simulate incoming inference data.
* Optionally consume events through Apache Kafka.
* Detect distribution changes with Evidently.
* Expose drift measurements as Prometheus metrics.
* Visualize drift trends in Grafana.
* Trigger alerts when drift persists beyond defined thresholds.
* Provide a foundation for concept-drift monitoring.
* Support future retraining and model-lifecycle workflows.

---

## Estimated Time

**Approximately 90–120 minutes**

---

## Tools

This lab uses:

* Python 3.9 or later
* Pandas
* scikit-learn
* Evidently
* Optional Apache Kafka
* `kafka-python`
* Prometheus Python client
* Prometheus
* Grafana

---

## Architecture

The core architecture is:

```text
                   Reference Dataset
                          │
                          │
                          ▼
                    Drift Monitor
                          ▲
                          │
                          │
          Production / Streaming Data
                          │
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
             Kafka             Local Stream
                 │                 │
                 └────────┬────────┘
                          │
                          ▼
                       Evidently
                          │
                          ▼
                 Prometheus Metrics
                          │
                          ▼
                      Prometheus
                          │
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
              Grafana            Alerts
```

The lab focuses primarily on **data drift**.

It also establishes the foundation for **concept drift**, which requires additional information such as ground-truth labels or model-performance measurements.

---

# Step 1: Install the Prerequisites

Install the required Python packages:

```bash
pip install \
  pandas \
  scikit-learn \
  evidently \
  kafka-python \
  prometheus_client
```

Kafka is optional.

If Kafka is unavailable, the lab can operate using a locally simulated production stream.

If Prometheus and Grafana are already running from an earlier monitoring lab, you can reuse that environment.

---

# Step 2: Create the Project Structure

Create:

```text
lab_drift_pipeline/
├── data/
│   └── reference.csv
├── scripts/
│   ├── prepare_reference.py
│   ├── stream_data.py
│   └── drift_monitor.py
├── grafana_dashboards/
│   └── drift_dashboard.json
└── README.md
```

For the companion GitHub repository, I recommend mapping these files to:

```text
chapter-17/
├── lab-07-build-a-drift-detection-pipeline.md
├── lab-07-prepare-reference.py
├── lab-07-stream-data.py
├── lab-07-drift-monitor.py
└── lab-07-drift-dashboard.json
```

This keeps the repository naming consistent with the other chapter labs.

---

# Step 3: Prepare the Reference Dataset

The first step is to establish a **reference distribution** representing expected production conditions.

Run:

```bash
python scripts/prepare_reference.py
```

The script should create:

```text
data/reference.csv
```

The reference dataset can conceptually represent:

```text
Training Data
     ↓
Stable Distribution
     ↓
reference.csv
```

In production, reference data may come from:

* Training data
* Validation data
* A stable historical production window
* A curated representative dataset

The important requirement is that it represents the behavior considered normal.

---

## Reference-vs-Current Comparison

Later in the pipeline, every production batch is compared against this baseline:

```text
Reference Distribution
        │
        │ Compare
        ▼
Current Distribution
        │
        ▼
Statistical Drift Tests
        │
        ▼
Drift Result
```

---

# Step 4: Simulate Production Streaming Data

Start the simulated stream:

```bash
python scripts/stream_data.py
```

Initially, generated data should remain reasonably close to the reference distribution.

Conceptually:

```text
Reference
   │
   ├── Mean ≈ A
   ├── Variance ≈ B
   └── Categories ≈ C

Production Stream
   │
   ├── Mean ≈ A
   ├── Variance ≈ B
   └── Categories ≈ C
```

This represents a stable production environment.

---

## Kafka Option

When Kafka is available, the producer can publish inference events to a topic such as:

```text
inference-events
```

The architecture becomes:

```text
Application
    ↓
Inference Events
    ↓
Kafka Topic
    ↓
Drift Monitor
```

Kafka is useful when the production system already uses event-driven infrastructure.

---

## Local Simulation Option

When Kafka is unavailable:

```text
stream_data.py
      ↓
Synthetic Production Batches
      ↓
Drift Monitor
```

This keeps the lab easy to run without requiring a complete streaming platform.

---

# Step 5: Introduce Synthetic Drift

Once the stable pipeline is working, deliberately alter the incoming distribution.

Possible drift scenarios include:

* Shift a numeric feature mean.
* Increase or decrease variance.
* Change category frequencies.
* Introduce previously unseen categories.
* Change correlations among features.

For example:

```text
Stable Feature
Mean = 0
     ↓
Drift Introduced
Mean = 2
```

or:

```text
Category Distribution

Before:
A = 60%
B = 30%
C = 10%

After:
A = 25%
B = 25%
C = 50%
```

The purpose is to verify that the monitoring pipeline reacts correctly.

---

# Step 6: Run the Drift Monitor

Start:

```bash
python scripts/drift_monitor.py
```

The monitoring process should:

1. Load `reference.csv`.
2. Receive current production data.
3. Compare reference and current batches.
4. Run Evidently drift analysis.
5. Extract drift measurements.
6. Convert those measurements into Prometheus metrics.
7. Expose them through an HTTP metrics endpoint.

The workflow is:

```text
reference.csv
      │
      │
      ▼
   Evidently
      ▲
      │
Current Batch
      │
      ▼
Drift Evaluation
      │
      ▼
Operational Metrics
```

---

# Step 7: Expose Prometheus Metrics

The monitoring service exposes:

```text
http://localhost:8005/metrics
```

Verify:

```bash
curl \
  http://localhost:8005/metrics
```

Depending on your implementation, useful metrics may represent:

* Overall drift detected
* Number of drifting features
* Percentage of drifting features
* Individual feature drift status
* Drift score or severity

Conceptually:

```text
Evidently Result
      ↓
Python Metrics Layer
      ↓
Prometheus Gauge / Counter
      ↓
/metrics
```

> **Version Note**
>
> Evidently APIs can change between releases. Use the drift-report interfaces supported by the installed version rather than assuming an older preset or object name will still be available.

---

# Step 8: Integrate with Prometheus

If Prometheus runs in the same container/network environment as the drift monitor, add:

```yaml
scrape_configs:
  - job_name: "drift-monitor"

    static_configs:
      - targets:
          - "drift-monitor:8005"
```

If Prometheus runs directly on the same machine:

```yaml
scrape_configs:
  - job_name: "drift-monitor"

    static_configs:
      - targets:
          - "localhost:8005"
```

Reload or restart Prometheus after updating the configuration.

The monitoring path becomes:

```text
Drift Monitor
     ↓
/metrics :8005
     ↓
Prometheus
     ↓
Time-Series Storage
```

Drift telemetry can now live alongside:

* Kubernetes metrics
* GPU metrics
* Application metrics
* Inference metrics
* Infrastructure metrics

---

# Step 9: Verify Prometheus Scraping

Open Prometheus and verify that the drift-monitor target is active.

Conceptually:

```text
Drift Monitor
      ↓
Prometheus Target
      ↓
UP
```

Query your exported metrics.

For example, if your implementation exposes:

```text
drift_detected
```

query:

```promql
drift_detected
```

For percentage of drifting features:

```promql
drifted_features_percentage
```

Use the exact metric names produced by your implementation.

---

# Step 10: Build the Grafana Dashboard

The project includes:

```text
grafana_dashboards/drift_dashboard.json
```

For the GitHub repository, use:

```text
lab-07-drift-dashboard.json
```

Import the dashboard into Grafana or create equivalent panels manually.

Recommended panels include:

* Overall drift status
* Percentage of drifting features
* Feature-level drift status
* Drift severity over time
* Sustained drift alerts

A dashboard might look like:

```text
┌──────────────────────────────┐
│ Overall Drift Status         │
├──────────────────────────────┤
│ % Features with Drift        │
├──────────────────────────────┤
│ Feature Drift by Name        │
├──────────────────────────────┤
│ Drift Severity Over Time     │
├──────────────────────────────┤
│ Alert Status                 │
└──────────────────────────────┘
```

---

# Step 11: Test the Stable State

Run the stream without intentional drift:

```bash
python scripts/stream_data.py
```

Expected behavior:

```text
Reference Data
      ≈
Current Data
      ↓
Little / No Drift
      ↓
Prometheus Metrics Stable
      ↓
Grafana Shows Healthy State
```

This establishes that the system does not trigger drift simply because new batches are arriving.

---

# Step 12: Test the Drift State

Introduce synthetic drift:

```bash
python scripts/stream_data.py \
  --drift-mode
```

Expected flow:

```text
Reference Distribution
        ↓
Current Distribution Changes
        ↓
Evidently Detects Differences
        ↓
Prometheus Metrics Change
        ↓
Grafana Drift Indicators Rise
        ↓
Alert Threshold Reached
```

This test validates the complete monitoring pipeline.

---

# Step 13: Add Alerting

Drift detection becomes operationally useful when sustained changes can trigger an alert.

A simple alerting concept is:

```text
Drift Percentage
      ↓
Threshold
      ↓
Sustained Duration
      ↓
Alert
```

For example:

```promql
drifted_features_percentage > 30
```

combined with a sustained interval such as:

```text
for: 10m
```

can detect prolonged drift.

> **Important**
>
> Threshold values are application-specific. Do not treat a single generic threshold as suitable for every model.

---

## Alerting Workflow

```text
Drift Metric
    ↓
PrometheusRule
    ↓
Threshold Exceeded
    ↓
Sustained?
    ↓
Alertmanager
    ↓
Notification / Review
```

---

# Step 14: Understand Data Drift

Data drift means the statistical distribution of incoming input data has changed.

Examples include:

```text
Feature Mean Changes
Feature Variance Changes
Category Frequencies Change
New Categories Appear
Feature Correlations Change
```

Data drift can signal:

* Changing user behavior
* Seasonal patterns
* Sensor changes
* Upstream data changes
* New market conditions
* Pipeline bugs

However:

```text
Data Drift
    ≠
Automatic Model Failure
```

A distribution change does not necessarily mean prediction quality has degraded.

---

# Step 15: Extend Toward Concept Drift

Concept drift concerns changes in the relationship between inputs and expected outcomes.

Conceptually:

```text
Same Input Pattern
       ↓
Different Real-World Outcome
       ↓
Model Relationship No Longer Holds
```

Detecting concept drift generally requires:

* Ground-truth labels
* Delayed outcomes
* Model-performance measurements

Useful metrics might include:

* Accuracy
* Precision
* Recall
* F1 score
* RMSE
* MAE
* Task-specific business metrics

The expanded monitoring architecture becomes:

```text
Input Features
      ↓
Data Drift
      │
      │
Predictions + Ground Truth
      ↓
Performance Monitoring
      ↓
Concept Drift
```

---

# Step 16: Connect Drift Detection to Model Lifecycle Management

A mature system can connect drift signals to model-management workflows.

However, drift should not automatically trigger retraining.

A safer workflow is:

```text
Drift Detected
      ↓
Alert
      ↓
Human / Automated Review
      ↓
Check Model Performance
      ↓
Check Business Impact
      ↓
Retraining Decision
      ↓
Train Candidate Model
      ↓
Evaluate
      ↓
Approve
      ↓
Deploy
```

This avoids reacting incorrectly to harmless or expected distribution changes.

---

# Step 17: Design a Controlled Retraining Trigger

A robust retraining decision can combine multiple signals:

```text
Data Drift
    +
Performance Degradation
    +
Sufficient New Data
    +
Business Impact
    ↓
Retraining Candidate
```

This is generally safer than:

```text
Any Drift
   ↓
Immediately Retrain
```

because drift may reflect:

* Seasonality
* Temporary events
* Valid changes
* Short-lived anomalies

---

# Step 18: Troubleshooting

## Drift Metrics Do Not Appear

Check:

```bash
curl \
  http://localhost:8005/metrics
```

If metrics are not exposed, inspect the drift-monitor process before troubleshooting Prometheus.

---

## Prometheus Cannot Scrape the Monitor

Verify the target hostname and port.

For local Prometheus:

```text
localhost:8005
```

For containerized networking:

```text
drift-monitor:8005
```

Make sure Prometheus can resolve and reach the target.

---

## Grafana Panels Are Empty

Test the metric directly in Prometheus first.

Troubleshoot:

```text
Drift Monitor
     ↓
/metrics
     ↓
Prometheus
     ↓
Grafana
```

If Prometheus has data but Grafana does not, check:

* Data source
* Query syntax
* Time range
* Panel variables

---

## No Drift Is Detected

Verify that:

* `--drift-mode` is active.
* The incoming batch is sufficiently different.
* The expected feature columns are present.
* Reference and current schemas match.
* Evidently is evaluating the intended features.

---

## Too Much Drift Is Detected

Check whether:

* The reference dataset is representative.
* Production batches are too small.
* Feature transformations differ.
* Data preprocessing is inconsistent.
* The drift threshold is too sensitive.

---

## Evidently API Errors

Evidently APIs may vary by release.

Verify the installed version:

```bash
pip show evidently
```

Then adapt the drift-report code to the interfaces supported by that release.

---

# Step 19: Clean Up

Stop the monitoring process:

```bash
pkill -f drift_monitor.py
```

If Kafka was started specifically for the lab using Docker, stop the corresponding containers:

```bash
docker stop \
  kafka \
  zookeeper
```

Do not stop shared Kafka, Prometheus, or Grafana services if they are being used by other labs or applications.

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Installed the required Python packages.
* [ ] Created the project structure.
* [ ] Generated a reference dataset.
* [ ] Saved `reference.csv`.
* [ ] Generated stable production data.
* [ ] Tested optional Kafka streaming or local simulation.
* [ ] Started the drift monitor.
* [ ] Compared current data against reference data.
* [ ] Generated Evidently drift results.
* [ ] Exposed Prometheus metrics.
* [ ] Verified the `/metrics` endpoint.
* [ ] Configured Prometheus scraping.
* [ ] Verified the drift-monitor target.
* [ ] Queried drift metrics in Prometheus.
* [ ] Created or imported a Grafana dashboard.
* [ ] Observed the stable state.
* [ ] Introduced synthetic drift.
* [ ] Observed drift metrics change.
* [ ] Reviewed alerting behavior.
* [ ] Understood the difference between data drift and concept drift.
* [ ] Reviewed controlled retraining workflows.
* [ ] Cleaned up lab-specific processes.

---

# Expected Results

At the end of the lab:

* A reference dataset should represent the expected feature distribution.
* A simulated production stream should provide current data.
* Evidently should compare the reference and current batches.
* Drift results should be converted into Prometheus-compatible metrics.
* Prometheus should scrape the drift-monitor service.
* Grafana should visualize drift behavior over time.
* Synthetic drift should cause observable metric changes.
* Sustained drift should be capable of generating an alert when configured.

The resulting observability path is:

```text
Production Data
      ↓
Drift Detection
      ↓
Prometheus
      ↓
Grafana
      ↓
Alerting
```

---

# Learning Outcomes

After completing this lab, you should be able to:

* Explain the role of reference data in drift monitoring.
* Distinguish between reference and current production distributions.
* Simulate production inference data.
* Use Kafka or local streaming patterns for monitoring input.
* Use Evidently to evaluate data drift.
* Convert statistical drift results into operational metrics.
* Expose custom Prometheus metrics from Python.
* Integrate model-monitoring metrics with Prometheus.
* Visualize drift trends using Grafana.
* Configure alerts for sustained drift.
* Explain the difference between data drift and concept drift.
* Understand why drift alone should not automatically trigger retraining.
* Design a controlled model-lifecycle response to persistent drift.

---

# Key Takeaway

**Model drift monitoring should be treated as part of production observability, not as an occasional offline analysis task.**

A production-oriented architecture connects statistical drift detection to the same operational systems used for infrastructure and application monitoring:

```text
Reference Data
      ↓
Production Data
      ↓
Drift Detection
      ↓
Prometheus
      ↓
Grafana
      ↓
Alerts
      ↓
Review
      ↓
Retraining Decision
```

Data drift indicates that the input distribution has changed, but it does not automatically prove that model quality has deteriorated. The strongest monitoring systems therefore combine **data drift, model-performance metrics, ground-truth outcomes, infrastructure telemetry, and controlled retraining workflows** before taking automated action.
