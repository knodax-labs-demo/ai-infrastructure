# Hands-On Lab: Optimize Cloud AI Workload Costs

In this lab, you will systematically reduce the end-to-end cost of an AI training and inference workload while maintaining acceptable performance, reliability, and model quality.

Rather than focusing only on the hourly price of GPU infrastructure, you will measure the **total cost required to complete useful AI work**. You will establish a baseline, instrument the environment, test optimization techniques across compute, storage, networking, scheduling, and runtime efficiency, and compare the results.

The workflow is:

```text
Baseline
   ↓
Instrument
   ↓
Optimize
   ↓
Measure
   ↓
Compare
   ↓
Report
```

The objective is to identify configurations that provide the best practical balance among **cost, throughput, latency, reliability, GPU utilization, and model accuracy**.

---

## Lab Objective

Reduce the total cost of AI training and inference workloads without sacrificing required performance or model quality.

You will evaluate techniques including:

* Spot capacity
* Checkpointing
* Autoscaling
* Mixed-precision training
* Batch-size optimization
* Gradient accumulation
* GPU utilization improvements
* Storage lifecycle policies
* Data locality
* Runtime optimization
* Dynamic inference batching
* Caching
* Commitment-based pricing
* Spot-aware Kubernetes scheduling

---

## Estimated Time

**Approximately 90–150 minutes**, depending on the number of experiments performed.

Some optimization techniques, such as storage lifecycle policies or commitment-based pricing analysis, may be evaluated conceptually rather than through long-running experiments.

---

## Tools

This lab can use:

* AWS, Google Cloud, or Microsoft Azure
* GPU-enabled virtual machines
* PyTorch
* Docker
* Kubernetes
* `kubectl`
* Cloud provider CLI
* NVIDIA GPU tools
* Prometheus and Grafana
* NVIDIA DCGM Exporter
* Optional OpenCost or another cost-monitoring platform
* Cost Tracking Worksheet

---

## Cost Tracking Worksheet

Use the accompanying worksheet:

```text
lab-07-cost-tracking-worksheet.xlsx
```

to record each experiment.

The worksheet should capture the same metrics for every configuration so that results remain comparable.

---

## Prerequisites

Before starting the lab, make sure you have access to one of:

* Amazon Web Services
* Google Cloud
* Microsoft Azure

You should have permission to provision GPU resources and, where available, interruptible or Spot capacity.

The lab can be performed using:

* GPU virtual machines
* Kubernetes
* A combination of both

Kubernetes is particularly useful for the autoscaling and scheduling exercises.

Install the appropriate cloud CLI:

```text
AWS           → aws
Google Cloud  → gcloud
Azure         → az
```

If you are using Kubernetes, verify:

```bash
kubectl get nodes
```

Verify Docker:

```bash
docker --version
```

Verify access to the NVIDIA GPU:

```bash
nvidia-smi
```

---

## Example Workload

A practical workload for this lab is:

```text
Training:
PyTorch + ResNet-50 + CIFAR-10

Inference:
FastAPI or NVIDIA Triton Inference Server
```

You may use another workload if necessary, but keep it consistent across experiments.

> **Experimental Control**
>
> Keep the dataset, model architecture, evaluation criteria, and workload behavior fixed whenever possible. Change **one major infrastructure or runtime variable at a time** so that its impact can be measured accurately.

---

## Cost Warning

This lab provisions billable cloud resources such as:

* GPU instances
* Persistent storage
* Object storage
* Network services
* Public IP addresses
* Load balancers
* Kubernetes worker nodes

Actual cost depends on:

* Cloud provider
* Region
* GPU type
* Spot availability
* Experiment duration
* Storage usage
* Network transfer

Use short experimental runs and modest GPU instances where possible.

> **⚠️ Cost Warning**
>
> Review current cloud pricing, configure billing alerts, and terminate resources immediately after each experiment when they are no longer required.

---

# Step 1: Design the Experiment

Begin by defining an **unoptimized baseline**.

A reasonable baseline might use:

* One on-demand GPU
* One cloud region
* FP32 training
* Batch size of 128, if supported
* No Spot capacity
* No autoscaling
* No mixed precision
* Standard storage
* Compute and storage in the same region

The baseline should represent a straightforward deployment before optimization techniques are introduced.

---

## Baseline Architecture

```text
Dataset
   ↓
Object Storage
   ↓
On-Demand GPU
   ↓
FP32 Training
   ↓
Model
```

Record the baseline configuration in the Cost Tracking Worksheet.

---

## Metrics to Capture

For every experiment, record the same key performance indicators.

| Metric                 | Purpose                                      |
| ---------------------- | -------------------------------------------- |
| Throughput             | Measures useful work completed per unit time |
| Training duration      | Measures total job completion time           |
| p50 latency            | Measures typical inference latency           |
| p95 latency            | Measures tail inference latency              |
| GPU utilization        | Shows accelerator efficiency                 |
| GPU memory utilization | Shows accelerator memory pressure            |
| Model accuracy         | Confirms quality is preserved                |
| Compute cost           | Measures accelerator/VM expense              |
| Storage cost           | Measures persistent-data expense             |
| Network cost           | Measures data-transfer expense               |
| Total workload cost    | Measures end-to-end cost                     |
| Savings vs. baseline   | Quantifies improvement                       |

A useful formula is:

```text
Total Workload Cost =
Compute Cost
+ Storage Cost
+ Network Cost
+ Other Infrastructure Cost
```

Calculate savings as:

```text
Savings (%) =
((Baseline Cost - Optimized Cost) / Baseline Cost) × 100
```

---

# Step 2: Instrument the Workload

Before attempting optimization, establish visibility into how the workload uses infrastructure.

Monitoring should remain enabled during the baseline and all subsequent experiments.

You want to determine whether the workload is:

```text
Compute-Bound
Memory-Bound
Storage-Bound
Network-Bound
Data-Loading-Bound
```

Without measurement, optimization decisions are largely assumptions.

---

## Monitor GPU Utilization

Use `nvidia-smi`:

```bash
nvidia-smi \
  --query-gpu=name,utilization.gpu,utilization.memory,memory.total,pstate,power.draw \
  --format=csv \
  -l 5
```

Observe:

* GPU model
* GPU utilization
* Memory utilization
* Total GPU memory
* Performance state
* Power consumption

A consistently underutilized GPU may indicate that another part of the pipeline is limiting performance.

---

## Monitor Storage

On Linux, use:

```bash
iostat -xz 5
```

Observe:

* Disk utilization
* Queue length
* Read throughput
* Write throughput
* I/O wait

---

## Monitor Network Activity

Use:

```bash
ifstat 5
```

This can help identify workloads that are limited by dataset transfer or remote storage access.

---

## Kubernetes Monitoring

For Kubernetes environments, use monitoring tools such as:

* Prometheus
* Grafana
* NVIDIA DCGM Exporter

Install NVIDIA DCGM Exporter:

```bash
helm repo add nvidia \
  https://nvidia.github.io/dcgm-exporter/helm-charts
```

Update Helm:

```bash
helm repo update
```

Install:

```bash
helm install dcgm \
  nvidia/dcgm-exporter \
  -n monitoring \
  --create-namespace
```

GPU telemetry can then be collected for Kubernetes workloads.

A Kubernetes cost-monitoring system such as OpenCost can also help connect cluster utilization to infrastructure cost.

---

# Step 3: Tag Resources for Cost Attribution

Optimization is easier when infrastructure costs can be associated with a specific experiment.

Apply consistent tags or labels such as:

```text
Project
Team
Environment
Experiment
Lab
```

---

## AWS Example

```bash
--tag-specifications \
'ResourceType=instance,Tags=[{Key=Project,Value=Lab13-07}]'
```

## Google Cloud Example

```bash
--labels=project=lab13-07
```

## Azure Example

```bash
--tags Project=Lab13-07
```

Consistent cost attribution allows billing reports to isolate resources created for this lab.

---

# Step 4: Establish the On-Demand Baseline

Launch a GPU instance using standard on-demand pricing.

Use:

* FP32 precision
* Fixed batch size
* No autoscaling
* No Spot capacity
* Same-region compute and storage

Run the training workload and record:

* Throughput
* GPU utilization
* Training duration
* Final accuracy
* Compute cost
* Storage cost
* Network cost
* Total cost

Do not make optimization changes until this baseline is recorded.

---

## AWS Example

```bash
aws ec2 run-instances \
  --image-id ami-xxxxxxxx \
  --instance-type g5.2xlarge \
  --key-name yourkey \
  --count 1 \
  --block-device-mappings \
  '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200}}]' \
  --tag-specifications \
  'ResourceType=instance,Tags=[{Key=Project,Value=Lab13-07}]'
```

---

## Google Cloud Example

```bash
gcloud compute instances create lab13-07-ondemand \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=count=1,type=nvidia-tesla-a100 \
  --boot-disk-size=200GB \
  --labels=project=lab13-07
```

---

## Azure Example

```bash
az vm create \
  -g rg-lab13-07 \
  -n lab13-07-ondemand \
  --image Ubuntu2204 \
  --size Standard_NC4as_T4_v3 \
  --storage-sku Premium_LRS \
  --tags Project=Lab13-07
```

The exact instance type depends on regional GPU availability and quota.

---

# Step 5: Run the Optimization Experiments

Apply optimization techniques **individually first**.

After each experiment:

1. Record the new configuration.
2. Run the same workload.
3. Capture the same metrics.
4. Calculate total workload cost.
5. Compare with the baseline.
6. Record operational side effects.

This avoids attributing a savings result to the wrong change.

---

# Experiment 1: Use Spot Capacity with Checkpointing

Spot or interruptible capacity can reduce compute expense for fault-tolerant workloads.

However, capacity may be reclaimed.

Training should therefore use persistent checkpoints.

A checkpoint should contain enough information to resume work, including:

* Model state
* Optimizer state
* Training step or epoch
* Scheduler state if applicable
* Other required training metadata

A simplified PyTorch pattern is:

```python
start_step = load_checkpoint_if_exists(
    model,
    optimizer
)

for step, (x, y) in enumerate(
    loader,
    start=start_step
):
    optimizer.zero_grad(
        set_to_none=True
    )

    yhat = model(
        x.cuda(
            non_blocking=True
        )
    )

    loss = criterion(
        yhat,
        y.cuda(
            non_blocking=True
        )
    )

    loss.backward()

    optimizer.step()

    if step % CKPT_EVERY == 0:
        save_checkpoint(
            model,
            optimizer,
            step
        )
```

Store checkpoints on persistent storage rather than local ephemeral disks.

---

## AWS Spot Example

```bash
aws ec2 run-instances \
  --instance-market-options 'MarketType=spot' \
  --instance-type g5.2xlarge \
  --image-id ami-xxxxxxxx \
  --tag-specifications \
  'ResourceType=instance,Tags=[{Key=Project,Value=Lab13-07}]'
```

---

## Google Cloud Spot Example

```bash
gcloud compute instances create lab13-07-spot \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --provisioning-model=SPOT \
  --boot-disk-size=200GB \
  --labels=project=lab13-07
```

---

## Azure Spot Example

```bash
az vm create \
  -g rg-lab13-07 \
  -n lab13-07-spot \
  --image Ubuntu2204 \
  --size Standard_NC4as_T4_v3 \
  --priority Spot \
  --max-price -1 \
  --eviction-policy Deallocate \
  --tags Project=Lab13-07
```

Record:

* Compute cost
* Job completion time
* Number of interruptions
* Time lost to interruptions
* Checkpoint overhead
* Recovery time

An inexpensive Spot instance is not beneficial if repeated interruptions make the total workload more expensive.

---

# Experiment 2: Enable Autoscaling

Autoscaling allows infrastructure capacity to grow and shrink with demand.

For inference:

```text
Traffic Rises
    ↓
Demand Metric Rises
    ↓
HPA Adds Replicas
    ↓
Node Autoscaler Adds Capacity
    ↓
Traffic Distributed Across
More Inference Workers
```

When traffic falls, excess capacity can be removed.

Production AI services may scale using:

* CPU utilization
* Memory utilization
* GPU utilization
* Request rate
* Queue depth
* Concurrent requests
* Inference latency

---

## Example Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler

metadata:
  name: inference-hpa
  namespace: ai

spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-api

  minReplicas: 2
  maxReplicas: 50

  metrics:
    - type: Resource

      resource:
        name: cpu

        target:
          type: Utilization
          averageUtilization: 70
```

Record:

* Replica count
* Resource utilization
* p50 latency
* p95 latency
* Infrastructure cost
* Scale-up behavior
* Scale-down behavior

---

# Experiment 3: Enable Mixed-Precision Training

Modern GPUs can often perform lower-precision operations more efficiently than full FP32 operations.

Mixed precision can improve:

* Training throughput
* GPU memory efficiency
* Training duration
* Total training cost

while maintaining model quality for many workloads.

A simplified pattern is:

```python
with autocast():

    yhat = model(
        x.cuda(
            non_blocking=True
        )
    )

    loss = criterion(
        yhat,
        y.cuda(
            non_blocking=True
        )
    )

scaler.scale(
    loss
).backward()

scaler.step(
    optimizer
)

scaler.update()
```

Compare this run with the FP32 baseline.

Record:

* Throughput
* GPU memory usage
* Training duration
* Accuracy
* Total compute cost

Do not keep the optimization if model quality falls outside the required tolerance.

---

# Experiment 4: Tune Batch Size and Gradient Accumulation

Batch size directly affects:

* GPU memory consumption
* GPU utilization
* Throughput
* Convergence behavior

Increase batch size gradually while monitoring GPU memory.

For example:

```text
Batch 64
   ↓
Batch 128
   ↓
Batch 256
   ↓
Batch 512
```

Stop increasing when:

* GPU memory becomes constrained
* Performance stops improving
* Model convergence changes undesirably

If the desired effective batch size does not fit in GPU memory, use gradient accumulation.

Conceptually:

```text
Mini-Batch 1 ─┐
Mini-Batch 2 ─┤
Mini-Batch 3 ─┼─► Accumulate Gradients
Mini-Batch 4 ─┘
                    ↓
              Optimizer Step
```

Record:

* Batch size
* Effective batch size
* GPU memory
* Throughput
* Training duration
* Accuracy
* Cost

The objective is not the largest possible batch. It is the configuration that delivers the best cost/performance balance.

---

# Experiment 5: Improve GPU Utilization

Low GPU utilization often means the accelerator is waiting for something else.

Potential bottlenecks include:

```text
CPU Preprocessing
       ↓
Data Loader
       ↓
Storage
       ↓
Network
       ↓
GPU
```

Before purchasing a larger GPU, determine whether the current accelerator is being fed efficiently.

Potential improvements include:

* Local NVMe caching
* More DataLoader workers
* Pinned memory
* Persistent DataLoader workers
* Asynchronous transfers
* More efficient preprocessing

Example:

```python
loader = DataLoader(
    ds,
    batch_size=B,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)
```

Measure the impact rather than assuming that more workers or caching will always improve the workload.

---

# Experiment 6: Optimize Storage Lifecycle

AI workflows often accumulate:

* Checkpoints
* Logs
* Model versions
* Intermediate files
* Evaluation outputs
* Temporary artifacts

Keeping every object indefinitely in high-cost storage can generate unnecessary expense.

A lifecycle policy can:

```text
New Artifact
     ↓
Standard Storage
     ↓
Lower-Cost Storage
     ↓
Archive
     ↓
Delete After Retention Period
```

Before applying lifecycle policies, consider:

* Retrieval latency
* Retrieval charges
* Minimum storage duration
* Data retention requirements
* Compliance requirements

---

## Example Amazon S3 Lifecycle Policy

```json
{
  "Rules": [
    {
      "ID": "MoveOlderArtifacts",
      "Filter": {
        "Prefix": "experiments/"
      },
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 14,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 45,
          "StorageClass": "GLACIER"
        }
      ],
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 30,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```

Google Cloud Storage and Azure Blob Storage provide comparable lifecycle-management capabilities.

Record the expected or measured effect on:

* Storage cost
* Retrieval cost
* Data accessibility
* Operational complexity

---

# Experiment 7: Improve Data Locality

Large datasets can generate significant network charges when compute and storage are placed in different regions.

Poor locality:

```text
Dataset
Region A
    │
    │ Cross-Region Transfer
    ▼
GPU Compute
Region B
```

Improved locality:

```text
Region A
   │
   ├── Dataset
   ├── Checkpoints
   └── GPU Compute
```

Whenever architectural and compliance requirements permit, place:

* Compute
* Primary dataset
* Checkpoints
* Frequently accessed object storage

in the same region.

Record:

* Network charges
* Data startup time
* Training throughput
* Total workload cost

This can be especially important for workloads that repeatedly read very large datasets.

---

# Experiment 8: Optimize the Container and Runtime

Container and framework configuration can also influence AI infrastructure efficiency.

Use runtime images that contain:

* Required CUDA libraries
* Required ML frameworks
* Necessary runtime dependencies

Avoid unnecessary:

* Development tools
* Build toolchains
* Large unused packages

Smaller images may reduce:

* Image transfer time
* Container startup time
* Registry storage

although GPU compute generally remains the larger cost driver.

---

## PyTorch Runtime Optimization

For compatible models and PyTorch versions, evaluate:

```python
model = torch.compile(
    model
)
```

Compare:

* Training duration
* Inference throughput
* Startup overhead
* Model correctness
* Total cost

Compilation does not improve every workload, so measure the actual result.

---

# Experiment 9: Optimize Inference with Dynamic Batching

GPU inference becomes more cost-efficient when more useful work can be performed per accelerator execution cycle.

NVIDIA Triton Inference Server supports **dynamic batching**, which combines compatible requests.

Without batching:

```text
Request 1 → GPU
Request 2 → GPU
Request 3 → GPU
Request 4 → GPU
```

With dynamic batching:

```text
Request 1 ─┐
Request 2 ─┤
Request 3 ─┼─► Batch ─► GPU
Request 4 ─┘
```

This can increase throughput and GPU utilization.

---

## Example Triton Configuration

```text
max_batch_size: 64

dynamic_batching {
  preferred_batch_size: [4, 8, 16, 32]
  max_queue_delay_microseconds: 1000
}

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
```

Test multiple:

* Preferred batch sizes
* Queue delays
* Concurrency levels

Measure:

* Throughput
* GPU utilization
* p50 latency
* p95 latency
* Infrastructure cost

The goal is to improve throughput without violating application latency requirements.

---

## Optional Response Caching

For deterministic or safely reusable responses, application-level caching may provide further savings.

For example:

```text
Request
   ↓
Cache Lookup
   │
   ├── Hit ─────► Return Cached Result
   │
   └── Miss ────► GPU Inference
```

A cache such as Redis can reduce repeated inference work.

Only cache requests where reuse is semantically correct and compatible with freshness requirements.

---

# Experiment 10: Evaluate Commitment-Based Pricing

Spot capacity is appropriate for interruptible workloads, but not every workload is intermittent.

Some systems maintain predictable baseline usage.

Examples include:

* Continuously running inference services
* Stable production GPU capacity
* Long-running training infrastructure

For these workloads, evaluate commitment-based pricing options such as:

* AWS Savings Plans
* AWS Reserved Instances where applicable
* Google Cloud committed-use discounts
* Azure reservations or savings mechanisms

You do **not** need to purchase a commitment for this lab.

Instead, estimate:

```text
Monthly On-Demand Cost
        vs.
Estimated Commitment Cost
```

Record the result as a planning scenario rather than a measured live experiment.

---

# Step 6: Configure Spot-Aware Kubernetes Scheduling

A Kubernetes cluster can combine:

```text
Standard Nodes
+
Spot Nodes
```

Critical services can remain on stable capacity while restartable training jobs use less expensive interruptible capacity.

Useful Kubernetes mechanisms include:

* Labels
* Taints
* Tolerations
* Node selectors
* Affinity
* Workload priorities

A hybrid cluster can look like:

```text
                    Kubernetes Cluster
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
          Standard Node Pool     Spot Node Pool
                 │                   │
                 ▼                   ▼
       Critical Inference       Training Jobs
       Control Services         Batch Workloads
```

Training jobs placed on Spot capacity should save checkpoints to persistent storage.

---

## Simplified Scheduling Example

```yaml
apiVersion: apps/v1
kind: Deployment

metadata:
  name: trainer
  namespace: ai

spec:
  replicas: 1

  selector:
    matchLabels:
      app: trainer

  template:
    metadata:
      labels:
        app: trainer

    spec:
      tolerations:
        - key: "spot"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"

      nodeSelector:
        lifecycle: spot

      containers:
        - name: trainer
          image: yourrepo/trainer:latest

          resources:
            limits:
              nvidia.com/gpu: "1"
```

> **Kubernetes Note**
>
> A finite training workload is generally better represented as a Kubernetes `Job` than a `Deployment`. The example above focuses specifically on the Spot scheduling concepts.

---

# Step 7: Run, Measure, and Record

Repeat the same measurement workflow for every major optimization.

Use:

```text
Change One Variable
       ↓
Run Workload
       ↓
Collect Metrics
       ↓
Calculate Cost
       ↓
Record Results
       ↓
Compare with Baseline
```

Add one row to the Cost Tracking Worksheet after each experiment.

Include short notes describing:

* What changed
* Whether performance improved
* Whether accuracy changed
* Operational side effects
* Failures or interruptions
* Additional complexity introduced

---

## Recommended Experiment Order

A practical sequence is:

```text
Spot
  ↓
Mixed Precision
  ↓
Batch Size / Gradient Accumulation
  ↓
Data Locality
  ↓
Storage Lifecycle
  ↓
Autoscaling
  ↓
Runtime Optimization
  ↓
Inference Batching / Caching
```

After testing each technique individually, combine the strongest optimizations.

---

# Step 8: Build the Final Optimized Configuration

Once the individual experiments are complete, select the best-performing techniques and combine them.

For example:

```text
Spot GPU
   +
Checkpointing
   +
Mixed Precision
   +
Optimized Batch Size
   +
Efficient DataLoader
   +
Same-Region Data
   +
Storage Lifecycle
   +
Runtime Optimization
```

Run the workload again.

This final experiment is important because individual optimizations may interact with each other.

An optimization that works well independently may provide less benefit when combined with another change.

---

# Step 9: Analyze the Results

Compare the baseline and experimental runs.

A useful table is:

| Experiment           |   Cost | Throughput | Duration | GPU Util. | Accuracy | Savings |
| -------------------- | -----: | ---------: | -------: | --------: | -------: | ------: |
| Baseline             | $_____ |      _____ |    _____ |    _____% |   _____% |      0% |
| Spot                 | $_____ |      _____ |    _____ |    _____% |   _____% |  _____% |
| Mixed Precision      | $_____ |      _____ |    _____ |    _____% |   _____% |  _____% |
| Batch Tuning         | $_____ |      _____ |    _____ |    _____% |   _____% |  _____% |
| Data Locality        | $_____ |      _____ |    _____ |    _____% |   _____% |  _____% |
| Autoscaling          | $_____ |      _____ |    _____ |    _____% |   _____% |  _____% |
| Runtime Optimization | $_____ |      _____ |    _____ |    _____% |   _____% |  _____% |
| Final Combined       | $_____ |      _____ |    _____ |    _____% |   _____% |  _____% |

For inference experiments, also record:

| Experiment       |   Cost | Throughput |      p50 |      p95 | Error Rate |
| ---------------- | -----: | ---------: | -------: | -------: | ---------: |
| Baseline         | $_____ |      _____ | _____ ms | _____ ms |     _____% |
| Dynamic Batching | $_____ |      _____ | _____ ms | _____ ms |     _____% |
| Autoscaling      | $_____ |      _____ | _____ ms | _____ ms |     _____% |
| Cache            | $_____ |      _____ | _____ ms | _____ ms |     _____% |

---

## Cost vs. Performance

Do not automatically select the lowest-cost configuration.

For example:

```text
Configuration A
Cost: $2.00
Runtime: 30 minutes

Configuration B
Cost: $1.50
Runtime: 3 hours
```

Configuration B has a lower apparent infrastructure price, but it may provide poorer operational value.

Evaluate:

```text
Cost
+
Performance
+
Reliability
+
Latency
+
Model Quality
+
Operational Complexity
```

together.

---

## Useful Cost-Efficiency Metrics

For training:

```text
Cost per Training Run
```

or:

```text
Cost per Valid Model
```

For inference:

```text
Cost per 1,000 Requests
```

or:

```text
Cost per 1 Million Inferences
```

For token-based workloads:

```text
Cost per 1 Million Tokens
```

These metrics often provide more useful engineering insight than hourly GPU cost alone.

---

# Step 10: Capture the Lab Results

Record:

* Baseline infrastructure
* GPU type
* Region
* Instance pricing model
* Training duration
* Throughput
* GPU utilization
* GPU memory utilization
* Model accuracy
* p50 latency
* p95 latency
* Compute cost
* Storage cost
* Network cost
* Total workload cost
* Percentage savings
* Operational tradeoffs

Also identify:

```text
Most Effective Optimization:
____________________________

Least Effective Optimization:
____________________________

Final Recommended Configuration:
____________________________
```

---

# Expected Outcomes

There is no universal savings percentage.

Results depend on:

* Workload behavior
* GPU type
* Cloud pricing
* Region
* Spot availability
* Storage usage
* Network patterns
* Model characteristics
* Optimization maturity

You may observe that:

* Spot capacity reduces training cost substantially for restartable workloads.
* Mixed precision shortens training by improving GPU efficiency.
* Batch tuning increases GPU utilization.
* Improved data locality reduces network expense.
* Lifecycle policies reduce long-term artifact-storage cost.
* Autoscaling reduces idle inference capacity.
* Dynamic batching improves inference throughput.
* Caching eliminates some repeated inference computation.

The largest savings often result from combining complementary techniques.

---

# Troubleshooting

## GPU Utilization Is Low

Check:

* DataLoader workers
* CPU utilization
* Storage latency
* Network throughput
* Preprocessing overhead
* Batch size

Do not immediately assume that a larger GPU is required.

---

## Spot Jobs Fail Repeatedly

Check:

* Checkpoint frequency
* Checkpoint persistence
* Spot availability
* Recovery logic
* Alternative instance types
* Alternative availability zones or regions

---

## Mixed Precision Changes Model Quality

Compare:

* Validation accuracy
* Training stability
* Loss progression

Do not retain mixed precision if model quality falls outside acceptable tolerance.

---

## Autoscaling Does Not Respond

Verify:

```bash
kubectl get hpa
```

Check metrics availability:

```bash
kubectl top pods
```

Verify that the autoscaler has access to the required CPU, memory, or custom metrics.

---

## Training Is Storage-Bound

Check:

```bash
iostat -xz 5
```

Consider:

* Local caching
* Faster storage
* More efficient file formats
* Larger sequential reads
* Data sharding
* Prefetching

---

## Network Cost Is High

Verify whether:

* Dataset and compute are in different regions.
* Checkpoints are written cross-region.
* Inference traffic crosses zones or regions unnecessarily.
* Large container images are repeatedly transferred.

---

# Clean Up

Terminate all cloud resources created for the lab.

Verify:

* GPU instances stopped or terminated
* Spot instances terminated
* Persistent disks removed if unnecessary
* Kubernetes GPU nodes removed
* Load balancers deleted
* Public IP addresses released
* Temporary object-storage data deleted if appropriate

For Kubernetes resources:

```bash
kubectl get all -A
```

Review cloud billing dashboards after cleanup.

> **⚠️ Important**
>
> GPU instances, persistent disks, public IP addresses, and managed networking resources may continue generating charges if they are left provisioned after the experiment.

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Created an unoptimized baseline
* [ ] Recorded the baseline in the Cost Tracking Worksheet
* [ ] Monitored GPU utilization
* [ ] Monitored storage behavior
* [ ] Monitored network behavior
* [ ] Applied consistent resource tags or labels
* [ ] Tested Spot capacity
* [ ] Implemented or reviewed checkpointing
* [ ] Tested autoscaling
* [ ] Tested mixed precision
* [ ] Tuned batch size
* [ ] Evaluated gradient accumulation
* [ ] Investigated GPU utilization bottlenecks
* [ ] Evaluated storage lifecycle optimization
* [ ] Evaluated data locality
* [ ] Tested runtime optimization
* [ ] Evaluated dynamic inference batching
* [ ] Considered caching where appropriate
* [ ] Estimated commitment-based pricing
* [ ] Evaluated Spot-aware Kubernetes scheduling
* [ ] Recorded each experiment separately
* [ ] Calculated total workload cost
* [ ] Calculated percentage savings
* [ ] Compared performance and model quality
* [ ] Combined the strongest optimizations
* [ ] Completed a final optimized run
* [ ] Removed billable cloud resources

---

# Learning Outcomes

After completing this lab, you should be able to:

* Establish a measurable cloud AI cost baseline.
* Connect infrastructure cost with actual resource utilization.
* Calculate total workload cost rather than focusing only on hourly GPU price.
* Evaluate Spot capacity for fault-tolerant AI training.
* Use checkpointing to reduce the impact of interruptions.
* Apply autoscaling to variable AI workloads.
* Use mixed precision to improve GPU efficiency.
* Tune batch size and gradient accumulation.
* Diagnose underutilized GPU infrastructure.
* Optimize storage lifecycle and data locality.
* Evaluate runtime and container optimizations.
* Improve inference efficiency through dynamic batching and caching.
* Evaluate commitment-based pricing for predictable workloads.
* Schedule restartable training workloads on Spot Kubernetes nodes.
* Compare cost, performance, reliability, and model quality together.
* Build a repeatable framework for continuous AI infrastructure cost optimization.

---

# Key Takeaway

**Cloud AI cost optimization is not about choosing the GPU with the lowest hourly price. The meaningful metric is the total cost required to complete a useful training or inference workload while meeting performance, reliability, latency, and model-quality requirements.**

Spot capacity, autoscaling, mixed precision, batch tuning, data locality, storage lifecycle management, efficient runtime configuration, dynamic batching, and other techniques address different sources of infrastructure waste. The strongest results generally come from combining complementary optimizations and continuously measuring their effect rather than applying them blindly.

Cost optimization should therefore be treated as an ongoing engineering discipline:

```text
Measure
   ↓
Identify Waste
   ↓
Optimize
   ↓
Validate
   ↓
Measure Again
```
