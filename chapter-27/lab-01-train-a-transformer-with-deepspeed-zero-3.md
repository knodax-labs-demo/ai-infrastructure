# Hands-On Lab: Train a Transformer with DeepSpeed ZeRO-3

In this lab, you will train a Transformer model across multiple GPUs using **DeepSpeed ZeRO Stage 3 (ZeRO-3)**.

Unlike conventional data-parallel training, where each GPU maintains a complete copy of the model parameters, gradients, and optimizer state, ZeRO-3 partitions these training states across participating workers. This substantially reduces per-GPU memory requirements and enables larger models or batch configurations to run on the same hardware.

You will fine-tune **BERT** on the IMDb sentiment dataset, integrate DeepSpeed with the Hugging Face `Trainer`, enable optional CPU optimizer offloading, monitor GPU memory and throughput, save distributed checkpoints, and resume training after interruption.

The complete architecture is:

```text
Dataset
   ↓
Hugging Face Trainer
   ↓
DeepSpeed ZeRO-3
   ↓
Distributed Workers
   ├── GPU 0 → Parameter / Gradient / Optimizer Shards
   ├── GPU 1 → Parameter / Gradient / Optimizer Shards
   ├── GPU 2 → Parameter / Gradient / Optimizer Shards
   └── GPU N → Parameter / Gradient / Optimizer Shards
              ↓
       Distributed Training
              ↓
        ZeRO Checkpoint
              ↓
    Consolidated Model for Inference
```

---

## Lab Objectives

By completing this lab, you will learn how to:

* Configure DeepSpeed ZeRO-3.
* Integrate DeepSpeed with Hugging Face `Trainer`.
* Train a Transformer across multiple GPUs.
* Understand parameter, gradient, and optimizer-state sharding.
* Calculate effective global batch size.
* Use CPU optimizer offloading.
* Monitor GPU memory and throughput.
* Compare ZeRO-3 with conventional distributed training.
* Save and resume distributed checkpoints.
* Distinguish training checkpoints from deployment artifacts.
* Understand when ZeRO-3 becomes valuable for larger models.

---

## Estimated Time

**Approximately 120–180 minutes**

---

## Tools

This lab uses:

* Linux
* Python
* PyTorch
* CUDA
* NCCL
* DeepSpeed
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Accelerate
* NVIDIA GPUs

---

# Step 1: Verify the Prerequisites

This lab requires a Linux-based system or cluster with multiple NVIDIA GPUs.

Verify GPU availability:

```bash
nvidia-smi
```

Verify that PyTorch can detect CUDA:

```bash
python -c \
'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

Install the required packages:

```bash
pip install \
  torch \
  transformers \
  datasets \
  accelerate \
  deepspeed
```

Verify DeepSpeed:

```bash
deepspeed --version
```

A working environment should provide:

```text
NVIDIA Driver
      ↓
CUDA
      ↓
PyTorch
      ↓
NCCL
      ↓
DeepSpeed
```

---

# Step 2: Understand Why ZeRO-3 Is Needed

Conventional distributed data parallelism typically replicates training state.

With four GPUs:

```text
GPU 0
├── Full Parameters
├── Full Gradients
└── Full Optimizer State

GPU 1
├── Full Parameters
├── Full Gradients
└── Full Optimizer State

GPU 2
├── Full Parameters
├── Full Gradients
└── Full Optimizer State

GPU 3
├── Full Parameters
├── Full Gradients
└── Full Optimizer State
```

This creates substantial memory duplication.

ZeRO-3 changes the architecture:

```text
                Distributed Model State

GPU 0 → Parameter Shard 0
GPU 1 → Parameter Shard 1
GPU 2 → Parameter Shard 2
GPU 3 → Parameter Shard 3

GPU 0 → Gradient Shard 0
GPU 1 → Gradient Shard 1
GPU 2 → Gradient Shard 2
GPU 3 → Gradient Shard 3

GPU 0 → Optimizer Shard 0
GPU 1 → Optimizer Shard 1
GPU 2 → Optimizer Shard 2
GPU 3 → Optimizer Shard 3
```

The main goal is:

```text
Reduce Replicated Training State
              ↓
Reduce Per-GPU Memory
              ↓
Train Larger Workloads
```

---

# Step 3: Understand the ZeRO Stages

The ZeRO family progressively partitions training state.

```text
ZeRO-1
   ↓
Shard Optimizer State

ZeRO-2
   ↓
Shard Optimizer State
+
Shard Gradients

ZeRO-3
   ↓
Shard Optimizer State
+
Shard Gradients
+
Shard Parameters
```

A simplified comparison is:

| Training Mode | Optimizer State | Gradients  | Parameters |
| ------------- | --------------- | ---------- | ---------- |
| DDP           | Replicated      | Replicated | Replicated |
| ZeRO-1        | Sharded         | Replicated | Replicated |
| ZeRO-2        | Sharded         | Sharded    | Replicated |
| ZeRO-3        | Sharded         | Sharded    | Sharded    |

ZeRO-3 therefore provides the most aggressive model-state sharding of the three stages.

---

# Step 4: Create the Training Script

Create:

```text
lab-07-train-ds-zero3.py
```

Start with:

```python
from datasets import load_dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "bert-base-uncased"


dataset = load_dataset(
    "imdb"
)


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


tokenized_dataset = (
    dataset.map(
        tokenize,
        batched=True
    )
)


tokenized_dataset.set_format(
    "torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label",
    ],
)


model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
)
```

---

# Step 5: Understand the Dataset

IMDb provides movie reviews labeled as:

```text
Positive
or
Negative
```

The model input path is:

```text
Movie Review
     ↓
Tokenizer
     ↓
input_ids
attention_mask
     ↓
BERT
     ↓
Classification Head
     ↓
Positive / Negative
```

The lab uses BERT Base because it is large enough to demonstrate the distributed workflow while remaining practical for experimentation.

---

# Step 6: Create the ZeRO-3 Configuration

Create:

```text
lab-07-ds-config-zero3.json
```

Add:

```json
{
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 2,

  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 500000000,
    "stage3_param_persistence_threshold": 100000,

    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },

  "bf16": {
    "enabled": true
  },

  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

---

# Step 7: Understand the ZeRO-3 Configuration

The critical setting is:

```json
"stage": 3
```

which enables parameter, gradient, and optimizer-state partitioning.

Other important settings include:

| Setting                              | Purpose                                      |
| ------------------------------------ | -------------------------------------------- |
| `overlap_comm`                       | Overlap communication with computation       |
| `contiguous_gradients`               | Reduce gradient-memory fragmentation         |
| `reduce_bucket_size`                 | Control collective communication bucket size |
| `stage3_prefetch_bucket_size`        | Control parameter prefetching                |
| `stage3_param_persistence_threshold` | Keep selected small parameters resident      |
| `offload_optimizer`                  | Move optimizer state to CPU memory           |

These values are reasonable for experimentation but may require tuning on larger systems.

---

# Step 8: Understand CPU Optimizer Offloading

The configuration uses:

```json
"offload_optimizer": {
  "device": "cpu",
  "pin_memory": true
}
```

The architecture becomes:

```text
GPU
├── Parameter Shard
├── Gradient Shard
├── Activations
└── Temporary Buffers

CPU
└── Optimizer State
```

The benefit is:

```text
Less GPU Memory
```

but the tradeoff is:

```text
Additional CPU Memory
+
CPU ↔ GPU Data Movement
+
Potentially Lower Throughput
```

Offloading therefore improves capacity, not necessarily speed.

---

# Step 9: Understand BF16

The configuration enables:

```json
"bf16": {
  "enabled": true
}
```

BF16 can reduce memory usage and improve accelerator throughput on supported GPUs.

However, hardware support varies.

Conceptually:

```text
FP32
 ↓
Higher Precision
Higher Memory

BF16
 ↓
Lower Memory
Higher Tensor Throughput
```

If the target hardware does not support BF16 efficiently, use an appropriate FP16 configuration instead.

---

# Step 10: Understand the Global Batch Size

The effective global batch size is:

```text
Global Batch Size
=
Micro-Batch Size per GPU
×
Gradient Accumulation Steps
×
Data-Parallel World Size
```

With:

```text
Micro-Batch Size = 4
Gradient Accumulation = 2
Workers = 2
```

the result is:

```text
4 × 2 × 2 = 16
```

If you scale to four workers:

```text
4 × 2 × 4 = 32
```

Changing GPU count can therefore change the optimization configuration.

---

# Step 11: Integrate DeepSpeed with Hugging Face Trainer

Add to the Python script:

```python
training_args = TrainingArguments(
    output_dir="./outputs",

    per_device_train_batch_size=4,

    gradient_accumulation_steps=2,

    eval_strategy="steps",

    eval_steps=500,

    num_train_epochs=1,

    save_steps=500,

    logging_steps=50,

    report_to="none",

    deepspeed=(
        "lab-07-ds-config-zero3.json"
    ),
)


train_dataset = (
    tokenized_dataset[
        "train"
    ]
    .shuffle(seed=42)
    .select(
        range(5000)
    )
)


eval_dataset = (
    tokenized_dataset[
        "test"
    ]
    .select(
        range(1000)
    )
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


trainer.train()
```

---

# Step 12: Understand Trainer and DeepSpeed Responsibilities

The high-level division is:

```text
Hugging Face Trainer
├── Training Loop
├── Dataset Iteration
├── Evaluation
├── Logging
└── Checkpoint Scheduling

DeepSpeed
├── Distributed Initialization
├── ZeRO State Partitioning
├── Optimizer Integration
├── Communication
└── Distributed Checkpoint State
```

This allows application code to remain familiar while DeepSpeed handles lower-level distributed infrastructure.

---

# Step 13: Run Training

Launch with two GPUs:

```bash
deepspeed \
  --num_gpus=2 \
  lab-07-train-ds-zero3.py
```

DeepSpeed starts one worker per GPU.

Conceptually:

```text
DeepSpeed Launcher
       ↓
Worker 0 → GPU 0
Worker 1 → GPU 1
       ↓
ZeRO-3 Distributed Runtime
```

---

# Step 14: Observe ZeRO-3 State Sharding

During execution:

```text
Worker 0
├── Parameter Shard A
├── Gradient Shard A
└── Optimizer State Offloaded

Worker 1
├── Parameter Shard B
├── Gradient Shard B
└── Optimizer State Offloaded
```

When computation requires parameters owned elsewhere, DeepSpeed coordinates the necessary communication.

The model behaves logically as one training model even though the training state is physically distributed.

---

# Step 15: Monitor GPU Memory

Open another terminal:

```bash
watch -n 1 nvidia-smi
```

Record:

* Memory usage
* GPU utilization
* Power
* Temperature
* Worker processes

Do not assume that GPU memory will equal only the local model-state shard.

GPU memory also contains:

```text
Activations
Temporary Buffers
Communication Buffers
CUDA Context
Framework Allocations
```

---

# Step 16: Establish a DDP Baseline

Run the same BERT workload without:

```python
deepspeed="lab-07-ds-config-zero3.json"
```

Record:

| Metric               | DDP / Standard | ZeRO-3 |
| -------------------- | -------------: | -----: |
| Peak GPU memory      |         Record | Record |
| Runtime              |         Record | Record |
| Samples/sec          |         Record | Record |
| GPU utilization      |         Record | Record |
| Maximum stable batch |         Record | Record |

This comparison demonstrates that ZeRO-3 primarily targets **memory scalability**.

---

# Step 17: Measure Memory Savings

A simplified relationship is:

```text
DDP:
Each GPU ≈ Full Training State

ZeRO-3:
Each GPU ≈ Fraction of Training State
```

As worker count increases:

```text
More Workers
      ↓
Smaller State Shard per Worker
```

although actual memory use depends on many additional runtime allocations.

---

# Step 18: Measure Throughput

Memory savings do not guarantee faster training.

Record:

```text
Samples / Second
```

and:

```text
Training Step Duration
```

The tradeoff may be:

```text
Less Memory
   ↕
More Communication
```

and, when offloading:

```text
Less GPU Memory
   ↕
More CPU-GPU Transfer
```

---

# Step 19: Understand Why BERT May Show Modest Savings

BERT Base is relatively small compared with modern LLMs.

Therefore:

```text
BERT Base
     ↓
ZeRO-3 Benefit Visible
But Not Dramatic
```

For multi-billion-parameter models:

```text
Very Large Model
      ↓
Replicated Training State
May Exceed GPU Memory
      ↓
ZeRO-3 Becomes Much More Valuable
```

The lab demonstrates the mechanism before applying it at larger scale.

---

# Step 20: Understand Communication Overhead

ZeRO-3 reduces memory duplication by increasing distributed coordination.

Conceptually:

```text
Need Parameter
     ↓
Gather Parameter Shards
     ↓
Compute
     ↓
Release / Reshard
```

Training performance therefore depends on:

* GPU compute
* Interconnect bandwidth
* Network latency
* Collective communication
* Bucket sizing
* CPU offload speed

---

# Step 21: Understand Compute vs. Communication

Efficient scaling requires:

```text
Useful GPU Compute
       >
Communication Overhead
```

If communication dominates:

```text
More GPUs
   ↓
More Coordination
   ↓
Limited Speedup
```

This is why high-speed GPU interconnects and networking become increasingly important for distributed LLM training.

---

# Step 22: Save ZeRO-3 Checkpoints

The Hugging Face Trainer creates checkpoint directories under:

```text
outputs/
```

For example:

```text
outputs/
└── checkpoint-500/
```

A ZeRO-3 checkpoint may contain distributed training state rather than only one conventional model file.

Conceptually:

```text
Checkpoint
├── Parameter Shards
├── Optimizer State
├── Scheduler / Trainer State
└── Training Metadata
```

---

# Step 23: Resume from a Checkpoint

Modify the training call:

```python
trainer.train(
    resume_from_checkpoint=(
        "./outputs/checkpoint-500"
    )
)
```

Then relaunch:

```bash
deepspeed \
  --num_gpus=2 \
  lab-07-train-ds-zero3.py
```

The expected path is:

```text
Checkpoint
     ↓
DeepSpeed + Trainer
     ↓
Restore Distributed State
     ↓
Resume Training
```

---

# Step 24: Validate Recovery

Do not assume recovery works merely because checkpoint files exist.

Test it:

1. Start training.
2. Wait for a checkpoint.
3. Stop the workload.
4. Relaunch from the checkpoint.
5. Confirm that the training step resumes.
6. Verify continued loss progression.

The infrastructure principle is:

```text
Checkpoint Creation
        ≠
Recovery Validation
```

A checkpoint is useful only if it can be restored successfully.

---

# Step 25: Use Durable Checkpoint Storage

Local node storage is vulnerable to instance failure.

Production checkpoints should reside on durable storage such as:

```text
Distributed Filesystem
Object Storage
Persistent Volume
Shared High-Performance Storage
```

A common pattern is:

```text
Training Cluster
      ↓
Distributed Checkpoint
      ↓
Durable Storage
      ↓
Restarted Cluster
      ↓
Restore
```

---

# Step 26: Distinguish Training Checkpoints from Deployment Models

A ZeRO-3 training checkpoint serves:

```text
Resume Distributed Training
```

A deployment artifact serves:

```text
Load Model for Inference
```

These are not necessarily the same thing.

The lifecycle is:

```text
Distributed Training
        ↓
ZeRO Checkpoint
        ↓
Consolidate Weights
        ↓
Standard Model Artifact
        ↓
Inference Runtime
```

---

# Step 27: Understand Model Consolidation

ZeRO-3 partitions parameters across workers.

Inference environments frequently expect:

```text
Complete Model State
```

DeepSpeed provides mechanisms to reconstruct consolidated weights from distributed checkpoints.

This operation can require substantial CPU memory for large models.

The key architectural distinction is:

```text
Training State
      ≠
Deployment Artifact
```

---

# Step 28: Experiment with CPU Offloading Disabled

Remove:

```json
"offload_optimizer": {
  "device": "cpu",
  "pin_memory": true
}
```

Run again.

Compare:

| Metric     | CPU Offload | No CPU Offload |
| ---------- | ----------: | -------------: |
| GPU memory |      Record |         Record |
| CPU memory |      Record |         Record |
| Runtime    |      Record |         Record |
| Throughput |      Record |         Record |

You may observe:

```text
CPU Offload
   ↓
Lower GPU Memory
   ↓
Potentially Lower Throughput
```

---

# Step 29: Experiment with Larger Batch Sizes

Increase:

```json
"train_micro_batch_size_per_gpu"
```

gradually.

For example:

```text
4
↓
8
↓
16
```

Record the largest configuration that trains reliably.

One of ZeRO-3's practical advantages is that reduced model-state memory can leave more GPU memory available for:

```text
Activations
+
Larger Micro-Batches
```

---

# Step 30: Experiment with Gradient Accumulation

Try:

```text
1
2
4
8
```

for:

```json
"gradient_accumulation_steps"
```

The effective batch size is:

```text
Micro-Batch
×
Accumulation
×
Workers
```

This provides another way to increase effective batch size without increasing the instantaneous per-GPU micro-batch.

---

# Step 31: Experiment with a Larger Transformer

After BERT works, try:

```text
roberta-large
```

or another appropriately sized model.

The expected progression is:

```text
BERT Base
     ↓
Verify ZeRO-3
     ↓
RoBERTa Large
     ↓
Observe Greater Memory Benefit
```

Increase workload size gradually rather than jumping directly to the largest model that might fit.

---

# Step 32: Compare ZeRO-2 and ZeRO-3

Run equivalent workloads with:

```text
ZeRO-2
```

and:

```text
ZeRO-3
```

Record:

| Metric                   | ZeRO-2 | ZeRO-3 |
| ------------------------ | -----: | -----: |
| Peak memory              | Record | Record |
| Throughput               | Record | Record |
| Communication overhead   | Record | Record |
| Maximum model/batch size | Record | Record |

The typical tradeoff is:

```text
More Aggressive Sharding
       ↓
Lower Memory
       ↕
More Communication
```

---

# Step 33: Compare DDP, ZeRO-2, and ZeRO-3

Conceptually:

```text
DDP
 ↓
Replicated Training State
 ↓
Highest Memory Duplication

ZeRO-2
 ↓
Shard Gradients + Optimizer
 ↓
Moderate Memory Reduction

ZeRO-3
 ↓
Shard Parameters + Gradients + Optimizer
 ↓
Maximum State Sharding
```

Which configuration is best depends on workload size and hardware.

---

# Step 34: Understand ZeRO-Infinity

For very large models, DeepSpeed can extend offloading beyond GPU and CPU memory.

Conceptually:

```text
GPU
 ↓
CPU
 ↓
NVMe
```

This enables training workloads whose state exceeds aggregate GPU memory.

However:

```text
More Offloading
     ↓
More Data Movement
     ↓
Potentially Lower Throughput
```

Capacity and performance must therefore be balanced carefully.

---

# Step 35: Monitor CPU Memory

Because optimizer offloading is enabled, also monitor host memory.

On Linux:

```bash
free -h
```

or:

```bash
htop
```

The complete memory architecture is:

```text
GPU Memory
├── Parameter Shards
├── Gradient Shards
├── Activations
└── Buffers

CPU Memory
├── Optimizer State
└── Runtime Data
```

GPU memory savings can shift pressure to CPU memory.

---

# Step 36: Monitor PCIe and Network Activity

Offload and distributed sharding create additional data movement.

Potential paths include:

```text
CPU RAM
   ↕
PCIe
   ↕
GPU
```

and:

```text
GPU
   ↕
NVLink / PCIe
   ↕
GPU
```

across nodes:

```text
GPU
   ↕
NIC
   ↕
Network
   ↕
NIC
   ↕
GPU
```

For large-scale training, these paths become critical performance factors.

---

# Step 37: Measure Scaling Efficiency

Compare throughput on:

```text
1 GPU
2 GPUs
4 GPUs
```

If:

```text
T1 = throughput on 1 GPU
TN = throughput on N GPUs
```

then:

```text
Scaling Efficiency
=
TN
────────
N × T1
× 100%
```

This identifies whether adding GPUs produces useful performance gains.

---

# Step 38: Understand Memory Efficiency vs. Compute Efficiency

ZeRO-3 can improve:

```text
Memory Efficiency
```

without necessarily maximizing:

```text
Compute Efficiency
```

The production objective is therefore not:

```text
Use the Most Aggressive Sharding
```

but:

```text
Use Enough Sharding
to Fit the Workload
while Preserving Acceptable Throughput
```

---

# Step 39: Record Benchmark Results

Create a table such as:

| Configuration        |   GPUs | Peak GPU Memory | Runtime | Samples/sec | Max Batch |
| -------------------- | -----: | --------------: | ------: | ----------: | --------: |
| Standard / DDP       | Record |          Record |  Record |      Record |    Record |
| ZeRO-2               | Record |          Record |  Record |      Record |    Record |
| ZeRO-3               | Record |          Record |  Record |      Record |    Record |
| ZeRO-3 + CPU Offload | Record |          Record |  Record |      Record |    Record |

Change only one major variable at a time.

---

# Step 40: Understand Experimental Discipline

Avoid changing:

```text
ZeRO Stage
Batch Size
Precision
GPU Count
Offload
Model Size
```

all at once.

Instead:

```text
Baseline
   ↓
Change One Variable
   ↓
Measure
   ↓
Record
   ↓
Compare
```

This makes benchmark results interpretable.

---

# Step 41: Production Considerations

A production ZeRO-3 workflow should consider:

* GPU topology
* Interconnect bandwidth
* Multi-node networking
* CPU memory capacity
* NVMe performance
* Distributed checkpointing
* Durable storage
* Failure recovery
* Scheduler integration
* Dataset throughput
* Monitoring
* Experiment tracking
* Model versioning
* Artifact consolidation

---

# Step 42: Multi-Node ZeRO-3

The same model can scale beyond one machine.

Conceptually:

```text
Node 0
├── GPU 0
└── GPU 1
      │
      │ High-Speed Network
      │
Node 1
├── GPU 0
└── GPU 1
```

ZeRO shards model state across the complete data-parallel process group.

At larger scale, network performance becomes increasingly important.

---

# Step 43: Scheduler Integration

Production training is typically launched through:

```text
Slurm
```

or:

```text
Kubernetes
```

or another cluster scheduler.

The architecture becomes:

```text
Training Configuration
        ↓
Scheduler
        ↓
Allocate GPU Nodes
        ↓
Launch DeepSpeed Workers
        ↓
Train
        ↓
Checkpoint
        ↓
Monitor / Recover
```

---

# Step 44: Additional Challenges

After completing the core lab, try:

* Replace BERT Base with RoBERTa Large.
* Compare DDP, ZeRO-2, and ZeRO-3.
* Disable CPU optimizer offloading.
* Increase micro-batch size.
* Change gradient accumulation.
* Measure checkpoint write time.
* Measure checkpoint recovery time.
* Run across multiple nodes.
* Measure communication overhead.
* Explore ZeRO-Infinity.
* Track CPU and GPU memory simultaneously.
* Measure scaling efficiency.

---

# Step 45: Clean Up

Remove generated training artifacts if no longer needed:

```bash
rm -rf outputs/
```

Deactivate the environment:

```bash
deactivate
```

---

# Recommended Repository Structure

For the standalone lab:

```text
deepspeed-zero3-lab/
├── train_ds_zero3.py
├── ds_config_zero3.json
└── outputs/
```

For your book companion repository:

```text
chapter-27/
├── lab-07-train-a-transformer-with-deepspeed-zero-3.md
├── lab-07-train-ds-zero3.py
└── lab-07-ds-config-zero3.json
```

I recommend not committing:

```text
outputs/
```

because it contains generated checkpoints that can become large and are reproducible from the lab.

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Verified multiple NVIDIA GPUs.
* [ ] Verified CUDA and PyTorch.
* [ ] Installed DeepSpeed.
* [ ] Created the BERT training script.
* [ ] Loaded the IMDb dataset.
* [ ] Tokenized the reviews.
* [ ] Created the ZeRO-3 configuration.
* [ ] Enabled Stage 3 optimization.
* [ ] Enabled communication overlap.
* [ ] Enabled contiguous gradients.
* [ ] Enabled CPU optimizer offloading.
* [ ] Configured BF16 or an appropriate alternative.
* [ ] Integrated DeepSpeed with `Trainer`.
* [ ] Calculated effective global batch size.
* [ ] Launched training with multiple GPUs.
* [ ] Verified DeepSpeed workers.
* [ ] Monitored GPU memory.
* [ ] Monitored GPU utilization.
* [ ] Compared ZeRO-3 with a non-DeepSpeed baseline.
* [ ] Recorded throughput.
* [ ] Recorded maximum stable batch size.
* [ ] Created a distributed checkpoint.
* [ ] Resumed from a checkpoint.
* [ ] Distinguished checkpoint state from deployment weights.
* [ ] Tested CPU-offload behavior.
* [ ] Reviewed ZeRO-2 versus ZeRO-3.
* [ ] Reviewed larger-model scaling.
* [ ] Reviewed multi-node considerations.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Explain why conventional replicated training consumes substantial GPU memory.
* Describe the differences between ZeRO-1, ZeRO-2, and ZeRO-3.
* Configure DeepSpeed ZeRO-3.
* Integrate ZeRO-3 with Hugging Face Trainer.
* Explain parameter, gradient, and optimizer-state sharding.
* Calculate global batch size.
* Explain CPU optimizer offloading.
* Measure GPU-memory reduction.
* Compare memory capacity with training throughput.
* Save and resume distributed ZeRO checkpoints.
* Distinguish training checkpoints from deployment artifacts.
* Explain why communication overhead grows with more aggressive sharding.
* Identify when ZeRO-3 becomes useful for large Transformer models.

---

# Key Takeaway

**DeepSpeed ZeRO-3 makes distributed training more memory-scalable by partitioning parameters, gradients, and optimizer states across workers instead of replicating the complete training state on every GPU.**

The core architecture is:

```text
Transformer Model
       ↓
DeepSpeed ZeRO-3
       ↓
┌─────────────┬─────────────┬─────────────┐
│ GPU 0       │ GPU 1       │ GPU N       │
│ State Shard │ State Shard │ State Shard │
└─────────────┴─────────────┴─────────────┘
       ↓
Distributed Communication
       ↓
Training
```

Compared with conventional DDP:

```text
DDP
 ↓
Replicate Model State
 ↓
High Per-GPU Memory
```

ZeRO-3 provides:

```text
ZeRO-3
 ↓
Shard Model State
 ↓
Lower Per-GPU Memory
```

But that memory efficiency introduces tradeoffs:

```text
More Sharding
     ↓
Less Memory
     +
More Communication
     +
Potential Offload Overhead
```

The broader progression is:

```text
Single GPU
    ↓
DDP
    ↓
ZeRO-1
    ↓
ZeRO-2
    ↓
ZeRO-3
    ↓
CPU / NVMe Offload
    ↓
Large-Scale Distributed Training
```

The correct production configuration is therefore not simply the one with the lowest GPU-memory usage. It is the configuration that provides enough memory capacity to fit the workload while maintaining acceptable **throughput, communication efficiency, checkpoint reliability, and operational complexity**.
