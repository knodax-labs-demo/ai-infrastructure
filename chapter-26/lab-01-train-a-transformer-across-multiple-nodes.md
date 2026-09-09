# Hands-On Lab: Train a Transformer Across Multiple Nodes

In this lab, you will train a Transformer model across multiple GPU nodes using **PyTorch DistributedDataParallel (DDP)**.

You will fine-tune **BERT** for sentiment classification using the IMDb dataset while running one distributed training process per GPU. PyTorch will synchronize gradients across all participating workers using the **NCCL** communication backend, while `DistributedSampler` ensures that each process receives a different portion of the dataset.

You will also configure multi-node rendezvous settings, launch the job with `torchrun`, save a distributed training checkpoint, resume from that checkpoint, perform distributed evaluation, and analyze the performance tradeoffs introduced by scaling across GPUs and nodes.

The complete architecture is:

```text id="mb4p9g"
Node 0
├── GPU 0 → DDP Process 0
├── GPU 1 → DDP Process 1
├── GPU 2 → DDP Process 2
└── GPU 3 → DDP Process 3
           │
           │ NCCL Gradient Synchronization
           │
Node 1     │
├── GPU 0 → DDP Process 4
├── GPU 1 → DDP Process 5
├── GPU 2 → DDP Process 6
└── GPU 3 → DDP Process 7
           ↓
     Synchronized Model
           ↓
       Checkpoint
           ↓
 Distributed Evaluation
```

---

## Lab Objectives

By completing this lab, you will learn how to:

* Configure multiple GPU nodes for distributed training.
* Understand node rank, local rank, global rank, and world size.
* Launch multi-node workloads with `torchrun`.
* Use NCCL for GPU-to-GPU collective communication.
* Train BERT with PyTorch DDP.
* Partition datasets with `DistributedSampler`.
* Synchronize gradients across workers.
* Compute distributed evaluation metrics.
* Save checkpoints from a designated rank.
* Resume distributed training after interruption.
* Calculate effective global batch size.
* Measure scaling efficiency.
* Identify communication bottlenecks.
* Understand when FSDP or ZeRO may become necessary.

---

## Estimated Time

**Approximately 120–180 minutes**

---

## Tools

This lab uses:

* Python
* PyTorch 2.x
* PyTorch Distributed
* DistributedDataParallel
* NCCL
* `torchrun`
* Hugging Face Transformers
* Hugging Face Datasets
* NVIDIA GPUs
* Two or more networked GPU nodes

---

# Step 1: Verify the Prerequisites

You need at least two GPU-enabled nodes that can communicate with one another over the network.

Each node should have:

* A compatible NVIDIA GPU.
* NVIDIA drivers.
* CUDA-compatible PyTorch.
* NCCL support.
* The same Python environment.
* The same training script.
* Network connectivity to the rendezvous node.

Install on every node:

```bash id="y70mt5"
pip install \
  torch \
  torchvision \
  torchaudio \
  transformers \
  datasets
```

Verify GPUs:

```bash id="k1mfxo"
python -c \
'import torch; print(torch.cuda.device_count())'
```

Each node should report the expected number of visible GPUs.

---

# Step 2: Understand the Distributed Terminology

Before launching the workload, distinguish these concepts:

| Term        | Meaning                                 |
| ----------- | --------------------------------------- |
| Node        | Physical or virtual machine             |
| GPU         | Accelerator device on a node            |
| Process     | One training worker                     |
| Local Rank  | Process index within one node           |
| Global Rank | Unique process index across the job     |
| World Size  | Total number of participating processes |
| Node Rank   | Unique index assigned to each node      |

For GPU-based DDP, the standard architecture is:

```text id="de9fh6"
1 GPU
  =
1 Process
```

For two nodes with four GPUs each:

```text id="9dmr1h"
Nodes              = 2
GPUs per node      = 4
Processes per node = 4
WORLD_SIZE         = 8
Global ranks       = 0–7
```

---

# Step 3: Understand Rank Assignment

For example:

```text id="tug2ir"
Node 0
├── GPU 0 → LOCAL_RANK=0 → RANK=0
├── GPU 1 → LOCAL_RANK=1 → RANK=1
├── GPU 2 → LOCAL_RANK=2 → RANK=2
└── GPU 3 → LOCAL_RANK=3 → RANK=3

Node 1
├── GPU 0 → LOCAL_RANK=0 → RANK=4
├── GPU 1 → LOCAL_RANK=1 → RANK=5
├── GPU 2 → LOCAL_RANK=2 → RANK=6
└── GPU 3 → LOCAL_RANK=3 → RANK=7
```

`torchrun` calculates global rank and world size for the processes.

---

# Step 4: Configure Multi-Node Networking

Choose one node as rank 0.

Set on every node:

```bash id="0s8jqp"
export MASTER_ADDR="10.0.0.10"
export MASTER_PORT=29500
export NUM_NODES=2
```

Replace:

```text id="rdpmf0"
10.0.0.10
```

with the private IP address or resolvable hostname of node 0.

On the first node:

```bash id="m63738"
export NODE_RANK=0
```

On the second:

```bash id="fyjlbx"
export NODE_RANK=1
```

---

## Networking Architecture

```text id="88z03m"
Node 1
    │
    │ Rendezvous
    │
    ▼
Node 0
MASTER_ADDR
MASTER_PORT
```

Every participating node must be able to reach:

```text id="edz1wn"
MASTER_ADDR:MASTER_PORT
```

Production environments should restrict this traffic with appropriate network and firewall policies.

---

# Step 5: Understand WORLD_SIZE

A common mistake is to treat world size as the number of machines.

Normally:

```text id="0l0x8u"
WORLD_SIZE
    =
Total Distributed Processes
```

For:

```text id="pmlpem"
2 nodes
×
4 processes per node
```

the result is:

```text id="y07fnv"
WORLD_SIZE = 8
```

not:

```text id="go70vz"
WORLD_SIZE = 2
```

---

# Step 6: Create the DDP Training Script

Create:

```text id="e7b085"
lab-07-train-ddp.py
```

Add:

```python id="mugqfx"
import os

import torch
import torch.distributed as dist

from torch.nn.parallel import (
    DistributedDataParallel as DDP
)

from torch.utils.data import (
    DataLoader,
    DistributedSampler,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from datasets import load_dataset


MODEL_NAME = "bert-base-uncased"

CHECKPOINT_PATH = (
    "bert_ddp_checkpoint.pt"
)

NUM_EPOCHS = 2


def setup():
    dist.init_process_group(
        backend="nccl"
    )


def cleanup():
    dist.destroy_process_group()


def main():
    setup()

    rank = dist.get_rank()

    world_size = (
        dist.get_world_size()
    )

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    torch.cuda.set_device(
        local_rank
    )

    device = torch.device(
        "cuda",
        local_rank
    )

    if rank == 0:
        print(
            f"Distributed job started "
            f"with WORLD_SIZE={world_size}"
        )

    # Load dataset
    dataset = load_dataset(
        "imdb"
    )

    # Load tokenizer
    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    tokenized = (
        dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"],
        )
    )

    tokenized.set_format(
        "torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label",
        ],
    )

    train_dataset = (
        tokenized["train"]
    )

    test_dataset = (
        tokenized["test"]
    )

    # Distributed samplers
    train_sampler = (
        DistributedSampler(
            train_dataset,
            shuffle=True,
        )
    )

    test_sampler = (
        DistributedSampler(
            test_dataset,
            shuffle=False,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=train_sampler,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        sampler=test_sampler,
        pin_memory=True,
    )

    # Create BERT model
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
    )

    model.to(
        device
    )

    model = DDP(
        model,
        device_ids=[
            local_rank
        ],
        output_device=(
            local_rank
        ),
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,
        )
    )

    start_epoch = 0

    # Resume if checkpoint exists
    if os.path.exists(
        CHECKPOINT_PATH
    ):
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

        model.module.load_state_dict(
            checkpoint[
                "model_state"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state"
            ]
        )

        start_epoch = (
            checkpoint["epoch"]
            + 1
        )

        if rank == 0:
            print(
                f"Resuming from "
                f"epoch {start_epoch}"
            )

    # Training
    for epoch in range(
        start_epoch,
        NUM_EPOCHS
    ):
        model.train()

        train_sampler.set_epoch(
            epoch
        )

        total_loss = 0.0

        for batch in train_loader:

            input_ids = (
                batch["input_ids"]
                .to(
                    device,
                    non_blocking=True
                )
            )

            attention_mask = (
                batch[
                    "attention_mask"
                ]
                .to(
                    device,
                    non_blocking=True
                )
            )

            labels = (
                batch["label"]
                .to(
                    device,
                    non_blocking=True
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                labels=labels,
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        # Ensure all workers finish epoch
        dist.barrier()

        if rank == 0:

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state":
                        model
                        .module
                        .state_dict(),

                    "optimizer_state":
                        optimizer
                        .state_dict(),
                },
                CHECKPOINT_PATH,
            )

            average_loss = (
                total_loss
                / max(
                    len(
                        train_loader
                    ),
                    1
                )
            )

            print(
                f"Epoch {epoch + 1} "
                f"completed. "
                f"Local rank-0 loss: "
                f"{average_loss:.4f}. "
                "Checkpoint saved."
            )

        dist.barrier()

    # Distributed evaluation
    model.eval()

    correct = torch.tensor(
        0,
        device=device,
        dtype=torch.long,
    )

    total = torch.tensor(
        0,
        device=device,
        dtype=torch.long,
    )

    with torch.no_grad():

        for batch in test_loader:

            input_ids = (
                batch["input_ids"]
                .to(
                    device,
                    non_blocking=True
                )
            )

            attention_mask = (
                batch[
                    "attention_mask"
                ]
                .to(
                    device,
                    non_blocking=True
                )
            )

            labels = (
                batch["label"]
                .to(
                    device,
                    non_blocking=True
                )
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
            )

            predictions = (
                outputs
                .logits
                .argmax(
                    dim=-1
                )
            )

            correct += (
                predictions
                == labels
            ).sum()

            total += (
                labels.numel()
            )

    # Aggregate metrics across workers
    dist.all_reduce(
        correct,
        op=dist.ReduceOp.SUM,
    )

    dist.all_reduce(
        total,
        op=dist.ReduceOp.SUM,
    )

    if rank == 0:

        accuracy = (
            correct.float()
            / total.float()
        ).item()

        print(
            "Distributed test "
            f"accuracy: "
            f"{accuracy:.4f}"
        )

    cleanup()


if __name__ == "__main__":
    main()
```

---

# Step 7: Understand DDP Initialization

The critical call is:

```python id="e0c3y2"
dist.init_process_group(
    backend="nccl"
)
```

This establishes the distributed process group.

The architecture becomes:

```text id="2uvb5u"
Process 0
Process 1
Process 2
...
Process N
     ↓
Distributed Process Group
     ↓
NCCL Collectives
```

---

# Step 8: Map Each Process to a GPU

Each process receives:

```text id="qho8ku"
LOCAL_RANK
```

from `torchrun`.

The script uses:

```python id="zbmy4h"
local_rank = int(
    os.environ["LOCAL_RANK"]
)

torch.cuda.set_device(
    local_rank
)
```

This ensures:

```text id="5kurr4"
Process 0 → GPU 0
Process 1 → GPU 1
Process 2 → GPU 2
...
```

within each node.

---

# Step 9: Wrap the Model with DDP

The BERT model is wrapped with:

```python id="njg6iw"
model = DDP(
    model,
    device_ids=[
        local_rank
    ],
    output_device=(
        local_rank
    ),
)
```

Each process now owns:

```text id="o5vrqj"
One BERT Replica
+
One GPU
+
One Data Partition
```

---

# Step 10: Understand Gradient Synchronization

During training:

```text id="0crjyw"
Forward Pass
      ↓
Loss
      ↓
Backward Pass
      ↓
Local Gradients
      ↓
NCCL All-Reduce
      ↓
Synchronized Gradients
      ↓
Optimizer Step
```

DDP performs gradient synchronization automatically during backward propagation.

Each model replica therefore applies logically synchronized updates.

---

# Step 11: Understand DistributedSampler

Without dataset partitioning:

```text id="h1yyt0"
Process 0 → Entire Dataset
Process 1 → Entire Dataset
Process 2 → Entire Dataset
```

That would duplicate training work.

Instead:

```text id="zm42ic"
DistributedSampler
       ↓
Dataset
├── Partition 0 → Process 0
├── Partition 1 → Process 1
├── Partition 2 → Process 2
└── Partition N → Process N
```

Each worker receives a different subset.

---

# Step 12: Use `set_epoch()`

The training loop calls:

```python id="p18mgf"
train_sampler.set_epoch(
    epoch
)
```

This allows distributed shuffling to vary correctly between epochs while remaining coordinated across ranks.

Without this call, the shuffle ordering may repeat across epochs.

---

# Step 13: Understand the Global Batch Size

With:

```text id="l1pxj9"
Per-Process Batch Size = 8
```

and:

```text id="a2dk2g"
2 nodes
×
4 GPUs per node
=
8 processes
```

the effective batch size is:

```text id="z29ww9"
8 × 8 = 64
```

More generally:

```text id="64mi2y"
Global Batch Size
=
Per-Process Batch Size
×
Number of DDP Processes
×
Gradient Accumulation Steps
```

---

# Step 14: Understand Why Scaling Changes Optimization

Adding GPUs changes more than throughput.

For example:

```text id="mxepud"
1 GPU
Batch = 8
```

versus:

```text id="judlkm"
8 GPUs
Batch per GPU = 8
Global Batch = 64
```

This can affect:

* Gradient statistics
* Convergence
* Learning-rate behavior
* Number of optimizer updates
* Generalization

Distributed training should therefore evaluate model convergence as well as raw throughput.

---

# Step 15: Launch the Multi-Node Job

Assume:

```text id="gqq2cs"
2 Nodes
4 GPUs per Node
```

Run on **both nodes**:

```bash id="mm0d23"
torchrun \
  --nnodes=$NUM_NODES \
  --nproc-per-node=4 \
  --node-rank=$NODE_RANK \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  lab-07-train-ddp.py
```

Start the commands close enough together for all workers to join the same rendezvous.

---

# Step 16: Verify the Distributed Topology

Expected configuration:

```text id="8fucwc"
Nodes:               2
GPUs per node:       4
Processes per node:  4
WORLD_SIZE:          8
Global ranks:        0–7
```

The process topology is:

```text id="xyycuf"
Node 0
Ranks 0–3

Node 1
Ranks 4–7
```

---

# Step 17: Understand Rank 0

Global rank 0 typically handles tasks that should happen only once, such as:

* Main console logging
* Checkpoint creation
* Final metric reporting
* Certain metadata operations

This prevents every process from trying to write the same artifact.

---

# Step 18: Save the Distributed Checkpoint

After each epoch, rank 0 creates:

```text id="5knm71"
bert_ddp_checkpoint.pt
```

The checkpoint contains:

```text id="hsfqeh"
Epoch
+
Model State
+
Optimizer State
```

The code saves:

```python id="lztzz5"
model.module.state_dict()
```

rather than:

```python id="rud75w"
model.state_dict()
```

because the underlying BERT model is wrapped by DDP.

---

# Step 19: Understand Why Optimizer State Matters

AdamW maintains state such as momentum-related statistics.

Therefore, recovery requires more than:

```text id="39hip5"
Model Weights
```

A better training checkpoint contains:

```text id="uo9ywj"
Model State
+
Optimizer State
+
Training Position
```

Production checkpoints may also include:

* Learning-rate scheduler
* RNG states
* Training configuration
* Data-loader state
* Gradient-scaler state
* Experiment metadata

---

# Step 20: Validate the Checkpoint

Create:

```text id="fv2qkl"
lab-07-validate-checkpoint.py
```

Add:

```python id="qfmp37"
import torch

from transformers import (
    AutoModelForSequenceClassification,
)


MODEL_NAME = (
    "bert-base-uncased"
)

CHECKPOINT_PATH = (
    "bert_ddp_checkpoint.pt"
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
)


model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
)


model.load_state_dict(
    checkpoint[
        "model_state"
    ]
)

model.eval()


print(
    "Checkpoint loaded successfully."
)
```

Run:

```bash id="dqy2q2"
python \
  lab-07-validate-checkpoint.py
```

---

# Step 21: Test Checkpoint Recovery

Increase:

```python id="xut394"
NUM_EPOCHS = 4
```

Run training until at least one checkpoint is created.

Then interrupt the job.

Restart using the same `torchrun` command.

The script should:

```text id="7ne8li"
Find Checkpoint
      ↓
Load Model State
      ↓
Load Optimizer State
      ↓
Read Last Epoch
      ↓
Resume Next Epoch
```

---

# Step 22: Understand the Shared Checkpoint Requirement

In this simple design, every worker checks:

```text id="4kn3k1"
bert_ddp_checkpoint.pt
```

Therefore, all nodes must see the same checkpoint.

A common architecture is:

```text id="i7jaov"
Node 0
Node 1
Node 2
   ↓
Shared Filesystem
   ↓
Checkpoint
```

Alternatives include object storage or distributed checkpointing systems.

---

# Step 23: Understand the Role of Barriers

The code uses:

```python id="49i9w1"
dist.barrier()
```

before and after checkpoint creation.

A barrier means:

```text id="5tz8wr"
All Workers
    ↓
Reach Synchronization Point
    ↓
Wait
    ↓
Continue Together
```

This reduces the chance that some workers move ahead while checkpoint-related work is still occurring.

---

# Step 24: Perform Distributed Evaluation

Each worker evaluates a separate dataset partition.

Locally:

```text id="29p8na"
Worker 0
→ Local Correct
→ Local Total

Worker 1
→ Local Correct
→ Local Total

...
```

Then:

```python id="8z7z9g"
dist.all_reduce(
    correct,
    op=dist.ReduceOp.SUM
)
```

and:

```python id="c5rsna"
dist.all_reduce(
    total,
    op=dist.ReduceOp.SUM
)
```

aggregate statistics across the process group.

---

# Step 25: Understand Metric Aggregation

The distributed metric path is:

```text id="q3pzft"
Rank 0 Correct ─┐
Rank 1 Correct ─┤
Rank 2 Correct ─┼─► All-Reduce SUM
Rank N Correct ─┘
                    ↓
               Global Correct
```

The same happens for:

```text id="akm6gt"
Total Examples
```

Then:

```text id="gc7v44"
Global Accuracy
=
Global Correct / Global Total
```

---

# Step 26: Monitor GPU Utilization

On each node:

```bash id="7j7a60"
watch -n 1 nvidia-smi
```

Observe:

* GPU utilization
* Memory utilization
* Temperature
* Power consumption
* Process placement

Each GPU should normally have one training process.

---

# Step 27: Monitor NCCL Behavior

Distributed performance depends heavily on communication.

Useful areas to observe include:

```text id="5bc33z"
Gradient Synchronization Time
Network Throughput
GPU Utilization
Step Duration
```

The basic training step becomes:

```text id="1pp37v"
Compute Forward
      ↓
Compute Backward
      ↓
Communicate Gradients
      ↓
Optimizer Step
```

As node count grows, communication may consume a larger fraction of each training step.

---

# Step 28: Understand Compute vs. Communication

Distributed scaling depends on the ratio:

```text id="6gp5zp"
Useful Compute
      vs.
Communication Overhead
```

If each training step performs substantial GPU computation:

```text id="eng499"
Compute >> Communication
```

scaling can be efficient.

If synchronization dominates:

```text id="2ah46x"
Communication >> Compute
```

adding workers may provide limited benefit.

---

# Step 29: Measure Training Throughput

Measure:

```text id="s3zau3"
Samples / Second
```

or, for language models:

```text id="zqp9ol"
Tokens / Second
```

Compare:

| Configuration    | Throughput | Step Time | GPU Utilization |
| ---------------- | ---------: | --------: | --------------: |
| 1 GPU            |     Record |    Record |          Record |
| 4 GPUs / 1 node  |     Record |    Record |          Record |
| 8 GPUs / 2 nodes |     Record |    Record |          Record |

---

# Step 30: Calculate Scaling Efficiency

Suppose:

```text id="0a7yh8"
1 GPU throughput = T1
N GPU throughput = TN
```

Ideal throughput would be:

```text id="muqzwf"
N × T1
```

A simple scaling-efficiency metric is:

```text id="cb7bxu"
Scaling Efficiency
=
TN
──────
N × T1
× 100%
```

For example:

```text id="w89gt7"
1 GPU = 100 samples/s
8 GPUs = 680 samples/s

Efficiency =
680 / 800 × 100
= 85%
```

---

# Step 31: Understand Why Scaling Is Not Linear

Factors include:

* Gradient synchronization
* Network bandwidth
* Network latency
* PCIe topology
* NVLink availability
* Dataset pipeline speed
* Batch size
* Model size
* CPU bottlenecks
* Storage throughput
* Load imbalance

Therefore:

```text id="zt2mqp"
2× GPUs
    ≠
Automatically 2× Throughput
```

---

# Step 32: Understand NCCL

NCCL provides optimized collective communication for NVIDIA GPUs.

Important collective operations include:

```text id="myxnme"
All-Reduce
Broadcast
Reduce
All-Gather
Reduce-Scatter
```

DDP relies heavily on:

```text id="q4blst"
All-Reduce
```

for gradient synchronization.

---

# Step 33: Understand Intra-Node vs. Inter-Node Communication

Within one machine:

```text id="m5wk63"
GPU
 ↓
PCIe / NVLink / NVSwitch
 ↓
GPU
```

Across machines:

```text id="94xkrh"
GPU
 ↓
PCIe / NIC
 ↓
Network
 ↓
NIC / PCIe
 ↓
GPU
```

Inter-node communication is generally more expensive, making network architecture critical for large training clusters.

---

# Step 34: High-Performance Networking

Larger distributed workloads may use:

* InfiniBand
* RoCE
* High-bandwidth Ethernet
* GPUDirect RDMA

A simplified path can become:

```text id="d7fb9f"
GPU
 ↓
GPUDirect RDMA
 ↓
High-Speed Network
 ↓
Remote GPU
```

Reducing unnecessary CPU copies can improve collective-communication performance.

---

# Step 35: Add Mixed Precision

A useful extension is:

```text id="tgg80m"
FP16
```

or:

```text id="wk97o4"
BF16
```

where supported.

Mixed precision can:

* Reduce memory usage
* Increase throughput
* Improve Tensor Core utilization

A mature distributed workload should measure both model quality and throughput after enabling reduced precision.

---

# Step 36: Understand DDP Memory Scaling

DDP gives each process a complete model replica.

Conceptually:

```text id="x4qsdt"
GPU 0 → Full Model
GPU 1 → Full Model
GPU 2 → Full Model
GPU 3 → Full Model
```

Therefore, DDP primarily distributes:

```text id="jtuvja"
Data / Compute
```

but does not fundamentally eliminate per-GPU model replication.

---

# Step 37: Know When DDP Stops Being Enough

For a larger model:

```text id="o55bwf"
Model
+
Gradients
+
Optimizer State
```

may no longer fit on one GPU.

At that point, consider:

```text id="mdkd4l"
FSDP
DeepSpeed ZeRO
Tensor Parallelism
Pipeline Parallelism
```

depending on the workload.

---

# Step 38: Compare DDP and FSDP

## DDP

```text id="o7243d"
Each GPU
   ↓
Full Model Replica
```

## FSDP

```text id="95vagh"
Model State
   ↓
Sharded Across GPUs
```

This is why FSDP becomes especially important for larger Transformer models.

---

# Step 39: Understand Sharded Checkpointing

The lab uses:

```text id="rdun13"
Rank 0
   ↓
Writes Complete Checkpoint
```

For very large models, this becomes inefficient.

A distributed checkpoint architecture is closer to:

```text id="ja3q08"
Rank 0 → Shard 0
Rank 1 → Shard 1
Rank 2 → Shard 2
...
```

This distributes checkpoint memory and I/O pressure.

---

# Step 40: Test Failure Recovery

A useful experiment is:

1. Start distributed training.
2. Wait until an epoch checkpoint is saved.
3. Stop all workers.
4. Restart the complete job.
5. Confirm that training resumes.
6. Verify final accuracy.

This demonstrates basic infrastructure resilience.

---

# Step 41: Production Considerations

A production distributed-training platform should consider:

* Job scheduling
* GPU placement
* High-speed networking
* Shared storage
* Dataset locality
* Distributed checkpointing
* Logging
* Metrics
* Fault recovery
* Secret management
* Cluster security
* Job reproducibility
* Experiment tracking
* Model artifact management

---

# Step 42: Scheduler Integration

A production workload is often launched through:

```text id="s4u3yd"
Slurm
```

or:

```text id="sv9gtc"
Kubernetes
```

rather than manually running `torchrun` on individual servers.

Conceptually:

```text id="hlbi4m"
Training Job
     ↓
Scheduler
     ↓
Allocate GPU Nodes
     ↓
Launch Distributed Workers
     ↓
Monitor Job
     ↓
Recover / Restart
```

---

# Step 43: Understand Elastic Training

Static DDP generally expects the configured workers to remain available.

More advanced systems may use elastic behavior:

```text id="vog680"
Worker Failure
      ↓
Detect Failure
      ↓
Reconfigure Workers
      ↓
Resume Training
```

PyTorch elastic capabilities can be explored as an extension.

---

# Step 44: Additional Experiments

Try:

* 1 GPU.
* 2 GPUs.
* 4 GPUs.
* 2 nodes.
* Additional nodes.

Record:

```text id="okq73t"
Throughput
Latency per Step
GPU Utilization
Network Utilization
Scaling Efficiency
```

This converts the lab from a simple distributed-training demonstration into an infrastructure-scaling experiment.

---

# Step 45: Clean Up

Stop the distributed processes if they are still running.

Remove the checkpoint if no longer required:

```bash id="5qh7gf"
rm -f \
  bert_ddp_checkpoint.pt
```

Deactivate your Python environment:

```bash id="hllw9h"
deactivate
```

---

# Recommended Repository Structure

For a standalone lab:

```text id="0r20ko"
transformer-ddp-lab/
├── train_ddp.py
├── validate_checkpoint.py
└── bert_ddp_checkpoint.pt
```

For your book companion repository:

```text id="jcb9rz"
chapter-26/
├── lab-07-train-a-transformer-across-multiple-nodes.md
├── lab-07-train-ddp.py
└── lab-07-validate-checkpoint.py
```

I recommend not committing:

```text id="dehaz8"
bert_ddp_checkpoint.pt
```

because it is a generated training artifact and can be reproduced by running the lab.

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Prepared two or more GPU nodes.
* [ ] Installed matching PyTorch environments.
* [ ] Verified GPU visibility on every node.
* [ ] Configured `MASTER_ADDR`.
* [ ] Configured `MASTER_PORT`.
* [ ] Assigned unique node ranks.
* [ ] Understood `WORLD_SIZE`.
* [ ] Created the DDP training script.
* [ ] Initialized the NCCL process group.
* [ ] Assigned each process to a local GPU.
* [ ] Loaded the IMDb dataset.
* [ ] Loaded BERT.
* [ ] Created `DistributedSampler`.
* [ ] Wrapped BERT with DDP.
* [ ] Called `train_sampler.set_epoch()`.
* [ ] Launched the job with `torchrun`.
* [ ] Verified one process per GPU.
* [ ] Observed GPU utilization.
* [ ] Completed distributed training.
* [ ] Saved the rank-0 checkpoint.
* [ ] Validated the checkpoint.
* [ ] Tested checkpoint recovery.
* [ ] Performed distributed evaluation.
* [ ] Aggregated metrics with `all_reduce`.
* [ ] Calculated global accuracy.
* [ ] Compared single-GPU and distributed throughput.
* [ ] Reviewed global batch-size effects.
* [ ] Reviewed communication overhead.
* [ ] Reviewed FSDP and ZeRO as scaling extensions.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Explain multi-node distributed training.
* Distinguish node rank, local rank, global rank, and world size.
* Configure PyTorch distributed networking.
* Launch multi-node jobs with `torchrun`.
* Use NCCL for GPU collective communication.
* Train a Transformer with DDP.
* Partition datasets with `DistributedSampler`.
* Explain DDP gradient synchronization.
* Calculate effective global batch size.
* Save and restore distributed checkpoints.
* Aggregate evaluation metrics across processes.
* Measure distributed scaling efficiency.
* Explain why scaling can become communication-bound.
* Identify when FSDP, ZeRO, or other model-parallel techniques may be required.

---

# Key Takeaway

**Distributed Transformer training is not simply the act of adding more GPUs. It requires coordinated process launch, dataset partitioning, gradient synchronization, communication infrastructure, checkpoint recovery, and careful performance measurement.**

The core architecture is:

```text id="r66yxt"
Dataset
   ↓
DistributedSampler
   ↓
Multiple GPU Processes
   ↓
DDP Model Replicas
   ↓
Forward / Backward
   ↓
NCCL Gradient Synchronization
   ↓
Optimizer Updates
   ↓
Checkpoint
```

DDP works especially well when each GPU can hold a complete model replica and the workload has enough computation to justify the communication overhead.

As models grow, the architecture evolves:

```text id="ymhncy"
Single GPU
    ↓
Single-Node DDP
    ↓
Multi-Node DDP
    ↓
FSDP / ZeRO
    ↓
Tensor + Pipeline Parallelism
    ↓
Large-Scale Distributed Training
```

The broader infrastructure principle is:

```text id="cnyejk"
More GPUs
    ↓
More Compute Capacity
    +
More Communication
    +
Larger Global Batch
    +
Greater Operational Complexity
```

Successful scaling therefore requires optimizing **compute, communication, data loading, networking, checkpointing, and training configuration together**, rather than treating distributed execution as an automatic linear speedup.
