# Hands-On Lab: Train a PyTorch Model Across Multiple GPUs with DDP

In this lab, you will use **PyTorch Distributed Data Parallel (DDP)** to train a ResNet-18 model across multiple GPUs.

DDP creates one training process per GPU, maintains a model replica on each accelerator, partitions the training dataset among those processes, and synchronizes gradients during backpropagation.

The lab demonstrates the core workflow used for efficient multi-GPU training with PyTorch.

---

## Distributed Training Code

The training loop calculates the average loss for each epoch and reports the result from each GPU process:

```python
print(
    f"[GPU {local_rank}] "
    f"Epoch {epoch + 1}, "
    f"Loss: {running_loss / len(trainloader):.4f}"
)
```

After training completes, only **rank 0** saves the final model checkpoint:

```python
if rank == 0:
    torch.save(
        model.module.state_dict(),
        "resnet_ddp.pth"
    )

cleanup()
```

The script finishes with:

```python
if __name__ == "__main__":
    train()
```

The `DistributedSampler` ensures that each training process receives a different portion of the dataset.

The relationship is:

```text
                  Training Dataset
                         │
                         ▼
                DistributedSampler
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Process / GPU 0       Process / GPU 1
              │                     │
              ▼                     ▼
        Model Replica 0       Model Replica 1
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                Gradient Synchronization
                         │
                         ▼
                   Updated Models
```

DDP maintains a model replica on each GPU and synchronizes gradients during backpropagation.

Only rank `0` saves the final checkpoint, preventing multiple processes from attempting to write the same model file.

---

## Step 5: Launch Distributed Training

Launch the distributed training job using PyTorch's `torchrun` utility:

```bash
torchrun --standalone --nproc-per-node=2 train_ddp.py
```

The option:

```text
--nproc-per-node=2
```

creates **two training processes**, with one process assigned to each GPU.

The relationship is:

```text
torchrun
   │
   ├── Process 0
   │      ↓
   │    GPU 0
   │
   └── Process 1
          ↓
        GPU 1
```

If your machine contains more GPUs, adjust the number accordingly.

For example, with four GPUs:

```bash
torchrun --standalone --nproc-per-node=4 train_ddp.py
```

During training, each process works on its assigned portion of the CIFAR-10 dataset while DDP synchronizes gradients across the GPUs.

You should see output similar to:

```text
[GPU 0] Epoch 1, Loss: 1.2345
[GPU 1] Epoch 1, Loss: 1.2287
[GPU 0] Epoch 2, Loss: 0.8721
[GPU 1] Epoch 2, Loss: 0.8698
```

> **Important**
>
> `torchrun` is the recommended launcher for this script. It supplies the process information required by `dist.init_process_group()` and sets the `LOCAL_RANK` environment variable used to map each process to its GPU.

---

## Step 6: Monitor GPU Utilization

While distributed training is running, open another terminal and monitor GPU activity:

```bash
watch -n 1 nvidia-smi
```

Observe:

* GPU utilization
* GPU memory consumption
* Power usage
* Active Python processes
* Activity across multiple GPUs

You should see multiple GPUs being used simultaneously.

A typical process relationship is:

```text
Python Process 0 ──────► GPU 0
Python Process 1 ──────► GPU 1
```

Each process owns its corresponding model replica and processes a different portion of the training data.

> **Note**
>
> GPU utilization may fluctuate because CIFAR-10 and ResNet-18 are relatively small workloads. Larger models and datasets generally make multi-GPU utilization more apparent.

On systems where `watch` is unavailable, run:

```bash
nvidia-smi
```

repeatedly during training.

---

## Step 7: Load the Trained Model

The training script saves the final checkpoint as:

```text
resnet_ddp.pth
```

Only rank `0` writes this file.

You can load the checkpoint into a standard non-DDP ResNet-18 model for subsequent evaluation or inference.

```python
import torch
import torchvision.models as models

model = models.resnet18(
    weights=None,
    num_classes=10
)

state_dict = torch.load(
    "resnet_ddp.pth",
    map_location="cpu"
)

model.load_state_dict(state_dict)

model.eval()
```

The checkpoint contains the model's learned parameters rather than the DDP wrapper.

The workflow is:

```text
DDP Training
     ↓
model.module.state_dict()
     ↓
resnet_ddp.pth
     ↓
Standard ResNet-18
     ↓
Evaluation / Inference
```

Saving:

```python
model.module.state_dict()
```

instead of the complete DDP wrapper makes the checkpoint easier to load into a normal PyTorch model that is not running under Distributed Data Parallel.

---

## Step 8: Stop the Workload and Release Resources

Normally, the distributed training processes terminate automatically when training completes.

If you need to stop the workload manually on Linux, you can use:

```bash
pkill -f train_ddp.py
```

After the training processes terminate, GPU memory allocated by those processes is released automatically.

Verify GPU status:

```bash
nvidia-smi
```

The training processes should no longer appear.

---

## Cloud GPU Cleanup

If you are using cloud GPU infrastructure, stop or terminate the GPU resources when they are no longer required.

Check for resources such as:

* GPU virtual machines
* Attached disks
* Persistent volumes
* Snapshots
* Static IP addresses
* Other billable infrastructure

> **⚠️ Cost Warning**
>
> Multi-GPU cloud instances can be expensive. Stop or terminate them immediately after completing the lab to avoid unnecessary charges.

---

## How PyTorch DDP Works

PyTorch Distributed Data Parallel uses a **one-process-per-GPU architecture**.

For two GPUs:

```text
                     Training Job
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
             Rank 0              Rank 1
                │                   │
                ▼                   ▼
              GPU 0               GPU 1
                │                   │
                ▼                   ▼
          Model Replica       Model Replica
                │                   │
                ▼                   ▼
          Local Forward       Local Forward
                │                   │
                ▼                   ▼
          Local Backward      Local Backward
                │                   │
                └─────────┬─────────┘
                          │
                          ▼
                  Gradient Synchronization
                          │
                          ▼
                  Updated Model Replicas
```

Each process:

1. Runs on a separate GPU.
2. Maintains its own model replica.
3. Processes a different portion of the dataset.
4. Computes gradients locally.
5. Participates in gradient synchronization.
6. Updates its model replica using the synchronized gradients.

This design allows the workload to process multiple training batches concurrently across multiple GPUs.

---

## Important DDP Components

| Component                 | Purpose                                                |
| ------------------------- | ------------------------------------------------------ |
| `torchrun`                | Launches distributed training processes                |
| `LOCAL_RANK`              | Identifies the GPU assigned to a process               |
| `init_process_group()`    | Initializes distributed communication                  |
| `DistributedDataParallel` | Wraps the model for distributed training               |
| `DistributedSampler`      | Partitions training data among processes               |
| `rank`                    | Identifies a process within the distributed job        |
| `world_size`              | Total number of participating processes                |
| `model.module`            | Provides access to the underlying model wrapped by DDP |

---

## Why Use DistributedSampler?

Without a distributed sampler, every GPU could process the same training examples.

For example:

```text
Without DistributedSampler

GPU 0 → Images 1–64
GPU 1 → Images 1–64
```

This duplicates work.

With `DistributedSampler`:

```text
With DistributedSampler

GPU 0 → Dataset Partition A
GPU 1 → Dataset Partition B
```

Each process receives a different portion of the dataset, enabling the GPUs to work on different training examples concurrently.

---

## Why Save Only from Rank 0?

Every DDP process contains a synchronized model replica.

If every process attempted to save:

```text
resnet_ddp.pth
```

simultaneously, multiple processes could try to write the same file.

Instead:

```python
if rank == 0:
    torch.save(
        model.module.state_dict(),
        "resnet_ddp.pth"
    )
```

ensures that only one process writes the checkpoint.

The pattern is:

```text
Rank 0 ─────► Save Checkpoint

Rank 1 ─────► Do Not Save

Rank 2 ─────► Do Not Save

Rank 3 ─────► Do Not Save
```

---

## Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Configured PyTorch Distributed Data Parallel
* [ ] Initialized the distributed process group
* [ ] Assigned one process to each GPU
* [ ] Used `DistributedSampler` to partition training data
* [ ] Wrapped ResNet-18 with DDP
* [ ] Launched training with `torchrun`
* [ ] Observed activity on multiple GPUs
* [ ] Monitored GPU utilization using `nvidia-smi`
* [ ] Completed distributed training
* [ ] Saved the model checkpoint from rank 0
* [ ] Loaded the saved checkpoint into a standard PyTorch model
* [ ] Stopped or terminated any cloud GPU resources

---

## Learning Outcomes

After completing this lab, you should be able to:

* Configure PyTorch **Distributed Data Parallel** for multi-GPU training.
* Run one distributed training process per GPU.
* Partition training data using `DistributedSampler`.
* Train ResNet-18 across multiple GPUs.
* Understand how DDP synchronizes gradients between model replicas.
* Monitor GPU utilization during distributed training.
* Save a DDP-trained model checkpoint correctly.
* Load a distributed-training checkpoint for subsequent evaluation or inference.
* Explain the relationship among ranks, processes, GPUs, model replicas, and gradient synchronization.

---

## Key Takeaway

**PyTorch Distributed Data Parallel scales neural-network training across multiple GPUs by running one process and one model replica per GPU, partitioning the training data among those processes, and synchronizing gradients during backpropagation.**

This architecture allows multiple GPUs to process different portions of the workload concurrently while keeping the model parameters synchronized across the distributed training job.
