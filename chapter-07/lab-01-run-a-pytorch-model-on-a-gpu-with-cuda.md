# Hands-On Lab: Run a PyTorch Model on a GPU with CUDA

In this lab, you will train and run a simple PyTorch neural network on a CUDA-enabled NVIDIA GPU, monitor GPU utilization, and compare CPU and GPU execution.

You will use the MNIST handwritten-digit dataset because it is small, widely available, and can be trained quickly. The lab demonstrates how PyTorch moves models and tensors between CPU and GPU devices and how CUDA acceleration affects neural-network workloads.

## Goal

Train and run a simple PyTorch neural network on a CUDA-enabled GPU, monitor GPU utilization, and compare CPU and GPU execution.

## Estimated Time

**60–90 minutes**

## Tools

This lab uses:

* Python
* PyTorch
* CUDA-enabled NVIDIA GPU
* NVIDIA drivers
* `nvidia-smi`

---

## Prerequisites

Before starting the lab, make sure you have:

* Python installed
* PyTorch installed
* Access to a CUDA-capable NVIDIA GPU
* A compatible NVIDIA driver
* Internet connectivity for downloading the MNIST dataset

If you are using a cloud GPU instance, verify that the instance includes the required GPU drivers and an appropriate PyTorch environment.

You can complete this lab using either:

* A local GPU workstation
* A compatible cloud GPU instance

> **Cloud Cost Note**
>
> If you use a cloud GPU instance, stop or terminate it immediately after completing the lab to avoid unnecessary charges.

---

## Step 1: Verify GPU and CUDA Availability

First, verify that the operating system can detect the NVIDIA GPU.

Run:

```bash
nvidia-smi
```

The output should display information such as:

* GPU model
* Driver version
* GPU memory
* Memory utilization
* GPU utilization
* Active processes

Next, verify that PyTorch can access CUDA.

You can run the following in Python:

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

If the environment is configured correctly, you should see:

```text
CUDA available: True
```

PyTorch should also display the detected NVIDIA GPU.

> **Important**
>
> If `torch.cuda.is_available()` returns `False`, verify your NVIDIA driver, PyTorch installation, and CUDA compatibility before continuing.

---

## Step 2: Load the MNIST Dataset

This lab uses the MNIST dataset to demonstrate GPU-based training without requiring a large dataset or lengthy training process.

Create the dataset and DataLoader:

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)
```

The DataLoader organizes the training dataset into batches of:

```text
64 images
```

These batches will later be transferred to the GPU during training.

The data flow is:

```text
MNIST Dataset
      ↓
DataLoader
      ↓
Batch of 64 Images
      ↓
CPU Memory
      ↓
GPU Memory
```

---

## Step 3: Define the Neural Network

Create a simple fully connected neural network containing three linear layers.

Each MNIST image contains:

```text
28 × 28 = 784 pixels
```

The image is flattened before being passed through the network.

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

model = Net()
```

The network architecture is:

```text
MNIST Image
  28 × 28
     ↓
Flatten
784 Values
     ↓
Linear Layer
784 → 128
     ↓
ReLU
     ↓
Linear Layer
128 → 64
     ↓
ReLU
     ↓
Linear Layer
64 → 10
     ↓
Digit Classes
0 through 9
```

The final layer produces **10 outputs**, corresponding to the digits `0` through `9`.

---

## Step 4: Move the Model to the GPU

PyTorch uses a device abstraction that allows the same application to run on either a CPU or CUDA-enabled GPU.

Define the device:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print("Using device:", device)
```

When CUDA is available, you should see:

```text
Using device: cuda
```

Calling:

```python
model.to(device)
```

moves the neural-network parameters into GPU memory.

Input tensors must also be transferred to the same device before they are passed to the model.

The relationship is:

```text
Model Parameters ────────┐
                         │
                         ▼
                    GPU Memory
                         ▲
                         │
Input Tensors ───────────┘
```

> **Important**
>
> The model and the input tensors must be on the same device. A GPU model cannot directly process tensors that remain in CPU memory.

---

## Step 5: Configure the Loss Function and Optimizer

Configure cross-entropy loss for classification:

```python
criterion = nn.CrossEntropyLoss()
```

Import the PyTorch optimizer module:

```python
import torch.optim as optim
```

Create an Adam optimizer:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

The two components perform different roles:

| Component         | Purpose                                |
| ----------------- | -------------------------------------- |
| **Loss Function** | Measures prediction error              |
| **Optimizer**     | Updates model parameters               |
| **Learning Rate** | Controls the size of parameter updates |

The training process follows:

```text
Prediction
    ↓
Loss Calculation
    ↓
Backpropagation
    ↓
Gradient Calculation
    ↓
Optimizer Update
```

---

## Step 6: Train the Model on the GPU

Run the training process for two epochs.

```python
for epoch in range(2):
    running_loss = 0.0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)

    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {avg_loss:.4f}"
    )
```

During each training iteration:

1. Images are transferred to the selected device.
2. Labels are transferred to the same device.
3. Existing gradients are cleared.
4. The model performs a forward pass.
5. The loss is calculated.
6. Backpropagation calculates gradients.
7. The optimizer updates the model parameters.

The training workflow is:

```text
Training Batch
      ↓
Transfer to GPU
      ↓
Forward Pass
      ↓
Prediction
      ↓
Loss
      ↓
Backpropagation
      ↓
Gradient Update
      ↓
Next Batch
```

If:

```text
device = cuda
```

the model calculations and tensors are processed using the GPU.

---

## Step 7: Monitor GPU Utilization

While the model is training, open another terminal and monitor the GPU:

```bash
watch -n 1 nvidia-smi
```

This refreshes `nvidia-smi` approximately once per second.

Observe:

* GPU utilization
* GPU memory consumption
* Power usage
* Temperature
* Active processes

The Python training process should appear in the process list while training is active.

On systems where `watch` is unavailable, run:

```bash
nvidia-smi
```

repeatedly during training.

Monitoring helps verify that the workload is actually using the GPU rather than unintentionally executing on the CPU.

---

## Step 8: Test Model Inference

After training, evaluate the model using the MNIST test dataset.

Create the test dataset:

```python
testset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=100,
    shuffle=False
)
```

Place the model in evaluation mode:

```python
model.eval()
```

Run inference:

```python
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

accuracy = 100 * correct / total

print(
    f"Accuracy on test data: "
    f"{accuracy:.2f}%"
)
```

The inference workflow is:

```text
Test Images
     ↓
GPU
     ↓
Neural Network
     ↓
Class Scores
     ↓
Predicted Digit
     ↓
Compare with Label
     ↓
Accuracy
```

Using:

```python
torch.no_grad()
```

disables gradient calculations because gradients are unnecessary during inference.

This reduces:

* Memory usage
* Computational overhead
* Unnecessary gradient tracking

---

## Step 9: Compare CPU and GPU Performance

Repeat the training process using both:

```text
CPU
```

and:

```text
GPU
```

Keep the following configuration identical:

* Neural-network architecture
* Dataset
* Batch size
* Number of epochs
* Optimizer
* Learning rate

Record the total training time for each configuration.

A simple comparison table can be used:

| Device  | Training Time | Final Loss | GPU Utilization |
| ------- | ------------: | ---------: | --------------: |
| **CPU** |     _____ sec |      _____ |             N/A |
| **GPU** |     _____ sec |      _____ |         _____ % |

You can measure execution time using Python's `time` module:

```python
import time

start_time = time.perf_counter()

# Training code here

end_time = time.perf_counter()

print(
    "Training time:",
    round(end_time - start_time, 2),
    "seconds"
)
```

### Analyze the Results

Compare:

* Total training time
* Final training loss
* GPU utilization
* GPU memory consumption

Do not assume the GPU will always be dramatically faster.

MNIST and this neural network are relatively small. For small workloads, overhead such as:

* CUDA initialization
* CPU-to-GPU memory transfers
* Kernel-launch overhead
* Data-loading overhead

can represent a significant percentage of total execution time.

Larger neural networks and more computationally intensive workloads generally provide a clearer demonstration of GPU acceleration.

---

## Optional: Force CPU Execution

To explicitly test CPU execution, set:

```python
device = torch.device("cpu")
```

Then recreate and move the model:

```python
model = Net()
model = model.to(device)
```

Run the same training process and record the execution time.

For the GPU test, use:

```python
device = torch.device("cuda")
```

assuming CUDA is available.

> **Benchmarking Tip**
>
> For a fair comparison, perform the CPU and GPU tests using the same model architecture, dataset, batch size, optimizer, learning rate, and number of epochs.

---

## Step 10: Clean Up

If you performed the lab locally, the downloaded MNIST dataset is stored under:

```text
./data
```

You can retain the dataset for future experiments or remove it if disk space is required.

If you used a cloud GPU instance:

1. Stop or terminate the GPU instance.
2. Check attached storage.
3. Check snapshots.
4. Check static or reserved IP addresses.
5. Verify that other related billable resources are no longer running.

> **⚠️ Cloud Cost Warning**
>
> GPU instances can generate charges quickly. Do not leave a cloud GPU instance running after completing the lab.

---

## Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Verified the NVIDIA GPU using `nvidia-smi`
* [ ] Confirmed that PyTorch reports `CUDA available: True`
* [ ] Loaded the MNIST training dataset
* [ ] Created the neural network
* [ ] Moved the model to the GPU
* [ ] Moved training tensors to the GPU
* [ ] Trained the model for two epochs
* [ ] Monitored GPU utilization with `nvidia-smi`
* [ ] Evaluated the trained model
* [ ] Calculated test accuracy
* [ ] Compared CPU and GPU execution
* [ ] Recorded benchmark results
* [ ] Cleaned up any cloud GPU resources

---

## Learning Outcomes

After completing this lab, you should be able to:

* Verify that an NVIDIA GPU and CUDA are available to PyTorch.
* Move PyTorch models and tensors between CPU and GPU devices.
* Train a neural network using CUDA acceleration.
* Evaluate a trained neural network on a GPU.
* Monitor GPU utilization and memory consumption with `nvidia-smi`.
* Compare CPU and GPU execution using the same workload.
* Explain why GPU acceleration benefits computationally intensive AI workloads more than very small workloads.

---

## Key Takeaway

**CUDA enables PyTorch to execute computationally intensive tensor operations on NVIDIA GPUs. Effective GPU acceleration requires both the model and its tensors to reside on the GPU, while monitoring and benchmarking help determine whether the workload actually benefits from hardware acceleration.**
