# Hands-On Lab: Track Machine Learning Experiments with MLflow

In this lab, you will train a simple PyTorch model on the MNIST handwritten-digit dataset while using **MLflow Tracking** to record experiment parameters, metrics, model artifacts, and supporting files.

You will log training configuration, capture loss values across epochs, store the trained model, compare experiments with different hyperparameters, and load a previously logged model for evaluation or inference.

The lab demonstrates how experiment tracking improves **traceability, reproducibility, and comparison of machine learning training runs**.

---

## Goal

Track parameters, metrics, and model artifacts from a PyTorch training experiment using MLflow Tracking.

## Estimated Time

**60–90 minutes**

## Tools

This lab uses:

* Python
* PyTorch
* TorchVision
* MLflow
* Matplotlib
* Optional CUDA-enabled GPU

---

## Prerequisites

Before beginning the lab, make sure you have:

* Python installed
* PyTorch installed
* TorchVision installed
* `pip` available
* A web browser
* Optional access to a CUDA-enabled NVIDIA GPU

You can verify Python with:

```bash
python --version
```

Verify PyTorch:

```bash
python -c "import torch; print(torch.__version__)"
```

---

## Experiment Tracking Workflow

The lab follows this general workflow:

```text
Training Code
     ↓
Hyperparameters
Metrics
Artifacts
Model
     ↓
MLflow Tracking Server
     ↓
Experiment
     ↓
Multiple Runs
     ↓
Comparison / Reproduction
```

MLflow provides a central record of how each training run was executed and what outputs it produced.

---

## Step 1: Install and Start MLflow

Install MLflow in your Python environment:

```bash
pip install mlflow
```

Start the local MLflow Tracking Server:

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5000
```

Keep this terminal running while completing the lab.

By default, the MLflow web interface is available at:

```text
http://127.0.0.1:5000
```

Open the address in your web browser.

You should see the MLflow interface.

The basic relationship is:

```text
Python Training Script
        ↓
MLflow Client
        ↓
http://127.0.0.1:5000
        ↓
MLflow Tracking Server
        ↓
Experiment Runs
```

---

## Step 2: Verify the MLflow Environment

Open another terminal or Python environment.

Verify that MLflow is installed:

```python
import mlflow

print("MLflow version:", mlflow.__version__)
```

Next, configure the tracking server:

```python
import mlflow

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)
```

Create or select an experiment:

```python
mlflow.set_experiment(
    "mnist-pytorch"
)
```

The tracking URI tells the MLflow client where experiment information should be sent.

The experiment name provides a logical container for related runs.

The hierarchy is:

```text
MLflow Tracking Server
        ↓
mnist-pytorch Experiment
        │
        ├── Run 1
        ├── Run 2
        ├── Run 3
        └── ...
```

---

## Step 3: Load the MNIST Dataset

MNIST contains handwritten digit images and is well suited for demonstrating experiment tracking because it can be trained quickly.

Create the training dataset:

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.ToTensor()

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

The transformation converts each image into a PyTorch tensor.

The DataLoader organizes the dataset into batches of:

```text
64 samples
```

The dataset is shuffled so that each epoch can process examples in a different order.

The data flow is:

```text
MNIST Dataset
      ↓
Tensor Conversion
      ↓
DataLoader
      ↓
Training Batches
```

---

## Step 4: Define the PyTorch Model

Create a simple fully connected neural network:

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(
            28 * 28,
            128
        )

        self.fc2 = nn.Linear(
            128,
            64
        )

        self.fc3 = nn.Linear(
            64,
            10
        )

    def forward(self, x):
        x = x.view(
            -1,
            28 * 28
        )

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        return self.fc3(x)
```

Each MNIST image contains:

```text
28 × 28 = 784 pixels
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

The final layer produces ten values corresponding to the ten possible digit classes.

---

## Step 5: Train the Model with MLflow Logging

Next, integrate MLflow directly into the PyTorch training workflow.

The run will record:

* Learning rate
* Batch size
* Execution device
* Training loss
* Final trained model
* MLflow run ID

Import the required libraries:

```python
import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch
```

Configure the execution device:

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)
```

Create the model:

```python
model = Net().to(device)
```

Configure the loss function:

```python
criterion = nn.CrossEntropyLoss()
```

Configure the Adam optimizer:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Start an MLflow run:

```python
with mlflow.start_run() as run:

    mlflow.log_param(
        "learning_rate",
        0.001
    )

    mlflow.log_param(
        "batch_size",
        64
    )

    mlflow.log_param(
        "device",
        str(device)
    )

    for epoch in range(2):

        running_loss = 0.0

        for images, labels in trainloader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = (
            running_loss
            / len(trainloader)
        )

        mlflow.log_metric(
            "training_loss",
            avg_loss,
            step=epoch
        )

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {avg_loss:.4f}"
        )

    mlflow.pytorch.log_model(
        model,
        name="model"
    )

    print(
        "Run ID:",
        run.info.run_id
    )
```

MLflow now maintains a structured record of the experiment.

The relationship is:

```text
Training Run
    │
    ├── Parameters
    │     ├── learning_rate
    │     ├── batch_size
    │     └── device
    │
    ├── Metrics
    │     └── training_loss
    │
    └── Model
          └── PyTorch Artifact
```

The run ID uniquely identifies the training execution and can later be used to retrieve its artifacts and model.

---

## Step 6: Review the Experiment in MLflow

Open:

```text
http://127.0.0.1:5000
```

in your browser.

Select the:

```text
mnist-pytorch
```

experiment.

You should see the training run created in the previous step.

MLflow displays information such as:

* Run ID
* Start time
* Duration
* Parameters
* Metrics
* Artifacts
* Model outputs

Select the run and inspect:

```text
learning_rate = 0.001
batch_size = 64
device = cpu or cuda
```

You should also see the recorded:

```text
training_loss
```

metric for each epoch.

The logged model should appear among the run artifacts or model outputs, depending on the MLflow version.

---

## Step 7: Log a Training-Loss Artifact

Experiment tracking can include files in addition to parameters and scalar metrics.

Common artifacts include:

* Training plots
* Confusion matrices
* Evaluation reports
* Configuration files
* Model checkpoints
* Dataset summaries
* Diagnostic outputs

Create a simple loss curve:

```python
import matplotlib.pyplot as plt
import mlflow

losses = [
    0.9,
    0.5,
    0.3
]

plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.savefig(
    "loss_curve.png"
)

plt.close()
```

Log the file as an MLflow artifact:

```python
with mlflow.start_run():

    mlflow.log_artifact(
        "loss_curve.png"
    )
```

The image is stored with the MLflow run and can be reviewed later through the MLflow interface.

> **Note**
>
> The `losses` values above are example values. In a real training workflow, capture the actual loss values from the training loop and use those values to generate the plot.

---

## Step 8: Compare Multiple Training Runs

One of the primary benefits of MLflow is the ability to compare multiple experiment configurations.

Run the training workflow with:

```text
Run 1
learning_rate = 0.001
```

Then change the learning rate:

```text
Run 2
learning_rate = 0.01
```

Each execution creates a separate run:

```text
mnist-pytorch
     │
     ├── Run 1
     │    ├── learning_rate = 0.001
     │    ├── training_loss
     │    └── model
     │
     └── Run 2
          ├── learning_rate = 0.01
          ├── training_loss
          └── model
```

Open the MLflow interface and select both runs.

Compare:

* Learning rate
* Final training loss
* Loss progression
* Training duration
* Device
* Resulting model

Use a results table such as:

| Run       | Learning Rate | Batch Size | Device | Final Loss |
| --------- | ------------: | ---------: | ------ | ---------: |
| **Run 1** |         0.001 |         64 | ______ |     ______ |
| **Run 2** |         0.010 |         64 | ______ |     ______ |

Consider:

1. Which learning rate produced the lower final loss?
2. Which run converged more smoothly?
3. Did either learning rate cause unstable training?
4. Were training times significantly different?
5. Did GPU execution change the experiment duration?

This demonstrates how experiment tracking allows configurations to be compared using recorded evidence rather than memory or manually maintained notes.

---

## Step 9: Load a Model from MLflow

MLflow can retrieve a model associated with a previous experiment run.

Use the run ID printed during training.

Replace:

```text
<run_id>
```

with the actual MLflow run ID.

```python
import mlflow.pytorch

run_id = "<run_id>"

model_uri = (
    f"runs:/{run_id}/model"
)

loaded_model = (
    mlflow.pytorch.load_model(
        model_uri
    )
)

loaded_model.eval()
```

The workflow is:

```text
MLflow Run
    ↓
Run ID
    ↓
Model URI
    ↓
Logged PyTorch Model
    ↓
Load Model
    ↓
Evaluation / Inference
```

Retrieving the model directly from the experiment run preserves its relationship with:

* Training parameters
* Metrics
* Artifacts
* Run metadata

This is an important foundation for reproducibility.

---

## Optional: Verify the Loaded Model

You can verify that the loaded model can process an MNIST sample.

```python
loaded_model = loaded_model.to(device)

images, labels = next(
    iter(trainloader)
)

images = images.to(device)

loaded_model.eval()

with torch.no_grad():

    outputs = loaded_model(
        images
    )

    predictions = outputs.argmax(
        dim=1
    )

print(
    "Predictions:",
    predictions[:10]
)
```

This confirms that the model retrieved from MLflow can be used independently of the original training run.

---

## Step 10: Review Experiment Traceability

MLflow creates a relationship between the model and the experiment that produced it.

Without experiment tracking:

```text
model.pth
    ↓
Which learning rate?
Which batch size?
Which training run?
Which loss?
Which environment?
```

With MLflow:

```text
MLflow Run
    │
    ├── Parameters
    ├── Metrics
    ├── Artifacts
    ├── Model
    ├── Run ID
    └── Timestamps
```

This makes it much easier to understand how a particular model was produced.

---

## Step 11: Stop the MLflow Server

When the lab is complete, return to the terminal running:

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5000
```

Press:

```text
Ctrl+C
```

to stop the server.

If you created local MLflow files that are no longer required, you can remove them.

However, do not delete experiment data that you intend to use for:

* Future comparisons
* Model reproduction
* Analysis
* Debugging
* Auditing

---

## Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Installed MLflow
* [ ] Started the local MLflow Tracking Server
* [ ] Opened the MLflow web interface
* [ ] Created the `mnist-pytorch` experiment
* [ ] Loaded the MNIST dataset
* [ ] Created the PyTorch neural network
* [ ] Logged the learning rate
* [ ] Logged the batch size
* [ ] Logged the execution device
* [ ] Logged training-loss metrics
* [ ] Stored the trained PyTorch model
* [ ] Recorded the MLflow run ID
* [ ] Reviewed the run through the MLflow interface
* [ ] Logged an additional artifact
* [ ] Executed a second run with a different learning rate
* [ ] Compared multiple experiment runs
* [ ] Loaded a previously logged model
* [ ] Stopped the MLflow server

---

## Learning Outcomes

After completing this lab, you should be able to:

* Configure and run a local MLflow Tracking environment.
* Organize training runs within an MLflow experiment.
* Log hyperparameters and training metrics.
* Store trained PyTorch models with experiment runs.
* Log additional files and visualizations as artifacts.
* Compare multiple training configurations through the MLflow interface.
* Retrieve a previously logged PyTorch model.
* Understand the relationship among experiments, runs, parameters, metrics, artifacts, and models.
* Explain how MLflow supports experiment traceability and reproducibility.

---

## Key Takeaway

**MLflow provides a structured way to record how machine learning models are trained. By associating parameters, metrics, artifacts, and models with individual experiment runs, MLflow makes it easier to compare training configurations, reproduce results, and understand exactly how a particular model was produced.**
