# Hands-On Lab: Serve an Image Classifier with FastAPI

In this lab, you will deploy a pretrained image classification model as a REST API using **FastAPI**.

The application uses a pretrained **ResNet18** model from Torchvision and demonstrates the complete model-serving workflow:

```text
Image Upload
    ↓
Request Validation
    ↓
Image Preprocessing
    ↓
PyTorch Inference
    ↓
Top-K Predictions
    ↓
JSON Response
```

The service automatically uses a CUDA-enabled GPU when one is available and otherwise runs on the CPU.

You will first run and test the application locally and then optionally package it as a Docker container.

---

## Lab Objective

Build a FastAPI-based image classification service that:

* Accepts JPEG and PNG images
* Loads a pretrained ResNet18 model
* Applies the preprocessing associated with the pretrained weights
* Performs inference with PyTorch
* Returns structured top-k ImageNet predictions
* Reports inference time
* Exposes a health endpoint
* Automatically uses CUDA when available
* Can optionally run inside Docker

---

## Estimated Time

**Approximately 60–90 minutes**

---

## Tools

This lab uses:

* Python 3.10 or later
* PyTorch
* TorchVision
* FastAPI
* Uvicorn
* Pillow
* Requests
* Optional CUDA-enabled GPU
* Optional Docker

---

## Architecture

The completed workflow follows this pattern:

```text
                  Client
                    │
                    ▼
              FastAPI Service
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
     /healthz               /predict
                               │
                               ▼
                       Validate Image
                               │
                               ▼
                       Decode with PIL
                               │
                               ▼
                  TorchVision Preprocess
                               │
                               ▼
                         ResNet18
                               │
                               ▼
                           Softmax
                               │
                               ▼
                     Top-K Predictions
                               │
                               ▼
                         JSON Response
```

---

## Prerequisites

Before beginning the lab, verify that Python is available:

```bash
python --version
```

You should have:

* Python 3.10 or later
* `pip`
* Network access for downloading model weights and labels
* Uvicorn
* Optional CUDA-compatible environment
* Docker only if you complete the containerization section

A GPU is not required.

PyTorch will select:

```text
CUDA GPU
```

when available, otherwise:

```text
CPU
```

---

# Step 1: Create the Project

Create a project directory:

```bash
mkdir fastapi-image-classifier
cd fastapi-image-classifier
```

Organize the project as:

```text
fastapi-image-classifier/
├── app/
│   └── main.py
├── scripts/
│   └── client.py
├── imagenet_classes.txt
├── requirements.txt
└── Dockerfile
```

The structure separates the serving application from the test client.

| File                   | Purpose                              |
| ---------------------- | ------------------------------------ |
| `app/main.py`          | FastAPI model-serving application    |
| `scripts/client.py`    | Python client for testing inference  |
| `imagenet_classes.txt` | Human-readable ImageNet class labels |
| `requirements.txt`     | Python dependencies                  |
| `Dockerfile`           | Optional container definition        |

---

# Step 2: Install the Dependencies

Create:

```text
requirements.txt
```

Add:

```text
fastapi
uvicorn[standard]
pillow
torch
torchvision
pydantic
python-multipart
requests
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

You may also create a virtual environment first:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Then install:

```bash
pip install -r requirements.txt
```

> **Production Note**
>
> The lab intentionally uses unpinned package names to keep the example simple. Production deployments should test and pin compatible package versions for reproducibility.

---

# Step 3: Download the ImageNet Class Labels

ResNet18 produces predictions across **1,000 ImageNet classes**.

Download the human-readable labels:

```bash
wget \
  https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

If `wget` is unavailable:

```bash
curl -O \
  https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

Verify:

```bash
ls -l imagenet_classes.txt
```

The file should remain in the project root:

```text
fastapi-image-classifier/
├── imagenet_classes.txt
└── ...
```

The model produces numeric output indices such as:

```text
207
```

and the labels file translates the index into a human-readable name such as:

```text
golden retriever
```

---

# Step 4: Build the FastAPI Application

Create:

```text
app/main.py
```

Add:

```python
import io
import time
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from torchvision import models
from torchvision.models import ResNet18_Weights


app = FastAPI(
    title="Image Classifier API"
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


WEIGHTS = ResNet18_Weights.DEFAULT

MODEL = models.resnet18(
    weights=WEIGHTS
).to(DEVICE)

MODEL.eval()


PREPROCESS = WEIGHTS.transforms()


with open(
    "imagenet_classes.txt",
    "r"
) as f:
    LABELS = [
        line.strip()
        for line in f.readlines()
    ]


class Prediction(BaseModel):
    index: int
    label: str
    probability: float


class PredictResponse(BaseModel):
    model: str
    device: str
    top_k: int
    time_ms: float
    predictions: List[Prediction]


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "device": DEVICE
    }


@app.post(
    "/predict",
    response_model=PredictResponse
)
async def predict(
    image: UploadFile = File(...),
    top_k: int = Query(
        5,
        ge=1,
        le=20
    )
):

    if image.content_type not in {
        "image/jpeg",
        "image/png"
    }:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported file type: "
                f"{image.content_type}"
            )
        )

    try:
        raw = await image.read()

        img = Image.open(
            io.BytesIO(raw)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to process "
                "the uploaded image."
            )
        ) from exc


    tensor = (
        PREPROCESS(img)
        .unsqueeze(0)
        .to(DEVICE)
    )


    start_time = time.perf_counter()


    with torch.inference_mode():

        logits = MODEL(tensor)

        probabilities = F.softmax(
            logits,
            dim=1
        )

        (
            top_probabilities,
            top_indices
        ) = probabilities.topk(
            top_k,
            dim=1
        )


    elapsed_ms = (
        time.perf_counter()
        - start_time
    ) * 1000


    predictions = [

        Prediction(
            index=int(
                index.item()
            ),

            label=LABELS[
                int(index.item())
            ],

            probability=float(
                probability.item()
            )
        )

        for probability, index
        in zip(
            top_probabilities[0],
            top_indices[0]
        )
    ]


    return PredictResponse(
        model="resnet18",
        device=DEVICE,
        top_k=top_k,
        time_ms=elapsed_ms,
        predictions=predictions
    )
```

---

## How the Application Works

The model is loaded once when the application starts:

```text
Application Startup
       ↓
Load ResNet18
       ↓
Load Pretrained Weights
       ↓
Move Model to CPU/GPU
       ↓
Set Evaluation Mode
       ↓
Service Ready
```

This is more efficient than loading the model for every request.

---

## Device Selection

The application selects:

```python
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
```

So the serving path is:

```text
CUDA Available?
      │
   ┌──┴──┐
   │     │
  Yes    No
   │     │
   ▼     ▼
  GPU    CPU
```

---

## Preprocessing

The application uses:

```python
PREPROCESS = WEIGHTS.transforms()
```

This is important because the preprocessing is tied to the pretrained model weights.

The request flow becomes:

```text
Uploaded Image
      ↓
Decode
      ↓
RGB Conversion
      ↓
Resize / Crop
      ↓
Tensor Conversion
      ↓
Normalization
      ↓
ResNet18
```

---

## Inference Mode

Inference is executed using:

```python
with torch.inference_mode():
```

This disables gradient-related work that is unnecessary during prediction.

Benefits include:

* Lower memory usage
* Reduced overhead
* Cleaner inference execution

---

# Step 5: Run the API

From the project root, start Uvicorn:

```bash
uvicorn \
  app.main:app \
  --host 0.0.0.0 \
  --port 8000
```

The service should now be available at:

```text
http://localhost:8000
```

---

## Test the Health Endpoint

Run:

```bash
curl \
  http://localhost:8000/healthz
```

A CPU response may look like:

```json
{
  "status": "ok",
  "device": "cpu"
}
```

With CUDA available:

```json
{
  "status": "ok",
  "device": "cuda"
}
```

The health endpoint can later be used with systems such as:

* Kubernetes probes
* Load balancers
* Monitoring platforms
* Deployment pipelines

---

## Interactive API Documentation

FastAPI automatically exposes Swagger documentation at:

```text
http://localhost:8000/docs
```

Open this URL in your browser.

You can use the interface to:

* Inspect endpoints
* Upload an image
* Modify `top_k`
* Execute requests
* Inspect responses

---

# Step 6: Test Image Classification

Place a test image in the project directory:

```text
sample.jpg
```

Send it to the API:

```bash
curl -X POST \
  "http://localhost:8000/predict?top_k=5" \
  -F "image=@sample.jpg"
```

A response may resemble:

```json
{
  "model": "resnet18",
  "device": "cpu",
  "top_k": 5,
  "time_ms": 52.4,
  "predictions": [
    {
      "index": 207,
      "label": "golden retriever",
      "probability": 0.72
    }
  ]
}
```

The exact:

* Labels
* Probabilities
* Inference time

depend on the uploaded image and hardware.

---

## Prediction Flow

```text
sample.jpg
    ↓
POST /predict
    ↓
Validate MIME Type
    ↓
Decode Image
    ↓
Preprocess
    ↓
ResNet18
    ↓
Softmax
    ↓
Top-K
    ↓
JSON
```

---

# Step 7: Test Request Validation

The service accepts:

```text
image/jpeg
image/png
```

Try submitting an unsupported file:

```bash
curl -X POST \
  "http://localhost:8000/predict" \
  -F "image=@sample.txt"
```

The service should reject the request with an HTTP:

```text
415 Unsupported Media Type
```

Malformed image content should result in:

```text
400 Bad Request
```

This demonstrates why inference APIs should validate inputs before model execution.

---

# Step 8: Create a Python Client

Create:

```text
scripts/client.py
```

Add:

```python
import sys

import requests


URL = (
    "http://localhost:8000/predict"
)


if len(sys.argv) != 2:
    print(
        "Usage: "
        "python scripts/client.py "
        "<image-path>"
    )

    sys.exit(1)


image_path = sys.argv[1]


with open(
    image_path,
    "rb"
) as image_file:

    files = {
        "image": (
            image_path,
            image_file,
            "image/jpeg"
        )
    }


    response = requests.post(
        URL,
        files=files,
        params={
            "top_k": 5
        },
        timeout=30
    )


response.raise_for_status()

print(
    response.json()
)
```

Run:

```bash
python \
  scripts/client.py \
  sample.jpg
```

The client:

```text
Reads Image
    ↓
Creates Multipart Request
    ↓
POST /predict
    ↓
Receives JSON
    ↓
Prints Predictions
```

---

# Step 9: Test Multiple Top-K Values

The endpoint supports:

```text
1 ≤ top_k ≤ 20
```

Test a single result:

```bash
curl -X POST \
  "http://localhost:8000/predict?top_k=1" \
  -F "image=@sample.jpg"
```

Test ten results:

```bash
curl -X POST \
  "http://localhost:8000/predict?top_k=10" \
  -F "image=@sample.jpg"
```

This lets the client control how many prediction candidates are returned.

---

# Step 10: Observe CPU and GPU Execution

The health endpoint reports the execution device.

Run:

```bash
curl \
  http://localhost:8000/healthz
```

If using CUDA, monitor the GPU in another terminal:

```bash
watch -n 1 nvidia-smi
```

Then send repeated inference requests.

You should see the serving process appear in GPU memory and utilization statistics.

---

# Step 11: Containerize the API

Create:

```text
Dockerfile
```

Add:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY app/ ./app/

COPY imagenet_classes.txt .

EXPOSE 8000

CMD [
    "uvicorn",
    "app.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]
```

The image contains:

```text
Container
   │
   ├── Python
   ├── PyTorch
   ├── TorchVision
   ├── FastAPI
   ├── Uvicorn
   ├── app/main.py
   └── imagenet_classes.txt
```

---

## Build the Image

Run:

```bash
docker build \
  -t fastapi-image-classifier \
  .
```

Verify:

```bash
docker images \
  | grep fastapi-image-classifier
```

---

## Run the Container

Run:

```bash
docker run \
  --rm \
  -p 8000:8000 \
  fastapi-image-classifier
```

Test the health endpoint:

```bash
curl \
  http://localhost:8000/healthz
```

Test prediction:

```bash
curl -X POST \
  "http://localhost:8000/predict?top_k=5" \
  -F "image=@sample.jpg"
```

The Docker deployment should behave like the locally executed application.

---

## GPU Container Note

The Dockerfile above is a straightforward CPU-oriented example.

GPU-enabled container deployment requires:

* CUDA-compatible container base image
* NVIDIA drivers on the host
* NVIDIA Container Toolkit/runtime
* CUDA-compatible PyTorch packages
* Appropriate GPU container configuration

Conceptually:

```text
Host NVIDIA GPU
      ↓
NVIDIA Driver
      ↓
NVIDIA Container Runtime
      ↓
CUDA-Compatible Container
      ↓
PyTorch
      ↓
ResNet18 Inference
```

---

# Step 12: Verify the Complete Workflow

Test several images.

For each image, observe:

* Returned labels
* Probabilities
* Top-k behavior
* Inference time
* CPU or GPU execution

Also verify:

* JPEG files are accepted
* PNG files are accepted
* Unsupported file types are rejected
* `/healthz` responds correctly
* `/docs` is accessible
* Containerized behavior matches local behavior

Use a table such as:

| Test              | Result     |
| ----------------- | ---------- |
| Health endpoint   | __________ |
| JPEG upload       | __________ |
| PNG upload        | __________ |
| Invalid file type | __________ |
| `top_k=1`         | __________ |
| `top_k=5`         | __________ |
| Local deployment  | __________ |
| Docker deployment | __________ |
| CPU/GPU device    | __________ |

---

# Understanding the Complete Serving Pipeline

The complete model-serving path is:

```text
Client
   ↓
HTTP Request
   ↓
FastAPI
   ↓
Input Validation
   ↓
PIL Image Decode
   ↓
TorchVision Transform
   ↓
PyTorch Tensor
   ↓
CPU or GPU
   ↓
ResNet18
   ↓
Logits
   ↓
Softmax
   ↓
Top-K Selection
   ↓
Pydantic Response
   ↓
JSON
   ↓
Client
```

This pattern is common in production model-serving systems even when the underlying model, framework, or deployment infrastructure changes.

---

# Production Improvements

This lab intentionally implements a lightweight serving architecture.

A production system may additionally include:

## Authentication

Protect endpoints using:

* API keys
* OAuth
* JWT
* Service-to-service identity

## Observability

Collect:

* Request count
* Error rate
* p50 latency
* p95 latency
* p99 latency
* Model inference latency
* CPU utilization
* GPU utilization
* Memory utilization

## Request Controls

Add:

* Maximum upload size
* Request timeout
* Rate limiting
* Concurrency controls
* Queue limits

## Model Versioning

Return model metadata such as:

```json
{
  "model": "resnet18",
  "version": "1.0"
}
```

## Scaling

Deploy multiple replicas behind:

* Kubernetes Service
* Ingress
* Cloud load balancer
* API gateway

## Accelerated Serving

For higher-throughput GPU workloads, evaluate:

* Dynamic batching
* NVIDIA Triton
* TorchServe-style serving patterns
* Dedicated inference runtimes
* Accelerator-aware autoscaling

---

# Troubleshooting

## API Does Not Start

Check:

```bash
uvicorn \
  app.main:app \
  --host 0.0.0.0 \
  --port 8000
```

Confirm that you are running the command from the project root.

Verify:

```text
imagenet_classes.txt
```

exists.

---

## ModuleNotFoundError

Install dependencies:

```bash
pip install \
  -r requirements.txt
```

---

## File Upload Fails

Verify that:

```text
python-multipart
```

is installed.

FastAPI requires it for multipart file uploads.

---

## Model Weight Download Fails

Verify that the system has network access.

Pretrained weights may need to be downloaded the first time the application starts.

---

## CUDA Is Not Used

Verify:

```bash
nvidia-smi
```

Then run:

```python
import torch

print(
    torch.cuda.is_available()
)
```

If the result is:

```text
False
```

check:

* NVIDIA driver
* CUDA compatibility
* PyTorch build
* Container runtime configuration if using Docker

---

## Image Is Rejected

The API accepts:

```text
image/jpeg
image/png
```

Verify the client is sending the correct MIME type.

---

# Cleanup

Stop Uvicorn using:

```text
Ctrl+C
```

If Docker is running the service:

```bash
docker ps
```

Stop the container if necessary:

```bash
docker stop \
  <CONTAINER_ID>
```

Optionally remove the image:

```bash
docker rmi \
  fastapi-image-classifier
```

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Created the FastAPI project structure
* [ ] Installed the required dependencies
* [ ] Downloaded the ImageNet class labels
* [ ] Loaded pretrained ResNet18 weights
* [ ] Configured automatic CPU/GPU selection
* [ ] Used the preprocessing associated with the pretrained weights
* [ ] Created `/healthz`
* [ ] Created `/predict`
* [ ] Validated uploaded file types
* [ ] Decoded uploaded images
* [ ] Executed inference using `torch.inference_mode()`
* [ ] Calculated top-k predictions
* [ ] Returned structured JSON responses
* [ ] Started the service with Uvicorn
* [ ] Tested the API with `curl`
* [ ] Used FastAPI `/docs`
* [ ] Created a Python client
* [ ] Tested multiple top-k values
* [ ] Verified CPU or GPU execution
* [ ] Built the optional Docker image
* [ ] Tested the containerized application
* [ ] Verified unsupported files are rejected

---

# Learning Outcomes

After completing this lab, you should be able to:

* Expose a pretrained PyTorch model through a FastAPI REST API.
* Load pretrained TorchVision model weights.
* Apply the preprocessing transforms associated with model weights.
* Process uploaded JPEG and PNG files.
* Execute inference using `torch.inference_mode()`.
* Convert model logits into probabilities.
* Return structured top-k predictions.
* Use Pydantic models to define API responses.
* Implement a health endpoint for model-serving applications.
* Automatically select CPU or CUDA execution.
* Test a serving endpoint with `curl` and Python.
* Package a model-serving application in Docker.
* Explain the complete workflow from client request to model response.

---

# Key Takeaway

**Serving a machine learning model requires more than loading the model and calling it. A production-oriented inference service must also handle input validation, preprocessing, device management, safe inference execution, structured responses, health checks, testing, and deployment packaging.**

FastAPI provides a lightweight foundation for exposing PyTorch models as application-accessible services, while TorchVision's pretrained weights and associated transformations help keep inference preprocessing consistent with the original model configuration.

The resulting serving pipeline is:

```text
Image
   ↓
API
   ↓
Validation
   ↓
Preprocessing
   ↓
Model Inference
   ↓
Top-K Predictions
   ↓
JSON Response
```
