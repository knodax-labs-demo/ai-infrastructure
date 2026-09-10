# Hands-On Lab: Deploy a Model on NVIDIA Triton Inference Server

In this lab, you will deploy a pretrained image-classification model using **NVIDIA Triton Inference Server**.

You will export a pretrained **ResNet-50** model from PyTorch to ONNX, organize it using Triton's model repository structure, start Triton with Docker, and send a real inference request through the Triton HTTP API.

You will also inspect Triton's readiness, model metadata, and Prometheus metrics endpoints, then experiment with **dynamic batching** and multiple model instances. Optional extensions introduce TensorRT optimization and Kubernetes deployment.

The complete workflow is:

```text
PyTorch Model
    ↓
Export to ONNX
    ↓
Triton Model Repository
    ↓
Triton Inference Server
    ↓
HTTP Request
    ↓
Inference
    ↓
Prediction Logits
    ↓
Client Processing
```

---

## Lab Objectives

By the end of this lab, you should be able to:

* Export a pretrained PyTorch model to ONNX.
* Create and configure a Triton model repository.
* Run Triton Inference Server with Docker.
* Send inference requests through Triton's HTTP endpoint.
* Access Triton health, metadata, and Prometheus metrics.
* Configure dynamic batching.
* Configure multiple model instances.
* Understand how TensorRT can optimize GPU inference.
* Deploy Triton to Kubernetes as an optional extension.

---

## Estimated Time

**Approximately 90–120 minutes**

---

## Tools

This lab uses:

* Python 3.10 or later
* PyTorch
* TorchVision
* NumPy
* Pillow
* Requests
* Docker
* NVIDIA Triton Inference Server
* Optional NVIDIA GPU
* Optional NVIDIA Container Toolkit
* Optional Kubernetes
* Optional TensorRT

---

## Triton Ports

Triton commonly exposes:

| Port   | Purpose            |
| ------ | ------------------ |
| `8000` | HTTP inference API |
| `8001` | gRPC API           |
| `8002` | Prometheus metrics |

Make sure these ports are available before starting Triton.

---

## Architecture

The lab implements the following separation:
<img width="1439" height="562" alt="triton-1" src="https://github.com/user-attachments/assets/47beef68-8fe3-4e1c-a46d-4a3860c320ca" />

The client handles preprocessing and response interpretation, while Triton manages model loading, execution, batching, concurrency, and metrics.

This separation allows the serving backend to be optimized without significantly changing the client application.

---

# Step 1: Install the Required Python Packages

Install the client-side dependencies:

```bash
pip install \
  torch \
  torchvision \
  pillow \
  numpy \
  requests
```

Verify PyTorch:

```bash
python -c "import torch; print(torch.__version__)"
```

If using an NVIDIA GPU:

```bash
nvidia-smi
```

Verify Docker:

```bash
docker --version
```

If GPU-enabled containers will be used, make sure the NVIDIA Container Toolkit is configured.

---

# Step 2: Create the Project and Model Repository

Create the working directory:

```bash
mkdir -p \
  ~/triton-model-lab/models/resnet50_onnx/1
```

Move into it:

```bash
cd ~/triton-model-lab
```

The project should have this structure:

```text
triton-model-lab/
├── models/
│   └── resnet50_onnx/
│       ├── config.pbtxt
│       └── 1/
│           └── model.onnx
├── export_onnx.py
└── client_http.py
```

Triton uses a standardized model repository hierarchy:

```text
Model Repository
      │
      ▼
Model Name
      │
      ▼
Version Directory
      │
      ▼
Model File
```

For this lab:

```text
models/
└── resnet50_onnx/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

The directory:

```text
1/
```

represents model version 1.

Future versions could be placed in:

```text
2/
3/
4/
```

and so on.

---

# Step 3: Create the Triton Model Configuration

Create:

```text
models/resnet50_onnx/config.pbtxt
```

Add:

```text
name: "resnet50_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 16

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [1000]
  }
]

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 1000
}
```

This configuration describes:

* Model name
* Serving backend
* Maximum batch size
* Input tensor
* Output tensor
* GPU execution
* Dynamic batching behavior

---

## Input Shape

The model expects:

```text
[3, 224, 224]
```

representing:

```text
Channels × Height × Width
```

or:

```text
RGB × 224 × 224
```

---

## Output Shape

The model produces:

```text
[1000]
```

values corresponding to the 1,000 ImageNet classes.

---

## Dynamic Batching

The configuration enables:

```text
preferred_batch_size: [4, 8, 16]
```

with a maximum queue delay of:

```text
1000 microseconds
```

The serving path is:

```text
Request 1 ─┐
Request 2 ─┤
Request 3 ─┼─► Dynamic Batch ─► GPU
Request 4 ─┘
```

This can improve accelerator utilization when sufficient concurrent traffic exists.

> **CPU-Only Note**
>
> If you are completing the lab without a compatible NVIDIA GPU, change:
>
> ```text
> kind: KIND_GPU
> ```
>
> to:
>
> ```text
> kind: KIND_CPU
> ```

---

# Step 4: Export ResNet-50 to ONNX

Create:

```text
export_onnx.py
```

Add:

```python
import torch
import torchvision as tv


def main():

    model = tv.models.resnet50(
        weights="DEFAULT"
    ).eval()

    dummy = torch.randn(
        1,
        3,
        224,
        224
    )

    torch.onnx.export(
        model,
        dummy,
        "models/resnet50_onnx/1/model.onnx",

        input_names=[
            "input"
        ],

        output_names=[
            "logits"
        ],

        dynamic_axes={
            "input": {
                0: "batch"
            },

            "logits": {
                0: "batch"
            }
        },

        opset_version=17
    )

    print(
        "Exported to "
        "models/resnet50_onnx/1/model.onnx"
    )


if __name__ == "__main__":
    main()
```

Run:

```bash
python export_onnx.py
```

Verify:

```bash
ls -lh \
  models/resnet50_onnx/1/model.onnx
```

---

## Why Dynamic Axes Matter

Without a dynamic batch dimension, the exported model could effectively be limited to the batch size represented by the dummy tensor.

The export explicitly defines:

```text
Batch Dimension = Dynamic
```

for both input and output.

This allows Triton to serve batches such as:

```text
1
4
8
16
```

within the configured limits.

---

## Verify Name Consistency

The ONNX names:

```text
input
logits
```

must match the names in:

```text
config.pbtxt
```

A mismatch can prevent Triton from loading the model.

---

# Step 5: Start Triton with Docker

For reproducible environments, use a specific Triton release tag compatible with your NVIDIA driver rather than relying permanently on an unversioned image.

Set the selected tag:

```bash
export TRITON_TAG=<compatible-triton-tag>
```

For a GPU environment:

```bash
docker run \
  --rm \
  -it \
  --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "$PWD/models:/models" \
  nvcr.io/nvidia/tritonserver:${TRITON_TAG} \
  tritonserver \
  --model-repository=/models \
  --exit-on-error=false
```

Triton now exposes:

```text
8000 → HTTP
8001 → gRPC
8002 → Metrics
```

and mounts:

```text
$PWD/models
```

inside the container as:

```text
/models
```

---

## Startup Flow

```text
Docker Container
      ↓
Triton Starts
      ↓
Reads /models
      ↓
Reads config.pbtxt
      ↓
Loads model.onnx
      ↓
Initializes ONNX Runtime
      ↓
Creates Model Instance
      ↓
Service Ready
```

Inspect the startup logs.

Verify that:

```text
resnet50_onnx
```

loads successfully.

A configuration error, shape mismatch, backend problem, or model-loading failure should appear in the Triton logs.

---

## CPU-Only Execution

For CPU testing:

1. Remove:

```text
--gpus all
```

from the Docker command.

2. Change the model configuration to:

```text
kind: KIND_CPU
```

---

# Step 6: Verify Server Health

Open another terminal.

Check Triton readiness:

```bash
curl -s \
  http://localhost:8000/v2/health/ready
```

A successful response indicates that Triton is ready to receive inference traffic.

---

## Check Model Metadata

Run:

```bash
curl -s \
  http://localhost:8000/v2/models/resnet50_onnx
```

This returns model metadata such as:

* Model name
* Platform
* Inputs
* Outputs
* Datatypes
* Shapes

---

## Check Metrics

Run:

```bash
curl -s \
  http://localhost:8002/metrics \
  | head
```

Triton exposes Prometheus-compatible metrics that can later be collected by monitoring systems.

The observability path is:

```text
Triton
   │
   ├── Health
   ├── Model Metadata
   └── Metrics
          ↓
      Prometheus
          ↓
       Grafana
```

These signals are useful for identifying:

* Model availability
* Inference failures
* Request activity
* Latency changes
* GPU utilization
* Memory usage

---

# Step 7: Create the HTTP Inference Client

Create:

```text
client_http.py
```

Add:

```python
import argparse

import numpy as np
import requests
from PIL import Image


def preprocess(img):

    img = (
        img
        .convert("RGB")
        .resize((256, 256))
    )

    offset = (
        256 - 224
    ) // 2

    img = img.crop(
        (
            offset,
            offset,
            offset + 224,
            offset + 224
        )
    )

    x = (
        np.asarray(img)
        .astype("float32")
        / 255.0
    )

    mean = np.array(
        [
            0.485,
            0.456,
            0.406
        ],
        dtype=np.float32
    )

    std = np.array(
        [
            0.229,
            0.224,
            0.225
        ],
        dtype=np.float32
    )

    x = (
        x - mean
    ) / std

    x = np.transpose(
        x,
        (2, 0, 1)
    )[None, ...]

    return x


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image_path"
    )

    parser.add_argument(
        "--url",
        default=(
            "http://localhost:8000/"
            "v2/models/"
            "resnet50_onnx/infer"
        )
    )

    args = parser.parse_args()

    x = preprocess(
        Image.open(
            args.image_path
        )
    )

    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": list(
                    x.shape
                ),
                "datatype": "FP32",
                "data": (
                    x
                    .flatten()
                    .tolist()
                )
            }
        ],

        "outputs": [
            {
                "name": "logits"
            }
        ]
    }

    response = requests.post(
        args.url,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    output = (
        response
        .json()["outputs"][0]["data"]
    )

    logits = np.array(
        output,
        dtype=np.float32
    ).reshape(
        (
            x.shape[0],
            1000
        )
    )

    exps = np.exp(
        logits
        - logits.max(
            axis=1,
            keepdims=True
        )
    )

    probabilities = (
        exps
        / exps.sum(
            axis=1,
            keepdims=True
        )
    )

    top5 = (
        probabilities[0]
        .argsort()[-5:][::-1]
    )

    print(
        "Top-5 indices:",
        top5.tolist()
    )

    print(
        "Top-5 probabilities:",
        probabilities[0][top5].tolist()
    )


if __name__ == "__main__":
    main()
```

---

# Step 8: Send an Inference Request

Run:

```bash
python client_http.py \
  path/to/image.jpg
```

The client performs:

```text
Image
  ↓
RGB Conversion
  ↓
Resize to 256 × 256
  ↓
Center Crop to 224 × 224
  ↓
Normalize
  ↓
Convert HWC → CHW
  ↓
Add Batch Dimension
  ↓
HTTP Request to Triton
```

The request shape is:

```text
[1, 3, 224, 224]
```

Triton returns:

```text
1000 logits
```

The client then applies softmax and displays the top five class indices and probabilities.

Example:

```text
Top-5 indices: [207, 208, 176, 222, 209]
Top-5 probabilities: [0.72, 0.11, 0.05, 0.03, 0.02]
```

The exact results depend on the input image.

---

# Step 9: Understand the Triton Request Flow

The complete inference path is:

```text
Image
   ↓
Python Client
   ↓
Preprocessing
   ↓
FP32 Tensor
   ↓
HTTP /v2/models/resnet50_onnx/infer
   ↓
Triton
   ↓
ONNX Runtime
   ↓
ResNet-50
   ↓
GPU / CPU
   ↓
1000 Logits
   ↓
HTTP Response
   ↓
Softmax
   ↓
Top-5 Predictions
```

This architecture separates application logic from model execution.

---

# Step 10: Experiment with Dynamic Batching

The current configuration contains:

```text
dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 1000
}
```

Dynamic batching attempts to combine independent requests before executing them.

Without batching:

```text
Request 1 → GPU
Request 2 → GPU
Request 3 → GPU
Request 4 → GPU
```

With batching:

```text
Request 1 ─┐
Request 2 ─┤
Request 3 ─┼─► Batch ─► GPU
Request 4 ─┘
```

Potential benefits include:

* Higher throughput
* Better GPU utilization
* Lower cost per inference

The tradeoff is that Triton may wait briefly to assemble a larger batch.

This can increase queueing latency.

---

## Measure the Tradeoff

Generate concurrent requests and record:

| Configuration | Throughput |      p50 |      p95 |      p99 | GPU Util. |
| ------------- | ---------: | -------: | -------: | -------: | --------: |
| No batching   |      _____ | _____ ms | _____ ms | _____ ms |    _____% |
| Batch 4       |      _____ | _____ ms | _____ ms | _____ ms |    _____% |
| Batch 8       |      _____ | _____ ms | _____ ms | _____ ms |    _____% |
| Batch 16      |      _____ | _____ ms | _____ ms | _____ ms |    _____% |

Do not optimize for throughput alone.

The preferred configuration should satisfy the application's latency requirements.

---

# Step 11: Experiment with Multiple Model Instances

Modify:

```text
instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
```

to:

```text
instance_group [
  {
    kind: KIND_GPU
    count: 2
  }
]
```

Conceptually:

```text
Incoming Requests
        ↓
      Triton
        │
    ┌───┴───┐
    │       │
    ▼       ▼
Instance 1 Instance 2
    │       │
    └───┬───┘
        ▼
       GPU
```

Multiple model instances may increase concurrent execution.

However, they also consume additional GPU memory.

Measure:

* Throughput
* p50 latency
* p95 latency
* GPU utilization
* GPU memory usage

More instances do not automatically mean better performance.

---

# Step 12: Optional — Map Predictions to ImageNet Labels

The client currently displays numeric ImageNet class indices.

For a friendlier application, load the ImageNet labels and map:

```text
207
```

to something like:

```text
golden retriever
```

The response path becomes:

```text
Model Index
    ↓
ImageNet Label Lookup
    ↓
Human-Readable Prediction
```

This is optional because the main objective of the lab is serving infrastructure rather than user-interface formatting.

---

# Step 13: Optional — TensorRT Optimization

For NVIDIA GPU environments, the ONNX model can be converted into a TensorRT engine.

Create:

```bash
mkdir -p \
  models/resnet50_trt/1
```

A typical workflow is:

```text
PyTorch
   ↓
ONNX
   ↓
TensorRT Build
   ↓
model.plan
   ↓
Triton TensorRT Backend
```

The exact `trtexec` command depends on:

* TensorRT version
* GPU architecture
* Dynamic shape configuration
* Precision mode
* Optimization profiles

An FP16 engine can often improve performance on compatible NVIDIA hardware.

---

## TensorRT Model Configuration

Create:

```text
models/resnet50_trt/config.pbtxt
```

Add:

```text
name: "resnet50_trt"
platform: "tensorrt_plan"
max_batch_size: 16

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [1000]
  }
]

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 1000
}
```

After creating:

```text
model.plan
```

restart Triton and verify that both models load.

---

## Compare ONNX Runtime and TensorRT

Use the same input data and workload.

Record:

| Backend      | Throughput |      p50 |      p95 |      p99 | GPU Util. |
| ------------ | ---------: | -------: | -------: | -------: | --------: |
| ONNX Runtime |      _____ | _____ ms | _____ ms | _____ ms |    _____% |
| TensorRT     |      _____ | _____ ms | _____ ms | _____ ms |    _____% |

Do not assume TensorRT will improve every workload.

Measure the actual result.

---

# Step 14: Optional — Deploy Triton to Kubernetes

After validating Triton locally with Docker, it can be moved into Kubernetes.

Create:

```text
lab-07-k8s-triton.yaml
```

Add:

```yaml
apiVersion: apps/v1
kind: Deployment

metadata:
  name: triton

spec:
  replicas: 1

  selector:
    matchLabels:
      app: triton

  template:
    metadata:
      labels:
        app: triton

    spec:
      containers:
        - name: triton

          image: nvcr.io/nvidia/tritonserver:<compatible-triton-tag>

          args:
            - tritonserver
            - --model-repository=/models

          ports:
            - containerPort: 8000
            - containerPort: 8001
            - containerPort: 8002

          volumeMounts:
            - name: model-repo
              mountPath: /models

          resources:
            limits:
              nvidia.com/gpu: 1

      volumes:
        - name: model-repo

          persistentVolumeClaim:
            claimName: triton-model-repo

---
apiVersion: v1
kind: Service

metadata:
  name: triton-svc

spec:
  selector:
    app: triton

  ports:
    - name: http
      port: 8000
      targetPort: 8000

    - name: grpc
      port: 8001
      targetPort: 8001

    - name: metrics
      port: 8002
      targetPort: 8002
```

Apply:

```bash
kubectl apply \
  -f lab-07-k8s-triton.yaml
```

Verify:

```bash
kubectl get pods
```

and:

```bash
kubectl get svc
```

---

## Kubernetes Architecture

```text
Client
   ↓
Kubernetes Service
   ↓
Triton Pod
   │
   ├── HTTP 8000
   ├── gRPC 8001
   └── Metrics 8002
           │
           ▼
        GPU Node
```

This example assumes that:

```text
triton-model-repo
```

already exists as a PersistentVolumeClaim and contains the Triton model repository.

---

## Production Kubernetes Improvements

A production Triton deployment would typically add:

* Readiness probes
* Liveness probes
* Resource requests
* Security contexts
* Autoscaling
* Authentication
* Network policies
* TLS
* Object-backed model storage
* External load balancing
* Monitoring
* GPU-aware node scheduling

GPU worker nodes must also expose NVIDIA GPU resources to Kubernetes.

---

# Step 15: Monitor Inference

Triton's metrics endpoint is:

```text
http://localhost:8002/metrics
```

Inspect it:

```bash
curl -s \
  http://localhost:8002/metrics \
  | head
```

During testing, monitor:

* Request count
* Inference execution
* Queueing
* Latency
* GPU utilization
* GPU memory usage

If using NVIDIA GPU hardware, also run:

```bash
watch -n 1 nvidia-smi
```

---

## Serving Optimization Loop

A useful workflow is:

```text
Generate Load
     ↓
Measure Throughput
     ↓
Measure Latency
     ↓
Measure GPU Utilization
     ↓
Change One Setting
     ↓
Repeat
```

Possible settings include:

* Dynamic batch size
* Queue delay
* Number of model instances
* TensorRT
* FP16
* Concurrency
* Model version

---

# Troubleshooting

## Model Fails to Load

Inspect Triton startup logs.

Verify that `config.pbtxt` matches the ONNX model for:

* Input names
* Output names
* Datatypes
* Dimensions
* Batch behavior

Check:

```text
input
logits
FP32
[3, 224, 224]
[1000]
```

---

## HTTP 400 During Inference

Verify that the request uses:

```text
Input Name: input
Datatype: FP32
Shape: [1, 3, 224, 224]
```

Also verify that the JSON payload follows Triton's HTTP inference protocol.

---

## Triton Is Running but Model Is Not Ready

Check:

```bash
curl -s \
  http://localhost:8000/v2/models/resnet50_onnx
```

Review Docker logs for model-loading errors.

---

## GPU Is Not Used

Verify:

```bash
nvidia-smi
```

Confirm Docker has GPU access:

```bash
docker run \
  --rm \
  --gpus all \
  nvidia/cuda:12.0.0-base-ubuntu22.04 \
  nvidia-smi
```

Also confirm:

```text
kind: KIND_GPU
```

in the Triton model configuration.

---

## Dynamic Batching Does Not Improve Performance

A single sequential client is usually insufficient to demonstrate the benefit of dynamic batching.

Generate concurrent traffic.

Then compare:

* Throughput
* Queue latency
* Execution latency
* GPU utilization

---

## GPU Memory Usage Is Too High

Reduce:

* Model instance count
* Preferred batch size
* Maximum batch size
* Concurrent request volume

Monitor:

```bash
nvidia-smi
```

---

# Lab Validation

Before completing the lab, verify that:

* [ ] ResNet-50 was exported to ONNX.
* [ ] `model.onnx` exists in the versioned Triton model directory.
* [ ] `config.pbtxt` exists.
* [ ] Triton loads `resnet50_onnx`.
* [ ] The readiness endpoint responds successfully.
* [ ] Model metadata can be retrieved.
* [ ] The Prometheus metrics endpoint is accessible.
* [ ] At least one real image was sent to Triton.
* [ ] The client returned five ImageNet class indices.
* [ ] The client returned probabilities.
* [ ] Dynamic batching was configured.
* [ ] Multiple model instances were reviewed or tested.
* [ ] GPU utilization was observed if using GPU hardware.
* [ ] TensorRT was evaluated if completing the optional section.
* [ ] Kubernetes deployment was tested if completing the optional section.

---

# Results Worksheet

Record your observations:

| Test             | Configuration      | Result     |
| ---------------- | ------------------ | ---------- |
| Model Load       | ONNX Runtime       | __________ |
| Health Check     | `/v2/health/ready` | __________ |
| Inference        | Single Request     | __________ |
| Dynamic Batching | 4 / 8 / 16         | __________ |
| Model Instances  | 1                  | __________ |
| Model Instances  | 2                  | __________ |
| GPU Utilization  | Baseline           | __________ |
| TensorRT         | Optional           | __________ |
| Kubernetes       | Optional           | __________ |

For performance testing:

| Configuration | Throughput |      p50 |      p95 |      p99 | GPU Memory |
| ------------- | ---------: | -------: | -------: | -------: | ---------: |
| Baseline      |      _____ | _____ ms | _____ ms | _____ ms |      _____ |
| Dynamic Batch |      _____ | _____ ms | _____ ms | _____ ms |      _____ |
| 2 Instances   |      _____ | _____ ms | _____ ms | _____ ms |      _____ |
| TensorRT      |      _____ | _____ ms | _____ ms | _____ ms |      _____ |

---

# Learning Outcomes

After completing this lab, you should be able to:

* Export a pretrained PyTorch model to ONNX.
* Understand Triton's model repository structure.
* Configure model inputs and outputs with `config.pbtxt`.
* Run NVIDIA Triton Inference Server using Docker.
* Submit HTTP inference requests using Triton's model API.
* Preprocess image data for ResNet-50.
* Interpret returned prediction logits.
* Check Triton readiness and model metadata.
* Use Triton's Prometheus metrics endpoint.
* Explain how dynamic batching improves GPU utilization.
* Evaluate the tradeoff between throughput and latency.
* Configure multiple model instances.
* Understand the role of TensorRT in GPU inference optimization.
* Deploy Triton into Kubernetes as part of a broader AI serving platform.

---

# Key Takeaway

**NVIDIA Triton Inference Server separates model execution from client application logic and provides a production-oriented serving layer with standardized APIs, model repositories, dynamic batching, concurrency management, observability, and support for multiple inference backends.**

The core serving path is:

```text
Model
   ↓
ONNX
   ↓
Triton Repository
   ↓
Triton Server
   ↓
HTTP / gRPC
   ↓
Dynamic Batching
   ↓
GPU / CPU
   ↓
Predictions
```

Triton's major advantage is not simply that it runs a model. It provides infrastructure for **efficiently operating model inference under concurrent production workloads**, where throughput, latency, GPU utilization, memory consumption, observability, and cost must be balanced together.
