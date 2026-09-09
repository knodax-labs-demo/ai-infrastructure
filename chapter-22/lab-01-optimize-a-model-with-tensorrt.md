# Hands-On Lab: Optimize a Model with TensorRT

In this lab, you will take a pretrained **ResNet-18 image-classification model**, export it from **PyTorch to ONNX**, optimize it with **NVIDIA TensorRT**, and benchmark inference performance.

You will create an **FP16 TensorRT engine**, explore the requirements for **INT8 optimization**, and compare latency, throughput, and model quality against a baseline implementation. The lab is suitable for NVIDIA Jetson platforms and other NVIDIA GPU systems with compatible CUDA and TensorRT installations.

The complete workflow is:

```text
PyTorch Model
      ↓
ONNX Export
      ↓
TensorRT Optimization
      ↓
FP16 / INT8 Engine
      ↓
Benchmark
      ↓
Validate Accuracy
      ↓
Deploy
```

---

## Lab Objective

By completing this lab, you will learn how to:

* Export a pretrained PyTorch model to ONNX.
* Use ONNX as an intermediate deployment representation.
* Establish a measurable baseline.
* Build a TensorRT FP16 engine.
* Benchmark TensorRT inference.
* Understand INT8 quantization and calibration requirements.
* Compare latency and throughput across precision modes.
* Evaluate performance together with model accuracy.
* Calculate relative inference speedup.
* Understand TensorRT engine portability constraints.
* Explore Torch-TensorRT as an alternative optimization workflow.

---

## Estimated Time

**Approximately 90–120 minutes**

---

## Tools

This lab uses:

* Python
* PyTorch
* TorchVision
* ONNX
* ONNX Runtime
* NVIDIA CUDA
* NVIDIA TensorRT
* `trtexec`
* NumPy
* Optional Torch-TensorRT

---

# Step 1: Verify the Environment

You need either:

* A supported NVIDIA Jetson platform, or
* Another NVIDIA GPU system with compatible CUDA and TensorRT installations

Install the Python packages required for the basic workflow:

```bash
pip install \
  torch \
  torchvision \
  onnx \
  onnxruntime \
  numpy
```

> **Jetson Note**
>
> On Jetson platforms, install PyTorch and TorchVision versions compatible with the installed JetPack release rather than blindly replacing NVIDIA-provided builds with the newest PyPI packages.

Verify TensorRT's command-line utility:

```bash
trtexec \
  --help
```

If the command is not found, search the TensorRT installation directories. On some systems, `trtexec` is installed outside the default shell `PATH`.

---

# Step 2: Understand the Optimization Pipeline

The lab uses this progression:

```text
PyTorch
   ↓
ONNX
   ↓
TensorRT
   ├── FP16
   └── INT8
        ↓
Benchmark
        ↓
Accuracy Validation
```

Each stage serves a different purpose:

| Stage         | Purpose                              |
| ------------- | ------------------------------------ |
| PyTorch       | Model development and baseline       |
| ONNX          | Portable intermediate representation |
| TensorRT FP16 | Reduced-precision GPU inference      |
| TensorRT INT8 | More aggressive quantized inference  |
| Benchmarking  | Measure real performance             |
| Validation    | Confirm acceptable model quality     |

---

# Step 3: Export ResNet-18 to ONNX

Create:

```text
lab-07-export-resnet18.py
```

Add:

```python
import torch
from torchvision.models import ResNet18_Weights, resnet18


# Load pretrained ResNet-18
weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)

model.eval()


# Create representative input
dummy_input = torch.randn(
    1,
    3,
    224,
    224
)


# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17
)

print(
    "ONNX model saved: resnet18.onnx"
)
```

Run:

```bash
python3 \
  lab-07-export-resnet18.py
```

Expected artifact:

```text
resnet18.onnx
```

---

## Why ONNX?

ONNX provides an interoperable representation between model-development frameworks and optimized deployment runtimes.

Conceptually:

```text
PyTorch Model
      ↓
ONNX Graph
      ↓
TensorRT Import
      ↓
Optimized Engine
```

The ONNX file becomes the portable intermediate artifact used to create TensorRT engines.

---

# Step 4: Verify the ONNX Model

A useful additional validation step is to check that the exported graph is structurally valid.

Run:

```bash
python3 - <<'PY'
import onnx

model = onnx.load(
    "resnet18.onnx"
)

onnx.checker.check_model(
    model
)

print(
    "ONNX model validation passed."
)
PY
```

This helps detect export problems before attempting TensorRT conversion.

---

# Step 5: Establish an ONNX Runtime Baseline

Create:

```text
lab-07-benchmark-onnx.py
```

Add:

```python
import time

import numpy as np
import onnxruntime as ort


session = ort.InferenceSession(
    "resnet18.onnx",
    providers=[
        "CPUExecutionProvider"
    ]
)

input_name = (
    session
    .get_inputs()[0]
    .name
)

x = np.random.rand(
    1,
    3,
    224,
    224
).astype(
    np.float32
)


# Warm-up
for _ in range(10):
    session.run(
        None,
        {
            input_name: x
        }
    )


# Benchmark
iterations = 100

start = time.perf_counter()

for _ in range(iterations):
    session.run(
        None,
        {
            input_name: x
        }
    )

elapsed = (
    time.perf_counter()
    - start
)

average_latency = (
    elapsed
    / iterations
) * 1000

throughput = (
    iterations
    / elapsed
)

print(
    f"Average latency: "
    f"{average_latency:.2f} ms"
)

print(
    f"Throughput: "
    f"{throughput:.2f} inferences/sec"
)
```

Run:

```bash
python3 \
  lab-07-benchmark-onnx.py
```

Record the results.

---

## Baseline Metrics

Capture at least:

| Metric             |        Result |
| ------------------ | ------------: |
| Average latency    |        Record |
| Throughput         |        Record |
| Execution provider |           CPU |
| Input shape        | `1×3×224×224` |

Actual results vary with:

* CPU
* GPU
* Jetson model
* Power mode
* Thermal conditions
* Software versions
* Runtime configuration

---

## Important Benchmarking Note

The example baseline uses:

```text
ONNX Runtime
      +
CPUExecutionProvider
```

while TensorRT runs on the NVIDIA GPU.

Therefore, this comparison measures the improvement between **different execution environments**, not only the effect of TensorRT graph optimization.

For stricter benchmarking, compare equivalent GPU-backed execution paths where possible.

---

# Step 6: Build the TensorRT FP16 Engine

Run:

```bash
trtexec \
  --onnx=resnet18.onnx \
  --saveEngine=resnet18_fp16.engine \
  --fp16
```

The resulting artifact should be:

```text
resnet18_fp16.engine
```

The optimization path is:

```text
ONNX
  ↓
TensorRT Builder
  ↓
Graph Optimization
  ↓
FP16 Precision
  ↓
GPU-Specific Engine
```

TensorRT may fuse operations, select optimized kernels, optimize memory usage, and use lower precision where supported.

---

## What FP16 Does

FP16 uses 16-bit floating-point operations instead of FP32 where supported.

Conceptually:

```text
FP32
 ↓
More Numerical Precision
More Memory
More Compute

FP16
 ↓
Lower Memory Requirements
Potentially Higher Throughput
Potentially Lower Latency
```

The actual benefit depends on the GPU's FP16 capabilities.

---

# Step 7: Benchmark the FP16 Engine

Run:

```bash
trtexec \
  --loadEngine=resnet18_fp16.engine \
  --shapes=input:1x3x224x224
```

Review the TensorRT output for metrics such as:

* Latency
* Throughput
* Host latency
* GPU compute time
* Enqueue time

Record the values relevant to your system.

---

## Example Results Table

| Mode                  | Avg. Latency | Throughput | Precision |
| --------------------- | -----------: | ---------: | --------- |
| ONNX Runtime baseline |       Record |     Record | FP32      |
| TensorRT FP16         |       Record |     Record | FP16      |

Do not use assumed performance values.

Measure them on the actual target hardware.

---

# Step 8: Calculate FP16 Speedup

Use:

```text
Speedup =
Baseline Latency
─────────────────
FP16 Latency
```

For example:

```text
Baseline = 40 ms
FP16     = 20 ms
```

Then:

```text
Speedup =
40 / 20
= 2×
```

This simple ratio makes optimization improvements easier to interpret.

---

# Step 9: Understand INT8 Optimization

INT8 reduces numerical representation from floating point to 8-bit integer values.

Conceptually:

```text
FP32
 ↓
FP16
 ↓
INT8
 ↓
Lower Precision
Lower Memory
Potentially Higher Throughput
```

However, INT8 introduces an additional requirement:

```text
Reduced Precision
       ↓
Quantization
       ↓
Need to Preserve Accuracy
```

This is why INT8 optimization requires more careful validation than simply enabling FP16.

---

# Step 10: Understand INT8 Calibration

One common workflow is **post-training quantization with calibration**.

The process is:

```text
FP32 Model
    ↓
Representative Calibration Data
    ↓
Observe Activation Ranges
    ↓
Determine Quantization Parameters
    ↓
Build INT8 Engine
    ↓
Validate Accuracy
```

Representative calibration data should resemble actual production inputs.

For an ImageNet-style classifier, calibration samples should represent realistic image distributions rather than random noise.

---

## Why Calibration Matters

INT8 must map a large floating-point numerical range into a much smaller integer range.

Poor calibration can cause:

```text
Bad Quantization Ranges
        ↓
Information Loss
        ↓
Accuracy Degradation
```

Representative data helps TensorRT select more appropriate quantization ranges.

---

# Step 11: INT8 Optimization Options

A mature TensorRT INT8 workflow generally uses one of two approaches:

### Post-Training Quantization

```text
Trained Model
    ↓
Representative Calibration Dataset
    ↓
Quantization
    ↓
INT8 Engine
```

### Quantization-Aware Training

```text
Training
    ↓
Simulate Quantization Effects
    ↓
Model Learns to Tolerate INT8
    ↓
Quantized Deployment
```

QAT can preserve model quality better for models that are sensitive to post-training quantization.

---

# Step 12: Build a Valid INT8 Engine

The exact TensorRT INT8 workflow depends on:

* TensorRT version
* Quantization API
* Calibration strategy
* Explicit vs. implicit quantization
* Model architecture

Do not assume that an arbitrary calibration cache or historical command is appropriate for every TensorRT release.

Use the INT8 or explicit-quantization workflow supported by your installed version.

The conceptual goal is to produce:

```text
resnet18_int8.engine
```

from:

```text
resnet18.onnx
```

using representative calibration or another supported quantization approach.

---

# Step 13: Benchmark INT8

After producing a valid engine, run:

```bash
trtexec \
  --loadEngine=resnet18_int8.engine \
  --shapes=input:1x3x224x224
```

Record:

* Average latency
* Throughput
* GPU compute time
* Memory behavior

Use the same input shape and equivalent benchmark settings used for FP16.

---

# Step 14: Compare FP32, FP16, and INT8

Use:

| Mode          | Average Latency | Throughput | Accuracy | Relative Performance |
| ------------- | --------------: | ---------: | -------: | -------------------: |
| Baseline      |          Record |     Record | Baseline |                 1.0× |
| TensorRT FP16 |          Record |     Record |  Measure |            Calculate |
| TensorRT INT8 |          Record |     Record |  Measure |            Calculate |

Calculate:

```text
FP16 Speedup =
Baseline Latency / FP16 Latency
```

and:

```text
INT8 Speedup =
Baseline Latency / INT8 Latency
```

---

# Step 15: Evaluate Accuracy

Inference speed alone is not enough.

For FP16 and especially INT8, evaluate the optimized model against a representative validation dataset.

For image classification, measure metrics such as:

* Top-1 accuracy
* Top-5 accuracy

The optimization decision should consider:

```text
Latency
   +
Throughput
   +
Accuracy
   +
Memory
   +
Power
   ↓
Deployment Decision
```

A faster model is not necessarily better if its predictive quality degrades beyond acceptable limits.

---

# Step 16: Interpret the Precision Tradeoff

A typical conceptual relationship is:

```text
FP32
├── Highest Numerical Precision
├── Larger Memory Footprint
└── Usually Lower Accelerator Efficiency

FP16
├── Reduced Precision
├── Lower Memory Use
└── Strong GPU Acceleration

INT8
├── Much Lower Precision
├── Lower Memory Use
├── Potentially Higher Throughput
└── Requires Quantization Validation
```

Actual results vary significantly by hardware and model.

---

# Step 17: Understand TensorRT Engine Portability

TensorRT engine files should be treated as deployment artifacts tied to a compatible target environment.

An engine can depend on:

* GPU architecture
* TensorRT version
* CUDA environment
* Precision mode
* Optimization profiles
* Build configuration

Therefore:

```text
TensorRT Engine
      ≠
Universal Portable Model
```

The safer pattern is:

```text
Portable ONNX
     ↓
Target Environment
     ↓
Build / Validate TensorRT Engine
```

This is particularly important for Jetson deployments.

---

# Step 18: Optional — Explore Torch-TensorRT

Torch-TensorRT allows developers to access TensorRT acceleration while staying closer to PyTorch.

Create:

```text
lab-07-torch-tensorrt.py
```

A simplified example is:

```python
import torch
import torch_tensorrt

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)

model = (
    model
    .eval()
    .cuda()
)


trt_model = torch_tensorrt.compile(
    model,
    inputs=[
        torch_tensorrt.Input(
            (
                1,
                3,
                224,
                224
            )
        )
    ],
    enabled_precisions={
        torch.float16
    }
)

print(
    "Torch-TensorRT compilation complete."
)
```

> **Version Note**
>
> Torch-TensorRT compilation and serialization APIs can change between releases. Use the workflow supported by the installed Torch-TensorRT version.

---

# Step 19: Understand Torch-TensorRT vs. ONNX/TensorRT

The traditional path is:

```text
PyTorch
   ↓
ONNX
   ↓
TensorRT
```

Torch-TensorRT provides another path:

```text
PyTorch
   ↓
Torch-TensorRT
   ↓
TensorRT-Accelerated Execution
```

The first approach emphasizes deployment interoperability.

The second keeps developers closer to PyTorch workflows.

---

# Step 20: Monitor Hardware During Benchmarking

For NVIDIA GPU systems, monitor GPU behavior while running TensorRT.

Depending on the platform, useful tools include:

```bash
nvidia-smi
```

For Jetson systems:

```bash
tegrastats
```

Observe:

* GPU utilization
* Memory
* Temperature
* Power behavior
* CPU activity

This helps connect model-level performance with hardware-level behavior.

---

# Step 21: Improve Benchmark Quality

For meaningful comparisons, keep the following consistent:

* Batch size
* Input shape
* Warm-up iterations
* Number of timed iterations
* Power mode
* GPU clock state
* Thermal state
* Precision
* Input preprocessing

A good benchmark process is:

```text
Warm-Up
   ↓
Stabilize GPU
   ↓
Run Many Iterations
   ↓
Record Latency Distribution
   ↓
Calculate Throughput
   ↓
Repeat
```

Avoid drawing conclusions from a single inference.

---

# Step 22: Consider Batch Size

So far, the lab uses:

```text
Batch = 1
```

This is appropriate for latency-sensitive inference.

Higher batch sizes may increase throughput:

```text
Batch 1
   ↓
Lower Request Latency

Batch 8
   ↓
Potentially Higher GPU Utilization
   ↓
Potentially Higher Throughput
```

The correct choice depends on the workload.

---

# Step 23: Understand Latency vs. Throughput

These metrics measure different things.

### Latency

```text
Time per inference
```

Useful for:

* Interactive applications
* Real-time inference
* Edge AI

### Throughput

```text
Inferences per second
```

Useful for:

* Batch processing
* High-volume serving

A production optimization should align with application requirements.

---

# Step 24: Production Optimization Workflow

A mature workflow may look like:

```text
Trained PyTorch Model
        ↓
Validate Accuracy
        ↓
Export ONNX
        ↓
Validate ONNX
        ↓
Build FP16 Engine
        ↓
Benchmark
        ↓
Validate Accuracy
        ↓
Build INT8 Engine
        ↓
Calibrate / Quantize
        ↓
Benchmark
        ↓
Validate Accuracy
        ↓
Choose Deployment Precision
```

The goal is not simply to choose the lowest precision.

The goal is to find the best tradeoff for the workload.

---

# Step 25: Clean Up

Remove generated artifacts if they are no longer needed:

```bash
rm -f \
  resnet18.onnx \
  resnet18_fp16.engine \
  resnet18_int8.engine
```

These are generated deployment artifacts and do not necessarily need to be committed to Git.

---

# Recommended Folder Structure

For a standalone lab:

```text
lab_tensorrt_opt/
├── export_resnet18.py
├── benchmark_onnx.py
├── torch_tensorrt_example.py
├── README.md
├── resnet18.onnx
├── resnet18_fp16.engine
└── resnet18_int8.engine
```

For your book companion repository:

```text
chapter-22/
├── lab-07-optimize-a-model-with-tensorrt.md
├── lab-07-export-resnet18.py
├── lab-07-benchmark-onnx.py
└── lab-07-torch-tensorrt.py
```

I recommend **not committing**:

```text
*.onnx
*.engine
```

unless you intentionally want to distribute generated model artifacts.

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Verified CUDA and TensorRT availability.
* [ ] Verified `trtexec`.
* [ ] Loaded pretrained ResNet-18.
* [ ] Exported the model to ONNX.
* [ ] Validated the ONNX model.
* [ ] Established a baseline.
* [ ] Recorded baseline latency.
* [ ] Recorded baseline throughput.
* [ ] Built a TensorRT FP16 engine.
* [ ] Benchmarked FP16 inference.
* [ ] Calculated FP16 speedup.
* [ ] Reviewed INT8 quantization requirements.
* [ ] Identified representative calibration data.
* [ ] Built or reviewed the workflow for a valid INT8 engine.
* [ ] Benchmarked INT8 where supported.
* [ ] Compared latency across modes.
* [ ] Compared throughput across modes.
* [ ] Evaluated model accuracy.
* [ ] Reviewed TensorRT engine portability constraints.
* [ ] Explored Torch-TensorRT.
* [ ] Monitored hardware behavior.
* [ ] Cleaned generated artifacts.

---

# Expected Results

At the end of the lab:

* A pretrained ResNet-18 model should be exported to ONNX.
* The ONNX graph should be usable as an intermediate deployment representation.
* A baseline latency should be measured.
* A TensorRT FP16 engine should be created.
* FP16 inference performance should be measured.
* INT8 requirements should be understood and, where supported, tested.
* Model accuracy should be evaluated alongside performance.
* Relative speedup should be calculated from measured results.
* Hardware characteristics should be considered when interpreting benchmark results.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Export PyTorch models to ONNX.
* Explain why ONNX is useful for deployment.
* Build TensorRT engines.
* Use FP16 for inference optimization.
* Explain INT8 quantization and calibration.
* Understand the role of representative calibration data.
* Benchmark TensorRT inference using `trtexec`.
* Calculate relative inference speedup.
* Distinguish latency from throughput.
* Evaluate performance and model quality together.
* Explain why TensorRT engines are environment-specific deployment artifacts.
* Understand Torch-TensorRT as an alternative PyTorch-oriented workflow.
* Design more rigorous GPU inference benchmarks.

---

# Key Takeaway

**Inference optimization should be measured on the actual target hardware and evaluated as a tradeoff among latency, throughput, accuracy, memory, and hardware efficiency.**

The optimization workflow is:

```text
PyTorch
   ↓
ONNX
   ↓
TensorRT
   ├── FP16
   └── INT8
        ↓
Benchmark
        ↓
Accuracy Validation
        ↓
Deployment Decision
```

FP16 often provides a practical balance between numerical precision and accelerator performance, while INT8 can reduce computational and memory requirements further but requires stronger quantization and accuracy validation.

Most importantly:

```text
Optimization
     ≠
Assumed Speedup

Optimization
     =
Measured Improvement
on Target Hardware
```

This workflow provides a practical foundation for deploying optimized computer-vision and other neural-network workloads on NVIDIA Jetson and data-center GPU systems.
