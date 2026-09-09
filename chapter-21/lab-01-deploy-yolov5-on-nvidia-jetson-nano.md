# Hands-On Lab: Deploy YOLOv5 on NVIDIA Jetson Nano

In this lab, you will deploy a **YOLOv5 object detection model** on an **NVIDIA Jetson Nano** and optimize inference using **TensorRT**.

You will first run the standard PyTorch YOLOv5 model to establish a performance baseline. You will then export the model to ONNX, generate a TensorRT engine, run optimized inference, and compare the performance of the PyTorch and TensorRT deployments. You will also monitor Jetson resource utilization during inference and optionally expose the optimized model through a lightweight FastAPI service.

The complete Edge AI workflow is:

```text
PyTorch Model
      ↓
YOLOv5
      ↓
ONNX Export
      ↓
TensorRT Optimization
      ↓
Jetson GPU Inference
      ↓
Camera / Images
      ↓
Object Detection
      ↓
Optional Edge API
```

> **Platform Note**
>
> Jetson Nano is an older Jetson platform with more limited software compatibility than newer Jetson devices. JetPack, CUDA, cuDNN, TensorRT, PyTorch, and TorchVision versions must be compatible with one another. Do not assume that the newest PyPI packages will work correctly on the device.

---

## Lab Objectives

By completing this lab, you will learn how to:

* Prepare an NVIDIA Jetson Nano for AI inference.
* Verify CUDA access from PyTorch.
* Run YOLOv5 object detection on camera input.
* Establish a PyTorch inference baseline.
* Export YOLOv5 to ONNX.
* Build a TensorRT inference engine.
* Run YOLOv5 using TensorRT.
* Compare PyTorch and TensorRT inference performance.
* Monitor Jetson CPU, GPU, memory, temperature, and power behavior.
* Understand Edge AI deployment constraints.
* Optionally expose inference through FastAPI.

---

## Estimated Time

**Approximately 90–150 minutes**

---

## Hardware and Software

This lab requires:

* NVIDIA Jetson Nano Developer Kit
* Preferably 4 GB RAM
* Compatible JetPack installation
* 16–32 GB or larger storage
* USB webcam or supported CSI camera
* Reliable Internet access
* Stable Jetson-compatible power supply
* Python 3
* Git
* PyTorch
* TorchVision
* CUDA
* TensorRT
* YOLOv5

---

# Step 1: Prepare the Jetson Environment

Update the operating system packages:

```bash
sudo apt update
```

Then:

```bash
sudo apt upgrade -y
```

Install the basic development tools:

```bash
sudo apt install -y \
  git \
  python3-pip \
  python3-venv
```

Create a dedicated Python virtual environment:

```bash
python3 -m venv yolov5-env
```

Activate it:

```bash
source yolov5-env/bin/activate
```

Upgrade Python packaging tools:

```bash
python3 -m pip install \
  --upgrade \
  pip \
  setuptools \
  wheel
```

---

# Step 2: Verify the NVIDIA Software Stack

Before installing YOLOv5 dependencies, verify the existing Jetson environment.

The compatibility chain is:

```text
JetPack
   ↓
CUDA
   ↓
cuDNN
   ↓
TensorRT
   ↓
PyTorch
   ↓
TorchVision
```

These components must be mutually compatible.

> **Important**
>
> Do not blindly run:
>
> ```bash
> pip install torch torchvision
> ```
>
> on Jetson Nano. Standard PyPI builds may replace a working NVIDIA-optimized PyTorch installation with an incompatible package.

Install PyTorch and TorchVision versions supported by the JetPack release running on the device.

Then verify PyTorch:

```bash
python3 -c \
"import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

A correctly configured GPU-enabled environment should report:

```text
True
```

for CUDA availability.

You can also inspect the GPU:

```bash
nvidia-smi
```

if the installed Jetson environment provides it.

On Jetson systems, `tegrastats` is often the more useful runtime monitoring command and will be used later in the lab.

---

# Step 3: Clone the YOLOv5 Repository

Clone YOLOv5:

```bash
git clone \
  https://github.com/ultralytics/yolov5.git
```

Move into the repository:

```bash
cd yolov5
```

The project now contains components such as:

```text
yolov5/
├── detect.py
├── export.py
├── models/
├── utils/
├── data/
└── requirements.txt
```

---

# Step 4: Install YOLOv5 Dependencies

Install the required dependencies:

```bash
pip install \
  -r requirements.txt
```

Because Jetson Nano is an ARM-based NVIDIA platform, carefully review dependency changes.

If the installation attempts to replace working Jetson-compatible versions of:

```text
torch
torchvision
```

preserve the versions already validated for your JetPack release.

The important principle is:

```text
Newest Package
      ≠
Best Jetson Package
```

Compatibility matters more than using the latest release.

---

# Step 5: Verify the Camera

Before running inference, confirm that the camera is recognized.

For a USB camera, inspect available video devices:

```bash
ls \
  /dev/video*
```

You may see:

```text
/dev/video0
```

YOLOv5 uses:

```text
--source 0
```

to access the default camera.

If you are using a CSI camera, the camera pipeline may require additional Jetson-specific configuration.

---

# Step 6: Run Baseline YOLOv5 Inference

Run YOLOv5 using the standard PyTorch model:

```bash
python detect.py \
  --source 0 \
  --weights yolov5s.pt \
  --conf-thres 0.4
```

The options mean:

| Option                 | Purpose                          |
| ---------------------- | -------------------------------- |
| `--source 0`           | Use the default camera           |
| `--weights yolov5s.pt` | Load the YOLOv5s model           |
| `--conf-thres 0.4`     | Require 40% detection confidence |

YOLOv5s is useful for Jetson Nano because it is smaller than larger YOLOv5 model variants.

---

## Expected Pipeline

```text
Camera
   ↓
Video Frame
   ↓
YOLOv5 Preprocessing
   ↓
PyTorch Model
   ↓
GPU Inference
   ↓
Non-Maximum Suppression
   ↓
Bounding Boxes
   ↓
Display
```

If everything is configured correctly, recognized objects should appear with bounding boxes and class labels.

---

# Step 7: Record the PyTorch Baseline

Before optimization, record the standard inference performance.

Capture:

* Inference time
* Approximate FPS
* GPU utilization
* Memory utilization
* Temperature
* Power mode
* Input resolution

This establishes a baseline for comparing TensorRT.

Example:

| Metric             | PyTorch Baseline |
| ------------------ | ---------------: |
| Inference time     |    Record result |
| FPS                |    Record result |
| GPU utilization    |    Record result |
| Memory utilization |    Record result |
| Temperature        |    Record result |

Do not use theoretical performance numbers. Measure the actual device.

---

# Step 8: Export YOLOv5 to ONNX

Export:

```bash
python export.py \
  --weights yolov5s.pt \
  --include onnx
```

The export should produce:

```text
yolov5s.onnx
```

The deployment path becomes:

```text
YOLOv5 PyTorch
      ↓
ONNX Export
      ↓
yolov5s.onnx
```

---

## Why ONNX?

ONNX acts as an intermediate model representation between training frameworks and optimized inference runtimes.

Conceptually:

```text
PyTorch
   ↓
ONNX
   ↓
Deployment Runtime
```

This makes the model easier to move into inference systems such as TensorRT.

---

# Step 9: Export YOLOv5 to TensorRT

Generate a TensorRT engine:

```bash
python export.py \
  --weights yolov5s.pt \
  --include engine
```

The result should resemble:

```text
yolov5s.engine
```

The optimization path is:

```text
PyTorch Model
      ↓
ONNX / Export Graph
      ↓
TensorRT Builder
      ↓
Hardware-Specific Optimization
      ↓
TensorRT Engine
```

TensorRT performs inference-oriented optimizations for NVIDIA GPUs.

---

## TensorRT Portability Warning

A TensorRT engine is not a universal model file.

It can depend on:

* TensorRT version
* CUDA environment
* GPU architecture
* Model configuration
* Precision mode
* Build environment

Therefore, for Jetson deployments:

```text
Build / Validate Engine
        ↓
Target Jetson Device
```

is generally safer than building the engine elsewhere and assuming it will work.

---

# Step 10: Run TensorRT Inference

Run:

```bash
python detect.py \
  --source 0 \
  --weights yolov5s.engine \
  --conf-thres 0.4
```

The inference path is now:

```text
Camera
   ↓
Frame
   ↓
Preprocessing
   ↓
TensorRT Engine
   ↓
Jetson GPU
   ↓
Detection Results
```

Observe the inference timing reported by YOLOv5.

Compare responsiveness with the original PyTorch deployment.

---

# Step 11: Perform a Controlled Performance Comparison

For a meaningful comparison, use the same input source.

Run PyTorch:

```bash
python detect.py \
  --source data/images \
  --weights yolov5s.pt \
  --conf-thres 0.4
```

Then TensorRT:

```bash
python detect.py \
  --source data/images \
  --weights yolov5s.engine \
  --conf-thres 0.4
```

Keep the following equivalent:

* Input images
* Image resolution
* Confidence threshold
* Device power mode
* Thermal conditions
* Background workloads

---

## Performance Results

Use a table like:

| Deployment | Inference Time |    FPS | GPU Utilization | Notes     |
| ---------- | -------------: | -----: | --------------: | --------- |
| PyTorch    |         Record | Record |          Record | Baseline  |
| TensorRT   |         Record | Record |          Record | Optimized |

Calculate speedup if useful:

```text
Speedup =
PyTorch Inference Time
──────────────────────
TensorRT Inference Time
```

For example, if measured results were:

```text
PyTorch = 100 ms
TensorRT = 50 ms
```

then:

```text
Speedup = 2×
```

Use your actual measurements rather than assuming a fixed TensorRT acceleration factor.

---

# Step 12: Monitor the Jetson During Inference

Run:

```bash
tegrastats
```

This provides runtime information such as:

* Memory consumption
* CPU utilization
* GPU activity
* Temperature
* Power-related information

Keep `tegrastats` running while executing YOLOv5.

Conceptually:

```text
YOLOv5 Inference
       │
       ├── GPU Utilization
       ├── CPU Utilization
       ├── Memory
       ├── Temperature
       └── Power
              ↓
          tegrastats
```

---

# Step 13: Compare Resource Behavior

Performance should not be evaluated using inference latency alone.

Record:

| Metric          | PyTorch | TensorRT |
| --------------- | ------: | -------: |
| Inference time  |  Record |   Record |
| FPS             |  Record |   Record |
| GPU utilization |  Record |   Record |
| CPU utilization |  Record |   Record |
| Memory          |  Record |   Record |
| Temperature     |  Record |   Record |

An optimized Edge AI deployment should consider:

```text
Latency
+
Throughput
+
Memory
+
Power
+
Temperature
```

rather than only raw speed.

---

# Step 14: Understand Edge AI Constraints

Jetson Nano has substantially fewer resources than a data-center GPU system.

Common constraints include:

* Limited GPU compute
* Limited RAM
* Shared system/GPU memory
* Thermal limits
* Power limits
* ARM software compatibility
* Storage capacity
* Camera bandwidth

Therefore, Edge AI optimization frequently involves tradeoffs among:

```text
Model Accuracy
      ↕
Model Size
      ↕
Latency
      ↕
Power
      ↕
Memory
```

---

# Step 15: Optional — Reduce Model Input Resolution

For constrained environments, reducing input resolution may improve inference performance.

Compare your standard configuration with a smaller inference size where appropriate.

The tradeoff is:

```text
Lower Resolution
      ↓
Less Computation
      ↓
Higher FPS
      ↓
Potentially Lower Detection Accuracy
```

Always evaluate the effect on your target workload.

---

# Step 16: Optional — Expose the Model Through FastAPI

Once TensorRT inference works correctly, the edge device can expose detection through a network API.

Install:

```bash
pip install \
  fastapi \
  uvicorn \
  python-multipart
```

A high-level architecture is:

```text
Client
   ↓
HTTP Request
   ↓
FastAPI
   ↓
Image Upload
   ↓
Preprocessing
   ↓
TensorRT Engine
   ↓
Jetson GPU
   ↓
Postprocessing
   ↓
JSON Response
```

---

## API Design

A simple endpoint could accept:

```text
POST /detect
```

with an uploaded image.

The response might resemble:

```json
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.94,
      "box": [
        102,
        50,
        340,
        420
      ]
    }
  ]
}
```

---

## Important API Design Principle

Do not reload the model on every request.

Avoid:

```text
Request
   ↓
Load TensorRT Engine
   ↓
Inference
```

Instead:

```text
Application Startup
      ↓
Load TensorRT Engine Once
      ↓
Keep Engine in Memory
      ↓
Request 1 ─┐
Request 2 ─┼──► Inference
Request 3 ─┘
```

This dramatically reduces per-request overhead.

---

# Step 17: Validate Object Detection

Test several objects supported by the pretrained model.

Verify:

* Camera input is captured.
* CUDA is available.
* PyTorch inference works.
* Bounding boxes appear correctly.
* ONNX export succeeds.
* TensorRT engine creation succeeds.
* TensorRT inference works.
* TensorRT results remain sensible.
* Performance measurements are recorded.

Test a variety of objects and environmental conditions.

---

# Step 18: Validate Sustained Inference

Run the model continuously for several minutes while monitoring:

```bash
tegrastats
```

Observe whether:

* Temperature continues increasing.
* GPU utilization remains stable.
* Memory usage grows unexpectedly.
* FPS decreases over time.
* Thermal throttling occurs.
* The application remains responsive.

This is important because short benchmark runs may not reveal thermal or memory constraints.

---

# Step 19: Understand the Complete Edge AI Workflow

The complete lab architecture is:

```text
Model Development
      ↓
PyTorch Model
      ↓
Edge Deployment
      ↓
Jetson Nano
      ↓
Baseline Inference
      ↓
ONNX Export
      ↓
TensorRT Optimization
      ↓
Optimized GPU Inference
      ↓
Camera / Sensor Input
      ↓
Detection Results
      ↓
Optional Edge API
```

This demonstrates the transition from a general-purpose ML framework to a hardware-optimized edge deployment.

---

# Step 20: Troubleshooting

## CUDA Is Not Available

Run:

```bash
python3 -c \
"import torch; print(torch.cuda.is_available())"
```

If the result is:

```text
False
```

verify:

* JetPack installation
* PyTorch build
* CUDA compatibility
* TorchVision compatibility

---

## PyTorch Installation Breaks CUDA

If CUDA worked before installing `requirements.txt` but fails afterward, check whether pip replaced the Jetson-specific PyTorch build.

Inspect:

```bash
pip show torch
```

and:

```bash
pip show torchvision
```

Restore versions compatible with your JetPack environment.

---

## Camera Is Not Detected

Check:

```bash
ls \
  /dev/video*
```

If no video device exists, verify:

* Camera connection
* USB compatibility
* CSI configuration
* Device permissions

---

## TensorRT Export Fails

Check compatibility among:

```text
YOLOv5
Python
PyTorch
ONNX
CUDA
TensorRT
JetPack
```

TensorRT export support can depend strongly on the software versions installed on the Jetson.

---

## TensorRT Engine Does Not Load

Remember:

```text
TensorRT Engine
      ≠
Universal Portable Artifact
```

Build or validate the engine on the target Jetson environment.

---

## Inference Is Slow

Check:

* Input resolution
* Power mode
* Thermal throttling
* Background workloads
* Camera pipeline
* Model size
* TensorRT engine
* Available memory

Monitor:

```bash
tegrastats
```

during the test.

---

## Device Becomes Unstable

AI inference can create substantial power demand.

Verify:

* Stable power supply
* Proper cooling
* Adequate airflow
* Appropriate power configuration

Hardware stability is part of Edge AI infrastructure.

---

# Step 21: Clean Up

Stop detection using:

```text
Ctrl+C
```

Deactivate the environment:

```bash
deactivate
```

If disk space is constrained, review generated artifacts such as:

```text
yolov5s.onnx
yolov5s.engine
runs/
```

and remove files no longer needed.

Also clean unnecessary package caches where appropriate.

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Prepared the Jetson Nano.
* [ ] Created the Python virtual environment.
* [ ] Verified JetPack-compatible PyTorch.
* [ ] Confirmed CUDA availability.
* [ ] Cloned YOLOv5.
* [ ] Installed compatible dependencies.
* [ ] Verified camera input.
* [ ] Ran YOLOv5 with PyTorch.
* [ ] Recorded baseline inference performance.
* [ ] Exported YOLOv5 to ONNX.
* [ ] Created a TensorRT engine.
* [ ] Ran TensorRT inference.
* [ ] Compared PyTorch and TensorRT latency.
* [ ] Compared FPS.
* [ ] Monitored resource utilization with `tegrastats`.
* [ ] Observed temperature and memory behavior.
* [ ] Tested sustained inference.
* [ ] Reviewed Edge AI resource constraints.
* [ ] Reviewed the optional FastAPI architecture.
* [ ] Cleaned up temporary resources.

---

# Expected Results

At the end of the lab:

* YOLOv5 should run on the Jetson Nano.
* PyTorch should have access to CUDA.
* Camera or image input should be processed successfully.
* The model should produce object detections.
* YOLOv5 should export to ONNX.
* A TensorRT engine should be generated when supported by the installed environment.
* TensorRT inference should run on the Jetson GPU.
* PyTorch and TensorRT performance should be measurable and comparable.
* `tegrastats` should provide hardware utilization data.
* The optimized model should be suitable for further integration into an Edge AI application or API.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Deploy an object-detection model on NVIDIA Jetson hardware.
* Explain the role of JetPack in the Jetson software stack.
* Verify CUDA-enabled PyTorch.
* Run YOLOv5 camera inference.
* Establish an inference performance baseline.
* Export PyTorch models to ONNX.
* Explain ONNX as an intermediate deployment representation.
* Optimize inference using TensorRT.
* Explain why TensorRT engines are hardware and software dependent.
* Compare PyTorch and TensorRT performance experimentally.
* Monitor Edge AI resource utilization.
* Recognize thermal, memory, and power constraints in edge systems.
* Design a lightweight API around an optimized Edge AI model.

---

# Key Takeaway

**Edge AI deployment is not simply a matter of copying a trained model onto a smaller computer. The model, runtime, accelerator libraries, memory limits, power envelope, and hardware-specific optimization stack must work together.**

The deployment workflow can be summarized as:

```text
General-Purpose ML Model
          ↓
        PyTorch
          ↓
         ONNX
          ↓
       TensorRT
          ↓
NVIDIA Jetson GPU
          ↓
Optimized Edge Inference
```

YOLOv5 provides the object-detection model, ONNX provides a portable intermediate representation, and TensorRT optimizes the model for NVIDIA inference hardware. On Jetson Nano, these optimizations must be evaluated within the constraints of the installed JetPack software stack, available memory, thermal behavior, and power configuration.

The broader Edge AI principle is:

**Develop with a general-purpose ML framework, convert into a deployment-friendly representation, optimize for the target accelerator, and validate performance on the actual edge device.**
