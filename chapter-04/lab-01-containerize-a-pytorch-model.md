# Hands-On Lab: Containerize a PyTorch Model

In this lab, you will package a simple PyTorch model and FastAPI inference application inside a Docker container. You will build a Docker image, run the container, and send an image to the API to generate a prediction.

This lab demonstrates how containers package application code, model artifacts, runtime components, and dependencies into a reproducible deployment unit.

## Lab Objective

Package a PyTorch model and FastAPI inference service into a Docker container, run the application locally, and test the inference endpoint.

## Estimated Time

**60–90 minutes**

## Cost

**Free** when using Docker locally. Minimal cloud charges may apply if you perform the lab on a cloud virtual machine.

---

## Prerequisites

Before beginning the lab, make sure Docker is installed and running.

Verify the installation:

```bash
docker --version
```

You will also need:

* Python
* PyTorch
* Docker
* Internet connectivity for downloading packages and container images

If you plan to experiment with GPU-enabled containers later, you will additionally need:

* A compatible NVIDIA GPU
* NVIDIA drivers
* NVIDIA Container Toolkit

---

## Step 1: Create and Save a PyTorch Model

Create a project directory:

```bash
mkdir pytorch-api
cd pytorch-api
```

Create a file named:

```text
train_model.py
```

Add the following code:

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pretrained ResNet18 weights
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Replace the final layer with a 10-class output layer
model.fc = nn.Linear(model.fc.in_features, 10)

# Save the model for this lab
torch.save(model, "model.pt")

print("Model saved as model.pt")
```

Run the script:

```bash
python train_model.py
```

This creates:

```text
model.pt
```

The model file will be packaged with the inference application.

> **Note**
>
> This lab replaces the final classification layer but does not train it. Therefore, the returned class number is used only to demonstrate the containerized inference workflow and should not be interpreted as a meaningful image classification result.

---

## Step 2: Build the FastAPI Inference Application

Create a file named:

```text
app.py
```

Add the following code:

```python
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as T

app = FastAPI()

# Load the model
model = torch.load(
    "model.pt",
    map_location="cpu",
    weights_only=False
)

model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(x)

    prediction = output.argmax(dim=1).item()

    return {"prediction": prediction}
```

The application loads the PyTorch model when the service starts and exposes a:

```text
/predict
```

endpoint.

The inference workflow is:

```text
Uploaded Image
      ↓
Image Preprocessing
      ↓
PyTorch Model
      ↓
Prediction
      ↓
JSON Response
```

The uploaded image is resized, converted into a tensor, and passed through the model. The API returns the predicted class index as JSON.

---

## Step 3: Define the Application Dependencies

Create a file named:

```text
requirements.txt
```

Add:

```text
fastapi
uvicorn
python-multipart
torch
torchvision
pillow
```

> **Production Note**
>
> Production environments should normally pin dependencies to tested versions to improve reproducibility. Unpinned versions are used here so the lab does not depend on older package combinations that may no longer be readily available.

---

## Step 4: Create the Dockerfile

Create a file named:

```text
Dockerfile
```

Add the following:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model.pt app.py ./

EXPOSE 8000

CMD [
    "uvicorn",
    "app:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]
```

The Dockerfile defines the runtime environment for the application.

It performs the following steps:

1. Starts with a lightweight Python image.
2. Sets `/app` as the working directory.
3. Copies the dependency file.
4. Installs the required Python packages.
5. Copies the model and application code.
6. Exposes port `8000`.
7. Starts the FastAPI application using Uvicorn.

Your project directory should now look like this:

```text
pytorch-api/
├── app.py
├── Dockerfile
├── model.pt
├── requirements.txt
└── train_model.py
```

---

## Step 5: Build the Docker Image

From the project directory, run:

```bash
docker build -t pytorch-api:latest .
```

Verify that the image was created:

```bash
docker images
```

You should see an image named:

```text
pytorch-api
```

with the tag:

```text
latest
```

---

## Step 6: Run the Container

Start the container and map host port `8000` to container port `8000`:

```bash
docker run --rm -p 8000:8000 pytorch-api:latest
```

Docker starts the FastAPI application inside the container.

The API is now available at:

```text
http://localhost:8000
```

The port mapping can be interpreted as:

```text
Local Machine :8000
        ↓
Docker Port Mapping
        ↓
Container :8000
        ↓
FastAPI Application
```

---

## Step 7: Test the Inference API

Open the following address in your browser:

```text
http://localhost:8000/docs
```

FastAPI displays its interactive Swagger API documentation.

To test the API:

1. Expand the **POST `/predict`** endpoint.
2. Select **Try it out**.
3. Upload an image.
4. Choose **Execute**.

A successful response will look similar to:

```json
{
  "prediction": 3
}
```

Remember that the class number is only demonstrating the inference workflow because the replacement classification layer was not trained.

### Test with `curl`

You can also test the endpoint from another terminal:

```bash
curl -X POST \
  http://localhost:8000/predict \
  -F "file=@cat.jpg"
```

Replace:

```text
cat.jpg
```

with the path to an image available on your computer.

At this point, the complete workflow is running inside Docker:

```text
Image
  ↓
FastAPI
  ↓
PyTorch Model
  ↓
Prediction
```

---

## Optional: Explore GPU Containerization

The container created in this lab is intentionally **CPU-based**.

Simply adding:

```bash
--gpus all
```

to the `docker run` command is not sufficient to make this particular image GPU-enabled because its PyTorch installation and base image have not been configured specifically for CUDA.

A GPU-enabled container requires:

* An NVIDIA GPU on the host
* Compatible NVIDIA drivers
* NVIDIA Container Toolkit
* A CUDA-compatible container image
* A CUDA-enabled PyTorch installation

Once a GPU-capable image has been built, it can be launched using:

```bash
docker run --gpus all \
  --rm \
  -p 8000:8000 \
  pytorch-api-gpu:latest
```

The relationship is:

```text
Host NVIDIA GPU
      ↓
NVIDIA Driver
      ↓
NVIDIA Container Toolkit
      ↓
Docker Container
      ↓
CUDA-Compatible PyTorch
      ↓
GPU-Accelerated Inference
```

> **Important**
>
> Docker makes the GPU available to the container, but the software inside the container must also support GPU acceleration.

---

## Cleanup

If the container is running in the foreground, stop it using:

```text
Ctrl+C
```

Because the container was started with:

```text
--rm
```

Docker automatically removes the container after it stops.

The Docker image remains available for future use.

Verify available images:

```bash
docker images
```

When you no longer need the image, remove it:

```bash
docker rmi pytorch-api:latest
```

---

## Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Created and saved a PyTorch model
* [ ] Created a FastAPI inference application
* [ ] Defined the Python dependencies
* [ ] Created a Dockerfile
* [ ] Built the Docker image
* [ ] Started the Docker container
* [ ] Accessed FastAPI documentation
* [ ] Uploaded an image to `/predict`
* [ ] Received a JSON prediction
* [ ] Tested the API using `curl`
* [ ] Understood the additional requirements for GPU-enabled containers
* [ ] Stopped and cleaned up the container

---

## Learning Outcomes

After completing this lab, you should be able to:

* Package a PyTorch model and inference API into a Docker image.
* Understand the relationship between a **Dockerfile, image, container, and AI application**.
* Build and run a reproducible containerized inference service.
* Expose a containerized AI application through port mapping.
* Test an inference API using FastAPI documentation or `curl`.
* Understand the additional requirements for GPU-enabled AI containers.

---

## Key Takeaway

**Containerization packages the model, application code, runtime, and dependencies into a consistent deployment unit, making AI applications easier to reproduce and deploy across environments.**
