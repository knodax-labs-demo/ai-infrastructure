# Hands-On Lab: Build a CI/CD Pipeline for AI Model Deployment

In this lab, you will build an end-to-end CI/CD workflow for an AI model using **PyTorch, Flask, Docker, GitHub Actions, and Docker Hub**.

The workflow trains a PyTorch model on the MNIST dataset, validates the generated model artifact, packages the model and inference API in a Docker image, and automatically publishes the image to a container registry. An optional Kubernetes stage demonstrates how a versioned container image can be deployed and rolled back.

The lab illustrates how CI/CD connects model development, validation, packaging, deployment, and recovery within a repeatable MLOps workflow.

```text
Code Change
    ↓
Train
    ↓
Validate
    ↓
Build
    ↓
Tag
    ↓
Push
    ↓
Deploy
    ↓
Verify
    ↓
Roll Back if Required
```

---

## Goal

Build a CI/CD pipeline that:

* Trains a PyTorch model
* Validates the generated model artifact
* Packages the model and inference application in Docker
* Pushes the image to Docker Hub
* Optionally deploys the image to Kubernetes
* Supports rollback when a deployment fails

## Estimated Time

**90–120 minutes**

## Tools

This lab uses:

* Python
* PyTorch
* TorchVision
* Flask
* Docker
* GitHub
* GitHub Actions
* Docker Hub
* Optional Kubernetes and `kubectl`

---

## Prerequisites

Before beginning, make sure you have:

* A GitHub account
* A Docker Hub account
* Git installed
* Docker installed locally if you want to test the image
* Python installed
* Basic familiarity with GitHub repositories
* Optional access to a Kubernetes cluster

Verify Git:

```bash
git --version
```

Verify Docker:

```bash
docker --version
```

Verify Python:

```bash
python --version
```

If you are completing the Kubernetes portion, verify:

```bash
kubectl version --client
```

---

## Step 1: Prepare the Repository

Create a new GitHub repository for the lab.

A suitable project structure is:

```text
ai-model-cicd/
├── .github/
│   └── workflows/
│       └── cicd.yml
├── app.py
├── model.py
├── train.py
├── requirements.txt
└── Dockerfile
```

Each file has a specific role:

| File                         | Purpose                                  |
| ---------------------------- | ---------------------------------------- |
| `model.py`                   | Defines the neural-network architecture  |
| `train.py`                   | Trains the model and creates `model.pth` |
| `app.py`                     | Provides the Flask inference API         |
| `requirements.txt`           | Defines Python dependencies              |
| `Dockerfile`                 | Builds the inference container           |
| `.github/workflows/cicd.yml` | Defines the CI/CD workflow               |

Keeping the model definition in a separate file prevents training and inference code from defining different architectures.

---

## Step 2: Define the Model

Create:

```text
model.py
```

Add:

```python
import torch
import torch.nn as nn


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

        x = torch.relu(
            self.fc1(x)
        )

        x = torch.relu(
            self.fc2(x)
        )

        return self.fc3(x)
```

Each MNIST image contains:

```text
28 × 28 = 784 pixels
```

The network processes the image through two hidden layers and produces ten output values corresponding to the digit classes `0` through `9`.

```text
MNIST Image
  28 × 28
     ↓
Flatten
784 Values
     ↓
Linear
784 → 128
     ↓
ReLU
     ↓
Linear
128 → 64
     ↓
ReLU
     ↓
Linear
64 → 10
     ↓
Digit Prediction
```

---

## Step 3: Train and Save the Model

Create:

```text
train.py
```

Add:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import Net


trainset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

model = Net()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(1):
    running_loss = 0.0

    for images, labels in trainloader:
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

    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {avg_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "model.pth"
)

print("Model trained and saved.")
```

The script:

1. Downloads MNIST.
2. Creates training batches.
3. Initializes the model.
4. Trains for one epoch.
5. Saves the learned parameters to:

```text
model.pth
```

One epoch keeps the CI/CD demonstration relatively short.

> **Production Note**
>
> Real training pipelines typically include validation datasets, multiple epochs, model-quality thresholds, reproducibility controls, experiment tracking, and more extensive testing before a model is approved for deployment.

---

## Step 4: Create the Inference API

Create:

```text
app.py
```

Add:

```python
from flask import Flask, request, jsonify
import torch

from model import Net


app = Flask(__name__)

model = Net()

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location="cpu"
    )
)

model.eval()


@app.route(
    "/health",
    methods=["GET"]
)
def health():
    return jsonify(
        {"status": "healthy"}
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():
    data = request.get_json()

    if not data or "input" not in data:
        return jsonify(
            {"error": "Missing input"}
        ), 400

    tensor = torch.tensor(
        data["input"],
        dtype=torch.float32
    )

    with torch.no_grad():
        output = model(tensor)

        predicted = torch.argmax(
            output,
            dim=1
        )

    return jsonify(
        {
            "prediction":
            predicted.item()
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
```

The application exposes two endpoints:

| Endpoint   | Method | Purpose                                |
| ---------- | ------ | -------------------------------------- |
| `/health`  | GET    | Verify that the application is running |
| `/predict` | POST   | Perform model inference                |

The startup flow is:

```text
Container Starts
      ↓
Flask Application Starts
      ↓
Net() Created
      ↓
model.pth Loaded
      ↓
model.eval()
      ↓
Inference API Ready
```

Using:

```python
torch.no_grad()
```

during inference prevents unnecessary gradient calculations and reduces memory and computational overhead.

---

## Step 5: Define the Python Dependencies

Create:

```text
requirements.txt
```

Add:

```text
torch
torchvision
flask
```

For this introductory lab, the dependency file is intentionally simple.

> **Production Note**
>
> Production projects should generally pin and test dependency versions to improve reproducibility and reduce the risk of unexpected package changes.

---

## Step 6: Create the Dockerfile

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

COPY model.py .
COPY app.py .
COPY model.pth .

EXPOSE 5000

CMD ["python", "app.py"]
```

The image contains:

```text
Container Image
    │
    ├── Python Runtime
    ├── PyTorch
    ├── Flask
    ├── model.py
    ├── app.py
    └── model.pth
```

Notice that:

```text
model.pth
```

must exist before Docker executes the:

```dockerfile
COPY model.pth .
```

instruction.

This is why the pipeline trains the model **before** building the image.

Separating training from runtime also prevents the inference container from retraining the model whenever it starts.

---

## Step 7: Configure Docker Hub Credentials

Create a Docker Hub repository for the container image.

For example:

```text
ai-model
```

In the GitHub repository, configure repository secrets for:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

The workflow uses these secrets to authenticate with Docker Hub.

The relationship is:

```text
GitHub Actions
      ↓
Repository Secrets
      │
      ├── DOCKERHUB_USERNAME
      └── DOCKERHUB_TOKEN
      ↓
Docker Hub Authentication
```

> **Security Note**
>
> Do not place registry passwords, tokens, API keys, or other credentials directly in source files or workflow YAML.

The Docker Hub repository referenced by the workflow must match the repository you created.

---

## Step 8: Create the GitHub Actions Workflow

Create:

```text
.github/workflows/cicd.yml
```

Add:

```yaml
name: ML CI/CD Pipeline

on:
  push:
    branches:
      - main

jobs:
  build-and-push:
    runs-on: ubuntu-latest

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        run: python train.py

      - name: Validate model artifact
        run: test -f model.pth

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build Docker image
        run: |
          docker build \
            -t ${{ secrets.DOCKERHUB_USERNAME }}/ai-model:${{ github.sha }} \
            -t ${{ secrets.DOCKERHUB_USERNAME }}/ai-model:latest \
            .

      - name: Push Docker image
        run: |
          docker push \
            ${{ secrets.DOCKERHUB_USERNAME }}/ai-model:${{ github.sha }}

          docker push \
            ${{ secrets.DOCKERHUB_USERNAME }}/ai-model:latest
```

Whenever code is pushed to:

```text
main
```

the workflow performs:

```text
Git Push
   ↓
Checkout Repository
   ↓
Set Up Python
   ↓
Install Dependencies
   ↓
Train Model
   ↓
Validate model.pth
   ↓
Authenticate to Docker Hub
   ↓
Build Image
   ↓
Create Immutable SHA Tag
   ↓
Create latest Tag
   ↓
Push Both Tags
```

---

## Understanding Image Tags

The workflow creates two tags.

### Commit-Specific Tag

```text
<dockerhub-user>/ai-model:<git-commit-sha>
```

For example:

```text
sk123/ai-model:8a4f932...
```

This creates a traceable relationship between:

```text
Source Code Commit
       ↓
Git SHA
       ↓
Docker Image Tag
       ↓
Deployable Artifact
```

### Latest Tag

```text
<dockerhub-user>/ai-model:latest
```

The `latest` tag provides a convenient reference to the most recently published image.

> **Deployment Best Practice**
>
> Production deployments should generally reference an explicit immutable image version, such as the Git commit SHA, instead of relying only on `latest`.

---

## Step 9: Validate the CI/CD Pipeline

Commit the project files:

```bash
git add .
```

Create a commit:

```bash
git commit -m "Add AI model CI/CD pipeline"
```

Push to GitHub:

```bash
git push origin main
```

Open the repository's:

```text
Actions
```

section.

Select the **ML CI/CD Pipeline** workflow run.

Verify that the following stages complete successfully:

* Repository checkout
* Python setup
* Dependency installation
* Model training
* Model artifact validation
* Docker Hub authentication
* Docker image build
* Docker image push

The core flow is:

```text
Push
  ↓
Train
  ↓
Validate
  ↓
Build
  ↓
Tag
  ↓
Push
```

After the workflow finishes, open the Docker Hub repository.

Confirm that you can see:

```text
latest
```

and a commit-specific SHA tag.

---

## Step 10: Optional — Test the Published Container Locally

Pull the published image:

```bash
docker pull \
  <dockerhub-user>/ai-model:<image-tag>
```

Run it:

```bash
docker run \
  --rm \
  -p 5000:5000 \
  <dockerhub-user>/ai-model:<image-tag>
```

Test the health endpoint:

```bash
curl -s \
  http://localhost:5000/health
```

You should receive a response similar to:

```json
{
  "status": "healthy"
}
```

This verifies that the published image can start successfully.

---

## Step 11: Deploy to Kubernetes

If a Kubernetes environment is available, deploy the published container image.

Assume an existing Deployment named:

```text
ai-app
```

with a container also named:

```text
ai-app
```

Update the image:

```bash
kubectl set image \
  deployment/ai-app \
  ai-app=<dockerhub-user>/ai-model:<image-tag>
```

Follow the rollout:

```bash
kubectl rollout status \
  deployment/ai-app
```

The deployment flow becomes:

```text
GitHub
   ↓
GitHub Actions
   ↓
Docker Hub
   ↓
Versioned Container Image
   ↓
Kubernetes Deployment
   ↓
Inference Pods
```

> **Security Note**
>
> Kubernetes credentials and cluster-access information should be stored and handled securely. Do not embed cluster credentials directly in workflow source files.

Production deployment workflows may also include:

* Approval gates
* Staging environments
* Model validation criteria
* Security scanning
* Integration tests
* Canary releases
* Blue/green deployment
* Progressive rollout policies

---

## Step 12: Test the Deployed Service

After deployment, verify that the application is healthy before sending inference requests.

Check the Deployment:

```bash
kubectl get deployment ai-app
```

Check the Pods:

```bash
kubectl get pods
```

If the service is externally reachable, test:

```text
/health
```

first.

A successful response should resemble:

```json
{
  "status": "healthy"
}
```

Then submit a properly formatted MNIST input to:

```text
/predict
```

The application should return a predicted digit.

The validation sequence should be:

```text
Deployment Completed
       ↓
Pods Ready
       ↓
Health Check
       ↓
Inference Request
       ↓
Valid Prediction
       ↓
Deployment Accepted
```

Production pipelines commonly automate these post-deployment checks before declaring a release successful.

---

## Step 13: Roll Back a Failed Deployment

A CI/CD pipeline should provide a recovery path when a new release fails.

Kubernetes maintains rollout history for Deployments.

Roll back to the previous revision:

```bash
kubectl rollout undo \
  deployment/ai-app
```

Verify the rollback:

```bash
kubectl rollout status \
  deployment/ai-app
```

The recovery workflow is:

```text
New Version Deployed
       ↓
Health / Validation Failure
       ↓
Rollback Triggered
       ↓
Previous Revision Restored
       ↓
Health Verification
       ↓
Service Recovered
```

Immutable image tags make this process easier because each image can be mapped to a specific source-code revision.

For example:

```text
Git Commit A
     ↓
ai-model:a83fd2
     ↓
Deployment Revision 4
```

and:

```text
Git Commit B
     ↓
ai-model:f27cd1
     ↓
Deployment Revision 5
```

If Revision 5 fails, the system can return to the previously known-good version.

---

## Step 14: Understand the Complete CI/CD Flow

The completed workflow demonstrates the relationship among model development, CI, container packaging, and deployment.

```text
Developer
    ↓
Code Change
    ↓
Git Push
    ↓
GitHub Actions
    │
    ├── Install Dependencies
    ├── Train Model
    ├── Validate Artifact
    ├── Build Container
    └── Push Image
           ↓
       Docker Hub
           ↓
      Versioned Image
           ↓
       Kubernetes
           ↓
      Inference API
           ↓
   Health Validation
           ↓
        Success
           │
           └──────────────┐
                          │
                     Failure?
                          │
                          ▼
                       Rollback
```

This is the foundation of a repeatable AI delivery workflow.

---

## CI/CD Pipeline Responsibilities

| Stage        | Purpose                                       |
| ------------ | --------------------------------------------- |
| **Code**     | Capture model and application changes         |
| **Train**    | Produce a model artifact                      |
| **Validate** | Confirm required outputs exist                |
| **Build**    | Package model and API                         |
| **Tag**      | Associate image with a source revision        |
| **Push**     | Publish the deployable artifact               |
| **Deploy**   | Release the image to the runtime environment  |
| **Verify**   | Confirm service health and inference behavior |
| **Rollback** | Recover from a failed release                 |

---

## Production Improvements

The lab intentionally uses a simplified CI/CD workflow.

A production AI pipeline could add:

### Model Validation

Require criteria such as:

* Minimum accuracy
* Maximum validation loss
* Bias or fairness checks
* Regression checks against the previous model

### Software Testing

Add:

* Unit tests
* API tests
* Integration tests
* Container startup tests

### Security Controls

Add:

* Dependency scanning
* Container vulnerability scanning
* Secret scanning
* Image signing
* Provenance metadata

### Deployment Controls

Add:

* Staging environments
* Manual approval
* Canary deployment
* Blue/green deployment
* Automated rollback

### MLOps Integration

Integrate:

* Experiment tracking
* Model registry
* Dataset versioning
* Model lineage
* Monitoring
* Drift detection

A more mature pipeline might therefore become:

```text
Code
  ↓
Train
  ↓
Evaluate
  ↓
Quality Gate
  ↓
Test
  ↓
Security Scan
  ↓
Build
  ↓
Sign
  ↓
Publish
  ↓
Stage
  ↓
Validate
  ↓
Production
  ↓
Monitor
  ↓
Rollback / Retrain
```

---

## Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Created the GitHub repository structure
* [ ] Created the shared PyTorch model definition
* [ ] Created the training script
* [ ] Trained and saved `model.pth`
* [ ] Created the Flask inference API
* [ ] Added a health endpoint
* [ ] Added a prediction endpoint
* [ ] Created `requirements.txt`
* [ ] Created the Dockerfile
* [ ] Created a Docker Hub repository
* [ ] Added Docker Hub credentials as GitHub secrets
* [ ] Created the GitHub Actions workflow
* [ ] Triggered the workflow with a push to `main`
* [ ] Verified model training
* [ ] Verified `model.pth`
* [ ] Built the Docker image
* [ ] Tagged the image using the Git SHA
* [ ] Published the image to Docker Hub
* [ ] Verified the published image
* [ ] Deployed the image to Kubernetes, if completing the optional stage
* [ ] Verified application health after deployment
* [ ] Tested rollback behavior, if using Kubernetes

---

## Learning Outcomes

After completing this lab, you should be able to:

* Build an automated CI/CD pipeline for an AI model.
* Train and validate a PyTorch model within a CI workflow.
* Share a single model definition between training and inference.
* Package a trained model and inference API using Docker.
* Automatically tag and publish container images to a registry.
* Use Git commit identifiers to create traceable deployment artifacts.
* Understand how GitHub Actions connects source changes to model delivery.
* Deploy a versioned AI container to Kubernetes.
* Verify application health after deployment.
* Use Kubernetes rollout history to recover from failed deployments.
* Explain the flow from **Code → Train → Validate → Build → Push → Deploy → Verify → Roll Back**.

---

## Key Takeaway

**AI CI/CD extends traditional software delivery by incorporating model training, validation, packaging, versioning, and deployment into an automated pipeline. GitHub Actions can connect source-code changes to repeatable model builds, while Docker provides a portable deployment artifact and Kubernetes provides controlled rollout and rollback capabilities.**

Using immutable image tags such as the Git commit SHA preserves traceability between the source code, trained model, container image, and deployed version, making AI deployments easier to reproduce, audit, and recover.
