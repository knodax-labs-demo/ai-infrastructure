# Hands-On Lab: Deploy a PyTorch Model on Minikube

In this lab, you will deploy the containerized PyTorch inference application from the previous chapter to a local Kubernetes cluster using **Minikube**.

You will create a Kubernetes Deployment, expose the application through a Service, scale the number of replicas, inspect application logs, troubleshoot the environment, and clean up the resources when finished.

This exercise demonstrates how containerized AI applications move from **Docker into Kubernetes** and reinforces the relationship between **Pods, Deployments, and Services**.

By the end of the lab, you will have a working AI inference API running on a local Kubernetes cluster.

## Goal

Deploy a containerized PyTorch model API on Kubernetes using Minikube.

## Estimated Time

**60–90 minutes**

## Cost

**Free** when run locally.

## Requirements

Before beginning the lab, make sure the following tools are installed:

* Docker
* Minikube
* `kubectl`

You will also reuse the **FastAPI + PyTorch containerized inference application** created in the previous lab.

Verify the required tools:

```bash
docker --version
minikube version
kubectl version --client
```

---

## Step 1: Start Minikube

Start a local Kubernetes cluster with sufficient CPU and memory:

```bash
minikube start --memory=4096 --cpus=4
```

Verify that the cluster is running:

```bash
kubectl get nodes
```

You should see one Minikube node with a status of:

```text
Ready
```

Minikube creates a lightweight Kubernetes environment on your local system, making it useful for development, experimentation, and learning before moving workloads to larger cloud-based Kubernetes clusters.

---

## Step 2: Build the AI Model Container Image

Reuse the **FastAPI + PyTorch application** created in the previous containerization lab.

Your project should contain files similar to:

```text
pytorch-api/
├── app.py
├── Dockerfile
├── model.pt
├── requirements.txt
└── train_model.py
```

Configure your terminal to use Minikube's Docker environment:

```bash
eval $(minikube docker-env)
```

Build the image:

```bash
docker build -t ai-model:latest .
```

Verify that the image is available:

```bash
docker images | grep ai-model
```

You should see:

```text
ai-model
```

in the output.

> **Why Build Inside Minikube?**
>
> Building the image inside Minikube's Docker environment makes it directly available to the local Kubernetes cluster. You do not need to push the image to an external container registry for this lab.

---

## Step 3: Create the Kubernetes Deployment

Create a file named:

```text
deployment.yaml
```

Add the following configuration:

```yaml
apiVersion: apps/v1
kind: Deployment

metadata:
  name: ai-model-deployment

spec:
  replicas: 2

  selector:
    matchLabels:
      app: ai-model

  template:
    metadata:
      labels:
        app: ai-model

    spec:
      containers:
        - name: ai-model
          image: ai-model:latest
          imagePullPolicy: IfNotPresent

          ports:
            - containerPort: 8000
```

This Deployment tells Kubernetes to maintain **two replicas** of the AI model application.

The important configuration elements are:

| Configuration                   | Purpose                                |
| ------------------------------- | -------------------------------------- |
| `replicas: 2`                   | Runs two copies of the application     |
| `app: ai-model`                 | Identifies the application Pods        |
| `image: ai-model:latest`        | Specifies the Docker image             |
| `imagePullPolicy: IfNotPresent` | Allows use of the locally built image  |
| `containerPort: 8000`           | Documents the FastAPI application port |

Apply the Deployment:

```bash
kubectl apply -f deployment.yaml
```

Check the Deployment:

```bash
kubectl get deployments
```

Check the Pods:

```bash
kubectl get pods
```

The Pods should transition to:

```text
Running
```

once the containers start successfully.

The relationship is:

```text
Deployment
    │
    ├── Pod 1
    │    └── AI Model Container
    │
    └── Pod 2
         └── AI Model Container
```

---

## Step 4: Expose the AI Application with a Service

Pods are ephemeral, and their IP addresses can change. Applications should therefore normally be accessed through a Kubernetes **Service** rather than directly through individual Pod addresses.

Create a file named:

```text
service.yaml
```

Add:

```yaml
apiVersion: v1
kind: Service

metadata:
  name: ai-model-service

spec:
  selector:
    app: ai-model

  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000

  type: NodePort
```

The Service selects Pods with the label:

```text
app: ai-model
```

and forwards traffic from:

```text
Service Port 80
        ↓
Pod Port 8000
        ↓
FastAPI Application
```

The `NodePort` Service type makes the application reachable from outside the Minikube cluster.

Apply the Service:

```bash
kubectl apply -f service.yaml
```

Verify it:

```bash
kubectl get svc
```

You should see:

```text
ai-model-service
```

with an automatically assigned NodePort.

---

## Step 5: Access the Inference API

Use Minikube to retrieve the Service URL:

```bash
minikube service ai-model-service --url
```

The output should resemble:

```text
http://127.0.0.1:<NODE_PORT>
```

For example:

```text
http://127.0.0.1:32768
```

Open the returned address in your browser and append:

```text
/docs
```

For example:

```text
http://127.0.0.1:32768/docs
```

FastAPI's interactive API documentation should appear.

Test the:

```text
POST /predict
```

endpoint by uploading an image.

A successful response should resemble:

```json
{
  "prediction": 3
}
```

The request path is now:

```text
Browser / Client
      ↓
Kubernetes Service
      ↓
Pod
      ↓
FastAPI
      ↓
PyTorch Model
      ↓
Prediction
```

---

## Step 6: Scale the Deployment

One of the advantages of Kubernetes is the ability to change application capacity without rebuilding the container image.

Increase the Deployment from two replicas to four:

```bash
kubectl scale deployment ai-model-deployment --replicas=4
```

Verify the Deployment:

```bash
kubectl get deployment ai-model-deployment
```

Check the Pods:

```bash
kubectl get pods
```

You should now see four Pods running the same AI model API.

The architecture now resembles:

```text
                 Kubernetes Service
                        │
           ┌────────────┼────────────┐
           │            │            │
           ▼            ▼            ▼
         Pod 1        Pod 2        Pod 3
           │            │            │
      AI Model      AI Model      AI Model

                        │
                        ▼
                      Pod 4
                        │
                   AI Model
```

Kubernetes manages these replicas as a single Deployment and continuously works to maintain the requested number of Pods.

---

## Step 7: Inspect Application Logs

List the Pods:

```bash
kubectl get pods
```

Select one Pod name and inspect its logs:

```bash
kubectl logs <POD_NAME>
```

For example:

```bash
kubectl logs ai-model-deployment-xxxxxxxxxx-xxxxx
```

Replace `<POD_NAME>` with the actual Pod name.

The logs may contain:

* FastAPI startup messages
* Uvicorn server messages
* HTTP requests
* Inference requests
* Application errors
* Python exceptions

Log inspection is one of the most fundamental troubleshooting techniques when working with containerized AI workloads on Kubernetes.

### Follow Logs in Real Time

You can also stream logs:

```bash
kubectl logs -f <POD_NAME>
```

Press:

```text
Ctrl+C
```

to stop following the logs.

---

## Step 8: Inspect the Deployment and Service

View detailed information about the Deployment:

```bash
kubectl describe deployment ai-model-deployment
```

This displays information such as:

* Replica configuration
* Pod template
* Labels
* Selectors
* Container image
* Deployment strategy
* Kubernetes events

Inspect the Service:

```bash
kubectl describe service ai-model-service
```

This shows information including:

* Service type
* Selector
* Service port
* Target port
* NodePort
* Endpoints

You can also inspect all major resources:

```bash
kubectl get all
```

These commands are useful when diagnosing problems such as:

```text
Pod does not start
        ↓
Check Pod status and events

Service does not respond
        ↓
Check Service selector and endpoints

Application returns errors
        ↓
Check container logs
```

---

## Optional: Inspect an Individual Pod

Get detailed information about a Pod:

```bash
kubectl describe pod <POD_NAME>
```

This command is particularly useful for identifying issues such as:

* Image loading failures
* Container crashes
* Scheduling problems
* Configuration errors
* Restart events

You can also display additional Pod information:

```bash
kubectl get pods -o wide
```

---

## Step 9: Clean Up the Environment

When the lab is complete, remove the Service:

```bash
kubectl delete -f service.yaml
```

Remove the Deployment:

```bash
kubectl delete -f deployment.yaml
```

Verify that the resources were removed:

```bash
kubectl get all
```

Stop Minikube:

```bash
minikube stop
```

If you no longer need the cluster and want to remove it completely:

```bash
minikube delete
```

> **Cleanup Note**
>
> Cleaning up resources is an important habit. Although Minikube is running locally in this lab, the same discipline becomes critical when using cloud Kubernetes platforms where compute, storage, networking, and load-balancing resources can continue generating charges.

---

## Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Started a Minikube Kubernetes cluster
* [ ] Verified that the Kubernetes node was `Ready`
* [ ] Built the PyTorch inference container image
* [ ] Created a Kubernetes Deployment
* [ ] Started two initial application replicas
* [ ] Created a Kubernetes Service
* [ ] Accessed the FastAPI `/docs` interface
* [ ] Sent an inference request to `/predict`
* [ ] Scaled the Deployment from two Pods to four
* [ ] Inspected application logs
* [ ] Described the Deployment
* [ ] Described the Service
* [ ] Deleted the Kubernetes resources
* [ ] Stopped or deleted Minikube

---

## Learning Outcomes

After completing this lab, you should be able to:

* Deploy a containerized PyTorch inference API on Kubernetes.
* Understand how a **Deployment** manages AI application replicas.
* Understand how Kubernetes **Pods** run containerized AI applications.
* Expose Pods through a Kubernetes **Service**.
* Scale an AI workload using `kubectl scale`.
* Inspect Pods, logs, Deployments, and Services for troubleshooting.
* Understand the basic workflow for moving an AI application from Docker to Kubernetes.

---

## Key Takeaway

**Docker packages the AI application, while Kubernetes deploys, manages, exposes, and scales it. Minikube provides a simple local environment for practicing these Kubernetes concepts before moving to larger production clusters.**
