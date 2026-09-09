# Hands-On Lab: Deploy MLflow on Kubernetes

In this lab, you will deploy an **MLflow Tracking Server on Kubernetes** using PostgreSQL as the backend metadata store and MinIO as an S3-compatible artifact store.

You will configure persistent storage, Kubernetes Secrets, internal Services, optional Ingress access, and an end-to-end MLflow experiment to verify that parameters and metrics are stored in PostgreSQL while artifacts are stored separately in MinIO.

The architecture demonstrates how Kubernetes can provide a persistent infrastructure foundation for experiment tracking in an MLOps environment.

---

## Goal

Deploy an MLflow Tracking Server on Kubernetes with:

* PostgreSQL as the backend metadata store
* MinIO as the artifact store
* Persistent Kubernetes storage
* Kubernetes Secrets
* Internal Service discovery
* Optional Ingress access
* End-to-end experiment validation

## Estimated Time

**90–120 minutes**

## Tools

This lab uses:

* Kubernetes cluster
* `kubectl`
* Helm
* MLflow
* PostgreSQL
* MinIO
* Docker if a custom MLflow image is required

You can use:

* Minikube
* kind
* Amazon EKS
* Google Kubernetes Engine
* Azure Kubernetes Service
* Another appropriately configured Kubernetes environment

## Namespace

This lab uses:

```text
mlops
```

---

## Architecture Overview

By the end of the lab, you will have the following architecture:

```text
ML Client
    │
    ▼
MLflow Tracking Server
    │
    ├──────────────► PostgreSQL
    │                 Metadata
    │
    └──────────────► MinIO
                      Artifacts
```

The responsibilities are separated:

| Component              | Purpose                                                |
| ---------------------- | ------------------------------------------------------ |
| MLflow Tracking Server | Receives experiment tracking requests                  |
| PostgreSQL             | Stores runs, parameters, metrics, and tags             |
| MinIO                  | Stores models, plots, checkpoints, and other artifacts |
| Kubernetes Service     | Provides stable internal connectivity                  |
| Kubernetes Secret      | Supplies credentials and connection information        |
| Ingress                | Optionally provides external HTTP access               |
| Persistent Volumes     | Preserve database and object-storage data              |

---

## Lab Files

Use the following files:

```text
chapter-12/
├── lab-08-deploy-mlflow-on-kubernetes.md
├── lab-08-mkbucket.yaml
├── lab-08-mlflow-deploy.yaml
└── lab-08-mlflow-ingress.yaml
```

---

## Step 1: Create the Namespace and Verify Storage

Create a dedicated namespace for the MLOps components:

```bash
kubectl create namespace mlops
```

Verify that the namespace exists:

```bash
kubectl get namespace mlops
```

Next, inspect the available Kubernetes StorageClasses:

```bash
kubectl get storageclass
```

PostgreSQL and MinIO require persistent storage, so verify that the cluster has an appropriate StorageClass before continuing.

The storage relationship is:

```text
PostgreSQL Pod
     │
     ▼
PersistentVolumeClaim
     │
     ▼
Persistent Storage

MinIO Pod
     │
     ▼
PersistentVolumeClaim
     │
     ▼
Persistent Storage
```

> **Important**
>
> If the cluster does not have a default StorageClass, configure an appropriate StorageClass for your Kubernetes environment before installing PostgreSQL or MinIO.

---

## Step 2: Install MinIO as the Artifact Store

MLflow uses an artifact store for files generated during experiments.

Examples include:

* Trained models
* Model checkpoints
* Plots
* Evaluation reports
* Images
* Configuration files
* Other experiment outputs

In this lab, MinIO provides an S3-compatible object store inside Kubernetes.

Add the MinIO Helm repository:

```bash
helm repo add minio https://charts.min.io/
```

Update the local Helm repository information:

```bash
helm repo update
```

Install MinIO:

```bash
helm install minio minio/minio \
  --namespace mlops \
  --set rootUser=admin \
  --set rootPassword=admin12345 \
  --set resources.requests.memory=256Mi \
  --set mode=standalone \
  --set replicas=1
```

Verify the MinIO resources:

```bash
kubectl -n mlops get pods
```

and:

```bash
kubectl -n mlops get svc
```

---

## Access the MinIO Console

For temporary local access, forward the MinIO Service ports:

```bash
kubectl -n mlops port-forward \
  svc/minio \
  9000:9000 \
  9001:9001
```

Open:

```text
http://localhost:9001
```

Sign in using:

```text
Username: admin
Password: admin12345
```

Create a bucket named:

```text
mlflow-artifacts
```

The resulting storage path is conceptually:

```text
MLflow
   ↓
S3-Compatible API
   ↓
MinIO
   ↓
mlflow-artifacts Bucket
   ↓
Experiment Artifacts
```

> **Lab Security Note**
>
> The credentials in this lab are intentionally simple for a learning environment. Production systems should use Kubernetes Secrets or an external secrets-management platform, credential rotation, TLS, and appropriate access controls.

---

## Step 3: Install PostgreSQL as the Backend Store

MLflow uses its backend store for structured experiment metadata such as:

* Experiments
* Runs
* Parameters
* Metrics
* Tags
* Run status
* Timestamps

Add the Bitnami Helm repository:

```bash
helm repo add bitnami \
  https://charts.bitnami.com/bitnami
```

Update Helm repositories:

```bash
helm repo update
```

Install PostgreSQL:

```bash
helm install pg bitnami/postgresql \
  --namespace mlops \
  --set global.postgresql.auth.postgresPassword=pgpass \
  --set global.postgresql.auth.username=mlflow \
  --set global.postgresql.auth.password=mlflowpass \
  --set global.postgresql.auth.database=mlflowdb \
  --set primary.persistence.size=5Gi
```

Verify that PostgreSQL is running:

```bash
kubectl -n mlops get pods
```

Inspect its Service:

```bash
kubectl -n mlops get svc
```

The MLflow server will connect to PostgreSQL through the Kubernetes Service DNS name rather than through an external IP.

The internal path is:

```text
MLflow Pod
    │
    ▼
pg-postgresql.mlops.svc.cluster.local
    │
    ▼
PostgreSQL
    │
    ▼
mlflowdb
```

---

## Step 4: Create MLflow Secrets

Create a Kubernetes Secret containing:

* PostgreSQL connection string
* Artifact-store URI
* MinIO access key
* MinIO secret key

Run:

```bash
kubectl -n mlops create secret generic mlflow-secrets \
  --from-literal=BACKEND_URI="postgresql://mlflow:mlflowpass@pg-postgresql.mlops.svc.cluster.local:5432/mlflowdb" \
  --from-literal=ARTIFACT_URI="s3://mlflow-artifacts" \
  --from-literal=AWS_ACCESS_KEY_ID="admin" \
  --from-literal=AWS_SECRET_ACCESS_KEY="admin12345"
```

Verify the Secret:

```bash
kubectl -n mlops get secret mlflow-secrets
```

The Secret keeps these configuration values separate from the MLflow Deployment manifest.

The MinIO endpoint itself will be configured separately so that MLflow's S3 client sends artifact requests to MinIO instead of Amazon S3.

---

## Step 5: Create the MinIO Artifact Bucket with Kubernetes

If you already created the `mlflow-artifacts` bucket through the MinIO web console, this step is optional.

To automate bucket creation, create:

```text
lab-08-mkbucket.yaml
```

Add:

```yaml
apiVersion: batch/v1
kind: Job

metadata:
  name: mkbucket
  namespace: mlops

spec:
  template:
    spec:
      restartPolicy: Never

      containers:
        - name: mc
          image: minio/mc:latest

          env:
            - name: MC_HOST_minio
              value: "http://admin:admin12345@minio.mlops.svc.cluster.local:9000"

          command:
            - "sh"
            - "-c"

          args:
            - |
              mc ls minio || true
              mc mb -p minio/mlflow-artifacts || true
```

Apply the Job:

```bash
kubectl apply -f lab-08-mkbucket.yaml
```

Inspect its output:

```bash
kubectl -n mlops logs job/mkbucket
```

The Job creates:

```text
mlflow-artifacts
```

if the bucket does not already exist.

---

## Step 6: Deploy the MLflow Tracking Server

Create:

```text
lab-08-mlflow-deploy.yaml
```

Add:

```yaml
apiVersion: apps/v1
kind: Deployment

metadata:
  name: mlflow
  namespace: mlops

spec:
  replicas: 1

  selector:
    matchLabels:
      app: mlflow

  template:
    metadata:
      labels:
        app: mlflow

    spec:
      containers:
        - name: mlflow
          image: ghcr.io/mlflow/mlflow:v2.14.1
          imagePullPolicy: IfNotPresent

          ports:
            - containerPort: 5000

          envFrom:
            - secretRef:
                name: mlflow-secrets

          env:
            - name: MLFLOW_S3_ENDPOINT_URL
              value: "http://minio.mlops.svc.cluster.local:9000"

            - name: AWS_DEFAULT_REGION
              value: "us-east-1"

          command:
            - "/bin/sh"
            - "-c"

          args:
            - >
              mlflow server
              --host 0.0.0.0
              --port 5000
              --backend-store-uri "$BACKEND_URI"
              --default-artifact-root "$ARTIFACT_URI"

---
apiVersion: v1
kind: Service

metadata:
  name: mlflow
  namespace: mlops

spec:
  type: ClusterIP

  selector:
    app: mlflow

  ports:
    - name: http
      port: 5000
      targetPort: 5000
```

Apply the configuration:

```bash
kubectl apply \
  -f lab-08-mlflow-deploy.yaml
```

Verify the resources:

```bash
kubectl -n mlops get pods,svc
```

Wait until the MLflow Pod reaches:

```text
Running
```

If startup fails, inspect the logs:

```bash
kubectl -n mlops logs deploy/mlflow
```

The resulting architecture is:

```text
                 Kubernetes Cluster
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
     MLflow         PostgreSQL         MinIO
      Pod               Pod              Pod
        │               │                │
        │               │                │
        ├──────────────►│                │
        │    Metadata                    │
        │                                │
        └───────────────────────────────►│
                      Artifacts
```

> **Dependency Note**
>
> The MLflow container must contain the database and S3 client dependencies required by the selected MLflow configuration. If the selected image does not contain them, build a small custom MLflow image containing the required packages.

---

## Step 7: Access the MLflow UI

For a local lab, use Kubernetes port forwarding:

```bash
kubectl -n mlops port-forward \
  svc/mlflow \
  5000:5000
```

Open:

```text
http://localhost:5000
```

in your browser.

You should see the MLflow Tracking interface.

The access path is:

```text
Browser
   ↓
localhost:5000
   ↓
kubectl Port Forward
   ↓
MLflow Service
   ↓
MLflow Pod
```

For production environments, MLflow can instead be exposed through an Ingress or gateway with TLS and an appropriate authentication mechanism.

---

## Step 8: Optional — Configure Ingress

If your cluster has an Ingress controller, create:

```text
lab-08-mlflow-ingress.yaml
```

Add:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress

metadata:
  name: mlflow
  namespace: mlops

spec:
  rules:
    - host: mlflow.localtest.me

      http:
        paths:
          - path: /
            pathType: Prefix

            backend:
              service:
                name: mlflow

                port:
                  number: 5000
```

Apply the resource:

```bash
kubectl apply \
  -f lab-08-mlflow-ingress.yaml
```

Verify it:

```bash
kubectl -n mlops get ingress
```

The request path becomes:

```text
Browser
   ↓
Ingress
   ↓
MLflow Service
   ↓
MLflow Pod
```

> **Production Security Note**
>
> Before exposing MLflow outside the cluster, add TLS, authentication, authorization, and suitable network restrictions.

---

## Step 9: Validate Experiment Tracking

Keep the MLflow port-forward active.

From your local machine, create a simple MLflow client script:

```python
import mlflow


mlflow.set_tracking_uri(
    "http://localhost:5000"
)

mlflow.set_experiment(
    "kubernetes-lab"
)

with mlflow.start_run():

    mlflow.log_param(
        "learning_rate",
        0.001
    )

    mlflow.log_metric(
        "loss",
        0.42,
        step=1
    )

    with open(
        "hello.txt",
        "w"
    ) as f:
        f.write(
            "MLflow artifact test"
        )

    mlflow.log_artifact(
        "hello.txt"
    )
```

Run the script.

Then open the MLflow UI and verify that:

* The `kubernetes-lab` experiment exists.
* A new run appears.
* `learning_rate` is recorded.
* `loss` is recorded.
* `hello.txt` appears as an artifact.

Next, inspect the MinIO:

```text
mlflow-artifacts
```

bucket and verify that the artifact was stored successfully.

The complete experiment path is:

```text
                   MLflow Client
                        │
                        ▼
               MLflow Tracking Server
                    /         \
                   /           \
                  ▼             ▼
           PostgreSQL          MinIO
             │                  │
             │                  │
      Parameters/Metrics     Artifacts
```

This verifies that experiment metadata and artifacts are being routed to the correct storage systems.

---

## Step 10: Understand Metadata and Artifact Separation

MLflow separates structured tracking information from larger experiment files.

### PostgreSQL Stores

```text
Experiments
Runs
Parameters
Metrics
Tags
Status
Timestamps
```

### MinIO Stores

```text
Models
Checkpoints
Plots
Reports
Images
Files
Other Artifacts
```

This separation is important because databases and object stores are optimized for different types of data.

```text
                 MLflow
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
     PostgreSQL           MinIO
     Structured           Objects
      Metadata            / Files
```

---

## Step 11: Production Security Considerations

The configuration in this lab is intentionally simplified.

A production environment should strengthen several areas.

### Credential Management

Avoid embedding credentials directly into command lines or manifests.

Use mechanisms such as:

* Kubernetes Secrets
* External secrets-management systems
* Cloud secrets managers
* Credential rotation

### Network Security

Restrict communication among MLflow, PostgreSQL, and MinIO using appropriate Kubernetes network controls.

A production architecture might follow:

```text
External Client
      ↓
Authenticated Gateway
      ↓
TLS
      ↓
MLflow
   │      │
   ▼      ▼
Postgres  MinIO
Internal Network Only
```

### External Access

Protect MLflow with:

* TLS
* Authentication
* Authorization
* Network restrictions
* Appropriate firewall policies

### Container Security

Production container images should be:

* Version-pinned
* Scanned for vulnerabilities
* Obtained from trusted registries
* Updated according to security policy

### Storage Protection

Persistent data should have:

* Backup policies
* Recovery procedures
* Appropriate retention
* Access controls
* Encryption where required

---

## Troubleshooting

### MLflow Pod Does Not Start

Inspect the Pod:

```bash
kubectl -n mlops get pods
```

Review logs:

```bash
kubectl -n mlops logs deploy/mlflow
```

If necessary, describe the Pod:

```bash
kubectl -n mlops describe pod \
  <MLFLOW_POD_NAME>
```

---

### PostgreSQL Connection Fails

Verify:

* `BACKEND_URI`
* PostgreSQL Service name
* PostgreSQL username
* PostgreSQL password
* Database name
* PostgreSQL Pod status

Check:

```bash
kubectl -n mlops get svc
```

and:

```bash
kubectl -n mlops get pods
```

---

### Artifacts Are Not Stored

Verify:

```text
MLFLOW_S3_ENDPOINT_URL
```

Check:

* MinIO credentials
* MinIO Service
* `mlflow-artifacts` bucket
* Artifact URI
* MinIO availability

Verify the MinIO resources:

```bash
kubectl -n mlops get pods,svc
```

---

### MLflow UI Cannot Be Reached

Verify the MLflow Pod:

```bash
kubectl -n mlops get pods
```

Verify the Service:

```bash
kubectl -n mlops get svc mlflow
```

Restart the port forward if required:

```bash
kubectl -n mlops port-forward \
  svc/mlflow \
  5000:5000
```

If using Ingress:

```bash
kubectl -n mlops get ingress
```

---

### MinIO Permission Errors

Check:

* Access key
* Secret key
* Bucket permissions
* Bucket existence
* Endpoint URL

Systematically checking each component helps determine whether the problem originates with:

```text
MLflow
PostgreSQL
MinIO
Persistent Storage
Kubernetes Networking
Secrets
Ingress
```

---

## Step 12: Clean Up

After completing the lab, remove the installed resources.

Uninstall MinIO:

```bash
helm -n mlops uninstall minio
```

Uninstall PostgreSQL:

```bash
helm -n mlops uninstall pg
```

Delete the namespace:

```bash
kubectl delete namespace mlops
```

Deleting the namespace removes namespaced resources created during the lab.

> **Storage Warning**
>
> Persistent volumes may or may not be deleted automatically. Their behavior depends on the Kubernetes StorageClass reclaim policy. Inspect persistent storage separately when complete removal is required.

If using a managed cloud Kubernetes cluster, also verify that no billable infrastructure remains.

---

## Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Created the `mlops` namespace
* [ ] Verified Kubernetes persistent-storage availability
* [ ] Installed MinIO
* [ ] Created the `mlflow-artifacts` bucket
* [ ] Installed PostgreSQL
* [ ] Created the `mlflowdb` database configuration
* [ ] Created the MLflow Kubernetes Secret
* [ ] Deployed the MLflow Tracking Server
* [ ] Created the MLflow ClusterIP Service
* [ ] Connected MLflow to PostgreSQL
* [ ] Connected MLflow to MinIO
* [ ] Accessed the MLflow web interface
* [ ] Configured Ingress if completing the optional section
* [ ] Created an MLflow experiment
* [ ] Logged a parameter
* [ ] Logged a metric
* [ ] Logged an artifact
* [ ] Verified metadata through MLflow
* [ ] Verified artifact storage in MinIO
* [ ] Reviewed production security considerations
* [ ] Removed lab resources when testing was complete

---

## Learning Outcomes

After completing this lab, you should be able to:

* Deploy an MLflow Tracking Server on Kubernetes.
* Configure PostgreSQL as the MLflow backend metadata store.
* Configure MinIO as an S3-compatible artifact store.
* Understand why experiment metadata and artifacts are stored separately.
* Configure Kubernetes Secrets for MLflow connection information.
* Use Kubernetes Services for internal component communication.
* Use port forwarding to access MLflow locally.
* Configure optional Ingress access.
* Validate end-to-end experiment tracking.
* Understand the role of persistent storage in an MLflow deployment.
* Troubleshoot MLflow, PostgreSQL, MinIO, storage, and Kubernetes networking.
* Explain the additional security controls required for a production deployment.

---

## Key Takeaway

**A production-style MLflow deployment separates experiment metadata from larger model artifacts. PostgreSQL provides structured persistent storage for runs, parameters, metrics, and tags, while MinIO provides scalable S3-compatible storage for models and other experiment artifacts.**

Kubernetes provides the orchestration, networking, persistent storage, and configuration mechanisms required to operate these components together. This separation creates a stronger foundation for persistent, reproducible, and scalable MLOps experiment tracking.
