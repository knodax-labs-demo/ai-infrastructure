# Hands-On Lab: Serve a Model Across AWS and Google Cloud

In this lab, you will deploy the same AI inference service across **Amazon EKS** and **Google Kubernetes Engine (GKE)** and expose both deployments through a global DNS or traffic-management layer.

The lab demonstrates a practical multi-cloud serving pattern in which:

* The same application artifact is deployed to both cloud providers.
* AWS and Google Cloud each maintain their own Kubernetes inference environment.
* Each environment exposes a load-balanced endpoint.
* A global routing layer directs users toward a healthy cloud endpoint.
* Cross-cloud failover can redirect requests if one provider becomes unavailable.

The completed architecture provides two layers of resilience:

```text
Intra-Cloud High Availability
        +
Cross-Cloud Failover
        ↓
Multi-Cloud AI Inference
```

---

## Lab Objective

By completing this lab, you will:

* Build one inference container image.
* Publish the image to Amazon ECR.
* Publish the same image to Google Artifact Registry.
* Create an Amazon EKS cluster.
* Create a Google Kubernetes Engine cluster.
* Deploy equivalent Kubernetes workloads in both clouds.
* Expose each deployment through a cloud load balancer.
* Configure global DNS or traffic routing.
* Validate inference through a common global endpoint.
* Identify which cloud served a request.
* Simulate a provider-side outage.
* Validate cross-cloud failover.
* Understand the difference between pod-level, cluster-level, and cloud-provider-level resilience.

---

## Estimated Time

**Approximately 120–180 minutes**

---

## Tools

This lab uses:

* AWS account
* Google Cloud account
* Docker
* AWS CLI
* Google Cloud CLI
* `kubectl`
* `eksctl`
* Amazon ECR
* Google Artifact Registry
* Amazon EKS
* Google Kubernetes Engine
* Kubernetes
* Optional Route 53
* Optional Cloudflare or another global traffic-management platform
* Optional custom domain

---

## Architecture

The overall multi-cloud architecture is:

```text
                         Users
                           │
                           ▼
               Global DNS / Traffic Manager
                           │
               ┌───────────┴───────────┐
               │                       │
               ▼                       ▼
          AWS Endpoint             GCP Endpoint
               │                       │
               ▼                       ▼
        AWS Load Balancer        GCP Load Balancer
               │                       │
               ▼                       ▼
          Amazon EKS                  GKE
         ┌─────┼─────┐          ┌─────┼─────┐
         │     │     │          │     │     │
        Pod   Pod   Pod         Pod   Pod   Pod
         │     │     │          │     │     │
         └─────┴─────┘          └─────┴─────┘
              AI Inference Service
```

Each cloud provides local Kubernetes high availability, while the global routing layer provides cross-cloud resilience.

---

# Step 1: Verify the Prerequisites

Before beginning, verify access to both cloud environments.

You need permissions to create:

### AWS

* ECR repositories
* EKS clusters
* Worker nodes
* Load balancers
* DNS records or health checks if using Route 53

### Google Cloud

* Artifact Registry repositories
* GKE clusters
* Worker nodes
* Load balancers

You should also have:

```text
docker
kubectl
eksctl
aws
gcloud
```

installed and authenticated.

Verify:

```bash
docker --version
```

```bash
kubectl version --client
```

```bash
eksctl version
```

```bash
aws --version
```

```bash
gcloud --version
```

The inference application should expose endpoints similar to:

```text
/healthz
/readyz
/predict
```

---

# Step 2: Build the Docker Image

Build the inference application once:

```bash
docker build \
  -t fastapi-inference:multi \
  .
```

The key idea is:

```text
One Application
      ↓
One Container Build
      ↓
Equivalent Deployment
Across Multiple Clouds
```

Using the same application artifact reduces configuration drift and makes it easier to compare behavior across providers.

---

# Step 3: Create an Amazon ECR Repository

Create:

```bash
aws ecr create-repository \
  --repository-name fastapi-inference \
  --region us-east-1
```

Authenticate Docker:

```bash
aws ecr get-login-password \
  --region us-east-1 \
  | docker login \
      --username AWS \
      --password-stdin \
      <account-id>.dkr.ecr.us-east-1.amazonaws.com
```

Define:

```bash
AWS_URI=<account-id>.dkr.ecr.us-east-1.amazonaws.com/fastapi-inference:latest
```

Tag:

```bash
docker tag \
  fastapi-inference:multi \
  $AWS_URI
```

Push:

```bash
docker push \
  $AWS_URI
```

The AWS flow is now:

```text
Local Image
    ↓
Amazon ECR
    ↓
Amazon EKS
```

---

# Step 4: Create a Google Artifact Registry Repository

Enable Artifact Registry:

```bash
gcloud services enable \
  artifactregistry.googleapis.com
```

Create the Docker repository:

```bash
gcloud artifacts repositories create inference \
  --repository-format=docker \
  --location=us-central1
```

Configure authentication:

```bash
gcloud auth configure-docker \
  us-central1-docker.pkg.dev
```

Define:

```bash
GCP_URI=us-central1-docker.pkg.dev/<project-id>/inference/fastapi-inference:latest
```

Tag the same image:

```bash
docker tag \
  fastapi-inference:multi \
  $GCP_URI
```

Push:

```bash
docker push \
  $GCP_URI
```

The Google Cloud flow is:

```text
Local Image
    ↓
Artifact Registry
    ↓
GKE
```

Both clouds are now using equivalent inference application code.

---

# Step 5: Create the Amazon EKS Cluster

Create:

```bash
eksctl create cluster \
  --name ai-eks \
  --region us-east-1 \
  --nodes 3
```

After creation, verify:

```bash
kubectl get nodes
```

You should see three worker nodes.

The AWS environment now looks like:

```text
Amazon EKS
   ├── Worker Node 1
   ├── Worker Node 2
   └── Worker Node 3
```

---

# Step 6: Create the GKE Cluster

Create:

```bash
gcloud container clusters create ai-gke \
  --region us-central1 \
  --num-nodes 3
```

Retrieve credentials:

```bash
gcloud container clusters get-credentials \
  ai-gke \
  --region us-central1
```

Verify:

```bash
kubectl get nodes
```

At this point, you have Kubernetes infrastructure in both cloud providers.

---

# Step 7: Create the Common Kubernetes Deployment

Create:

```text
lab-07-deployment.yaml
```

Add:

```yaml
apiVersion: apps/v1
kind: Deployment

metadata:
  name: inference-api

spec:
  replicas: 3

  selector:
    matchLabels:
      app: inference-api

  template:
    metadata:
      labels:
        app: inference-api

    spec:
      containers:
        - name: api
          image: <CLOUD_IMAGE_URI>

          ports:
            - containerPort: 8000

          readinessProbe:
            httpGet:
              path: /readyz
              port: 8000

            initialDelaySeconds: 5
            periodSeconds: 5

          livenessProbe:
            httpGet:
              path: /healthz
              port: 8000

            initialDelaySeconds: 15
            periodSeconds: 10
```

Replace:

```text
<CLOUD_IMAGE_URI>
```

with either:

```text
$AWS_URI
```

or:

```text
$GCP_URI
```

depending on the target environment.

---

## Why Reuse the Same Manifest?

The goal is:

```text
Common Kubernetes Architecture
          +
Cloud-Specific Image URI
          ↓
Consistent Deployment Model
```

This is one of Kubernetes' main benefits in multi-cloud architecture: the core workload definition can remain largely portable.

---

# Step 8: Create the Kubernetes Service

Create:

```text
lab-07-service.yaml
```

Add:

```yaml
apiVersion: v1
kind: Service

metadata:
  name: inference-svc

spec:
  type: LoadBalancer

  selector:
    app: inference-api

  ports:
    - name: http
      port: 80
      targetPort: 8000
```

The Service exposes:

```text
Cloud Load Balancer
       ↓
Kubernetes Service
       ↓
Ready Inference Pods
```

Only ready pods should receive normal service traffic.

---

# Step 9: Deploy to Amazon EKS

Switch `kubectl` to EKS:

```bash
aws eks update-kubeconfig \
  --name ai-eks \
  --region us-east-1
```

Verify:

```bash
kubectl get nodes
```

Update the Deployment image to the AWS image URI.

Then apply:

```bash
kubectl apply \
  -f lab-07-deployment.yaml
```

```bash
kubectl apply \
  -f lab-07-service.yaml
```

Verify:

```bash
kubectl get pods
```

and:

```bash
kubectl get svc \
  inference-svc
```

Wait until all three replicas are ready.

---

# Step 10: Record the AWS Endpoint

Run:

```bash
kubectl get svc \
  inference-svc
```

The external endpoint may resemble:

```text
a1b2c3d4e5.us-east-1.elb.amazonaws.com
```

Record it:

```bash
AWS_ENDPOINT=<aws-load-balancer-hostname>
```

Test:

```bash
curl \
  http://$AWS_ENDPOINT/healthz
```

---

# Step 11: Deploy to Google Kubernetes Engine

Switch `kubectl` to GKE:

```bash
gcloud container clusters get-credentials \
  ai-gke \
  --region us-central1
```

Verify:

```bash
kubectl get nodes
```

Update the Deployment image to:

```text
$GCP_URI
```

Apply:

```bash
kubectl apply \
  -f lab-07-deployment.yaml
```

```bash
kubectl apply \
  -f lab-07-service.yaml
```

Verify:

```bash
kubectl get pods
```

and:

```bash
kubectl get svc \
  inference-svc
```

---

# Step 12: Record the GCP Endpoint

Run:

```bash
kubectl get svc \
  inference-svc
```

The Service may receive an external IP such as:

```text
34.12.45.67
```

Record:

```bash
GCP_ENDPOINT=<gcp-external-address>
```

Test:

```bash
curl \
  http://$GCP_ENDPOINT/healthz
```

---

# Step 13: Verify Both Cloud Deployments Independently

Before creating global routing, make sure both environments work independently.

Test AWS:

```bash
curl \
  http://$AWS_ENDPOINT/healthz
```

Test GCP:

```bash
curl \
  http://$GCP_ENDPOINT/healthz
```

Then send inference requests to each endpoint.

Conceptually:

```text
Client
 ├──► AWS → Prediction
 └──► GCP → Prediction
```

Do not proceed until both deployments are independently healthy.

---

# Step 14: Add a Cloud Identifier

For testing, configure the application to return or log the cloud environment.

For example:

```text
CLOUD_PROVIDER=AWS
```

in EKS and:

```text
CLOUD_PROVIDER=GCP
```

in GKE.

The response might contain:

```json
{
  "prediction": "class_A",
  "cloud": "AWS"
}
```

or:

```json
{
  "prediction": "class_A",
  "cloud": "GCP"
}
```

This makes multi-cloud routing tests easier to understand.

In production, provider identity may be better placed in logs or traces rather than exposed to external clients.

---

# Step 15: Configure Global DNS Routing

Create a DNS name such as:

```text
inference.mycompany.com
```

Conceptually:

```text
inference.mycompany.com
          │
          ├── AWS EKS
          │      ↓
          │  AWS Endpoint
          │
          └── GCP GKE
                 ↓
             GCP Endpoint
```

The global routing platform could use:

* Latency-based routing
* Geographic routing
* Weighted routing
* Primary/secondary failover

Potential platforms include:

* Amazon Route 53
* Cloudflare
* Another global traffic-management platform

---

## Health-Aware Routing

Multiple DNS records alone do not create reliable failover.

The routing layer needs some way to determine whether an endpoint is healthy.

The correct logic is:

```text
Global DNS
    ↓
Health Check
    │
    ├── AWS Healthy → Eligible
    │
    └── AWS Unhealthy → Remove / Deprioritize
```

and similarly for GCP.

---

# Step 16: Understand DNS-Based Routing

A latency-aware model might behave like:

```text
User in Eastern U.S.
       ↓
AWS us-east-1


User near Central U.S.
       ↓
AWS or GCP
Depending on Routing Policy
```

A failover-oriented policy may instead use:

```text
Primary: AWS
Secondary: GCP
```

The exact behavior depends on the DNS or global traffic-management platform.

---

# Step 17: Test the Global Health Endpoint

Run:

```bash
curl -s \
  http://inference.mycompany.com/healthz
```

If correctly configured, the request should reach one of the healthy cloud deployments.

---

# Step 18: Send a Global Inference Request

Run:

```bash
curl -s -X POST \
  http://inference.mycompany.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

The global request path is:

```text
Client
   ↓
inference.mycompany.com
   ↓
Global Routing Policy
   ↓
AWS or GCP
   ↓
Cloud Load Balancer
   ↓
Kubernetes Service
   ↓
Inference Pod
   ↓
Prediction
```

---

# Step 19: Validate Which Cloud Served the Request

If you added the `CLOUD_PROVIDER` identifier, inspect the response:

```json
{
  "prediction": "class_A",
  "cloud": "AWS"
}
```

Repeat the request:

```bash
for i in {1..10}; do
  curl -s \
    http://inference.mycompany.com/healthz

  echo
done
```

Depending on your routing policy, repeated requests may continue going to the same provider or may vary.

---

# Step 20: Understand the Two Levels of Resilience

The architecture provides two distinct layers.

## Layer 1: Intra-Cloud High Availability

Within AWS:

```text
AWS Load Balancer
       ↓
EKS
 ┌─────┼─────┐
Pod   Pod   Pod
```

Within GCP:

```text
GCP Load Balancer
       ↓
GKE
 ┌─────┼─────┐
Pod   Pod   Pod
```

Each Kubernetes cluster handles:

* Pod failures
* Health-aware routing
* Replica replacement
* Local load balancing

---

## Layer 2: Cross-Cloud Resilience

Above the clusters:

```text
Global Routing
      │
   ┌──┴──┐
   │     │
  AWS   GCP
```

If one provider becomes unavailable, the global layer can redirect traffic toward the healthy environment.

---

# Step 21: Simulate an AWS-Side Failure

To test provider-level failover correctly, make the AWS inference environment unhealthy from the perspective of the global health check.

Switch to EKS:

```bash
aws eks update-kubeconfig \
  --name ai-eks \
  --region us-east-1
```

Scale the deployment to zero:

```bash
kubectl scale deployment \
  inference-api \
  --replicas=0
```

Verify:

```bash
kubectl get pods
```

Check endpoints:

```bash
kubectl get endpoints \
  inference-svc
```

There should no longer be healthy inference pod endpoints.

---

## Why Not Delete One Pod?

Deleting a single pod is not a provider failure.

If you delete one:

```text
Pod Deleted
    ↓
Deployment Replaces Pod
    ↓
Service Remains Healthy
```

That tests Kubernetes self-healing, not multi-cloud failover.

For this lab, scaling the entire serving deployment to zero creates a more meaningful provider-side service outage.

---

# Step 22: Observe Cross-Cloud Failover

Repeatedly test:

```bash
while true; do
  curl -s \
    http://inference.mycompany.com/healthz

  echo
  sleep 5
done
```

Expected behavior:

```text
AWS Healthy
    ↓
AWS May Receive Traffic
    ↓
AWS Becomes Unhealthy
    ↓
Health Check Detects Failure
    ↓
Global Router Updates Routing
    ↓
Traffic Goes to GCP
```

The exact transition time depends on:

* Health-check interval
* DNS TTL
* Resolver caching
* Routing platform
* Failover policy

---

# Step 23: Restore AWS

Restore:

```bash
kubectl scale deployment \
  inference-api \
  --replicas=3
```

Watch:

```bash
kubectl get pods \
  -w
```

Once all replicas are ready, verify:

```bash
curl \
  http://$AWS_ENDPOINT/healthz
```

After global health checks detect recovery, AWS may become eligible for traffic again depending on the routing policy.

---

# Step 24: Validate the Complete Architecture

The final architecture is:

```text
                           Users
                             │
                             ▼
                    Global Routing
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
             AWS Cloud                GCP Cloud
                 │                       │
                 ▼                       ▼
            AWS Load Balancer       GCP Load Balancer
                 │                       │
                 ▼                       ▼
                EKS                     GKE
           ┌─────┼─────┐          ┌─────┼─────┐
           │     │     │          │     │     │
          Pod   Pod   Pod         Pod   Pod   Pod
```

The resilience hierarchy is:

```text
Pod Failure
    ↓
Kubernetes Recovery


Cluster / Cloud Service Failure
    ↓
Global Traffic Failover
```

---

# Step 25: Production Improvements

This lab demonstrates the foundation. Production multi-cloud AI systems require additional design considerations.

## TLS

Use:

```text
HTTPS
```

rather than plain HTTP for public inference traffic.

---

## Authentication

Add:

* OIDC
* OAuth 2.0
* API gateway authentication
* Workload identity
* mTLS for sensitive service-to-service traffic

---

## Cloud-Specific Secrets

Avoid embedding credentials in manifests.

Use:

* AWS Secrets Manager
* Google Secret Manager
* Kubernetes Secrets
* External Secrets Operator or similar mechanisms

---

## Multi-Zone Clusters

Each individual cloud cluster should also span multiple failure domains where appropriate.

Conceptually:

```text
AWS
├── Zone A
├── Zone B
└── Zone C

GCP
├── Zone A
├── Zone B
└── Zone C
```

Cross-cloud resilience should not replace good intra-cloud architecture.

---

## Observability

Monitor each provider separately and globally.

Useful signals include:

* Request rate
* Error rate
* p50/p95/p99 latency
* Pod health
* Load-balancer health
* DNS health-check status
* Provider routing percentage
* Model version
* GPU utilization
* Failover events

---

## Data and Model Consistency

Multi-cloud serving becomes more complex when the inference service requires:

* Model artifacts
* Feature stores
* Vector databases
* Databases
* Caches
* Stateful dependencies

Those dependencies may also require cross-cloud replication or provider-independent designs.

---

## Cost Awareness

Running complete serving environments in two clouds increases cost.

Evaluate:

```text
Compute Cost
+
Load Balancers
+
Kubernetes Control Plane
+
Container Registry
+
Cross-Cloud Networking
+
DNS / Traffic Management
```

Multi-cloud should be driven by resilience, regulatory, portability, or business requirements rather than added solely for architectural novelty.

---

# Step 26: Clean Up AWS

Delete EKS:

```bash
eksctl delete cluster \
  --name ai-eks \
  --region us-east-1
```

After deletion, verify that AWS resources created for the cluster are removed as expected.

Also review:

* ECR repository
* Load balancers
* Health checks
* DNS records
* Associated networking resources

---

# Step 27: Clean Up Google Cloud

Delete GKE:

```bash
gcloud container clusters delete \
  ai-gke \
  --region us-central1
```

Also review:

* Artifact Registry repository
* Load balancer resources
* External addresses
* DNS-related resources

Remove resources you no longer need to avoid continuing charges.

---

# Recommended Folder Structure

The standalone lab structure can be:

```text
lab_multicloud_inference/
├── app/
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── Dockerfile
└── README.md
```

For your book companion repository, I recommend:

```text
chapter-20/
├── lab-07-serve-a-model-across-aws-and-google-cloud.md
├── lab-07-deployment.yaml
├── lab-07-service.yaml
└── lab-07-Dockerfile
```

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Built one inference image.
* [ ] Created an Amazon ECR repository.
* [ ] Pushed the image to ECR.
* [ ] Created a Google Artifact Registry repository.
* [ ] Pushed the same image to Artifact Registry.
* [ ] Created an EKS cluster.
* [ ] Created a GKE cluster.
* [ ] Verified worker nodes in both clouds.
* [ ] Created a reusable Kubernetes Deployment.
* [ ] Configured three inference replicas.
* [ ] Added readiness probes.
* [ ] Added liveness probes.
* [ ] Created a `LoadBalancer` Service.
* [ ] Deployed the workload to EKS.
* [ ] Deployed the same workload to GKE.
* [ ] Recorded both external endpoints.
* [ ] Verified each cloud endpoint independently.
* [ ] Configured a global DNS or traffic-management endpoint.
* [ ] Added health-aware routing.
* [ ] Sent inference traffic through the global endpoint.
* [ ] Identified which cloud served requests.
* [ ] Scaled AWS inference to zero.
* [ ] Confirmed the AWS service became unhealthy.
* [ ] Observed traffic redirect toward GCP.
* [ ] Restored AWS serving capacity.
* [ ] Reviewed production multi-cloud considerations.
* [ ] Cleaned up cloud resources.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Explain how Kubernetes supports a common deployment model across clouds.
* Publish equivalent application images to provider-specific registries.
* Deploy the same inference service to Amazon EKS and GKE.
* Expose each cloud environment through a Kubernetes `LoadBalancer` Service.
* Explain global DNS and traffic-management routing.
* Distinguish geographic routing from health-based failover.
* Explain why multiple DNS records alone do not guarantee failover.
* Validate which cloud handles a request.
* Test provider-level failure rather than only pod-level failure.
* Explain the difference between intra-cloud HA and cross-cloud resilience.
* Understand the operational complexity introduced by multi-cloud architectures.
* Identify production requirements around TLS, identity, observability, data consistency, and cost.

---

# Key Takeaway

**Multi-cloud AI inference combines local Kubernetes resilience with a global routing layer that can redirect traffic across cloud providers when an entire serving environment becomes unavailable.**

The architecture can be summarized as:

```text
One Application
      ↓
Two Cloud Registries
      ↓
EKS + GKE
      ↓
Multiple Replicas Per Cloud
      ↓
Cloud Load Balancers
      ↓
Health-Aware Global Routing
      ↓
Cross-Cloud AI Inference
```

Kubernetes provides a relatively consistent workload deployment model across providers, but multi-cloud availability requires more than duplicating clusters. The global routing layer must understand endpoint health, each cloud must remain independently operational, and application dependencies such as models, data stores, secrets, networking, and observability must also be designed for cross-cloud operation.

The result is two complementary resilience layers: **high availability within each cloud and provider-level failover across clouds**.
