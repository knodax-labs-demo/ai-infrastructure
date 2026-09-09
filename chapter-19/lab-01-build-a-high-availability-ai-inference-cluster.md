# Hands-On Lab: Build a High-Availability AI Inference Cluster

In this lab, you will deploy a **high-availability AI inference service on Kubernetes**.

The environment will combine several complementary resilience mechanisms:

* Multiple inference replicas
* Kubernetes Service load balancing
* Readiness probes
* Liveness probes
* Automatic pod recovery
* Horizontal Pod Autoscaling
* Pod Disruption Budget
* Optional topology-aware scheduling across nodes or zones
* Failure injection and recovery testing

The goal is to demonstrate that high availability is not provided by a single Kubernetes feature. Reliable inference infrastructure is created by combining **redundancy, health-aware routing, automatic recovery, disruption protection, elastic scaling, and failure-domain distribution**.

---

## Lab Objective

Build a self-healing and automatically scalable AI inference cluster that continues serving requests when individual inference pods fail.

The completed architecture follows this pattern:

```text id="haarch1"
                 Client
                    │
                    ▼
             Load Balancer
                    │
                    ▼
          Kubernetes Service
                    │
          ┌─────────┼─────────┐
          │         │         │
          ▼         ▼         ▼
       Pod 1      Pod 2      Pod 3
          │         │         │
          └──── AI Inference ─┘
                    │
                    ▼
             Health Probes
                    │
          ┌─────────┴─────────┐
          │                   │
     Pod Recovery            HPA
          │                   │
          ▼                   ▼
   Replace Failures      Scale Capacity
```

---

## Estimated Time

**Approximately 90–120 minutes**

---

## Tools

This lab uses:

* Kubernetes cluster
* At least two worker nodes
* `kubectl`
* Kubernetes Metrics Server for CPU-based HPA
* Containerized inference application
* Optional multi-zone cluster
* Optional load-testing utility such as `hey` or ApacheBench

The example uses a CPU-based inference service.

GPU workloads additionally require:

* GPU-enabled Kubernetes worker nodes
* NVIDIA device plugin or GPU Operator
* GPU resource requests and limits
* Appropriate accelerator-aware scheduling

---

# Step 1: Verify the Prerequisites

Verify cluster access:

```bash id="ha01"
kubectl cluster-info
```

List the nodes:

```bash id="ha02"
kubectl get nodes
```

For this lab, use a cluster with at least:

```text id="ha03"
2 worker nodes
```

Verify your inference application exposes:

```text id="ha04"
/healthz
/readyz
```

These endpoints allow Kubernetes to distinguish between:

```text id="ha05"
Process Alive?
     ↓
Liveness Probe

Ready for Traffic?
     ↓
Readiness Probe
```

If you plan to use CPU-based autoscaling, confirm that Metrics Server is available:

```bash id="ha06"
kubectl top nodes
```

and:

```bash id="ha07"
kubectl top pods -A
```

If these return resource metrics, HPA has the data it needs for the CPU-based example.

---

# Step 2: Create the Namespace

Create:

```bash id="ha08"
kubectl create namespace ai-ha
```

Verify:

```bash id="ha09"
kubectl get namespace ai-ha
```

Using a dedicated namespace keeps the lab isolated and makes cleanup straightforward.

---

# Step 3: Deploy Multiple Inference Replicas

Create:

```text id="ha10"
lab-07-deployment.yaml
```

Add:

```yaml id="ha11"
apiVersion: apps/v1
kind: Deployment

metadata:
  name: ha-inference
  namespace: ai-ha

spec:
  replicas: 3

  selector:
    matchLabels:
      app: ha-inference

  template:
    metadata:
      labels:
        app: ha-inference

    spec:
      containers:
        - name: model-api
          image: YOUR_REGISTRY/secure-ml:latest

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

          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"

            limits:
              cpu: "500m"
              memory: "1Gi"
```

Replace:

```text id="ha12"
YOUR_REGISTRY/secure-ml:latest
```

with the container image used by your inference application.

Apply:

```bash id="ha13"
kubectl apply \
  -f lab-07-deployment.yaml
```

Verify:

```bash id="ha14"
kubectl -n ai-ha get pods
```

You should see three inference pods.

---

## Why Multiple Replicas Matter

With one replica:

```text id="ha15"
Pod Fails
   ↓
No Serving Capacity
```

With three replicas:

```text id="ha16"
Pod 1 ──┐
Pod 2 ──┼──► Inference Service
Pod 3 ──┘
```

If one pod fails:

```text id="ha17"
Pod 1 → Failed
Pod 2 → Healthy
Pod 3 → Healthy
```

the service can continue using the remaining ready replicas while Kubernetes creates a replacement.

---

# Step 4: Understand Readiness and Liveness

The Deployment uses two different probes.

## Readiness Probe

```yaml id="ha18"
readinessProbe:
  httpGet:
    path: /readyz
    port: 8000
```

The readiness probe controls whether Kubernetes should send traffic to the pod.

Conceptually:

```text id="ha19"
Pod Running
    ↓
Model Still Loading
    ↓
Readiness = False
    ↓
No Service Traffic
    ↓
Model Ready
    ↓
Readiness = True
    ↓
Receive Traffic
```

---

## Liveness Probe

```yaml id="ha20"
livenessProbe:
  httpGet:
    path: /healthz
    port: 8000
```

The liveness probe determines whether the application is still functioning.

If it repeatedly fails:

```text id="ha21"
Application Unhealthy
      ↓
Liveness Probe Fails
      ↓
Kubernetes Restarts Container
```

Readiness and liveness solve different problems and should not be treated as interchangeable.

---

# Step 5: Create the Load-Balanced Service

Create:

```text id="ha22"
lab-07-service.yaml
```

Add:

```yaml id="ha23"
apiVersion: v1
kind: Service

metadata:
  name: ha-inference-svc
  namespace: ai-ha

spec:
  selector:
    app: ha-inference

  ports:
    - name: http
      port: 80
      targetPort: 8000

  type: LoadBalancer
```

Apply:

```bash id="ha24"
kubectl apply \
  -f lab-07-service.yaml
```

Verify:

```bash id="ha25"
kubectl -n ai-ha get svc \
  ha-inference-svc
```

The request path becomes:

```text id="ha26"
Client
   ↓
LoadBalancer
   ↓
Kubernetes Service
   ↓
Ready Pods Only
```

If a pod fails its readiness probe, Kubernetes removes it from active Service endpoints until it becomes ready again.

---

## Local Kubernetes Note

A Service of type:

```text id="ha27"
LoadBalancer
```

typically provisions an external load balancer on supported cloud platforms.

Local environments such as Minikube or other local clusters may require an additional load-balancer mechanism before an external address becomes available.

---

# Step 6: Configure Horizontal Pod Autoscaling

Create an HPA:

```bash id="ha28"
kubectl -n ai-ha autoscale deployment \
  ha-inference \
  --cpu-percent=70 \
  --min=3 \
  --max=10
```

Verify:

```bash id="ha29"
kubectl -n ai-ha get hpa
```

The scaling model is:

```text id="ha30"
Minimum = 3 Pods
      ↓
Traffic Increases
      ↓
CPU > 70%
      ↓
HPA Adds Replicas
      ↓
Up to 10 Pods
```

When demand later falls:

```text id="ha31"
Traffic Falls
    ↓
Resource Usage Falls
    ↓
HPA Scales Down
    ↓
Never Below 3
```

---

## Why Resource Requests Matter

CPU-based HPA relies on utilization relative to requested resources.

The Deployment contains:

```yaml id="ha32"
resources:
  requests:
    cpu: "250m"
```

Without appropriate requests, CPU utilization-based autoscaling may not behave as intended.

---

## Better AI Autoscaling Signals

CPU is convenient for a lab but may not represent actual inference pressure.

Production AI services may scale using:

* Requests per second
* Queue depth
* Concurrent requests
* Inference latency
* Token throughput
* GPU utilization
* GPU memory
* Batch backlog

Conceptually:

```text id="ha33"
AI Workload
     ↓
Operational Metric
     ↓
Autoscaler
     ↓
Replica Count
```

---

# Step 7: Configure a Pod Disruption Budget

Create:

```text id="ha34"
lab-07-pdb.yaml
```

Add:

```yaml id="ha35"
apiVersion: policy/v1
kind: PodDisruptionBudget

metadata:
  name: ha-inference-pdb
  namespace: ai-ha

spec:
  minAvailable: 2

  selector:
    matchLabels:
      app: ha-inference
```

Apply:

```bash id="ha36"
kubectl apply \
  -f lab-07-pdb.yaml
```

Verify:

```bash id="ha37"
kubectl -n ai-ha get pdb
```

---

## What the PDB Does

With:

```text id="ha38"
replicas = 3
minAvailable = 2
```

supported voluntary disruptions should preserve at least two available replicas.

Examples include:

* Node draining
* Some maintenance operations
* Voluntary pod eviction

The protection model is:

```text id="ha39"
3 Available
    ↓
Voluntary Eviction Requested
    ↓
Would Availability Fall Below 2?
       │
   ┌───┴───┐
   │       │
  Yes      No
   │       │
 Block    Allow
```

---

## What a PDB Does Not Do

A Pod Disruption Budget does **not** prevent involuntary failures such as:

* Hardware failure
* Node crash
* Network outage
* Application crash
* Sudden power loss

Therefore:

```text id="ha40"
PDB
 +
Replica Redundancy
 +
Automatic Recovery
```

are complementary mechanisms.

---

# Step 8: Distribute Replicas Across Nodes

Running three replicas does not provide strong resilience if all three run on the same node.

Without topology control:

```text id="ha41"
Worker Node 1
├── Pod 1
├── Pod 2
└── Pod 3

Worker Node 1 Fails
      ↓
All Replicas Lost
```

To spread replicas across worker nodes, add this under the Pod specification:

```yaml id="ha42"
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: kubernetes.io/hostname
    whenUnsatisfiable: DoNotSchedule

    labelSelector:
      matchLabels:
        app: ha-inference
```

The desired distribution becomes:

```text id="ha43"
Node 1 → Pod 1
Node 2 → Pod 2
Node 3 → Pod 3
```

depending on available cluster capacity.

---

# Step 9: Optional — Spread Across Availability Zones

For multi-zone clusters, use:

```yaml id="ha44"
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: DoNotSchedule

    labelSelector:
      matchLabels:
        app: ha-inference
```

Verify available topology labels:

```bash id="ha45"
kubectl get nodes \
  --show-labels
```

The resilience model becomes:

```text id="ha46"
Availability Zone A → Pod 1
Availability Zone B → Pod 2
Availability Zone C → Pod 3
```

If one zone fails:

```text id="ha47"
Zone A → Unavailable
Zone B → Still Serving
Zone C → Still Serving
```

Topology-aware placement reduces the blast radius of infrastructure failures.

---

# Step 10: Verify Pod Distribution

Run:

```bash id="ha48"
kubectl -n ai-ha get pods \
  -o wide
```

Inspect the:

```text id="ha49"
NODE
```

column.

Verify that inference replicas are distributed across failure domains as expected.

---

# Step 11: Test Automatic Pod Recovery

List the pods:

```bash id="ha50"
kubectl -n ai-ha get pods
```

Select one:

```text id="ha51"
ha-inference-xxxxxxxxxx-yyyyy
```

Delete it:

```bash id="ha52"
kubectl -n ai-ha delete pod \
  <pod-name>
```

Watch:

```bash id="ha53"
kubectl -n ai-ha get pods \
  -w
```

Expected flow:

```text id="ha54"
Pod Deleted
    ↓
Replica Count Drops
    ↓
Deployment Detects Missing Replica
    ↓
Replacement Pod Created
    ↓
Container Starts
    ↓
Readiness Probe Passes
    ↓
Service Begins Routing Traffic
```

This demonstrates Kubernetes self-healing.

---

# Step 12: Validate Service Availability During Failure

Retrieve the Service:

```bash id="ha55"
kubectl -n ai-ha get svc \
  ha-inference-svc
```

If an external address is available, run:

```bash id="ha56"
while true; do
  curl -s \
    http://<external-ip>/healthz

  echo
  sleep 1
done
```

While the loop runs, delete a pod from another terminal:

```bash id="ha57"
kubectl -n ai-ha delete pod \
  <pod-name>
```

Monitor:

```bash id="ha58"
kubectl -n ai-ha get pods \
  -w
```

The goal is to observe:

```text id="ha59"
Pod Failure
    ↓
Failed Pod Removed from Ready Endpoints
    ↓
Other Pods Continue Serving
    ↓
Replacement Pod Starts
    ↓
Readiness Passes
    ↓
New Pod Joins Service
```

---

## Important Availability Note

Multiple replicas reduce failure impact but do not guarantee zero downtime under every condition.

Availability may still be affected by:

* Insufficient cluster capacity
* Networking failures
* Slow application startup
* Incorrect probes
* Load-balancer behavior
* Node or zone concentration
* Resource exhaustion

Production resilience should be validated with realistic failure and load testing.

---

# Step 13: Test Horizontal Pod Autoscaling

Generate sustained traffic against the inference service using a load-testing tool.

While traffic is running, monitor:

```bash id="ha60"
kubectl -n ai-ha get hpa \
  -w
```

In another terminal:

```bash id="ha61"
kubectl -n ai-ha get pods \
  -w
```

Expected progression:

```text id="ha62"
3 Pods
   ↓
Load Increases
   ↓
CPU Target Exceeded
   ↓
4 Pods
   ↓
5 Pods
   ↓
...
   ↓
Up to 10 Pods
```

When the workload falls, the deployment should eventually move back toward:

```text id="ha63"
3 replicas
```

---

# Step 14: Examine Service Endpoints

Check the endpoints backing the Service:

```bash id="ha64"
kubectl -n ai-ha get endpoints \
  ha-inference-svc
```

This helps confirm which pod IPs are currently eligible to receive traffic.

The Service effectively maintains:

```text id="ha65"
Service
   ↓
Ready Endpoint 1
Ready Endpoint 2
Ready Endpoint 3
```

Pods that are not ready should not appear as normal ready serving targets.

---

# Step 15: Examine Deployment State

Run:

```bash id="ha66"
kubectl -n ai-ha get deployment \
  ha-inference
```

For more detail:

```bash id="ha67"
kubectl -n ai-ha describe deployment \
  ha-inference
```

Observe:

* Desired replicas
* Current replicas
* Ready replicas
* Available replicas
* Deployment events

---

# Step 16: Examine Cluster Events

Run:

```bash id="ha68"
kubectl -n ai-ha get events \
  --sort-by=.metadata.creationTimestamp
```

Events can reveal:

* Pod scheduling
* Container startup
* Probe failures
* Pod termination
* Replica creation
* Scaling operations

This information is particularly useful when validating recovery behavior.

---

# Step 17: Understand the Complete HA Architecture

The complete request and recovery architecture is:

```text id="ha69"
                     Clients
                        │
                        ▼
                External Load Balancer
                        │
                        ▼
                 Kubernetes Service
                        │
             ┌──────────┼──────────┐
             │          │          │
             ▼          ▼          ▼
          Pod A       Pod B       Pod C
             │          │          │
          Ready?      Ready?      Ready?
             │          │          │
             └──────────┼──────────┘
                        │
                        ▼
                 AI Inference
```

Control plane behavior:

```text id="ha70"
Deployment
   ├── Maintains Replica Count
   │
   ├── Replaces Failed Pods
   │
   └── Performs Rollouts
   │
   ├── HPA
   │     └── Adjusts Capacity
   │
   └── PDB
         └── Protects Against
             Excessive Voluntary Evictions
```

Topology-aware scheduling adds:

```text id="ha71"
Replicas
   ↓
Different Nodes / Zones
   ↓
Reduced Failure Blast Radius
```

---

# Step 18: High Availability vs. Scalability

These concepts are related but different.

## High Availability

Goal:

```text id="ha72"
Continue Serving
When Components Fail
```

Mechanisms include:

* Multiple replicas
* Health checks
* Automatic recovery
* Failure-domain spread
* PDB

---

## Scalability

Goal:

```text id="ha73"
Handle More Demand
```

Mechanisms include:

* HPA
* Additional replicas
* Cluster autoscaling
* Larger inference capacity

A mature platform requires both:

```text id="ha74"
High Availability
       +
Scalability
       =
Resilient Inference Platform
```

---

# Step 19: GPU-Based Extension

The example uses CPU resources:

```yaml id="ha75"
resources:
  requests:
    cpu: "250m"
    memory: "512Mi"
```

For an NVIDIA GPU inference service, the pod may instead require:

```yaml id="ha76"
resources:
  limits:
    nvidia.com/gpu: "1"
```

The cluster also needs:

* NVIDIA GPU nodes
* NVIDIA driver
* Device plugin or GPU Operator
* Accelerator-aware scheduling

For GPU inference, scaling decisions may be better driven by:

```text id="ha77"
GPU Utilization
GPU Memory
Queue Depth
Concurrent Requests
Inference Latency
Token Throughput
```

rather than CPU alone.

---

# Step 20: Production Improvements

This lab establishes the foundation.

Production AI inference infrastructure may additionally use:

## Multi-Zone Deployment

```text id="ha78"
Zone A
Zone B
Zone C
```

to survive larger infrastructure failures.

---

## Multi-Region Deployment

For stronger geographic resilience:

```text id="ha79"
Region A
    │
    ├── AI Cluster
    │
Region B
    │
    └── AI Cluster
```

combined with global traffic management.

---

## Authentication and TLS

Use:

* HTTPS
* JWT/OIDC
* Workload identity
* API gateway
* mTLS when needed

---

## Model-Specific Readiness

Instead of checking only whether FastAPI is running, readiness should ideally confirm that:

* Model loaded successfully
* Required GPU is available
* Model repository is reachable
* Dependencies are available

---

## Centralized Observability

Monitor:

* Replica health
* Restart rate
* p50/p95/p99 latency
* Throughput
* Error rate
* HPA status
* GPU utilization
* Queue depth
* Node failures
* Probe failures

---

## Automated Failure Testing

Regularly test:

* Pod termination
* Node loss
* Zone failure
* Network failure
* Dependency failure
* Traffic spikes

This evolves basic recovery testing toward chaos engineering.

---

# Step 21: Clean Up

Delete the namespace:

```bash id="ha80"
kubectl delete namespace ai-ha
```

This removes namespace-scoped resources including:

* Deployment
* Service
* Horizontal Pod Autoscaler
* Pod Disruption Budget

Verify:

```bash id="ha81"
kubectl get namespace ai-ha
```

---

# Recommended Lab Folder Structure

The standalone lab can use:

```text id="ha82"
lab_ha_cluster/
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── pdb.yaml
└── README.md
```

For your book companion repository, I recommend:

```text id="ha83"
chapter-19/
├── lab-07-build-a-high-availability-ai-inference-cluster.md
├── lab-07-deployment.yaml
├── lab-07-service.yaml
└── lab-07-pdb.yaml
```

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Created the `ai-ha` namespace.
* [ ] Deployed three baseline inference replicas.
* [ ] Configured readiness probes.
* [ ] Configured liveness probes.
* [ ] Defined CPU and memory requests and limits.
* [ ] Created the load-balanced Kubernetes Service.
* [ ] Verified Service endpoints.
* [ ] Configured HPA.
* [ ] Set minimum replicas to three.
* [ ] Allowed scaling up to ten replicas.
* [ ] Created the Pod Disruption Budget.
* [ ] Configured `minAvailable: 2`.
* [ ] Reviewed node-level topology spreading.
* [ ] Reviewed zone-level topology spreading.
* [ ] Verified pod placement across nodes where possible.
* [ ] Deliberately deleted an inference pod.
* [ ] Observed automatic pod recreation.
* [ ] Verified remaining replicas continued serving.
* [ ] Generated load and observed HPA behavior.
* [ ] Reviewed Kubernetes events.
* [ ] Cleaned up the namespace.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Explain why multiple replicas are required for highly available inference.
* Use Kubernetes Deployments to maintain desired serving capacity.
* Distinguish readiness from liveness.
* Use Services to load balance across healthy replicas.
* Configure Horizontal Pod Autoscaling.
* Explain the relationship between resource requests and CPU-based HPA.
* Configure a Pod Disruption Budget.
* Explain what PDBs do and do not protect against.
* Distribute workloads across nodes or availability zones.
* Validate Kubernetes self-healing by deliberately deleting a pod.
* Observe service continuity during pod recovery.
* Use Kubernetes events and endpoints to analyze resilience.
* Distinguish high availability from scalability.
* Explain how these patterns extend to GPU-based inference.

---

# Key Takeaway

**High availability for AI inference is achieved by combining multiple complementary resilience mechanisms rather than relying on one Kubernetes feature.**

The core architecture is:

```text id="ha84"
Multiple Replicas
       +
Load Balancing
       +
Health Probes
       +
Automatic Recovery
       +
Pod Disruption Budget
       +
Topology Distribution
       +
Autoscaling
       ↓
Highly Available AI Inference
```

Multiple replicas provide redundant serving capacity, readiness probes prevent traffic from reaching unavailable workloads, liveness probes support automatic recovery, and the Deployment controller restores failed replicas. HPA allows capacity to grow with demand, while a PDB helps protect availability during voluntary maintenance. Topology-aware scheduling further reduces the risk that a single node or availability-zone failure removes all serving capacity.

For production AI systems, this foundation can be extended with **multi-zone and multi-region deployment, GPU-aware scheduling, secure traffic management, centralized monitoring, model-specific readiness checks, persistent model storage, and automated chaos testing**.
