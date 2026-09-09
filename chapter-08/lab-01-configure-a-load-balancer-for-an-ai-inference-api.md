# Hands-On Lab: Configure a Load Balancer for an AI Inference API

In this lab, you will expose an AI inference API through a resilient and scalable load-balancing architecture using Kubernetes.

You will configure **Layer 4 and Layer 7 traffic distribution, health and readiness checks, rate limiting, autoscaling, canary deployment, observability, and failure testing**.

The architecture can be adapted to managed Kubernetes platforms such as:

* Amazon Elastic Kubernetes Service (EKS)
* Google Kubernetes Engine (GKE)
* Azure Kubernetes Service (AKS)
* Appropriately configured self-managed Kubernetes clusters

An optional Envoy configuration demonstrates advanced Layer 7 capabilities such as **least-request routing, retries, outlier detection, and circuit breaking**.

The emphasis of this lab is on understanding how production AI inference traffic can be distributed safely while protecting backend resources from failures, overload, and sudden traffic spikes.

---

## Lab Objective

Configure a resilient load-balancing architecture for an AI inference API using Kubernetes.

You will work with:

* Layer 4 load balancing
* Layer 7 application-aware routing
* Health checks
* Readiness checks
* Rate limiting
* Backpressure
* Horizontal autoscaling
* Canary deployment
* Observability
* Failure and recovery testing

---

## Estimated Time

**90–120 minutes**

## Cost

Local Kubernetes environments may be free.

Managed cloud Kubernetes clusters can generate charges for:

* Worker nodes
* Load balancers
* Public IP addresses
* Storage
* Monitoring
* Network traffic

> **⚠️ Cost Warning**
>
> If you use EKS, GKE, AKS, or another cloud Kubernetes environment, remove externally provisioned load balancers and other billable resources immediately after completing the lab.

---

## Lab Outcome

By the end of this lab, you should have a working AI inference API behind Kubernetes load-balancing services and understand how to combine:

* Layer 4 load balancing
* Layer 7 application-aware routing
* Health and readiness checks
* Rate limiting and backpressure
* Horizontal autoscaling
* Canary releases
* Basic inference observability
* Failure and recovery testing

---

## Lab Files

Use the following accompanying files:

```text
chapter-08/
├── lab-07-configure-a-load-balancer-for-an-ai-inference-api.md
├── lab-07-k8s-inference-fastapi.yaml
└── lab-07-envoy.yaml
```

### Kubernetes Manifest

```text
lab-07-k8s-inference-fastapi.yaml
```

Contains resources for:

* Namespace
* FastAPI inference application
* Deployments
* Services
* Horizontal Pod Autoscaler
* Ingress
* Canary deployment

### Envoy Configuration

```text
lab-07-envoy.yaml
```

Provides an optional Layer 7 load balancer with:

* Least-request routing
* Retries
* Outlier detection
* Circuit breakers
* Request limits

---

## Step 1: Verify the Prerequisites

Before starting the lab, make sure you have access to a Kubernetes cluster.

You can use:

* Amazon EKS
* Google Kubernetes Engine
* Azure Kubernetes Service
* Minikube or another suitable local Kubernetes environment
* A self-managed Kubernetes cluster

Verify that `kubectl` is configured:

```bash
kubectl cluster-info
```

Check the cluster nodes:

```bash
kubectl get nodes
```

All required nodes should report:

```text
Ready
```

### Layer 7 Requirements

For the Layer 7 portion of the lab, install an Ingress controller such as **ingress-nginx**.

Verify available Ingress classes:

```bash
kubectl get ingressclass
```

### Autoscaling Requirements

CPU-based Horizontal Pod Autoscaling requires a source of resource metrics such as Kubernetes Metrics Server.

Check:

```bash
kubectl top nodes
```

You can also check Pod metrics:

```bash
kubectl top pods -A
```

If metrics are unavailable, the CPU-based HPA portion of the lab will not operate correctly.

### Optional HTTPS Requirements

For full HTTPS testing, you will also need:

* A DNS hostname
* A TLS certificate
* A corresponding private key

---

## Step 2: Understand the Architecture

The inference service exposes three primary endpoints:

| Endpoint   | Purpose                        |
| ---------- | ------------------------------ |
| `/healthz` | Application health check       |
| `/readyz`  | Readiness check                |
| `/v1/echo` | Sample inference-style request |

A Kubernetes **ClusterIP Service** provides internal connectivity.

A **LoadBalancer Service** provides Layer 4 external traffic distribution.

An **NGINX Ingress** provides Layer 7 HTTP routing, TLS termination, and request controls.

A **Horizontal Pod Autoscaler** adjusts inference replicas based on resource utilization.

A **canary Ingress** sends a percentage of requests to a second application version.

The architecture is:

```text
                    Client
                      │
                      ▼
          ┌──────────────────────┐
          │ External Load Balancer│
          │       Layer 4         │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │    NGINX Ingress     │
          │       Layer 7        │
          │ Routing / TLS / Rate │
          │       Limiting       │
          └──────────┬───────────┘
                     │
                     ▼
            Kubernetes Service
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       Pod 1       Pod 2       Pod 3
          │          │          │
          └────── AI Inference ─┘
```

A canary path adds a second application version:

```text
                    Ingress
                       │
              ┌────────┴────────┐
              │                 │
            ~90%              ~10%
              │                 │
              ▼                 ▼
         Stable Model       Canary Model
```

The optional Envoy configuration can replace or complement the Layer 7 proxy when more advanced traffic-management and resilience controls are required.

---

## Step 3: Deploy the Inference Stack

Apply the Kubernetes manifest:

```bash
kubectl apply -f lab-07-k8s-inference-fastapi.yaml
```

The manifest creates the resources required by the exercise, including:

* Namespace
* Application configuration
* Inference Deployments
* Services
* Horizontal Pod Autoscaler
* Ingress
* Canary configuration

Verify the resources:

```bash
kubectl -n ai get pods,svc,ingress,hpa
```

Confirm that the application Pods reach:

```text
Running
```

and:

```text
Ready
```

before proceeding.

### Troubleshoot an Unhealthy Pod

If a Pod does not become ready, inspect it:

```bash
kubectl -n ai describe pod <POD_NAME>
```

Review the logs:

```bash
kubectl -n ai logs <POD_NAME>
```

---

## Step 4: Optional — Configure TLS

If you are using HTTPS through the Ingress, create a Kubernetes TLS Secret.

```bash
kubectl -n ai create secret tls inference-tls \
  --cert=server.crt \
  --key=server.key
```

Verify the Secret:

```bash
kubectl -n ai get secret inference-tls
```

Your certificate should match the hostname used by the Ingress.

---

## Step 5: Test the Inference API Locally

Before testing external load balancing, verify that the application itself works correctly.

Port-forward the internal ClusterIP Service:

```bash
kubectl -n ai port-forward svc/inference-svc 8080:80
```

Leave this terminal running.

Open another terminal and test the health endpoint:

```bash
curl -s http://localhost:8080/healthz
```

Then send a sample request:

```bash
curl -s -X POST \
  http://localhost:8080/v1/echo \
  -H "Content-Type: application/json" \
  -d '{"text":"hello"}'
```

This validates:

```text
Client
  ↓
Port Forward
  ↓
ClusterIP Service
  ↓
Inference Pod
```

> **Troubleshooting Principle**
>
> Validate each networking layer independently. First verify the application and internal Service, then test the external load balancer, and finally test the Layer 7 Ingress.

---

## Step 6: Test Layer 4 Load Balancing

The `inference-lb` Service uses Kubernetes:

```text
type: LoadBalancer
```

Check its status:

```bash
kubectl -n ai get svc inference-lb
```

In a managed cloud Kubernetes environment, the platform will typically provision an external load balancer.

Wait until an external IP address or hostname appears.

Test:

```bash
curl -s http://<EXTERNAL_LB>/healthz
```

For example:

```text
http://203.0.113.10/healthz
```

or:

```text
http://example-lb.cloudprovider.com/healthz
```

At this stage, the request path is:

```text
Client
   ↓
External Load Balancer
   ↓
Kubernetes Service
   ↓
Inference Pods
```

This represents **Layer 4 traffic distribution** without the application-aware routing rules provided by the Ingress.

---

## Step 7: Add Layer 7 Routing with NGINX Ingress

Layer 7 load balancing understands HTTP-level information and can apply application-aware policies.

In this lab, NGINX Ingress provides:

* HTTP routing
* TLS termination
* Rate limiting
* Proxy timeout configuration
* Canary routing

Configure a DNS record such as:

```text
api.example.com
```

to point to the endpoint used by your Ingress deployment.

Verify the Ingress:

```bash
kubectl -n ai get ingress
```

Once DNS and TLS are configured, test:

```bash
curl -s https://api.example.com/healthz
```

Send an inference request:

```bash
curl -s -X POST \
  https://api.example.com/v1/echo \
  -H "Content-Type: application/json" \
  -d '{"text":"gamma"}'
```

The request path becomes:

```text
Client
   ↓
HTTPS
   ↓
NGINX Ingress
   │
   ├── TLS Termination
   ├── Routing
   ├── Rate Limiting
   └── Timeout Controls
          ↓
   Kubernetes Service
          ↓
      Inference Pods
```

These controls demonstrate how an application-aware proxy can protect AI inference resources before requests reach CPU- or GPU-backed model servers.

---

## Step 8: Test the Canary Release

Deploying a new model or application version to all users simultaneously can introduce unnecessary risk.

The canary configuration routes approximately:

```text
10%
```

of eligible traffic to the canary backend.

The remaining traffic continues to use the stable deployment.

Generate multiple requests:

```bash
for i in {1..50}; do
  curl -s https://api.example.com/healthz
  echo
done
```

Observe whether responses identify traffic handled by different versions.

The rollout concept is:

```text
Stable 100%
    ↓
Canary 10%
Stable 90%
    ↓
Canary 25%
Stable 75%
    ↓
Canary 50%
Stable 50%
    ↓
Canary 100%
```

If latency, errors, or application behavior degrade:

```text
Problem Detected
      ↓
Stop Rollout
      ↓
Return Traffic
to Stable Version
```

> **Production Principle**
>
> Progressive delivery allows a new model or inference version to be validated under real traffic before a complete rollout.

---

## Step 9: Test Horizontal Autoscaling

The lab's Horizontal Pod Autoscaler uses CPU utilization as a simple scaling signal.

Watch the HPA:

```bash
kubectl -n ai get hpa -w
```

In another terminal, generate load against the inference endpoint.

Also watch the Pods:

```bash
kubectl -n ai get pods -w
```

The HPA begins with a minimum number of replicas and can scale the Deployment up to its configured maximum as demand increases.

The basic feedback loop is:

```text
Traffic Increases
       ↓
CPU Utilization Increases
       ↓
HPA Detects Demand
       ↓
More Replicas Created
       ↓
Traffic Distributed Across
More Inference Pods
```

After traffic decreases, continue monitoring to observe scale-down behavior.

### Production AI Scaling Signals

CPU utilization is used here because it is easy to demonstrate.

GPU-backed inference environments may benefit from metrics such as:

* GPU utilization
* GPU memory utilization
* Request rate
* Queue depth
* Concurrent requests
* Active inference requests
* Token throughput
* Batch size
* Request latency

These signals can be integrated through custom or external metrics.

---

## Step 10: Optional — Use Envoy as an Intelligent Layer 7 Load Balancer

The accompanying:

```text
lab-07-envoy.yaml
```

demonstrates an alternative Layer 7 architecture using Envoy Proxy.

The configuration can demonstrate:

* `LEAST_REQUEST` load balancing
* Retries
* Per-try timeouts
* Outlier detection
* Removal of unhealthy backends
* Circuit breaking
* Connection limits
* Request limits

A simplified request path is:

```text
Client
   ↓
Envoy Proxy
   │
   ├── Least-Request Routing
   ├── Health Awareness
   ├── Retries
   ├── Outlier Detection
   └── Circuit Breaking
          ↓
   Inference Backends
```

Unlike simple round-robin distribution, an intelligent Layer 7 proxy can make routing decisions based on backend health, request behavior, and configured load-balancing policies.

---

## Step 11: Understand Health Checks and Readiness

The application exposes:

```text
/healthz
```

and:

```text
/readyz
```

These endpoints serve different purposes.

| Check               | Purpose                                                   |
| ------------------- | --------------------------------------------------------- |
| **Liveness/Health** | Determines whether the application is functioning         |
| **Readiness**       | Determines whether the application should receive traffic |

A running process is not necessarily ready to perform inference.

For example:

```text
Container Running
      ↓
Model Still Loading
      ↓
Readiness = False
      ↓
Do Not Send Traffic
      ↓
Model Ready
      ↓
Readiness = True
      ↓
Accept Requests
```

A model-serving backend may need time to:

* Load model weights
* Allocate GPU memory
* Warm caches
* Initialize dependencies
* Establish downstream connections

Proper readiness probes prevent traffic from reaching such instances prematurely.

---

## Step 12: Understand Rate Limiting and Backpressure

AI inference APIs can consume expensive CPU and GPU resources.

If traffic arrives faster than the backend can process it, queues can grow rapidly.

```text
Traffic Spike
     ↓
Request Queue Grows
     ↓
Latency Increases
     ↓
Backend Saturation
     ↓
Failures
```

Rate limiting helps control how quickly requests can enter the system:

```text
Incoming Requests
       ↓
Rate Limiter
       ↓
Accepted Traffic
       ↓
Inference Backend
```

Other resilience mechanisms include:

* Request limits
* Queue limits
* Circuit breakers
* Outlier detection
* Bounded retries
* Timeouts

> **Retry Warning**
>
> Retries must be carefully bounded. Excessive retries can amplify an outage by generating additional traffic against systems that are already overloaded.

---

## Step 13: Add Minimum Viable Observability

A production inference architecture should be evaluated using measurable behavior.

Important metrics include:

| Metric              | Why It Matters                      |
| ------------------- | ----------------------------------- |
| Requests per second | Measures throughput                 |
| p50 latency         | Typical request latency             |
| p95 latency         | Tail performance                    |
| p99 latency         | Extreme tail latency                |
| HTTP error rate     | Measures failed requests            |
| Queue depth         | Indicates backend pressure          |
| Replica count       | Shows scaling behavior              |
| CPU utilization     | Measures compute pressure           |
| GPU utilization     | Measures accelerator usage          |
| GPU memory          | Detects accelerator memory pressure |

The sample application includes a:

```text
/metrics
```

placeholder that can be extended with Prometheus-compatible metrics.

NGINX Ingress and Envoy can also expose proxy-level metrics including:

* Request volume
* HTTP response codes
* Upstream latency
* Retry counts
* Backend health

Grafana can visualize these measurements.

OpenTelemetry can provide distributed tracing across:

```text
Client
   ↓
Load Balancer
   ↓
Ingress / Gateway
   ↓
Inference Application
   ↓
Model Backend
```

### Why Tail Latency Matters

Average latency can hide slow requests.

For example:

```text
Average: 120 ms
p95:     450 ms
p99:    1800 ms
```

The average appears acceptable, while a small percentage of users experience much slower responses.

For production AI inference, **p95 and p99 latency are often more informative than average latency alone**.

---

## Step 14: Perform Load Testing

Generate controlled traffic to observe load-balancing and autoscaling behavior.

One option is `hey`:

```bash
hey -z 60s \
  -q 50 \
  -c 100 \
  -m POST \
  -H "Content-Type: application/json" \
  -d '{"text":"load"}' \
  https://api.example.com/v1/echo
```

Alternatively, use `wrk` with an appropriate POST request script:

```bash
wrk -t4 \
  -c100 \
  -d60s \
  -s post.lua \
  https://api.example.com/v1/echo
```

While the load test runs, observe:

```bash
kubectl -n ai get pods -w
```

and:

```bash
kubectl -n ai get hpa -w
```

Monitor:

* Pod count
* HPA activity
* Request throughput
* Latency
* Error rates
* CPU utilization
* GPU utilization if applicable
* Proxy behavior

> **Cost Warning**
>
> Avoid generating unnecessary traffic against cloud or GPU environments that incur significant usage charges.

---

## Step 15: Perform a Failure Test

A resilient inference architecture should continue serving traffic when an individual Pod becomes unavailable.

While requests are being generated, delete one inference Pod:

```bash
kubectl -n ai delete pod \
  -l app=inference-api \
  --wait=false
```

Continue observing client responses.

Watch Kubernetes:

```bash
kubectl -n ai get pods -w
```

The expected sequence is:

```text
Inference Pod Deleted
        ↓
Pod Stops Receiving Traffic
        ↓
Healthy Pods Continue Serving
        ↓
Deployment Detects Missing Replica
        ↓
Replacement Pod Created
        ↓
New Pod Starts
        ↓
Readiness Probe Passes
        ↓
Pod Receives Traffic
```

This demonstrates the relationship between:

* Kubernetes reconciliation
* Deployments
* Readiness probes
* Services
* Load balancing

---

## Step 16: Analyze the Results

Compare system behavior under:

* Normal traffic
* Elevated traffic
* Rate limiting
* Autoscaling
* Canary deployment
* Pod failure
* Envoy routing, if used

Consider the following questions:

1. Did latency remain within an acceptable range?
2. Did traffic continue reaching healthy backends after a Pod failed?
3. Did the HPA create replicas quickly enough?
4. What happened to p95 and p99 latency as traffic increased?
5. Did rate limiting prevent backend overload?
6. Did the canary receive approximately the expected traffic percentage?
7. Did the system recover automatically from Pod failure?
8. Did Envoy outlier detection or circuit breaking improve resilience?

A production-ready inference platform should do more than return successful responses.

It should also:

* Maintain predictable latency
* Route traffic to healthy backends
* Recover from failures
* Scale when demand rises
* Avoid overwhelming model servers
* Provide enough telemetry to diagnose problems

---

## Step 17: Capture the Lab Results

Record evidence of the completed lab.

Capture:

* External load-balancer endpoint or DNS name
* Successful inference requests
* Initial replica count
* Peak replica count
* Replica count after scale-down
* HPA scaling events
* p95 latency
* p99 latency
* Canary percentage
* Canary behavior
* Pod failure behavior
* Recovery time
* Envoy resilience behavior, if tested

Use a table such as:

| Test             | What to Measure                              | Result     |
| ---------------- | -------------------------------------------- | ---------- |
| **Baseline**     | Throughput, p95/p99 latency, replicas        | __________ |
| **Rate Limited** | Request handling and limited traffic         | __________ |
| **Autoscaling**  | Replica growth and latency                   | __________ |
| **Canary**       | Traffic distribution                         | __________ |
| **Pod Failure**  | Availability and recovery                    | __________ |
| **Envoy**        | Retries, outliers, circuit breaking, latency | __________ |

---

## Step 18: Clean Up

Remove the main Kubernetes resources:

```bash
kubectl delete -f lab-07-k8s-inference-fastapi.yaml
```

If you deployed the optional Envoy configuration:

```bash
kubectl delete -f lab-07-envoy.yaml
```

If you created TLS resources separately, remove them as appropriate.

For example:

```bash
kubectl -n ai delete secret inference-tls
```

Verify remaining resources:

```bash
kubectl -n ai get all
```

In a managed cloud environment, also verify that the cloud provider removed:

* External load balancers
* Public IP addresses
* Persistent disks
* Kubernetes worker resources
* Other billable networking resources

> **⚠️ Important**
>
> Cloud load balancers and public IP addresses may continue generating charges even after the inference application itself has been removed.

---

## Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Deployed the inference application
* [ ] Verified that Pods reached `Running` and `Ready`
* [ ] Tested the application through the internal ClusterIP Service
* [ ] Tested Layer 4 traffic through a LoadBalancer Service
* [ ] Configured or tested Layer 7 routing
* [ ] Verified health checks
* [ ] Verified readiness checks
* [ ] Tested rate limiting
* [ ] Tested the canary deployment
* [ ] Observed HPA behavior
* [ ] Generated controlled load
* [ ] Monitored latency and errors
* [ ] Deleted an inference Pod
* [ ] Verified automatic recovery
* [ ] Tested Envoy features if completing the optional section
* [ ] Recorded the lab results
* [ ] Removed Kubernetes and cloud resources

---

## Lab Extensions

After completing the core exercise, you can extend the architecture in several ways.

### Add Authentication

Add JWT-based authentication at the gateway or integrate Envoy external authorization.

### Use AI-Specific Autoscaling Metrics

Replace CPU-only autoscaling with metrics such as:

* GPU utilization
* GPU memory
* Queue depth
* Concurrent requests
* Request latency
* Token generation rate

### Add Accelerator-Aware Routing

An advanced router could consider the capabilities and current utilization of different GPUs:

```text
Incoming Request
       ↓
Intelligent Router
       │
       ├── H100 Backend
       ├── A100 Backend
       └── T4 Backend
```

Routing weights could reflect each backend's measured inference capacity.

### Explore Kubernetes Gateway API

Where supported, investigate Kubernetes Gateway API as a modern traffic-management abstraction.

### Add Complete Observability

Integrate:

* Prometheus
* Grafana
* OpenTelemetry

to create a feedback loop across:

```text
Traffic
   ↓
Load Balancing
   ↓
Inference
   ↓
Metrics
   ↓
Autoscaling
   ↓
Updated Capacity
```

---

## Learning Outcomes

After completing this lab, you should be able to:

* Explain the difference between Layer 4 and Layer 7 load balancing.
* Expose an AI inference API through Kubernetes Services and Ingress.
* Configure health and readiness checks.
* Understand the role of rate limiting and backpressure.
* Scale inference workloads with the Horizontal Pod Autoscaler.
* Use canary routing for progressive model or application releases.
* Monitor throughput, errors, and tail latency.
* Test how an inference platform responds to Pod failure.
* Explain how Envoy can provide advanced routing and resilience capabilities.
* Understand why GPU-backed AI inference requires workload-aware traffic management.

---

## Key Takeaway

**AI inference load balancing involves much more than distributing requests evenly among servers. Effective designs combine Layer 4 and Layer 7 traffic management with backend health awareness, autoscaling, rate limiting, failure isolation, progressive deployment, and observability.**

GPU-backed services introduce additional considerations such as **batching, queue depth, concurrency, accelerator utilization, and heterogeneous GPU capacity**. Kubernetes provides the orchestration foundation, while technologies such as NGINX Ingress and Envoy provide application-aware traffic management.

By testing load, failures, scaling, and canary behavior together, you can validate that an inference architecture remains responsive as operating conditions change.
