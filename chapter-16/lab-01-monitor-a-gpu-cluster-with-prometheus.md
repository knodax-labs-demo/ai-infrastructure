# Hands-On Lab: Monitor a GPU Cluster with Prometheus

In this lab, you will build a practical observability environment for a Kubernetes-based NVIDIA GPU cluster using **Prometheus, NVIDIA DCGM Exporter, Grafana, and Alertmanager**.

NVIDIA DCGM Exporter collects GPU telemetry from GPU-enabled Kubernetes nodes and exposes the metrics in Prometheus format. Prometheus collects and stores the resulting time-series data, while Grafana provides dashboards for analyzing GPU utilization, memory consumption, temperature, and power usage. Prometheus alerting rules detect selected GPU conditions, and Alertmanager provides the foundation for routing resulting alerts to notification and incident-management systems.

The lab uses the **kube-prometheus-stack** Helm chart to simplify deployment of the core Kubernetes monitoring components.

---

## Lab Objectives

By completing this lab, you will learn how to:

* Deploy Prometheus, Grafana, and Alertmanager on Kubernetes.
* Collect NVIDIA GPU metrics with DCGM Exporter.
* Deploy DCGM Exporter across GPU nodes using a DaemonSet.
* Configure Prometheus discovery with a `ServiceMonitor`.
* Query accelerator telemetry using PromQL.
* Visualize GPU behavior in Grafana.
* Create GPU-focused Prometheus alerting rules.
* Associate GPU activity with Kubernetes workloads where supported.
* Troubleshoot the complete GPU monitoring path.

---

## Architecture

The monitoring architecture is:

```text
NVIDIA GPU
    ↓
NVIDIA Driver / DCGM
    ↓
DCGM Exporter
    ↓
Service :9400
    ↓
ServiceMonitor
    ↓
Prometheus
   ├──────────────► Grafana
   │                  ↓
   │              Dashboards
   │
   └──────────────► PrometheusRule
                        ↓
                    Alertmanager
                        ↓
                   Notifications
```

The broader monitoring stack also collects standard Kubernetes infrastructure telemetry through components such as:

* Node Exporter
* kube-state-metrics
* Kubernetes control-plane exporters

This creates a unified view of both cluster resources and expensive GPU accelerators.

---

## Estimated Time

**Approximately 90–120 minutes**

---

## Tools

This lab uses:

* Kubernetes
* NVIDIA GPU-enabled worker node
* NVIDIA driver
* NVIDIA Kubernetes device plugin or GPU Operator
* `kubectl`
* Helm
* NVIDIA DCGM Exporter
* Prometheus
* Grafana
* Alertmanager
* Prometheus Operator
* PromQL

---

# Step 1: Verify the Prerequisites

Before beginning, make sure the Kubernetes cluster contains at least one NVIDIA GPU-enabled node.

Verify cluster access:

```bash
kubectl get nodes
```

Verify Helm:

```bash
helm version
```

GPU worker nodes require a functioning NVIDIA driver and a Kubernetes mechanism for advertising GPU resources.

If your cluster already uses:

* NVIDIA GPU Operator
* NVIDIA device plugin
* A cloud-managed GPU stack

review the existing configuration before installing additional GPU components.

> **Important**
>
> Avoid installing duplicate device plugins or exporters when the cluster already provides equivalent functionality.

---

## Install the NVIDIA Device Plugin if Required

If the NVIDIA device plugin is not already installed, add the NVIDIA Helm repository:

```bash
helm repo add nvidia \
  https://nvidia.github.io/k8s-device-plugin
```

Update Helm:

```bash
helm repo update
```

Create a namespace:

```bash
kubectl create namespace gpu-operator \
  --dry-run=client \
  -o yaml \
  | kubectl apply -f -
```

Install the device plugin:

```bash
helm install nvidia-device-plugin \
  nvidia/k8s-device-plugin \
  -n gpu-operator
```

Verify the nodes:

```bash
kubectl get nodes
```

Inspect the GPU node:

```bash
kubectl describe node <gpu-node-name>
```

Look for:

```text
nvidia.com/gpu
```

under:

```text
Capacity
Allocatable
```

A properly configured node may show something similar to:

```text
Capacity:
  nvidia.com/gpu: 1

Allocatable:
  nvidia.com/gpu: 1
```

---

# Step 2: Create the Monitoring Namespace

Create:

```bash
kubectl create namespace monitoring
```

Verify:

```bash
kubectl get namespace monitoring
```

Using a dedicated namespace creates a clean operational boundary for:

* Prometheus
* Grafana
* Alertmanager
* DCGM Exporter
* Monitoring CRDs
* Alert rules

---

# Step 3: Install Prometheus, Grafana, and Alertmanager

The `kube-prometheus-stack` Helm chart provides:

```text
kube-prometheus-stack
        │
        ├── Prometheus
        ├── Grafana
        ├── Alertmanager
        ├── Prometheus Operator
        ├── Node Exporter
        └── kube-state-metrics
```

The Prometheus Operator adds Kubernetes custom resources such as:

```text
ServiceMonitor
PrometheusRule
```

which simplify discovery and alert configuration.

---

## Add the Helm Repository

Run:

```bash
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts
```

Update:

```bash
helm repo update
```

---

## Install the Stack

Run:

```bash
helm install monitoring \
  prometheus-community/kube-prometheus-stack \
  -n monitoring
```

Verify:

```bash
kubectl -n monitoring get pods
```

Wait until the major components reach the `Running` state.

You should eventually see resources corresponding to:

```text
Prometheus
Grafana
Alertmanager
Node Exporter
kube-state-metrics
Prometheus Operator
```

---

# Step 4: Identify and Label GPU Nodes

DCGM Exporter should run only on GPU-enabled nodes.

For this lab, apply a simple custom label:

```bash
kubectl label nodes \
  <gpu-node-name> \
  gpu=true
```

For multiple GPU nodes, repeat the command.

Verify:

```bash
kubectl get nodes \
  --show-labels
```

The placement model becomes:

```text
CPU Node
gpu label absent
      ↓
No DCGM Exporter


GPU Node
gpu=true
      ↓
DCGM Exporter
```

> **Production Note**
>
> Production clusters may already use labels created by NVIDIA GPU Operator, Node Feature Discovery, or the cloud platform. Prefer existing standardized labels when available.

---

# Step 5: Deploy NVIDIA DCGM Exporter

NVIDIA DCGM Exporter exposes accelerator telemetry in Prometheus format.

Depending on the GPU and environment, metrics can include:

* GPU utilization
* GPU framebuffer memory
* Temperature
* Power consumption
* Clock behavior
* ECC errors
* Other accelerator health information

Running DCGM Exporter as a `DaemonSet` places one exporter on each selected GPU node.

Create:

```text
lab-07-dcgm-exporter.yaml
```

Add:

```yaml
apiVersion: apps/v1
kind: DaemonSet

metadata:
  name: dcgm-exporter
  namespace: monitoring

  labels:
    app: dcgm-exporter

spec:
  selector:
    matchLabels:
      app: dcgm-exporter

  template:
    metadata:
      labels:
        app: dcgm-exporter

    spec:
      nodeSelector:
        gpu: "true"

      tolerations:
        - effect: NoSchedule
          operator: Exists

        - effect: NoExecute
          operator: Exists

      containers:
        - name: dcgm-exporter
          image: nvcr.io/nvidia/k8s/dcgm-exporter:latest
          imagePullPolicy: IfNotPresent

          ports:
            - name: metrics
              containerPort: 9400

          securityContext:
            privileged: true

          env:
            - name: DCGM_EXPORTER_KUBERNETES
              value: "true"

          volumeMounts:
            - name: pod-resources
              mountPath: /var/lib/kubelet/pod-resources
              readOnly: true

      volumes:
        - name: pod-resources

          hostPath:
            path: /var/lib/kubelet/pod-resources
            type: Directory

---
apiVersion: v1
kind: Service

metadata:
  name: dcgm-exporter
  namespace: monitoring

  labels:
    app: dcgm-exporter

spec:
  clusterIP: None

  selector:
    app: dcgm-exporter

  ports:
    - name: metrics
      port: 9400
      targetPort: metrics
```

Apply:

```bash
kubectl apply \
  -f lab-07-dcgm-exporter.yaml
```

Verify:

```bash
kubectl -n monitoring get pods \
  -l app=dcgm-exporter \
  -o wide
```

There should normally be one exporter Pod on each labeled GPU node.

---

## DCGM Exporter Architecture

```text
GPU Node 1
   ↓
DCGM Exporter :9400
        │
        │
GPU Node 2
   ↓    │
DCGM Exporter :9400
        │
        ▼
Headless Service
        ↓
Prometheus
```

> **Production Note**
>
> The lab uses `latest` for simplicity. Production environments should pin exporter and Helm chart versions that have been validated with your Kubernetes version, NVIDIA driver, CUDA version, and DCGM environment.

---

# Step 6: Configure Prometheus Scraping

Prometheus Operator discovers targets using Kubernetes resources such as `ServiceMonitor`.

Create:

```text
lab-07-dcgm-servicemonitor.yaml
```

Add:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor

metadata:
  name: dcgm-exporter
  namespace: monitoring

  labels:
    release: monitoring

spec:
  selector:
    matchLabels:
      app: dcgm-exporter

  namespaceSelector:
    matchNames:
      - monitoring

  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
```

Apply:

```bash
kubectl apply \
  -f lab-07-dcgm-servicemonitor.yaml
```

Verify:

```bash
kubectl -n monitoring get servicemonitor
```

The discovery path is now:

```text
DCGM Exporter Pod
       ↓
Service
       ↓
ServiceMonitor
       ↓
Prometheus Operator
       ↓
Prometheus Target
```

The:

```text
release: monitoring
```

label corresponds to the Helm release used in this lab.

If your Helm release or Prometheus selector differs, adjust the label accordingly.

---

# Step 7: Inspect Raw GPU Metrics

Before troubleshooting Prometheus, first verify that DCGM Exporter itself is producing metrics.

Select an exporter Pod:

```bash
POD=$(kubectl -n monitoring get pod \
  -l app=dcgm-exporter \
  -o jsonpath='{.items[0].metadata.name}')
```

Port-forward it:

```bash
kubectl -n monitoring port-forward \
  pod/$POD \
  9400:9400
```

In another terminal:

```bash
curl -s \
  http://localhost:9400/metrics \
  | head -n 40
```

You may see metrics such as:

```text
DCGM_FI_DEV_GPU_UTIL
DCGM_FI_DEV_GPU_TEMP
DCGM_FI_DEV_FB_USED
DCGM_FI_DEV_POWER_USAGE
```

The exact set depends on:

* GPU hardware
* DCGM version
* DCGM Exporter version
* Driver configuration
* Enabled metrics

> **Important**
>
> Treat the actual `/metrics` endpoint as the authoritative source for metric names available in your environment.

---

# Step 8: Access Prometheus

Find the Prometheus Service:

```bash
kubectl -n monitoring get svc \
  | grep prometheus
```

Port-forward it:

```bash
kubectl -n monitoring port-forward \
  svc/monitoring-kube-prometheus-prometheus \
  9090:9090
```

Open:

```text
http://localhost:9090
```

---

# Step 9: Query GPU Metrics with PromQL

## GPU Utilization

Run:

```promql
avg by (instance, gpu) (
  DCGM_FI_DEV_GPU_UTIL
)
```

This provides average GPU utilization grouped by instance and GPU.

---

## GPU Memory Utilization

Run:

```promql
100 *
(
  DCGM_FI_DEV_FB_USED
  /
  DCGM_FI_DEV_FB_TOTAL
)
```

This estimates framebuffer memory utilization as a percentage.

---

## GPU Temperature

Run:

```promql
max by (instance, gpu) (
  DCGM_FI_DEV_GPU_TEMP
)
```

---

## GPU Power Consumption

Run:

```promql
avg by (instance, gpu) (
  DCGM_FI_DEV_POWER_USAGE
)
```

---

## Monitoring View

You now have:

```text
GPU Utilization
GPU Memory
GPU Temperature
GPU Power
       ↓
    Prometheus
       ↓
      PromQL
```

If a query returns no data:

1. Inspect DCGM Exporter's `/metrics` endpoint.
2. Verify the exact metric name.
3. Use Prometheus query autocompletion.
4. Check Prometheus target discovery.

---

# Step 10: Access Grafana

Retrieve the Grafana administrator password:

```bash
kubectl -n monitoring get secret \
  monitoring-grafana \
  -o jsonpath="{.data.admin-password}" \
  | base64 -d

echo
```

Port-forward Grafana:

```bash
kubectl -n monitoring port-forward \
  svc/monitoring-grafana \
  3000:80
```

Open:

```text
http://localhost:3000
```

Sign in using:

```text
Username: admin
Password: <retrieved-password>
```

---

# Step 11: Build a GPU Dashboard

Create Grafana panels for:

* GPU Utilization
* GPU Memory Utilization
* GPU Temperature
* GPU Power Usage

A useful dashboard layout is:

```text
┌────────────────────────────┐
│ GPU Utilization            │
├────────────────────────────┤
│ GPU Memory Utilization     │
├────────────────────────────┤
│ GPU Temperature            │
├────────────────────────────┤
│ GPU Power Consumption      │
└────────────────────────────┘
```

Use the PromQL queries from the previous step.

Useful visualization types include:

* Time series
* Gauge
* Stat

For larger clusters, create dashboard variables for:

```text
Node
GPU
Instance
Namespace
Pod
```

when corresponding labels are available.

---

# Step 12: Configure GPU Alerts

Monitoring becomes operationally valuable when selected conditions can generate alerts.

Prometheus Operator uses:

```text
PrometheusRule
```

for declarative alert definitions.

Create:

```text
lab-07-gpu-alerts.yaml
```

Add:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule

metadata:
  name: gpu-alerts
  namespace: monitoring

  labels:
    release: monitoring

spec:
  groups:
    - name: gpu.rules

      rules:

        - alert: GPUScrapeMissing

          expr: >
            up{job="dcgm-exporter"} == 0

          for: 10m

          labels:
            severity: warning

          annotations:
            summary: "DCGM exporter scrape failing"
            description: "Prometheus cannot scrape a DCGM exporter target."


        - alert: GPUHighTemperature

          expr: >
            max by (instance, gpu)
            (DCGM_FI_DEV_GPU_TEMP) > 80

          for: 5m

          labels:
            severity: warning

          annotations:
            summary: "High GPU temperature"
            description: "GPU temperature has remained above the configured threshold."


        - alert: GPUMemoryPressure

          expr: >
            100 *
            (DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL)
            > 90

          for: 10m

          labels:
            severity: warning

          annotations:
            summary: "High GPU memory utilization"
            description: "GPU memory utilization has remained above 90%."


        - alert: GPUECCErrorsSpike

          expr: >
            rate(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL[5m]) > 0

          for: 5m

          labels:
            severity: critical

          annotations:
            summary: "GPU ECC errors increasing"
            description: "Single-bit ECC errors are increasing."
```

Apply:

```bash
kubectl apply \
  -f lab-07-gpu-alerts.yaml
```

Verify:

```bash
kubectl -n monitoring get prometheusrule
```

---

## Alerting Flow

```text
GPU Metric
    ↓
Prometheus
    ↓
PrometheusRule
    ↓
Condition True?
    ↓
Required Duration Met?
    ↓
Alert
    ↓
Alertmanager
    ↓
Notification System
```

Potential notification destinations include organizationally supported:

* Email
* Chat systems
* Pager systems
* Incident-management platforms

> **Threshold Note**
>
> Values such as `80°C` and `90%` memory utilization are examples for the lab. Production thresholds should reflect GPU specifications, workload behavior, service objectives, and operational policies.

---

# Step 13: Explore Workload-Level GPU Visibility

Infrastructure-level utilization becomes more useful when you can associate it with workloads.

The DCGM Exporter configuration mounts:

```text
/var/lib/kubelet/pod-resources
```

which may allow GPU telemetry to include workload context.

Depending on the environment and exporter configuration, labels may identify:

```text
Namespace
Pod
Container
GPU
Node
```

This provides a path such as:

```text
GPU Utilization
      ↓
GPU
      ↓
Node
      ↓
Pod
      ↓
Namespace
      ↓
Application / Team
```

This can help with:

* Chargeback
* Showback
* Capacity planning
* Underutilization analysis
* Troubleshooting
* GPU scheduling decisions

If Multi-Instance GPU (MIG) is enabled, confirm compatibility across:

* NVIDIA driver
* DCGM
* Device plugin
* DCGM Exporter
* Kubernetes GPU configuration

---

# Step 14: Generate GPU Activity

If the GPU is idle, utilization metrics may remain near zero.

Run a GPU workload or AI training/inference job while observing Grafana and Prometheus.

Monitor:

```bash
nvidia-smi
```

and compare it with:

```promql
DCGM_FI_DEV_GPU_UTIL
```

You should see utilization rise while the workload executes.

Conceptually:

```text
AI Workload Starts
       ↓
GPU Utilization Rises
       ↓
DCGM Detects Change
       ↓
Exporter Publishes Metric
       ↓
Prometheus Scrapes
       ↓
Grafana Displays Change
```

---

# Step 15: Validate Prometheus Discovery

Verify the Service:

```bash
kubectl -n monitoring get svc \
  dcgm-exporter
```

Verify the ServiceMonitor:

```bash
kubectl -n monitoring get servicemonitor \
  dcgm-exporter
```

Check Prometheus and confirm that DCGM Exporter appears as an active scrape target.

The expected pipeline is:

```text
GPU
 ↓
DCGM
 ↓
Exporter
 ↓
Service
 ↓
ServiceMonitor
 ↓
Prometheus
```

Every stage must function for metrics to reach the dashboard.

---

# Step 16: Troubleshooting

## DCGM Exporter Pods Are Not Running

Check:

```bash
kubectl -n monitoring get pods \
  -l app=dcgm-exporter
```

Inspect logs:

```bash
kubectl -n monitoring logs \
  <dcgm-exporter-pod>
```

Verify the underlying node:

```bash
kubectl describe node \
  <gpu-node-name>
```

Check:

* NVIDIA driver
* GPU availability
* Node labels
* Pod scheduling
* DaemonSet events

---

## Exporter Runs but Metrics Are Missing

Inspect directly:

```bash
curl -s \
  http://localhost:9400/metrics
```

If GPU metrics are absent here, the problem is upstream of Prometheus.

Check:

```text
GPU
 ↓
Driver
 ↓
DCGM
 ↓
Exporter
```

---

## Metrics Exist but Prometheus Does Not See Them

Check:

```bash
kubectl -n monitoring get svc \
  dcgm-exporter
```

Then:

```bash
kubectl -n monitoring get servicemonitor \
  dcgm-exporter
```

Verify:

* Service selector
* Port name
* ServiceMonitor selector
* Namespace
* Helm release label

The ServiceMonitor expects:

```text
port: metrics
```

and the Service must expose a port with that exact name.

---

## Grafana Panels Are Empty

First test the PromQL query directly in Prometheus.

If Prometheus returns data but Grafana does not:

* Verify the Prometheus data source.
* Check dashboard time range.
* Check query variables.
* Verify panel query syntax.

Troubleshoot in this order:

```text
GPU
 ↓
DCGM Exporter
 ↓
Service
 ↓
ServiceMonitor
 ↓
Prometheus
 ↓
Grafana
```

---

## Exporter Enters CrashLoopBackOff

Run:

```bash
kubectl -n monitoring describe pod \
  <dcgm-exporter-pod>
```

Then:

```bash
kubectl -n monitoring logs \
  <dcgm-exporter-pod>
```

Verify:

* NVIDIA driver availability
* Container permissions
* GPU visibility
* DCGM compatibility
* Volume mount availability

---

## MIG Metrics Are Missing

If MIG is enabled, verify compatibility among:

```text
NVIDIA Driver
      ↓
DCGM
      ↓
Device Plugin
      ↓
DCGM Exporter
      ↓
Kubernetes
```

MIG visibility can vary according to software versions and configuration.

---

# Useful Diagnostic Commands

```bash
kubectl -n monitoring get pods \
  -l app=dcgm-exporter
```

```bash
kubectl -n monitoring logs \
  <dcgm-exporter-pod>
```

```bash
kubectl -n monitoring get svc \
  dcgm-exporter
```

```bash
kubectl -n monitoring get servicemonitor \
  dcgm-exporter
```

```bash
kubectl -n monitoring get prometheusrule \
  gpu-alerts
```

---

# Step 17: Clean Up

Remove the GPU alert rules:

```bash
kubectl delete \
  -f lab-07-gpu-alerts.yaml
```

Delete the ServiceMonitor:

```bash
kubectl delete \
  -f lab-07-dcgm-servicemonitor.yaml
```

Delete DCGM Exporter:

```bash
kubectl delete \
  -f lab-07-dcgm-exporter.yaml
```

Remove the Prometheus monitoring stack:

```bash
helm -n monitoring uninstall monitoring
```

Delete the namespace:

```bash
kubectl delete namespace monitoring
```

If the NVIDIA device plugin was installed only for this lab, you may remove it after confirming that no other workloads depend on it.

> **Important**
>
> Do not remove an existing GPU Operator or NVIDIA device-plugin installation that is used by other cluster workloads.

---

# Expected Results

At the end of the lab:

* Prometheus should show active DCGM Exporter targets.
* GPU utilization metrics should be available.
* GPU framebuffer memory metrics should be available.
* GPU temperature telemetry should be available where supported.
* GPU power telemetry should be available where supported.
* Grafana should display GPU dashboards.
* Prometheus should contain GPU-related alerting rules.
* Alertmanager should be available for alert-routing workflows.

The resulting architecture provides visibility into:

```text
GPU Health
GPU Utilization
GPU Memory
GPU Temperature
GPU Power
Cluster Resources
Workload Context
```

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Verified an NVIDIA GPU-enabled Kubernetes node.
* [ ] Verified `nvidia.com/gpu` capacity.
* [ ] Created the `monitoring` namespace.
* [ ] Installed `kube-prometheus-stack`.
* [ ] Verified Prometheus.
* [ ] Verified Grafana.
* [ ] Verified Alertmanager.
* [ ] Labeled GPU nodes.
* [ ] Deployed DCGM Exporter as a DaemonSet.
* [ ] Created the DCGM Exporter Service.
* [ ] Confirmed metrics on port `9400`.
* [ ] Created a `ServiceMonitor`.
* [ ] Confirmed Prometheus discovers the exporter.
* [ ] Queried GPU utilization using PromQL.
* [ ] Queried GPU memory utilization.
* [ ] Queried GPU temperature.
* [ ] Queried GPU power consumption.
* [ ] Accessed Grafana.
* [ ] Created GPU dashboard panels.
* [ ] Created GPU alerting rules.
* [ ] Verified the `PrometheusRule`.
* [ ] Investigated workload-level GPU labels where supported.
* [ ] Generated GPU activity and observed metric changes.
* [ ] Reviewed troubleshooting procedures.
* [ ] Removed lab resources when complete.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Explain how NVIDIA DCGM Exporter, Prometheus, Grafana, and Alertmanager work together.
* Deploy GPU telemetry collection across Kubernetes GPU nodes.
* Use a DaemonSet to run node-level monitoring software.
* Expose metrics through a Kubernetes Service.
* Configure automatic Prometheus discovery with `ServiceMonitor`.
* Query GPU telemetry using PromQL.
* Build dashboards for GPU utilization and health.
* Configure Prometheus GPU alerting rules.
* Understand how Alertmanager fits into an operational monitoring architecture.
* Associate accelerator usage with Kubernetes workloads where supported.
* Systematically troubleshoot the path from GPU hardware to monitoring dashboards.
* Use GPU telemetry as a foundation for capacity planning and AI infrastructure operations.

---

# Key Takeaway

**GPU clusters require accelerator-specific observability in addition to standard Kubernetes monitoring. NVIDIA DCGM Exporter provides GPU telemetry, Prometheus stores and queries the resulting time-series metrics, Grafana makes accelerator behavior visible through dashboards, and Alertmanager supports operational notification workflows.**

The complete monitoring path is:

```text
GPU
 ↓
DCGM
 ↓
DCGM Exporter
 ↓
ServiceMonitor
 ↓
Prometheus
 ↓
Grafana + Alerting
```

This architecture provides the foundation for monitoring production AI infrastructure and can later be extended with workload attribution, long-term metric storage, model-serving telemetry, distributed tracing, SLO-based alerts, and automated capacity management.
