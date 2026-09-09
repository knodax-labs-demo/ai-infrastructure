# Hands-On Lab: Compare GPU Cost and Performance Across Clouds

In this lab, you will deploy GPU-enabled virtual machines on **AWS, Google Cloud, and Microsoft Azure**, run the same PyTorch benchmark on each platform, and compare their performance and cost.

The objective is not simply to determine which cloud is fastest, but to understand how **GPU architecture, VM configuration, pricing, and workload characteristics** influence infrastructure decisions. By keeping the benchmark consistent across all three environments, you can make a more meaningful comparison of performance and cost efficiency.

## Goal

Compare GPU performance and cost across **AWS, Google Cloud, and Microsoft Azure** using a consistent PyTorch workload.

## Estimated Time

**90–120 minutes**

## Cost

GPU instances can generate charges quickly. Actual costs vary significantly by Region, VM configuration, operating system, pricing model, and current cloud-provider pricing.

> **⚠️ Cost Warning**
>
> Check current pricing before launching resources, and **stop or terminate all resources immediately after completing the lab**.

---

## Prerequisites

Before beginning the lab, make sure you have:

* An active **AWS account** with billing enabled
* An active **Google Cloud account** with billing enabled
* An active **Microsoft Azure account** with billing enabled
* Sufficient GPU quota on each platform
* Basic familiarity with Linux
* An SSH client
* Basic experience with `nvidia-smi` and PyTorch

To make the comparison meaningful, use the **same benchmark script and equivalent software configuration** on all three platforms.

---

## Step 1: Define the Benchmark

For this lab, you will use PyTorch to perform repeated matrix multiplication on the GPU.

Matrix multiplication is a fundamental operation in neural networks and provides a simple way to demonstrate GPU computational performance. This is intentionally a lightweight benchmark rather than a comprehensive measure of overall AI performance.

Create a file named:

```text
gpu_benchmark.py
```

Add the following code:

```python
import time
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA-compatible GPU is not available.")

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)

x = torch.rand((10000, 10000), device=device)

# Warm up the GPU
y = x @ x
torch.cuda.synchronize()

# Run benchmark
start = time.perf_counter()

for _ in range(10):
    y = x @ x

torch.cuda.synchronize()
end = time.perf_counter()

print("Time taken:", round(end - start, 3), "seconds")
```

The initial matrix multiplication serves as a warm-up operation.

`torch.cuda.synchronize()` ensures that asynchronous GPU operations have completed before timing is measured, making the comparison more meaningful.

> **Benchmarking Note**
>
> This benchmark measures a specific GPU compute operation. It should not be interpreted as a comprehensive benchmark of overall AI or cloud-platform performance.

---

## Step 2: Launch an AWS GPU Instance

On AWS, launch an appropriate GPU-enabled Amazon EC2 instance.

For example, you can use a **G5 instance**, such as:

```text
g5.xlarge
```

This configuration provides an NVIDIA A10G GPU and can be used for the benchmark.

Select a current AWS Deep Learning AMI based on Ubuntu when available, as these images simplify GPU driver and framework configuration.

Configure:

* GPU-enabled EC2 instance
* Ubuntu-based Deep Learning AMI
* SSH access from **My IP**
* Sufficient storage for the lab

After connecting to the instance, verify the GPU:

```bash
nvidia-smi
```

Activate or configure an appropriate PyTorch environment and run:

```bash
python gpu_benchmark.py
```

Record:

* AWS Region
* EC2 instance type
* GPU model
* PyTorch version
* Benchmark execution time
* Current hourly price

---

## Step 3: Launch a Google Cloud GPU VM

On Google Cloud, create a GPU-enabled Compute Engine VM.

A configuration based on the **G2 machine family with an NVIDIA L4 GPU** can be used for this comparison, subject to availability and quota in your selected Region.

Use an appropriate Ubuntu or GPU-enabled image and install the required NVIDIA drivers if they are not already included.

Verify the GPU:

```bash
nvidia-smi
```

Install or activate a compatible PyTorch environment and run exactly the same benchmark:

```bash
python gpu_benchmark.py
```

Record:

* Google Cloud Region
* Machine type
* GPU model
* PyTorch version
* Benchmark execution time
* Current hourly price

---

## Step 4: Launch an Azure GPU VM

On Microsoft Azure, create an appropriate **N-series GPU VM**.

Depending on current availability and Region, select a VM equipped with an NVIDIA GPU suitable for the benchmark.

Using an Azure Data Science or machine-learning-oriented VM image can simplify the installation of NVIDIA drivers and AI frameworks.

After connecting to the VM, verify the GPU:

```bash
nvidia-smi
```

Run the same PyTorch benchmark:

```bash
python gpu_benchmark.py
```

Record:

* Azure Region
* VM size
* GPU model
* PyTorch version
* Benchmark execution time
* Current hourly price

> **Important**
>
> GPU VM families, accelerator models, software images, availability, and pricing change over time. Verify the current configuration and price offered by each cloud provider rather than relying on fixed values.

---

## Step 5: Record the Results

Enter the results from your experiments in the following table:

| Cloud            | Instance/VM | GPU        | Benchmark Time | Cost/Hour |
| ---------------- | ----------- | ---------- | -------------: | --------: |
| **AWS**          | __________  | __________ |      _____ sec |    $_____ |
| **Google Cloud** | __________  | __________ |      _____ sec |    $_____ |
| **Azure**        | __________  | __________ |      _____ sec |    $_____ |

For greater consistency, run the benchmark **three times on each VM** and calculate the average runtime.

You can use:

```text
Average Runtime = (Run 1 + Run 2 + Run 3) / 3
```

Also record the Region used because cloud pricing and hardware availability can vary by location.

### Optional Detailed Results

| Cloud            |     Run 1 |     Run 2 |     Run 3 |   Average |
| ---------------- | --------: | --------: | --------: | --------: |
| **AWS**          | _____ sec | _____ sec | _____ sec | _____ sec |
| **Google Cloud** | _____ sec | _____ sec | _____ sec | _____ sec |
| **Azure**        | _____ sec | _____ sec | _____ sec | _____ sec |

---

## Step 6: Compare Cost and Performance

The fastest GPU is not necessarily the most economical choice.

A more expensive GPU may complete a workload quickly enough to produce a lower total cost, while a less expensive GPU may provide better value for workloads that do not require maximum performance.

For a simple comparison, estimate the cost of running the benchmark workload using:

```text
Estimated Workload Cost =
(Benchmark Time in Seconds / 3,600) × Hourly VM Cost
```

For example, if a benchmark takes `30` seconds on a VM costing `$1.50/hour`:

```text
Estimated Workload Cost =
(30 / 3,600) × 1.50

= $0.0125
```

You can also compare relative performance per dollar across the three configurations.

However, this matrix multiplication benchmark measures only one aspect of GPU performance. Real AI workloads can also be affected by:

* GPU memory capacity
* Memory bandwidth
* GPU architecture
* CPU resources
* System memory
* Storage throughput
* Networking
* Framework optimization
* Multi-GPU communication
* Distributed-training capabilities

---

## Step 7: Analyze Your Results

Review your measurements and answer the following questions:

1. Which cloud configuration provided the best **absolute performance**?
2. Which configuration provided the best **cost-to-performance ratio**?
3. How significant was the performance difference between the GPUs?
4. Would your choice change for a workload requiring substantially more GPU memory?
5. Would your choice change for a multi-GPU workload?
6. Was your preferred GPU available in all desired Regions?
7. Would Spot or preemptible capacity make sense for this workload?

Cloud availability is an important consideration because the preferred GPU type may not always be available in the desired Region.

Spot or preemptible capacity may significantly reduce costs for fault-tolerant workloads, but these resources can be interrupted.

---

## Step 8: Lab Deliverables

At the end of the lab, you should have:

* [ ] `nvidia-smi` output identifying the GPU used on AWS
* [ ] `nvidia-smi` output identifying the GPU used on Google Cloud
* [ ] `nvidia-smi` output identifying the GPU used on Azure
* [ ] PyTorch benchmark results for all three clouds
* [ ] A completed cost-and-performance comparison table
* [ ] Average runtime based on multiple benchmark runs
* [ ] A short cost-and-performance analysis

Your analysis should answer:

> **Which configuration delivered the fastest benchmark result?**

> **Which configuration provided the best cost-to-performance ratio?**

> **Which configuration would you select for your workload, and why?**

---

## Step 9: Clean Up Resources

> **⚠️ Important: Stop GPU Resources When Finished**
>
> GPU resources are significantly more expensive than typical general-purpose VMs.

After collecting your results:

1. **Stop or terminate the AWS GPU instance.**
2. **Stop or delete the Google Cloud GPU VM.**
3. **Stop or delete the Azure GPU VM.**
4. Review attached disks and persistent storage.
5. Review snapshots.
6. Review reserved or static IP addresses.
7. Check for other associated resources.
8. Review the billing or cost-management dashboard for each provider.

Remember that stopping a VM may stop compute charges while **storage, snapshots, IP addresses, and other resources may continue generating charges**.

---

## What You Learned

This lab demonstrated that selecting AI compute infrastructure requires more than comparing GPU specifications or hourly prices.

You deployed GPU resources across **AWS, Google Cloud, and Microsoft Azure**, executed a consistent workload, measured performance, and evaluated the relationship between **execution time and cost**.

You also saw why benchmarking representative workloads is important before committing to a larger infrastructure design. In production environments, these measurements can be expanded to include memory utilization, throughput, storage performance, networking, availability, scalability, and total cost of ownership.
