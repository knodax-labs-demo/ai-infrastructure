# Hands-On Lab: Spin Up Your First AI GPU VM

In this lab, you will launch a cloud virtual machine equipped with an NVIDIA GPU and configure it for basic AI workloads. You will connect to the VM, verify that the operating system can access the GPU, install PyTorch, and run a small GPU computation from Python.

This provides a simple introduction to the type of accelerated computing environment used for AI training and inference.

You can complete the lab using either:

* **Amazon Web Services (AWS)**
* **Google Cloud**

> **Note:** Exact instance types and console options may change over time, but the underlying workflow remains the same.

## Lab Goal

Launch a GPU-enabled cloud VM and verify that Python and PyTorch can successfully use the GPU.

## Estimated Time

**45–90 minutes**, depending on account setup and GPU quota availability.

## Pricing and Cost Considerations

GPU-enabled virtual machines are considerably more expensive than standard CPU instances, so cost awareness is important in this lab.

Cloud providers generally charge for GPU compute based on how long the instance runs, with additional charges possible for:

* Storage
* Snapshots
* Static IP addresses
* Network data transfer

AWS EC2 On-Demand Instances charge for compute without requiring a long-term commitment, while Google Cloud provides both standard and lower-cost Spot VM options.

For a short lab, expect compute costs to range from **less than $1 to several dollars per hour**, depending on the GPU selected and pricing model.

> **⚠️ Cost Guardrail**
>
> Use the smallest suitable GPU instance available. Complete the exercises in one session and **stop or terminate the VM immediately afterward**.
>
> Stopping a VM usually stops compute charges, but attached storage and certain other resources may continue generating charges.

## Prerequisites

Before starting, make sure you have:

* An active **AWS or Google Cloud account** with billing enabled
* Sufficient **GPU service quota**
* An **SSH client**
* A stable Internet connection

macOS and Linux include SSH by default. Windows users can use PowerShell or Windows Terminal.

---

# Option A: AWS EC2

AWS provides GPU-enabled Amazon EC2 instances that can be used for AI training and inference.

For this lab, using an AWS **Deep Learning AMI (DLAMI)** simplifies setup because these images are designed specifically for machine learning environments and can include NVIDIA drivers, CUDA-related components, and popular development tools.

## Step 1: Create an SSH Key Pair

Open the AWS Management Console and navigate to:

**EC2 → Network & Security → Key Pairs → Create key pair**

Enter a name such as:

```text
ai-lab-key
```

Select **RSA** as the key type and download the private key in the appropriate format.

The `.pem` format works with OpenSSH on macOS, Linux, and modern Windows environments.

> **Security:** Store the private key securely. AWS does not retain a downloadable copy after creation.

## Step 2: Check Your GPU Quota

Before launching the instance, verify that your AWS account has sufficient quota for GPU-based EC2 instances.

GPU instance families are subject to separate vCPU quotas, and new accounts may initially have little or no GPU capacity available.

If necessary:

1. Open **Service Quotas**.
2. Locate the relevant EC2 GPU quota.
3. Request an increase.
4. Wait for approval before continuing.

## Step 3: Launch the GPU Instance

Navigate to:

**EC2 → Instances → Launch instances**

Configure the instance with settings similar to the following:

| Setting           | Configuration                                   |
| ----------------- | ----------------------------------------------- |
| **Name**          | `ai-lab-aws`                                    |
| **AMI**           | Ubuntu-based Deep Learning AMI with GPU support |
| **Instance Type** | Small GPU instance available to your account    |
| **Key Pair**      | `ai-lab-key`                                    |
| **SSH**           | Port 22 from **My IP**                          |
| **Storage**       | Approximately 50–100 GB                         |

Select an instance containing an NVIDIA GPU, such as an appropriate instance from a GPU-enabled EC2 family available in your region.

> **Security:** Avoid allowing SSH access from `0.0.0.0/0`. Restricting port 22 to **My IP** reduces unnecessary exposure.

Launch the instance and wait until its status changes to **Running**.

## Step 4: Connect to the Instance

Locate the instance's **Public IPv4 address** or public DNS name.

On macOS or Linux, protect the downloaded private key:

```bash
chmod 400 ~/Downloads/ai-lab-key.pem
```

Connect to the Ubuntu instance:

```bash
ssh -i ~/Downloads/ai-lab-key.pem ubuntu@<PUBLIC_IP>
```

Replace `<PUBLIC_IP>` with the public address assigned to your instance.

## Step 5: Verify the GPU

Run:

```bash
nvidia-smi
```

The command should display information about the NVIDIA GPU, including:

* GPU model
* Driver version
* GPU memory
* Memory utilization
* Running GPU processes

If `nvidia-smi` runs successfully, the operating system can communicate with the GPU.

## Step 6: Create a Python Environment

Check whether Conda is available:

```bash
conda --version
```

If available, create a dedicated environment:

```bash
conda create -n ai-lab python=3.11 -y
conda activate ai-lab
```

Using a dedicated environment isolates the lab's Python packages from other software installed on the VM.

## Step 7: Install PyTorch

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install a PyTorch build appropriate for the NVIDIA environment on the VM. Because supported PyTorch and CUDA combinations change over time, use the installation command recommended by the current PyTorch documentation.

Verify that PyTorch can detect the GPU:

```bash
python - <<'PY'
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

A successful result should include:

```text
CUDA available: True
```

## Step 8: Run a GPU Computation

Perform a small matrix multiplication directly on the GPU:

```bash
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available")

device = torch.device("cuda")

x = torch.rand((1024, 1024), device=device)
y = torch.mm(x, x)

print("GPU:", torch.cuda.get_device_name(0))
print("Result shape:", y.shape)
print("GPU computation successful!")
PY
```

If the command completes successfully, PyTorch has allocated tensors on the GPU and executed the matrix multiplication using GPU acceleration.

> **Success:** You have now created and validated your first cloud-based AI GPU environment.

---

# Option B: Google Cloud

The same basic exercise can be completed using Google Cloud Compute Engine.

The workflow is similar:

**Create GPU VM → Configure NVIDIA Driver → Connect → Install PyTorch → Verify GPU**

## Step 1: Prepare the Google Cloud Project

1. Open Google Cloud Console.
2. Create or select a project.
3. Make sure billing is enabled.
4. Enable the **Compute Engine API**.
5. Verify the GPU quota for your intended region.

GPU availability varies by region and zone, and a quota increase may be required.

## Step 2: Create a GPU VM

Navigate to:

**Compute Engine → VM instances → Create instance**

Configure a GPU-enabled machine using an available NVIDIA accelerator.

For an introductory lab:

* Choose the smallest practical GPU configuration.
* Use a supported Ubuntu image.
* Allocate approximately **50–100 GB** of boot-disk storage.
* Limit network access to what is required for the lab.

Create the instance and wait for it to enter the **Running** state.

## Step 3: Install and Verify the NVIDIA Driver

Follow the current Google Cloud instructions for installing the appropriate NVIDIA driver for your selected accelerator.

After installation, connect to the VM and run:

```bash
nvidia-smi
```

You should see the NVIDIA accelerator together with its driver and memory information.

## Step 4: Install PyTorch and Test the GPU

Create a Python environment and install the PyTorch build recommended for your current driver environment.

Then run:

```bash
python - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.rand((1024, 1024), device="cuda")
    y = torch.mm(x, x)

    print("Result shape:", y.shape)
    print("GPU computation successful!")
PY
```

If `CUDA available` reports `True` and the matrix multiplication completes successfully, your Google Cloud GPU environment is ready for AI workloads.

---

# Optional: Access Jupyter Through an SSH Tunnel

You can run Jupyter on the VM without exposing the notebook server directly to the Internet.

Install and start Jupyter on the VM:

```bash
pip install jupyter
jupyter notebook --no-browser --port 8888
```

From your local computer, create an SSH tunnel:

```bash
ssh -i ~/Downloads/ai-lab-key.pem \
    -N -L 8888:localhost:8888 \
    ubuntu@<PUBLIC_IP>
```

Then open:

```text
http://localhost:8888
```

Enter the Jupyter token displayed in the VM terminal.

This approach keeps the notebook service bound to the VM while securely forwarding traffic through SSH.

---

# Troubleshooting

### `Permission denied (publickey)`

Verify that:

* You are using the correct private key.
* You are using the correct username.
* The private key has restrictive permissions.

For many Ubuntu-based AWS images, the default username is `ubuntu`.

### `nvidia-smi` Is Unavailable

Confirm that:

* You launched a GPU-enabled VM.
* The NVIDIA driver is installed correctly.

A CPU-only VM will not expose an NVIDIA GPU.

### PyTorch Reports `CUDA available: False`

The installed PyTorch build may not be compatible with the current NVIDIA environment.

Verify the driver and install a supported PyTorch configuration.

### GPU Instance Cannot Be Launched

Check:

* Account GPU quota
* Regional GPU availability
* Selected GPU type
* Available zones

You may need to request a quota increase or select another supported region, zone, or GPU type.

---

# Clean Up Your Resources

> **⚠️ Important: Avoid Unexpected Charges**
>
> GPU resources can continue generating charges while running.

After completing the lab:

1. **Stop the VM** if you plan to use it again soon.
2. **Terminate or delete the VM** if you no longer need it.
3. Check for persistent disks.
4. Check for snapshots.
5. Check for reserved resources.
6. Check for static public IP addresses.

Keep downloaded SSH private keys secure and **never store them in a public GitHub repository**.

If a private key is accidentally exposed, replace it rather than continuing to use it.

---

# Lab Summary

In this lab, you:

* Launched a GPU-enabled cloud VM
* Connected remotely using SSH
* Verified the NVIDIA GPU using `nvidia-smi`
* Configured a Python environment
* Confirmed that PyTorch could access the GPU
* Executed a matrix multiplication directly on the accelerator

The same fundamental workflow applies to much larger AI infrastructure environments:

**Provision Compute → Verify Accelerator → Configure Software → Validate Workload**
