# Hands-On Lab: AWS EC2 for AI Workloads

In this lab, you will launch a GPU-enabled Amazon EC2 instance, connect to it through SSH, verify access to the NVIDIA GPU, and run a simple PyTorch workload.

Using an AWS Deep Learning AMI simplifies the process because the environment includes many of the drivers, libraries, and machine learning frameworks commonly required for GPU workloads. This lab provides practical experience with cloud-based accelerated computing and demonstrates how an AI engineer can provision GPU infrastructure on demand.

## Goal

Launch an Amazon EC2 GPU instance and verify that PyTorch can use the GPU.

## Estimated Time

**60–90 minutes**

## Cost

GPU instances are billed while they are running and can cost significantly more than standard CPU instances. Pricing varies by instance type, Region, operating system, and purchasing option.

> **⚠️ Cost Warning**
>
> Check the current EC2 pricing before beginning the lab, and **stop or terminate the instance immediately after completing the exercises**.

---

## Prerequisites

Before starting the lab, you should have:

* An active AWS account
* Basic familiarity with Linux and the command line
* An SSH client
* Internet connectivity

macOS and Linux include SSH by default, while Windows users can use PowerShell or Windows Terminal.

The AWS CLI is optional because the entire lab can be completed through the AWS Management Console.

---

## Step 1: Create an SSH Key Pair

An EC2 key pair allows you to securely authenticate to the Linux instance without using a traditional password.

1. Open the **AWS Management Console** and navigate to **EC2 → Key Pairs**.
2. Choose **Create key pair**.
3. Configure the key pair:

| Setting                | Value        |
| ---------------------- | ------------ |
| **Name**               | `ai-keypair` |
| **Key pair type**      | RSA          |
| **Private key format** | `.pem`       |

4. Download the key and store it securely, such as:

```text
~/.ssh/ai-keypair.pem
```

On macOS or Linux, restrict access to the private key:

```bash
chmod 400 ~/.ssh/ai-keypair.pem
```

This permission prevents other users from reading the private key and is normally required when using the key with SSH.

> **Security:** Never commit your `.pem` private key to GitHub or another public repository.

---

## Step 2: Check the EC2 GPU Quota

GPU instance families are subject to EC2 service quotas, and some accounts may initially have insufficient quota to launch them.

Before creating the instance:

1. Open **Service Quotas**.
2. Locate the relevant EC2 quota for the GPU instance family you plan to use.
3. Verify that sufficient capacity is available.
4. Submit a quota-increase request if necessary.

Approval may not be immediate, so this step should ideally be completed before beginning the lab.

---

## Step 3: Launch the GPU Instance

Navigate to:

**EC2 → Instances → Launch instances**

Configure the instance with the following settings:

| Setting            | Configuration                                     |
| ------------------ | ------------------------------------------------- |
| **Name**           | `ai-ec2-lab`                                      |
| **AMI**            | Appropriate AWS Deep Learning AMI based on Ubuntu |
| **Instance Type**  | `g5.xlarge`, or another suitable GPU instance     |
| **Key Pair**       | `ai-keypair`                                      |
| **Security Group** | Allow SSH (TCP 22) from **My IP**                 |
| **Storage**        | Approximately 100 GB                              |

A `g5.xlarge` instance provides an NVIDIA A10G GPU and is suitable for introductory GPU experimentation.

Deep Learning AMI software configurations change over time, so verify the AMI description and included frameworks before launching the instance.

> **🔒 Security Note**
>
> Do not configure SSH access from `0.0.0.0/0` unless there is a specific reason to do so. Restricting SSH to your IP address significantly reduces unnecessary exposure.

Choose **Launch instance** and wait until the instance reaches the **Running** state and its status checks have completed.

---

## Step 4: Connect to the EC2 Instance

Select the instance and copy its **Public IPv4 address**.

From your local terminal, connect using the private key:

```bash
ssh -i ~/.ssh/ai-keypair.pem ubuntu@<INSTANCE_PUBLIC_IP>
```

Replace `<INSTANCE_PUBLIC_IP>` with the actual public IP address assigned to your instance.

After authentication succeeds, your terminal session will be running inside the GPU-enabled EC2 server.

---

## Step 5: Verify the NVIDIA GPU

Run:

```bash
nvidia-smi
```

The output should display information about the installed NVIDIA GPU, including:

* GPU model
* Driver version
* GPU memory
* GPU utilization
* Active processes

For a `g5.xlarge` instance, you should normally see an **NVIDIA A10G** GPU.

If `nvidia-smi` does not recognize a GPU, verify that you selected a GPU-enabled instance type and an appropriate Deep Learning AMI.

---

## Step 6: Activate the PyTorch Environment

AWS Deep Learning AMIs may provide preconfigured environments for supported machine learning frameworks.

First, inspect the available Conda environments:

```bash
conda env list
```

If an appropriate PyTorch environment is available, activate it using the environment name displayed by the command:

```bash
conda activate <PYTORCH_ENVIRONMENT>
```

The exact environment name can vary between Deep Learning AMI releases.

Using the environment provided with the AMI helps avoid unnecessary manual configuration of framework and GPU dependencies.

---

## Step 7: Run a PyTorch GPU Test

Run the following Python program to verify that PyTorch can access the GPU and perform a simple matrix multiplication:

```bash
python - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.rand((2048, 2048), device="cuda")
    y = x @ x

    print("Matrix multiplication successful:", y.shape)
PY
```

A successful configuration should produce output similar to:

```text
CUDA available: True
PyTorch version: <version>
GPU: NVIDIA A10G
Matrix multiplication successful: torch.Size([2048, 2048])
```

The most important result is:

```text
CUDA available: True
```

This confirms that PyTorch can communicate with the NVIDIA GPU and execute GPU-accelerated operations.

---

## Step 8: Optional — Run JupyterLab Securely

You can also use the EC2 instance as a remote Jupyter development environment.

If JupyterLab is not already installed in the active environment, install it:

```bash
pip install jupyterlab
```

Start JupyterLab on the EC2 instance:

```bash
jupyter lab --no-browser --port 8888
```

From a separate terminal on your local computer, establish an SSH tunnel:

```bash
ssh -i ~/.ssh/ai-keypair.pem \
    -N -L 8888:localhost:8888 \
    ubuntu@<INSTANCE_PUBLIC_IP>
```

Then open the following address in your local browser:

```text
http://localhost:8888
```

Enter the Jupyter token displayed in the EC2 terminal.

The SSH tunnel allows you to access JupyterLab without exposing port `8888` directly to the Internet.

---

## Step 9: Clean Up the Environment

> **⚠️ Important: GPU Instances Can Generate Charges Quickly**

If you plan to continue using the instance, **stop it** when it is not needed.

If the lab is complete, **terminate the instance** and review associated resources that may continue generating charges, including:

* EBS volumes
* EBS snapshots
* Elastic IP addresses
* Other attached or reserved resources

Remember that stopping an instance generally stops instance compute charges, but attached storage and certain other resources may continue to incur charges.

---

## Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Launched a GPU-enabled EC2 instance
* [ ] Connected securely using SSH
* [ ] Used `nvidia-smi` to identify the GPU
* [ ] Confirmed that PyTorch reports `CUDA available: True`
* [ ] Executed a matrix operation on the GPU
* [ ] Optionally accessed JupyterLab through an SSH tunnel
* [ ] Stopped or terminated resources that are no longer required

---

## What You Learned

This lab demonstrated the basic workflow for provisioning **GPU-accelerated AI infrastructure in AWS**.

You created a cloud GPU server, established secure administrative access, verified the NVIDIA GPU environment, and executed an AI workload using PyTorch.

These same foundational concepts extend to larger environments containing multiple GPUs, distributed training clusters, containerized AI workloads, and production inference systems.
