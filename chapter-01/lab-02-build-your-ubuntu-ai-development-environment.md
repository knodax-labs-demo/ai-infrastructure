# Hands-On Lab: Build Your Ubuntu AI Development Environment

In this lab, you will configure an **Ubuntu Linux environment for AI development and infrastructure work**.

You will install essential Linux utilities, configure Git, create an isolated Python environment, install core AI and data-science libraries, install PyTorch and JupyterLab, and verify that the environment operates correctly.

If the system contains an NVIDIA GPU, you will also verify or install the NVIDIA driver, confirm CUDA access through PyTorch, install Docker, and optionally configure **NVIDIA Container Toolkit** so containers can access the GPU.

The resulting environment provides a reusable foundation for many of the labs throughout this book.

The complete environment stack is:

```text
Ubuntu Linux
     ↓
Development Tools
     ↓
Python Virtual Environment
     ↓
PyTorch + AI Libraries
     ↓
JupyterLab
     ↓
Docker
     ↓
Optional NVIDIA GPU
     ↓
GPU-Enabled Containers
```

---

## Lab Objectives

By completing this lab, you will learn how to:

* Inspect an Ubuntu AI development system.
* Install common Linux development and administration tools.
* Configure basic host firewall protection.
* Configure Git and SSH credentials.
* Create an isolated Python environment.
* Install common AI and data-science packages.
* Configure NVIDIA GPU drivers when required.
* Install and verify PyTorch.
* Verify CUDA access from Python.
* Run JupyterLab securely on a remote server.
* Install and test Docker.
* Enable NVIDIA GPU access inside containers.
* Create a reusable AI project directory.
* Troubleshoot AI environments layer by layer.

---

## Estimated Time

**Approximately 60–90 minutes**

---

## Prerequisites

You need:

* Ubuntu 22.04 or newer
* A physical machine or VM
* A user account with `sudo` privileges
* Internet connectivity
* Optional NVIDIA GPU

> **Version Note**
>
> GPU drivers, PyTorch builds, Docker packages, CUDA compatibility, and NVIDIA Container Toolkit versions change frequently. Use the currently supported installation instructions for your Ubuntu release and hardware rather than hard-coding old package versions.
>
> Cloud GPU images may already contain NVIDIA drivers, CUDA libraries, PyTorch, or other AI software. Verify what is already installed before modifying the environment.

---

# Step 1: Inspect the System

Check the Ubuntu version:

```bash
lsb_release -a
```

Check the current user:

```bash
whoami
```

Look for an NVIDIA GPU:

```bash
lspci | grep -i nvidia || echo "No NVIDIA GPU detected"
```

If no GPU is detected, you can still complete all CPU-based portions of the lab.

---

# Step 2: Understand the Environment Layers

An AI development system is composed of several separate layers:

```text
Physical / Virtual Hardware
          ↓
Operating System
          ↓
NVIDIA Driver
          ↓
Python Environment
          ↓
AI Framework
          ↓
Application
```

Containers add another layer:

```text
Hardware
   ↓
Host Driver
   ↓
Container Runtime
   ↓
NVIDIA Container Runtime
   ↓
Container
   ↓
AI Framework
```

Understanding these layers is important when troubleshooting.

---

# Step 3: Update Ubuntu

Refresh the package index:

```bash
sudo apt update
```

Upgrade installed packages:

```bash
sudo apt -y upgrade
```

If major system components or the kernel were upgraded, reboot:

```bash
sudo reboot
```

Reconnect after the machine restarts.

---

# Step 4: Install Essential Linux Tools

Install common development and administration utilities:

```bash
sudo apt -y install \
  build-essential \
  git \
  curl \
  wget \
  unzip \
  zip \
  tar \
  ca-certificates \
  htop \
  iotop \
  tree \
  tmux \
  pkg-config \
  software-properties-common \
  nano \
  vim \
  gnupg \
  lsb-release
```

These packages provide:

| Tool              | Purpose                        |
| ----------------- | ------------------------------ |
| `build-essential` | Compiler and build tools       |
| `git`             | Source control                 |
| `curl`, `wget`    | Network downloads              |
| `htop`            | Interactive process monitoring |
| `iotop`           | Disk I/O monitoring            |
| `tmux`            | Persistent terminal sessions   |
| `tree`            | Directory visualization        |
| `nano`, `vim`     | Text editors                   |
| `gnupg`           | Package-signing support        |

---

# Step 5: Configure Basic Firewall Protection

For a remotely accessed Ubuntu server, you can use **UFW** for host-level firewall protection.

If you are connected through SSH, allow SSH **before** enabling the firewall:

```bash
sudo ufw allow OpenSSH
```

Enable UFW:

```bash
sudo ufw enable
```

Check status:

```bash
sudo ufw status
```

> **Important**
>
> Cloud infrastructure commonly has additional network controls such as AWS security groups, Azure network security groups, or Google Cloud firewall rules. UFW does not replace those controls.

Do not expose JupyterLab directly to the public Internet unless it is properly secured.

---

# Step 6: Configure Git

Verify Git:

```bash
git --version
```

Configure your name:

```bash
git config --global user.name "Your Name"
```

Configure your email:

```bash
git config --global user.email "you@example.com"
```

Verify:

```bash
git config --global --list
```

---

# Step 7: Configure an SSH Key for Git

Generate an Ed25519 key:

```bash
ssh-keygen \
  -t ed25519 \
  -C "you@example.com"
```

Start the SSH agent:

```bash
eval "$(ssh-agent -s)"
```

Add the private key:

```bash
ssh-add ~/.ssh/id_ed25519
```

Display the public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Register the **public key** with your Git hosting service when needed.

> Never share:
>
> ```text
> ~/.ssh/id_ed25519
> ```
>
> That file contains your private key.

---

# Step 8: Install Python

Install Python and virtual-environment support:

```bash
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip
```

Verify:

```bash
python3 --version
```

---

# Step 9: Create an Isolated Python Environment

Create:

```bash
python3 -m venv ~/ai-env
```

Activate it:

```bash
source ~/ai-env/bin/activate
```

Verify:

```bash
python --version
```

and:

```bash
which python
```

The path should point into:

```text
~/ai-env/
```

rather than the operating-system Python installation.

---

# Step 10: Understand Why Virtual Environments Matter

Without isolation:

```text
Operating-System Python
        ↓
Package A
Package B
Package C
        ↓
Potential Dependency Conflicts
```

With a virtual environment:

```text
Ubuntu Python
      │
      └── ai-env/
            ├── PyTorch
            ├── pandas
            ├── Jupyter
            └── Project Dependencies
```

This makes environments easier to reproduce and maintain.

---

# Step 11: Upgrade Python Packaging Tools

Run:

```bash
python -m pip install \
  --upgrade \
  pip \
  setuptools \
  wheel
```

Verify:

```bash
python -m pip --version
```

---

# Step 12: Install the Core AI Development Stack

Install:

```bash
python -m pip install \
  numpy \
  pandas \
  scipy \
  scikit-learn \
  matplotlib \
  jupyterlab \
  ipywidgets \
  tqdm
```

This gives you a practical starter environment for:

* Numerical computing
* Data analysis
* Machine learning
* Visualization
* Notebook development

Avoid installing every possible AI package into one environment.

A better principle is:

```text
Install What the Project Needs
           ↓
Keep Environment Small
           ↓
Improve Reproducibility
```

---

# Step 13: Check for an Existing NVIDIA Driver

If the machine has an NVIDIA GPU, run:

```bash
nvidia-smi
```

If this command already works and displays the GPU correctly, **do not reinstall the NVIDIA driver unnecessarily**.

Typical information includes:

```text
GPU Model
Driver Version
GPU Memory
GPU Utilization
Running Processes
```

---

# Step 14: Install the NVIDIA Driver When Required

Skip this step if `nvidia-smi` already works.

Install Ubuntu's driver utilities:

```bash
sudo apt install -y \
  ubuntu-drivers-common
```

Inspect available drivers:

```bash
ubuntu-drivers devices
```

Install Ubuntu's recommended driver:

```bash
sudo ubuntu-drivers install
```

Reboot:

```bash
sudo reboot
```

After reconnecting:

```bash
nvidia-smi
```

---

# Step 15: Understand Driver and CUDA Compatibility

A GPU AI environment involves several compatibility layers:

```text
NVIDIA GPU
    ↓
NVIDIA Driver
    ↓
CUDA Runtime Requirements
    ↓
PyTorch Build
    ↓
Python Application
```

A common misconception is that you must always install the complete CUDA Toolkit on the host before using PyTorch.

That is not necessarily required because PyTorch distributions may include the CUDA runtime components required by the framework.

The host still requires a compatible NVIDIA driver.

---

# Step 16: Install PyTorch

Activate your environment if necessary:

```bash
source ~/ai-env/bin/activate
```

For CPU-only experimentation:

```bash
python -m pip install \
  torch \
  torchvision \
  torchaudio
```

For NVIDIA GPU environments, use the installation command recommended by the **current PyTorch installation documentation** for your:

```text
Operating System
Python Version
Package Manager
CUDA Environment
```

This avoids embedding an outdated CUDA wheel version in the lab.

---

# Step 17: Verify PyTorch

Run:

```bash
python - <<'PY'
import torch

print(
    "PyTorch version:",
    torch.__version__
)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    x = torch.rand(
        (2048, 2048),
        device="cuda"
    )

    y = x @ x

    torch.cuda.synchronize()

    print(
        "GPU matrix multiplication "
        "successful:",
        y.shape
    )
PY
```

A correctly configured GPU environment should report:

```text
CUDA available: True
```

and display the GPU name.

---

# Step 18: Understand What the PyTorch Test Proves

The test verifies several layers simultaneously:

```text
Python
  ↓
PyTorch Import
  ↓
CUDA Detection
  ↓
GPU Allocation
  ↓
GPU Computation
```

Checking only:

```bash
nvidia-smi
```

does not prove that PyTorch can use the GPU.

Likewise:

```text
CUDA available: True
```

provides stronger application-level validation.

---

# Step 19: Start JupyterLab

Verify:

```bash
jupyter lab --version
```

Start JupyterLab:

```bash
jupyter lab \
  --no-browser \
  --port 8888
```

Jupyter prints a local URL containing an authentication token.

---

# Step 20: Access Remote JupyterLab Securely

If Ubuntu is running remotely, keep Jupyter accessible only through the remote host and establish a tunnel **from your local computer**:

```bash
ssh -N \
  -L 8888:localhost:8888 \
  <user>@<server-ip>
```

Then open:

```text
http://localhost:8888
```

in your local browser.

The connection becomes:

```text
Local Browser
     ↓
localhost:8888
     ↓
SSH Tunnel
     ↓
Remote localhost:8888
     ↓
JupyterLab
```

This is safer than exposing Jupyter's port directly to the Internet.

---

# Step 21: Test JupyterLab

Create a notebook and run:

```python
import pandas as pd
import torch

print(
    "CUDA available:",
    torch.cuda.is_available()
)

print(
    "pandas:",
    pd.__version__
)
```

If a GPU is available:

```python
if torch.cuda.is_available():
    print(
        torch.cuda.get_device_name(0)
    )
```

---

# Step 22: Install Docker

Containers provide reproducible application environments.

Use Docker's currently supported Ubuntu installation procedure rather than relying on an old package sequence embedded permanently in the lab.

After installation:

```bash
docker --version
```

Verify the engine:

```bash
sudo docker run hello-world
```

A successful result confirms:

```text
Docker Client
     ↓
Docker Daemon
     ↓
Pull Container Image
     ↓
Create Container
     ↓
Execute Container
```

---

# Step 23: Understand Docker Permissions

Docker can optionally be configured so commands do not require:

```text
sudo
```

However, membership in the Docker group grants significant control over the host.

Treat:

```text
docker group membership
```

as a privileged configuration decision rather than merely a convenience setting.

After configuring permissions, test:

```bash
docker run hello-world
```

---

# Step 24: Understand Containers for AI Development

Without containers:

```text
Host OS
  ↓
Python Environment
  ↓
Application
```

With Docker:

```text
Host OS
   ↓
Docker
   ↓
Container Image
   ↓
Python + Dependencies
   ↓
AI Application
```

Containers improve reproducibility by packaging dependencies with the application.

---

# Step 25: Enable NVIDIA GPUs in Containers

Complete this section only if:

```bash
nvidia-smi
```

works successfully on the host.

Install **NVIDIA Container Toolkit** using NVIDIA's currently supported installation procedure for your Ubuntu version.

Configure Docker:

```bash
sudo nvidia-ctk \
  runtime configure \
  --runtime=docker
```

Restart Docker:

```bash
sudo systemctl restart docker
```

---

# Step 26: Test GPU Access from a Container

Run a currently compatible NVIDIA CUDA container image:

```bash
docker run \
  --rm \
  --gpus all \
  <compatible-nvidia-cuda-image> \
  nvidia-smi
```

Expected architecture:

```text
Host NVIDIA GPU
       ↓
Host NVIDIA Driver
       ↓
NVIDIA Container Toolkit
       ↓
Docker Runtime
       ↓
CUDA Container
       ↓
nvidia-smi
```

If the GPU appears inside the container, containerized applications can access the host GPU.

---

# Step 27: Understand Host vs. Container CUDA

A useful mental model is:

```text
HOST
├── NVIDIA GPU
├── NVIDIA Driver
└── Docker Runtime

CONTAINER
├── CUDA User-Space Libraries
├── AI Framework
└── Application
```

The NVIDIA Container Toolkit connects the two environments.

This separation is important when debugging GPU containers.

---

# Step 28: Create a Reusable AI Project Structure

Create:

```bash
mkdir -p \
  ~/projects/ai-starter/{data,notebooks,scripts,models,logs}
```

Move into it:

```bash
cd ~/projects/ai-starter
```

Inspect:

```bash
tree
```

Expected:

```text
ai-starter/
├── data/
├── notebooks/
├── scripts/
├── models/
└── logs/
```

---

# Step 29: Understand the Directory Structure

The directories have different roles:

| Directory    | Purpose                       |
| ------------ | ----------------------------- |
| `data/`      | Input or processed datasets   |
| `notebooks/` | Jupyter notebooks             |
| `scripts/`   | Python and automation scripts |
| `models/`    | Model artifacts               |
| `logs/`      | Runtime and experiment logs   |

Separating artifacts makes projects easier to navigate and automate.

---

# Step 30: Use tmux for Remote Development

Start a persistent session:

```bash
tmux new -s ai-work
```

Detach with:

```text
Ctrl-b
then d
```

Reattach:

```bash
tmux attach \
  -t ai-work
```

The workflow is:

```text
SSH Session
    ↓
tmux
    ↓
Long-Running Command
```

If SSH disconnects:

```text
Process Continues
```

and you can reconnect later.

---

# Step 31: Know When Not to Use tmux

`tmux` is useful for:

```text
Development
Testing
Administration
Small Experiments
```

For production workloads, prefer:

```text
Kubernetes
Slurm
Batch Scheduler
Managed Training Platform
Workflow Orchestrator
```

These systems provide better lifecycle management and recovery.

---

# Step 32: Verify the Complete Environment

Check Python:

```bash
python --version
```

Check Git:

```bash
git --version
```

Check Jupyter:

```bash
jupyter lab --version
```

Check Docker:

```bash
docker --version
```

For a GPU environment:

```bash
nvidia-smi
```

Repeat the PyTorch CUDA test.

If NVIDIA Container Toolkit is configured, verify container GPU access as well.

---

# Step 33: Create a Verification Checklist

Your final environment should look conceptually like:

```text
Ubuntu
  │
  ├── Git
  ├── Linux Tools
  ├── Firewall
  │
  ├── Python Virtual Environment
  │       ├── NumPy
  │       ├── pandas
  │       ├── scikit-learn
  │       ├── PyTorch
  │       └── JupyterLab
  │
  ├── Docker
  │
  └── Optional GPU
          ├── NVIDIA Driver
          └── NVIDIA Container Toolkit
```

---

# Troubleshooting

## `nvidia-smi` Fails

First verify:

```text
Does the machine actually have an NVIDIA GPU?
```

Then:

```text
GPU
 ↓
Driver
 ↓
nvidia-smi
```

Do not troubleshoot PyTorch until this layer works.

---

## PyTorch Reports `CUDA available: False`

If:

```bash
nvidia-smi
```

works but:

```python
torch.cuda.is_available()
```

returns:

```text
False
```

investigate:

```text
PyTorch Build
CUDA Compatibility
Python Environment
Driver Compatibility
```

Confirm that the installed PyTorch build supports GPU acceleration.

---

## JupyterLab Cannot Be Reached

Check that Jupyter is running:

```bash
jupyter lab list
```

Verify the SSH tunnel:

```text
Local Port 8888
      ↓
SSH
      ↓
Remote localhost:8888
```

Avoid opening remote port 8888 publicly simply to solve a tunnel configuration problem.

---

## Docker Cannot Access the GPU

First verify the host:

```bash
nvidia-smi
```

Then verify Docker:

```bash
docker run hello-world
```

Then verify NVIDIA Container Toolkit.

Troubleshoot in this order:

```text
GPU Hardware
     ↓
NVIDIA Driver
     ↓
Docker
     ↓
NVIDIA Container Toolkit
     ↓
GPU Container
```

---

# Layered Troubleshooting Strategy

Avoid repeatedly installing different combinations of:

```text
CUDA
NVIDIA Driver
PyTorch
Container Toolkit
```

without first identifying the failing layer.

Use:

```text
1. Hardware
      ↓
2. Operating System
      ↓
3. NVIDIA Driver
      ↓
4. Python Environment
      ↓
5. PyTorch
      ↓
6. Docker
      ↓
7. NVIDIA Container Toolkit
      ↓
8. Application
```

Test each layer before moving to the next.

This isolates failures far more effectively than reinstalling the entire stack.

---

# Production Improvements

For production AI infrastructure, consider:

* Infrastructure-as-code for environment creation
* Configuration management
* Reproducible container images
* Pinned dependency versions
* Automated security updates
* Centralized logging
* Metrics and monitoring
* Secret management
* Vulnerability scanning
* GPU health monitoring
* Automated environment validation
* Immutable infrastructure
* CI/CD pipelines
* Kubernetes or workload schedulers

A development environment can evolve into:

```text
Manual Ubuntu Setup
        ↓
Setup Script
        ↓
Configuration Management
        ↓
Container Image
        ↓
Infrastructure as Code
        ↓
Automated AI Platform
```

---

# Recommended Repository Structure

For this lab:

```text
chapter-01/
├── lab-01-spin-up-your-first-ai-gpu-vm.md
├── lab-02-build-your-ubuntu-ai-development-environment.md
└── lab-03-compare-gpu-cost-and-performance-across-clouds.md
```

A reusable project created during the lab can use:

```text
ai-starter/
├── data/
├── notebooks/
├── scripts/
├── models/
└── logs/
```

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Identified the Ubuntu version.
* [ ] Verified the current user.
* [ ] Checked for an NVIDIA GPU.
* [ ] Updated Ubuntu packages.
* [ ] Installed essential Linux tools.
* [ ] Configured basic UFW protection where appropriate.
* [ ] Verified Git.
* [ ] Configured Git identity.
* [ ] Created an SSH key if needed.
* [ ] Installed Python.
* [ ] Created an isolated Python environment.
* [ ] Activated the environment.
* [ ] Upgraded `pip`, `setuptools`, and `wheel`.
* [ ] Installed core AI libraries.
* [ ] Verified or installed the NVIDIA driver where applicable.
* [ ] Verified `nvidia-smi`.
* [ ] Installed PyTorch.
* [ ] Verified PyTorch.
* [ ] Verified CUDA from PyTorch where applicable.
* [ ] Executed a GPU matrix operation.
* [ ] Installed and started JupyterLab.
* [ ] Accessed remote Jupyter through an SSH tunnel where applicable.
* [ ] Installed Docker.
* [ ] Successfully ran `hello-world`.
* [ ] Installed NVIDIA Container Toolkit where applicable.
* [ ] Verified GPU access from a container.
* [ ] Created the AI starter project structure.
* [ ] Tested `tmux`.
* [ ] Verified the complete development environment.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Build an Ubuntu-based AI development environment.
* Create isolated Python environments.
* Install and verify PyTorch.
* Explain the relationship between the GPU, NVIDIA driver, CUDA runtime, and AI framework.
* Securely access JupyterLab on a remote machine.
* Install and validate Docker.
* Explain how NVIDIA Container Toolkit exposes GPUs to containers.
* Organize a basic AI development project.
* Use Linux monitoring and administration tools.
* Troubleshoot GPU environments systematically.
* Explain why environment reproducibility is important for AI infrastructure.

---

# Key Takeaway

A reliable AI development environment should be built and verified **layer by layer**.

The core stack is:

```text
Ubuntu Linux
     ↓
Python Environment
     ↓
AI Libraries
     ↓
PyTorch
     ↓
JupyterLab
     ↓
Docker
```

For GPU workloads:

```text
NVIDIA GPU
     ↓
NVIDIA Driver
     ↓
PyTorch CUDA Support
     ↓
GPU Application
```

For containerized GPU workloads:

```text
NVIDIA GPU
     ↓
Host Driver
     ↓
Docker
     ↓
NVIDIA Container Toolkit
     ↓
GPU Container
```

The broader infrastructure principle is:

```text
Build One Layer
      ↓
Verify It
      ↓
Add the Next Layer
      ↓
Verify Again
```

When something fails, troubleshoot in the same order:

```text
Hardware
   ↓
Driver
   ↓
Runtime
   ↓
Framework
   ↓
Container
   ↓
Application
```

That layered approach scales from a single Ubuntu development machine to much larger AI infrastructure environments.
