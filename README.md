# AI Infrastructure from Beginner to Advanced — Hands-On Labs

This repository contains the hands-on labs, exercises, configuration examples, scripts, and supporting resources for the book **AI Infrastructure from Beginner to Advanced**.

The labs are designed to complement the concepts covered in the book and provide practical experience with the technologies used to build, deploy, operate, and scale modern artificial intelligence and machine learning infrastructure.

## About the Labs

The exercises in this repository cover a range of AI infrastructure technologies and practices, including:

* Linux environments for AI development
* CPU and GPU computing
* NVIDIA GPUs and CUDA
* Python and PyTorch
* Docker and containerized AI workloads
* Kubernetes and GPU-enabled Kubernetes clusters
* Cloud-based AI infrastructure
* AWS, Microsoft Azure, and Google Cloud
* Data ingestion and processing pipelines
* Distributed training
* Model serving and inference
* MLOps and model lifecycle management
* CI/CD for AI and machine learning workloads
* Monitoring, observability, and performance optimization
* Large Language Model (LLM) infrastructure
* Enterprise AI infrastructure patterns

The labs progress from foundational exercises to more advanced infrastructure scenarios so that readers can gradually develop practical AI infrastructure engineering skills.

## Repository Structure

Labs are organized primarily by chapter.

```text
ai-infrastructure-labs/
│
├── chapter-01/
│   ├── lab-01-...
│   └── lab-02-...
│
├── chapter-02/
│   ├── lab-01-...
│   └── lab-02-...
│
├── chapter-03/
│   └── ...
│
├── ...
│
└── README.md
```

Each lab may contain step-by-step instructions, commands, source code, configuration files, deployment manifests, or other resources required to complete the exercise.

## Prerequisites

Requirements vary depending on the lab. Some exercises can be completed on a local computer, while others may require cloud infrastructure or access to GPU resources.

Depending on the exercise, you may need:

* A Linux, macOS, or Windows development environment
* Python and `pip`
* Git
* Docker
* Kubernetes or Minikube
* NVIDIA GPU drivers and CUDA
* PyTorch or other machine learning frameworks
* AWS, Microsoft Azure, or Google Cloud accounts
* Command-line tools such as AWS CLI, Azure CLI, or Google Cloud CLI
* Appropriate cloud permissions and credentials

Always review the prerequisites listed in an individual lab before starting it.

## Important Note About AI-Assisted Content

Some of the labs, exercises, code examples, commands, configurations, and supporting materials in this repository were created or refined with the assistance of **generative AI tools**.

Although the materials are reviewed and organized for educational purposes, AI-generated or AI-assisted content can contain errors, outdated syntax, incorrect assumptions, incomplete configurations, or code that requires modification for a particular environment.

Therefore, you should treat the examples as **educational starting points rather than guaranteed production-ready implementations**.

You may need to modify or troubleshoot portions of the code depending on factors such as:

* Operating system and version
* Python or package versions
* GPU model and driver version
* CUDA version
* Docker or Kubernetes version
* Cloud provider services and APIs
* SDK and CLI versions
* Library and framework updates
* Account permissions and security policies
* Regional cloud-service availability
* Changes introduced after publication

Cloud platforms, AI frameworks, APIs, and infrastructure tools evolve rapidly. Commands or configurations that worked when a lab was created may require adjustments as these technologies change.

Part of working with AI infrastructure is learning how to diagnose compatibility problems, interpret errors, consult current documentation, and adapt configurations to your environment. Readers are encouraged to use the labs as a foundation for experimentation and deeper learning.

## Cloud Costs

Some labs may create **billable cloud resources**, including virtual machines, GPU instances, storage, managed Kubernetes clusters, databases, networking components, or other services.

Before running a cloud-based lab:

1. Review the resources that will be created.
2. Check the current pricing from the applicable cloud provider.
3. Use the smallest practical resource configuration for learning.
4. Stop or delete resources when they are no longer required.
5. Verify that all billable resources have been removed after completing the lab.

GPU-enabled cloud instances can be significantly more expensive than standard CPU instances.

You are responsible for any charges associated with resources created in your cloud accounts.

## Security

The labs are intended primarily for learning and experimentation. Example configurations may prioritize clarity and simplicity over production-level security.

Do not place passwords, API keys, access keys, tokens, private keys, or other credentials directly in source code or commit them to a public Git repository.

For production environments, follow your organization's security requirements and the security best practices recommended by the applicable technology or cloud provider.

## Troubleshooting

If a command or code example does not work exactly as shown:

1. Read the complete error message.
2. Verify software and dependency versions.
3. Confirm that required services are running.
4. Check environment variables and configuration files.
5. Verify cloud permissions and credentials.
6. Check GPU drivers and CUDA compatibility when applicable.
7. Consult the latest official documentation for the technology being used.
8. Modify the example as necessary for your environment.

Troubleshooting is an important part of developing practical AI infrastructure engineering skills.

## Educational Purpose

The materials in this repository are provided for **educational and informational purposes**. They are intended to demonstrate AI infrastructure concepts and provide hands-on learning opportunities.

The examples should be reviewed, tested, secured, and adapted before being considered for production use.

## Companion Book

These labs accompany:

**AI Infrastructure from Beginner to Advanced**

The book provides the concepts, architecture, explanations, and technical background behind the exercises, while this repository provides practical activities for applying those concepts.

For the best learning experience, follow the relevant chapter in the book before or while completing its associated labs.

## Author

**SK Singh**
Founder, KnoDAX

**KnoDAX — Learn. Get Certified. Get Ahead.**

## Contributions and Feedback

AI infrastructure technologies change rapidly. If you discover an issue, outdated command, compatibility problem, or improvement to an exercise, feedback and contributions are welcome.

When reporting an issue, please include relevant information such as the lab name, operating system, software versions, error message, and the steps that produced the problem.

This information makes it easier to reproduce and correct the issue.

## Disclaimer

The code, commands, configurations, and exercises in this repository are provided **"as is"** for educational purposes without warranties or guarantees of any kind.

Users are responsible for reviewing and testing the materials before using them in their own environments. Neither the author nor KnoDAX is responsible for data loss, service interruption, security incidents, cloud charges, or other damages resulting from the use or modification of these materials.
