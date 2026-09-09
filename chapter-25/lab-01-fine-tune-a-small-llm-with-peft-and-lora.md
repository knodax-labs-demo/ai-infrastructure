# Hands-On Lab: Fine-Tune a Small LLM with PEFT and LoRA

In this lab, you will fine-tune a small language model using **Parameter-Efficient Fine-Tuning (PEFT)** and **Low-Rank Adaptation (LoRA)**.

Rather than updating every parameter in the pretrained model, LoRA inserts a small number of trainable low-rank parameters into selected model layers while keeping most original weights frozen. This significantly reduces the amount of trainable state required for adaptation and demonstrates why PEFT is attractive for modern LLM infrastructure.

You will use **DistilGPT-2** with a small sentiment dataset, configure LoRA adapters, train only the adapter parameters, generate text with the adapted model, save the adapter independently, and reload it later.

The complete workflow is:

```text
Base Language Model
        ↓
Prepare Dataset
        ↓
Tokenize
        ↓
Insert LoRA Adapters
        ↓
Freeze Base Parameters
        ↓
Train Adapters
        ↓
Evaluate
        ↓
Save Adapter
        ↓
Reload with Base Model
```

---

## Lab Objectives

By completing this lab, you will learn how to:

* Load a small causal language model.
* Prepare a simple text dataset for causal language modeling.
* Tokenize training examples.
* Apply LoRA using Hugging Face PEFT.
* Identify trainable versus frozen parameters.
* Fine-tune only the adapter parameters.
* Reduce training-state requirements compared with full fine-tuning.
* Generate text using an adapted model.
* Save LoRA adapters separately from the base model.
* Reload an adapter for inference.
* Compare base-model and adapted-model behavior.
* Understand how the same approach can scale to larger LLMs.

---

## Estimated Time

**Approximately 90–150 minutes**

---

## Tools

This lab uses:

* Python 3.9+
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* Accelerate
* Optional CUDA GPU
* Optional `bitsandbytes` for later QLoRA experiments

---

# Step 1: Create the Project

Create the lab directory:

```bash
mkdir peft-lab
cd peft-lab
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```text
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install \
  torch \
  transformers \
  datasets \
  peft \
  accelerate
```

For later experiments with quantized PEFT:

```bash
pip install bitsandbytes
```

> **Note**
>
> `bitsandbytes` is not required for this basic LoRA exercise because the base model is not quantized. It becomes useful when exploring QLoRA or other quantized fine-tuning techniques.

---

# Step 2: Understand PEFT and LoRA

Full-model fine-tuning updates all or most model parameters:

```text
Base Model
   ↓
Update Millions / Billions of Weights
   ↓
Large Gradient State
   ↓
Large Optimizer State
   ↓
High Memory Requirement
```

LoRA changes the pattern:

```text
Base Model
   ↓
Freeze Most Parameters
   ↓
Insert Small Trainable Adapters
   ↓
Train Only Adapter Parameters
```

Conceptually:

```text
Original Weight Matrix
        W
        ↓
     Frozen

LoRA Update
   A × B
     ↓
Trainable
```

The effective weight becomes conceptually:

```text
W' = W + ΔW
```

where:

```text
ΔW = B × A
```

and the low-rank matrices contain far fewer parameters than the original weight matrix.

---

# Step 3: Load the Base Model

Create:

```text
lab-07-train-peft.py
```

Start with:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = "distilgpt2"


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)

tokenizer.pad_token = (
    tokenizer.eos_token
)


model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
)

model.config.pad_token_id = (
    tokenizer.pad_token_id
)
```

---

## Why DistilGPT-2?

DistilGPT-2 is useful for this lab because it is:

* Small enough for experimentation.
* A causal language model.
* Compatible with the Hugging Face ecosystem.
* Easier to train with limited hardware.
* Suitable for demonstrating PEFT concepts.

The goal is not to build a production sentiment model. The goal is to understand the infrastructure pattern.

---

# Step 4: Understand Padding

GPT-2-family tokenizers do not define a padding token by default.

For this lab:

```python
tokenizer.pad_token = (
    tokenizer.eos_token
)
```

The end-of-sequence token is reused as padding.

Then:

```python
model.config.pad_token_id = (
    tokenizer.pad_token_id
)
```

keeps model and tokenizer configuration aligned.

---

# Step 5: Load the Training Dataset

Add:

```python
from datasets import (
    load_dataset
)


dataset = (
    load_dataset(
        "glue",
        "sst2"
    )["train"]
)

dataset = (
    dataset.select(
        range(2000)
    )
)
```

The lab uses a subset of **SST-2** to keep the workload manageable.

SST-2 contains:

```text
Sentence
+
Binary Sentiment Label
```

where:

```text
0 → negative
1 → positive
```

---

# Step 6: Convert the Dataset to a Causal-LM Format

DistilGPT-2 is being used here as a causal language model rather than a sequence-classification model.

Add:

```python
def format_example(
    example
):
    sentiment = (
        "positive"
        if example["label"] == 1
        else "negative"
    )

    return {
        "text": (
            f"Review: "
            f"{example['sentence']}\n"
            f"Sentiment: "
            f"{sentiment}"
        )
    }


dataset = (
    dataset.map(
        format_example
    )
)
```

A formatted example becomes:

```text
Review: a charming and often affecting journey
Sentiment: positive
```

The model is therefore trained to continue text in the pattern:

```text
Review: ...
Sentiment: ...
```

---

# Step 7: Understand the Training Objective

Instead of:

```text
Input Sentence
      ↓
Classification Head
      ↓
Positive / Negative
```

this lab uses:

```text
Text Prompt
      ↓
Causal Language Model
      ↓
Generate Sentiment Text
```

This keeps the experiment focused on causal-LM fine-tuning.

---

# Step 8: Tokenize the Dataset

Add:

```python
def tokenize(
    batch
):
    encoded = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )

    labels = []

    for (
        input_ids,
        attention_mask
    ) in zip(
        encoded["input_ids"],
        encoded["attention_mask"]
    ):
        labels.append(
            [
                token_id
                if mask == 1
                else -100

                for (
                    token_id,
                    mask
                ) in zip(
                    input_ids,
                    attention_mask
                )
            ]
        )

    encoded[
        "labels"
    ] = labels

    return encoded


tokenized = (
    dataset.map(
        tokenize,
        batched=True,
        remove_columns=(
            dataset.column_names
        ),
    )
)
```

The resulting dataset contains:

```text
input_ids
attention_mask
labels
```

---

# Step 9: Understand the Label Mask

Padding tokens should not contribute to training loss.

Therefore:

```text
Real Token
    ↓
Use token ID as label

Padding Token
    ↓
Use -100
```

PyTorch loss functions commonly ignore positions marked:

```text
-100
```

This prevents the model from learning to predict artificial padding.

---

# Step 10: Configure LoRA

Add:

```python
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


lora_config = LoraConfig(
    task_type=(
        TaskType.CAUSAL_LM
    ),

    r=8,

    lora_alpha=16,

    lora_dropout=0.1,

    target_modules=[
        "c_attn"
    ],

    bias="none",
)


model = get_peft_model(
    model,
    lora_config
)
```

---

# Step 11: Understand the LoRA Configuration

The main parameters are:

| Parameter        | Purpose                   |
| ---------------- | ------------------------- |
| `r`              | Low-rank dimension        |
| `lora_alpha`     | LoRA scaling              |
| `lora_dropout`   | Regularization            |
| `target_modules` | Layers receiving adapters |
| `bias`           | Bias-training behavior    |

In this lab:

```text
r = 8
```

means the adaptation is represented using low-rank matrices of rank 8.

---

# Step 12: Understand Target Modules

The configuration uses:

```python
target_modules=[
    "c_attn"
]
```

For GPT-2-style architectures, `c_attn` participates in attention projections.

Conceptually:

```text
Transformer Layer
      ↓
Attention Projection
      ↓
c_attn
      ↓
LoRA Adapter Added
```

The base weight remains largely frozen while LoRA contributes a trainable update.

---

# Step 13: Inspect Trainable Parameters

Run:

```python
model.print_trainable_parameters()
```

You should observe that only a small fraction of the total model parameters are trainable.

The important comparison is:

```text
Total Parameters
       vs.
Trainable Parameters
```

rather than relying on a fixed percentage.

---

# Step 14: Understand the Infrastructure Benefit

Full fine-tuning requires training state for much of the model:

```text
Base Parameters
+
Gradients
+
Optimizer State
```

PEFT reduces that requirement:

```text
Frozen Base Model
      +
LoRA Parameters
      +
LoRA Gradients
      +
LoRA Optimizer State
```

This can substantially reduce memory pressure and storage requirements.

---

# Step 15: Configure Training

Add:

```python
import torch

from transformers import (
    Trainer,
    TrainingArguments,
)


training_args = (
    TrainingArguments(
        output_dir="outputs",

        per_device_train_batch_size=8,

        num_train_epochs=2,

        learning_rate=2e-4,

        logging_steps=10,

        save_strategy="epoch",

        report_to="none",

        fp16=(
            torch.cuda.is_available()
        ),
    )
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)
```

---

# Step 16: Understand the Training Configuration

This lab uses:

```text
Batch Size      = 8
Epochs          = 2
Learning Rate   = 2e-4
FP16            = Enabled when CUDA is available
```

These settings keep the experiment relatively lightweight.

If GPU memory is limited, reduce:

```python
per_device_train_batch_size
```

and optionally use gradient accumulation.

---

# Step 17: Understand Effective Batch Size

If you later use gradient accumulation:

```text
Per-Device Batch Size
        ×
Gradient Accumulation Steps
        =
Effective Batch Size
```

For example:

```text
Batch Size = 2
Accumulation = 4

Effective Batch Size = 8
```

This is useful when GPU memory cannot accommodate a larger batch directly.

---

# Step 18: Train the LoRA Adapters

Run:

```python
trainer.train()
```

During training, monitor:

* Training loss
* Step count
* Epoch progress
* GPU memory usage
* Training time

The main optimization flow is:

```text
Forward Pass
     ↓
Loss
     ↓
Backward Pass
     ↓
Gradients
     ↓
LoRA Parameters Updated
```

Most base-model parameters remain frozen.

---

# Step 19: Monitor GPU Usage

If training on an NVIDIA GPU:

```bash
nvidia-smi
```

Observe:

* GPU utilization
* GPU memory
* Power usage
* Temperature

The purpose is to connect model-training behavior with infrastructure utilization.

---

# Step 20: Generate Text with the Adapted Model

After training:

```python
model.eval()


prompt = (
    "Review: "
    "The movie was exciting "
    "and beautifully made.\n"
    "Sentiment:"
)


inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inputs = {
    key:
        value.to(
            model.device
        )

    for (
        key,
        value
    ) in inputs.items()
}


with torch.no_grad():
    outputs = (
        model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=(
                tokenizer
                .eos_token_id
            ),
        )
    )


result = (
    tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
)


print(result)
```

---

# Step 21: Understand the Generation Test

The prompt follows the training format:

```text
Review: ...
Sentiment:
```

The adapted model may continue with:

```text
positive
```

or:

```text
negative
```

or additional text.

This is an educational fine-tuning experiment, not a rigorous sentiment classifier.

---

# Step 22: Compare Before and After Adaptation

Test the same prompts with:

```text
Base DistilGPT-2
```

and:

```text
LoRA-Adapted DistilGPT-2
```

Use identical generation settings.

For example:

```text
Review: This was one of the best films I have seen this year.
Sentiment:
```

```text
Review: The story was dull and the acting was disappointing.
Sentiment:
```

```text
Review: A thoughtful and beautifully performed movie.
Sentiment:
```

Record how consistently each model follows the expected output pattern.

---

# Step 23: Understand Why Evaluation Matters

Decreasing training loss does not automatically mean the adapted model is better.

A basic evaluation should ask:

```text
Does the model follow the format?
Does it produce the expected sentiment?
Does adaptation improve behavior?
Does it generalize beyond training examples?
```

Production evaluation would require:

* Held-out validation data
* Quantitative metrics
* Multiple prompts
* Regression tests
* Safety evaluation where appropriate

---

# Step 24: Save the LoRA Adapter

Add:

```python
ADAPTER_DIR = (
    "lora-finetuned"
)


model.save_pretrained(
    ADAPTER_DIR
)

tokenizer.save_pretrained(
    ADAPTER_DIR
)
```

The resulting directory contains the adapter configuration and adapter weights.

---

# Step 25: Understand Adapter Storage

Instead of saving:

```text
Entire Base Model
+
Fine-Tuned Copy
```

for every task, PEFT allows:

```text
Shared Base Model
      +
Adapter A
Adapter B
Adapter C
```

Conceptually:

```text
               ┌── LoRA Adapter A
Base Model ────┼── LoRA Adapter B
               └── LoRA Adapter C
```

This is one of the strongest operational advantages of PEFT.

---

# Step 26: Inspect Adapter Size

Compare:

```text
Base Model Checkpoint Size
```

with:

```text
LoRA Adapter Size
```

The adapter should generally be much smaller because it contains only the additional trainable parameters rather than another complete model checkpoint.

---

# Step 27: Reload the Adapter

Create:

```text
lab-07-reload-adapter.py
```

Add:

```python
from peft import (
    PeftModel
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = (
    "distilgpt2"
)

ADAPTER_DIR = (
    "lora-finetuned"
)


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        ADAPTER_DIR
    )
)

tokenizer.pad_token = (
    tokenizer.eos_token
)


base_model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
)

base_model.config.pad_token_id = (
    tokenizer.pad_token_id
)


lora_model = (
    PeftModel
    .from_pretrained(
        base_model,
        ADAPTER_DIR
    )
)

lora_model.eval()


print(
    "LoRA adapter loaded."
)
```

---

# Step 28: Understand Adapter Reloading

The serving path becomes:

```text
Base Model
    ↓
Load Pretrained Weights
    ↓
Load LoRA Adapter
    ↓
Combined Inference Model
```

The base model and adaptation remain separate artifacts.

---

# Step 29: Understand Multi-Adapter Architecture

A shared base model can support multiple task-specific adapters:

```text
                 ┌── Sentiment Adapter
                 │
Base LLM ─────────┼── Support Adapter
                 │
                 ├── Legal Adapter
                 │
                 └── Finance Adapter
```

Each adapter can be:

* Versioned independently
* Distributed independently
* Updated independently
* Removed independently
* Evaluated independently

This can significantly reduce duplicate model storage.

---

# Step 30: Understand PEFT vs Full Fine-Tuning

## Full Fine-Tuning

```text
Base Model
   ↓
Update Most / All Parameters
   ↓
Large Gradient Memory
   ↓
Large Optimizer State
   ↓
Large Checkpoint
```

## PEFT / LoRA

```text
Base Model
   ↓
Freeze Parameters
   ↓
Add Small Adapters
   ↓
Train Adapter Parameters
   ↓
Small Adapter Checkpoint
```

---

# Step 31: Compare Infrastructure Requirements

| Area                 | Full Fine-Tuning | LoRA           |
| -------------------- | ---------------- | -------------- |
| Trainable parameters | Large            | Small          |
| Gradient memory      | High             | Lower          |
| Optimizer state      | High             | Lower          |
| Adapter storage      | N/A              | Small          |
| Base-model reuse     | Limited          | Strong         |
| Task-specific copies | Large            | Small adapters |

Actual savings depend on architecture and LoRA configuration.

---

# Step 32: Experiment with LoRA Rank

Try:

```text
r = 4
r = 8
r = 16
```

The conceptual tradeoff is:

```text
Lower Rank
   ↓
Fewer Parameters
Lower Memory
Lower Adaptation Capacity
```

versus:

```text
Higher Rank
   ↓
More Parameters
More Memory
Potentially More Adaptation Capacity
```

Measure rather than assume which rank works best.

---

# Step 33: Experiment with Target Modules

The lab targets:

```text
c_attn
```

Try other compatible modules where appropriate.

Compare:

* Trainable parameter count
* Training time
* Adapter size
* Output quality

The placement of LoRA adapters can affect both efficiency and model behavior.

---

# Step 34: Understand QLoRA

A natural extension is **QLoRA**.

LoRA:

```text
Full-Precision Base Model
        +
LoRA Adapters
```

QLoRA:

```text
Quantized Base Model
        +
Trainable LoRA Adapters
```

Conceptually:

```text
Large Model
    ↓
Quantize Base Weights
    ↓
Reduce Memory
    ↓
Train LoRA Adapters
```

This can make adaptation of larger models possible on more limited hardware.

---

# Step 35: Understand the QLoRA Infrastructure Benefit

The main goal is:

```text
Reduce Base-Model Memory
        +
Reduce Trainable Parameters
        ↓
Lower Fine-Tuning Infrastructure Requirement
```

This is especially valuable for multi-billion-parameter models.

---

# Step 36: Measure Training Efficiency

Record:

| Metric               | Result |
| -------------------- | -----: |
| Total parameters     | Record |
| Trainable parameters | Record |
| Adapter size         | Record |
| Training time        | Record |
| Peak GPU memory      | Record |
| Epochs               |      2 |
| Batch size           | Record |

This turns the lab into an infrastructure experiment rather than only a model-training exercise.

---

# Step 37: Evaluate Multiple Prompts

Use positive and negative examples.

For example:

```text
Review: The performances were excellent and the story was moving.
Sentiment:
```

```text
Review: The movie was tedious and badly written.
Sentiment:
```

```text
Review: A clever and entertaining film.
Sentiment:
```

Record:

| Prompt          | Base Model | LoRA Model | Expected |
| --------------- | ---------- | ---------- | -------- |
| Positive review | Record     | Record     | Positive |
| Negative review | Record     | Record     | Negative |
| Positive review | Record     | Record     | Positive |

---

# Step 38: Understand Training vs Serving Artifacts

The training environment produces:

```text
Base Model
+
Training Dataset
+
LoRA Configuration
+
Adapter Weights
```

At inference time, you generally need:

```text
Base Model
+
LoRA Adapter
+
Tokenizer
```

This separation reduces deployment artifact duplication.

---

# Step 39: Optional FastAPI Serving Architecture

A later extension can expose the adapted model through FastAPI:

```text
Client
   ↓
FastAPI
   ↓
Base Model
   +
LoRA Adapter
   ↓
Generation
   ↓
Response
```

Load the model and adapter once at application startup rather than on every request.

---

# Step 40: Production Improvements

A production PEFT workflow should consider:

* Model and adapter versioning
* Dataset versioning
* Reproducible training
* Evaluation datasets
* Experiment tracking
* Adapter registry
* Security
* Model provenance
* GPU scheduling
* Distributed training where needed
* Checkpoint management
* Monitoring
* Rollback

---

# Step 41: Production Adapter Lifecycle

A mature workflow may look like:

```text
Base Model
    ↓
Training Dataset
    ↓
LoRA Training
    ↓
Evaluation
    ↓
Adapter Registry
    ↓
Approval
    ↓
Deployment
    ↓
Monitoring
    ↓
Rollback / Update
```

This treats LoRA adapters as independently managed ML artifacts.

---

# Step 42: Clean Up

Deactivate the environment:

```bash
deactivate
```

Generated directories may include:

```text
outputs/
lora-finetuned/
```

The `outputs/` directory may contain intermediate checkpoints.

The `lora-finetuned/` directory contains the final adapter and tokenizer artifacts.

---

# Recommended Repository Structure

For the standalone lab:

```text
peft-lab/
├── train_peft.py
├── reload_adapter.py
├── outputs/
└── lora-finetuned/
```

For your book companion repository:

```text
chapter-25/
├── lab-07-fine-tune-a-small-llm-with-peft-and-lora.md
├── lab-07-train-peft.py
└── lab-07-reload-adapter.py
```

I would normally avoid committing large generated checkpoints in:

```text
outputs/
```

Instead, commit the source code and document how to reproduce the adapter.

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Created the PEFT project.
* [ ] Created a Python virtual environment.
* [ ] Installed Transformers.
* [ ] Installed PEFT.
* [ ] Installed Datasets and Accelerate.
* [ ] Loaded DistilGPT-2.
* [ ] Configured the padding token.
* [ ] Loaded the SST-2 dataset.
* [ ] Selected a manageable training subset.
* [ ] Converted examples to a causal-LM format.
* [ ] Tokenized the dataset.
* [ ] Masked padding labels with `-100`.
* [ ] Created the LoRA configuration.
* [ ] Applied LoRA to `c_attn`.
* [ ] Printed trainable parameter counts.
* [ ] Configured Hugging Face Trainer.
* [ ] Trained the LoRA adapters.
* [ ] Monitored training loss.
* [ ] Generated text using the adapted model.
* [ ] Compared multiple prompts.
* [ ] Saved the adapter.
* [ ] Compared adapter size with the base model.
* [ ] Reloaded the base model.
* [ ] Reloaded the adapter.
* [ ] Verified inference after reload.
* [ ] Reviewed LoRA rank tradeoffs.
* [ ] Reviewed QLoRA as an extension.
* [ ] Reviewed production adapter lifecycle concepts.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Explain Parameter-Efficient Fine-Tuning.
* Explain how LoRA reduces trainable parameter count.
* Prepare a text dataset for causal-language-model fine-tuning.
* Apply LoRA adapters with Hugging Face PEFT.
* Identify frozen and trainable parameters.
* Train only adapter parameters.
* Measure adapter-training efficiency.
* Generate text with an adapted model.
* Save LoRA adapters independently.
* Reload adapters with a shared base model.
* Compare base and adapted model behavior.
* Explain how PEFT can reduce GPU-memory, optimizer-state, and storage requirements.
* Describe how LoRA can scale to larger LLM adaptation workflows.
* Explain the relationship between LoRA and QLoRA.

---

# Key Takeaway

**LoRA makes model adaptation more infrastructure-efficient by keeping the pretrained foundation model largely frozen and training only a small set of low-rank adapter parameters.**

The basic architecture is:

```text
Pretrained LLM
      ↓
Freeze Base Weights
      ↓
Insert LoRA Adapters
      ↓
Train Small Parameter Set
      ↓
Save Adapter
      ↓
Reuse Base Model
```

Instead of maintaining separate complete model copies:

```text
Model A
Model B
Model C
```

PEFT enables:

```text
Shared Base Model
      ├── Adapter A
      ├── Adapter B
      └── Adapter C
```

This provides major benefits for:

* Training memory
* Optimizer state
* Checkpoint size
* Storage
* Versioning
* Multi-task adaptation
* Deployment flexibility

The broader infrastructure principle is:

```text
Model Customization
        ↓
Minimize Trainable State
        ↓
Reuse Foundation Model
        ↓
Store Lightweight Adapters
        ↓
Reduce Infrastructure Cost
```

PEFT does not eliminate the need for good data, careful evaluation, reproducibility, and deployment governance. However, it provides a practical path to adapting increasingly large language models without requiring full-model fine-tuning for every task.
