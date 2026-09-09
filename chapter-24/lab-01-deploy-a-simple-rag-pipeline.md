# Hands-On Lab: Deploy a Simple RAG Pipeline

In this lab, you will build and deploy a complete **Retrieval-Augmented Generation (RAG)** pipeline that answers questions using information retrieved from your own documents.

You will split source documents into chunks, generate vector embeddings, store them in a **FAISS vector index**, retrieve relevant chunks at query time, provide that context to a large language model, and expose the entire workflow through a **FastAPI** endpoint.

The complete architecture is:

```text
Source Documents
      ↓
Chunking
      ↓
Embeddings
      ↓
FAISS Vector Index
      ↓
User Question
      ↓
Query Embedding
      ↓
Semantic Retrieval
      ↓
Relevant Context
      ↓
LLM
      ↓
Grounded Answer
      ↓
FastAPI Response
```

---

## Lab Objectives

By completing this lab, you will learn how to:

* Build an end-to-end RAG pipeline.
* Split documents into manageable chunks.
* Generate embeddings for document chunks.
* Store vectors in FAISS.
* Perform semantic similarity search.
* Retrieve relevant context at query time.
* Augment an LLM prompt with retrieved content.
* Generate grounded answers.
* Expose the RAG pipeline through FastAPI.
* Test the complete request path.
* Identify production improvements such as caching, reranking, observability, and security.

---

## Estimated Time

**Approximately 120–180 minutes**

---

## Tools

This lab uses:

* Python 3.9+
* OpenAI-compatible embedding and LLM APIs
* FAISS
* NumPy
* `tiktoken`
* FastAPI
* Uvicorn
* Pydantic

---

# Step 1: Create the Project

Create:

```bash
mkdir rag-lab
cd rag-lab
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```text
.venv\Scripts\activate
```

---

# Step 2: Install the Dependencies

Install:

```bash
pip install \
  faiss-cpu \
  openai \
  fastapi \
  uvicorn \
  tiktoken \
  pydantic \
  numpy
```

Set the API key.

On macOS or Linux:

```bash
export OPENAI_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

---

# Step 3: Create the Sample Documents

Create:

```text
docs/
```

Add files such as:

```text
docs/
├── ai_infra.txt
├── etl_vs_elt.txt
└── kafka_streaming.txt
```

Each file should contain enough text to produce several chunks.

The initial project becomes:

```text
rag-lab/
├── docs/
│   ├── ai_infra.txt
│   ├── etl_vs_elt.txt
│   └── kafka_streaming.txt
```

In production, these documents could come from:

* Object storage
* Databases
* Enterprise document repositories
* Knowledge bases
* Content-management systems
* APIs

---

# Step 4: Understand the Indexing Pipeline

The indexing stage performs:

```text
Documents
   ↓
Read Text
   ↓
Chunk Text
   ↓
Generate Embeddings
   ↓
Store Vectors
   ↓
FAISS Index
```

Chunking improves retrieval granularity.

Instead of embedding one large document:

```text
Large Document
     ↓
One Vector
```

you create:

```text
Large Document
     ↓
Chunk 1 → Vector
Chunk 2 → Vector
Chunk 3 → Vector
...
```

---

# Step 5: Build the Embedding Index

Create:

```text
lab-07-build-index.py
```

Add:

```python
import glob
import os
import pickle

import faiss
import numpy as np
import tiktoken

from openai import OpenAI


client = OpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY"
    )
)

EMBED_MODEL = (
    "text-embedding-3-small"
)


def embed(texts):
    response = (
        client
        .embeddings
        .create(
            model=EMBED_MODEL,
            input=texts
        )
    )

    return [
        item.embedding
        for item
        in response.data
    ]


def chunk_text(
    text,
    size=500,
    overlap=50
):
    encoding = (
        tiktoken
        .get_encoding(
            "cl100k_base"
        )
    )

    tokens = encoding.encode(
        text
    )

    chunks = []

    step = size - overlap

    for i in range(
        0,
        len(tokens),
        step
    ):
        chunk_tokens = tokens[
            i:i + size
        ]

        if not chunk_tokens:
            continue

        chunks.append(
            encoding.decode(
                chunk_tokens
            )
        )

    return chunks


docs = []
metadatas = []


for file in glob.glob(
    "docs/*.txt"
):
    with open(
        file,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    chunks = chunk_text(
        text
    )

    docs.extend(
        chunks
    )

    metadatas.extend(
        [
            {
                "source": file
            }
        ] * len(chunks)
    )


if not docs:
    raise ValueError(
        "No document chunks were created. "
        "Add .txt files to the docs directory."
    )


embeddings = embed(
    docs
)

vectors = np.array(
    embeddings,
    dtype="float32"
)

dimension = (
    vectors.shape[1]
)

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    vectors
)


with open(
    "faiss_index.pkl",
    "wb"
) as f:
    pickle.dump(
        (
            index,
            docs,
            metadatas
        ),
        f
    )


print(
    f"Index built with "
    f"{len(docs)} chunks."
)
```

---

# Step 6: Understand Chunking

The function uses:

```text
Chunk Size = 500 tokens
Overlap    = 50 tokens
```

Conceptually:

```text
Document
   ↓
┌───────────────┐
│ Chunk 1       │
└───────────────┘
          ┌───────────────┐
          │ Chunk 2       │
          └───────────────┘
                    ┌───────────────┐
                    │ Chunk 3       │
                    └───────────────┘
```

The overlap helps preserve information that crosses chunk boundaries.

---

# Step 7: Generate the Index

Run:

```bash
python lab-07-build-index.py
```

Expected artifact:

```text
faiss_index.pkl
```

The file stores:

```text
FAISS Index
+
Document Chunks
+
Metadata
```

The source text is now represented as vectors that can be searched semantically.

---

# Step 8: Understand Vector Retrieval

At query time:

```text
Question
   ↓
Embedding Model
   ↓
Query Vector
   ↓
FAISS Search
   ↓
Nearest Document Vectors
   ↓
Relevant Chunks
```

The system retrieves semantically similar text rather than requiring exact keyword matches.

---

# Step 9: Build the RAG Runtime

Create:

```text
lab-07-rag.py
```

Add:

```python
import os
import pickle

import numpy as np

from openai import OpenAI


client = OpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY"
    )
)

EMBED_MODEL = (
    "text-embedding-3-small"
)

LLM_MODEL = (
    "gpt-4o-mini"
)


with open(
    "faiss_index.pkl",
    "rb"
) as f:
    (
        index,
        docs,
        metadatas
    ) = pickle.load(f)


def embed_query(query):
    response = (
        client
        .embeddings
        .create(
            model=EMBED_MODEL,
            input=[query]
        )
    )

    vector = np.array(
        response
        .data[0]
        .embedding,
        dtype="float32"
    )

    return vector.reshape(
        1,
        -1
    )


def retrieve(
    query,
    k=3
):
    query_vector = (
        embed_query(
            query
        )
    )

    k = min(
        k,
        len(docs)
    )

    distances, indices = (
        index.search(
            query_vector,
            k
        )
    )

    results = []

    for i in indices[0]:
        if i >= 0:
            results.append(
                {
                    "text":
                        docs[i],

                    "metadata":
                        metadatas[i]
                }
            )

    return results


def rag_answer(query):
    retrieved = retrieve(
        query,
        k=3
    )

    contexts = [
        item["text"]
        for item
        in retrieved
    ]

    context_text = (
        "\n\n---\n\n"
        .join(contexts)
    )

    prompt = f"""
Use the following retrieved context to answer the question.

Answer only from the provided context.

If the answer cannot be determined from the context,
say that the provided documents do not contain enough
information.

Context:

{context_text}

Question:

{query}
"""

    response = (
        client
        .chat
        .completions
        .create(
            model=LLM_MODEL,
            messages=[
                {
                    "role":
                        "user",

                    "content":
                        prompt
                }
            ]
        )
    )

    return {
        "answer":
            response
            .choices[0]
            .message
            .content,

        "sources":
            [
                item[
                    "metadata"
                ]
                for item
                in retrieved
            ]
    }
```

---

# Step 10: Understand the RAG Flow

The runtime performs:

```text
Question
   ↓
Query Embedding
   ↓
Vector Search
   ↓
Top-K Chunks
   ↓
Prompt Augmentation
   ↓
LLM
   ↓
Grounded Answer
```

This is the classic:

```text
Retrieve
   ↓
Augment
   ↓
Generate
```

pattern.

---

# Step 11: Test Retrieval and Generation

Open Python:

```bash
python
```

Then:

```python
from lab_07_rag import rag_answer

result = rag_answer(
    "What is the difference between ETL and ELT?"
)

print(
    result["answer"]
)
```

The system should retrieve relevant content from:

```text
etl_vs_elt.txt
```

and use that context to answer.

---

# Step 12: Test Questions Across Documents

Try:

```text
What is AI infrastructure?
```

```text
How does Kafka support streaming systems?
```

```text
What is the difference between ETL and ELT?
```

The retrieved context should change according to the query.

---

# Step 13: Understand Grounding

Without RAG:

```text
Question
   ↓
LLM Internal Knowledge
   ↓
Answer
```

With RAG:

```text
Question
   ↓
Retrieve Your Documents
   ↓
Provide Context
   ↓
LLM
   ↓
Grounded Answer
```

The retrieved context helps constrain the response to information from your knowledge base.

---

# Step 14: Build the FastAPI Service

Create:

```text
lab-07-server.py
```

Add:

```python
from fastapi import FastAPI
from pydantic import BaseModel

from lab_07_rag import (
    rag_answer
)


app = FastAPI(
    title="Simple RAG API",
    version="1.0"
)


class Query(BaseModel):
    question: str


@app.get(
    "/health"
)
def health():
    return {
        "status":
            "healthy"
    }


@app.post(
    "/ask"
)
def ask(
    q: Query
):
    result = rag_answer(
        q.question
    )

    return {
        "question":
            q.question,

        "answer":
            result[
                "answer"
            ],

        "sources":
            result[
                "sources"
            ]
    }
```

---

# Step 15: Start the API

Run:

```bash
uvicorn lab_07_server:app \
  --reload \
  --port 8000
```

The API becomes available at:

```text
http://127.0.0.1:8000
```

---

# Step 16: Test the Health Endpoint

Run:

```bash
curl \
  http://127.0.0.1:8000/health
```

Expected:

```json
{
  "status": "healthy"
}
```

---

# Step 17: Test the RAG Endpoint

Run:

```bash
curl -X POST \
  http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question":
      "Explain Kafka streaming for AI systems"
  }'
```

A successful response should resemble:

```json
{
  "question":
    "Explain Kafka streaming for AI systems",

  "answer":
    "Kafka enables real-time data ingestion...",

  "sources": [
    {
      "source":
        "docs/kafka_streaming.txt"
    }
  ]
}
```

---

# Step 18: Understand the Complete Request Path

The complete serving path is:

```text
Client
   ↓
FastAPI
   ↓
Question
   ↓
Embedding API
   ↓
Query Vector
   ↓
FAISS
   ↓
Relevant Chunks
   ↓
Prompt
   ↓
LLM
   ↓
Answer
   ↓
JSON Response
```

This architecture separates retrieval logic from client applications.

---

# Step 19: Return Source Metadata

Returning document metadata helps users understand where the answer came from.

The RAG result now includes:

```json
{
  "answer":
    "...",

  "sources": [
    {
      "source":
        "docs/etl_vs_elt.txt"
    }
  ]
}
```

This is a useful first step toward RAG citations.

---

# Step 20: Add Retrieval Scores

FAISS returns:

```text
distances
indices
```

The current example uses only the indices.

A stronger RAG service can also preserve the similarity score.

Conceptually:

```text
Retrieved Chunk
      +
Similarity Score
      +
Source Metadata
```

This supports:

* Debugging
* Relevance thresholds
* Retrieval analysis
* Reranking

---

# Step 21: Add a Retrieval Threshold

Without a threshold, the system always returns the nearest chunks, even if none are especially relevant.

A better production pattern is:

```text
Query
   ↓
Retrieve Top-K
   ↓
Check Relevance
   │
   ├── Relevant
   │      ↓
   │     LLM
   │
   └── Too Weak
          ↓
"No sufficient information"
```

This can reduce answers based on weak retrieval.

---

# Step 22: Understand FAISS Limitations

FAISS is excellent for local experimentation.

Advantages include:

* Fast local vector search
* Simple setup
* No external database
* Useful for prototypes

However, a local in-memory or file-backed FAISS index does not automatically provide:

* Distributed scaling
* Replication
* Multi-node durability
* Fine-grained access control
* Managed backups
* Native metadata filtering at enterprise scale
* Multi-tenant isolation

For larger systems, consider a production vector database or vector-enabled search system.

---

# Step 23: Add Structured Logging

A production RAG service should record events such as:

```text
Request ID
Query
Retrieval Time
Embedding Time
LLM Time
Total Latency
Retrieved Sources
Token Usage
Errors
```

Be careful not to log sensitive document content or user queries unless appropriate controls are in place.

---

# Step 24: Measure RAG Latency

The total response time can be decomposed as:

```text
Total RAG Latency
      =
Query Embedding
      +
Vector Retrieval
      +
Prompt Construction
      +
LLM Generation
```

This breakdown helps identify performance bottlenecks.

---

# Step 25: Add Caching

Frequently repeated requests can benefit from caching.

One architecture is:

```text
Question
   ↓
Cache Lookup
   │
   ├── Hit
   │     ↓
   │  Return Response
   │
   └── Miss
         ↓
      RAG Pipeline
         ↓
      Cache Result
```

Possible cache technologies include:

```text
Redis
```

Caching may reduce:

* Embedding API calls
* Retrieval work
* LLM calls
* Latency
* Cost

---

# Step 26: Add Retrieval Caching

Queries with repeated or highly similar retrieval patterns may also cache retrieval results.

Conceptually:

```text
Question
   ↓
Embedding
   ↓
Retrieval Cache
   │
   ├── Hit
   └── FAISS Search
```

This is separate from caching the final generated answer.

---

# Step 27: Add Reranking

Basic retrieval may return semantically related but suboptimal chunks.

A more advanced architecture is:

```text
Query
   ↓
Vector Search
   ↓
Top 20 Chunks
   ↓
Reranker
   ↓
Top 5 Chunks
   ↓
LLM
```

Reranking can improve retrieval precision before context is sent to the model.

---

# Step 28: Add Hybrid Retrieval

Vector search is semantic.

Keyword search is lexical.

A hybrid design combines both:

```text
Query
   ├── Vector Search
   └── Keyword Search
           ↓
       Merge Results
           ↓
        Rerank
           ↓
          LLM
```

This is useful when exact terms such as:

* Product codes
* Names
* Error messages
* Identifiers
* Technical terminology

matter strongly.

---

# Step 29: Improve Chunking

The lab uses fixed token chunks.

Production systems may instead use:

* Paragraph-aware chunking
* Heading-aware chunking
* Semantic chunking
* Sentence-aware splitting
* Document-structure-aware splitting

The goal is:

```text
Small Enough
for Precise Retrieval
       +
Large Enough
for Useful Context
```

---

# Step 30: Add Document Metadata

Metadata can include:

```text
source
title
section
author
timestamp
document_id
page
category
```

For example:

```json
{
  "source":
    "docs/kafka_streaming.txt",

  "section":
    "Consumer Groups",

  "document_id":
    "kafka-001"
}
```

Metadata enables richer citations and filtering.

---

# Step 31: Add Authentication

The current API is open.

Production deployments should consider:

* API keys
* OAuth 2.0
* OIDC
* JWT
* Service identities

The architecture becomes:

```text
Client
   ↓
Authentication
   ↓
Authorization
   ↓
RAG API
```

---

# Step 32: Add Tenant Isolation

For multi-user or enterprise systems:

```text
User / Tenant
      ↓
Authorization
      ↓
Allowed Documents
      ↓
Retrieval
```

The retrieval layer must not return documents that the caller is not authorized to access.

This is a critical production RAG requirement.

---

# Step 33: Containerize the Service

A later extension can package:

```text
FastAPI
+
RAG Runtime
+
Dependencies
```

into a Docker image.

Conceptually:

```text
Docker Image
     ↓
Container
     ↓
RAG API
```

The FAISS index could be included in the image, mounted as a volume, or loaded from external storage.

---

# Step 34: Deploy to Kubernetes

A production deployment might look like:

```text
Load Balancer
      ↓
Kubernetes Service
      ↓
RAG API Pods
  ┌────┼────┐
 Pod  Pod  Pod
      ↓
Vector Infrastructure
      ↓
LLM / Embedding APIs
```

Kubernetes can add:

* Horizontal scaling
* Health probes
* Rolling deployments
* Resource limits
* Secret management
* Fault recovery

---

# Step 35: Add Observability

Monitor:

* Request rate
* Error rate
* p50/p95/p99 latency
* Embedding latency
* Retrieval latency
* LLM latency
* Token usage
* Cache hit rate
* Retrieval relevance
* Model failures

A common architecture is:

```text
RAG Service
     ↓
Metrics / Logs / Traces
     ↓
Observability Platform
```

---

# Step 36: Understand Production RAG Architecture

A more mature system may evolve toward:

```text
                     Documents
                         ↓
                  Ingestion Pipeline
                         ↓
                  Chunk / Enrich
                         ↓
                    Embeddings
                         ↓
                  Vector Database
                         ↓
User Query → API → Retrieval → Reranking
                         ↓
                     Context
                         ↓
                       LLM
                         ↓
                     Answer
                         ↓
                 Sources / Citations
```

Additional layers may include:

```text
Authentication
Authorization
Caching
Guardrails
Observability
Evaluation
Security
```

---

# Step 37: Validate Grounded Responses

Test questions that are:

### Supported by Documents

Expected behavior:

```text
Relevant Context
      ↓
Answer
```

### Unsupported by Documents

Expected behavior:

```text
No Sufficient Context
       ↓
"Provided documents do not contain enough information."
```

This test is important because a grounded system should be able to decline unsupported answers.

---

# Step 38: Evaluate Retrieval Separately

Do not evaluate only the final answer.

Check:

```text
Question
   ↓
Retrieved Chunks
```

and ask:

* Did the correct document appear?
* Was the most relevant section retrieved?
* Were irrelevant chunks included?
* Did chunk boundaries remove important context?

RAG quality depends heavily on retrieval quality.

---

# Step 39: Understand RAG Failure Modes

Common RAG problems include:

* Wrong chunks retrieved
* Important chunks not retrieved
* Poor chunk boundaries
* Weak embeddings
* Too much context
* Too little context
* Stale documents
* Unauthorized documents retrieved
* LLM ignores context
* LLM misinterprets context

Therefore:

```text
RAG Quality
     ≠
LLM Quality Alone
```

Instead:

```text
RAG Quality
     =
Data Quality
+
Chunking
+
Embeddings
+
Retrieval
+
Reranking
+
Prompting
+
LLM
```

---

# Step 40: Clean Up

Stop FastAPI:

```text
Ctrl+C
```

Deactivate:

```bash
deactivate
```

Generated artifacts such as:

```text
faiss_index.pkl
```

can be removed and rebuilt when required.

---

# Recommended Folder Structure

For the standalone lab:

```text
rag-lab/
├── docs/
│   ├── ai_infra.txt
│   ├── etl_vs_elt.txt
│   └── kafka_streaming.txt
├── build_index.py
├── rag.py
├── server.py
└── faiss_index.pkl
```

For your book companion repository:

```text
chapter-24/
├── lab-07-deploy-a-simple-rag-pipeline.md
├── lab-07-build-index.py
├── lab-07-rag.py
└── lab-07-server.py
```

I would generally avoid committing:

```text
faiss_index.pkl
```

because it is a generated index artifact and can be rebuilt from the source documents.

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Created the RAG project.
* [ ] Created a Python virtual environment.
* [ ] Installed FAISS.
* [ ] Installed the API dependencies.
* [ ] Configured the model API key.
* [ ] Added sample documents.
* [ ] Split documents into overlapping chunks.
* [ ] Generated embeddings.
* [ ] Built the FAISS index.
* [ ] Stored source metadata.
* [ ] Generated query embeddings.
* [ ] Retrieved the top matching chunks.
* [ ] Built a context-augmented prompt.
* [ ] Generated a grounded answer.
* [ ] Tested multiple source documents.
* [ ] Created the FastAPI service.
* [ ] Added the `/health` endpoint.
* [ ] Added the `/ask` endpoint.
* [ ] Returned source metadata.
* [ ] Tested the API with `curl`.
* [ ] Tested an unsupported question.
* [ ] Reviewed caching opportunities.
* [ ] Reviewed reranking.
* [ ] Reviewed hybrid retrieval.
* [ ] Reviewed production security.
* [ ] Reviewed observability.
* [ ] Reviewed scaling options.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Explain the architecture of RAG.
* Split source documents into retrieval-friendly chunks.
* Generate vector embeddings.
* Build a FAISS index.
* Perform semantic similarity search.
* Retrieve relevant context.
* Use retrieved content to augment an LLM prompt.
* Generate document-grounded answers.
* Expose RAG through FastAPI.
* Return basic source information with answers.
* Explain why retrieval quality strongly affects answer quality.
* Recognize limitations of local FAISS deployments.
* Identify production improvements including reranking, caching, authentication, observability, and distributed vector storage.

---

# Key Takeaway

**Retrieval-Augmented Generation connects large language models with external knowledge by retrieving relevant information at query time and supplying that information as context for generation.**

The core architecture is:

```text
Documents
   ↓
Chunks
   ↓
Embeddings
   ↓
Vector Index
   ↓
Semantic Retrieval
   ↓
Relevant Context
   ↓
LLM
   ↓
Grounded Answer
```

FAISS provides a lightweight vector-retrieval layer for this lab, while FastAPI turns the RAG workflow into a reusable service.

The most important architectural principle is:

```text
RAG
 =
Retrieval Quality
+
Context Quality
+
Generation Quality
```

A strong LLM cannot compensate for consistently poor retrieval. Production RAG systems therefore treat **data ingestion, chunking, embeddings, retrieval, reranking, metadata, security, evaluation, caching, and observability** as first-class infrastructure components.

The broader progression is:

```text
Simple RAG Prototype
        ↓
Source Metadata
        ↓
Better Retrieval
        ↓
Reranking
        ↓
Caching
        ↓
Authentication
        ↓
Observability
        ↓
Distributed Vector Infrastructure
        ↓
Production RAG Platform
```
