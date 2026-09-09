# Hands-On Lab: Build a Data Ingestion Pipeline

In this lab, you will build a local data ingestion pipeline that combines **object storage, event streaming, and relational storage**.

You will store raw CSV data in MinIO, stream transaction events through Apache Kafka, and load those events into PostgreSQL for downstream analytics and AI/ML workloads.

The pipeline follows this architecture:

```text
CSV Data
    ↓
MinIO Object Storage
    ↓
Apache Kafka
    ↓
PostgreSQL
    ↓
AI/ML Workloads
```

## Goal

Build a data ingestion pipeline that stores raw CSV data in object storage, streams transaction events through Apache Kafka, and loads the streamed data into PostgreSQL.

## Estimated Time

**90–120 minutes**

## Cost

**Free** when run locally using Docker.

## Tools

This lab uses:

* Python
* Docker
* Docker Compose
* Apache Kafka
* PostgreSQL
* MinIO

---

## Prerequisites

Before beginning the lab, make sure Docker and Docker Compose are installed and running.

Verify Docker:

```bash
docker --version
```

Verify Docker Compose:

```bash
docker compose version
```

You should also have Python installed:

```bash
python --version
```

---

## Step 1: Set Up the Environment

This lab uses a local containerized environment so that the complete data pipeline can run on a single machine.

The infrastructure components are:

| Component          | Role                                     |
| ------------------ | ---------------------------------------- |
| **MinIO**          | S3-compatible object storage             |
| **Apache Kafka**   | Real-time event streaming                |
| **PostgreSQL**     | Structured relational storage            |
| **Docker Compose** | Infrastructure deployment and management |

Create a working directory:

```bash
mkdir ai-ingestion-lab
cd ai-ingestion-lab
```

The pipeline developed in this lab follows this flow:

```text
Raw CSV Dataset
      ↓
MinIO
Object Storage
      ↓
Kafka
Event Streaming
      ↓
PostgreSQL
Structured Storage
      ↓
Analytics / AI / ML
```

---

## Step 2: Deploy the Infrastructure with Docker Compose

Create a file named:

```text
compose.yaml
```

Add the following configuration:

```yaml
services:

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"

    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: password

    ports:
      - "9000:9000"
      - "9001:9001"

  kafka:
    image: bitnami/kafka:latest

    environment:
      KAFKA_CFG_NODE_ID: 1
      KAFKA_CFG_PROCESS_ROLES: broker,controller
      KAFKA_CFG_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_CFG_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
      KAFKA_CFG_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
      KAFKA_CFG_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE: "true"

    ports:
      - "9092:9092"

  postgres:
    image: postgres:14

    environment:
      POSTGRES_USER: aiuser
      POSTGRES_PASSWORD: aipass
      POSTGRES_DB: aidb

    ports:
      - "5432:5432"
```

Start the environment:

```bash
docker compose up -d
```

Verify that the containers are running:

```bash
docker compose ps
```

You should see the MinIO, Kafka, and PostgreSQL containers running.

At this point, your local environment contains:

```text
┌─────────────────┐
│      MinIO      │
│ Object Storage  │
└─────────────────┘

┌─────────────────┐
│      Kafka      │
│ Event Streaming │
└─────────────────┘

┌─────────────────┐
│   PostgreSQL    │
│   Relational DB │
└─────────────────┘
```

---

## Step 3: Create Sample Transaction Data

Create a file named:

```text
transactions.csv
```

For example:

```csv
user_id,amount
1,125
2,450
3,75
4,920
5,210
```

You can also use another CSV dataset if you prefer.

This file represents raw source data that will first be stored in object storage.

---

## Step 4: Ingest Raw Data into Object Storage

Create a Python script named:

```text
ingest_to_minio.py
```

Add:

```python
from minio import Minio

client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="password",
    secure=False
)

bucket = "rawdata"

if not client.bucket_exists(bucket):
    client.make_bucket(bucket)

client.fput_object(
    bucket,
    "transactions.csv",
    "transactions.csv"
)

print("Uploaded transactions.csv to MinIO")
```

Install the MinIO Python client:

```bash
pip install minio
```

Run the script:

```bash
python ingest_to_minio.py
```

You should see:

```text
Uploaded transactions.csv to MinIO
```

The raw CSV file is now stored in the:

```text
rawdata
```

bucket.

This demonstrates a common AI data architecture pattern in which original datasets are retained in scalable object storage before downstream processing begins.

---

## Step 5: Verify the MinIO Upload

Open the MinIO console in your browser:

```text
http://localhost:9001
```

Sign in using:

```text
Username: admin
Password: password
```

Open the:

```text
rawdata
```

bucket.

You should see:

```text
transactions.csv
```

The first stage of the pipeline is now complete:

```text
transactions.csv
        ↓
      MinIO
   rawdata bucket
```

---

## Step 6: Stream Transaction Events with Kafka

The next stage introduces real-time data.

Instead of processing only static files, the pipeline uses Kafka to receive transaction events as they occur.

Create:

```text
kafka_producer.py
```

Add:

```python
from kafka import KafkaProducer
import json
import random
import time

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)

for i in range(10):
    record = {
        "user_id": i,
        "amount": random.randint(10, 1000)
    }

    producer.send("transactions", record)

    print("Produced:", record)

    time.sleep(1)

producer.flush()
```

Install the required Python library:

```bash
pip install kafka-python
```

Run the producer:

```bash
python kafka_producer.py
```

You should see output similar to:

```text
Produced: {'user_id': 0, 'amount': 245}
Produced: {'user_id': 1, 'amount': 712}
Produced: {'user_id': 2, 'amount': 94}
...
```

The producer generates ten transaction events and sends them to the Kafka topic:

```text
transactions
```

In a production environment, similar events might originate from:

* Web applications
* Mobile applications
* Financial systems
* Sensors
* IoT devices
* Clickstreams
* APIs
* Operational databases

The architecture now includes:

```text
Application / Event Source
          ↓
      Kafka Producer
          ↓
   transactions Topic
          ↓
        Kafka
```

---

## Step 7: Consume Events and Store Them in PostgreSQL

The next component reads transaction events from Kafka and writes them into PostgreSQL.

Create:

```text
kafka_consumer.py
```

Add:

```python
from kafka import KafkaConsumer
import json
import psycopg2

consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda message: json.loads(
        message.decode("utf-8")
    )
)

conn = psycopg2.connect(
    dbname="aidb",
    user="aiuser",
    password="aipass",
    host="localhost",
    port=5432
)

cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        user_id INT,
        amount INT
    );
""")

conn.commit()

for message in consumer:
    record = message.value

    cur.execute(
        """
        INSERT INTO transactions (user_id, amount)
        VALUES (%s, %s)
        """,
        (
            record["user_id"],
            record["amount"]
        )
    )

    conn.commit()

    print("Inserted:", record)
```

Install the required libraries:

```bash
pip install psycopg2-binary kafka-python
```

Run the consumer:

```bash
python kafka_consumer.py
```

The consumer continues listening for events until you stop it.

As new transaction events arrive in Kafka, they are inserted into the PostgreSQL:

```text
transactions
```

table.

The streaming portion of the architecture now looks like:

```text
Kafka Producer
      ↓
transactions Topic
      ↓
Kafka Consumer
      ↓
PostgreSQL
transactions Table
```

---

## Step 8: Generate Additional Events

Because the Kafka consumer continues running, open another terminal and run the producer again:

```bash
python kafka_producer.py
```

In the consumer terminal, you should see messages similar to:

```text
Inserted: {'user_id': 0, 'amount': 531}
Inserted: {'user_id': 1, 'amount': 87}
Inserted: {'user_id': 2, 'amount': 643}
```

This demonstrates a basic real-time ingestion pattern:

```text
Incoming Event
      ↓
Kafka
      ↓
Consumer
      ↓
Database Insert
```

Press:

```text
Ctrl+C
```

when you are ready to stop the consumer.

---

## Step 9: Verify the End-to-End Pipeline

Verify each stage to confirm that data is moving successfully through the architecture.

### Verify MinIO

Open:

```text
http://localhost:9001
```

Confirm that:

```text
rawdata/transactions.csv
```

exists.

### Verify Kafka

Review the producer output and confirm that events were successfully generated.

Review the consumer output and confirm that those events were received.

### Verify PostgreSQL

Query PostgreSQL directly:

```bash
docker compose exec postgres \
  psql -U aiuser -d aidb \
  -c "SELECT * FROM transactions;"
```

The result should resemble:

```text
 user_id | amount
---------+--------
       0 |    531
       1 |     87
       2 |    643
       3 |    312
       4 |    755
       ...
```

You have now validated the complete pipeline:

```text
Raw CSV
   ↓
MinIO
   ↓
Object Storage

Real-Time Events
   ↓
Kafka Producer
   ↓
Kafka Topic
   ↓
Kafka Consumer
   ↓
PostgreSQL
   ↓
Analytics / AI / ML
```

---

## Step 10: Inspect the PostgreSQL Table

You can open an interactive PostgreSQL session:

```bash
docker compose exec postgres \
  psql -U aiuser -d aidb
```

List the available tables:

```sql
\dt
```

Inspect the table structure:

```sql
\d transactions
```

Query the data:

```sql
SELECT * FROM transactions;
```

Calculate the number of ingested events:

```sql
SELECT COUNT(*) FROM transactions;
```

Calculate the average transaction amount:

```sql
SELECT AVG(amount) FROM transactions;
```

Exit PostgreSQL:

```text
\q
```

This demonstrates how ingested streaming data can become available for downstream SQL-based analytics or AI feature-processing pipelines.

---

## Step 11: Clean Up the Environment

When the lab is complete, stop the containers and remove the associated Docker volumes:

```bash
docker compose down -v
```

Verify that the containers were removed:

```bash
docker compose ps
```

Because this lab uses local Docker infrastructure, no cloud resources need to be terminated.

> **Cleanup Note**
>
> The `-v` option removes Docker volumes associated with the Compose environment. This means the PostgreSQL and MinIO data created during the lab will also be removed.

---

## Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Started MinIO, Kafka, and PostgreSQL with Docker Compose
* [ ] Created a sample CSV dataset
* [ ] Created a MinIO bucket
* [ ] Uploaded `transactions.csv` to MinIO
* [ ] Verified the CSV file through the MinIO console
* [ ] Created a Kafka producer
* [ ] Published transaction events to Kafka
* [ ] Created a Kafka consumer
* [ ] Created the PostgreSQL `transactions` table
* [ ] Consumed Kafka events
* [ ] Inserted events into PostgreSQL
* [ ] Queried the stored transaction records
* [ ] Verified the complete ingestion pipeline
* [ ] Removed the local Docker environment

---

## Learning Outcomes

After completing this lab, you should be able to:

* Store raw datasets in **S3-compatible object storage**.
* Produce and consume real-time events with **Apache Kafka**.
* Load streaming data into a **relational database**.
* Use **Docker Compose** to deploy supporting data infrastructure.
* Understand the roles of object storage, event streaming, and databases in AI data platforms.
* Trace data through a simple end-to-end ingestion pipeline.
* Understand how ingested data can support downstream analytics and AI/ML workloads.

---

## Key Takeaway

**AI data pipelines often combine multiple storage and processing patterns. Object storage provides scalable storage for raw datasets, streaming platforms move events in real time, and databases provide structured access to processed data for analytics and AI/ML workloads.**
