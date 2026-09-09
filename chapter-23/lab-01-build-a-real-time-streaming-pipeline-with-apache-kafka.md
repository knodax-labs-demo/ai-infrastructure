# Hands-On Lab: Build a Real-Time Streaming Pipeline with Apache Kafka

In this lab, you will build an end-to-end **real-time streaming data pipeline** using **Apache Kafka**, Python, and Kafka UI.

The pipeline will continuously generate synthetic user-activity events, publish them to Kafka, validate and aggregate the incoming data, generate derived features, and deliver those features to downstream consumers that could support analytics or machine learning workloads.

You will also use Kafka UI and application logs to observe the system, inspect partitions and consumer groups, experiment with throughput and lag, and optionally route invalid records to a dead-letter queue.

The complete pipeline is:

```text
Synthetic Events
      ↓
Kafka Producer
      ↓
transactions Topic
      ↓
Stream Processor
      ├── Validation
      ├── Windowing
      ├── Aggregation
      └── Feature Generation
             ↓
        features Topic
             ↓
      Feature Consumer
             ↓
Analytics / ML / Storage

Invalid Events
      ↓
dlq-transactions
```

---

## Lab Objectives

By completing this lab, you will learn how to:

* Run Apache Kafka locally.
* Use Kafka in KRaft mode.
* Observe Kafka using a browser-based Kafka UI.
* Publish synthetic events in real time.
* Partition related records using Kafka message keys.
* Consume and validate streaming data.
* Perform fixed-window aggregation.
* Generate derived features from event streams.
* Work with Kafka consumer groups.
* Manage consumer offsets explicitly.
* Observe consumer lag.
* Route invalid events to a dead-letter queue.
* Experiment with throughput and window size.
* Understand the limitations of in-memory state.
* Recognize where Flink, Kafka Streams, or Spark Structured Streaming become appropriate.

---

## Estimated Time

**Approximately 120–180 minutes**

---

## Tools

This lab uses:

* Docker Desktop
* Docker Compose
* Apache Kafka
* Kafka KRaft mode
* Kafka UI
* Python 3.9+
* `kafka-python`
* Pydantic
* Faker
* `python-dateutil`
* Optional CSV output

---

# Step 1: Verify the Prerequisites

Ensure Docker is running.

Verify Python:

```bash
python --version
```

Depending on your environment, you may need:

```bash
python3 --version
```

Verify Docker:

```bash
docker --version
```

Verify Docker Compose:

```bash
docker compose version
```

---

# Step 2: Create the Project Structure

Create the project:

```bash
mkdir kafka-streaming-lab
cd kafka-streaming-lab
mkdir app
```

The completed project will look like:

```text
kafka-streaming-lab/
├── docker-compose.yml
└── app/
    ├── requirements.txt
    ├── producer.py
    ├── processor.py
    └── consumer.py
```

The architecture separates the system into:

```text
Infrastructure
     ↓
docker-compose.yml

Applications
     ↓
producer.py
processor.py
consumer.py
```

This separation makes each part easier to start, stop, inspect, and troubleshoot independently.

---

# Step 3: Configure Kafka and Kafka UI

Create:

```text
docker-compose.yml
```

Add:

```yaml
services:
  kafka:
    image: bitnami/kafka:3.7
    container_name: kafka

    ports:
      - "9092:9092"
      - "29092:29092"

    environment:
      - KAFKA_ENABLE_KRAFT=yes
      - KAFKA_CFG_NODE_ID=1
      - KAFKA_CFG_PROCESS_ROLES=broker,controller
      - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=1@kafka:9093
      - KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER

      - KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP=INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT,CONTROLLER:PLAINTEXT

      - KAFKA_CFG_LISTENERS=INTERNAL://:29092,EXTERNAL://:9092,CONTROLLER://:9093

      - KAFKA_CFG_ADVERTISED_LISTENERS=INTERNAL://kafka:29092,EXTERNAL://localhost:9092

      - KAFKA_CFG_INTER_BROKER_LISTENER_NAME=INTERNAL
      - KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE=true

    volumes:
      - kafka_data:/bitnami/kafka

  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: kafka-ui

    ports:
      - "8080:8080"

    environment:
      - KAFKA_CLUSTERS_0_NAME=local
      - KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS=kafka:29092

    depends_on:
      - kafka

volumes:
  kafka_data:
```

This creates:

```text
Host Python Apps
      ↓
localhost:9092
      ↓
Kafka Broker

Kafka UI Container
      ↓
kafka:29092
      ↓
Kafka Broker
```

Kafka is running in **KRaft mode**, so ZooKeeper is not required.

> **Local Development Note**
>
> This configuration uses plaintext listeners for simplicity. Production Kafka deployments should add authentication, authorization, encryption, replication, monitoring, and appropriate operational controls.

---

# Step 4: Start Kafka

Run:

```bash
docker compose up -d
```

Verify:

```bash
docker compose ps
```

Open Kafka UI:

```text
http://localhost:8080
```

You should see:

```text
Cluster: local
```

---

# Step 5: Install the Python Dependencies

Create:

```text
app/requirements.txt
```

Add:

```text
kafka-python==2.0.2
faker==25.9.2
pydantic==2.8.2
python-dateutil==2.9.0.post0
```

Move into:

```bash
cd app
```

Create a virtual environment:

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

Install:

```bash
pip install -r requirements.txt
```

---

# Step 6: Create the Kafka Topics

Create these topics in Kafka UI:

| Topic              | Partitions | Purpose             |
| ------------------ | ---------: | ------------------- |
| `transactions`     |          3 | Raw events          |
| `features`         |          3 | Aggregated features |
| `dlq-transactions` |          1 | Invalid records     |

The flow is:

```text
transactions
     ↓
Stream Processor
     ├── Valid
     │     ↓
     │  features
     │
     └── Invalid
           ↓
      dlq-transactions
```

Multiple partitions provide a foundation for parallel processing.

---

# Step 7: Build the Event Producer

Create:

```text
app/producer.py
```

Add:

```python
import json
import os
import random
import time

from datetime import datetime, timezone

from faker import Faker
from kafka import KafkaProducer


BOOTSTRAP = os.getenv(
    "BOOTSTRAP",
    "localhost:9092"
)

TOPIC = os.getenv(
    "TOPIC",
    "transactions"
)

EPS = float(
    os.getenv(
        "EVENTS_PER_SECOND",
        "5"
    )
)

fake = Faker()

EVENT_TYPES = [
    "view",
    "add_to_cart",
    "purchase",
]

DEVICES = [
    "web",
    "ios",
    "android",
]

COUNTRIES = [
    "US",
    "IN",
    "BR",
    "DE",
    "GB",
    "CA",
]


producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,

    value_serializer=lambda value:
        json.dumps(value).encode("utf-8"),

    key_serializer=lambda key:
        str(key).encode("utf-8")
        if key is not None
        else None,

    linger_ms=10,
)


def make_event():
    user_id = random.randint(
        1,
        500
    )

    event_type = random.choices(
        EVENT_TYPES,
        weights=[
            0.7,
            0.2,
            0.1,
        ],
    )[0]

    amount = (
        round(
            random.uniform(
                5,
                200
            ),
            2
        )
        if event_type == "purchase"
        else 0.0
    )

    event = {
        "event_time":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "user_id":
            user_id,

        "event_type":
            event_type,

        "amount":
            amount,

        "device":
            random.choice(
                DEVICES
            ),

        "country":
            random.choice(
                COUNTRIES
            ),
    }

    return event, user_id


def main():
    if EPS <= 0:
        raise ValueError(
            "EVENTS_PER_SECOND must be greater than zero."
        )

    print(
        f"Producing events to '{TOPIC}' "
        f"at approximately {EPS} events/sec"
    )

    interval = 1.0 / EPS

    try:
        while True:
            payload, key = make_event()

            producer.send(
                TOPIC,
                key=key,
                value=payload
            )

            time.sleep(
                interval
            )

    except KeyboardInterrupt:
        print(
            "\nStopping producer..."
        )

    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
```

---

# Step 8: Understand the Producer

Each event contains:

```json
{
  "event_time": "...",
  "user_id": 123,
  "event_type": "purchase",
  "amount": 76.58,
  "device": "web",
  "country": "US"
}
```

The producer generates:

```text
view
add_to_cart
purchase
```

The approximate event distribution is:

```text
View         ~70%
Add to Cart  ~20%
Purchase     ~10%
```

---

## Kafka Message Key

The producer uses:

```text
user_id
```

as the Kafka key.

This is important because:

```text
Same User ID
     ↓
Same Kafka Key
     ↓
Typically Same Partition
```

as long as the partition configuration remains stable.

That helps preserve per-user ordering and becomes useful for stateful processing.

---

# Step 9: Start the Producer

Run:

```bash
python producer.py
```

In Kafka UI navigate to:

```text
Topics
   ↓
transactions
   ↓
Messages
```

You should see events arriving continuously.

---

# Step 10: Understand the Stream Processing Stage

The next component performs:

```text
Kafka Event
    ↓
Validation
    ↓
Determine Time Window
    ↓
Group by User
    ↓
Aggregate
    ↓
Create Feature Record
    ↓
Publish to features
```

Invalid records take another path:

```text
Invalid Event
      ↓
Validation Failure
      ↓
DLQ
```

---

# Step 11: Build the Stream Processor

Create:

```text
app/processor.py
```

Add:

```python
import json
import os
import time

from collections import defaultdict
from datetime import datetime, timezone

from dateutil import parser as dtparser
from kafka import KafkaConsumer, KafkaProducer
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)


BOOTSTRAP = os.getenv(
    "BOOTSTRAP",
    "localhost:9092"
)

SRC_TOPIC = os.getenv(
    "SRC_TOPIC",
    "transactions"
)

SINK_TOPIC = os.getenv(
    "SINK_TOPIC",
    "features"
)

DLQ_TOPIC = os.getenv(
    "DLQ_TOPIC",
    "dlq-transactions"
)

WINDOW_SEC = int(
    os.getenv(
        "WINDOW_SEC",
        "60"
    )
)


class Txn(BaseModel):
    event_time: str

    user_id: int = Field(
        ge=1
    )

    event_type: str

    amount: float = Field(
        ge=0
    )

    device: str
    country: str


def window_start(
    ts_iso: str
) -> int:

    timestamp = (
        dtparser
        .isoparse(ts_iso)
    )

    epoch = int(
        timestamp.timestamp()
    )

    return (
        epoch // WINDOW_SEC
    ) * WINDOW_SEC


consumer = KafkaConsumer(
    SRC_TOPIC,

    bootstrap_servers=BOOTSTRAP,

    group_id="processor-1",

    value_deserializer=lambda value:
        json.loads(
            value.decode("utf-8")
        ),

    key_deserializer=lambda key:
        int(
            key.decode("utf-8")
        )
        if key
        else None,

    auto_offset_reset="earliest",

    enable_auto_commit=False,

    max_poll_records=200,
)


producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,

    value_serializer=lambda value:
        json.dumps(value).encode("utf-8"),
)


state = defaultdict(
    lambda: {
        "event_count": 0,
        "purchase_count": 0,
        "revenue": 0.0,
    }
)


last_flush = time.time()


def flush_ready(
    current_epoch: int
):
    current_window = (
        current_epoch
        // WINDOW_SEC
    ) * WINDOW_SEC

    completed = []

    for (
        user_id,
        win_start
    ), aggregate in list(
        state.items()
    ):

        if win_start < current_window:

            output = {
                "user_id":
                    user_id,

                "window_start":
                    win_start,

                "window_end":
                    win_start
                    + WINDOW_SEC,

                "event_count":
                    aggregate[
                        "event_count"
                    ],

                "purchase_count":
                    aggregate[
                        "purchase_count"
                    ],

                "revenue":
                    round(
                        aggregate[
                            "revenue"
                        ],
                        2
                    ),

                "emitted_at":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
            }

            producer.send(
                SINK_TOPIC,
                value=output
            )

            completed.append(
                (
                    user_id,
                    win_start
                )
            )

    producer.flush()

    for key in completed:
        del state[key]


try:
    print(
        f"Processing '{SRC_TOPIC}' "
        f"→ '{SINK_TOPIC}' "
        f"using {WINDOW_SEC}-second windows"
    )

    while True:
        records = consumer.poll(
            timeout_ms=1000
        )

        if not records:
            flush_ready(
                int(
                    time.time()
                )
            )

            continue

        for _, messages in records.items():

            for message in messages:

                try:
                    txn = Txn(
                        **message.value
                    )

                    win_start = (
                        window_start(
                            txn.event_time
                        )
                    )

                    key = (
                        txn.user_id,
                        win_start
                    )

                    aggregate = (
                        state[key]
                    )

                    aggregate[
                        "event_count"
                    ] += 1

                    if (
                        txn.event_type
                        == "purchase"
                    ):
                        aggregate[
                            "purchase_count"
                        ] += 1

                        aggregate[
                            "revenue"
                        ] += float(
                            txn.amount
                        )

                except (
                    ValidationError,
                    ValueError
                ) as exc:

                    producer.send(
                        DLQ_TOPIC,

                        value={
                            "error":
                                "validation_error",

                            "reason":
                                str(exc),

                            "payload":
                                message.value,
                        },
                    )

        producer.flush()
        consumer.commit()

        now = time.time()

        if (
            now - last_flush
            > 5
        ):
            flush_ready(
                int(now)
            )

            last_flush = now


except KeyboardInterrupt:
    print(
        "\nStopping processor..."
    )


finally:
    flush_ready(
        int(
            time.time()
        ) + WINDOW_SEC
    )

    consumer.commit()
    consumer.close()

    producer.flush()
    producer.close()
```

---

# Step 12: Understand Validation

Pydantic validates:

```text
user_id >= 1
amount >= 0
```

before an event enters the aggregation state.

The flow is:

```text
Incoming JSON
     ↓
Pydantic Validation
     │
     ├── Valid
     │     ↓
     │ Aggregation
     │
     └── Invalid
           ↓
     dlq-transactions
```

This prevents malformed data from silently contaminating downstream features.

---

# Step 13: Understand Windowed Aggregation

The processor uses fixed windows.

Default:

```text
WINDOW_SEC = 60
```

For each user and time window, it calculates:

```text
event_count
purchase_count
revenue
```

The state key is:

```text
(user_id, window_start)
```

For example:

```text
User 123
Window 10:00–10:01
        ↓
9 Events
2 Purchases
$153.17 Revenue
```

---

# Step 14: Understand Feature Generation

A completed aggregate resembles:

```json
{
  "user_id": 123,
  "window_start": 1787854920,
  "window_end": 1787854980,
  "event_count": 9,
  "purchase_count": 2,
  "revenue": 153.17,
  "emitted_at": "..."
}
```

These values are examples of **streaming features** that could feed:

* Fraud detection
* Recommendation systems
* Personalization
* User-behavior analytics
* Real-time scoring
* Feature stores

---

# Step 15: Understand Offset Management

The processor disables automatic commits:

```python
enable_auto_commit=False
```

and commits explicitly:

```python
consumer.commit()
```

The conceptual flow is:

```text
Consume Records
      ↓
Process
      ↓
Produce Results
      ↓
Commit Offset
```

This gives more control than automatic offset commits.

However:

> **Important**
>
> This simple implementation does not provide exactly-once processing guarantees.

A failure between producing output and committing offsets can potentially lead to repeated processing.

---

# Step 16: Start the Stream Processor

Open a second terminal.

Activate the same environment and run:

```bash
python processor.py
```

Leave both:

```text
producer.py
processor.py
```

running.

---

# Step 17: Build the Feature Consumer

Create:

```text
app/consumer.py
```

Add:

```python
import csv
import json
import os

from kafka import KafkaConsumer


BOOTSTRAP = os.getenv(
    "BOOTSTRAP",
    "localhost:9092"
)

TOPIC = os.getenv(
    "TOPIC",
    "features"
)

WRITE_CSV = (
    os.getenv(
        "WRITE_CSV",
        "false"
    ).lower()
    == "true"
)


consumer = KafkaConsumer(
    TOPIC,

    bootstrap_servers=BOOTSTRAP,

    value_deserializer=lambda value:
        json.loads(
            value.decode("utf-8")
        ),

    auto_offset_reset="earliest",

    group_id="features-reader",
)


file_handle = None
writer = None


try:
    if WRITE_CSV:

        file_handle = open(
            "features.csv",
            "w",
            newline="",
            encoding="utf-8",
        )

    print(
        f"Consuming features from '{TOPIC}'..."
    )

    for message in consumer:

        record = message.value

        print(
            record
        )

        if WRITE_CSV:

            if writer is None:

                writer = csv.DictWriter(
                    file_handle,
                    fieldnames=record.keys(),
                )

                writer.writeheader()

            writer.writerow(
                record
            )

            file_handle.flush()


except KeyboardInterrupt:
    print(
        "\nStopping feature consumer..."
    )


finally:
    consumer.close()

    if file_handle:
        file_handle.close()
```

---

# Step 18: Start the Feature Consumer

Open a third terminal:

```bash
python consumer.py
```

To write results to CSV on macOS or Linux:

```bash
WRITE_CSV=true python consumer.py
```

On Windows PowerShell:

```powershell
$env:WRITE_CSV="true"
python consumer.py
```

---

# Step 19: Observe the Complete Pipeline

The system is now:

```text
producer.py
    ↓
transactions
    ↓
processor.py
    ↓
features
    ↓
consumer.py
```

With error handling:

```text
              ┌──► features
transactions ─┤
              └──► dlq-transactions
```

---

# Step 20: Inspect the Pipeline in Kafka UI

Open:

```text
http://localhost:8080
```

Inspect:

```text
Topics
Partitions
Messages
Consumer Groups
Offsets
Consumer Lag
```

Useful locations include:

```text
Topics → transactions → Messages
Topics → features → Messages
Topics → dlq-transactions → Messages
```

---

# Step 21: Validate Consumer Lag

Inspect:

```text
processor-1
```

in Kafka UI.

Consumer lag represents approximately:

```text
Latest Topic Offset
        -
Consumer Processed Offset
        =
Consumer Lag
```

Under normal load:

```text
Producer Rate
      ≤
Processor Capacity
```

lag should remain relatively low.

---

# Step 22: Test the Dead-Letter Queue

Temporarily modify the producer so that some events contain:

```text
amount < 0
```

For example:

```python
amount = -10.0
```

The Pydantic rule:

```python
amount: float = Field(
    ge=0
)
```

should reject the record.

Expected path:

```text
Invalid Event
     ↓
Validation Error
     ↓
dlq-transactions
```

Inspect:

```text
Kafka UI
  ↓
Topics
  ↓
dlq-transactions
```

Verify that the rejected payload and reason are visible.

Then restore the producer.

---

# Step 23: Increase Event Throughput

Stop the producer and restart it with:

```bash
EVENTS_PER_SECOND=50 \
python producer.py
```

Observe:

* Topic message rate
* Processor behavior
* Consumer lag
* CPU usage

The principle is:

```text
Input Rate
   >
Processing Rate
   ↓
Consumer Lag Grows
```

This is an important operational signal in production streaming systems.

---

# Step 24: Experiment with Window Size

Run the processor with:

```bash
WINDOW_SEC=30 \
python processor.py
```

Now feature records should be emitted more frequently.

Compare:

```text
60-second Window
       ↓
Less Frequent
Larger Aggregates
```

with:

```text
30-second Window
       ↓
More Frequent
Smaller Aggregates
```

Window size affects both system behavior and downstream feature freshness.

---

# Step 25: Test Failure Recovery

Stop:

```text
processor.py
```

but leave the producer running.

Events continue accumulating in:

```text
transactions
```

Wait briefly, then restart:

```bash
python processor.py
```

The consumer should resume based on its committed offsets.

Conceptually:

```text
Processor Stops
      ↓
Kafka Retains Events
      ↓
Lag Grows
      ↓
Processor Restarts
      ↓
Reads from Committed Offset
      ↓
Catches Up
```

This demonstrates one of Kafka's central architectural advantages.

---

# Step 26: Experiment with Schema Evolution

Add a new field to producer events:

```python
"marketing_channel":
    random.choice(
        [
            "organic",
            "search",
            "social",
            "email",
        ]
    )
```

Observe how the processor behaves.

In the current Pydantic configuration, extra fields are ignored by default.

This demonstrates a simple form of schema evolution, but production systems should use stronger schema governance.

Common approaches include:

* Avro
* Protobuf
* JSON Schema
* Schema Registry
* Compatibility rules

---

# Step 27: Understand Consumer Groups

Kafka distributes partitions among consumers with the same:

```text
group_id
```

Conceptually:

```text
Topic
├── Partition 0 ──► Consumer A
├── Partition 1 ──► Consumer B
└── Partition 2 ──► Consumer C
```

This enables horizontal processing.

However, there is an important limitation in this lab.

---

# Step 28: Understand the In-Memory State Limitation

The processor keeps aggregation state in:

```text
Python Process Memory
```

That works for one local processor but is not sufficient for distributed stateful stream processing.

If multiple processors share the same consumer group:

```text
Partition Rebalance
      ↓
Events Move to Another Process
      ↓
Previous In-Memory State
May Remain Elsewhere
```

This can make aggregates incorrect.

Production frameworks solve this more robustly.

---

# Step 29: Production Stateful Processing

Frameworks such as:

```text
Apache Flink
Kafka Streams
Spark Structured Streaming
```

provide stronger support for:

* Partitioned state
* Checkpoints
* Recovery
* Event-time processing
* Watermarks
* Distributed execution
* Fault tolerance

The lab's Python processor exists to expose the mechanics clearly, not to replace these systems.

---

# Step 30: Understand Event Time vs. Processing Time

The event contains:

```text
event_time
```

and the processor uses it to determine:

```text
window_start
```

Conceptually:

```text
Event Created
    ↓
event_time
    ↓
Window Assignment
```

This differs from simply grouping events according to when the processor happens to receive them.

Production stream processors typically provide more sophisticated event-time semantics and late-event handling.

---

# Step 31: Understand the AI Feature Pipeline

The generated records can function as real-time ML features.

For example:

```text
User Activity
     ↓
60-Second Window
     ↓
event_count
purchase_count
revenue
     ↓
Feature Vector
     ↓
ML Model
```

Potential use cases include:

```text
Fraud Detection
Recommendations
Churn Prediction
Personalization
Real-Time Scoring
```

---

# Step 32: Optional — Persist Features to CSV

Run:

```bash
WRITE_CSV=true \
python consumer.py
```

This creates:

```text
features.csv
```

The architecture becomes:

```text
features Topic
      ↓
consumer.py
      ↓
features.csv
```

CSV is useful for the lab, but production feature pipelines would normally use more appropriate storage systems.

---

# Step 33: Production Storage Extensions

Possible sinks include:

```text
Kafka
  ↓
PostgreSQL
```

or:

```text
Kafka
  ↓
Redis
```

or:

```text
Kafka
  ↓
Feature Store
```

or:

```text
Kafka
  ↓
Data Lake / Lakehouse
```

---

# Step 34: Production Observability

A mature streaming platform should monitor:

* Producer throughput
* Consumer throughput
* Consumer lag
* Error rate
* DLQ volume
* Partition balance
* Broker health
* Processing latency
* Window completion delay
* JVM and host metrics
* Disk utilization

A common architecture is:

```text
Kafka / Applications
        ↓
Prometheus
        ↓
Grafana
        ↓
Alerts
```

---

# Step 35: Production Security

The local lab uses:

```text
PLAINTEXT
```

listeners.

Production environments should consider:

```text
TLS
Authentication
Authorization
ACLs
Secret Management
Network Segmentation
Encryption
Audit Logging
```

Kafka should not be exposed publicly without appropriate security controls.

---

# Step 36: Expected Results

The `transactions` topic should receive events such as:

```json
{
  "event_time":
    "2026-08-27T18:22:31.481233+00:00",

  "user_id":
    123,

  "event_type":
    "purchase",

  "amount":
    76.58,

  "device":
    "web",

  "country":
    "US"
}
```

After a window closes, `features` should contain records similar to:

```json
{
  "user_id":
    123,

  "window_start":
    1787854920,

  "window_end":
    1787854980,

  "event_count":
    9,

  "purchase_count":
    2,

  "revenue":
    153.17,

  "emitted_at":
    "2026-08-27T18:23:05.678901+00:00"
}
```

Exact values will differ because events are generated randomly.

---

# Step 37: Validate the End-to-End Pipeline

Verify:

```text
Producer
   ↓
transactions
   ↓
Processor
   ↓
features
   ↓
Consumer
```

and separately:

```text
Invalid Event
   ↓
dlq-transactions
```

Kafka UI should also show:

* Topics
* Partitions
* Messages
* Consumer groups
* Offsets
* Lag

---

# Step 38: Additional Challenges

After completing the core lab, consider these extensions.

## Replace In-Memory Processing

Use:

* Apache Flink
* Kafka Streams
* Spark Structured Streaming

---

## Add PostgreSQL

```text
Kafka
   ↓
Processor
   ↓
PostgreSQL
```

Persist streaming features for analysis.

---

## Add Redis

```text
Kafka
   ↓
Processor
   ↓
Redis
   ↓
Low-Latency Feature Lookup
```

---

## Containerize the Applications

Add:

```text
producer
processor
consumer
```

to Docker Compose.

Then the entire environment becomes:

```text
Docker Compose
├── Kafka
├── Kafka UI
├── Producer
├── Processor
└── Consumer
```

---

## Add Monitoring

Use:

```text
Prometheus
+
Grafana
```

to monitor:

* Throughput
* Lag
* Errors
* Processing rate
* DLQ volume

---

# Step 39: Clean Up

Stop the three Python applications using:

```text
Ctrl+C
```

Return to the project root.

Stop Kafka and remove the local Kafka volume:

```bash
docker compose down -v
```

If you want to preserve the Kafka data:

```bash
docker compose down
```

Deactivate the Python environment:

```bash
deactivate
```

---

# Recommended Repository Structure

For the standalone lab:

```text
kafka-streaming-lab/
├── docker-compose.yml
└── app/
    ├── requirements.txt
    ├── producer.py
    ├── processor.py
    └── consumer.py
```

For your book companion repository:

```text
chapter-23/
├── lab-07-build-a-real-time-streaming-pipeline-with-apache-kafka.md
├── lab-07-docker-compose.yml
├── lab-07-requirements.txt
├── lab-07-producer.py
├── lab-07-processor.py
└── lab-07-consumer.py
```

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Started Kafka locally.
* [ ] Started Kafka UI.
* [ ] Verified the `local` Kafka cluster.
* [ ] Created the `transactions` topic.
* [ ] Created the `features` topic.
* [ ] Created the `dlq-transactions` topic.
* [ ] Installed the Python dependencies.
* [ ] Created the event producer.
* [ ] Published events continuously.
* [ ] Used `user_id` as the Kafka key.
* [ ] Observed raw events in Kafka UI.
* [ ] Created the stream processor.
* [ ] Validated events with Pydantic.
* [ ] Implemented fixed-window aggregation.
* [ ] Generated `event_count`.
* [ ] Generated `purchase_count`.
* [ ] Generated `revenue`.
* [ ] Published results to `features`.
* [ ] Created the feature consumer.
* [ ] Read derived features.
* [ ] Optionally persisted features to CSV.
* [ ] Inspected the consumer group.
* [ ] Observed consumer lag.
* [ ] Sent an invalid event to the DLQ.
* [ ] Increased producer throughput.
* [ ] Changed window size.
* [ ] Tested processor restart and offset recovery.
* [ ] Explored schema evolution.
* [ ] Reviewed distributed-state limitations.
* [ ] Cleaned up the environment.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Explain the architecture of a real-time Kafka pipeline.
* Produce JSON events to Kafka.
* Use Kafka keys and partitions.
* Validate incoming streaming records.
* Implement simple windowed aggregation.
* Generate derived real-time features.
* Work with Kafka consumers and consumer groups.
* Explain consumer offsets and lag.
* Route invalid events to a DLQ.
* Experiment with throughput and backlogs.
* Understand basic schema-evolution concerns.
* Explain why in-memory state is not sufficient for distributed processing.
* Recognize when Flink, Kafka Streams, or Spark Structured Streaming should be used.
* Relate streaming infrastructure to AI feature pipelines.

---

# Key Takeaway

**Real-time AI infrastructure transforms continuously arriving events into validated, structured, and immediately useful information that can feed analytics and machine learning systems.**

The core architecture is:

```text
Events
   ↓
Kafka
   ↓
Validation
   ↓
Windowed Processing
   ↓
Feature Generation
   ↓
Downstream AI / Analytics
```

Kafka provides durable event transport and partitioned consumption, while the stream processor performs validation, aggregation, and feature generation. Consumer groups and offsets support recoverable processing, Kafka UI provides operational visibility, and the dead-letter queue separates malformed data from valid records.

The most important architectural limitation of this lab is deliberate:

```text
In-Memory Python State
          ↓
Good for Learning
          ↓
Not Distributed-State Infrastructure
```

Production streaming systems strengthen the pattern with **distributed state management, checkpoints, event-time semantics, schema governance, stronger delivery guarantees, security, and comprehensive observability**.

The broader AI infrastructure principle is:

```text
Continuous Events
       ↓
Reliable Streaming Platform
       ↓
Stateful Processing
       ↓
Fresh Features
       ↓
Real-Time AI
```
