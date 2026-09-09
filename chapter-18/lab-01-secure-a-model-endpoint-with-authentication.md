# Hands-On Lab: Secure a Model Endpoint with Authentication

In this lab, you will secure a FastAPI-based machine learning inference endpoint using several common API-security controls.

The API will support:

* JWT authentication for user requests
* API-key authentication for service-to-service requests
* Rate limiting
* Environment-based secrets management
* Docker containerization
* Kubernetes Secrets
* Health and readiness probes
* TLS-ready deployment architecture

The objective is to demonstrate how a basic model-serving API can be evolved into a more production-oriented endpoint.

---

## Lab Objective

Build a secured inference service that supports two distinct authentication paths:

```text
Authenticated User
      ↓
JWT
      ↓
/predict/user


Service Client
      ↓
API Key
      ↓
/predict/service
```

Both inference paths are protected by rate limiting.

Sensitive values such as:

```text
JWT_SECRET
API_KEY
```

are supplied at runtime rather than embedded directly in source code.

---

## Architecture

The completed architecture follows this general pattern:

```text
                    Clients
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
       User Client             Service Client
          │                         │
          │ JWT                     │ API Key
          ▼                         ▼
    /predict/user            /predict/service
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
                  Rate Limiter
                       │
                       ▼
                  FastAPI API
                       │
                       ▼
               Inference Logic
                       │
                       ▼
                  Prediction
```

For Kubernetes deployment:

```text
Client
   ↓
TLS / Ingress
   ↓
Kubernetes Service
   ↓
FastAPI Pod
   │
   ├── JWT authentication
   ├── API-key authentication
   ├── Rate limiting
   ├── /healthz
   └── /readyz
        │
        ▼
Kubernetes Secrets
```

---

## Estimated Time

**Approximately 90–120 minutes**

---

## Tools

This lab uses:

* Python 3.10 or later
* FastAPI
* Uvicorn
* Python JOSE
* Passlib
* SlowAPI
* Docker
* Optional Kubernetes
* Optional Kubernetes Ingress controller
* `kubectl`

---

# Step 1: Create the Project

Create a project directory:

```bash
mkdir -p lab_secure_endpoint/app
mkdir -p lab_secure_endpoint/k8s

cd lab_secure_endpoint
```

Use this structure:

```text
lab_secure_endpoint/
├── app/
│   └── main.py
├── k8s/
│   └── deployment.yaml
├── requirements.txt
└── Dockerfile
```

For the companion GitHub repository, I recommend:

```text
chapter-18/
├── lab-07-secure-a-model-endpoint-with-authentication.md
├── lab-07-main.py
├── lab-07-Dockerfile
└── lab-07-k8s-deployment.yaml
```

---

# Step 2: Create the FastAPI Application

Create:

```text
app/main.py
```

Add:

```python
import os
import time

from fastapi import (
    APIKeyHeader,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Security,
)
from fastapi.responses import JSONResponse
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
)
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SECRET_KEY = os.getenv(
    "JWT_SECRET",
    "change-me"
)

API_KEY = os.getenv(
    "API_KEY",
    "super-secret"
)

ALGORITHM = "HS256"
TOKEN_EXPIRE_SECONDS = 3600


# ---------------------------------------------------------
# FastAPI and Rate Limiting
# ---------------------------------------------------------

limiter = Limiter(
    key_func=get_remote_address
)

app = FastAPI(
    title="Secure Model Inference API",
    version="1.0"
)

app.state.limiter = limiter


@app.exception_handler(
    RateLimitExceeded
)
async def rate_limit_handler(
    request: Request,
    exc: RateLimitExceeded
):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded"
        }
    )


# ---------------------------------------------------------
# Authentication Configuration
# ---------------------------------------------------------

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token"
)

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


# ---------------------------------------------------------
# Demo Users
# ---------------------------------------------------------

fake_users = {
    "alice": {
        "username": "alice",
        "hashed_pw": pwd_context.hash(
            "alicepass"
        ),
    },

    "admin": {
        "username": "admin",
        "hashed_pw": pwd_context.hash(
            "adminpass"
        ),
    },
}


# ---------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------

class Token(BaseModel):
    access_token: str
    token_type: str


class Features(BaseModel):
    features: list[float]


# ---------------------------------------------------------
# Authentication Functions
# ---------------------------------------------------------

def authenticate_user(
    username: str,
    password: str
):
    user = fake_users.get(
        username
    )

    if not user:
        return None

    if not pwd_context.verify(
        password,
        user["hashed_pw"]
    ):
        return None

    return user


def create_access_token(
    data: dict,
    expires_in: int = TOKEN_EXPIRE_SECONDS
):
    payload = data.copy()

    payload["exp"] = int(
        time.time() + expires_in
    )

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


async def get_current_user(
    token: str = Depends(
        oauth2_scheme
    )
):
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        username = payload.get(
            "sub"
        )

        if not username:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )

        return username

    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )


async def validate_api_key(
    api_key: str = Security(
        api_key_header
    )
):
    if api_key == API_KEY:
        return api_key

    raise HTTPException(
        status_code=403,
        detail="Invalid API key"
    )


# ---------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------

@app.post(
    "/token",
    response_model=Token
)
async def login(
    form: OAuth2PasswordRequestForm = Depends()
):
    user = authenticate_user(
        form.username,
        form.password
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Bad credentials"
        )

    token = create_access_token(
        {
            "sub": form.username
        }
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }


# ---------------------------------------------------------
# Health Endpoints
# ---------------------------------------------------------

@app.get("/healthz")
def health():
    return {
        "status": "ok"
    }


@app.get("/readyz")
def ready():
    return {
        "status": "ready"
    }


# ---------------------------------------------------------
# Protected Inference Endpoints
# ---------------------------------------------------------

@app.post("/predict/user")
@limiter.limit("30/minute")
async def predict_user(
    request: Request,
    data: Features,
    user: str = Depends(
        get_current_user
    ),
):
    score = sum(
        data.features
    )

    prediction = (
        "class_A"
        if score > 5
        else "class_B"
    )

    return {
        "authenticated_user": user,
        "prediction": prediction
    }


@app.post("/predict/service")
@limiter.limit("30/minute")
async def predict_service(
    request: Request,
    data: Features,
    api_key: str = Depends(
        validate_api_key
    ),
):
    score = sum(
        data.features
    )

    prediction = (
        "class_A"
        if score > 5
        else "class_B"
    )

    return {
        "authentication": "api_key",
        "prediction": prediction
    }
```

---

## Authentication Design

The lab intentionally separates the two authentication mechanisms.

### User Authentication

```text
Username + Password
       ↓
POST /token
       ↓
JWT Issued
       ↓
Authorization: Bearer <token>
       ↓
/predict/user
```

### Service Authentication

```text
Service
   ↓
X-API-Key
   ↓
/predict/service
```

This makes the authentication model clearer than requiring both credentials on every request.

---

# Step 3: Create the Dependencies

Create:

```text
requirements.txt
```

Add:

```text
fastapi
uvicorn[standard]
python-jose
passlib[bcrypt]
slowapi
python-multipart
```

Install:

```bash
pip install \
  -r requirements.txt
```

`python-multipart` is required because FastAPI's OAuth2 password flow accepts username and password as form data.

---

# Step 4: Configure Secrets

Set the JWT signing secret:

```bash
export JWT_SECRET='change-me'
```

Set the service API key:

```bash
export API_KEY='super-secret'
```

The application reads them using:

```python
os.getenv()
```

rather than embedding operational secrets directly into business logic.

The configuration pattern is:

```text
Environment
    ↓
JWT_SECRET
API_KEY
    ↓
FastAPI Application
```

> **Production Note**
>
> The example values are intentionally simple. Production secrets should be random, securely stored, rotated, and delivered through an approved secrets-management system.

---

# Step 5: Run the API Locally

Start:

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000
```

---

## Check Health

Run:

```bash
curl \
  http://localhost:8000/healthz
```

Expected:

```json
{
  "status": "ok"
}
```

---

## Check Readiness

Run:

```bash
curl \
  http://localhost:8000/readyz
```

Expected:

```json
{
  "status": "ready"
}
```

These endpoints remain unauthenticated because infrastructure components such as Kubernetes and load balancers may need to access them automatically.

---

# Step 6: Obtain a JWT

Use the demo account:

```text
Username: alice
Password: alicepass
```

Request a token:

```bash
curl -X POST \
  http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=alice&password=alicepass"
```

A response should resemble:

```json
{
  "access_token": "<JWT_TOKEN>",
  "token_type": "bearer"
}
```

Store it:

```bash
TOKEN="<paste-JWT-token>"
```

---

## JWT Flow

```text
Credentials
    ↓
Authentication
    ↓
Signed JWT
    ↓
Client Stores Token
    ↓
Bearer Token on Requests
```

The token contains an expiration value and is cryptographically signed.

---

# Step 7: Call the User-Protected Endpoint

Send:

```bash
curl -X POST \
  http://localhost:8000/predict/user \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

Expected:

```json
{
  "authenticated_user": "alice",
  "prediction": "class_A"
}
```

Try without a JWT:

```bash
curl -X POST \
  http://localhost:8000/predict/user \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

The request should be rejected.

---

# Step 8: Call the Service-Protected Endpoint

Use:

```bash
curl -X POST \
  http://localhost:8000/predict/service \
  -H "X-API-Key: super-secret" \
  -H "Content-Type: application/json" \
  -d '{"features":[6.0,2.2,4.0,1.0]}'
```

Expected:

```json
{
  "authentication": "api_key",
  "prediction": "class_A"
}
```

Try an invalid key:

```bash
curl -X POST \
  http://localhost:8000/predict/service \
  -H "X-API-Key: invalid-key" \
  -H "Content-Type: application/json" \
  -d '{"features":[6.0,2.2,4.0,1.0]}'
```

Expected:

```text
403 Forbidden
```

---

## API-Key Security

API keys are convenient for controlled service integrations, but long-lived static credentials require careful management.

A more mature architecture may evolve toward:

```text
Static API Key
      ↓
Rotated API Key
      ↓
Short-Lived Credential
      ↓
Workload Identity / OAuth / mTLS
```

---

# Step 9: Test Rate Limiting

Both inference endpoints use:

```text
30 requests per minute
```

If a client exceeds the configured limit:

```text
HTTP 429 Too Many Requests
```

The protection path is:

```text
Request
   ↓
Rate Limiter
   │
   ├── Within Limit → Continue
   │
   └── Limit Exceeded → HTTP 429
```

Rate limiting can reduce:

* Accidental overload
* Excessive API usage
* Automated abuse
* Some denial-of-service patterns

Production limits should reflect:

* Model latency
* GPU capacity
* Traffic profile
* Client type
* SLA/SLO requirements

---

# Step 10: Containerize the Application

Create:

```text
Dockerfile
```

Add:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD [
    "uvicorn",
    "app.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]
```

Build:

```bash
docker build \
  -t secure-endpoint \
  .
```

Run:

```bash
docker run \
  --rm \
  -p 8000:8000 \
  -e JWT_SECRET='change-me' \
  -e API_KEY='super-secret' \
  secure-endpoint
```

Verify:

```bash
curl \
  http://localhost:8000/healthz
```

---

## Why Runtime Secrets Matter

The container image contains:

```text
Application Code
Dependencies
Runtime
```

but not environment-specific secrets.

Instead:

```text
Container Image
      +
Runtime Secrets
      ↓
Running Application
```

This separation is important because images may be:

* Stored in registries
* Cached on worker nodes
* Scanned
* Shared
* Reused across environments

---

# Step 11: Deploy to Kubernetes

Create:

```text
lab-07-k8s-deployment.yaml
```

Add:

```yaml
apiVersion: apps/v1
kind: Deployment

metadata:
  name: secure-ml
  namespace: ai-sec

spec:
  replicas: 1

  selector:
    matchLabels:
      app: secure-ml

  template:
    metadata:
      labels:
        app: secure-ml

    spec:
      containers:
        - name: api
          image: YOUR_REGISTRY/secure-endpoint:latest

          ports:
            - containerPort: 8000

          env:
            - name: JWT_SECRET
              valueFrom:
                secretKeyRef:
                  name: secure-ml-secrets
                  key: JWT_SECRET

            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: secure-ml-secrets
                  key: API_KEY

          readinessProbe:
            httpGet:
              path: /readyz
              port: 8000

            initialDelaySeconds: 5
            periodSeconds: 10

          livenessProbe:
            httpGet:
              path: /healthz
              port: 8000

            initialDelaySeconds: 10
            periodSeconds: 20

---
apiVersion: v1
kind: Service

metadata:
  name: secure-ml-svc
  namespace: ai-sec

spec:
  selector:
    app: secure-ml

  ports:
    - port: 80
      targetPort: 8000
```

---

# Step 12: Create the Kubernetes Namespace

Run:

```bash
kubectl create namespace ai-sec
```

Verify:

```bash
kubectl get namespace ai-sec
```

---

# Step 13: Create Kubernetes Secrets

Create:

```bash
kubectl -n ai-sec create secret generic \
  secure-ml-secrets \
  --from-literal=JWT_SECRET='change-me' \
  --from-literal=API_KEY='super-secret'
```

Verify:

```bash
kubectl -n ai-sec get secret \
  secure-ml-secrets
```

The deployment retrieves the values at runtime:

```text
Kubernetes Secret
      ↓
Environment Variables
      ↓
FastAPI Container
```

> **Production Note**
>
> Kubernetes Secrets demonstrate the pattern, but production environments may require encryption at rest and integration with an external secrets-management system.

---

# Step 14: Deploy the Application

Apply:

```bash
kubectl apply \
  -f lab-07-k8s-deployment.yaml
```

Verify:

```bash
kubectl -n ai-sec get pods
```

Check the Service:

```bash
kubectl -n ai-sec get service
```

Inspect the deployment if necessary:

```bash
kubectl -n ai-sec describe deployment \
  secure-ml
```

---

# Step 15: Understand Health and Readiness

The deployment uses separate probes.

## Liveness

```text
/healthz
```

answers:

> Is the process alive?

If liveness repeatedly fails, Kubernetes may restart the container.

---

## Readiness

```text
/readyz
```

answers:

> Is the application ready to receive traffic?

If readiness fails:

```text
Pod Running
    ↓
Readiness = False
    ↓
Removed from Service Traffic
```

This distinction is particularly important for AI services where model initialization may take significant time.

---

# Step 16: Add TLS with an Ingress

Authentication credentials and inference traffic should not travel across unencrypted external connections.

The production path should be:

```text
Client
   ↓
HTTPS
   ↓
Ingress / API Gateway
   ↓
TLS Termination
   ↓
Kubernetes Service
   ↓
Secure ML Endpoint
```

TLS protects:

* JWTs
* API keys
* Inference inputs
* Predictions
* Request metadata

Internal service-to-service environments can optionally add mTLS where stronger mutual authentication is required.

---

# Step 17: Production Hardening

The authentication implemented in this lab is intentionally simplified.

A production architecture should consider several additional controls.

---

## Use an Identity Provider

Instead of implementing user authentication locally through `/token`, production systems typically use:

```text
OpenID Connect
OAuth 2.0
```

An external identity provider can manage:

* Users
* MFA
* Password policies
* Token issuance
* Session policies
* Token revocation
* Credential lifecycle

---

## Replace Long-Lived API Keys

Prefer, where appropriate:

```text
Workload Identity
OAuth 2.0 Client Credentials
Short-Lived Tokens
mTLS
```

over indefinitely valid static service credentials.

---

## Apply Kubernetes Security Controls

Consider:

* RBAC
* Namespace isolation
* NetworkPolicies
* Least-privilege service accounts
* Pod security controls
* Image scanning
* Signed images

---

## Protect Against Resource Abuse

Use:

* Rate limiting
* API gateway
* Web application firewall
* Request-size limits
* Concurrency controls
* Timeouts
* Quotas

For expensive GPU inference, request controls can directly protect infrastructure cost.

---

## Centralize Security Logging

Capture useful security metadata such as:

```text
Request ID
Authenticated Identity
Endpoint
Response Status
Authentication Failure
Model Version
Timestamp
```

Avoid unnecessarily logging:

* Passwords
* API keys
* JWTs
* Sensitive inference data

Security logs can be forwarded to:

```text
SIEM
Security Analytics
Incident Management
```

---

## Monitor Security Signals

Useful alerts include:

* Repeated login failures
* Repeated invalid API keys
* Unusual request volume
* Excessive HTTP 429 responses
* Access from unexpected clients
* Unexpected endpoint use

The architecture becomes:

```text
API Request
    ↓
Authentication
    ↓
Authorization
    ↓
Rate Limiting
    ↓
Inference
    ↓
Security Logging
    ↓
Monitoring / SIEM
    ↓
Alert
```

---

# Step 18: Test the Complete Security Workflow

Verify each path.

| Test                 | Expected Result     |
| -------------------- | ------------------- |
| `/healthz`           | `200 OK`            |
| `/readyz`            | `200 OK`            |
| Valid login          | JWT returned        |
| Invalid login        | Rejected            |
| Valid JWT            | Prediction returned |
| Missing JWT          | Rejected            |
| Invalid JWT          | Rejected            |
| Valid API key        | Prediction returned |
| Invalid API key      | `403`               |
| Excess requests      | `429`               |
| Docker health        | Working             |
| Kubernetes readiness | Working             |
| Kubernetes liveness  | Working             |

---

# Step 19: Troubleshooting

## `/token` Fails

Verify that:

```text
python-multipart
```

is installed.

Check:

```bash
pip show python-multipart
```

---

## JWT Is Rejected

Verify:

* The client copied the complete token.
* The same JWT secret is used for signing and verification.
* The token has not expired.
* The request contains:

```text
Authorization: Bearer <token>
```

---

## API Key Is Rejected

Verify:

```bash
echo "$API_KEY"
```

and compare with:

```text
X-API-Key
```

sent by the client.

---

## Rate Limiting Does Not Trigger

Generate enough requests within the configured interval.

Also remember that the lab limiter is based on client address.

---

## Kubernetes Pod Does Not Start

Check:

```bash
kubectl -n ai-sec get pods
```

Then:

```bash
kubectl -n ai-sec describe pod \
  <pod-name>
```

And:

```bash
kubectl -n ai-sec logs \
  <pod-name>
```

Verify:

* Container image
* Secret name
* Secret keys
* Namespace
* Container port

---

## Readiness Probe Fails

Test inside the cluster or Pod:

```text
/readyz
```

Verify that the application is listening on:

```text
0.0.0.0:8000
```

---

# Step 20: Clean Up

Remove the deployment and Service:

```bash
kubectl delete \
  -f lab-07-k8s-deployment.yaml
```

Delete the Secret:

```bash
kubectl -n ai-sec delete secret \
  secure-ml-secrets
```

Delete the namespace:

```bash
kubectl delete namespace ai-sec
```

Stop the local Docker container if necessary:

```bash
docker ps
```

Then:

```bash
docker stop \
  <container-id>
```

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Created the FastAPI application.
* [ ] Created demo users.
* [ ] Implemented password verification.
* [ ] Implemented JWT generation.
* [ ] Implemented JWT validation.
* [ ] Protected `/predict/user`.
* [ ] Implemented API-key validation.
* [ ] Protected `/predict/service`.
* [ ] Added rate limiting.
* [ ] Added `/healthz`.
* [ ] Added `/readyz`.
* [ ] Moved secrets to environment variables.
* [ ] Obtained a JWT.
* [ ] Tested authenticated user inference.
* [ ] Tested unauthenticated rejection.
* [ ] Tested service API-key authentication.
* [ ] Tested invalid API-key rejection.
* [ ] Tested rate limiting.
* [ ] Containerized the API.
* [ ] Passed secrets to Docker at runtime.
* [ ] Created Kubernetes Secrets.
* [ ] Deployed the application to Kubernetes.
* [ ] Added readiness and liveness probes.
* [ ] Reviewed TLS architecture.
* [ ] Reviewed production OIDC/OAuth options.
* [ ] Reviewed workload identity and mTLS.
* [ ] Reviewed RBAC and NetworkPolicy controls.
* [ ] Reviewed centralized security logging.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Explain why model endpoints require authentication.
* Implement JWT-based authentication for user requests.
* Implement API-key authentication for service clients.
* Generate and validate signed JWTs.
* Protect FastAPI endpoints using dependency injection.
* Apply rate limiting to inference APIs.
* Move secrets out of application code.
* Pass secrets into Docker containers at runtime.
* Store runtime credentials using Kubernetes Secrets.
* Configure Kubernetes health and readiness probes.
* Explain why TLS is required for externally exposed inference APIs.
* Distinguish user authentication from service authentication.
* Understand the role of OIDC, OAuth 2.0, workload identity, and mTLS in production environments.
* Apply additional infrastructure controls such as RBAC, NetworkPolicies, and centralized logging.

---

# Key Takeaway

**Securing an AI model endpoint requires multiple complementary controls rather than a single authentication mechanism.**

A practical security architecture combines:

```text
Identity
   ↓
Authentication
   ↓
Authorization
   ↓
Rate Limiting
   ↓
TLS
   ↓
Secrets Management
   ↓
Secure Model Inference
   ↓
Logging and Monitoring
```

JWTs are suitable for authenticated user sessions, API keys can support controlled service integrations, and rate limiting helps protect expensive inference resources from abuse. Secrets should remain outside source code and container images, while Kubernetes or external secret-management systems deliver them securely at runtime.

In production, these basic controls should evolve toward enterprise identity providers, short-lived credentials, workload identity, mTLS, RBAC, network isolation, centralized security logging, and continuous monitoring.
