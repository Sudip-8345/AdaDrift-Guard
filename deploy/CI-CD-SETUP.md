# CI/CD Setup — Self-Healing MLOps → AWS ECR

## Pipeline Overview

```
Push to main → Lint & Test → Build Docker Images → Push to Amazon ECR
```

| Stage            | Trigger       | What it does                                    |
|:-----------------|:--------------|:------------------------------------------------|
| **Test**         | push & PR     | flake8 lint, pytest, Docker build verification  |
| **Build & Push** | push to main  | Build API + Streamlit images, push to Amazon ECR|

---

## Required GitHub Secrets

Go to **Settings → Secrets and variables → Actions** in your repo and create these:

| Secret                 | Description                                         |
|:-----------------------|:----------------------------------------------------|
| `AWS_ACCESS_KEY_ID`    | IAM user access key with ECR permissions             |
| `AWS_SECRET_ACCESS_KEY`| IAM user secret key                                  |

### Optional GitHub Variable

| Variable      | Default      | Description          |
|:--------------|:-------------|:---------------------|
| `AWS_REGION`  | `us-east-1`  | AWS region for ECR   |

---

## AWS IAM Policy (minimum permissions)

Attach this policy to the IAM user whose credentials you add as secrets:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:DescribeRepositories",
        "ecr:CreateRepository",
        "ecr:TagResource"
      ],
      "Resource": "*"
    }
  ]
}
```

---

## ECR Repositories

Two repositories are auto-created by the workflow on first push:

| Repository                        | Image                              |
|:----------------------------------|:-----------------------------------|
| `self-healing-mlops-api`          | FastAPI backend (port 8000)        |
| `self-healing-mlops-streamlit`    | Streamlit dashboard (port 8501)    |

Each push tags images with both the commit SHA and `latest`.

### Pulling images manually

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Pull images
docker pull <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/self-healing-mlops-api:latest
docker pull <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/self-healing-mlops-streamlit:latest
```

---

## File Structure

```
.github/
  workflows/
    ci-cd.yml              ← GitHub Actions workflow (test + push to ECR)
tests/
  test_smoke.py            ← Smoke tests (import checks, health endpoint)
.dockerignore              ← Keeps images lean
```

---

## Local Testing

```bash
# Run tests locally
python -m pytest tests/ -v

# Lint locally
flake8 app/ src/ utils/ monitoring/ --max-line-length=120

# Build images locally
docker compose build
docker compose up
```
