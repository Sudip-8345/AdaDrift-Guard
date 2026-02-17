#!/bin/bash
# ──────────────────────────────────────────────────────────────
# EC2 Deployment Script for Self-Healing MLOps
# Called by GitHub Actions after images are pushed to ECR.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

DEPLOY_DIR="/home/${USER}/self-healing-mlops"
COMPOSE_FILE="${DEPLOY_DIR}/docker-compose.prod.yml"

echo "===== Deployment started at $(date) ====="

# ── 1. Install Docker & Docker Compose if missing ────────────
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo yum update -y 2>/dev/null || sudo apt-get update -y
    sudo yum install -y docker 2>/dev/null || sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to re-login for group changes."
fi

if ! command -v docker compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose plugin..."
    DOCKER_COMPOSE_VERSION="v2.24.5"
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -SL "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-$(uname -m)" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

# ── 2. Install & configure AWS CLI if missing ────────────────
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
    unzip -qo /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update
    rm -rf /tmp/aws /tmp/awscliv2.zip
fi

# ── 3. ECR Login ─────────────────────────────────────────────
echo "Logging into ECR (${AWS_REGION})..."
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# ── 4. Export image tags for docker-compose ───────────────────
export API_IMAGE="${ECR_REGISTRY}/${ECR_REPOSITORY_API}:${IMAGE_TAG}"
export STREAMLIT_IMAGE="${ECR_REGISTRY}/${ECR_REPOSITORY_STREAMLIT}:${IMAGE_TAG}"

echo "API Image:       ${API_IMAGE}"
echo "Streamlit Image: ${STREAMLIT_IMAGE}"

# ── 5. Create persistent directories ─────────────────────────
mkdir -p "${DEPLOY_DIR}/models" "${DEPLOY_DIR}/data"

# ── 6. Pull latest images ────────────────────────────────────
echo "Pulling images..."
docker pull "${API_IMAGE}"
docker pull "${STREAMLIT_IMAGE}"

# ── 7. Stop existing containers & deploy ──────────────────────
cd "${DEPLOY_DIR}"
echo "Stopping existing containers..."
docker compose -f "${COMPOSE_FILE}" down --remove-orphans 2>/dev/null || true

echo "Starting services..."
docker compose -f "${COMPOSE_FILE}" up -d

# ── 8. Health check ──────────────────────────────────────────
echo "Waiting for API health check..."
MAX_RETRIES=15
RETRY=0
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    RETRY=$((RETRY + 1))
    if [ $RETRY -ge $MAX_RETRIES ]; then
        echo "ERROR: API failed to start after ${MAX_RETRIES} attempts"
        docker compose -f "${COMPOSE_FILE}" logs api
        exit 1
    fi
    echo "  Attempt ${RETRY}/${MAX_RETRIES}..."
    sleep 5
done

echo "API is healthy!"

# ── 9. Cleanup old images ────────────────────────────────────
echo "Cleaning up dangling images..."
docker image prune -f

echo "===== Deployment completed at $(date) ====="
