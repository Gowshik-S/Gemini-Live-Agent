#!/usr/bin/env bash
# =============================================================================
# Rio — One-command Cloud Run deployment
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - A GCP project with Cloud Run, Cloud Build, and Secret Manager enabled
#   - The secret "gemini-api-key" created in Secret Manager:
#       echo -n "YOUR_KEY" | gcloud secrets create gemini-api-key --data-file=-
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these for your environment
# ---------------------------------------------------------------------------
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${RIO_SERVICE_NAME:-rio-cloud}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "==> Pre-flight checks"

if ! command -v gcloud &>/dev/null; then
    echo "ERROR: gcloud CLI not found.  Install it from https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if [[ "${PROJECT_ID}" == "your-gcp-project-id" ]]; then
    echo "ERROR: Set GCP_PROJECT_ID or edit PROJECT_ID in this script."
    echo "  export GCP_PROJECT_ID=my-project"
    exit 1
fi

# Ensure we're pointing at the right project
gcloud config set project "${PROJECT_ID}" --quiet

echo "  Project : ${PROJECT_ID}"
echo "  Region  : ${REGION}"
echo "  Service : ${SERVICE_NAME}"
echo "  Image   : ${IMAGE}"
echo ""

# ---------------------------------------------------------------------------
# Step 1 — Build container image with Cloud Build
# ---------------------------------------------------------------------------
echo "==> Building container image via Cloud Build"

# Submit from the cloud/ directory where the Dockerfile lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOUD_DIR="${SCRIPT_DIR}/cloud"

if [[ ! -d "${CLOUD_DIR}" ]]; then
    echo "ERROR: cloud/ directory not found at ${CLOUD_DIR}"
    exit 1
fi

gcloud builds submit "${CLOUD_DIR}" \
    --tag "${IMAGE}" \
    --project "${PROJECT_ID}" \
    --quiet

echo "  Image built: ${IMAGE}"

# ---------------------------------------------------------------------------
# Step 2 — Deploy to Cloud Run
# ---------------------------------------------------------------------------
echo "==> Deploying to Cloud Run"

gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --platform managed \
    --min-instances 1 \
    --max-instances 5 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --session-affinity \
    --no-cpu-throttling \
    --allow-unauthenticated \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
    --quiet

# ---------------------------------------------------------------------------
# Step 3 — Print the service URL
# ---------------------------------------------------------------------------
echo ""
echo "==> Deployment complete!"

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --format "value(status.url)")

WS_URL="${SERVICE_URL/https:/wss:}/ws/rio/live"

echo ""
echo "  HTTP : ${SERVICE_URL}"
echo "  WS   : ${WS_URL}"
echo ""
echo "Update your rio/config.yaml:"
echo "  cloud_url: \"${WS_URL}\""
echo ""
echo "Done."
