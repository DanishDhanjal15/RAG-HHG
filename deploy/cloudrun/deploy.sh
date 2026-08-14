#!/usr/bin/env bash
# Deploy the voice-RAG app to Google Cloud Run.
#
# Why Cloud Run rather than baking a VM: it scales to zero, so an idle demo costs
# nothing and stays inside the free tier (2M requests, 360k GB-s, 180k vCPU-s per
# month). A hackathon demo is idle ~99% of the time, which is exactly the shape
# Cloud Run is cheap for.
#
# Two decisions worth knowing before you read the flags:
#
# 1. The index is NOT baked into the image. It is fetched from the Hugging Face
#    repo at boot. Baking it would mean pushing ~700 MB from a home connection on
#    every deploy; fetching it means a datacenter-to-CDN download measured in
#    seconds. The image stays small and code deploys stay fast.
#
# 2. min-instances stays at 0. Setting it to 1 removes cold starts but bills for
#    a container running 720 h/month against a 180k vCPU-second (=50 h) free
#    allowance -- i.e. it would leave the free tier immediately. Cold start is
#    ~40-60 s; `--cpu-boost` and the startup probe below absorb it.
#
# Usage:
#   ./deploy.sh YOUR_GCP_PROJECT_ID
set -euo pipefail

PROJECT="${1:?usage: ./deploy.sh <gcp-project-id>}"
REGION="${REGION:-asia-south1}"          # Mumbai — closest to Sarvam's API and to Goa
SERVICE="${SERVICE:-vrag}"
REPO="${REPO:-vrag}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/app"

echo "==> project=${PROJECT} region=${REGION} service=${SERVICE}"

gcloud config set project "${PROJECT}" --quiet

echo "==> enabling APIs (idempotent, slow the first time)"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com --quiet

echo "==> artifact registry"
gcloud artifacts repositories describe "${REPO}" --location="${REGION}" --quiet >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location="${REGION}" \
  --description="voice RAG images" --quiet

# The Sarvam key goes in Secret Manager, not --set-env-vars: env vars are visible
# in the console, in `gcloud run services describe`, and in deploy logs.
echo "==> secret"
if ! gcloud secrets describe vrag-sarvam-key --quiet >/dev/null 2>&1; then
  if [[ -z "${VRAG_SARVAM_API_KEY:-}" ]]; then
    echo "!! set VRAG_SARVAM_API_KEY in your shell first, or the mic stays disabled"
    exit 1
  fi
  printf '%s' "${VRAG_SARVAM_API_KEY}" | \
    gcloud secrets create vrag-sarvam-key --data-file=- --quiet
fi

# Let the Cloud Run runtime service account read it.
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT}" --format='value(projectNumber)')"
gcloud secrets add-iam-policy-binding vrag-sarvam-key \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor --quiet >/dev/null

echo "==> building image with Cloud Build"
# Built in the cloud rather than locally: no dependency on a working local Docker,
# and .gcloudignore keeps the uploaded context to a couple of MB.
gcloud builds submit --tag "${IMAGE}" --timeout=40m --quiet

echo "==> deploying"
gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=3 \
  --concurrency=8 \
  --timeout=120 \
  --cpu-boost \
  --set-secrets="VRAG_SARVAM_API_KEY=vrag-sarvam-key:latest" \
  --quiet

URL="$(gcloud run services describe "${SERVICE}" --region="${REGION}" --format='value(status.url)')"
echo
echo "==> live: ${URL}"
echo "==> health: ${URL}/api/health"
echo
echo "First request wakes a cold container: index download + model load, ~40-60s."
echo "Verify before sharing the link:"
echo "  curl -s ${URL}/api/health"
