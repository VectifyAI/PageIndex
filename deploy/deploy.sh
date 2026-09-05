#!/usr/bin/env bash
# deploy/deploy.sh — build and deploy the books-mcp Cloud Run service.
# Usage: deploy/deploy.sh <GCP_PROJECT_ID> [REGION]
set -euo pipefail

PROJECT_ID="${1:?Usage: deploy.sh <GCP_PROJECT_ID> [REGION]}"
REGION="${2:-us-central1}"
SERVICE=books-mcp
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOK_LIBRARY_SRC="${BOOK_LIBRARY_SRC:-/Users/utkarshagrawal/github/repos/book-library}"

echo "Staging book-library data from $BOOK_LIBRARY_SRC ..."
rm -rf "$REPO_ROOT/deploy/_book_library_data"
mkdir -p "$REPO_ROOT/deploy/_book_library_data"
rsync -a --exclude='.git' "$BOOK_LIBRARY_SRC/" "$REPO_ROOT/deploy/_book_library_data/"
cp "$REPO_ROOT/deploy/Dockerfile" "$REPO_ROOT/Dockerfile"
trap 'rm -rf "$REPO_ROOT/deploy/_book_library_data" "$REPO_ROOT/Dockerfile"' EXIT

echo "Building via Cloud Build ..."
gcloud builds submit "$REPO_ROOT" \
  --project "$PROJECT_ID" \
  --tag "gcr.io/$PROJECT_ID/$SERVICE"

echo "Deploying to Cloud Run ..."
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --image "gcr.io/$PROJECT_ID/$SERVICE" \
  --allow-unauthenticated \
  --platform managed \
  --memory 512Mi \
  --max-instances 2

gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" --region "$REGION" \
  --format='value(status.url)'
