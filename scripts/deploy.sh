#!/usr/bin/env bash
# deploy.sh — Upload trained adapters to S3 and start the FastAPI server
#
# Usage:
#   bash scripts/deploy.sh [--bucket NAME] [--port PORT]
#
# What it does:
#   1. Validates that both adapter directories exist
#   2. Syncs adapters to s3://<bucket>/adapters/
#   3. Starts the FastAPI inference server

set -euo pipefail

BUCKET="codecolosseum-demo"
PORT=8000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket) BUCKET="$2"; shift 2 ;;
    --port)   PORT="$2";   shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "========================================"
echo "CodeColosseum Deploy"
echo "  Bucket : s3://${BUCKET}"
echo "  Port   : ${PORT}"
echo "========================================"

# ── 1. Validate adapters ──────────────────────────────────────────────────────
MISSING=0
for AGENT in coder reviewer; do
  ADAPTER_DIR="data/adapters/${AGENT}_adapter"
  if [ ! -d "${ADAPTER_DIR}" ]; then
    echo "ERROR: Adapter not found: ${ADAPTER_DIR}"
    MISSING=$((MISSING + 1))
  fi
done

if [ "${MISSING}" -gt 0 ]; then
  echo ""
  echo "Train the missing agents first:"
  echo "  bash scripts/train_coder.sh"
  echo "  bash scripts/train_reviewer.sh"
  exit 1
fi

# ── 2. Upload adapters to S3 ─────────────────────────────────────────────────
if command -v aws &>/dev/null; then
  echo "Syncing adapters to s3://${BUCKET}/adapters/ ..."
  aws s3 sync data/adapters/ "s3://${BUCKET}/adapters/" \
    --exclude "*.tmp" \
    --exclude "__pycache__/*"
  echo "Adapters uploaded."
else
  echo "WARNING: AWS CLI not found — skipping S3 upload (running locally only)."
fi

# ── 3. Start API server ───────────────────────────────────────────────────────
echo ""
echo "Starting FastAPI server on port ${PORT}..."
echo "API docs at: http://localhost:${PORT}/docs"
exec uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  --log-level info
