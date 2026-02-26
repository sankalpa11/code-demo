#!/usr/bin/env bash
# train_reviewer.sh — Train the Reviewer agent (locally or on SageMaker)
#
# Usage:
#   bash scripts/train_reviewer.sh              # local EC2 GPU training
#   bash scripts/train_reviewer.sh --sagemaker  # SageMaker managed job
#
# Flags:
#   --sagemaker          Launch a SageMaker training job instead of local
#   --bucket NAME        S3 bucket (default: codecolosseum-demo)
#   --region REGION      AWS region (default: us-east-1)
#   --data-dir PATH      Local data dir (default: data/final)

set -euo pipefail

MODE="local"
BUCKET="codecolosseum-demo"
REGION="us-east-1"
DATA_DIR="data/final"
INSTANCE="ml.g5.xlarge"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sagemaker) MODE="sagemaker"; shift ;;
    --bucket)    BUCKET="$2"; shift 2 ;;
    --region)    REGION="$2"; shift 2 ;;
    --data-dir)  DATA_DIR="$2"; shift 2 ;;
    --instance)  INSTANCE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=false

echo "========================================"
echo "CodeColosseum — Train Reviewer Agent"
echo "Mode   : ${MODE}"
echo "========================================"

if [ "${MODE}" = "local" ]; then
  DATA_FILE="${DATA_DIR}/reviewer_train.jsonl"
  if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Data file not found: ${DATA_FILE}"
    echo "Run:   python -m src.data.selector --fast"
    exit 1
  fi
  python -m src.training.trainer --agent reviewer --data-dir "${DATA_DIR}"

elif [ "${MODE}" = "sagemaker" ]; then
  echo "Launching SageMaker training job for reviewer..."
  AGENT_TYPE=reviewer \
  S3_BUCKET="${BUCKET}" \
  AWS_REGION="${REGION}" \
  python3 scripts/launch_sagemaker.py --agent reviewer --bucket "${BUCKET}" --region "${REGION}" --instance "${INSTANCE}"
fi
