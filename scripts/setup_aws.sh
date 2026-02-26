#!/usr/bin/env bash
# setup_aws.sh — Bootstrap AWS resources for CodeColosseum training
#
# Usage:
#   bash scripts/setup_aws.sh [bucket-name] [region]
#
# What it does:
#   1. Validates AWS CLI is configured
#   2. Creates an S3 bucket (with public access blocked)
#   3. Uploads training data to s3://<bucket>/data/
#   4. Creates an IAM role with SageMaker + S3 permissions
#
# Prerequisites:
#   - AWS CLI installed and configured (aws configure)
#   - Training data in data/final/*.jsonl (run selector.py first)

set -euo pipefail

BUCKET="${1:-codecolosseum-demo}"
REGION="${2:-us-east-1}"
ROLE_NAME="SageMakerCodeColosseumRole"

echo "========================================"
echo "CodeColosseum AWS Setup"
echo "========================================"
echo "  Bucket : s3://${BUCKET}"
echo "  Region : ${REGION}"
echo "  Role   : ${ROLE_NAME}"
echo ""

# ── 1. Check AWS CLI is configured ───────────────────────────────────────────
if ! aws sts get-caller-identity &>/dev/null; then
  echo "ERROR: AWS CLI not configured. Run: aws configure"
  exit 1
fi
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account: ${ACCOUNT_ID}"

# ── 2. Create S3 bucket ───────────────────────────────────────────────────────
if aws s3 ls "s3://${BUCKET}" &>/dev/null; then
  echo "Bucket s3://${BUCKET} already exists — skipping."
else
  echo "Creating bucket: s3://${BUCKET}"
  if [ "${REGION}" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "${BUCKET}" --region "${REGION}"
  else
    aws s3api create-bucket \
      --bucket "${BUCKET}" \
      --region "${REGION}" \
      --create-bucket-configuration LocationConstraint="${REGION}"
  fi
  # Block all public access for security
  aws s3api put-public-access-block \
    --bucket "${BUCKET}" \
    --public-access-block-configuration \
      "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
  echo "Bucket created with public access blocked."
fi

# ── 3. Upload training data ───────────────────────────────────────────────────
echo ""
echo "Uploading training data..."
UPLOADED=0
for AGENT in coder reviewer; do
  LOCAL="data/final/${AGENT}_train.jsonl"
  S3_KEY="s3://${BUCKET}/data/${AGENT}_train.jsonl"
  if [ -f "${LOCAL}" ]; then
    aws s3 cp "${LOCAL}" "${S3_KEY}"
    echo "  Uploaded: ${LOCAL} -> ${S3_KEY}"
    UPLOADED=$((UPLOADED + 1))
  else
    echo "  WARNING: ${LOCAL} not found — run data pipeline first:"
    echo "           python -m src.data.selector --fast"
  fi
done

if [ "${UPLOADED}" -eq 0 ]; then
  echo ""
  echo "ERROR: No training data uploaded. Run the data pipeline first."
  exit 1
fi

# ── 4. Create IAM role ────────────────────────────────────────────────────────
echo ""
if aws iam get-role --role-name "${ROLE_NAME}" &>/dev/null; then
  echo "IAM role ${ROLE_NAME} already exists — skipping."
else
  echo "Creating IAM role: ${ROLE_NAME}"
  TRUST_POLICY='{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "${TRUST_POLICY}"
  aws iam attach-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
  aws iam attach-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
  echo "IAM role created."
fi

ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" --query Role.Arn --output text)

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo "  Role ARN : ${ROLE_ARN}"
echo "  S3 data  : s3://${BUCKET}/data/"
echo ""
echo "Next — launch training:"
echo "  python scripts/launch_sagemaker.py --agent coder --bucket ${BUCKET} --region ${REGION}"
