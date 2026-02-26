"""
Script to launch CodeColosseum training jobs on AWS SageMaker.

What it does:
  1. Reads configuration from config/settings.py
  2. Packages src/ and config/ into a source archive
  3. Launches a SageMaker PyTorch estimator
  4. Passes hyperparameters as environment variables

Usage:
  python scripts/launch_sagemaker.py --agent coder --bucket my-bucket
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import (
    AWS_REGION,
    BASE_MODEL,
    LORA_CONFIG,
    S3_BUCKET,
    TRAINING_CONFIG,
)


import argparse
import os
from pathlib import Path

# ...

def launch_job(
    agent_type: str,
    bucket: str,
    region: str,
    instance_type: str = "ml.g5.xlarge",
) -> str:
    """Launch a SageMaker training job."""
    print(f"==> Launching SageMaker job for {agent_type}")
    
    # Explicit session management for debugging
    boto_sess = boto3.Session(region_name=region)
    creds = boto_sess.get_credentials()
    if creds:
        print(f"    AWS Identity : {creds.access_key[:5]}... (from {creds.method})")
        if creds.token:
            print(f"    AWS Token    : [PRESENT]")
        else:
            print(f"    AWS Token    : [NONE]")
    else:
        print("    AWS Identity : [NOT FOUND]")

    sess = sagemaker.Session(boto_session=boto_sess)
    
    # Get account ID through STS explicitly
    sts = boto_sess.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    
    # Use the role created by setup_aws.sh
    role = f"arn:aws:iam::{account_id}:role/SageMakerCodeColosseumRole"
    
    # S3 path for outputs
    output_path = f"s3://{bucket}/output"
    checkpoint_path = f"s3://{bucket}/checkpoints/{agent_type}"

    # Hyperparameters from config/settings.py
    # SageMaker passes these as environment variables to the container
    hyperparameters = {
        "AGENT_TYPE": agent_type,
        "NUM_EPOCHS": str(TRAINING_CONFIG["num_epochs"]),
        "BATCH_SIZE": str(TRAINING_CONFIG["batch_size"]),
        "LEARNING_RATE": str(TRAINING_CONFIG["learning_rate"]),
        "LORA_R": str(LORA_CONFIG["r"]),
    }

    estimator = PyTorch(
        entry_point="sagemaker_entry.py",
        source_dir="src/training",
        role=role,
        framework_version="2.1.0",
        py_version="py310",
        instance_count=1,
        instance_type=instance_type,
        output_path=output_path,
        checkpoint_s3_uri=checkpoint_path,
        hyperparameters=hyperparameters,
        # Required packages
        dependencies=["requirements.txt"],
        environment={
            "PYTHONPATH": "/opt/ml/code",
            "S3_BUCKET": bucket,
            "AWS_REGION": region,
        }
    )

    # Input data channel
    data_uri = f"s3://{bucket}/data/"
    
    print(f"Starting job with data from {data_uri}...")
    estimator.fit({"train": data_uri}, wait=False)
    
    job_name = estimator.latest_training_job.name
    print(f"Job launched successfully: {job_name}")
    print(f"Monitor at: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
    
    return job_name


def main():
    parser = argparse.ArgumentParser(description="Launch CodeColosseum training on SageMaker")
    parser.add_argument("--agent", choices=["coder", "reviewer"], required=True)
    parser.add_argument("--bucket", default=S3_BUCKET)
    parser.add_argument("--region", default=AWS_REGION)
    parser.add_argument("--instance", default="ml.g5.xlarge", help="g5.xlarge is recommended (24GB VRAM)")
    
    args = parser.parse_args()
    
    launch_job(
        agent_type=args.agent,
        bucket=args.bucket,
        region=args.region,
        instance_type=args.instance
    )


if __name__ == "__main__":
    main()
