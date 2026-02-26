"""
AWS utilities for S3 interaction.
Handles downloading trained adapters to the local cache.
"""

import logging
import os
import boto3
from pathlib import Path
from config.settings import S3_BUCKET, ADAPTERS_DIR

logger = logging.getLogger(__name__)

def download_adapters_from_s3(bucket_name: str = S3_BUCKET, local_dir: str = str(ADAPTERS_DIR)):
    """
    Syncs the 'adapters/' folder from S3 to the local filesystem.
    """
    s3 = boto3.client('s3')
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Syncing adapters from s3://{bucket_name}/adapters/ to {local_dir}...")
    
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix='adapters/'):
            if 'Contents' not in page:
                logger.warning(f"No adapters found in s3://{bucket_name}/adapters/")
                return

            for obj in page['Contents']:
                # Get the relative path after 'adapters/'
                s3_key = obj['Key']
                relative_path = s3_key.replace('adapters/', '', 1)
                
                if not relative_path: # Skip the folder itself
                    continue
                    
                target_file = local_path / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"Downloading {s3_key} -> {target_file}")
                s3.download_file(bucket_name, s3_key, str(target_file))
        
        logger.info("S3 Sync complete.")
    except Exception as exc:
        logger.error(f"S3 Download failed: {exc}")
        raise
