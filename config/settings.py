"""
Central configuration for CodeColosseum Demo.
All settings loaded from environment or defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ADAPTERS_DIR = DATA_DIR / "adapters"

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "codecolosseum-demo")

# Model Configuration
BASE_MODEL = os.getenv("BASE_MODEL", "deepseek-ai/deepseek-coder-6.7b-instruct")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")

# Training Configuration
TRAINING_CONFIG = {
    "learning_rate": float(os.getenv("LEARNING_RATE", "2e-4")),
    "batch_size": int(os.getenv("BATCH_SIZE", "2")),       # 2 is safe on T4 16GB
    "num_epochs": int(os.getenv("NUM_EPOCHS", "3")),
    "gradient_accumulation_steps": 4,                        # effective batch = 8
    "max_seq_length": 1024,                                  # 1024 is safe on T4
    "warmup_steps": 10,
}

# LoRA Configuration
LORA_CONFIG = {
    "r": int(os.getenv("LORA_R", "16")),
    "lora_alpha": int(os.getenv("LORA_ALPHA", "32")),
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
}

# Inference Configuration
INFERENCE_CONFIG = {
    "max_new_tokens": int(os.getenv("MAX_TOKENS", "512")),
    "temperature": float(os.getenv("TEMPERATURE", "0.2")),
    "top_p": float(os.getenv("TOP_P", "0.95")),
    "do_sample": True,
}

# Agent Definitions
AGENTS = {
    "coder": {
        "name": "Coder",
        "description": "Generates code from natural language instructions",
        "adapter_name": "coder_adapter",
        "prompt_template": """### Instruction:
{instruction}

### Response:
""",
    },
    "reviewer": {
        "name": "Reviewer", 
        "description": "Evaluates code quality and suggests improvements",
        "adapter_name": "reviewer_adapter",
        "prompt_template": """### Code to Review:
{code}

### Review:
""",
    },
}