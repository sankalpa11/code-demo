"""
AWS SageMaker entry point for QLoRA training.

SageMaker automatically mounts:
  Training data : /opt/ml/input/data/train/   (SM_CHANNEL_TRAIN)
  Model output  : /opt/ml/model/              (SM_MODEL_DIR)
  Checkpoints   : /opt/ml/checkpoints/        (SM_CHECKPOINT_DIR)

Set hyperparameters via SageMaker Estimator or environment variables:
  AGENT_TYPE    : "coder" (default) or "reviewer"
  NUM_EPOCHS    : override epoch count (optional)
  BATCH_SIZE    : override batch size  (optional)
  LEARNING_RATE : override LR          (optional)
  LORA_R        : override LoRA rank   (optional)

Example SageMaker launch (from scripts/launch_sagemaker.py):
  estimator.fit({"train": "s3://my-bucket/data/"})
"""

import logging
import os
import sys
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sagemaker_entry")

# ── SageMaker standard paths (fall back to local paths for testing) ───────────
SM_DATA_DIR  = Path(os.environ.get("SM_CHANNEL_TRAIN",   "/opt/ml/input/data/train"))
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR",       "/opt/ml/model"))
SM_CKPT_DIR  = Path(os.environ.get("SM_CHECKPOINT_DIR",  "/opt/ml/checkpoints"))

# ── Hyperparameters from environment ─────────────────────────────────────────
AGENT_TYPE    = os.environ.get("AGENT_TYPE", "coder")
NUM_EPOCHS    = int(os.environ["NUM_EPOCHS"])      if "NUM_EPOCHS"    in os.environ else None
BATCH_SIZE    = int(os.environ["BATCH_SIZE"])      if "BATCH_SIZE"    in os.environ else None
LEARNING_RATE = float(os.environ["LEARNING_RATE"]) if "LEARNING_RATE" in os.environ else None
LORA_R        = int(os.environ["LORA_R"])          if "LORA_R"        in os.environ else None


def main() -> None:
    logger.info("=" * 60)
    logger.info("SageMaker QLoRA Training Entry Point")
    logger.info("=" * 60)
    logger.info(
        "agent=%s | data=%s | output=%s | epochs=%s | batch=%s | lr=%s | lora_r=%s",
        AGENT_TYPE, SM_DATA_DIR, SM_MODEL_DIR,
        NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, LORA_R,
    )

    if AGENT_TYPE not in ("coder", "reviewer"):
        logger.error("AGENT_TYPE must be 'coder' or 'reviewer', got: %s", AGENT_TYPE)
        sys.exit(1)

    data_file = SM_DATA_DIR / f"{AGENT_TYPE}_train.jsonl"
    if not data_file.exists():
        logger.error("Training data not found: %s", data_file)
        logger.error("Files present in data dir: %s", list(SM_DATA_DIR.iterdir()))
        sys.exit(1)

    # Ensure project root is on path so config.settings imports work
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from src.training.qlora import QLoRATrainer  # noqa: E402

    trainer = QLoRATrainer(
        output_dir=str(SM_MODEL_DIR),
        lora_r=LORA_R,
    )

    try:
        adapter_path = trainer.train(
            data_file=str(data_file),
            agent_name=AGENT_TYPE,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            resume_from_checkpoint=SM_CKPT_DIR.is_dir() and any(SM_CKPT_DIR.iterdir()),
        )
        logger.info("Adapter saved to: %s", adapter_path)
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
