"""
QLoRA training for CodeColosseum agents.
Trains lightweight adapters on frozen 4-bit base model.

Production-ready changes vs prototype:
  - Structured logging replacing all bare print() calls
  - All hyperparameters driven from config/settings.py (no hardcoding)
  - Error handling with descriptive messages on model load and training
  - bf16 vs fp16 auto-detected from GPU compute capability
  - 90/10 train/eval split to monitor and catch overfitting
  - Checkpoint resumption support (saves GPU cost on crashes)
  - tokenizer.padding_side = "right" for correct causal LM training
  - Timestamp + GPU precision written to training_info.json
  - Removed unused DataCollatorForLanguageModeling import
"""

import gc
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple  # noqa: F401 (Tuple kept for public API)

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ── Centralized config ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import ADAPTERS_DIR, BASE_MODEL, LORA_CONFIG, TRAINING_CONFIG  # noqa: E402

# ── Structured logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("qlora")


def _supports_bf16() -> bool:
    """Return True only on Ampere+ GPUs (compute capability >= 8.0).

    V100 = cc 7.0, T4 = cc 7.5  -> must use fp16
    A10G = cc 8.6, A100 = cc 8.0 -> native bf16
    H100 = cc 9.0               -> native bf16
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


class QLoRATrainer:
    """
    Production QLoRA trainer for code-generation agents.

    Q    = 4-bit quantization  (~75 % memory reduction)
    LoRA = Low-rank adaptation  (~1 % of parameters trained)
    """

    def __init__(
        self,
        base_model_name: str = BASE_MODEL,
        output_dir: str = str(ADAPTERS_DIR),
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
    ) -> None:
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Allow per-run overrides while defaulting to settings.py
        r = lora_r or LORA_CONFIG["r"]
        alpha = lora_alpha or LORA_CONFIG["lora_alpha"]

        # Quantization config (the "Q" in QLoRA)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # LoRA config — all values from settings.py, nothing hardcoded
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=LORA_CONFIG["target_modules"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            bias=LORA_CONFIG["bias"],
            task_type="CAUSAL_LM",
        )

        self.tokenizer = None
        self.base_model = None

        logger.info(
            "QLoRATrainer ready | model=%s | lora_r=%d | lora_alpha=%d",
            base_model_name, r, alpha,
        )

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_base_model(self) -> None:
        """Load 4-bit quantized base model with error handling."""
        logger.info("Loading tokenizer: %s", self.base_model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.error("Tokenizer load failed: %s", exc)
            raise RuntimeError(
                f"Could not load tokenizer for '{self.base_model_name}'. "
                "Check internet access and the Hugging Face model name."
            ) from exc

        # Right-padding is required for causal LM to avoid silent label
        # misalignment when batches contain sequences of different lengths.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        logger.info("Loading base model in 4-bit (~3.5 GB download on first run)...")
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=self.bnb_config,
                device_map="auto",
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )
        except Exception as exc:
            logger.error("Model load failed: %s", exc)
            raise RuntimeError(
                f"Could not load model '{self.base_model_name}'. "
                "Ensure bitsandbytes>=0.42.0 is installed and a CUDA GPU is present."
            ) from exc

        self.base_model = prepare_model_for_kbit_training(
            self.base_model,
            use_gradient_checkpointing=True,
        )
        mem_gb = self.base_model.get_memory_footprint() / 1e9
        logger.info("Base model ready | memory=%.2f GB", mem_gb)

    # ── PEFT ─────────────────────────────────────────────────────────────────

    def create_peft_model(self):
        """Attach LoRA adapters to the frozen base model."""
        logger.info("Attaching LoRA adapters...")
        peft_model = get_peft_model(self.base_model, self.lora_config)
        peft_model.print_trainable_parameters()
        return peft_model

    # ── Dataset ───────────────────────────────────────────────────────────────

    def load_dataset(
        self,
        data_file: str,
        eval_split: float = 0.1,
    ) -> Tuple[Dataset, Dataset]:
        """
        Load, format, and split training data as raw text.

        NOTE: Tokenization is intentionally deferred to SFTTrainer, which
        handles per-batch dynamic padding. Pre-tokenising everything upfront
        with padding="max_length" wastes system RAM and crashes Colab.

        A 90/10 train/eval split allows validation loss to be monitored
        during training so overfitting can be caught and stopped early.

        Args:
            data_file:  Path to JSONL training file.
            eval_split: Fraction held out for evaluation (default 10 %).

        Returns:
            (train_dataset, eval_dataset) — each with a single "text" column.
        """
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Training data not found: {data_file}\n"
                "Run: python -m src.data.selector --fast"
            )

        logger.info("Reading dataset: %s", data_file)
        try:
            with open(data_path) as fh:
                examples = [json.loads(line) for line in fh if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL in {data_file}: {exc}") from exc

        if not examples:
            raise ValueError(f"Dataset is empty: {data_file}")

        logger.info("Loaded %d examples", len(examples))

        def format_example(ex: dict) -> dict:
            if "instruction" in ex:
                text = (
                    "### Instruction:\n"
                    + ex["instruction"]
                    + "\n\n### Response:\n"
                    + ex["code"]
                    + "</s>"
                )
            else:
                text = (
                    "### Code:\n"
                    + ex["code"]
                    + "\n\n### Review:\n"
                    + ex["review"]
                    + "</s>"
                )
            return {"text": text}

        # Raw text-only dataset — SFTTrainer tokenises on the fly
        raw_ds = Dataset.from_list([format_example(ex) for ex in examples])
        splits = raw_ds.train_test_split(test_size=eval_split, seed=42)
        train_ds, eval_ds = splits["train"], splits["test"]
        logger.info("Split: train=%d  eval=%d", len(train_ds), len(eval_ds))
        return train_ds, eval_ds

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        data_file: str,
        agent_name: str,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        resume_from_checkpoint: bool = True,
    ) -> str:
        """
        Train a LoRA adapter for a specific agent.

        All hyperparameter defaults come from config/settings.py.
        Pass explicit overrides for sweeps or SageMaker env vars.

        Args:
            data_file:               Path to JSONL training data.
            agent_name:              "coder" or "reviewer".
            num_epochs:              Training epochs (default: settings.py).
            batch_size:              Per-device batch size (default: settings.py).
            learning_rate:           LR (default: settings.py).
            resume_from_checkpoint:  Resume from latest checkpoint if present.

        Returns:
            Absolute path to the saved adapter directory.
        """
        epochs = num_epochs or TRAINING_CONFIG["num_epochs"]
        bs     = batch_size or TRAINING_CONFIG["batch_size"]
        lr     = learning_rate or TRAINING_CONFIG["learning_rate"]

        if self.base_model is None:
            self.load_base_model()

        model    = self.create_peft_model()
        train_ds, eval_ds = self.load_dataset(data_file)

        output_path = self.output_dir / f"{agent_name}_adapter"
        output_path.mkdir(parents=True, exist_ok=True)

        # Auto-detect bf16 vs fp16 based on GPU generation
        use_bf16 = _supports_bf16()
        use_fp16 = (not use_bf16) and torch.cuda.is_available()
        precision = "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32-cpu")
        logger.info("GPU precision: %s", precision)

        # Resume from latest checkpoint if --resume requested
        existing = sorted(output_path.glob("checkpoint-*"))
        checkpoint = str(existing[-1]) if (resume_from_checkpoint and existing) else None
        if checkpoint:
            logger.info("Resuming from checkpoint: %s", checkpoint)

        # SFTConfig = TrainingArguments + SFT-specific fields (TRL >= 0.10)
        training_args = SFTConfig(
            output_dir=str(output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            optim="paged_adamw_8bit",
            learning_rate=lr,
            warmup_steps=TRAINING_CONFIG["warmup_steps"],
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=use_fp16,
            bf16=use_bf16,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            # SFT-specific (moved from SFTTrainer.__init__ in TRL >= 0.10)
            max_seq_length=TRAINING_CONFIG["max_seq_length"],
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=self.tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
        )

        logger.info("=" * 60)
        logger.info("TRAINING %s AGENT", agent_name.upper())
        logger.info("=" * 60)
        logger.info("epochs=%d  batch=%d  lr=%s  output=%s", epochs, bs, lr, output_path)

        try:
            trainer.train(resume_from_checkpoint=checkpoint)
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            raise

        # Save adapter weights + tokenizer
        logger.info("Saving adapter to: %s", output_path)
        trainer.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save metadata (includes timestamp so adapter versions are traceable)
        info = {
            "agent": agent_name,
            "base_model": self.base_model_name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "epochs": epochs,
            "batch_size": bs,
            "learning_rate": lr,
            "lora_r": self.lora_config.r,
            "lora_alpha": self.lora_config.lora_alpha,
            "num_train_examples": len(train_ds),
            "num_eval_examples": len(eval_ds),
            "gpu_precision": precision,
        }
        with open(output_path / "training_info.json", "w") as fh:
            json.dump(info, fh, indent=2)

        logger.info("Training complete | adapter=%s", output_path)
        return str(output_path)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Free GPU memory after training.

        Uses synchronize() + reset_peak_memory_stats() so that PyTorch
        fully returns CUDA memory to the driver before the next model load.
        This is critical when training two agents back-to-back on 16 GB VRAM.
        """
        if self.base_model:
            del self.base_model
            self.base_model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()          # wait for all CUDA ops to finish
            torch.cuda.empty_cache()          # return cached blocks to the driver
            torch.cuda.reset_peak_memory_stats()  # reset peak tracker
        logger.info("GPU memory freed")


# ── Convenience wrapper ───────────────────────────────────────────────────────

def train_agent(
    agent_type: str,
    data_file: str,
    output_dir: str = str(ADAPTERS_DIR),
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> str:
    """
    Convenience function to train a single agent.

    Args:
        agent_type:    "coder" or "reviewer"
        data_file:     Path to JSONL training data
        output_dir:    Where to save the adapter (default: settings.ADAPTERS_DIR)
        num_epochs:    Override epoch count
        batch_size:    Override batch size
        learning_rate: Override learning rate

    Returns:
        Path to the saved adapter directory.

    Raises:
        ValueError: If agent_type is not "coder" or "reviewer".
    """
    if agent_type not in ("coder", "reviewer"):
        raise ValueError(
            f"agent_type must be 'coder' or 'reviewer', got: {agent_type!r}"
        )

    trainer = QLoRATrainer(output_dir=output_dir)
    try:
        return trainer.train(
            data_file=data_file,
            agent_name=agent_type,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    finally:
        trainer.cleanup()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for CodeColosseum agents"
    )
    parser.add_argument("agent", choices=["coder", "reviewer"], help="Agent to train")
    parser.add_argument("--data-file", default=None, help="Path to JSONL training file")
    parser.add_argument("--output-dir", default=str(ADAPTERS_DIR))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    data_file = args.data_file or f"data/final/{args.agent}_train.jsonl"

    if not Path(data_file).exists():
        logger.error("Data file not found: %s", data_file)
        sys.exit(1)

    train_agent(
        agent_type=args.agent,
        data_file=data_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
