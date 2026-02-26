"""
Training orchestration for both CodeColosseum agents.

Production-ready changes vs prototype:
  - Accepts data_dir parameter (works locally and on SageMaker)
  - Per-agent try/except so one failure does not skip the other
  - Structured logging replacing bare print()
  - Removed unused subprocess import
  - CLI exits with code 1 if any agent fails
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("trainer")


class TrainingOrchestrator:
    """
    Manages training of both Coder and Reviewer agents sequentially.

    Accepts data_dir and adapters_dir so the same code runs locally
    and on SageMaker (where paths are /opt/ml/input/data/train/ etc.).
    """

    def __init__(
        self,
        adapters_dir: str = "data/adapters",
        data_dir: str = "data/final",
    ) -> None:
        self.adapters_dir = Path(adapters_dir)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_dir)

    def _data_file(self, agent: str) -> str:
        """Return the JSONL training data path for a given agent."""
        return str(self.data_dir / f"{agent}_train.jsonl")

    def train_coder(self) -> Optional[str]:
        """
        Train the Coder agent.

        Returns:
            Adapter path on success, None on failure.
            Never raises — logs error and returns None.
        """
        from .qlora import train_agent

        logger.info("=" * 60)
        logger.info("TRAINING CODER AGENT")
        logger.info("=" * 60)

        try:
            adapter_path = train_agent(
                agent_type="coder",
                data_file=self._data_file("coder"),
                output_dir=str(self.adapters_dir),
            )
            logger.info("Coder adapter saved: %s", adapter_path)
            return adapter_path
        except FileNotFoundError as exc:
            logger.error("Coder training skipped — data not found: %s", exc)
            return None
        except Exception as exc:
            logger.error("Coder training FAILED: %s", exc, exc_info=True)
            return None

    def train_reviewer(self) -> Optional[str]:
        """
        Train the Reviewer agent.

        Returns:
            Adapter path on success, None on failure.
            Never raises — logs error and returns None.
        """
        from .qlora import train_agent

        logger.info("=" * 60)
        logger.info("TRAINING REVIEWER AGENT")
        logger.info("=" * 60)

        try:
            adapter_path = train_agent(
                agent_type="reviewer",
                data_file=self._data_file("reviewer"),
                output_dir=str(self.adapters_dir),
            )
            logger.info("Reviewer adapter saved: %s", adapter_path)
            return adapter_path
        except FileNotFoundError as exc:
            logger.error("Reviewer training skipped — data not found: %s", exc)
            return None
        except Exception as exc:
            logger.error("Reviewer training FAILED: %s", exc, exc_info=True)
            return None

    def train_both(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Train both agents sequentially.

        Each agent is wrapped in its own try/except inside train_coder /
        train_reviewer, so a coder failure does NOT prevent reviewer training.

        Returns:
            (coder_adapter_path, reviewer_adapter_path)
            Either may be None if that agent's training failed.
        """
        logger.info("Starting full training pipeline...")

        coder_path = self.train_coder()

        # Force a full GC cycle so the Coder model's GPU memory is fully
        # released before the Reviewer tries to load the same 6.7 B model.
        import gc as _gc
        import torch as _torch
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()
        logger.info("Memory flushed between agents.")

        reviewer_path = self.train_reviewer()

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info("  Coder   : %s", coder_path    or "FAILED")
        logger.info("  Reviewer: %s", reviewer_path or "FAILED")

        failed = [a for a, p in [("coder", coder_path), ("reviewer", reviewer_path)] if not p]
        if failed:
            logger.warning("Agents that failed to train: %s", failed)

        return coder_path, reviewer_path


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train CodeColosseum agents")
    parser.add_argument(
        "--agent",
        choices=["coder", "reviewer", "both"],
        default="both",
        help="Which agent to train (default: both)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/final",
        help="Directory containing *_train.jsonl files (default: data/final). "
             "Set to /opt/ml/input/data/train on SageMaker.",
    )
    parser.add_argument(
        "--adapters-dir",
        default="data/adapters",
        help="Directory to save LoRA adapters (default: data/adapters). "
             "Set to /opt/ml/model on SageMaker.",
    )
    args = parser.parse_args()

    orchestrator = TrainingOrchestrator(
        adapters_dir=args.adapters_dir,
        data_dir=args.data_dir,
    )

    if args.agent == "coder":
        path = orchestrator.train_coder()
        sys.exit(0 if path else 1)
    elif args.agent == "reviewer":
        path = orchestrator.train_reviewer()
        sys.exit(0 if path else 1)
    else:
        coder, reviewer = orchestrator.train_both()
        sys.exit(0 if (coder and reviewer) else 1)


if __name__ == "__main__":
    main()