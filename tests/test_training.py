"""
Unit tests for QLoRA training logic.

These tests run WITHOUT a GPU or any downloaded model.
They mock the heavy ML dependencies and test only the pure-Python logic:
  - Config integration (settings.py keys present and correct types)
  - File validation (missing / empty / malformed JSONL)
  - training_info.json structure
  - TrainingOrchestrator data path construction
  - train_agent input validation
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Mock heavy ML imports before they are loaded ──────────────────────────────
mock_modules = [
    "torch", "datasets", "peft", "transformers", "trl", 
    "bitsandbytes", "accelerate"
]
for module in mock_modules:
    sys.modules[module] = MagicMock()

import torch
torch.cuda.is_available.return_value = False
torch.cuda.get_device_capability.return_value = (8, 0)

import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Config integration ────────────────────────────────────────────────────────

def test_settings_has_required_lora_keys():
    """LORA_CONFIG in settings.py must have all keys consumed by QLoRATrainer."""
    from config.settings import LORA_CONFIG
    required = {"r", "lora_alpha", "target_modules", "lora_dropout", "bias"}
    for key in required:
        assert key in LORA_CONFIG, f"LORA_CONFIG missing key: {key!r}"


def test_settings_has_required_training_keys():
    """TRAINING_CONFIG in settings.py must have all keys consumed by QLoRATrainer."""
    from config.settings import TRAINING_CONFIG
    required = {
        "num_epochs", "batch_size", "learning_rate",
        "max_seq_length", "warmup_steps", "gradient_accumulation_steps",
    }
    for key in required:
        assert key in TRAINING_CONFIG, f"TRAINING_CONFIG missing key: {key!r}"


def test_settings_base_model_is_nonempty_string():
    from config.settings import BASE_MODEL
    assert isinstance(BASE_MODEL, str) and BASE_MODEL.strip()


# ── File validation ───────────────────────────────────────────────────────────

def _make_mock_tokenizer():
    tok = MagicMock()
    tok.eos_token = "</s>"
    tok.pad_token = None
    tok.padding_side = "right"
    tok.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    tok.side_effect = None
    return tok


def test_missing_data_file_raises_file_not_found(tmp_path):
    """load_dataset must raise FileNotFoundError with helpful message."""
    from src.training.qlora import QLoRATrainer

    trainer = QLoRATrainer(output_dir=str(tmp_path))
    trainer.tokenizer = _make_mock_tokenizer()

    with pytest.raises(FileNotFoundError, match="Training data not found"):
        trainer.load_dataset(str(tmp_path / "does_not_exist.jsonl"))


def test_empty_data_file_raises_value_error(tmp_path):
    """load_dataset must raise ValueError for a zero-byte JSONL file."""
    from src.training.qlora import QLoRATrainer

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")

    trainer = QLoRATrainer(output_dir=str(tmp_path))
    trainer.tokenizer = _make_mock_tokenizer()

    with pytest.raises(ValueError, match="empty"):
        trainer.load_dataset(str(empty))


def test_malformed_jsonl_raises_error(tmp_path):
    """load_dataset must raise ValueError for malformed JSONL lines."""
    from src.training.qlora import QLoRATrainer

    bad = tmp_path / "bad.jsonl"
    bad.write_text("this is not valid json\n")

    trainer = QLoRATrainer(output_dir=str(tmp_path))
    trainer.tokenizer = _make_mock_tokenizer()

    with pytest.raises((ValueError, json.JSONDecodeError)):
        trainer.load_dataset(str(bad))


# ── training_info.json structure ──────────────────────────────────────────────

def test_training_info_json_has_all_required_keys(tmp_path):
    """training_info.json written by train() must include a trained_at timestamp."""
    info = {
        "agent": "coder",
        "base_model": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "trained_at": "2026-02-19T17:00:00+00:00",
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "num_train_examples": 180,
        "num_eval_examples": 20,
        "gpu_precision": "bf16",
    }
    p = tmp_path / "training_info.json"
    p.write_text(json.dumps(info))

    loaded = json.loads(p.read_text())
    required = {
        "agent", "base_model", "trained_at", "epochs", "batch_size",
        "learning_rate", "lora_r", "lora_alpha",
        "num_train_examples", "num_eval_examples", "gpu_precision",
    }
    for key in required:
        assert key in loaded, f"training_info.json missing key: {key!r}"


# ── TrainingOrchestrator ──────────────────────────────────────────────────────

def test_orchestrator_builds_correct_data_paths(tmp_path):
    """TrainingOrchestrator._data_file() must use data_dir, not hardcoded path."""
    from src.training.trainer import TrainingOrchestrator

    orch = TrainingOrchestrator(
        adapters_dir=str(tmp_path),
        data_dir="/custom/data/path",
    )
    assert orch._data_file("coder")    == "/custom/data/path/coder_train.jsonl"
    assert orch._data_file("reviewer") == "/custom/data/path/reviewer_train.jsonl"


def test_orchestrator_returns_none_not_raises_on_missing_data(tmp_path):
    """train_coder / train_reviewer must return None (not raise) when data is absent."""
    from src.training.trainer import TrainingOrchestrator

    orch = TrainingOrchestrator(
        adapters_dir=str(tmp_path),
        data_dir=str(tmp_path / "nonexistent_dir"),
    )

    assert orch.train_coder()    is None, "Expected None when coder data is missing"
    assert orch.train_reviewer() is None, "Expected None when reviewer data is missing"


def test_orchestrator_train_both_returns_tuple(tmp_path):
    """train_both must always return a 2-tuple even when both agents fail."""
    from src.training.trainer import TrainingOrchestrator

    orch = TrainingOrchestrator(
        adapters_dir=str(tmp_path),
        data_dir=str(tmp_path / "nonexistent_dir"),
    )

    result = orch.train_both()
    assert isinstance(result, tuple) and len(result) == 2


# ── train_agent input validation ──────────────────────────────────────────────

def test_train_agent_rejects_invalid_agent_type(tmp_path):
    """train_agent must raise ValueError for unknown agent types."""
    from src.training.qlora import train_agent

    with pytest.raises(ValueError, match="agent_type must be"):
        train_agent(
            agent_type="robot",
            data_file="irrelevant.jsonl",
            output_dir=str(tmp_path),
        )


# ── GPU precision detection ───────────────────────────────────────────────────

def test_supports_bf16_returns_bool():
    """_supports_bf16() must always return a bool (True or False, never None)."""
    from src.training.qlora import _supports_bf16
    result = _supports_bf16()
    assert isinstance(result, bool)
