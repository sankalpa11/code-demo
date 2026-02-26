"""
Dynamic adapter management for Peft models.
Handles loading, hotswapping, and unloading LoRA adapters.
"""

import logging
import os
from pathlib import Path
from typing import Any
from peft import PeftModel
from config.settings import ADAPTERS_DIR

logger = logging.getLogger(__name__)


def _uses_cpu_or_disk_offload(model) -> bool:
    """
    Return True when the base model is dispatched across cpu/disk devices.
    In this case PEFT should load adapter weights with low_cpu_mem_usage=True
    so meta tensors are assigned correctly.
    """
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return False
    return any(str(device) in {"cpu", "disk", "meta"} for device in device_map.values())

class AdapterManager:
    """
    Manages multiple LoRA adapters on a single frozen base model.
    """
    def __init__(self, model, adapters_dir: str = str(ADAPTERS_DIR)):
        self.model = model
        self.adapters_dir = Path(adapters_dir)
        self.loaded_adapters = {} # name -> path
        self.current_adapter = None

    def load_adapter(self, adapter_name: str, adapter_path: str = None):
        """
        Loads an adapter into the model without activating it.
        """
        if adapter_name in self.loaded_adapters:
            logger.info(f"Adapter {adapter_name} already loaded.")
            return

        path = Path(adapter_path) if adapter_path else self.adapters_dir / adapter_name
        if not path.exists():
            raise FileNotFoundError(f"Adapter path not found: {path}")

        logger.info(f"Loading adapter {adapter_name} from {path}...")

        load_kwargs: dict[str, Any] = {}
        offload_mode = _uses_cpu_or_disk_offload(self.model)
        if offload_mode:
            offload_dir = Path(os.getenv("HF_OFFLOAD_DIR", "data/offload")).resolve()
            offload_dir.mkdir(parents=True, exist_ok=True)
            load_kwargs["low_cpu_mem_usage"] = True
            logger.info(
                "Adapter load in offload mode | low_cpu_mem_usage=True | offload_dir=%s",
                offload_dir,
            )

        if offload_mode:
            # PEFT can fail on CPU/disk-offloaded models when it tries to remap
            # offload indices / balanced memory for some model layouts.
            # Temporarily masking hf_device_map avoids that path.
            self._load_adapter_with_masked_device_map(adapter_name, path, load_kwargs)
        else:
            self._load_adapter_impl(adapter_name, path, load_kwargs)
            
        self.loaded_adapters[adapter_name] = str(path)
        logger.info(f"Adapter {adapter_name} loaded successfully.")

    def _load_adapter_impl(
        self,
        adapter_name: str,
        path: Path,
        load_kwargs: dict[str, Any],
    ) -> None:
        """Load adapter either as first PEFT wrapper or additional adapter."""
        if not isinstance(self.model, PeftModel):
            self.model = PeftModel.from_pretrained(
                self.model,
                str(path),
                adapter_name=adapter_name,
                **load_kwargs,
            )
        else:
            self.model.load_adapter(
                str(path),
                adapter_name=adapter_name,
                **load_kwargs,
            )

    def _load_adapter_with_masked_device_map(
        self,
        adapter_name: str,
        path: Path,
        load_kwargs: dict[str, Any],
    ) -> None:
        """
        Work around PEFT+offload key-remap issues by temporarily disabling
        hf_device_map visibility during adapter injection.
        """
        model_obj = self.model
        had_map = hasattr(model_obj, "hf_device_map")
        original_map = getattr(model_obj, "hf_device_map", None) if had_map else None
        if had_map:
            setattr(model_obj, "hf_device_map", None)

        try:
            self._load_adapter_impl(adapter_name, path, load_kwargs)
        finally:
            if had_map:
                try:
                    setattr(model_obj, "hf_device_map", original_map)
                except Exception:
                    pass
                try:
                    if getattr(self.model, "hf_device_map", None) is None:
                        setattr(self.model, "hf_device_map", original_map)
                except Exception:
                    pass

    def set_adapter(self, adapter_name: str):
        """
        Switches the active adapter.
        """
        if adapter_name not in self.loaded_adapters:
            # Try to load it if it's in the default dir
            self.load_adapter(adapter_name)

        if not isinstance(self.model, PeftModel):
            raise RuntimeError("No adapters loaded in model.")

        logger.info(f"Switching to adapter: {adapter_name}")
        self.model.set_adapter(adapter_name)
        self.current_adapter = adapter_name

    def unload_adapter(self, adapter_name: str):
        """
        Unloads an adapter from the model to free memory (if possible).
        Note: Peft doesn't fully 'delete' from memory easily, but we can stop using it.
        """
        if adapter_name in self.loaded_adapters:
            logger.info(f"Deactivating adapter: {adapter_name}")
            # In current PEFT, we usually just switch away
            del self.loaded_adapters[adapter_name]
            if self.current_adapter == adapter_name:
                self.current_adapter = None
