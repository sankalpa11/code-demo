"""
Abstract base class for CodeColosseum agents.
Provides a common interface for generation and prompt handling.
"""

import abc
import logging
import os
from typing import Dict, Any
import torch
from config.settings import INFERENCE_CONFIG

logger = logging.getLogger(__name__)

class BaseAgent(abc.ABC):
    """
    Abstract base class for all AI agents.
    """
    def __init__(self, model, tokenizer, adapter_manager, agent_config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_manager = adapter_manager
        self.config = agent_config
        self.name = agent_config.get("name", "Unknown Agent")
        self.adapter_name = agent_config.get("adapter_name")
        self.prompt_template = agent_config.get("prompt_template", "{instruction}")

    @abc.abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """
        Execute the agent's logic.
        """
        pass

    def _generate(self, prompt: str, **kwargs) -> str:
        """
        Low-level generation call.
        """
        # Ensure correct adapter is active
        if self.adapter_name:
            self.adapter_manager.set_adapter(self.adapter_name)
            # AdapterManager may wrap/replace the model object on first load.
            # Keep the agent model reference in sync so generation uses LoRA.
            self.model = self.adapter_manager.model

        # For accelerate-dispatched models, model.device may be meta/ambiguous.
        # Pick a concrete execution device for the input tensors.
        input_device = None
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for mapped in hf_device_map.values():
                mapped_str = str(mapped)
                if mapped_str == "cpu":
                    input_device = torch.device("cpu")
                    break
                if mapped_str == "mps":
                    input_device = torch.device("mps")
                    break
                if isinstance(mapped, int):
                    input_device = torch.device(f"cuda:{mapped}")
                    break
                if isinstance(mapped, torch.device) and mapped.type not in {"meta"}:
                    input_device = mapped
                    break

        if input_device is None:
            model_device = getattr(self.model, "device", None)
            if isinstance(model_device, torch.device) and model_device.type != "meta":
                input_device = model_device
            elif isinstance(model_device, str) and model_device not in {"meta", "disk"}:
                input_device = torch.device(model_device)
            else:
                input_device = torch.device("cpu")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(input_device)
        
        # Merge global inference config with per-call overrides
        gen_kwargs = {**INFERENCE_CONFIG, **kwargs}
        # Keep defaults when callers pass None.
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        # On CPU/disk-offloaded models, generation speed is extremely low.
        # Apply guardrails so API calls don't appear to hang indefinitely.
        offload_mode = False
        if isinstance(hf_device_map, dict):
            offload_mode = any(str(v) in {"cpu", "disk", "meta"} for v in hf_device_map.values())
        if offload_mode:
            try:
                token_cap = int(os.getenv("CPU_OFFLOAD_MAX_NEW_TOKENS", "48"))
            except ValueError:
                token_cap = 48
            requested_tokens = gen_kwargs.get("max_new_tokens")
            if isinstance(requested_tokens, int) and requested_tokens > token_cap:
                logger.warning(
                    "CPU offload mode: capping max_new_tokens from %s to %s for latency control.",
                    requested_tokens,
                    token_cap,
                )
                gen_kwargs["max_new_tokens"] = token_cap
            # Stop long-running requests and return partial decode if needed.
            try:
                max_time_sec = float(os.getenv("GENERATION_MAX_TIME_SEC", "120"))
            except ValueError:
                max_time_sec = 120.0
            gen_kwargs.setdefault("max_time", max_time_sec)

        # Defensive guards for numerically fragile decode paths.
        gen_kwargs.setdefault("remove_invalid_values", True)
        gen_kwargs.setdefault("renormalize_logits", True)
        # Interpret non-positive temperature as a greedy decode request.
        # transformers requires temperature > 0 when do_sample=True.
        temp = gen_kwargs.get("temperature")
        if temp is not None:
            try:
                temp_value = float(temp)
            except (TypeError, ValueError):
                temp_value = None
            if temp_value is not None and temp_value <= 0:
                gen_kwargs["do_sample"] = False
                gen_kwargs.pop("temperature", None)
                logger.info(
                    "Non-positive temperature requested (%s). Switching to greedy decode.",
                    temp,
                )
            elif temp_value is not None:
                gen_kwargs["temperature"] = temp_value

        if gen_kwargs.get("do_sample") is False:
            for key in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
                gen_kwargs.pop(key, None)
        
        logger.info(
            "Agent %s generating | adapter=%s | input_device=%s | config=%s",
            self.name,
            self.adapter_name,
            input_device,
            gen_kwargs,
        )
        
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            except (RuntimeError, ValueError) as exc:
                msg = str(exc)
                unstable_probs = "probability tensor contains either `inf`, `nan` or element < 0" in msg
                bad_temp = "has to be a strictly positive float" in msg
                if not (unstable_probs or bad_temp):
                    raise
                # Fallback: disable sampling and decode greedily.
                fallback_kwargs = dict(gen_kwargs)
                fallback_kwargs["do_sample"] = False
                fallback_kwargs.pop("temperature", None)
                fallback_kwargs.pop("top_p", None)
                fallback_kwargs.pop("top_k", None)
                fallback_kwargs.pop("typical_p", None)
                fallback_kwargs["remove_invalid_values"] = True
                fallback_kwargs["renormalize_logits"] = True
                logger.warning(
                    "Sampling became numerically unstable (%s). Retrying with greedy decode.",
                    exc,
                )
                outputs = self.model.generate(
                    **inputs,
                    **fallback_kwargs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode only the generated part (excluding the prompt)
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
