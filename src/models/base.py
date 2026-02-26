"""
Base model loading for inference.
Loads the DeepSeek-Coder model in 4-bit quantization using bitsandbytes.
"""

import logging
import os
from importlib import metadata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config.settings import BASE_MODEL, MODEL_REVISION

logger = logging.getLogger(__name__)


def _supports_bf16() -> bool:
    """Return True on Ampere+ CUDA GPUs."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def _has_bitsandbytes() -> bool:
    """Return True if bitsandbytes package metadata is available."""
    try:
        metadata.version("bitsandbytes")
        return True
    except metadata.PackageNotFoundError:
        return False


def _supports_mps() -> bool:
    """Return True when Apple Metal backend is available."""
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _cpu_offload_dir() -> Path:
    """Directory for accelerate disk-offload files."""
    offload_dir = Path(os.getenv("HF_OFFLOAD_DIR", "data/offload"))
    offload_dir.mkdir(parents=True, exist_ok=True)
    return offload_dir


def load_base_model_and_tokenizer(model_name: str = BASE_MODEL, revision: str = MODEL_REVISION):
    """
    Loads the base model in 4-bit and its tokenizer.
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = None
    for trust_remote_code in (False, True):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                padding_side="right",  # Crucial for batch inference
            )
            if trust_remote_code:
                logger.warning("Tokenizer required trust_remote_code=True.")
            break
        except Exception as exc:
            if not trust_remote_code:
                logger.warning(
                    "Tokenizer load without remote code failed (%s). Retrying with trust_remote_code=True.",
                    exc,
                )
                continue
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    force_cpu = os.getenv("FORCE_CPU_INFERENCE", "0") == "1"
    use_mps = (not force_cpu) and _supports_mps()
    # Keep non-CUDA paths in half precision by default to avoid CPU RAM spikes.
    compute_dtype = torch.bfloat16 if _supports_bf16() else torch.float16

    quantization_config = None
    if use_cuda:
        if _has_bitsandbytes():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            logger.info("Using 4-bit bitsandbytes quantization for inference.")
        else:
            logger.warning(
                "bitsandbytes not installed; loading non-quantized model on GPU."
            )
    elif use_mps:
        logger.info("Using Apple Metal (MPS) backend for inference.")
    else:
        if force_cpu:
            logger.warning("FORCE_CPU_INFERENCE=1 detected. Using CPU offload mode.")
        else:
            logger.warning("No CUDA/MPS GPU detected. Loading model on CPU offload mode (very slow).")

    logger.info(f"Loading base model: {model_name}...")
    base_kwargs = {
        "revision": revision,
    }
    if use_cuda:
        base_kwargs["device_map"] = "auto"
        base_kwargs["low_cpu_mem_usage"] = True
    elif use_mps:
        # MPS is a single-device backend; load then move modules there.
        base_kwargs["device_map"] = {"": "mps"}
        base_kwargs["low_cpu_mem_usage"] = True
    else:
        # CPU offload mode: keep most tensors on CPU with disk spill support.
        # This is the safest path on machines where MPS/CUDA memory is too low.
        base_kwargs["device_map"] = "auto"
        base_kwargs["max_memory"] = {"cpu": os.getenv("MAX_CPU_MEMORY", "12GiB")}
        base_kwargs["offload_folder"] = str(_cpu_offload_dir())
        base_kwargs["offload_state_dict"] = True
        base_kwargs["low_cpu_mem_usage"] = True
        base_kwargs["use_safetensors"] = True
    if quantization_config is not None:
        base_kwargs["quantization_config"] = quantization_config

    # transformers/model-class combinations differ across versions on
    # dtype keyword handling and remote-code behavior.
    if use_cuda:
        load_attempts = (
            {"trust_remote_code": False, "dtype": compute_dtype},
            {"trust_remote_code": False, "torch_dtype": compute_dtype},
            {"trust_remote_code": False},
            {"trust_remote_code": True, "torch_dtype": compute_dtype},
            {"trust_remote_code": True},
        )
    else:
        # Avoid implicit fp32 loads on CPU, which can trigger immediate OOM kills.
        cpu_dtypes = [torch.float16, torch.bfloat16]
        if os.getenv("ALLOW_CPU_FP32", "0") == "1":
            cpu_dtypes.append(torch.float32)

        attempts = []
        for dt in cpu_dtypes:
            attempts.append({"trust_remote_code": False, "torch_dtype": dt})
            attempts.append({"trust_remote_code": False, "dtype": dt})
            attempts.append({"trust_remote_code": True, "torch_dtype": dt})
        load_attempts = tuple(attempts)

    model = None
    last_exc = None

    # transformers allocator warmup can fail on non-CUDA devices with
    # huge single allocations; disable it for this load path.
    warmup_patched = False
    original_warmup = None
    _modeling_utils = None
    if not use_cuda:
        try:
            from transformers import modeling_utils as _modeling_utils  # type: ignore

            original_warmup = _modeling_utils.caching_allocator_warmup
            _modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
            warmup_patched = True
            logger.info("Disabled transformers allocator warmup for non-CUDA model load.")
        except Exception as exc:
            logger.warning("Could not patch allocator warmup: %s", exc)

    for idx, extra_kwargs in enumerate(load_attempts, start=1):
        dtype_arg = "none"
        if "dtype" in extra_kwargs:
            dtype_arg = "dtype"
        elif "torch_dtype" in extra_kwargs:
            dtype_arg = "torch_dtype"
        logger.info(
            "Model load attempt %d/%d | trust_remote_code=%s | dtype_arg=%s",
            idx,
            len(load_attempts),
            extra_kwargs["trust_remote_code"],
            dtype_arg,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **base_kwargs,
                **extra_kwargs,
            )
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("Model load attempt %d failed: %s", idx, exc)

    if warmup_patched and _modeling_utils is not None and original_warmup is not None:
        try:
            _modeling_utils.caching_allocator_warmup = original_warmup
        except Exception:
            pass

    if model is None:
        raise RuntimeError(
            f"Failed to load model '{model_name}' after {len(load_attempts)} attempts."
        ) from last_exc

    return model, tokenizer
