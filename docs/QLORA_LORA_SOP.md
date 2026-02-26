# Standard Operating Procedure (SOP)
# QLoRA-Based Code Generation & Review System

## 0. Purpose of This Document

This is the operational guide for running the project end-to-end.

It covers:

- What LoRA and QLoRA mean in this codebase
- How to train `coder_adapter` and `reviewer_adapter` on Colab T4
- How to transfer adapters to local machine
- How to run local inference reliably (including Mac CPU offload)
- How to debug known runtime and compatibility issues

If someone follows this from top to bottom, they should be able to reproduce your setup.

---

## 1. LoRA and QLoRA (Practical View)

### 1.1 What LoRA does

A full fine-tune updates all model weights. For a 6.7B model, that is expensive.

LoRA freezes base weights and trains only low-rank updates:

- Base layer weight: `W`
- Learned update: `ΔW = B @ A`
- Effective weight at runtime: `W + ΔW`

Because `A` and `B` are low-rank, trainable params are much smaller.

Result:

- Lower VRAM usage
- Smaller checkpoints
- Faster iteration

### 1.2 What QLoRA adds

QLoRA = LoRA + quantized base model.

In this project, the base model is loaded in 4-bit (`nf4`) using bitsandbytes, then LoRA adapters are trained.

Training code (`src/training/qlora.py`):

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

Base model stays frozen; only LoRA layers update.

---

## 2. What This Project Trains

Base model:

- `deepseek-ai/deepseek-coder-6.7b-instruct`

Adapters:

- `coder_adapter`: instruction -> code
- `reviewer_adapter`: code -> review

Important:

- Adapters are not full models.
- Inference always needs both base model + adapter.

---

## 3. Core Project Files

- `config/settings.py`: global config (training, LoRA, inference defaults)
- `src/training/qlora.py`: QLoRA trainer implementation
- `src/training/trainer.py`: train orchestrator
- `src/models/base.py`: base model loader (CUDA/MPS/CPU offload paths)
- `src/models/adapter.py`: adapter load/switch manager
- `src/agents/base.py`: generation pipeline and decode safeguards
- `src/api/main.py`: FastAPI startup wiring
- `src/api/routes.py`: API endpoints and error handling

---

## 4. Environment Strategy

Use separate stacks for training and inference.

### 4.1 Colab (training)

Use `requirements-colab.txt`.

Pinned core versions:

- `transformers==4.46.3`
- `trl==0.11.4`
- `peft==0.13.2`
- `accelerate==0.34.2`
- `datasets==2.21.0`

### 4.2 Local machine (inference)

Use `requirements.txt`.

Why split:

- Training compatibility on Colab is version-sensitive.
- Local inference has different hardware/runtime constraints.

---

## 5. End-to-End Flow

1. Setup local and Colab environments.
2. Prepare data (`data/final/*_train.jsonl`).
3. Train adapters on Colab.
4. Verify adapter artifacts.
5. Transfer adapters to local.
6. Start API.
7. Validate endpoints.
8. Apply troubleshooting map if needed.

---

## 6. Setup and Execution

### Step 1: Local bootstrap

```bash
cd /Users/sankalpaaryal/Documents/code-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Always run server with venv Python:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Step 2: Colab training setup

```bash
%cd /content/code-demo
!pip install -U --no-cache-dir -r requirements-colab.txt
```

Optional (if model download/auth issues):

```bash
!huggingface-cli login
```

### Step 3: Data

Required files:

- `data/final/coder_train.jsonl`
- `data/final/reviewer_train.jsonl`

If building from scratch:

```bash
python -m src.data.github_scraper
python -m src.data.formatter data/raw/scikit_learn_raw.jsonl data/processed
python -m src.data.formatter data/raw/flask_raw.jsonl data/processed
python -m src.data.formatter data/raw/pytest_raw.jsonl data/processed
```

Merge/split:

```python
from src.data.formatter import merge_and_split
merge_and_split(coder_files, reviewer_files, "data/final", train_ratio=0.9)
```

### Step 4: Train adapters

Both:

```bash
python -m src.training.trainer --agent both --data-dir data/final --adapters-dir data/adapters
```

Single:

```bash
python -m src.training.trainer --agent coder --data-dir data/final --adapters-dir data/adapters
python -m src.training.trainer --agent reviewer --data-dir data/final --adapters-dir data/adapters
```

What trainer does internally:

- Loads quantized base model
- Adds LoRA layers
- Builds SFT text format
- Runs SFTTrainer
- Saves adapter + tokenizer + metadata

### Step 5: Validate adapter output

```bash
ls -la data/adapters/coder_adapter
```

Must include:

- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer files
- `training_info.json`

### Step 6: Transfer adapter from Colab to local

In Colab:

```bash
zip -r /content/coder_adapter.zip data/adapters/coder_adapter
```

Download and unzip locally under:

- `data/adapters/coder_adapter`

---

## 7. Local Inference Modes

### 7.1 Preferred mode: CUDA GPU

Fastest and most stable for 6.7B inference.

### 7.2 Mac / low-memory mode: CPU offload

Use:

```bash
export FORCE_CPU_INFERENCE=1
export MAX_CPU_MEMORY=10GiB
export HF_OFFLOAD_DIR=/Users/sankalpaaryal/Documents/code-demo/data/offload
```

Recommended latency guardrails (important):

```bash
export CPU_OFFLOAD_MAX_NEW_TOKENS=32
export GENERATION_MAX_TIME_SEC=90
```

Why these matter:

- CPU+disk offload is much slower than GPU.
- Large `max_tokens` can look like request hangs.

---

## 8. Start API and Validate

Start server:

```bash
source .venv/bin/activate
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Expected startup lines:

- `Inference Engine ready.`
- `Application startup complete.`

Note:

- `GET /` returns 404 by design (no root route defined).

Health endpoint:

```bash
curl -s http://127.0.0.1:8000/health
```

Generate endpoint:

```bash
curl -s -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Write a Python function to check palindrome","max_tokens":32,"temperature":0.2}'
```

Review endpoint:

```bash
curl -s -X POST http://127.0.0.1:8000/review \
  -H "Content-Type: application/json" \
  -d '{"code":"def add(a,b): return a+b","max_tokens":64}'
```

---

## 9. Runtime Behavior (How Requests Flow)

For `/generate`:

1. `src/api/routes.py -> generate_code`
2. `src/agents/coder.py -> run`
3. `src/agents/base.py -> _generate`
4. `src/models/adapter.py -> set_adapter`
5. PEFT loads/activates `coder_adapter`
6. `model.generate(...)`
7. tokens decoded and returned

Key detail:

- Base model loads once at startup.
- Adapter loads lazily on first request and can be switched per agent.

---

## 10. Important Code-Level Safeguards (Current State)

These are already implemented and should be understood by anyone maintaining this system:

- Base model loader retries `dtype`/`torch_dtype` combinations for compatibility.
- Non-CUDA allocator warmup is disabled in problematic paths.
- Adapter manager uses offload-safe loading behavior for CPU/disk dispatch.
- Agent generation sanitizes decode settings.
- If `temperature <= 0`, generation auto-switches to greedy (`do_sample=False`).
- If sampling gets invalid probabilities (`nan/inf/<0`), it falls back to greedy decoding.
- Route handlers log full tracebacks (`exc_info=True`) for debugging.

---

## 11. Common Errors and Fixes

### A) `LlamaForCausalLM.__init__() got unexpected keyword argument 'dtype'`

Cause:

- transformers/model API mismatch.

Fix:

- Use compatible pinned stack (especially on Colab).
- Keep retry logic between `dtype` and `torch_dtype`.

### B) `SFTTrainer.__init__() got unexpected keyword argument ...`

Cause:

- TRL API mismatch.

Fix:

- Match TRL version to training code signature.

### C) `No package metadata was found for bitsandbytes` (local Mac)

Cause:

- bitsandbytes unavailable for your local runtime path.

Fix:

- Use CPU offload mode.

### D) `Invalid buffer size ...` / MPS OOM

Cause:

- memory allocator behavior on MPS with large models.

Fix:

- Force CPU offload mode (`FORCE_CPU_INFERENCE=1`).

### E) Adapter load errors (`KeyError ... lm_head`, `unhashable type: 'set'`)

Cause:

- PEFT + accelerate offload remap edge case.

Fix:

- Use offload-safe adapter loading in `src/models/adapter.py`.

### F) `temperature (=0.0) has to be strictly positive` with sampling

Cause:

- invalid combination: `do_sample=True` and `temperature=0`.

Fix:

- send positive temperature (e.g. `0.2`) or greedy mode (`temperature<=0` now auto-switches to greedy in current code).

### G) "It runs for many minutes and no output"

Cause:

- CPU+disk offload latency with high `max_tokens`.

Fix:

- lower token count and set offload caps:
  - `CPU_OFFLOAD_MAX_NEW_TOKENS`
  - `GENERATION_MAX_TIME_SEC`

### H) `pyenv: python: command not found`

Cause:

- shell not using virtualenv interpreter.

Fix:

```bash
source .venv/bin/activate
python -m uvicorn ...
```

or run explicit interpreter:

```bash
/Users/sankalpaaryal/Documents/code-demo/.venv/bin/python -m uvicorn ...
```

---

## 12. Validation Checklist

Treat system as production-ready only when all pass:

- API starts cleanly
- `/health` returns `200`
- `/generate` returns non-empty content
- `/review` returns non-empty content
- Adapter switching works repeatedly
- No memory crash on repeated requests

---

## 13. Quick Explanation for Others

One-minute explanation:

"We use QLoRA to fine-tune a 6.7B code model efficiently. The base model is frozen and quantized; we train small LoRA adapters instead of full weights. We train separate adapters for coding and reviewing, then load a shared base model at runtime and hot-swap adapters per request. This reduces training cost and storage while keeping practical quality."

---

## 14. Security and Ops Notes

- Keep secrets out of committed files.
- Rotate any exposed API/GitHub/HF tokens.
- Version adapters (`coder_adapter_v1`, `v2`, etc.) for rollback.
- Keep `training_info.json` for run traceability.

---

## 15. Final Reality Check

- Colab training does not remove local base model requirement.
- Local inference always needs base model download at least once.
- CPU offload is slower but workable when GPU is not available.
- Separate training and inference environments by design.

This file is the operational source of truth for this project.
