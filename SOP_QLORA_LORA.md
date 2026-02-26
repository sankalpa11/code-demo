

## 0) Purpose of This SOP

This SOP is a complete runbook for:

1. Understanding **LoRA** and **QLoRA** clearly.
2. Training adapters (`coder_adapter`, `reviewer_adapter`) on Colab T4.
3. Moving trained adapters to local machine.
4. Running inference API reliably on local hardware (including Mac CPU/offload path).
5. Debugging common failures with clear root-cause -> fix mapping.

Use this when teaching others or reproducing your setup from scratch.

---

## 1) LoRA and QLoRA in Simple, Correct Terms

### 1.1 LoRA (Low-Rank Adaptation)

A full fine-tune updates *all* model parameters. For a 6.7B model, that is expensive.

LoRA freezes base weights and trains small low-rank matrices in selected modules.

Core idea:

- Original linear layer: `W`
- LoRA replacement during training: `W + ΔW`
- `ΔW = B @ A`, where `A` and `B` are low-rank (rank `r << hidden_size`)

Result:

- Much lower trainable parameter count
- Lower GPU memory usage
- Small adapter checkpoints (not full model)

### 1.2 QLoRA

QLoRA = LoRA + quantized frozen base model.

In this project, base model is loaded in **4-bit** (NF4) for training with bitsandbytes, while LoRA weights are trained on top.

Representative code (`src/training/qlora.py`):

```python
self.bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

And LoRA config:

```python
self.lora_config = LoraConfig(
    r=r,
    lora_alpha=alpha,
    target_modules=LORA_CONFIG["target_modules"],
    lora_dropout=LORA_CONFIG["lora_dropout"],
    bias=LORA_CONFIG["bias"],
    task_type="CAUSAL_LM",
)
```

---

## 2) What This Project Trains

Two adapters on top of `deepseek-ai/deepseek-coder-6.7b-instruct`:

- `coder_adapter`: instruction -> generated code
- `reviewer_adapter`: code -> review/feedback

Important: adapters are **deltas only**. They are not standalone full models.

At inference you always need:

1. Base model weights (downloaded from Hugging Face)
2. Adapter weights (`adapter_model.safetensors`)

---

## 3) Repository Map (Critical Files)

- `config/settings.py`
  - Global training/inference defaults, LoRA config, agent definitions.
- `src/training/qlora.py`
  - QLoRA trainer (load model, attach LoRA, train, save adapter).
- `src/training/trainer.py`
  - Orchestrates training for coder/reviewer agents.
- `src/models/base.py`
  - Inference base model loader (CUDA/MPS/CPU-offload logic).
- `src/models/adapter.py`
  - Dynamic adapter loading and switching with PEFT.
- `src/agents/base.py`
  - Core generation path and adapter activation.
- `src/api/main.py`
  - FastAPI startup and agent wiring.
- `src/api/routes.py`
  - `/generate`, `/review`, `/health` endpoints.

---

## 4) Version and Environment Strategy

This project uses two practical environments:

### 4.1 Colab (Training)

Use `requirements-colab.txt` to avoid API mismatches during QLoRA training.

### 4.2 Local machine (Inference)

Use `requirements.txt` for API/inference workflow.

Why split? Training stack and local stack often diverge in optimized ways (CUDA, TRL behavior, bitsandbytes availability, etc.).

---

## 5) End-to-End Pipeline (High Level)

1. Prepare environment.
2. Prepare/verify datasets (`data/final/*_train.jsonl`).
3. Train adapter(s) with QLoRA in Colab.
4. Verify adapter artifacts.
5. Transfer adapter(s) from Colab to local `data/adapters`.
6. Start API locally with correct runtime mode.
7. Hit `/generate` and `/review` endpoints.
8. Debug with log-based runbook if failures appear.

---

## 6) Step-by-Step SOP

## Step 1: Local Project Bootstrap

```bash
cd /Users/sankalpaaryal/Documents/code-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Always run uvicorn from venv Python:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Why: avoids PATH confusion where global `uvicorn` uses a different interpreter.

---

## Step 2: Colab Setup for Training

In Colab:

```bash
%cd /content/code-demo
!pip install -U --no-cache-dir -r requirements-colab.txt
```

If model download rate limits occur, login/token as needed:

```bash
!huggingface-cli login
```

---

## Step 3: Data Preparation

You need:

- `data/final/coder_train.jsonl`
- `data/final/reviewer_train.jsonl`

### Option A: Use existing prepared data (fastest)

Skip scraping and go to training.

### Option B: Full data build

1. Scrape raw examples:

```bash
python -m src.data.github_scraper
```

2. Format to coder/reviewer pairs:

```bash
python -m src.data.formatter data/raw/scikit_learn_raw.jsonl data/processed
python -m src.data.formatter data/raw/flask_raw.jsonl data/processed
python -m src.data.formatter data/raw/pytest_raw.jsonl data/processed
```

3. Merge and split:

```bash
python - <<'PY'
from src.data.formatter import merge_and_split

coder_files = [
    "data/processed/scikit_learn_coder.jsonl",
    "data/processed/flask_coder.jsonl",
    "data/processed/pytest_coder.jsonl",
]
reviewer_files = [
    "data/processed/scikit_learn_reviewer.jsonl",
    "data/processed/flask_reviewer.jsonl",
    "data/processed/pytest_reviewer.jsonl",
]
merge_and_split(coder_files, reviewer_files, "data/final", train_ratio=0.9)
PY
```

Optional sampling/selection:

```bash
python -m src.data.selector --fast
```

---

## Step 4: Launch QLoRA Training

Train both agents:

```bash
python -m src.training.trainer --agent both --data-dir data/final --adapters-dir data/adapters
```

Single adapter:

```bash
python -m src.training.trainer --agent coder --data-dir data/final --adapters-dir data/adapters
python -m src.training.trainer --agent reviewer --data-dir data/final --adapters-dir data/adapters
```

Equivalent scripts:

```bash
bash scripts/train_coder.sh
bash scripts/train_reviewer.sh
```

### What the trainer is doing internally

In `src/training/qlora.py`:

1. Load tokenizer/model
2. Convert model for k-bit training
3. Attach LoRA layers
4. Read dataset and create `text` field
5. Train with `SFTTrainer`
6. Save adapter + tokenizer + metadata

Example dataset conversion in code:

```python
if "instruction" in ex:
    text = "### Instruction:\n" + ex["instruction"] + "\n\n### Response:\n" + ex["code"] + "</s>"
else:
    text = "### Code:\n" + ex["code"] + "\n\n### Review:\n" + ex["review"] + "</s>"
```

---

## Step 5: Validate Training Artifacts

For `coder_adapter`:

```bash
ls -la data/adapters/coder_adapter
```

Required files:

- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer files
- `training_info.json`

Check LoRA config sanity:

```bash
cat data/adapters/coder_adapter/adapter_config.json
```

You should see `peft_type: "LORA"`, rank `r`, and target modules list.

---

## Step 6: Transfer Adapter from Colab to Local

In Colab:

```bash
%cd /content/code-demo
!zip -r /content/coder_adapter.zip data/adapters/coder_adapter
```

Download:

```python
from google.colab import files
files.download('/content/coder_adapter.zip')
```

On local:

```bash
cd /Users/sankalpaaryal/Documents/code-demo
unzip ~/Downloads/coder_adapter.zip -d .
```

Place should become:

- `/Users/sankalpaaryal/Documents/code-demo/data/adapters/coder_adapter`

---

## Step 7: Local Inference Mode Selection

### 7.1 CUDA Linux/Windows (best)

- Use GPU + bitsandbytes 4-bit path.

### 7.2 Apple Silicon / constrained local memory

Use CPU offload mode when MPS fails due memory/warmup behavior.

Recommended env vars:

```bash
export FORCE_CPU_INFERENCE=1
export MAX_CPU_MEMORY=10GiB
export HF_OFFLOAD_DIR=/Users/sankalpaaryal/Documents/code-demo/data/offload
```

### Why this is needed

In `src/models/base.py`, CPU offload path sets:

```python
base_kwargs["device_map"] = "auto"
base_kwargs["max_memory"] = {"cpu": os.getenv("MAX_CPU_MEMORY", "12GiB")}
base_kwargs["offload_folder"] = str(_cpu_offload_dir())
base_kwargs["offload_state_dict"] = True
```

And non-CUDA allocator warmup is patched off to avoid oversized allocations:

```python
_modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
```

---

## Step 8: Start API

```bash
cd /Users/sankalpaaryal/Documents/code-demo
source .venv/bin/activate
export FORCE_CPU_INFERENCE=1
export MAX_CPU_MEMORY=10GiB
export HF_OFFLOAD_DIR=/Users/sankalpaaryal/Documents/code-demo/data/offload
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Healthy startup logs should include:

- `Initializing CodeColosseum Inference Engine...`
- `Inference Engine ready.`
- `Application startup complete.`

---

## Step 9: Test Endpoints

Health:

```bash
curl -s http://127.0.0.1:8000/health
```

Coder:

```bash
curl -s -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Write a Python function to check palindrome","max_tokens":160,"temperature":0.2}'
```

Reviewer:

```bash
curl -s -X POST http://127.0.0.1:8000/review \
  -H "Content-Type: application/json" \
  -d '{"code":"def add(a,b): return a+b","max_tokens":160}'
```

---

## Step 10: How Adapter Switching Works at Runtime

In `src/agents/base.py`, each request activates the correct adapter before generate:

```python
if self.adapter_name:
    self.adapter_manager.set_adapter(self.adapter_name)
    self.model = self.adapter_manager.model
```

Generation uses merged kwargs + safe input device selection (important for offloaded models):

```python
gen_kwargs = {**INFERENCE_CONFIG, **kwargs}
gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
```

Why this matters:

- Prevents adapter mismatch between coder/reviewer calls
- Avoids passing `None` into generation params
- Handles dispatch/offload device ambiguity safely

---

## Step 11: Adapter Loader Details (Important Advanced Part)

`src/models/adapter.py` uses offload-aware logic.

When base model is dispatched on CPU/disk/meta, adapter injection is done with:

```python
load_kwargs["low_cpu_mem_usage"] = True
```

And during offload mode, `hf_device_map` is temporarily masked during PEFT adapter injection to avoid known PEFT+accelerate bugs on some model layouts.

This prevents runtime errors like:

- `KeyError: 'base_model.model.model.lm_head'`
- `TypeError: unhashable type: 'set'`

---

## 12) Operational Troubleshooting Matrix (Root Cause -> Fix)

## A) `LlamaForCausalLM.__init__() got unexpected keyword argument 'dtype'`

Root cause:

- API mismatch (`dtype` vs `torch_dtype`) across transformers/model class versions.

Fix:

- Keep compatibility retry logic in model loading (`src/models/base.py` / training loader).
- Prefer pinned versions for training stack.

## B) `SFTTrainer.__init__() got unexpected keyword argument 'processing_class'`

Root cause:

- TRL version mismatch with your trainer call signature.

Fix:

- Align TRL + transformers versions in Colab.
- If needed, swap `processing_class` <-> `tokenizer` depending on TRL version.

## C) `Unable to create tensor ... features ('text')`

Root cause:

- Dataset column handling / padding/truncation mismatch in collator path.

Fix:

- Ensure SFT trainer receives correct `dataset_text_field` and tokenization behavior.
- Avoid stray unprocessed columns in collator path.

## D) `No package metadata was found for bitsandbytes` (local Mac)

Root cause:

- bitsandbytes generally not available for local CPU/MPS path.

Fix:

- Use non-4-bit CPU/MPS fallback logic (already in `src/models/base.py`).

## E) `Invalid buffer size: 12.56 GiB` / MPS OOM during model load

Root cause:

- allocator warmup and memory ceiling issues on local MPS for this model.

Fix:

- Force CPU offload mode with env vars (`FORCE_CPU_INFERENCE=1`, `MAX_CPU_MEMORY`, `HF_OFFLOAD_DIR`).

## F) `KeyError 'base_model.model.model.lm_head'` / `unhashable type: 'set'` during adapter load

Root cause:

- PEFT offload-dispatch remap path issue.

Fix:

- Offload-safe adapter loading with temporary `hf_device_map` masking in `src/models/adapter.py`.

---

## 13) Quality Gates Before Declaring Success

Mark run successful only if all checks pass:

1. API startup log contains `Inference Engine ready.`
2. `GET /health` returns 200.
3. First `/generate` request loads/activates `coder_adapter` without traceback.
4. Response body contains non-empty generated code text.
5. (Optional) `/review` returns non-empty review text.

---

## 14) Recommended Test Cases (Adapter Validation)

## 9.1 Functional tests

1. Simple utility function
- Input: palindrome function request
- Expect: syntactically valid Python function with true/false behavior.

2. Data structure function
- Input: linked-list reverse request
- Expect: clear function signature + logic.

3. Error handling
- Input: request with invalid/empty constraint
- Expect: graceful generated fallback, not API crash.

## 9.2 API stability tests

1. Repeated same prompt (3-5 times)
- Expect: stable latency, no adapter reload error.

2. Switch endpoints (`/generate` then `/review` then `/generate`)
- Expect: adapter hot-switch works; no 500.

## 9.3 Resource behavior tests

1. Start server with offload env vars
- Expect: no immediate OOM.

2. First generation call after cold start
- Expect: may be slower, but successful.

---

## 15) How to Explain This Project to Others (Teaching Script)

### 10.1 60-second version

"We fine-tune a large code model efficiently using QLoRA. The base model stays frozen and quantized; we only train small LoRA adapters. We train two adapters: one for code generation and one for code review. At inference, the API loads the base model once and hot-swaps adapters per endpoint. This cuts training cost and storage dramatically compared to full fine-tuning."

### 10.2 5-minute technical version

1. Data pipeline converts raw code corpora into supervised text format.
2. QLoRA training in Colab T4 uses 4-bit base + LoRA layers.
3. Adapters are saved as lightweight artifacts.
4. Local inference composes base model + adapter dynamically.
5. For constrained local hardware, CPU offload mode is used.
6. Request path: API route -> agent -> set adapter -> generate -> return output.

---

## 16) Reproducibility Checklist

- [ ] Use pinned Colab requirements for training.
- [ ] Record exact date, model revision, adapter metadata.
- [ ] Keep `training_info.json` for each adapter version.
- [ ] Store command history used for training run.
- [ ] Keep offload env vars documented for local run.

---

## 17) Security and Hygiene

- Never commit live API tokens into `.env`.
- Rotate any token that appeared in logs or committed files.
- Keep adapter directories versioned (`coder_adapter_v1`, `v2`, etc.) for rollback.

---

## 18) Command Cheat Sheet

### Train both adapters

```bash
python -m src.training.trainer --agent both --data-dir data/final --adapters-dir data/adapters
```

### Start API (local offload mode)

```bash
source .venv/bin/activate
export FORCE_CPU_INFERENCE=1
export MAX_CPU_MEMORY=10GiB
export HF_OFFLOAD_DIR=/Users/sankalpaaryal/Documents/code-demo/data/offload
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Health check

```bash
curl -s http://127.0.0.1:8000/health
```

### Generate code

```bash
curl -s -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Write a Python function to check palindrome","max_tokens":160,"temperature":0.2}'
```

### Review code

```bash
curl -s -X POST http://127.0.0.1:8000/review \
  -H "Content-Type: application/json" \
  -d '{"code":"def add(a,b): return a+b","max_tokens":160}'
```

---

## 19) Final Reality Check

- Training adapters in Colab does **not** remove need for base model locally.
- Local inference always needs base model download at least once.
- Adapter loading issues on CPU/disk offload are framework-edge cases; your current code now includes practical safeguards.

This SOP is now the primary source of truth for running this project end-to-end.

---

## 20) Source Code Walkthrough (Step-by-Step, with Explanations)

This section maps the runtime flow to actual source code so you can explain both the **what** and **why** to others.

## 20.1 Step 1: Configuration is loaded once (`config/settings.py`)

Core configuration source:

```python
TRAINING_CONFIG = {
    "learning_rate": float(os.getenv("LEARNING_RATE", "2e-4")),
    "batch_size": int(os.getenv("BATCH_SIZE", "2")),
    "num_epochs": int(os.getenv("NUM_EPOCHS", "3")),
    "gradient_accumulation_steps": 4,
    "max_seq_length": 1024,
    "warmup_steps": 10,
}

LORA_CONFIG = {
    "r": int(os.getenv("LORA_R", "16")),
    "lora_alpha": int(os.getenv("LORA_ALPHA", "32")),
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
```

What this means:

- `TRAINING_CONFIG` controls optimization behavior and sequence sizing.
- `LORA_CONFIG` defines exactly where LoRA matrices are inserted.
- Agent mapping (`AGENTS`) binds `coder -> coder_adapter` and `reviewer -> reviewer_adapter`.

Teaching line:

- "This file is the single source of truth; CLI scripts and trainers inherit from it unless explicitly overridden."

## 20.2 Step 2: Data scraping stage (`src/data/github_scraper.py`)

Repo selection is intentionally constrained for speed:

```python
RECOMMENDED_REPOS = [
    {"name": "scikit-learn/scikit-learn", "max_files": 1000},
    {"name": "pallets/flask", "max_files": 500},
    {"name": "pytest-dev/pytest", "max_files": 500},
]
```

Search and pull files via GitHub Search API:

```python
params = {
    "q": f"repo:{owner}/{repo} extension:py",
    "per_page": 100,
    "page": page
}
response = self.session.get(url, params=params)
```

What to explain:

- This is a fast data bootstrap, not a full-repo semantic crawler.
- It prioritizes enough training diversity over total repository coverage.

## 20.3 Step 3: Formatting into trainable pairs (`src/data/formatter.py`)

Coder example creation:

```python
instruction = random.choice(self.CODER_TEMPLATES).format(description=desc)
return {
    "instruction": instruction,
    "code": code or ex["code"][:500],
    "source": ex["source_file"]
}
```

Reviewer example creation:

```python
return {
    "code": code,
    "review": review,
    "source": ex["source_file"]
}
```

Merge and split logic:

```python
coder_split = int(len(all_coder) * train_ratio)
reviewer_split = int(len(all_reviewer) * train_ratio)
```

What to explain:

- The trainer later expects JSONL records containing fields that can be converted into `text` prompt strings.
- A 90/10 split is used for train/eval to track overfitting.

## 20.4 Step 4: Training entrypoint orchestration (`src/training/trainer.py`)

Agent-specific calls:

```python
adapter_path = train_agent(
    agent_type="coder",
    data_file=self._data_file("coder"),
    output_dir=str(self.adapters_dir),
)
```

Sequential pipeline with memory flush between agents:

```python
coder_path = self.train_coder()
_gc.collect()
if _torch.cuda.is_available():
    _torch.cuda.synchronize()
    _torch.cuda.empty_cache()
reviewer_path = self.train_reviewer()
```

Why this matters:

- 6.7B model reloads are heavy; flushing between coder/reviewer prevents memory fragmentation and OOM on smaller GPUs.
- Failure of one agent does not prevent the second from running.

## 20.5 Step 5: QLoRA trainer internals (`src/training/qlora.py`)

### A) Build quantized base model config

```python
self.bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

### B) Build LoRA config

```python
self.lora_config = LoraConfig(
    r=r,
    lora_alpha=alpha,
    target_modules=LORA_CONFIG["target_modules"],
    lora_dropout=LORA_CONFIG["lora_dropout"],
    bias=LORA_CONFIG["bias"],
    task_type="CAUSAL_LM",
)
```

### C) Load tokenizer/model

```python
self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
self.base_model = AutoModelForCausalLM.from_pretrained(
    self.base_model_name,
    quantization_config=self.bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
```

### D) Convert model for k-bit training and attach LoRA

```python
self.base_model = prepare_model_for_kbit_training(
    self.base_model,
    use_gradient_checkpointing=True,
)
peft_model = get_peft_model(self.base_model, self.lora_config)
```

### E) Convert dataset rows to SFT-ready prompt text

```python
if "instruction" in ex:
    text = "### Instruction:\n" + ex["instruction"] + "\n\n### Response:\n" + ex["code"] + "</s>"
else:
    text = "### Code:\n" + ex["code"] + "\n\n### Review:\n" + ex["review"] + "</s>"
```

### F) SFT training config

```python
training_args = SFTConfig(
    num_train_epochs=epochs,
    per_device_train_batch_size=bs,
    gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
    optim="paged_adamw_8bit",
    eval_strategy="epoch",
    save_strategy="epoch",
    max_seq_length=TRAINING_CONFIG["max_seq_length"],
    dataset_text_field="text",
)
```

### G) Save adapters and metadata

```python
trainer.model.save_pretrained(output_path)
self.tokenizer.save_pretrained(output_path)
```

What to explain:

- The base model remains frozen and quantized.
- Only adapter parameters are optimized and saved.
- `training_info.json` is critical for traceability and reproducibility.

## 20.6 Step 6: Inference base model loading (`src/models/base.py`)

Device decision logic:

```python
use_cuda = torch.cuda.is_available()
force_cpu = os.getenv("FORCE_CPU_INFERENCE", "0") == "1"
use_mps = (not force_cpu) and _supports_mps()
```

CPU offload mode for constrained local systems:

```python
base_kwargs["device_map"] = "auto"
base_kwargs["max_memory"] = {"cpu": os.getenv("MAX_CPU_MEMORY", "12GiB")}
base_kwargs["offload_folder"] = str(_cpu_offload_dir())
base_kwargs["offload_state_dict"] = True
```

Compatibility retries (`dtype`/`torch_dtype`, trust flags):

```python
load_attempts = (
    {"trust_remote_code": False, "dtype": compute_dtype},
    {"trust_remote_code": False, "torch_dtype": compute_dtype},
    {"trust_remote_code": False},
    {"trust_remote_code": True, "torch_dtype": compute_dtype},
    {"trust_remote_code": True},
)
```

Allocator warmup patch on non-CUDA:

```python
_modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
```

What to explain:

- This loader is defensive by design because transformers behavior changes across versions/platforms.
- It is the reason your Mac can still run inference via CPU offload when MPS path fails.

## 20.7 Step 7: Adapter loading and switching (`src/models/adapter.py`)

First load vs additional load:

```python
if not isinstance(self.model, PeftModel):
    self.model = PeftModel.from_pretrained(self.model, str(path), adapter_name=adapter_name, **load_kwargs)
else:
    self.model.load_adapter(str(path), adapter_name=adapter_name, **load_kwargs)
```

Runtime selection:

```python
self.model.set_adapter(adapter_name)
self.current_adapter = adapter_name
```

Offload workaround path:

```python
if offload_mode:
    self._load_adapter_with_masked_device_map(adapter_name, path, load_kwargs)
```

Why this exists:

- On CPU/disk offloaded models, PEFT can fail in some remap paths.
- Temporary `hf_device_map` masking avoids those specific PEFT+accelerate incompatibilities.

## 20.8 Step 8: Generation pipeline (`src/agents/base.py`)

### A) Ensure correct adapter is active per call

```python
self.adapter_manager.set_adapter(self.adapter_name)
self.model = self.adapter_manager.model
```

### B) Resolve safe input device for dispatched models

```python
hf_device_map = getattr(self.model, "hf_device_map", None)
# choose cpu/mps/cuda from map, fallback to model.device, else cpu
```

### C) Merge inference args and sanitize

```python
gen_kwargs = {**INFERENCE_CONFIG, **kwargs}
gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
gen_kwargs.setdefault("remove_invalid_values", True)
gen_kwargs.setdefault("renormalize_logits", True)
```

### D) Handle invalid temperature requests

```python
if temp_value is not None and temp_value <= 0:
    gen_kwargs["do_sample"] = False
    gen_kwargs.pop("temperature", None)
```

### E) Fallback when sampling is numerically unstable

```python
except (RuntimeError, ValueError) as exc:
    unstable_probs = "probability tensor contains either `inf`, `nan` or element < 0" in msg
    bad_temp = "has to be a strictly positive float" in msg
    if unstable_probs or bad_temp:
        fallback_kwargs["do_sample"] = False
```

What to explain:

- This logic is why API remains robust under CPU offload numerical edge cases.
- It degrades gracefully from sampled decoding to greedy decoding instead of returning 500 immediately.

## 20.9 Step 9: API startup and request handling (`src/api/main.py`, `src/api/routes.py`)

Startup wiring:

```python
model, tokenizer = load_base_model_and_tokenizer()
manager = AdapterManager(model)
app.state.coder_agent = CoderAgent(model, tokenizer, manager)
app.state.reviewer_agent = ReviewerAgent(model, tokenizer, manager)
```

Generate route:

```python
code = agent.run(
    instruction=request.instruction,
    max_new_tokens=request.max_tokens,
    temperature=request.temperature
)
```

Error handling with stack trace:

```python
except Exception as exc:
    logger.error(f"Generation failed: {exc}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(exc))
```

What to explain:

- Startup initializes only base components; adapters load lazily on first request.
- Route-level traceback logging is essential for debugging deployment failures.

## 20.10 Step 10: End-to-end call chain (exact order)

When user calls `/generate`:

1. `src/api/routes.py -> generate_code(...)`
2. `src/agents/coder.py -> run(...)`
3. `src/agents/base.py -> _generate(...)`
4. `src/models/adapter.py -> set_adapter(...)`
5. `PeftModel.from_pretrained(...)` (first load) or `.load_adapter(...)`
6. `model.generate(...)`
7. Decode generated IDs and return response JSON

This call chain is the best way to explain runtime architecture during demos.

## 20.11 Step 11: What changed in your current codebase (important)

These hardening updates are already integrated and should be part of your explanation:

- Offload-safe base model loading in `src/models/base.py`.
- Offload-safe adapter load workaround in `src/models/adapter.py`.
- Agent model reference sync after adapter wrap in `src/agents/base.py`.
- Generation argument sanitization and greedy fallback in `src/agents/base.py`.
- Full traceback logging in `src/api/routes.py`.

Use this summary when presenting "production hardening" work.

