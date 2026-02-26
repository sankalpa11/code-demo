# CodeColosseum (Simple README)

This project runs two AI agents on top of a shared DeepSeek-Coder base model:
- `Coder` agent: instruction -> code
- `Reviewer` agent: code -> review feedback

## Basic Program Flow

1. Load the base model (`src/models/base.py`).
2. Load/switch LoRA adapters (`src/models/adapter.py`).
3. Create agents (`src/agents/coder.py`, `src/agents/reviewer.py`).
4. Start FastAPI app (`src/api/main.py`).
5. Call endpoints:
- `POST /generate` -> uses `coder_adapter`
- `POST /review` -> uses `reviewer_adapter`
- `GET /health` -> service check

## Basic Steps Required to Run

### 1. Set up environment

```bash
cd /Users/sankalpaaryal/Documents/code-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 2. Make sure training data exists

Required files:
- `data/final/coder_train.jsonl`
- `data/final/reviewer_train.jsonl`

### 3. Train adapters (if not already present)

```bash
python -m src.training.trainer --agent coder --data-dir data/final --adapters-dir data/adapters
python -m src.training.trainer --agent reviewer --data-dir data/final --adapters-dir data/adapters
```

Expected output folders:
- `data/adapters/coder_adapter`
- `data/adapters/reviewer_adapter`

### 4. Start API server

```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test endpoints

Health:
```bash
curl http://127.0.0.1:8000/health
```

Generate code:
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Write a Python function to reverse a string"}'
```
