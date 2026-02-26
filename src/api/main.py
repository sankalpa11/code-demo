"""
Main FastAPI application entry point.
Handles model loading on startup and orchestrates agents.
"""

import logging
import sys
from pathlib import Path
from fastapi import FastAPI
import uvicorn

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.base import load_base_model_and_tokenizer
from src.models.adapter import AdapterManager
from src.agents.coder import CoderAgent
from src.agents.reviewer import ReviewerAgent
from src.api.routes import router

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="CodeColosseum API",
    description="Fine-tuned DeepSeek-Coder agents for code generation and review.",
    version="1.0.0"
)

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """
    Initialize model and agents on startup.
    """
    logger.info("Initializing CodeColosseum Inference Engine...")
    
    try:
        # 1. Load base model
        model, tokenizer = load_base_model_and_tokenizer()
        
        # 2. Initialize Adapter Manager
        manager = AdapterManager(model)
        
        # 3. Initialize Agents
        app.state.coder_agent = CoderAgent(model, tokenizer, manager)
        app.state.reviewer_agent = ReviewerAgent(model, tokenizer, manager)
        
        logger.info("Inference Engine ready.")
    except Exception as exc:
        logger.error(f"Startup failed: {exc}", exc_info=True)
        # We don't exit(1) here to allow the process to start so we can see 500 errors
        app.state.coder_agent = None
        app.state.reviewer_agent = None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
