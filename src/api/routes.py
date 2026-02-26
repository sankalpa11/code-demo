"""
API routes for CodeColosseum.
Exposes Coder and Reviewer capabilities via REST.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Data Models ──────────────────────────────────────────────────────────────

class GenerationRequest(BaseModel):
    instruction: str = Field(..., example="Write a python function to scrape a website using beautifulsoup")
    max_tokens: Optional[int] = Field(None, example=512)
    temperature: Optional[float] = Field(None, example=0.2)

class GenerationResponse(BaseModel):
    code: str
    agent: str

class ReviewRequest(BaseModel):
    code: str = Field(..., example="def hello(): print('hello')")
    max_tokens: Optional[int] = Field(None, example=512)

class ReviewResponse(BaseModel):
    review: str
    agent: str

# ── Dependency Injection for Agents ──────────────────────────────────────────
# Note: These will be injected from the main app state

def get_coder_agent():
    from .main import app
    return app.state.coder_agent

def get_reviewer_agent():
    from .main import app
    return app.state.reviewer_agent

# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/generate", response_model=GenerationResponse)
async def generate_code(request: GenerationRequest, agent=Depends(get_coder_agent)):
    """
    Generate code from natural language instruction.
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Coder agent not initialized (check adapter status)")
        
    try:
        code = agent.run(
            instruction=request.instruction,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return GenerationResponse(code=code, agent="Coder")
    except Exception as exc:
        logger.error(f"Generation failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/review", response_model=ReviewResponse)
async def review_code(request: ReviewRequest, agent=Depends(get_reviewer_agent)):
    """
    Perform a code review on the provided snippet.
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Reviewer agent not initialized (check adapter status)")
        
    try:
        review = agent.run(
            code=request.code,
            max_new_tokens=request.max_tokens
        )
        return ReviewResponse(review=review, agent="Reviewer")
    except Exception as exc:
        logger.error(f"Review failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/health")
async def health_check():
    return {"status": "ok"}
