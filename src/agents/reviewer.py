"""
Reviewer Agent implementation.
Specialized in analyzing code and providing feedback.
"""

from typing import Dict, Any
from .base import BaseAgent
from config.settings import AGENTS

class ReviewerAgent(BaseAgent):
    """
    Agent that reviews code.
    """
    def __init__(self, model, tokenizer, adapter_manager):
        super().__init__(model, tokenizer, adapter_manager, AGENTS["reviewer"])

    def run(self, code: str, **kwargs) -> str:
        """
        Generates a review for given code.
        """
        # Format the prompt using the configured template
        prompt = self.prompt_template.format(code=code)
        
        # Run generation
        return self._generate(prompt, **kwargs)
