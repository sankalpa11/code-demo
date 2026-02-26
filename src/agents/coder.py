"""
Coder Agent implementation.
Specialized in generating code from natural language instructions.
"""

from typing import Dict, Any
from .base import BaseAgent
from config.settings import AGENTS

class CoderAgent(BaseAgent):
    """
    Agent that generates code.
    """
    def __init__(self, model, tokenizer, adapter_manager):
        super().__init__(model, tokenizer, adapter_manager, AGENTS["coder"])

    def run(self, instruction: str, **kwargs) -> str:
        """
        Generates code for a given instruction.
        """
        # Format the prompt using the configured template
        prompt = self.prompt_template.format(instruction=instruction)
        
        # Run generation
        return self._generate(prompt, **kwargs)
