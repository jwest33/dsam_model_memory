from __future__ import annotations
from typing import Optional
from .config import cfg

# We try to use llama-cpp-python tokenizer when available; fall back to heuristic
class TokenizerAdapter:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or cfg.llm_model
        try:
            from llama_cpp import Llama
            # Lazy load a small tokenizer-only instance (no model weights needed for tokenize)
            # llama.cpp python binding requires a model path to load a full model; that's heavy.
            # So we use HTTP count as first choice when available.
            self.llama = None
        except Exception:
            self.llama = None

    def count_tokens(self, text: str) -> int:
        # Try server-side token counting if llama.cpp provides /tokenize (OpenAI compat typically doesn't).
        # We implement a fast heuristic here; you can replace with exact tokenizer if you prefer.
        # Heuristic: ~ 3.5 chars/token as a decent rule for English-like text.
        if not text:
            return 0
        approx = max(1, int(len(text) / 3.5))
        return approx
