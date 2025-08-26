"""
Singleton embedder to prevent repeated model loading.

This ensures the sentence transformer model is only loaded once.
"""

from typing import Optional
from embedding.embedder import TextEmbedder, FiveW1HEmbedder

_text_embedder_instance: Optional[TextEmbedder] = None
_five_w1h_embedder_instance: Optional[FiveW1HEmbedder] = None


def get_text_embedder() -> TextEmbedder:
    """Get or create the singleton TextEmbedder instance"""
    global _text_embedder_instance
    if _text_embedder_instance is None:
        _text_embedder_instance = TextEmbedder()
    return _text_embedder_instance


def get_five_w1h_embedder() -> FiveW1HEmbedder:
    """Get or create the singleton FiveW1HEmbedder instance"""
    global _five_w1h_embedder_instance
    if _five_w1h_embedder_instance is None:
        # Use the singleton text embedder
        text_embedder = get_text_embedder()
        _five_w1h_embedder_instance = FiveW1HEmbedder(text_embedder)
    return _five_w1h_embedder_instance
