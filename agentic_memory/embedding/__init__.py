"""Embedding module using llama.cpp for fast GPU embeddings."""
from .llama_embedder import LlamaEmbedder, get_llama_embedder

__all__ = ['LlamaEmbedder', 'get_llama_embedder']