import os
from dataclasses import dataclass

@dataclass
class Config:
    llm_base_url: str = os.getenv("AM_LLM_BASE_URL", "http://localhost:8000/v1")
    llm_model: str = os.getenv("AM_LLM_MODEL", "Qwen3-4b-instruct-2507")
    context_window: int = int(os.getenv("AM_CONTEXT_WINDOW", "8192"))
    db_path: str = os.getenv("AM_DB_PATH", "./amemory.sqlite3")
    index_path: str = os.getenv("AM_INDEX_PATH", "./faiss.index")
    embed_model_name: str = os.getenv("AM_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reserve_output_tokens: int = int(os.getenv("AM_RESERVE_OUTPUT_TOKENS", "1024"))
    reserve_system_tokens: int = int(os.getenv("AM_RESERVE_SYSTEM_TOKENS", "512"))
    # Retrieval weights
    w_semantic: float = float(os.getenv("AM_W_SEMANTIC", "0.55"))
    w_lexical: float = float(os.getenv("AM_W_LEXICAL", "0.20"))
    w_recency: float = float(os.getenv("AM_W_RECENCY", "0.10"))
    w_actor: float = float(os.getenv("AM_W_ACTOR", "0.07"))
    w_spatial: float = float(os.getenv("AM_W_SPATIAL", "0.03"))
    w_usage: float = float(os.getenv("AM_W_USAGE", "0.05"))
    mmr_lambda: float = float(os.getenv("AM_MMR_LAMBDA", "0.5"))

cfg = Config()
