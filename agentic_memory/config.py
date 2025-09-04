import os
from dataclasses import dataclass

def get_config_value(key: str, default: str) -> str:
    """Get config value from environment or ConfigManager if available"""
    # First check environment variable
    value = os.getenv(key)
    if value is not None:
        return value
    
    # Try to get from ConfigManager if it's been initialized
    try:
        from .config_manager import ConfigManager
        # Use default db path since we're bootstrapping
        db_path = os.getenv("AM_DB_PATH", "./amemory.sqlite3")
        manager = ConfigManager(db_path)
        stored_value = manager.get_value(key)
        if stored_value is not None:
            return str(stored_value)
    except:
        pass  # ConfigManager not available yet
    
    return default

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
    # Multi-part extraction
    use_multi_part_extraction: bool = os.getenv("AM_USE_MULTI_PART", "true").lower() in ['true', '1', 'yes']
    multi_part_threshold: int = int(os.getenv("AM_MULTI_PART_THRESHOLD", "200"))  # Min chars to trigger multi-part
    
    # Dynamic/Attention-based retrieval
    use_attention_retrieval: bool = os.getenv("AM_USE_ATTENTION", "true").lower() in ['true', '1', 'yes']
    embed_dim: int = int(os.getenv("AM_EMBED_DIM", "384"))  # Embedding dimension
    attention_heads: int = int(os.getenv("AM_ATTENTION_HEADS", "8"))  # Number of attention heads
    consolidation_state_path: str = os.getenv("AM_CONSOLIDATION_PATH", "./consolidation_state.pkl")
    
    # Liquid clustering parameters
    use_liquid_clustering: bool = os.getenv("AM_USE_LIQUID_CLUSTERS", "true").lower() in ['true', '1', 'yes']
    cluster_flow_rate: float = float(os.getenv("AM_CLUSTER_FLOW_RATE", "0.1"))
    cluster_merge_threshold: float = float(os.getenv("AM_CLUSTER_MERGE_THRESHOLD", "0.85"))
    cluster_energy_decay: float = float(os.getenv("AM_CLUSTER_ENERGY_DECAY", "0.99"))
    
    # Memory consolidation parameters  
    hebbian_learning_rate: float = float(os.getenv("AM_HEBBIAN_RATE", "0.01"))
    synaptic_decay_rate: float = float(os.getenv("AM_SYNAPTIC_DECAY", "0.001"))
    memory_budget: int = int(os.getenv("AM_MEMORY_BUDGET", "10000"))  # Max memories before pruning
    
    # Embedding drift parameters
    embedding_momentum_rate: float = float(os.getenv("AM_EMBED_MOMENTUM", "0.95"))
    drift_blend_ratio: float = float(os.getenv("AM_DRIFT_BLEND", "0.3"))  # How much drift affects embeddings

cfg = Config()
