"""
Configuration for 5W1H + MHN Memory Framework
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class MemoryConfig:
    """Configuration for Modern Hopfield Network memory"""
    
    # Memory capacity settings
    max_memory_slots: int = 512
    embedding_dim: int = 384
    temperature: float = 15.0  # Beta parameter for attention sharpness
    
    # Learning parameters
    base_learning_rate: float = 0.3
    salience_threshold: float = 0.3  # Minimum salience to store in processed memory
    similarity_threshold: float = 0.8  # Threshold for considering memories as duplicates
    
    # Memory update weights
    query_weight: float = 0.7  # Weight for query similarity in matching
    content_weight: float = 0.3  # Weight for content similarity in matching
    
    # Eviction strategy
    salience_weight: float = 1.2
    usage_weight: float = 0.6
    age_weight: float = 0.6
    
    # EMA smoothing
    salience_ema_alpha: float = 0.1  # For exponential moving average

@dataclass
class StorageConfig:
    """Configuration for storage backends"""
    
    # Base paths
    state_dir: Path = Path("state")
    raw_memory_path: Path = Path("state/raw_memories.json")
    processed_memory_path: Path = Path("state/processed_memories.json")
    hopfield_state_path: Path = Path("state/hopfield_state.npz")
    
    # ChromaDB settings
    use_chromadb: bool = True
    chromadb_path: Path = Path("state/chromadb")
    raw_collection_name: str = "raw_memories"
    processed_collection_name: str = "processed_memories"
    
    # Persistence
    auto_save: bool = True
    save_interval: int = 100  # Save every N operations

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    
    # Model settings
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_transformer: bool = True
    cache_embeddings: bool = True
    
    # Fallback to hash-based embeddings if transformer unavailable
    hash_dim: int = 384
    
    # Role embeddings for 5W1H
    add_role_embeddings: bool = True
    role_embedding_scale: float = 0.2

@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    
    # Server settings (from llama_server_client)
    server_url: str = "http://localhost:8000"
    timeout: int = 30
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 500
    
    # Salience computation
    use_llm_salience: bool = True
    salience_prompt_template: str = """Rate the importance of this observation on a scale of 0 to 1.

Goal: {goal}
Query: {query}
Observation: {observation}
Novelty: {novelty:.2f}
Overlap: {overlap:.2f}

Consider:
1. Relevance to the goal
2. Information novelty
3. Potential for future use

Respond with only a number between 0 and 1."""

@dataclass
class AgentConfig:
    """Configuration for the memory agent"""
    
    # Research settings (simplified - no web search)
    convergence_window: int = 3  # Consecutive low-salience items before stopping
    min_salience: float = 0.25
    decay_factor: float = 0.8
    
    # Episode management
    auto_link_episodes: bool = True
    episode_timeout: int = 300  # Seconds before new episode

@dataclass
class Config:
    """Main configuration container"""
    
    memory: MemoryConfig
    storage: StorageConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    agent: AgentConfig
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables"""
        config = cls(
            memory=MemoryConfig(),
            storage=StorageConfig(),
            embedding=EmbeddingConfig(),
            llm=LLMConfig(),
            agent=AgentConfig()
        )
        
        # Override with environment variables
        if os.getenv("MHN_MAX_MEMORY"):
            config.memory.max_memory_slots = int(os.getenv("MHN_MAX_MEMORY"))
        
        if os.getenv("MHN_EMBEDDING_DIM"):
            config.memory.embedding_dim = int(os.getenv("MHN_EMBEDDING_DIM"))
        
        if os.getenv("MHN_TEMPERATURE"):
            config.memory.temperature = float(os.getenv("MHN_TEMPERATURE"))
        
        if os.getenv("LLM_SERVER_URL"):
            config.llm.server_url = os.getenv("LLM_SERVER_URL")
        
        if os.getenv("USE_CHROMADB"):
            config.storage.use_chromadb = os.getenv("USE_CHROMADB").lower() in ("true", "1", "yes")
        
        if os.getenv("STATE_DIR"):
            config.storage.state_dir = Path(os.getenv("STATE_DIR"))
            config.storage.raw_memory_path = config.storage.state_dir / "raw_memories.json"
            config.storage.processed_memory_path = config.storage.state_dir / "processed_memories.json"
            config.storage.hopfield_state_path = config.storage.state_dir / "hopfield_state.npz"
            config.storage.chromadb_path = config.storage.state_dir / "chromadb"
        
        # Ensure directories exist
        config.storage.state_dir.mkdir(parents=True, exist_ok=True)
        if config.storage.use_chromadb:
            config.storage.chromadb_path.mkdir(parents=True, exist_ok=True)
        
        return config

# Global configuration instance
_config = None

def get_config() -> Config:
    """Get or create the global configuration"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config

def reset_config():
    """Reset the global configuration"""
    global _config
    _config = None