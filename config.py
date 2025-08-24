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
    
    # Memory settings (no arbitrary limits!)
    embedding_dim: int = 384
    temperature: float = 15.0  # Beta parameter for attention sharpness
    
    # Learning parameters
    base_learning_rate: float = 0.3
    salience_threshold: float = 0.4  # Minimum salience to store in processed memory (40% importance)
    similarity_threshold: float = 0.85  # Threshold for considering memories as duplicates (85% similar)
    
    # Memory update weights (sum to 1.0 for consistency)
    query_weight: float = 0.65  # Weight for query similarity in matching
    content_weight: float = 0.35  # Weight for content similarity in matching
    
    # EMA smoothing
    salience_ema_alpha: float = 0.1  # For exponential moving average

@dataclass
class StorageConfig:
    """Configuration for storage backends"""
    
    # Base paths
    state_dir: Path = Path("state")
    backup_dir: Path = Path("state/backups")  # For JSON exports
    
    # ChromaDB settings (primary storage)
    chromadb_path: Path = Path("state/chromadb")
    chromadb_required: bool = True  # Fail if ChromaDB not available
    
    # Cache settings
    cache_size: int = 1000  # Maximum cached items
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    
    # Persistence
    auto_backup: bool = True
    backup_interval: int = 100  # Backup every N operations
    max_backups: int = 10  # Keep last N backups

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
    
    # Generation parameters - reduced for conciseness
    temperature: float = 0.3  # Lower temperature for less randomness
    max_tokens: int = 150  # Reduced from 500 to encourage brevity
    repetition_penalty: float = 1.2  # Penalize repetitive outputs
    
    # Salience computation
    use_llm_salience: bool = True
    salience_prompt_template: str = """Rate importance (0-1 scale).
Goal: {goal}
Observation: {observation}
Output only a decimal number."""

@dataclass
class AgentConfig:
    """Configuration for the memory agent"""
    
    # Research settings
    convergence_window: int = 3  # Consecutive low-salience items before stopping
    min_salience: float = 0.35  # Minimum salience to continue processing (35% importance)
    decay_factor: float = 0.85  # Decay rate for iterative searches
    
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
        if os.getenv("MHN_EMBEDDING_DIM"):
            config.memory.embedding_dim = int(os.getenv("MHN_EMBEDDING_DIM"))
        
        if os.getenv("MHN_TEMPERATURE"):
            config.memory.temperature = float(os.getenv("MHN_TEMPERATURE"))
        
        if os.getenv("LLM_SERVER_URL"):
            config.llm.server_url = os.getenv("LLM_SERVER_URL")
        
        if os.getenv("CHROMADB_PATH"):
            config.storage.chromadb_path = Path(os.getenv("CHROMADB_PATH"))
        
        if os.getenv("STATE_DIR"):
            config.storage.state_dir = Path(os.getenv("STATE_DIR"))
            config.storage.chromadb_path = config.storage.state_dir / "chromadb"
            config.storage.backup_dir = config.storage.state_dir / "backups"
        
        # Ensure directories exist
        config.storage.state_dir.mkdir(parents=True, exist_ok=True)
        config.storage.chromadb_path.mkdir(parents=True, exist_ok=True)
        config.storage.backup_dir.mkdir(parents=True, exist_ok=True)
        
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