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
    embedding_dim: int = 768  # Base embedding dimension
    temperature: float = 15.0  # Beta parameter for attention sharpness
    
    # Learning parameters
    base_learning_rate: float = 0.3
    # No salience threshold - all memories are stored and dynamically clustered
    similarity_threshold: float = 0.85  # Threshold for considering memories as duplicates (85% similar)
    
    # Memory update weights (sum to 1.0 for consistency)
    query_weight: float = 0.65  # Weight for query similarity in matching
    content_weight: float = 0.35  # Weight for content similarity in matching
    
    # EMA smoothing
    salience_ema_alpha: float = 0.1  # For exponential moving average

@dataclass
class DualSpaceConfig:
    """Configuration for dual-space encoding system"""
    
    # Dimension settings (overrides MemoryConfig.embedding_dim)
    euclidean_dim: int = 768  # Dimension for Euclidean space (from sentence transformer)
    hyperbolic_dim: int = 64  # Dimension for hyperbolic space (projected)
    
    # Learning parameters
    learning_rate: float = 0.01
    momentum: float = 0.9
    
    # Residual bounds (will be made scale-aware)
    euclidean_bound: float = 0.35  # Max residual norm relative to anchor
    hyperbolic_bound: float = 0.75  # Max residual norm in hyperbolic space
    use_relative_bounds: bool = True  # Use relative bounds instead of fixed
    
    # Hyperbolic stability parameters
    max_norm: float = 0.999  # Maximum norm in Poincaré ball (1 - epsilon)
    epsilon: float = 1e-5  # Small value for numerical stability
    
    # Decay and adaptation
    decay_factor: float = 0.995
    min_residual_norm: float = 1e-6  # Minimum residual norm before zeroing
    
    # HDBSCAN parameters (exposed for UI control)
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    
    # Drift hygiene
    field_adaptation_limits: dict = None  # Per-field adaptation limits
    enable_forgetting: bool = True  # Allow zeroing residuals
    
    def __post_init__(self):
        if self.field_adaptation_limits is None:
            self.field_adaptation_limits = {
                'who': 0.2,  # Limit adaptation on 'who' field
                'when': 0.3,  # Limit adaptation on 'when' field  
                'what': 0.5,  # Allow more adaptation on content
                'where': 0.4,
                'why': 0.5,
                'how': 0.5
            }

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
    embedding_model_name: str = "all-mpnet-base-v2"  # Updated to match downloaded model
    use_transformer: bool = True
    cache_embeddings: bool = True
    
    # Fallback to hash-based embeddings if transformer unavailable
    hash_dim: int = 768
    
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
    
    # LLM-based analysis (used for clustering and insights)
    use_llm_analysis: bool = True
    analysis_prompt_template: str = """Analyze relevance.
Context: {context}
Content: {content}
Output brief analysis."""

@dataclass
class AgentConfig:
    """Configuration for the memory agent"""
    
    # Research settings
    convergence_window: int = 3  # Consecutive items with low relevance before stopping
    # No minimum salience - use dynamic clustering instead
    decay_factor: float = 0.85  # Decay rate for iterative searches
    
    # Episode management
    auto_link_episodes: bool = True
    episode_timeout: int = 300  # Seconds before new episode

@dataclass
class TemporalConfig:
    """Configuration for temporal grouping and chaining"""
    
    # Time window for temporal grouping (in minutes)
    temporal_group_window: int = 30  # Events within 30 minutes are in same temporal group
    max_temporal_gap: int = 60  # Maximum gap (minutes) before creating new temporal group
    
    # Conversation continuity (for chat interfaces)
    conversation_window: int = 5  # Minutes to consider as same conversation
    
    # Time-based decay for similarity
    temporal_decay_rate: float = 0.95  # Decay factor per hour for temporal relevance
    
    # Whether to use episode_id at all for temporal grouping
    use_episode_for_temporal: bool = False  # If False, only use time proximity

@dataclass
class Config:
    """Main configuration container"""
    
    memory: MemoryConfig
    dual_space: DualSpaceConfig
    storage: StorageConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    agent: AgentConfig
    temporal: TemporalConfig
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables"""
        config = cls(
            memory=MemoryConfig(),
            dual_space=DualSpaceConfig(),
            storage=StorageConfig(),
            embedding=EmbeddingConfig(),
            llm=LLMConfig(),
            agent=AgentConfig(),
            temporal=TemporalConfig()
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
