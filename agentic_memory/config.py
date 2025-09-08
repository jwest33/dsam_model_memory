"""
Unified configuration module using ConfigManager as single source of truth.
This replaces the old static Config dataclass with dynamic configuration management.
"""
import os
from typing import Any, Optional
from .config_manager import ConfigManager

# Singleton instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get or create the singleton ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        db_path = os.getenv("AM_DB_PATH", "./amemory.sqlite3")
        _config_manager = ConfigManager(db_path)
    return _config_manager

# Configuration access class for backward compatibility
class Config:
    """Configuration access wrapper for backward compatibility."""
    
    def __init__(self):
        self._manager = get_config_manager()
    
    def __getattr__(self, name: str) -> Any:
        """Get configuration value by attribute name."""
        # Map old attribute names to new config keys
        mapping = {
            'llm_base_url': 'AM_LLM_BASE_URL',
            'llm_model': 'AM_LLM_MODEL',
            'context_window': 'AM_CONTEXT_WINDOW',
            'db_path': 'AM_DB_PATH',
            'index_path': 'AM_INDEX_PATH',
            'embed_model_name': 'AM_EMBED_MODEL',
            'reserve_output_tokens': 'AM_RESERVE_OUTPUT_TOKENS',
            'reserve_system_tokens': 'AM_RESERVE_SYSTEM_TOKENS',
            'w_semantic': 'AM_W_SEMANTIC',
            'w_lexical': 'AM_W_LEXICAL',
            'w_recency': 'AM_W_RECENCY',
            'w_actor': 'AM_W_ACTOR',
            'w_spatial': 'AM_W_SPATIAL',
            'w_usage': 'AM_W_USAGE',
            'mmr_lambda': 'AM_MMR_LAMBDA',
            'use_multi_part_extraction': 'AM_USE_MULTI_PART',
            'multi_part_threshold': 'AM_MULTI_PART_THRESHOLD',
            'use_attention_retrieval': 'AM_USE_ATTENTION',
            'embed_dim': 'AM_EMBED_DIM',
            'attention_heads': 'AM_ATTENTION_HEADS',
            'consolidation_state_path': 'AM_CONSOLIDATION_PATH',
            'use_liquid_clustering': 'AM_USE_LIQUID_CLUSTERS',
            'cluster_flow_rate': 'AM_CLUSTER_FLOW_RATE',
            'cluster_merge_threshold': 'AM_CLUSTER_MERGE_THRESHOLD',
            'cluster_energy_decay': 'AM_CLUSTER_ENERGY_DECAY',
            'hebbian_learning_rate': 'AM_HEBBIAN_RATE',
            'synaptic_decay_rate': 'AM_SYNAPTIC_DECAY',
            'memory_budget': 'AM_MEMORY_BUDGET',
            'embedding_momentum_rate': 'AM_EMBED_MOMENTUM',
            'drift_blend_ratio': 'AM_DRIFT_BLEND',
        }
        
        # Get the config key
        config_key = mapping.get(name)
        if config_key:
            value = self._manager.get_value(config_key)
            if value is not None:
                return value
        
        # Check if it's a direct config key
        upper_name = name.upper()
        if upper_name.startswith('AM_'):
            value = self._manager.get_value(upper_name)
            if value is not None:
                return value
        
        # Fallback to environment variable
        env_key = f"AM_{upper_name}"
        if os.getenv(env_key):
            return os.getenv(env_key)
        
        raise AttributeError(f"Configuration '{name}' not found")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default fallback."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default
    
    def get_all(self) -> dict:
        """Get all configuration values as dictionary."""
        return self._manager.get_all_as_dict()

# Create global config instance for backward compatibility
cfg = Config()

# Export convenience functions
def get_server_config() -> dict:
    """Get server-related configuration."""
    manager = get_config_manager()
    return {
        'web_port': manager.get_value('AM_WEB_PORT'),
        'api_port': manager.get_value('AM_API_PORT'),
        'llama_port': manager.get_value('AM_LLAMA_PORT'),
        'llama_url': f"http://localhost:{manager.get_value('AM_LLAMA_PORT')}",
        'cache_ttl': manager.get_value('AM_CACHE_TTL'),
        'rate_limit_requests': manager.get_value('AM_RATE_LIMIT_REQUESTS'),
        'rate_limit_window': manager.get_value('AM_RATE_LIMIT_WINDOW'),
    }

def get_generation_defaults() -> dict:
    """Get default LLM generation parameters."""
    manager = get_config_manager()
    return {
        'temperature': manager.get_value('AM_DEFAULT_TEMPERATURE'),
        'max_tokens': manager.get_value('AM_DEFAULT_MAX_TOKENS'),
        'repetition_penalty': manager.get_value('AM_DEFAULT_REPETITION_PENALTY'),
        'top_p': manager.get_value('AM_DEFAULT_TOP_P'),
        'top_k': manager.get_value('AM_DEFAULT_TOP_K'),
    }

def get_upload_config() -> dict:
    """Get file upload configuration."""
    manager = get_config_manager()
    extensions = manager.get_value('AM_UPLOAD_EXTENSIONS')
    return {
        'max_size': manager.get_value('AM_MAX_UPLOAD_SIZE'),
        'allowed_extensions': set(extensions.split(',')) if extensions else set(),
    }

def get_model_path() -> str:
    """Get the model path with OS-appropriate handling."""
    manager = get_config_manager()
    path = manager.get_value('AM_MODEL_PATH')
    
    if not path:
        # Fallback to LLM_MODEL_PATH for backward compatibility
        path = os.getenv('LLM_MODEL_PATH', '')
    
    # Handle Windows path conversion if needed
    import platform
    if platform.system() == "Windows" and path.startswith("/c/"):
        # Convert Unix-style /c/path to Windows C:\path
        path = path.replace("/c/", "C:\\").replace("/", "\\")
    
    return path

# For backward compatibility with existing imports
def get_config_value(key: str, default: str) -> str:
    """Get config value from ConfigManager or environment."""
    manager = get_config_manager()
    value = manager.get_value(key)
    if value is not None:
        return str(value)
    
    # Fallback to environment
    value = os.getenv(key)
    if value is not None:
        return value
    
    return default