"""Configuration management system with persistence and UI controls"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
import sqlite3
from contextlib import contextmanager

class ConfigType(Enum):
    """Types of configuration values"""
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    CHOICE = "choice"  # For dropdowns with specific options

@dataclass
class ConfigSetting:
    """Definition of a configuration setting"""
    key: str
    display_name: str
    description: str
    type: ConfigType
    default_value: Any
    current_value: Any
    category: str
    editable: bool = True
    requires_restart: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[str]] = None
    unit: Optional[str] = None  # For display (e.g., "tokens", "seconds", "%")
    advanced: bool = False  # Hide in basic view
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a proposed value"""
        if self.type == ConfigType.BOOLEAN:
            if not isinstance(value, bool):
                try:
                    value = str(value).lower() in ['true', '1', 'yes', 'on']
                except:
                    return False, "Must be a boolean value"
                    
        elif self.type == ConfigType.INTEGER:
            try:
                value = int(value)
                if self.min_value is not None and value < self.min_value:
                    return False, f"Must be at least {self.min_value}"
                if self.max_value is not None and value > self.max_value:
                    return False, f"Must be at most {self.max_value}"
            except:
                return False, "Must be an integer"
                
        elif self.type == ConfigType.FLOAT:
            try:
                value = float(value)
                if self.min_value is not None and value < self.min_value:
                    return False, f"Must be at least {self.min_value}"
                if self.max_value is not None and value > self.max_value:
                    return False, f"Must be at most {self.max_value}"
            except:
                return False, "Must be a number"
                
        elif self.type == ConfigType.CHOICE:
            if self.choices and value not in self.choices:
                return False, f"Must be one of: {', '.join(self.choices)}"
                
        return True, None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'key': self.key,
            'display_name': self.display_name,
            'description': self.description,
            'type': self.type.value,
            'default_value': self.default_value,
            'current_value': self.current_value,
            'category': self.category,
            'editable': self.editable,
            'requires_restart': self.requires_restart,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'choices': self.choices,
            'unit': self.unit,
            'advanced': self.advanced
        }

class ConfigManager:
    """Manages configuration with persistence and UI integration"""
    
    def __init__(self, db_path: str = "./amemory.sqlite3"):
        self.db_path = db_path
        self._ensure_tables()
        self.settings = self._define_settings()
        self._load_overrides()
        
    @contextmanager
    def connect(self):
        """Database connection context manager"""
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()
    
    def _ensure_tables(self):
        """Create config tables if they don't exist"""
        with self.connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS config_overrides (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT
                )
            """)
            
            con.execute("""
                CREATE TABLE IF NOT EXISTS config_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    changed_at TEXT NOT NULL,
                    changed_by TEXT,
                    reason TEXT
                )
            """)
    
    def _define_settings(self) -> Dict[str, ConfigSetting]:
        """Define all configurable settings"""
        settings = {}
        
        # Retrieval Settings
        settings['AM_USE_ATTENTION'] = ConfigSetting(
            key='AM_USE_ATTENTION',
            display_name='Use Attention Mechanism',
            description='Enable attention-based dynamic memory retrieval',
            type=ConfigType.BOOLEAN,
            default_value=True,
            current_value=os.getenv('AM_USE_ATTENTION', 'true').lower() in ['true', '1', 'yes'],
            category='Retrieval',
            editable=True,
            requires_restart=False
        )
        
        settings['AM_USE_LIQUID_CLUSTERS'] = ConfigSetting(
            key='AM_USE_LIQUID_CLUSTERS',
            display_name='Use Liquid Clustering',
            description='Enable self-organizing memory clusters that flow based on usage',
            type=ConfigType.BOOLEAN,
            default_value=True,
            current_value=os.getenv('AM_USE_LIQUID_CLUSTERS', 'true').lower() in ['true', '1', 'yes'],
            category='Retrieval',
            editable=True,
            requires_restart=False
        )
        
        settings['AM_USE_MULTI_PART'] = ConfigSetting(
            key='AM_USE_MULTI_PART',
            display_name='Multi-Part Extraction',
            description='Break complex content into multiple memory records',
            type=ConfigType.BOOLEAN,
            default_value=True,
            current_value=os.getenv('AM_USE_MULTI_PART', 'true').lower() in ['true', '1', 'yes'],
            category='Extraction',
            editable=True,
            requires_restart=False
        )
        
        # Weight Settings (only used when attention is disabled)
        settings['AM_W_SEMANTIC'] = ConfigSetting(
            key='AM_W_SEMANTIC',
            display_name='Semantic Weight',
            description='Weight for semantic similarity (when attention disabled)',
            type=ConfigType.FLOAT,
            default_value=0.55,
            current_value=float(os.getenv('AM_W_SEMANTIC', '0.55')),
            category='Weights',
            editable=True,
            requires_restart=False,
            min_value=0.0,
            max_value=1.0,
            unit='%',
            advanced=True
        )
        
        settings['AM_W_LEXICAL'] = ConfigSetting(
            key='AM_W_LEXICAL',
            display_name='Lexical Weight',
            description='Weight for keyword matches (when attention disabled)',
            type=ConfigType.FLOAT,
            default_value=0.20,
            current_value=float(os.getenv('AM_W_LEXICAL', '0.20')),
            category='Weights',
            editable=True,
            requires_restart=False,
            min_value=0.0,
            max_value=1.0,
            unit='%',
            advanced=True
        )
        
        settings['AM_W_RECENCY'] = ConfigSetting(
            key='AM_W_RECENCY',
            display_name='Recency Weight',
            description='Weight for recent memories (when attention disabled)',
            type=ConfigType.FLOAT,
            default_value=0.10,
            current_value=float(os.getenv('AM_W_RECENCY', '0.10')),
            category='Weights',
            editable=True,
            requires_restart=False,
            min_value=0.0,
            max_value=1.0,
            unit='%',
            advanced=True
        )
        
        settings['AM_W_ACTOR'] = ConfigSetting(
            key='AM_W_ACTOR',
            display_name='Actor Weight',
            description='Weight for actor relevance (when attention disabled)',
            type=ConfigType.FLOAT,
            default_value=0.07,
            current_value=float(os.getenv('AM_W_ACTOR', '0.07')),
            category='Weights',
            editable=True,
            requires_restart=False,
            min_value=0.0,
            max_value=1.0,
            unit='%',
            advanced=True
        )
        
        settings['AM_W_SPATIAL'] = ConfigSetting(
            key='AM_W_SPATIAL',
            display_name='Spatial Weight',
            description='Weight for spatial proximity (when attention disabled)',
            type=ConfigType.FLOAT,
            default_value=0.03,
            current_value=float(os.getenv('AM_W_SPATIAL', '0.03')),
            category='Weights',
            editable=True,
            requires_restart=False,
            min_value=0.0,
            max_value=1.0,
            unit='%',
            advanced=True
        )
        
        settings['AM_W_USAGE'] = ConfigSetting(
            key='AM_W_USAGE',
            display_name='Usage Weight',
            description='Weight for usage patterns (when attention disabled)',
            type=ConfigType.FLOAT,
            default_value=0.05,
            current_value=float(os.getenv('AM_W_USAGE', '0.05')),
            category='Weights',
            editable=True,
            requires_restart=False,
            min_value=0.0,
            max_value=1.0,
            unit='%',
            advanced=True
        )
        
        # Memory Management
        settings['AM_MEMORY_BUDGET'] = ConfigSetting(
            key='AM_MEMORY_BUDGET',
            display_name='Memory Budget',
            description='Maximum number of memories before pruning',
            type=ConfigType.INTEGER,
            default_value=10000,
            current_value=int(os.getenv('AM_MEMORY_BUDGET', '10000')),
            category='Memory',
            editable=True,
            requires_restart=False,
            min_value=100,
            max_value=100000,
            unit='memories'
        )
        
        settings['AM_CONTEXT_WINDOW'] = ConfigSetting(
            key='AM_CONTEXT_WINDOW',
            display_name='Context Window',
            description='Maximum context size for LLM',
            type=ConfigType.INTEGER,
            default_value=8192,
            current_value=int(os.getenv('AM_CONTEXT_WINDOW', '8192')),
            category='LLM',
            editable=True,
            requires_restart=False,
            min_value=2048,
            max_value=32768,
            unit='tokens'
        )
        
        settings['AM_RESERVE_OUTPUT_TOKENS'] = ConfigSetting(
            key='AM_RESERVE_OUTPUT_TOKENS',
            display_name='Reserved Output Tokens',
            description='Tokens reserved for LLM output',
            type=ConfigType.INTEGER,
            default_value=1024,
            current_value=int(os.getenv('AM_RESERVE_OUTPUT_TOKENS', '1024')),
            category='LLM',
            editable=True,
            requires_restart=False,
            min_value=256,
            max_value=4096,
            unit='tokens'
        )
        
        # Clustering Parameters
        settings['AM_CLUSTER_FLOW_RATE'] = ConfigSetting(
            key='AM_CLUSTER_FLOW_RATE',
            display_name='Cluster Flow Rate',
            description='How quickly memories flow between clusters',
            type=ConfigType.FLOAT,
            default_value=0.1,
            current_value=float(os.getenv('AM_CLUSTER_FLOW_RATE', '0.1')),
            category='Clustering',
            editable=True,
            requires_restart=False,
            min_value=0.01,
            max_value=0.5,
            advanced=True
        )
        
        settings['AM_CLUSTER_MERGE_THRESHOLD'] = ConfigSetting(
            key='AM_CLUSTER_MERGE_THRESHOLD',
            display_name='Cluster Merge Threshold',
            description='Similarity threshold for merging clusters',
            type=ConfigType.FLOAT,
            default_value=0.85,
            current_value=float(os.getenv('AM_CLUSTER_MERGE_THRESHOLD', '0.85')),
            category='Clustering',
            editable=True,
            requires_restart=False,
            min_value=0.5,
            max_value=0.99,
            advanced=True
        )
        
        # Learning Parameters
        settings['AM_HEBBIAN_RATE'] = ConfigSetting(
            key='AM_HEBBIAN_RATE',
            display_name='Learning Rate',
            description='Rate of Hebbian learning for memory connections',
            type=ConfigType.FLOAT,
            default_value=0.01,
            current_value=float(os.getenv('AM_HEBBIAN_RATE', '0.01')),
            category='Learning',
            editable=True,
            requires_restart=False,
            min_value=0.001,
            max_value=0.1,
            advanced=True
        )
        
        settings['AM_SYNAPTIC_DECAY'] = ConfigSetting(
            key='AM_SYNAPTIC_DECAY',
            display_name='Memory Decay Rate',
            description='Rate at which unused memory connections weaken',
            type=ConfigType.FLOAT,
            default_value=0.001,
            current_value=float(os.getenv('AM_SYNAPTIC_DECAY', '0.001')),
            category='Learning',
            editable=True,
            requires_restart=False,
            min_value=0.0001,
            max_value=0.01,
            advanced=True
        )
        
        # Server Settings
        settings['AM_WEB_PORT'] = ConfigSetting(
            key='AM_WEB_PORT',
            display_name='Web Interface Port',
            description='Port for the Flask web interface',
            type=ConfigType.INTEGER,
            default_value=5001,
            current_value=int(os.getenv('AM_WEB_PORT', '5001')),
            category='Server',
            editable=True,
            requires_restart=True,
            min_value=1024,
            max_value=65535
        )
        
        settings['AM_API_PORT'] = ConfigSetting(
            key='AM_API_PORT',
            display_name='API Wrapper Port',
            description='Port for the FastAPI wrapper server',
            type=ConfigType.INTEGER,
            default_value=8001,
            current_value=int(os.getenv('AM_API_PORT', '8001')),
            category='Server',
            editable=True,
            requires_restart=True,
            min_value=1024,
            max_value=65535
        )
        
        settings['AM_LLAMA_PORT'] = ConfigSetting(
            key='AM_LLAMA_PORT',
            display_name='Llama Server Port',
            description='Port for the llama.cpp server',
            type=ConfigType.INTEGER,
            default_value=8000,
            current_value=int(os.getenv('AM_LLAMA_PORT', '8000')),
            category='Server',
            editable=True,
            requires_restart=True,
            min_value=1024,
            max_value=65535
        )
        
        settings['AM_CACHE_TTL'] = ConfigSetting(
            key='AM_CACHE_TTL',
            display_name='Cache TTL',
            description='Time to live for cached API responses',
            type=ConfigType.INTEGER,
            default_value=300,
            current_value=int(os.getenv('AM_CACHE_TTL', '300')),
            category='Server',
            editable=True,
            requires_restart=False,
            min_value=0,
            max_value=3600,
            unit='seconds'
        )
        
        settings['AM_RATE_LIMIT_REQUESTS'] = ConfigSetting(
            key='AM_RATE_LIMIT_REQUESTS',
            display_name='Rate Limit Requests',
            description='Maximum requests per time window',
            type=ConfigType.INTEGER,
            default_value=100,
            current_value=int(os.getenv('AM_RATE_LIMIT_REQUESTS', '100')),
            category='Server',
            editable=True,
            requires_restart=False,
            min_value=10,
            max_value=1000,
            advanced=True
        )
        
        settings['AM_RATE_LIMIT_WINDOW'] = ConfigSetting(
            key='AM_RATE_LIMIT_WINDOW',
            display_name='Rate Limit Window',
            description='Time window for rate limiting',
            type=ConfigType.INTEGER,
            default_value=60,
            current_value=int(os.getenv('AM_RATE_LIMIT_WINDOW', '60')),
            category='Server',
            editable=True,
            requires_restart=False,
            min_value=1,
            max_value=600,
            unit='seconds',
            advanced=True
        )
        
        # LLM Generation Settings
        settings['AM_DEFAULT_TEMPERATURE'] = ConfigSetting(
            key='AM_DEFAULT_TEMPERATURE',
            display_name='Default Temperature',
            description='Default temperature for LLM generation',
            type=ConfigType.FLOAT,
            default_value=0.3,
            current_value=float(os.getenv('AM_DEFAULT_TEMPERATURE', '0.3')),
            category='Generation',
            editable=True,
            requires_restart=False,
            min_value=0.0,
            max_value=2.0
        )
        
        settings['AM_DEFAULT_MAX_TOKENS'] = ConfigSetting(
            key='AM_DEFAULT_MAX_TOKENS',
            display_name='Default Max Tokens',
            description='Default maximum tokens for LLM generation',
            type=ConfigType.INTEGER,
            default_value=100,
            current_value=int(os.getenv('AM_DEFAULT_MAX_TOKENS', '100')),
            category='Generation',
            editable=True,
            requires_restart=False,
            min_value=10,
            max_value=4096,
            unit='tokens'
        )
        
        settings['AM_DEFAULT_REPETITION_PENALTY'] = ConfigSetting(
            key='AM_DEFAULT_REPETITION_PENALTY',
            display_name='Default Repetition Penalty',
            description='Default repetition penalty for LLM generation',
            type=ConfigType.FLOAT,
            default_value=1.2,
            current_value=float(os.getenv('AM_DEFAULT_REPETITION_PENALTY', '1.2')),
            category='Generation',
            editable=True,
            requires_restart=False,
            min_value=0.0,
            max_value=2.0
        )
        
        settings['AM_DEFAULT_TOP_P'] = ConfigSetting(
            key='AM_DEFAULT_TOP_P',
            display_name='Default Top-P',
            description='Default top-p sampling parameter',
            type=ConfigType.FLOAT,
            default_value=0.9,
            current_value=float(os.getenv('AM_DEFAULT_TOP_P', '0.9')),
            category='Generation',
            editable=True,
            requires_restart=False,
            min_value=0.0,
            max_value=1.0,
            advanced=True
        )
        
        settings['AM_DEFAULT_TOP_K'] = ConfigSetting(
            key='AM_DEFAULT_TOP_K',
            display_name='Default Top-K',
            description='Default top-k sampling parameter',
            type=ConfigType.INTEGER,
            default_value=40,
            current_value=int(os.getenv('AM_DEFAULT_TOP_K', '40')),
            category='Generation',
            editable=True,
            requires_restart=False,
            min_value=0,
            max_value=200,
            advanced=True
        )
        
        # File Upload Settings
        settings['AM_MAX_UPLOAD_SIZE'] = ConfigSetting(
            key='AM_MAX_UPLOAD_SIZE',
            display_name='Max Upload Size',
            description='Maximum file upload size',
            type=ConfigType.INTEGER,
            default_value=52428800,  # 50MB
            current_value=int(os.getenv('AM_MAX_UPLOAD_SIZE', '52428800')),
            category='Upload',
            editable=True,
            requires_restart=True,
            min_value=1048576,  # 1MB
            max_value=524288000,  # 500MB
            unit='bytes'
        )
        
        settings['AM_UPLOAD_EXTENSIONS'] = ConfigSetting(
            key='AM_UPLOAD_EXTENSIONS',
            display_name='Allowed Upload Extensions',
            description='Comma-separated list of allowed file extensions',
            type=ConfigType.STRING,
            default_value='txt,pdf,docx,md,html,json,csv,xml,log,py,js,java,cpp,c,h',
            current_value=os.getenv('AM_UPLOAD_EXTENSIONS', 'txt,pdf,docx,md,html,json,csv,xml,log,py,js,java,cpp,c,h'),
            category='Upload',
            editable=True,
            requires_restart=False
        )
        
        # Model Settings
        settings['AM_MODEL_PATH'] = ConfigSetting(
            key='AM_MODEL_PATH',
            display_name='Model Path',
            description='Path to the GGUF model file',
            type=ConfigType.STRING,
            default_value='',
            current_value=os.getenv('AM_MODEL_PATH', os.getenv('LLM_MODEL_PATH', '')),
            category='Model',
            editable=True,
            requires_restart=True
        )
        
        # System Settings (read-only)
        settings['AM_DB_PATH'] = ConfigSetting(
            key='AM_DB_PATH',
            display_name='Database Path',
            description='Path to SQLite database',
            type=ConfigType.STRING,
            default_value='./amemory.sqlite3',
            current_value=os.getenv('AM_DB_PATH', './amemory.sqlite3'),
            category='System',
            editable=False,
            requires_restart=True
        )
        
        settings['AM_EMBED_MODEL'] = ConfigSetting(
            key='AM_EMBED_MODEL',
            display_name='Embedding Model',
            description='Model used for text embeddings',
            type=ConfigType.CHOICE,
            default_value='sentence-transformers/all-MiniLM-L6-v2',
            current_value=os.getenv('AM_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
            category='System',
            editable=True,
            requires_restart=True,
            choices=[
                'sentence-transformers/all-MiniLM-L6-v2',
                'sentence-transformers/all-mpnet-base-v2',
                'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
            ]
        )
        
        settings['AM_INDEX_PATH'] = ConfigSetting(
            key='AM_INDEX_PATH',
            display_name='FAISS Index Path',
            description='Path to FAISS vector index',
            type=ConfigType.STRING,
            default_value='./faiss.index',
            current_value=os.getenv('AM_INDEX_PATH', './faiss.index'),
            category='System',
            editable=False,
            requires_restart=True
        )
        
        settings['AM_LLM_BASE_URL'] = ConfigSetting(
            key='AM_LLM_BASE_URL',
            display_name='LLM Base URL',
            description='Base URL for LLM API',
            type=ConfigType.STRING,
            default_value='http://localhost:8000/v1',
            current_value=os.getenv('AM_LLM_BASE_URL', 'http://localhost:8000/v1'),
            category='System',
            editable=True,
            requires_restart=False
        )
        
        settings['AM_LLM_MODEL'] = ConfigSetting(
            key='AM_LLM_MODEL',
            display_name='LLM Model Name',
            description='Model name for LLM API',
            type=ConfigType.STRING,
            default_value='Qwen3-4b-instruct-2507',
            current_value=os.getenv('AM_LLM_MODEL', 'Qwen3-4b-instruct-2507'),
            category='System',
            editable=True,
            requires_restart=False
        )
        
        settings['AM_RESERVE_SYSTEM_TOKENS'] = ConfigSetting(
            key='AM_RESERVE_SYSTEM_TOKENS',
            display_name='Reserved System Tokens',
            description='Tokens reserved for system prompts',
            type=ConfigType.INTEGER,
            default_value=512,
            current_value=int(os.getenv('AM_RESERVE_SYSTEM_TOKENS', '512')),
            category='LLM',
            editable=True,
            requires_restart=False,
            min_value=128,
            max_value=2048,
            unit='tokens',
            advanced=True
        )
        
        settings['AM_MMR_LAMBDA'] = ConfigSetting(
            key='AM_MMR_LAMBDA',
            display_name='MMR Lambda',
            description='Lambda parameter for Maximal Marginal Relevance',
            type=ConfigType.FLOAT,
            default_value=0.5,
            current_value=float(os.getenv('AM_MMR_LAMBDA', '0.5')),
            category='Retrieval',
            editable=True,
            requires_restart=False,
            min_value=0.0,
            max_value=1.0,
            advanced=True
        )
        
        settings['AM_MULTI_PART_THRESHOLD'] = ConfigSetting(
            key='AM_MULTI_PART_THRESHOLD',
            display_name='Multi-Part Threshold',
            description='Minimum characters to trigger multi-part extraction',
            type=ConfigType.INTEGER,
            default_value=200,
            current_value=int(os.getenv('AM_MULTI_PART_THRESHOLD', '200')),
            category='Extraction',
            editable=True,
            requires_restart=False,
            min_value=50,
            max_value=1000,
            unit='characters',
            advanced=True
        )
        
        settings['AM_EMBED_DIM'] = ConfigSetting(
            key='AM_EMBED_DIM',
            display_name='Embedding Dimension',
            description='Dimension of embedding vectors',
            type=ConfigType.INTEGER,
            default_value=384,
            current_value=int(os.getenv('AM_EMBED_DIM', '384')),
            category='System',
            editable=False,
            requires_restart=True
        )
        
        settings['AM_ATTENTION_HEADS'] = ConfigSetting(
            key='AM_ATTENTION_HEADS',
            display_name='Attention Heads',
            description='Number of attention heads',
            type=ConfigType.INTEGER,
            default_value=8,
            current_value=int(os.getenv('AM_ATTENTION_HEADS', '8')),
            category='Retrieval',
            editable=True,
            requires_restart=False,
            min_value=1,
            max_value=16,
            advanced=True
        )
        
        settings['AM_CLUSTER_ENERGY_DECAY'] = ConfigSetting(
            key='AM_CLUSTER_ENERGY_DECAY',
            display_name='Cluster Energy Decay',
            description='Energy decay rate for liquid clusters',
            type=ConfigType.FLOAT,
            default_value=0.99,
            current_value=float(os.getenv('AM_CLUSTER_ENERGY_DECAY', '0.99')),
            category='Clustering',
            editable=True,
            requires_restart=False,
            min_value=0.9,
            max_value=0.999,
            advanced=True
        )
        
        settings['AM_EMBED_MOMENTUM'] = ConfigSetting(
            key='AM_EMBED_MOMENTUM',
            display_name='Embedding Momentum',
            description='Momentum rate for embedding updates',
            type=ConfigType.FLOAT,
            default_value=0.95,
            current_value=float(os.getenv('AM_EMBED_MOMENTUM', '0.95')),
            category='Learning',
            editable=True,
            requires_restart=False,
            min_value=0.8,
            max_value=0.99,
            advanced=True
        )
        
        settings['AM_DRIFT_BLEND'] = ConfigSetting(
            key='AM_DRIFT_BLEND',
            display_name='Drift Blend Ratio',
            description='How much drift affects embeddings',
            type=ConfigType.FLOAT,
            default_value=0.3,
            current_value=float(os.getenv('AM_DRIFT_BLEND', '0.3')),
            category='Learning',
            editable=True,
            requires_restart=False,
            min_value=0.0,
            max_value=0.5,
            advanced=True
        )
        
        return settings
    
    def _load_overrides(self):
        """Load saved overrides from database"""
        with self.connect() as con:
            rows = con.execute("SELECT key, value FROM config_overrides").fetchall()
            for row in rows:
                if row['key'] in self.settings:
                    setting = self.settings[row['key']]
                    # Parse value based on type
                    value = self._parse_value(row['value'], setting.type)
                    if value is not None:
                        setting.current_value = value
    
    def _parse_value(self, value_str: str, config_type: ConfigType) -> Any:
        """Parse string value based on config type"""
        try:
            if config_type == ConfigType.BOOLEAN:
                return value_str.lower() in ['true', '1', 'yes', 'on']
            elif config_type == ConfigType.INTEGER:
                return int(value_str)
            elif config_type == ConfigType.FLOAT:
                return float(value_str)
            else:
                return value_str
        except:
            return None
    
    def get_setting(self, key: str) -> Optional[ConfigSetting]:
        """Get a specific setting"""
        return self.settings.get(key)
    
    def get_value(self, key: str) -> Any:
        """Get current value of a setting"""
        setting = self.settings.get(key)
        return setting.current_value if setting else None
    
    def get_categories(self) -> List[str]:
        """Get all setting categories"""
        return sorted(list(set(s.category for s in self.settings.values())))
    
    def get_settings_by_category(self, category: str, include_advanced: bool = False) -> List[ConfigSetting]:
        """Get all settings in a category"""
        settings = [s for s in self.settings.values() if s.category == category]
        if not include_advanced:
            settings = [s for s in settings if not s.advanced]
        return sorted(settings, key=lambda s: s.display_name)
    
    def update_setting(self, key: str, value: Any, user: str = None, reason: str = None) -> tuple[bool, Optional[str]]:
        """Update a setting value"""
        setting = self.settings.get(key)
        if not setting:
            return False, f"Unknown setting: {key}"
        
        if not setting.editable:
            return False, f"Setting {key} is not editable"
        
        # Validate new value
        valid, error = setting.validate(value)
        if not valid:
            return False, error
        
        # Convert value to appropriate type
        if setting.type == ConfigType.BOOLEAN:
            value = bool(value) if isinstance(value, bool) else str(value).lower() in ['true', '1', 'yes', 'on']
        elif setting.type == ConfigType.INTEGER:
            value = int(value)
        elif setting.type == ConfigType.FLOAT:
            value = float(value)
        
        old_value = setting.current_value
        setting.current_value = value
        
        # Save to database
        now = datetime.utcnow().isoformat()
        with self.connect() as con:
            # Save override
            con.execute(
                """INSERT OR REPLACE INTO config_overrides (key, value, updated_at, updated_by)
                   VALUES (?, ?, ?, ?)""",
                (key, str(value), now, user)
            )
            
            # Save history
            con.execute(
                """INSERT INTO config_history (key, old_value, new_value, changed_at, changed_by, reason)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (key, str(old_value), str(value), now, user, reason)
            )
        
        # Update environment variable so running system uses new value
        os.environ[key] = str(value)
        
        return True, None
    
    def reset_setting(self, key: str, user: str = None) -> tuple[bool, Optional[str]]:
        """Reset a setting to its default value"""
        setting = self.settings.get(key)
        if not setting:
            return False, f"Unknown setting: {key}"
        
        return self.update_setting(key, setting.default_value, user, "Reset to default")
    
    def reset_all(self, user: str = None) -> Dict[str, tuple[bool, Optional[str]]]:
        """Reset all settings to defaults"""
        results = {}
        for key in self.settings:
            results[key] = self.reset_setting(key, user)
        return results
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration"""
        return {
            'exported_at': datetime.utcnow().isoformat(),
            'settings': {
                key: {
                    'value': setting.current_value,
                    'is_default': setting.current_value == setting.default_value
                }
                for key, setting in self.settings.items()
            }
        }
    
    def import_config(self, config_data: Dict[str, Any], user: str = None) -> Dict[str, tuple[bool, Optional[str]]]:
        """Import configuration from exported data"""
        results = {}
        if 'settings' not in config_data:
            return {'error': (False, 'Invalid config format')}
        
        for key, data in config_data['settings'].items():
            if key in self.settings and 'value' in data:
                results[key] = self.update_setting(key, data['value'], user, "Imported from config")
        
        return results
    
    def get_history(self, key: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get configuration change history"""
        with self.connect() as con:
            if key:
                rows = con.execute(
                    """SELECT * FROM config_history 
                       WHERE key = ? 
                       ORDER BY changed_at DESC 
                       LIMIT ?""",
                    (key, limit)
                ).fetchall()
            else:
                rows = con.execute(
                    """SELECT * FROM config_history 
                       ORDER BY changed_at DESC 
                       LIMIT ?""",
                    (limit,)
                ).fetchall()
        
        return [dict(row) for row in rows]
    
    def validate_weights(self) -> tuple[bool, Optional[str]]:
        """Validate that retrieval weights sum to 1.0"""
        if not self.get_value('AM_USE_ATTENTION'):
            # When attention is disabled, weights should sum to 1.0
            weight_keys = ['AM_W_SEMANTIC', 'AM_W_LEXICAL', 'AM_W_RECENCY', 
                          'AM_W_ACTOR', 'AM_W_SPATIAL', 'AM_W_USAGE']
            total = sum(self.get_value(key) for key in weight_keys)
            if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                return False, f"Retrieval weights must sum to 1.0 (current: {total:.3f})"
        return True, None
    
    def get_all_as_dict(self) -> Dict[str, Any]:
        """Get all settings as a dictionary for the Config class"""
        return {
            key: setting.current_value
            for key, setting in self.settings.items()
        }
