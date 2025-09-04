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
