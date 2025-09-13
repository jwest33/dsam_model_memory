"""
Settings persistence manager for JAM.
Stores user preferences and custom weights in a JSON file.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class SettingsManager:
    """Manages persistent settings for the analyzer application."""
    
    # Default weights that never change
    DEFAULT_WEIGHTS = {
        'semantic': 0.68,
        'recency': 0.02,
        'actor': 0.10,
        'temporal': 0.10,
        'spatial': 0.05,
        'usage': 0.05
    }
    
    # Default settings
    DEFAULT_SETTINGS = {
        'weights': DEFAULT_WEIGHTS.copy(),
        'analyzer': {
            'top_k': 100,
            'auto_normalize_weights': True,
            'show_decomposition': True,
            'highlight_high_scores': True,
            'high_score_threshold': 0.8,
            'medium_score_threshold': 0.5
        },
        'browser': {
            'per_page': 50,
            'sort_by': 'when_ts',
            'sort_order': 'desc',
            'show_entities': True,
            'date_format': 'local'  # 'local' or 'iso'
        },
        'analytics': {
            'default_entity_limit': 1000,
            'default_temporal_days': 30,
            'entity_co_occurrence_limit': 100,
            'show_network_labels': True
        },
        'ui': {
            'theme': 'synthwave',  # for future theme support
            'compact_mode': False,
            'show_stats_in_header': True,
            'enable_animations': True
        }
    }
    
    def __init__(self, settings_path: Optional[str] = None):
        """Initialize the settings manager.
        
        Args:
            settings_path: Path to the settings file. Defaults to ./analyzer_settings.json
        """
        if settings_path is None:
            settings_path = os.path.join(os.path.dirname(__file__), '..', 'analyzer_settings.json')
        
        self.settings_path = Path(settings_path)
        self.settings = self._load_settings()
        
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file or create with defaults if not exists."""
        if self.settings_path.exists():
            try:
                with open(self.settings_path, 'r') as f:
                    loaded = json.load(f)
                    
                # Merge with defaults to handle new settings in updates
                settings = self._deep_merge(self.DEFAULT_SETTINGS.copy(), loaded)
                
                # Validate weights sum to 1.0
                if 'weights' in settings:
                    total = sum(settings['weights'].values())
                    if abs(total - 1.0) > 0.001:
                        # Auto-normalize if not equal to 1
                        for key in settings['weights']:
                            settings['weights'][key] /= total
                            
                return settings
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading settings, using defaults: {e}")
                return self.DEFAULT_SETTINGS.copy()
        else:
            # Create new settings file with defaults
            settings = self.DEFAULT_SETTINGS.copy()
            self._save_settings(settings)
            return settings
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries, with update values taking precedence."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _save_settings(self, settings: Optional[Dict] = None) -> bool:
        """Save settings to file.
        
        Args:
            settings: Settings to save. Uses current settings if None.
            
        Returns:
            True if successful, False otherwise.
        """
        if settings is None:
            settings = self.settings
            
        try:
            # Create directory if it doesn't exist
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            settings['_metadata'] = {
                'version': '1.0',
                'last_modified': datetime.now().isoformat()
            }
            
            with open(self.settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by dot-notation key.
        
        Args:
            key: Dot-separated key path (e.g., 'analyzer.top_k')
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        keys = key.split('.')
        value = self.settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any, save: bool = True) -> bool:
        """Set a setting value by dot-notation key.
        
        Args:
            key: Dot-separated key path (e.g., 'analyzer.top_k')
            value: Value to set
            save: Whether to save immediately
            
        Returns:
            True if successful
        """
        keys = key.split('.')
        target = self.settings
        
        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
            
        # Set the value
        target[keys[-1]] = value
        
        # Save if requested
        if save:
            return self._save_settings()
            
        return True
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weight configuration."""
        return self.settings.get('weights', self.DEFAULT_WEIGHTS.copy())
    
    def set_weights(self, weights: Dict[str, float], normalize: bool = True) -> Dict[str, float]:
        """Set weight configuration.
        
        Args:
            weights: New weights dictionary
            normalize: Whether to normalize to sum to 1.0
            
        Returns:
            Normalized weights
        """
        # Ensure all weight keys are present
        for key in self.DEFAULT_WEIGHTS:
            if key not in weights:
                weights[key] = self.DEFAULT_WEIGHTS[key]
        
        # Normalize if requested
        if normalize:
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                weights = {k: v/total for k, v in weights.items()}
        
        self.settings['weights'] = weights
        self._save_settings()
        
        return weights
    
    def reset_weights(self) -> Dict[str, float]:
        """Reset weights to defaults.
        
        Returns:
            Default weights
        """
        self.settings['weights'] = self.DEFAULT_WEIGHTS.copy()
        self._save_settings()
        return self.settings['weights']
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings.
        
        Returns:
            Complete settings dictionary
        """
        return self.settings.copy()
    
    def update_settings(self, updates: Dict[str, Any], save: bool = True) -> bool:
        """Update multiple settings at once.
        
        Args:
            updates: Dictionary of updates to apply
            save: Whether to save immediately
            
        Returns:
            True if successful
        """
        self.settings = self._deep_merge(self.settings, updates)
        
        if save:
            return self._save_settings()
            
        return True
    
    def reset_all(self) -> Dict[str, Any]:
        """Reset all settings to defaults.
        
        Returns:
            Default settings
        """
        self.settings = self.DEFAULT_SETTINGS.copy()
        self._save_settings()
        return self.settings
    
    def reset_section(self, section: str) -> Dict[str, Any]:
        """Reset a specific section to defaults.
        
        Args:
            section: Section name (e.g., 'analyzer', 'browser', 'ui')
            
        Returns:
            Updated settings for that section
        """
        if section in self.DEFAULT_SETTINGS:
            self.settings[section] = self.DEFAULT_SETTINGS[section].copy()
            self._save_settings()
            return self.settings[section]
        
        return {}
    
    def export_settings(self, path: str) -> bool:
        """Export settings to a file.
        
        Args:
            path: Path to export to
            
        Returns:
            True if successful
        """
        try:
            with open(path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting settings: {e}")
            return False
    
    def import_settings(self, path: str) -> bool:
        """Import settings from a file.
        
        Args:
            path: Path to import from
            
        Returns:
            True if successful
        """
        try:
            with open(path, 'r') as f:
                imported = json.load(f)
                
            # Validate and merge with defaults
            self.settings = self._deep_merge(self.DEFAULT_SETTINGS.copy(), imported)
            
            # Validate weights
            if 'weights' in self.settings:
                total = sum(self.settings['weights'].values())
                if abs(total - 1.0) > 0.001:
                    for key in self.settings['weights']:
                        self.settings['weights'][key] /= total
            
            return self._save_settings()
            
        except Exception as e:
            print(f"Error importing settings: {e}")
            return False
