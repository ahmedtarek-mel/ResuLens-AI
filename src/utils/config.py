"""
Configuration management for the Resume/Job Matching System
"""

import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Singleton configuration manager"""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        else:
            # Default configuration
            self._config = self._get_defaults()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "models": {
                "embedding_model": "all-MiniLM-L6-v2",
                "spacy_model": "en_core_web_sm"
            },
            "matching": {
                "semantic_weight": 0.40,
                "skill_weight": 0.35,
                "keyword_weight": 0.25
            },
            "thresholds": {
                "excellent_match": 85,
                "good_match": 70,
                "fair_match": 55,
                "poor_match": 40
            },
            "file_processing": {
                "max_file_size_mb": 10,
                "supported_formats": [".pdf", ".docx", ".doc", ".txt"]
            },
            "ui": {
                "theme": "dark",
                "page_title": "AI Resume Matcher",
                "page_icon": "🎯"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('models.embedding_model')
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Return entire configuration"""
        return self._config.copy()
    
    @property
    def embedding_model(self) -> str:
        return self.get('models.embedding_model', 'all-MiniLM-L6-v2')
    
    @property
    def spacy_model(self) -> str:
        return self.get('models.spacy_model', 'en_core_web_sm')
    
    @property
    def matching_weights(self) -> Dict[str, float]:
        return self.get('matching', {
            'semantic_weight': 0.40,
            'skill_weight': 0.35,
            'keyword_weight': 0.25
        })
    
    @property
    def thresholds(self) -> Dict[str, int]:
        return self.get('thresholds', {
            'excellent_match': 85,
            'good_match': 70,
            'fair_match': 55,
            'poor_match': 40
        })
