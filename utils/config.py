"""Configuration loading utilities."""

import os
from pathlib import Path
from typing import Dict, Any
import yaml


def load_config(config_name: str = "params") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_name: Name of config file (without .yaml extension)
        
    Returns:
        Dictionary containing configuration
    """
    # Get project root
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_paths(create_if_missing: bool = True) -> Dict[str, Path]:
    """
    Get project paths and optionally create directories.
    
    Args:
        create_if_missing: Create directories if they don't exist
        
    Returns:
        Dictionary of paths
    """
    project_root = Path(__file__).parent.parent
    
    paths = {
        'root': project_root,
        'config': project_root / 'config',
        'data': project_root / 'data',
        'raw_data': project_root / 'data' / 'raw',
        'processed_data': project_root / 'data' / 'processed',
        'cache': project_root / 'data' / 'cache',
        'models': project_root / 'models',
        'checkpoints': project_root / 'models' / 'checkpoints',
        'results': project_root / 'results',
        'figures': project_root / 'results' / 'figures',
        'metrics': project_root / 'results' / 'metrics',
        'backtests': project_root / 'results' / 'backtests',
        'logs': project_root / 'logs',
        'notebooks': project_root / 'notebooks',
    }
    
    if create_if_missing:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    return paths


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "model.num_channels")
        default: Default value if key not found
        
    Returns:
        Configuration value
        
    Example:
        >>> config = {'model': {'num_channels': [32, 64]}}
        >>> get_config_value(config, 'model.num_channels')
        [32, 64]
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

