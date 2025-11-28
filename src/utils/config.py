"""Configuration utilities for loading environment settings."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_data_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Extract and convert data paths from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of Path objects for data directories
    """
    data_config = config.get('data', {})

    return {
        'raw': Path(data_config.get('raw_path', 'data/raw')),
        'processed': Path(data_config.get('processed_path', 'data/processed')),
        'external': Path(data_config.get('external_path', 'data/external'))
    }
