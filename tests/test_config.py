"""Tests for configuration utilities."""

import pytest
from pathlib import Path
from src.utils.config import load_config, get_data_paths


def test_load_config():
    """Test loading configuration file."""
    config = load_config("config.yaml")

    assert config is not None
    assert 'data' in config


def test_get_data_paths():
    """Test extracting data paths from configuration."""
    config = load_config("config.yaml")
    paths = get_data_paths(config)

    assert 'raw' in paths
    assert 'processed' in paths
    assert 'external' in paths
    assert isinstance(paths['raw'], Path)
