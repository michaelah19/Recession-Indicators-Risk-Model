# Configuration Files

This directory contains configuration files for different experiments, models, and hyperparameters.

## Structure

Place experiment-specific configurations here as YAML files:

- `model_config.yaml` - Model architecture and hyperparameters
- `preprocessing_config.yaml` - Data preprocessing parameters
- `feature_config.yaml` - Feature engineering specifications

## Usage

Load configurations in your code using:

```python
from src.utils.config import load_config
config = load_config("config/model_config.yaml")
```
