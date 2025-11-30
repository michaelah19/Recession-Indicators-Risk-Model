"""
Feature Engineering Module

This module contains feature engineering, target creation, and feature selection
for recession prediction.

Modules:
- engineer: Feature engineering (lags, rolling, diffs, ratios, technical indicators)
- targets: Target variable creation
- selector: Feature selection and multicollinearity checks
"""

from src.features.engineer import FeatureEngineer
from src.features.targets import create_recession_targets
from src.features.selector import FeatureSelector

__all__ = [
    'FeatureEngineer',
    'create_recession_targets',
    'FeatureSelector',
]
