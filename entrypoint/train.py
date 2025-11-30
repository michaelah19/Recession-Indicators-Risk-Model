"""
Training Pipeline for Hybrid Recession Prediction Model

Main entrypoint for training the two-stage hybrid model:
1. Stage 1: Recession probability classifier
2. Stage 2: Economic indicator impact regressors

Usage:
    python entrypoint/train.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.config import load_config, get_data_paths
from src.utils.logger import setup_logger

from src.models.recession_classifier import (
    XGBoostRecessionClassifier,
    RandomForestRecessionClassifier,
    LogisticRecessionClassifier
)
from src.models.indicator_regressors import (
    LaborIndicatorRegressor,
    MarketsIndicatorRegressor,
    GDPIndicatorRegressor
)
from src.models.hybrid_predictor import HybridRecessionPredictor
from src.models.evaluator import RecessionModelEvaluator
from src.models.model_persistence import ModelPersistence


def load_training_data(config: dict, logger) -> tuple:
    """
    Load and prepare training data.

    Returns:
        (X, y_classification, y_regression, feature_names, target_names)
    """
    logger.info("=" * 80)
    logger.info("LOADING TRAINING DATA")
    logger.info("=" * 80)

    # Load features
    data_paths = get_data_paths(config)
    features_path = Path(data_paths['processed']) / "features_selected.parquet"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}\n"
            "Please run feature engineering first: python entrypoint/engineer_features.py"
        )

    df = pd.read_parquet(features_path)
    logger.info(f"✓ Loaded features: {df.shape}")

    # Filter to date range where all regression targets are available
    # (Market data starts ~1979, labor/GDP data starts ~1947)
    regression_targets_temp = [
        'unemployment_rate_change', 'unemployment_claims_change',
        'sp500_drawdown', 'nasdaq_drawdown', 'gdp_decline'
    ]

    # Find first date where all regression targets have data
    valid_mask = df[regression_targets_temp].notna().all(axis=1)
    if valid_mask.sum() < len(df):
        first_valid_date = df[valid_mask].index.min()
        last_valid_date = df[valid_mask].index.max()
        logger.info(f"\nFiltering to date range with complete regression targets:")
        logger.info(f"  Original: {df.index.min()} to {df.index.max()} ({len(df)} samples)")
        logger.info(f"  Filtered: {first_valid_date} to {last_valid_date} ({valid_mask.sum()} samples)")

        df = df[valid_mask].copy()
        logger.info(f"✓ Filtered dataset: {df.shape}")
    else:
        logger.info(f"✓ All samples have complete regression targets")

    # Identify target columns
    classification_targets = [
        'recession_current', 'recession_next_1q',
        'recession_next_2q', 'recession_within_2q'
    ]

    regression_targets = [
        'unemployment_rate_change', 'unemployment_claims_change',
        'sp500_drawdown', 'nasdaq_drawdown', 'gdp_decline'
    ]

    # Primary targets for training
    primary_class_target = 'recession_within_2q'

    # Identify feature columns
    all_targets = classification_targets + regression_targets
    feature_cols = [col for col in df.columns if col not in all_targets]

    logger.info(f"✓ Features: {len(feature_cols)}")
    logger.info(f"✓ Classification target: {primary_class_target}")
    logger.info(f"✓ Regression targets: {len(regression_targets)}")

    # Split features and targets
    X = df[feature_cols].copy()
    y_classification = df[primary_class_target].copy()
    y_regression = df[regression_targets].copy()

    # Log class distribution
    class_counts = y_classification.value_counts()
    logger.info(f"\nClass distribution:")
    logger.info(f"  Non-recession (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    logger.info(f"  Recession (1):     {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")

    # Log regression target NaN rates
    logger.info(f"\nRegression target NaN rates:")
    for target in regression_targets:
        nan_count = y_regression[target].isna().sum()
        logger.info(f"  {target}: {nan_count}/{len(df)} ({nan_count/len(df)*100:.1f}%)")

    return X, y_classification, y_regression, feature_cols, regression_targets


def create_temporal_splits(
    X: pd.DataFrame,
    y_classification: pd.Series,
    y_regression: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    logger=None
) -> dict:
    """
    Create temporal train/validation/test splits.

    Args:
        X: Features
        y_classification: Classification targets
        y_regression: Regression targets
        train_ratio: Training set ratio (default: 0.70)
        val_ratio: Validation set ratio (default: 0.15)
        logger: Logger instance

    Returns:
        Dictionary with train/val/test splits
    """
    if logger:
        logger.info("\n" + "=" * 80)
        logger.info("CREATING TEMPORAL SPLITS")
        logger.info("=" * 80)

    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    # Temporal split (no shuffling)
    train_end = train_size
    val_end = train_size + val_size

    X_train = X.iloc[:train_end].copy()
    X_val = X.iloc[train_end:val_end].copy()
    X_test = X.iloc[val_end:].copy()

    y_class_train = y_classification.iloc[:train_end].copy()
    y_class_val = y_classification.iloc[train_end:val_end].copy()
    y_class_test = y_classification.iloc[val_end:].copy()

    y_reg_train = y_regression.iloc[:train_end].copy()
    y_reg_val = y_regression.iloc[train_end:val_end].copy()
    y_reg_test = y_regression.iloc[val_end:].copy()

    if logger:
        logger.info(f"✓ Train set: {len(X_train)} samples ({train_ratio*100:.0f}%)")
        logger.info(f"    Date range: {X_train.index[0]} to {X_train.index[-1]}")
        logger.info(f"    Class distribution: {y_class_train.value_counts().to_dict()}")

        logger.info(f"✓ Validation set: {len(X_val)} samples ({val_ratio*100:.0f}%)")
        logger.info(f"    Date range: {X_val.index[0]} to {X_val.index[-1]}")
        logger.info(f"    Class distribution: {y_class_val.value_counts().to_dict()}")

        logger.info(f"✓ Test set: {len(X_test)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
        logger.info(f"    Date range: {X_test.index[0]} to {X_test.index[-1]}")
        logger.info(f"    Class distribution: {y_class_test.value_counts().to_dict()}")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_class_train': y_class_train, 'y_class_val': y_class_val, 'y_class_test': y_class_test,
        'y_reg_train': y_reg_train, 'y_reg_val': y_reg_val, 'y_reg_test': y_reg_test
    }


def initialize_model(config: dict, logger) -> HybridRecessionPredictor:
    """
    Initialize hybrid model with components.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Initialized HybridRecessionPredictor
    """
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING HYBRID MODEL")
    logger.info("=" * 80)

    # Get model config
    model_config = config.get('model', {})
    stage1_config = model_config.get('stage1', {})
    stage2_config = model_config.get('stage2', {})

    # Stage 1: Recession classifier
    classifier_type = stage1_config.get('classifier_type', 'xgboost')
    classifier_params = stage1_config.get('hyperparameters', {})

    logger.info(f"\nStage 1: Recession Classifier")
    logger.info(f"  Type: {classifier_type}")

    if classifier_type == 'xgboost':
        stage1 = XGBoostRecessionClassifier(**classifier_params)
    elif classifier_type == 'random_forest':
        stage1 = RandomForestRecessionClassifier(**classifier_params)
    elif classifier_type == 'logistic':
        stage1 = LogisticRecessionClassifier(**classifier_params)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    # Stage 2: Indicator regressors
    regressor_type = stage2_config.get('regressor_type', 'xgboost')
    regressor_params = stage2_config.get('hyperparameters', {})

    logger.info(f"\nStage 2: Indicator Regressors")
    logger.info(f"  Type: {regressor_type}")

    stage2_labor = LaborIndicatorRegressor(model_type=regressor_type, **regressor_params)
    stage2_markets = MarketsIndicatorRegressor(model_type=regressor_type, **regressor_params)
    stage2_gdp = GDPIndicatorRegressor(model_type=regressor_type, **regressor_params)

    # Threshold
    threshold = model_config.get('threshold', 0.5)

    # Create hybrid predictor
    hybrid = HybridRecessionPredictor(
        recession_classifier=stage1,
        labor_regressor=stage2_labor,
        markets_regressor=stage2_markets,
        gdp_regressor=stage2_gdp,
        threshold=threshold
    )

    logger.info(f"\n✓ Hybrid predictor initialized (threshold={threshold})")

    return hybrid


def train_model(hybrid, splits: dict, logger):
    """
    Train hybrid model.

    Args:
        hybrid: HybridRecessionPredictor instance
        splits: Dictionary with train/val/test data
        logger: Logger instance

    Returns:
        Trained model
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING MODEL")
    logger.info("=" * 80)

    # Train on training set
    hybrid.fit(
        X=splits['X_train'],
        y_classification=splits['y_class_train'],
        y_regression=splits['y_reg_train']
    )

    logger.info("\n✓ Model training complete")

    return hybrid


def evaluate_model(hybrid, splits: dict, config: dict, logger) -> dict:
    """
    Evaluate model on all splits.

    Args:
        hybrid: Trained HybridRecessionPredictor
        splits: Dictionary with train/val/test data
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Dictionary with metrics for all splits
    """
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING MODEL")
    logger.info("=" * 80)

    # Get threshold
    threshold = config.get('model', {}).get('threshold', 0.5)
    evaluator = RecessionModelEvaluator(threshold=threshold)

    all_metrics = {}

    # Evaluate on each split
    for split_name in ['train', 'val', 'test']:
        logger.info(f"\n{split_name.upper()} SET EVALUATION")
        logger.info("-" * 80)

        # Get data
        X = splits[f'X_{split_name}']
        y_class = splits[f'y_class_{split_name}']
        y_reg = splits[f'y_reg_{split_name}']

        # Predict
        predictions = hybrid.predict(X)

        # Evaluate
        metrics = evaluator.evaluate_hybrid(
            y_class_true=y_class,
            y_reg_true=y_reg,
            predictions=predictions
        )

        all_metrics[split_name] = metrics

        # Print summary
        summary = evaluator.format_metrics_summary(metrics)
        print(summary)

    return all_metrics


def save_artifacts(hybrid, metrics: dict, config: dict, logger) -> Path:
    """
    Save trained model and metrics.

    Args:
        hybrid: Trained model
        metrics: Evaluation metrics
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Path to saved artifacts
    """
    logger.info("\n" + "=" * 80)
    logger.info("SAVING MODEL ARTIFACTS")
    logger.info("=" * 80)

    # Get save directory
    save_dir = Path(config.get('model', {}).get('save_dir', 'models'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_dir / f"run_{timestamp}"

    # Prepare training info
    training_info = {
        'timestamp': timestamp,
        'model_config': config.get('model', {}),
        'training_config': config.get('training', {}),
    }

    # Save using ModelPersistence
    ModelPersistence.save_training_artifacts(
        save_dir=run_dir,
        model=hybrid,
        metrics=metrics,
        training_info=training_info
    )

    logger.info(f"\n✓ Artifacts saved to: {run_dir}")

    return run_dir


def main():
    """Run training pipeline."""
    # Load config
    config = load_config("config.yaml")

    # Setup logger
    logger = setup_logger(
        name="train",
        log_file=Path(config['logging']['dir']) / "train.log",
        level=config['logging']['level']
    )

    logger.info("\n" + "=" * 80)
    logger.info("HYBRID RECESSION PREDICTION MODEL - TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. Load data
        X, y_class, y_reg, feature_names, target_names = load_training_data(config, logger)

        # 2. Create splits
        train_ratio = config.get('training', {}).get('train_ratio', 0.70)
        val_ratio = config.get('training', {}).get('validation_ratio', 0.15)
        splits = create_temporal_splits(X, y_class, y_reg, train_ratio, val_ratio, logger)

        # 3. Initialize model
        hybrid = initialize_model(config, logger)

        # 4. Train model
        hybrid = train_model(hybrid, splits, logger)

        # 5. Evaluate model
        metrics = evaluate_model(hybrid, splits, config, logger)

        # 6. Save artifacts
        save_dir = save_artifacts(hybrid, metrics, config, logger)

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {save_dir}")
        logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return 0

    except Exception as e:
        logger.error(f"\n✗ TRAINING FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
