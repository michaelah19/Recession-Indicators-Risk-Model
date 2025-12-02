"""
Model Comparison Script

Trains hybrid models with different Stage 1 classifiers and compares performance.

Usage:
    python entrypoint/compare_models.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
import json

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


def load_data(config, logger):
    """Load and prepare data."""
    logger.info("Loading data...")

    data_paths = get_data_paths(config)
    features_path = Path(data_paths['processed']) / "features_selected.parquet"

    df = pd.read_parquet(features_path)
    logger.info(f"✓ Loaded features: {df.shape}")

    # Filter to complete regression targets
    regression_targets = [
        'unemployment_rate_change', 'unemployment_claims_change',
        'sp500_drawdown', 'nasdaq_drawdown', 'gdp_decline'
    ]
    valid_mask = df[regression_targets].notna().all(axis=1)
    df = df[valid_mask].copy()
    logger.info(f"✓ Filtered to {len(df)} samples with complete targets")

    # Split features and targets
    classification_targets = [
        'recession_current', 'recession_next_1q',
        'recession_next_2q', 'recession_within_2q'
    ]
    all_targets = classification_targets + regression_targets
    feature_cols = [col for col in df.columns if col not in all_targets]

    X = df[feature_cols].copy()
    y_class = df['recession_within_2q'].copy()
    y_reg = df[regression_targets].copy()

    return X, y_class, y_reg


def create_splits(X, y_class, y_reg, train_ratio=0.70, val_ratio=0.15):
    """Create temporal splits."""
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    train_end = train_size
    val_end = train_size + val_size

    return {
        'X_train': X.iloc[:train_end].copy(),
        'X_val': X.iloc[train_end:val_end].copy(),
        'X_test': X.iloc[val_end:].copy(),
        'y_class_train': y_class.iloc[:train_end].copy(),
        'y_class_val': y_class.iloc[train_end:val_end].copy(),
        'y_class_test': y_class.iloc[val_end:].copy(),
        'y_reg_train': y_reg.iloc[:train_end].copy(),
        'y_reg_val': y_reg.iloc[train_end:val_end].copy(),
        'y_reg_test': y_reg.iloc[val_end:].copy()
    }


def train_model(classifier_name, classifier, splits, config, logger):
    """Train a hybrid model with given Stage 1 classifier."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING: {classifier_name}")
    logger.info(f"{'='*80}")

    # Initialize Stage 2 regressors (same for all models)
    stage2_config = config.get('model', {}).get('stage2', {})
    regressor_type = stage2_config.get('regressor_type', 'xgboost')
    regressor_params = stage2_config.get('hyperparameters', {})

    stage2_labor = LaborIndicatorRegressor(model_type=regressor_type, **regressor_params)
    stage2_markets = MarketsIndicatorRegressor(model_type=regressor_type, **regressor_params)
    stage2_gdp = GDPIndicatorRegressor(model_type=regressor_type, **regressor_params)

    # Create hybrid predictor
    threshold = config.get('model', {}).get('threshold', 0.5)
    hybrid = HybridRecessionPredictor(
        recession_classifier=classifier,
        labor_regressor=stage2_labor,
        markets_regressor=stage2_markets,
        gdp_regressor=stage2_gdp,
        threshold=threshold
    )

    # Train
    logger.info(f"Training {classifier_name} hybrid model...")
    hybrid.fit(
        X=splits['X_train'],
        y_classification=splits['y_class_train'],
        y_regression=splits['y_reg_train']
    )

    # Evaluate on all splits
    evaluator = RecessionModelEvaluator(threshold=threshold)
    results = {}

    for split_name in ['train', 'val', 'test']:
        logger.info(f"\nEvaluating on {split_name} set...")

        X = splits[f'X_{split_name}']
        y_class = splits[f'y_class_{split_name}']
        y_reg = splits[f'y_reg_{split_name}']

        predictions = hybrid.predict(X)
        metrics = evaluator.evaluate_hybrid(
            y_class_true=y_class,
            y_reg_true=y_reg,
            predictions=predictions
        )

        results[split_name] = metrics

    logger.info(f"\n✓ {classifier_name} training complete")

    return hybrid, results


def compare_models(all_results, logger):
    """Compare all models and select best."""
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)

    comparison = []

    for model_name, results in all_results.items():
        # Extract key metrics
        test_metrics = results['test']['stage1_classification']

        comparison.append({
            'model': model_name,
            'test_roc_auc': test_metrics.get('roc_auc'),
            'test_pr_auc': test_metrics.get('pr_auc'),
            'test_f1': test_metrics['f1_score'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_specificity': test_metrics['specificity'],
            'test_accuracy': test_metrics['accuracy'],
            'train_roc_auc': results['train']['stage1_classification']['roc_auc'],
        })

    comparison_df = pd.DataFrame(comparison)

    # Sort by test ROC-AUC (primary metric)
    comparison_df = comparison_df.sort_values('test_roc_auc', ascending=False)

    print("\n" + "="*80)
    print("STAGE 1 CLASSIFICATION COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print()

    # Select best model
    best_model = comparison_df.iloc[0]['model']
    logger.info(f"\n✓ BEST MODEL: {best_model}")
    logger.info(f"  Test ROC-AUC: {comparison_df.iloc[0]['test_roc_auc']:.4f}")
    logger.info(f"  Test Recall: {comparison_df.iloc[0]['test_recall']:.4f}")
    logger.info(f"  Test Specificity: {comparison_df.iloc[0]['test_specificity']:.4f}")

    return comparison_df, best_model


def main():
    """Run model comparison."""
    # Load config
    config = load_config("config.yaml")

    # Setup logger
    logger = setup_logger(
        name="compare_models",
        log_file=Path(config['logging']['dir']) / "compare_models.log",
        level=config['logging']['level']
    )

    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON PIPELINE")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Load data
        X, y_class, y_reg = load_data(config, logger)

        # Create splits
        train_ratio = config.get('training', {}).get('train_ratio', 0.70)
        val_ratio = config.get('training', {}).get('validation_ratio', 0.15)
        splits = create_splits(X, y_class, y_reg, train_ratio, val_ratio)

        logger.info(f"\n✓ Train: {len(splits['X_train'])} samples")
        logger.info(f"✓ Val: {len(splits['X_val'])} samples")
        logger.info(f"✓ Test: {len(splits['X_test'])} samples")

        # Define models to compare
        stage1_config = config.get('model', {}).get('stage1', {})

        models_to_test = {
            'XGBoost': XGBoostRecessionClassifier(
                **stage1_config.get('hyperparameters', {})
            ),
            'Random Forest': RandomForestRecessionClassifier(),
            'Logistic Regression': LogisticRecessionClassifier()
        }

        # Train all models
        all_results = {}
        all_models = {}

        for model_name, classifier in models_to_test.items():
            hybrid, results = train_model(model_name, classifier, splits, config, logger)
            all_results[model_name] = results
            all_models[model_name] = hybrid

        # Compare models
        comparison_df, best_model = compare_models(all_results, logger)

        # Save results
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)

        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
        logger.info(f"\n✓ Comparison saved to: {output_dir / 'model_comparison.csv'}")

        # Save detailed results as JSON
        results_path = output_dir / "model_comparison_detailed.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {}
            for model_name, results in all_results.items():
                serializable_results[model_name] = {
                    split: {
                        'stage1_classification': {
                            k: float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
                            for k, v in metrics['stage1_classification'].items()
                            if k != 'confusion_matrix'
                        }
                    }
                    for split, metrics in results.items()
                }

            json.dump(serializable_results, f, indent=2)
        logger.info(f"✓ Detailed results saved to: {results_path}")

        # Save best model
        from src.models.model_persistence import ModelPersistence
        save_dir = Path(config.get('model', {}).get('save_dir', 'models'))
        best_model_dir = save_dir / "best_model"

        ModelPersistence.save_training_artifacts(
            save_dir=best_model_dir,
            model=all_models[best_model],
            metrics=all_results[best_model],
            training_info={
                'timestamp': datetime.now().isoformat(),
                'model_name': best_model,
                'comparison': comparison_df.to_dict('records')
            }
        )
        logger.info(f"✓ Best model saved to: {best_model_dir}")

        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON COMPLETE")
        logger.info("="*80)
        logger.info(f"Best model: {best_model}")
        logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return 0

    except Exception as e:
        logger.error(f"\n✗ COMPARISON FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
