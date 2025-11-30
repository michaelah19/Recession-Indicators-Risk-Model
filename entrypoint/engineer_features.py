"""
Feature Engineering Pipeline

Complete pipeline for creating recession prediction features:
1. Load quarterly aligned data
2. Engineer all features (lags, rolling, diffs, ratios, technical)
3. Create target variables
4. Select top 50 features
5. Save outputs

Outputs:
- data/processed/features_full.parquet (500+ features)
- data/processed/features_selected.parquet (top 50 features + targets)
- data/processed/feature_importance.csv (all features ranked)

Usage:
    python entrypoint/engineer_features.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.features.engineer import FeatureEngineer
from src.features.targets import (
    create_recession_targets,
    create_indicator_change_targets,
    add_targets_to_features,
    split_features_targets
)
from src.features.selector import FeatureSelector
from src.utils.config import load_config, get_data_paths
from src.utils.logger import setup_logger


def main():
    """Main feature engineering pipeline."""

    # Load configuration
    config_path = project_root / "config.yaml"
    config = load_config(config_path)
    data_paths = get_data_paths(config)

    # Setup logger
    logger = setup_logger(
        name="feature_engineering",
        log_file=project_root / "logs" / "feature_engineering.log",
        level=config['logging']['level']
    )

    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 80)

    try:
        # ==================================================================
        # STEP 1: Load Quarterly Aligned Data
        # ==================================================================
        logger.info("STEP 1: Loading quarterly aligned data...")
        logger.info("-" * 80)

        processed_dir = project_root / data_paths['processed']
        input_file = processed_dir / "quarterly_aligned.parquet"

        if not input_file.exists():
            raise FileNotFoundError(
                f"Quarterly aligned data not found: {input_file}\n"
                f"Run 'python entrypoint/preprocess.py' first!"
            )

        quarterly_df = pd.read_parquet(input_file)
        logger.info(f"Loaded data: {quarterly_df.shape}")
        logger.info(f"Date range: {quarterly_df.index.min().date()} to {quarterly_df.index.max().date()}")
        logger.info("")

        # ==================================================================
        # STEP 2: Engineer All Features
        # ==================================================================
        logger.info("STEP 2: Engineering features...")
        logger.info("-" * 80)

        engineer = FeatureEngineer(
            lags=[1, 2, 3, 4, 8],
            rolling_windows=[4, 8]
        )

        features_full = engineer.engineer_all_features(quarterly_df)

        logger.info(f"Features created: {features_full.shape}")
        logger.info("")

        # ==================================================================
        # STEP 3: Create Target Variables (Classification + Regression)
        # ==================================================================
        logger.info("STEP 3: Creating target variables...")
        logger.info("-" * 80)

        if 'recession' not in quarterly_df.columns:
            raise ValueError("Recession column not found in quarterly data!")

        # Add both classification and regression targets
        features_full = add_targets_to_features(
            features_df=features_full,
            recession_series=quarterly_df['recession'],
            quarterly_df=quarterly_df,
            include_regression_targets=True
        )

        logger.info(f"Features + Targets: {features_full.shape}")
        logger.info("")

        # Save full features
        features_full_path = processed_dir / "features_full.parquet"
        logger.info(f"Saving full features to {features_full_path}...")
        features_full.to_parquet(features_full_path, engine='pyarrow', compression='snappy')

        file_size_mb = features_full_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {features_full_path} ({file_size_mb:.2f} MB)")
        logger.info("")

        # ==================================================================
        # STEP 4: Feature Selection
        # ==================================================================
        logger.info("STEP 4: Selecting top features...")
        logger.info("-" * 80)

        # Split features and targets (remove rows with NaN targets)
        X, y = split_features_targets(features_full, target_col='recession_within_2q')

        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info("")

        # Select top 50 features
        selector = FeatureSelector(
            n_estimators=100,
            max_depth=10,
            random_state=config['training']['random_seed'],
            vif_threshold=10.0
        )

        selected_features, importance_df = selector.select_features(
            X=X,
            y=y,
            n_features=50,
            check_vif=True
        )

        logger.info(f"Selected features: {len(selected_features)}")
        logger.info("")

        # ==================================================================
        # STEP 5: Save Outputs
        # ==================================================================
        logger.info("STEP 5: Saving outputs...")
        logger.info("-" * 80)

        # Save feature importance
        importance_path = processed_dir / "feature_importance.csv"
        logger.info(f"Saving feature importance to {importance_path}...")
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Saved: {len(importance_df)} features ranked by importance")
        logger.info("")

        # Create selected features DataFrame (with all targets - classification + regression)
        classification_target_cols = [
            'recession_current',
            'recession_next_1q',
            'recession_next_2q',
            'recession_within_2q'
        ]

        regression_target_cols_requested = [
            'unemployment_rate_change',
            'unemployment_claims_change',
            'sp500_drawdown',
            'nasdaq_drawdown',
            'gdp_decline'
        ]

        # Only include regression targets that exist in features_full
        regression_target_cols = [col for col in regression_target_cols_requested if col in features_full.columns]

        if len(regression_target_cols) < len(regression_target_cols_requested):
            missing_targets = set(regression_target_cols_requested) - set(regression_target_cols)
            logger.warning(f"Some regression targets were not created: {missing_targets}")

        all_target_cols = classification_target_cols + regression_target_cols

        # Get selected features + all targets from full features
        selected_cols = selected_features + all_target_cols
        features_selected = features_full[selected_cols].copy()

        # Remove rows with NaN classification targets (keep regression targets even if NaN)
        valid_idx = features_selected['recession_within_2q'].notna()
        features_selected = features_selected[valid_idx]

        # Save selected features
        features_selected_path = processed_dir / "features_selected.parquet"
        logger.info(f"Saving selected features to {features_selected_path}...")
        features_selected.to_parquet(features_selected_path, engine='pyarrow', compression='snappy')

        file_size_mb = features_selected_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {features_selected_path} ({file_size_mb:.2f} MB)")
        logger.info(f"  Shape: {features_selected.shape}")
        logger.info(f"  Features: {len(selected_features)}")
        logger.info(f"  Classification targets: {len(classification_target_cols)}")
        logger.info(f"  Regression targets: {len(regression_target_cols)}")
        logger.info(f"  Total targets: {len(all_target_cols)}")
        logger.info("")

        # ==================================================================
        # SUCCESS SUMMARY
        # ==================================================================
        logger.info("=" * 80)
        logger.info("FEATURE ENGINEERING COMPLETE!")
        logger.info("=" * 80)

        logger.info("\nOutputs created:")
        logger.info(f"  1. {features_full_path}")
        logger.info(f"     - Shape: {features_full.shape}")
        logger.info(f"     - All engineered features + all targets")
        logger.info("")
        logger.info(f"  2. {features_selected_path}")
        logger.info(f"     - Shape: {features_selected.shape}")
        logger.info(f"     - Top {len(selected_features)} features + {len(all_target_cols)} targets")
        logger.info(f"     - {len(classification_target_cols)} classification + {len(regression_target_cols)} regression targets")
        logger.info(f"     - Ready for model training")
        logger.info("")
        logger.info(f"  3. {importance_path}")
        logger.info(f"     - Feature importance rankings")
        logger.info(f"     - {len(importance_df)} features")
        logger.info("")

        logger.info("Top 10 most important features:")
        for i, row in importance_df.head(10).iterrows():
            logger.info(f"  {i+1:2d}. {row['feature']:50s} {row['importance']:.4f}")
        logger.info("")

        logger.info("Next step: Train models")
        logger.info("  python entrypoint/train.py")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("FEATURE ENGINEERING FAILED!")
        logger.error(f"{'=' * 80}")
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
