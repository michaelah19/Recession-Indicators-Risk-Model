"""
Test Set Prediction Visualization Script

Generates plots comparing predicted vs actual values for:
- Recession predictions (probability and classification)
- Economic indicator forecasts (GDP, unemployment, SP500, etc.)

Usage:
    python entrypoint/visualize_predictions.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.models.model_persistence import load_model
from src.visualization.static_plots import (
    plot_full_timeline_with_predictions,
    plot_indicator_predictions_vs_actual,
    plot_all_indicators_comparison
)


def main():
    """Generate prediction comparison visualizations."""
    # Load config
    config = load_config("config.yaml")

    # Setup logger
    logger = setup_logger(
        name="visualize_predictions",
        log_file=Path(config['logging']['dir']) / "visualize_predictions.log",
        level=config['logging']['level']
    )

    logger.info("\n" + "="*80)
    logger.info("TEST SET PREDICTION VISUALIZATION")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Load test set data
        features_path = Path("data/processed/features_selected.parquet")
        if not features_path.exists():
            raise FileNotFoundError(
                f"Features not found: {features_path}\n"
                "Please run: python entrypoint/engineer_features.py first"
            )

        df = pd.read_parquet(features_path)
        logger.info(f"✓ Loaded features: {df.shape}")

        # Temporal split (same as training)
        train_size = int(0.70 * len(df))
        val_size = int(0.15 * len(df))
        train_end_idx = train_size
        test_start_idx = train_size + val_size

        train_df = df.iloc[:train_end_idx].copy()
        test_df = df.iloc[test_start_idx:].copy()
        logger.info(f"✓ Train set: {len(train_df)} samples")
        logger.info(f"✓ Test set: {len(test_df)} samples")

        # Extract features and targets
        # Target columns (no prefix)
        target_cols = ['recession_current', 'recession_next_1q', 'recession_next_2q',
                       'recession_within_2q', 'unemployment_rate_change',
                       'unemployment_claims_change', 'sp500_drawdown', 'nasdaq_drawdown',
                       'gdp_decline']

        feature_cols = [col for col in test_df.columns if col not in target_cols]
        X_test = test_df[feature_cols]
        X_train = train_df[feature_cols]

        # Stage 1 targets
        y_train_recession = train_df['recession_within_2q'].values
        y_test_recession = test_df['recession_within_2q'].values

        # Stage 2 targets (map display names to actual column names)
        stage2_targets = {
            'Unemployment Rate Change (pp)': 'unemployment_rate_change',
            'Unemployment Claims Change (%)': 'unemployment_claims_change',
            'S&P 500 Drawdown (%)': 'sp500_drawdown',
            'NASDAQ Drawdown (%)': 'nasdaq_drawdown',
            'GDP Decline (%)': 'gdp_decline'
        }

        # Extract dates if available
        if 'DATE' in train_df.columns:
            train_dates = pd.to_datetime(train_df['DATE'])
            test_dates = pd.to_datetime(test_df['DATE'])
        elif train_df.index.name == 'DATE':
            train_dates = pd.to_datetime(train_df.index)
            test_dates = pd.to_datetime(test_df.index)
        else:
            train_dates = pd.date_range(start='1980-01-01', periods=len(train_df), freq='ME')
            test_dates = pd.date_range(start='2016-01-01', periods=len(test_df), freq='ME')
            logger.warning("No DATE column found, using synthetic dates")

        # Load best model
        model_path = Path("models/best_model")
        if not model_path.exists():
            # Find most recent model
            model_dirs = sorted(Path("models").glob("run_*"))
            if not model_dirs:
                raise FileNotFoundError("No trained models found in models/")
            model_path = model_dirs[-1]
            logger.info(f"Using most recent model: {model_path}")
        else:
            logger.info(f"Using best model: {model_path}")

        model, metadata = load_model(model_path)
        logger.info(f"✓ Loaded model from {model_path}")

        # Make predictions
        logger.info("Making predictions on test set...")
        predictions = model.predict(X_test)

        # Handle predictions (could be DataFrame/Series or already numpy arrays)
        if hasattr(predictions['recession_probability'], 'values'):
            y_pred_proba = predictions['recession_probability'].values
        else:
            y_pred_proba = predictions['recession_probability']

        if hasattr(predictions['recession_predicted'], 'values'):
            y_pred_recession = predictions['recession_predicted'].values.astype(int)
        else:
            y_pred_recession = predictions['recession_predicted'].astype(int)

        logger.info(f"✓ Predictions complete")
        logger.info(f"  - Predicted recessions: {y_pred_recession.sum()}/{len(y_pred_recession)}")
        logger.info(f"  - Actual recessions: {y_test_recession.sum()}/{len(y_test_recession)}")

        # Create output directory
        output_dir = Path("reports/figures")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Full timeline recession predictions (train context + test predictions)
        logger.info("\n1. Creating full timeline with predictions...")
        plot_full_timeline_with_predictions(
            train_dates=train_dates,
            train_y_true=y_train_recession,
            test_dates=test_dates,
            test_y_true=y_test_recession,
            test_y_pred_proba=y_pred_proba,
            test_y_pred=y_pred_recession,
            save_path=output_dir / "recession_predictions_full_timeline.png",
            threshold=config['model'].get('threshold', 0.5)
        )

        # Plot 2: Individual indicator predictions vs actual (ALL test samples)
        logger.info("\n2. Creating indicator prediction plots for all test samples...")

        indicator_impacts = predictions['indicator_impacts']

        if indicator_impacts is not None and not indicator_impacts.empty:
            indicators_dict = {}

            for display_name, target_col in stage2_targets.items():
                # Prediction columns should match target column names
                if target_col not in indicator_impacts.columns:
                    logger.warning(f"  - Skipping {display_name}: prediction column '{target_col}' not found")
                    continue

                if target_col not in test_df.columns:
                    logger.warning(f"  - Skipping {display_name}: target column not found")
                    continue

                # Get actual and predicted values for ALL test samples
                y_true_indicator = test_df[target_col].values
                y_pred_indicator = indicator_impacts[target_col].values

                # Determine unit
                if 'Rate Change' in display_name or 'pp' in display_name:
                    unit = 'pp'
                else:
                    unit = '%'

                # Create individual plot with recession shading
                indicator_filename = display_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                plot_indicator_predictions_vs_actual(
                    indicator_name=display_name,
                    y_true=y_true_indicator,
                    y_pred=y_pred_indicator,
                    recession_true=y_test_recession,
                    dates=test_dates,
                    save_path=output_dir / f"indicator_{indicator_filename}.png",
                    unit=unit
                )

                # Store for combined plot (only recession periods with valid values)
                recession_mask = y_test_recession == 1
                valid_mask = ~(np.isnan(y_true_indicator) | np.isnan(y_pred_indicator)) & recession_mask
                if valid_mask.sum() > 0:
                    indicators_dict[display_name] = {
                        'y_true': y_true_indicator[valid_mask],
                        'y_pred': y_pred_indicator[valid_mask]
                    }

            # Plot 3: Combined indicators comparison
            if indicators_dict:
                logger.info("\n3. Creating combined indicators comparison...")
                plot_all_indicators_comparison(
                    indicators_dict=indicators_dict,
                    save_path=output_dir / "all_indicators_comparison.png"
                )
            else:
                logger.warning("No valid indicator predictions to plot")
        else:
            logger.warning("No indicator impacts found in predictions")

        logger.info("\n" + "="*80)
        logger.info("PREDICTION VISUALIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Plots saved to: {output_dir}")
        logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return 0

    except Exception as e:
        logger.error(f"\n✗ VISUALIZATION FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
