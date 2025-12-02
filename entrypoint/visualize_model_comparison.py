"""
Visualization Generation Script

Generates all static plots for model analysis and reporting.

Usage:
    python entrypoint/visualize.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.visualization.static_plots import generate_all_plots


def main():
    """Generate all visualizations."""
    # Load config
    config = load_config("config.yaml")

    # Setup logger
    logger = setup_logger(
        name="visualize",
        log_file=Path(config['logging']['dir']) / "visualize.log",
        level=config['logging']['level']
    )

    logger.info("\n" + "="*80)
    logger.info("VISUALIZATION GENERATION")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Load model comparison results
        comparison_path = Path("reports/model_comparison.csv")
        if not comparison_path.exists():
            raise FileNotFoundError(
                f"Model comparison not found: {comparison_path}\n"
                "Please run: python entrypoint/compare_models.py first"
            )

        comparison_df = pd.read_csv(comparison_path)
        logger.info(f"✓ Loaded model comparison: {len(comparison_df)} models")

        # Load feature importance
        importance_path = Path("data/processed/feature_importance.csv")
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path, header=None, names=['feature', 'importance'])
            logger.info(f"✓ Loaded feature importance: {len(importance_df)} features")
        else:
            logger.warning(f"Feature importance not found at {importance_path}")
            importance_df = None

        # Generate all plots
        output_dir = Path("reports/figures")
        generate_all_plots(
            comparison_df=comparison_df,
            importance_df=importance_df,
            output_dir=output_dir
        )

        logger.info("\n" + "="*80)
        logger.info("VISUALIZATION GENERATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Plots saved to: {output_dir}")
        logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return 0

    except Exception as e:
        logger.error(f"\n✗ VISUALIZATION FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
