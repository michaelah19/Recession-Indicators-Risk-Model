"""
Preprocess Economic Indicators Data

Loads all CSV files from data/raw/, validates data quality,
aligns to quarterly frequency, and saves to data/processed/

Steps:
1. Load all indicator CSV files (27 files)
2. Validate data quality
3. Load NBER recession labels
4. Align all data to quarterly frequency
5. Handle missing data (forward-fill up to 1 quarter)
6. Save to data/processed/quarterly_aligned.parquet

Usage:
    python entrypoint/preprocess.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.data.loader import CSVLoader
from src.data.validator import DataValidator
from src.data.preprocessor import DataPreprocessor
from src.utils.config import load_config, get_data_paths
from src.utils.logger import setup_logger


def main():
    """Main function to preprocess data."""

    # Load configuration
    config_path = project_root / "config.yaml"
    config = load_config(config_path)
    data_paths = get_data_paths(config)

    # Setup logger
    logger = setup_logger(
        name="preprocess",
        log_file=project_root / "logs" / "preprocessing.log",
        level=config['logging']['level']
    )

    logger.info("=" * 80)
    logger.info("DATA PREPROCESSING PIPELINE")
    logger.info("=" * 80)

    try:
        # Paths
        raw_data_dir = project_root / data_paths['raw']
        processed_data_dir = project_root / data_paths['processed']

        logger.info(f"Raw data directory: {raw_data_dir}")
        logger.info(f"Processed data directory: {processed_data_dir}")
        logger.info("")

        # =====================================================================
        # STEP 1: Load all indicator CSV files
        # =====================================================================
        logger.info("STEP 1: Loading indicator CSV files...")
        logger.info("-" * 80)

        loader = CSVLoader(data_dir=raw_data_dir)

        # Load all indicators (exclude NBER recession file and README)
        indicators = loader.load_all_indicators(
            exclude_patterns=['nber', 'README', 'pdf', 'Information']
        )

        logger.info(f"\nLoaded {len(indicators)} indicators successfully")
        logger.info("")

        # Get indicator summary
        indicator_info = loader.get_indicator_info(indicators)
        logger.info("Indicator Summary:")
        logger.info(f"\n{indicator_info.to_string()}")
        logger.info("")

        # =====================================================================
        # STEP 2: Validate data quality
        # =====================================================================
        logger.info("STEP 2: Validating data quality...")
        logger.info("-" * 80)

        validator = DataValidator(
            outlier_method='zscore',
            zscore_threshold=5.0,
            max_gap_days=365
        )

        validation_reports = validator.validate_all_indicators(
            indicators=indicators,
            check_outliers=True
        )

        # Print validation summary
        validator.print_validation_summary(
            reports=validation_reports,
            show_all=False  # Only show failed validations
        )

        # Save validation report
        validation_report_path = processed_data_dir / "validation_report.csv"
        validator.save_validation_report(
            reports=validation_reports,
            output_path=validation_report_path
        )
        logger.info(f"\nValidation report saved to: {validation_report_path}")
        logger.info("")

        # =====================================================================
        # STEP 3: Load NBER recession labels
        # =====================================================================
        logger.info("STEP 3: Loading NBER recession labels...")
        logger.info("-" * 80)

        nber_path = raw_data_dir / "nber_recessions.csv"

        if nber_path.exists():
            recession_df = pd.read_csv(nber_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded NBER recession data: {recession_df.shape}")
            logger.info(
                f"Recession quarters: {recession_df['recession'].sum()} "
                f"({recession_df['recession'].sum() / len(recession_df) * 100:.1f}%)"
            )
        else:
            logger.warning(f"NBER recession file not found: {nber_path}")
            logger.warning("Run 'python entrypoint/fetch_nber.py' first!")
            logger.warning("Continuing without recession labels...")
            recession_df = None

        logger.info("")

        # =====================================================================
        # STEP 4: Align to quarterly frequency
        # =====================================================================
        logger.info("STEP 4: Aligning to quarterly frequency...")
        logger.info("-" * 80)

        preprocessor = DataPreprocessor(
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            target_freq='QS',  # Quarter Start
            forward_fill_limit=1  # Forward-fill up to 1 quarter
        )

        quarterly_df = preprocessor.preprocess(
            indicators=indicators,
            recession_df=recession_df,
            output_filename="quarterly_aligned.parquet"
        )

        logger.info("")

        # =====================================================================
        # SUCCESS SUMMARY
        # =====================================================================
        logger.info("=" * 80)
        logger.info("PREPROCESSING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Output file: {processed_data_dir / 'quarterly_aligned.parquet'}")
        logger.info(f"Shape: {quarterly_df.shape}")
        logger.info(f"Date range: {quarterly_df.index.min().date()} to {quarterly_df.index.max().date()}")
        logger.info(f"Indicators: {len([c for c in quarterly_df.columns if not c.endswith('_is_missing') and c != 'recession'])}")

        # Missing data summary
        total_values = quarterly_df.shape[0] * quarterly_df.shape[1]
        missing_values = quarterly_df.isna().sum().sum()
        missing_pct = (missing_values / total_values) * 100
        logger.info(f"Missing values: {missing_values:,} ({missing_pct:.2f}%)")

        if 'recession' in quarterly_df.columns:
            logger.info(f"\nRecession periods included:")
            logger.info(
                f"  {quarterly_df['recession'].sum()} quarters "
                f"({quarterly_df['recession'].sum() / len(quarterly_df) * 100:.1f}%)"
            )

        logger.info("\nNext step: Run feature engineering")
        logger.info("  python entrypoint/engineer_features.py")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("PREPROCESSING FAILED!")
        logger.error(f"{'=' * 80}")
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
