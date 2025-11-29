"""
Fetch NBER Recession Data

Downloads NBER recession indicator (USREC) from FRED API,
converts to quarterly frequency, and saves to data/raw/

Usage:
    python entrypoint/fetch_nber.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.nber_fetcher import fetch_nber_recessions
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main():
    """Main function to fetch NBER recession data."""

    # Load configuration
    config_path = project_root / "config.yaml"
    config = load_config(config_path)

    # Setup logger
    logger = setup_logger(
        name="fetch_nber",
        log_file=project_root / "logs" / "fetch_nber.log",
        level=config['logging']['level']
    )

    logger.info("=" * 80)
    logger.info("FETCHING NBER RECESSION DATA")
    logger.info("=" * 80)

    try:
        # Get configuration
        data_paths = config['data']
        nber_config = config.get('nber', {})

        # Output path
        output_path = project_root / data_paths['raw'] / "nber_recessions.csv"

        # API key (optional)
        api_key = nber_config.get('fred_api_key')  # Can be None
        series_id = nber_config.get('series_id', 'USREC')

        logger.info(f"Fetching series: {series_id}")
        logger.info(f"Output path: {output_path}")

        # Fetch and save
        recession_df = fetch_nber_recessions(
            output_path=output_path,
            api_key=api_key,
            series_id=series_id,
            aggregation='max'  # Quarter is recession if any month is recession
        )

        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS!")
        logger.info("=" * 80)
        logger.info(f"Recession data saved to: {output_path}")
        logger.info(f"Shape: {recession_df.shape}")
        logger.info(f"Date range: {recession_df.index.min().date()} to {recession_df.index.max().date()}")
        logger.info(
            f"Recession quarters: {recession_df['recession'].sum()} "
            f"({recession_df['recession'].sum() / len(recession_df) * 100:.1f}%)"
        )
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("FAILED!")
        logger.error(f"{'=' * 80}")
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
