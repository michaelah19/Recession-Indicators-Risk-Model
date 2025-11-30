"""
Target Variable Creation

Creates recession prediction target variables:
1. recession_current: Is current quarter a recession (same as 'recession')
2. recession_next_1q: Will next quarter (1Q ahead) be a recession
3. recession_next_2q: Will 2 quarters ahead be a recession
4. recession_within_2q: PRIMARY TARGET - recession in next 1-2 quarters

Key principle: Targets are properly shifted forward to avoid data leakage
"""

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def create_recession_targets(
    recession_series: pd.Series,
    horizons: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Create recession prediction target variables.

    Args:
        recession_series: Binary series indicating recession quarters (0/1)
        horizons: Dictionary mapping target name -> quarters ahead
                  Default: {'current': 0, 'next_1q': 1, 'next_2q': 2}

    Returns:
        DataFrame with target columns:
        - recession_current: Current quarter recession status
        - recession_next_1q: Next quarter (1Q ahead)
        - recession_next_2q: 2 quarters ahead
        - recession_within_2q: Recession in next 1-2 quarters (PRIMARY)

    Example:
        >>> recession = quarterly_df['recession']
        >>> targets = create_recession_targets(recession)
        >>> print(targets.columns)
        ['recession_current', 'recession_next_1q', 'recession_next_2q', 'recession_within_2q']
    """
    logger.info("Creating recession target variables...")

    # Default horizons
    if horizons is None:
        horizons = {
            'current': 0,
            'next_1q': 1,
            'next_2q': 2
        }

    # Initialize DataFrame
    targets = pd.DataFrame(index=recession_series.index)

    # Create each target by shifting recession indicator forward
    for target_name, quarters_ahead in horizons.items():
        col_name = f"recession_{target_name}"

        if quarters_ahead == 0:
            # Current quarter - no shift needed
            targets[col_name] = recession_series.astype(int)
        else:
            # Future quarters - shift backward (negative shift = look forward)
            # shift(-1) means: target for row i = value from row i+1
            targets[col_name] = recession_series.shift(-quarters_ahead).astype('Int64')

        # Count how many target=1
        n_positive = targets[col_name].sum()
        pct_positive = (n_positive / len(targets[col_name].dropna())) * 100

        logger.info(
            f"  {col_name}: {n_positive} recession quarters "
            f"({pct_positive:.1f}%)"
        )

    # Create PRIMARY target: recession_within_2q
    # This is 1 if EITHER next_1q OR next_2q is a recession
    targets['recession_within_2q'] = (
        (targets['recession_next_1q'] == 1) |
        (targets['recession_next_2q'] == 1)
    ).astype('Int64')

    n_within_2q = targets['recession_within_2q'].sum()
    pct_within_2q = (n_within_2q / len(targets['recession_within_2q'].dropna())) * 100

    logger.info(
        f"  recession_within_2q (PRIMARY): {n_within_2q} quarters "
        f"({pct_within_2q:.1f}%)"
    )

    # Summary
    logger.info(f"\nTarget variables created:")
    logger.info(f"  - recession_current: Current quarter status")
    logger.info(f"  - recession_next_1q: 1 quarter ahead")
    logger.info(f"  - recession_next_2q: 2 quarters ahead")
    logger.info(f"  - recession_within_2q: PRIMARY - recession in next 1-2 quarters")
    logger.info(f"\nNote: Last 2 rows will have NaN for future targets (no future data)")

    return targets


def add_targets_to_features(
    features_df: pd.DataFrame,
    recession_series: pd.Series
) -> pd.DataFrame:
    """
    Add recession targets to feature DataFrame.

    Args:
        features_df: DataFrame with engineered features
        recession_series: Binary recession indicator

    Returns:
        DataFrame with features + target columns

    Example:
        >>> features = engineer.engineer_all_features(quarterly_df)
        >>> recession = quarterly_df['recession']
        >>> features_with_targets = add_targets_to_features(features, recession)
    """
    logger.info("Adding target variables to features DataFrame...")

    # Create targets
    targets = create_recession_targets(recession_series)

    # Combine features and targets
    # Use same index to ensure alignment
    features_with_targets = features_df.copy()

    for col in targets.columns:
        features_with_targets[col] = targets[col]

    logger.info(f"Features + Targets shape: {features_with_targets.shape}")

    return features_with_targets


def get_primary_target(targets_df: pd.DataFrame) -> pd.Series:
    """
    Get the primary target variable for modeling.

    Args:
        targets_df: DataFrame with target variables

    Returns:
        Primary target series (recession_within_2q)

    Example:
        >>> targets = create_recession_targets(recession)
        >>> y = get_primary_target(targets)
        >>> print(y.name)
        'recession_within_2q'
    """
    if 'recession_within_2q' not in targets_df.columns:
        raise ValueError("recession_within_2q not found in targets DataFrame")

    return targets_df['recession_within_2q']


def validate_targets(targets_df: pd.DataFrame) -> None:
    """
    Validate target variables for correctness.

    Checks:
    - All targets are binary (0/1 or NaN)
    - recession_within_2q = recession_next_1q | recession_next_2q
    - No unexpected NaNs in non-future targets

    Args:
        targets_df: DataFrame with target variables

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating target variables...")

    required_cols = [
        'recession_current',
        'recession_next_1q',
        'recession_next_2q',
        'recession_within_2q'
    ]

    # Check all required columns exist
    for col in required_cols:
        if col not in targets_df.columns:
            raise ValueError(f"Required target column missing: {col}")

    # Check all targets are binary (0/1 or NaN)
    for col in required_cols:
        unique_vals = targets_df[col].dropna().unique()
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(
                f"Target {col} has non-binary values: {unique_vals}"
            )

    # Verify recession_within_2q correctness
    expected_within_2q = (
        (targets_df['recession_next_1q'] == 1) |
        (targets_df['recession_next_2q'] == 1)
    ).astype('Int64')

    if not targets_df['recession_within_2q'].equals(expected_within_2q):
        logger.warning(
            "recession_within_2q does not match expected values "
            "(recession_next_1q | recession_next_2q)"
        )

    # Check for unexpected NaNs in current target
    current_nans = targets_df['recession_current'].isna().sum()
    if current_nans > 0:
        logger.warning(
            f"recession_current has {current_nans} NaN values "
            f"(should match original recession series)"
        )

    # Check future targets have NaNs at the end (expected)
    next_1q_nans = targets_df['recession_next_1q'].isna().sum()
    next_2q_nans = targets_df['recession_next_2q'].isna().sum()

    logger.info(f"  recession_current NaNs: {current_nans}")
    logger.info(f"  recession_next_1q NaNs: {next_1q_nans} (expected: ~1)")
    logger.info(f"  recession_next_2q NaNs: {next_2q_nans} (expected: ~2)")

    logger.info("Target validation complete ")


def split_features_targets(
    df: pd.DataFrame,
    target_col: str = 'recession_within_2q'
) -> tuple:
    """
    Split DataFrame into features (X) and target (y).

    Removes all target columns from X, keeps only specified target in y.
    Also removes rows with NaN targets (last 2 rows typically).

    Args:
        df: DataFrame with features and targets
        target_col: Name of target column to use (default: recession_within_2q)

    Returns:
        Tuple of (X, y) where:
        - X: DataFrame with only feature columns
        - y: Series with target values (NaN rows removed)

    Example:
        >>> X, y = split_features_targets(features_with_targets)
        >>> print(X.shape, y.shape)
        (683, 530) (683,)  # 2 rows removed due to NaN targets
    """
    logger.info(f"Splitting features and target ({target_col})...")

    # Identify target columns
    target_cols = [
        'recession_current',
        'recession_next_1q',
        'recession_next_2q',
        'recession_within_2q'
    ]

    # Get target
    y = df[target_col].copy()

    # Get features (exclude all target columns and original recession)
    feature_cols = [col for col in df.columns if col not in target_cols and col != 'recession']
    X = df[feature_cols].copy()

    # Remove rows where target is NaN
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(f"  Features (X): {X.shape}")
    logger.info(f"  Target (y): {y.shape}")
    logger.info(f"  Target class distribution:")
    logger.info(f"    Class 0 (no recession): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    logger.info(f"    Class 1 (recession): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

    return X, y
