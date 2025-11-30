"""
Feature Selector

Selects top features using:
1. Random Forest feature importance (handles multicollinearity better than linear methods)
2. VIF (Variance Inflation Factor) to remove highly correlated features

Selection process:
1. Train Random Forest on all features
2. Rank by feature importance
3. Select top N features by importance
4. Check VIF for multicollinearity
5. Iteratively remove features with VIF > 10 until all VIF < threshold
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select top features using importance and multicollinearity checks."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
        vif_threshold: float = 10.0
    ):
        """
        Initialize feature selector.

        Args:
            n_estimators: Number of trees for Random Forest (default: 100)
            max_depth: Max depth of trees (default: 10)
            random_state: Random seed (default: 42)
            vif_threshold: VIF threshold for multicollinearity (default: 10.0)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.vif_threshold = vif_threshold

        self.feature_importances_ = None
        self.selected_features_ = None

        logger.info(
            f"Feature Selector initialized: "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"vif_threshold={vif_threshold}"
        )

    def calculate_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest.

        Args:
            X: Feature DataFrame
            y: Target series

        Returns:
            DataFrame with columns ['feature', 'importance'] sorted by importance

        Example:
            >>> selector = FeatureSelector()
            >>> importance_df = selector.calculate_feature_importance(X, y)
            >>> print(importance_df.head())
        """
        logger.info("Calculating feature importance with Random Forest...")
        logger.info(f"  Features: {X.shape[1]}")
        logger.info(f"  Samples: {X.shape[0]}")

        # Handle infinity values (replace with NaN, then impute)
        X_clean = X.replace([np.inf, -np.inf], np.nan)

        # Count infinities replaced
        inf_count = (np.isinf(X.values).sum())
        if inf_count > 0:
            logger.info(f"  Replaced {inf_count} infinity values with NaN")

        # Handle missing values (Random Forest in sklearn doesn't handle NaN)
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_clean),
            columns=X.columns,
            index=X.index
        )

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1  # Use all cores
        )

        logger.info("Training Random Forest...")
        rf.fit(X_imputed, y)

        # Get feature importances
        importances = rf.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        })

        # Sort by importance (descending)
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df = importance_df.reset_index(drop=True)

        self.feature_importances_ = importance_df

        logger.info(f"Feature importance calculated for {len(importance_df)} features")
        logger.info(f"Top 5 features:")
        for i, row in importance_df.head(5).iterrows():
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

        return importance_df

    def calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VIF (Variance Inflation Factor) for each feature.

        VIF > 10 indicates high multicollinearity.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with columns ['feature', 'vif'] sorted by VIF

        Example:
            >>> vif_df = selector.calculate_vif(X_selected)
            >>> high_vif = vif_df[vif_df['vif'] > 10]
        """
        logger.info("Calculating VIF for multicollinearity check...")

        # Handle infinity values
        X_clean = X.replace([np.inf, -np.inf], np.nan)

        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_clean),
            columns=X.columns,
            index=X.index
        )

        # Calculate VIF for each feature
        vif_data = []

        for i, col in enumerate(X_imputed.columns):
            try:
                vif = variance_inflation_factor(X_imputed.values, i)
                vif_data.append({'feature': col, 'vif': vif})
            except Exception as e:
                logger.warning(f"Could not calculate VIF for {col}: {e}")
                vif_data.append({'feature': col, 'vif': np.nan})

        vif_df = pd.DataFrame(vif_data)
        vif_df = vif_df.sort_values('vif', ascending=False)
        vif_df = vif_df.reset_index(drop=True)

        # Count high VIF features
        high_vif = (vif_df['vif'] > self.vif_threshold).sum()
        logger.info(f"  Features with VIF > {self.vif_threshold}: {high_vif}")

        if high_vif > 0:
            logger.info(f"Top 5 highest VIF features:")
            for i, row in vif_df.head(5).iterrows():
                logger.info(f"    {row['feature']}: VIF = {row['vif']:.2f}")

        return vif_df

    def remove_high_vif_features(
        self,
        X: pd.DataFrame,
        max_iterations: int = 50
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Iteratively remove features with VIF > threshold.

        Process:
        1. Calculate VIF for all features
        2. Remove feature with highest VIF if > threshold
        3. Repeat until all VIF < threshold or max iterations reached

        Args:
            X: Feature DataFrame
            max_iterations: Maximum iterations (default: 50)

        Returns:
            Tuple of (X_filtered, removed_features)
            - X_filtered: DataFrame with low-VIF features
            - removed_features: List of removed feature names

        Example:
            >>> X_filtered, removed = selector.remove_high_vif_features(X_selected)
            >>> print(f"Removed {len(removed)} features due to high VIF")
        """
        logger.info(f"Removing features with VIF > {self.vif_threshold}...")

        X_filtered = X.copy()
        removed_features = []

        for iteration in range(max_iterations):
            # Calculate VIF
            vif_df = self.calculate_vif(X_filtered)

            # Find max VIF
            max_vif_row = vif_df.iloc[0]
            max_vif = max_vif_row['vif']

            # Stop if all VIF below threshold
            if max_vif <= self.vif_threshold or pd.isna(max_vif):
                logger.info(
                    f"  All features have VIF <= {self.vif_threshold} "
                    f"after {iteration} iterations"
                )
                break

            # Remove feature with highest VIF
            feature_to_remove = max_vif_row['feature']
            removed_features.append(feature_to_remove)
            X_filtered = X_filtered.drop(columns=[feature_to_remove])

            logger.info(
                f"  Iteration {iteration + 1}: Removed '{feature_to_remove}' "
                f"(VIF = {max_vif:.2f}), {X_filtered.shape[1]} features remaining"
            )

        if iteration == max_iterations - 1:
            logger.warning(
                f"Reached max iterations ({max_iterations}). "
                f"Some features may still have VIF > {self.vif_threshold}"
            )

        logger.info(f"VIF filtering complete:")
        logger.info(f"  Removed: {len(removed_features)} features")
        logger.info(f"  Remaining: {X_filtered.shape[1]} features")

        return X_filtered, removed_features

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 50,
        check_vif: bool = True
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Select top features using importance and VIF.

        Process:
        1. Calculate feature importance with Random Forest
        2. Select top n_features by importance
        3. (Optional) Remove high-VIF features iteratively
        4. Return final feature list

        Args:
            X: Feature DataFrame (all features)
            y: Target series
            n_features: Number of features to select initially (default: 50)
            check_vif: Whether to check VIF and remove correlated features (default: True)

        Returns:
            Tuple of (selected_features, importance_df)
            - selected_features: List of selected feature names
            - importance_df: Full importance DataFrame

        Example:
            >>> selector = FeatureSelector()
            >>> selected, importance_df = selector.select_features(X, y, n_features=50)
            >>> X_selected = X[selected]
        """
        logger.info("=" * 80)
        logger.info("FEATURE SELECTION PIPELINE")
        logger.info("=" * 80)

        # Step 1: Calculate feature importance
        importance_df = self.calculate_feature_importance(X, y)

        # Step 2: Select top n features by importance
        top_features = importance_df.head(n_features)['feature'].tolist()
        logger.info(f"\nSelected top {len(top_features)} features by importance")

        X_selected = X[top_features]

        # Step 3: Remove high-VIF features if requested
        if check_vif:
            logger.info(f"\nChecking multicollinearity (VIF threshold = {self.vif_threshold})...")
            X_selected, removed_features = self.remove_high_vif_features(X_selected)

            # Update selected features list
            selected_features = X_selected.columns.tolist()

            if removed_features:
                logger.info(f"\nRemoved {len(removed_features)} features due to high VIF:")
                for feat in removed_features:
                    logger.info(f"  - {feat}")
        else:
            selected_features = top_features

        # Store selected features
        self.selected_features_ = selected_features

        logger.info("=" * 80)
        logger.info(f"FEATURE SELECTION COMPLETE")
        logger.info(f"Final selected features: {len(selected_features)}")
        logger.info("=" * 80)

        return selected_features, importance_df

    def save_importance(
        self,
        importance_df: pd.DataFrame,
        output_path: Path
    ) -> None:
        """
        Save feature importance to CSV.

        Args:
            importance_df: Feature importance DataFrame
            output_path: Path to save CSV

        Example:
            >>> selector.save_importance(importance_df, Path("data/processed/feature_importance.csv"))
        """
        logger.info(f"Saving feature importance to {output_path}...")

        # Create parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        importance_df.to_csv(output_path, index=False)

        logger.info(f"Feature importance saved: {len(importance_df)} features")


def select_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 50,
    vif_threshold: float = 10.0,
    output_path: Optional[Path] = None
) -> Tuple[List[str], pd.DataFrame]:
    """
    Convenience function to select top features.

    Args:
        X: Feature DataFrame
        y: Target series
        n_features: Number of features to select (default: 50)
        vif_threshold: VIF threshold (default: 10.0)
        output_path: Optional path to save importance CSV

    Returns:
        Tuple of (selected_features, importance_df)

    Example:
        >>> X_train, y_train = split_features_targets(features_df)
        >>> selected, importance = select_top_features(
        ...     X_train, y_train,
        ...     n_features=50,
        ...     output_path=Path("data/processed/feature_importance.csv")
        ... )
        >>> X_selected = X_train[selected]
    """
    selector = FeatureSelector(vif_threshold=vif_threshold)
    selected_features, importance_df = selector.select_features(X, y, n_features=n_features)

    if output_path:
        selector.save_importance(importance_df, output_path)

    return selected_features, importance_df
