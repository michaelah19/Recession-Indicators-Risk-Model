"""
Feature Engineer

Creates 500+ features from quarterly economic indicators:
1. Lag features (t-1, t-2, t-3, t-4, t-8)
2. Rolling statistics (4Q, 8Q windows: mean, std, min, max, median)
3. Difference features (QoQ, YoY percentage changes)
4. Economic ratios (Buffett Indicator, Money Velocity, Credit/GDP, Yield Curve)
5. Technical indicators (RSI, MACD, Bollinger Bands for market data)

Key principle: NO DATA LEAKAGE - all features look backward only
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features from quarterly economic indicators."""

    def __init__(
        self,
        lags: List[int] = [1, 2, 3, 4, 8],
        rolling_windows: List[int] = [4, 8],
        exclude_cols: Optional[List[str]] = None
    ):
        """
        Initialize feature engineer.

        Args:
            lags: Lag periods to create (default: [1,2,3,4,8])
            rolling_windows: Window sizes for rolling statistics (default: [4,8])
            exclude_cols: Columns to exclude from feature engineering
                          (default: ['recession'] and any '_is_missing' columns)
        """
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.exclude_cols = exclude_cols or []

        logger.info(
            f"Feature Engineer initialized: "
            f"lags={lags}, rolling_windows={rolling_windows}"
        )

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of columns to create features from.

        Excludes recession column and missing indicators.

        Args:
            df: DataFrame with indicators

        Returns:
            List of column names to engineer features from
        """
        # Exclude recession and _is_missing columns by default
        exclude = set(self.exclude_cols)
        exclude.add('recession')

        feature_cols = [
            col for col in df.columns
            if col not in exclude and not col.endswith('_is_missing')
        ]

        logger.debug(f"Feature columns: {len(feature_cols)} indicators")
        return feature_cols

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for all indicators.

        For each indicator, creates features at t-1, t-2, t-3, t-4, t-8.
        Example: GDP -> GDP_lag_1, GDP_lag_2, GDP_lag_3, GDP_lag_4, GDP_lag_8

        Args:
            df: DataFrame with indicators (rows=dates, cols=indicators)

        Returns:
            DataFrame with lag features added
        """
        logger.info("Creating lag features...")

        feature_cols = self._get_feature_columns(df)
        lag_features = df.copy()

        for col in feature_cols:
            for lag in self.lags:
                lag_col_name = f"{col}_lag_{lag}"
                lag_features[lag_col_name] = df[col].shift(lag)

        n_new_features = len(feature_cols) * len(self.lags)
        logger.info(f"  Created {n_new_features} lag features")

        return lag_features

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics.

        For each indicator and window size, creates:
        - mean, std, min, max, median

        Example: GDP with 4Q window ->
                 GDP_rolling_4q_mean, GDP_rolling_4q_std, etc.

        Args:
            df: DataFrame with indicators

        Returns:
            DataFrame with rolling features added
        """
        logger.info("Creating rolling statistics features...")

        feature_cols = self._get_feature_columns(df)
        rolling_features = df.copy()

        stats = {
            'mean': 'mean',
            'std': 'std',
            'min': 'min',
            'max': 'max',
            'median': 'median'
        }

        for col in feature_cols:
            for window in self.rolling_windows:
                for stat_name, stat_func in stats.items():
                    feature_name = f"{col}_rolling_{window}q_{stat_name}"
                    rolling_features[feature_name] = (
                        df[col].rolling(window=window, min_periods=1)
                        .agg(stat_func)
                    )

        n_new_features = len(feature_cols) * len(self.rolling_windows) * len(stats)
        logger.info(f"  Created {n_new_features} rolling statistics features")

        return rolling_features

    def create_difference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create difference/change features.

        For each indicator, creates:
        - QoQ (quarter-over-quarter): (value - lag_1) / lag_1 * 100
        - YoY (year-over-year): (value - lag_4) / lag_4 * 100

        Example: GDP -> GDP_diff_qoq, GDP_diff_yoy

        Args:
            df: DataFrame with indicators

        Returns:
            DataFrame with difference features added
        """
        logger.info("Creating difference features...")

        feature_cols = self._get_feature_columns(df)
        diff_features = df.copy()

        for col in feature_cols:
            # Quarter-over-quarter change
            qoq_col = f"{col}_diff_qoq"
            diff_features[qoq_col] = df[col].pct_change(periods=1) * 100

            # Year-over-year change
            yoy_col = f"{col}_diff_yoy"
            diff_features[yoy_col] = df[col].pct_change(periods=4) * 100

        n_new_features = len(feature_cols) * 2
        logger.info(f"  Created {n_new_features} difference features")

        return diff_features

    def create_economic_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create economic ratio features.

        Ratios include:
        - Buffett Indicator: Market Cap / GDP (SPX500/GDP, NASDAQ/GDP)
        - Money Velocity: GDP / Money Supply (GDP/M1, GDP/M2)
        - Credit Ratios: Credit / GDP
        - Yield Curve: Long-term rate - Short-term rate
        - Others: Savings/GDP, Debt/Equity, etc.

        Args:
            df: DataFrame with indicators

        Returns:
            DataFrame with economic ratio features added
        """
        logger.info("Creating economic ratio features...")

        ratio_features = df.copy()
        ratios_created = []

        # Helper function to safely create ratio
        def safe_ratio(numerator, denominator, name):
            """Create ratio, handling division by zero."""
            ratio = numerator / denominator.replace(0, np.nan)
            ratio_features[name] = ratio
            ratios_created.append(name)

        # Buffett Indicator (Market Cap / GDP)
        if 'SPX500' in df.columns and 'Gross Domestic Product' in df.columns:
            safe_ratio(
                df['SPX500'],
                df['Gross Domestic Product'],
                'buffett_indicator_sp500'
            )

        if 'NASDAQ' in df.columns and 'Gross Domestic Product' in df.columns:
            safe_ratio(
                df['NASDAQ'],
                df['Gross Domestic Product'],
                'buffett_indicator_nasdaq'
            )

        # Money Velocity (GDP / Money Supply)
        if 'Gross Domestic Product' in df.columns and 'M1' in df.columns:
            safe_ratio(
                df['Gross Domestic Product'],
                df['M1'],
                'money_velocity_m1'
            )

        if 'Gross Domestic Product' in df.columns and 'M2' in df.columns:
            safe_ratio(
                df['Gross Domestic Product'],
                df['M2'],
                'money_velocity_m2'
            )

        # Credit to GDP Ratios
        if 'Bank Credit All Commercial Banks' in df.columns and 'Gross Domestic Product' in df.columns:
            safe_ratio(
                df['Bank Credit All Commercial Banks'],
                df['Gross Domestic Product'],
                'bank_credit_to_gdp'
            )

        if 'Consumer Loans Credit Cards and Other Revolving Plans All Commercial Banks' in df.columns and 'Gross Domestic Product' in df.columns:
            safe_ratio(
                df['Consumer Loans Credit Cards and Other Revolving Plans All Commercial Banks'],
                df['Gross Domestic Product'],
                'consumer_loans_to_gdp'
            )

        # Total Credit to GDP
        if all(col in df.columns for col in ['Bank Credit All Commercial Banks',
                                               'Consumer Loans Credit Cards and Other Revolving Plans All Commercial Banks',
                                               'Gross Domestic Product']):
            total_credit = (
                df['Bank Credit All Commercial Banks'] +
                df['Consumer Loans Credit Cards and Other Revolving Plans All Commercial Banks']
            )
            safe_ratio(
                total_credit,
                df['Gross Domestic Product'],
                'total_credit_to_gdp'
            )

        # Yield Curve (Long-term - Short-term rates)
        if '10-Year Real Interest Rate' in df.columns and 'Federal Funds Effective Rate' in df.columns:
            ratio_features['yield_curve_spread'] = (
                df['10-Year Real Interest Rate'] - df['Federal Funds Effective Rate']
            )
            ratios_created.append('yield_curve_spread')

        # Personal Savings to GDP
        if 'Personal Saving Rate' in df.columns and 'Gross Domestic Product' in df.columns:
            safe_ratio(
                df['Personal Saving Rate'],
                df['Gross Domestic Product'],
                'savings_to_gdp'
            )

        # Debt to Equity (Real Estate)
        if all(col in df.columns for col in ['Real Estate Loans Commercial Real Estate Loans All Commercial Banks',
                                               'Households Owners Equity in Real Estate Level']):
            safe_ratio(
                df['Real Estate Loans Commercial Real Estate Loans All Commercial Banks'],
                df['Households Owners Equity in Real Estate Level'],
                'real_estate_debt_to_equity'
            )

        # Market to Credit Ratio
        if 'SPX500' in df.columns and 'Bank Credit All Commercial Banks' in df.columns:
            safe_ratio(
                df['SPX500'],
                df['Bank Credit All Commercial Banks'],
                'sp500_to_credit'
            )

        # Unemployment to GDP
        if 'Unemployment Level' in df.columns and 'Gross Domestic Product' in df.columns:
            safe_ratio(
                df['Unemployment Level'],
                df['Gross Domestic Product'],
                'unemployment_to_gdp'
            )

        # Real vs Nominal GDP comparison
        if 'Real Gross Domestic Product' in df.columns and 'Gross Domestic Product' in df.columns:
            ratio_features['real_to_nominal_gdp'] = (
                df['Real Gross Domestic Product'] / df['Gross Domestic Product'].replace(0, np.nan)
            )
            ratios_created.append('real_to_nominal_gdp')

        # Delinquency to Consumer Loans
        if all(col in df.columns for col in ['Delinquency Rate on Credit Card Loans All Commercial Banks',
                                               'Consumer Loans Credit Cards and Other Revolving Plans All Commercial Banks']):
            safe_ratio(
                df['Delinquency Rate on Credit Card Loans All Commercial Banks'],
                df['Consumer Loans Credit Cards and Other Revolving Plans All Commercial Banks'],
                'delinquency_to_consumer_loans'
            )

        logger.info(f"  Created {len(ratios_created)} economic ratio features")

        return ratio_features

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators for market data (SPX500, NASDAQ).

        For each market index, creates:
        - RSI (Relative Strength Index, 14-period)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands (20-period, 2 std)
        - Rate of Change (ROC)
        - Simple Moving Averages (SMA 4Q, 8Q)

        Args:
            df: DataFrame with market indicators

        Returns:
            DataFrame with technical indicator features added
        """
        logger.info("Creating technical indicators...")

        tech_features = df.copy()
        indicators_created = []

        # Market columns to analyze
        market_cols = []
        if 'SPX500' in df.columns:
            market_cols.append('SPX500')
        if 'NASDAQ' in df.columns:
            market_cols.append('NASDAQ')

        for col in market_cols:
            # RSI (Relative Strength Index) - 14 period
            rsi_col = f"rsi_14_{col}"
            tech_features[rsi_col] = self._calculate_rsi(df[col], period=14)
            indicators_created.append(rsi_col)

            # MACD (Moving Average Convergence Divergence)
            macd_line, signal_line, histogram = self._calculate_macd(df[col])
            tech_features[f"macd_line_{col}"] = macd_line
            tech_features[f"macd_signal_{col}"] = signal_line
            tech_features[f"macd_histogram_{col}"] = histogram
            indicators_created.extend([
                f"macd_line_{col}",
                f"macd_signal_{col}",
                f"macd_histogram_{col}"
            ])

            # Bollinger Bands (20-period, 2 std)
            upper, lower, bandwidth, percent_b = self._calculate_bollinger_bands(
                df[col], period=20, num_std=2
            )
            tech_features[f"bb_upper_{col}"] = upper
            tech_features[f"bb_lower_{col}"] = lower
            tech_features[f"bb_bandwidth_{col}"] = bandwidth
            tech_features[f"bb_percent_b_{col}"] = percent_b
            indicators_created.extend([
                f"bb_upper_{col}",
                f"bb_lower_{col}",
                f"bb_bandwidth_{col}",
                f"bb_percent_b_{col}"
            ])

            # Rate of Change (ROC) - 4 and 8 quarters
            tech_features[f"roc_4q_{col}"] = df[col].pct_change(periods=4) * 100
            tech_features[f"roc_8q_{col}"] = df[col].pct_change(periods=8) * 100
            indicators_created.extend([f"roc_4q_{col}", f"roc_8q_{col}"])

            # Simple Moving Averages
            tech_features[f"sma_4q_{col}"] = df[col].rolling(window=4, min_periods=1).mean()
            tech_features[f"sma_8q_{col}"] = df[col].rolling(window=8, min_periods=1).mean()
            indicators_created.extend([f"sma_4q_{col}", f"sma_8q_{col}"])

            # Price relative to SMA
            tech_features[f"price_to_sma4_{col}"] = df[col] / tech_features[f"sma_4q_{col}"]
            tech_features[f"price_to_sma8_{col}"] = df[col] / tech_features[f"sma_8q_{col}"]
            indicators_created.extend([f"price_to_sma4_{col}", f"price_to_sma8_{col}"])

        logger.info(f"  Created {len(indicators_created)} technical indicator features")

        return tech_features

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            series: Price series
            period: RSI period (default: 14)

        Returns:
            RSI values (0-100)
        """
        # Calculate price changes
        delta = series.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self,
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            series: Price series
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        # Calculate EMAs
        ema_fast = series.ewm(span=fast, adjust=False, min_periods=1).mean()
        ema_slow = series.ewm(span=slow, adjust=False, min_periods=1).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()

        # Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(
        self,
        series: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            series: Price series
            period: Moving average period (default: 20)
            num_std: Number of standard deviations (default: 2)

        Returns:
            Tuple of (upper_band, lower_band, bandwidth, percent_b)
        """
        # Middle band (SMA)
        sma = series.rolling(window=period, min_periods=1).mean()

        # Standard deviation
        std = series.rolling(window=period, min_periods=1).std()

        # Upper and lower bands
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        # Bandwidth (volatility measure)
        bandwidth = (upper_band - lower_band) / sma

        # %B (position within bands)
        percent_b = (series - lower_band) / (upper_band - lower_band).replace(0, np.nan)

        return upper_band, lower_band, bandwidth, percent_b

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features: lags, rolling, differences, ratios, technical.

        This is the main method to call for complete feature engineering.

        Args:
            df: DataFrame with quarterly indicators (rows=dates, cols=indicators)

        Returns:
            DataFrame with all features added (~500+ columns)

        Example:
            >>> engineer = FeatureEngineer()
            >>> features_df = engineer.engineer_all_features(quarterly_df)
            >>> print(features_df.shape)
            (685, 550)  # ~500 features + 27 original + 27 missing indicators + recession
        """
        logger.info("=" * 80)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 80)

        # Start with original data
        features = df.copy()
        original_cols = len(features.columns)

        # 1. Lag features
        features = self.create_lag_features(features)

        # 2. Rolling statistics
        features = self.create_rolling_features(features)

        # 3. Difference features
        features = self.create_difference_features(features)

        # 4. Economic ratios
        features = self.create_economic_ratios(features)

        # 5. Technical indicators
        features = self.create_technical_indicators(features)

        # Summary
        total_cols = len(features.columns)
        new_features = total_cols - original_cols

        logger.info("=" * 80)
        logger.info(f"FEATURE ENGINEERING COMPLETE")
        logger.info(f"Original columns: {original_cols}")
        logger.info(f"New features created: {new_features}")
        logger.info(f"Total columns: {total_cols}")
        logger.info(f"Shape: {features.shape}")
        logger.info("=" * 80)

        return features


def engineer_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 2, 3, 4, 8],
    rolling_windows: List[int] = [4, 8]
) -> pd.DataFrame:
    """
    Convenience function to engineer all features.

    Args:
        df: DataFrame with quarterly indicators
        lags: Lag periods (default: [1,2,3,4,8])
        rolling_windows: Rolling window sizes (default: [4,8])

    Returns:
        DataFrame with all engineered features

    Example:
        >>> import pandas as pd
        >>> quarterly_df = pd.read_parquet("data/processed/quarterly_aligned.parquet")
        >>> features_df = engineer_features(quarterly_df)
    """
    engineer = FeatureEngineer(lags=lags, rolling_windows=rolling_windows)
    return engineer.engineer_all_features(df)
