# Feature Engineering

Complete documentation of the feature engineering pipeline that transforms 27 raw economic indicators into 5,416 engineered features, then selects the top 29 for model training.

---

## Overview

The core of this project is transforming raw economic indicators into predictive features that capture recession patterns. We process quarterly economic data through a sophisticated feature engineering pipeline.

**Summary**:
- **Input**: 27 economic indicators, 685 quarters (1854-2025)
- **Output**: 5,416 engineered features → 29 selected features
- **Selection**: Random Forest importance + VIF multicollinearity filtering

---

## Feature Engineering Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QUARTERLY ALIGNED DATA                            │
│              685 quarters × 27 indicators (1854-2025)                │
│                         ↓                                            │
│  ┌──────────────────────────────────────────────────────┐           │
│  │         FEATURE ENGINEERING PROCESS                  │           │
│  └──────────────────────────────────────────────────────┘           │
│                         ↓                                            │
│  ┌─────────────────────┬──────────────────┬────────────────────┐   │
│  │   LAG FEATURES      │  ROLLING STATS   │   DIFFERENCE       │   │
│  │   t-1, t-2, t-3,    │  4Q & 8Q windows │   QoQ, YoY         │   │
│  │   t-4, t-8          │  mean, std, min, │   % changes        │   │
│  │   (135 features)    │  max, median     │   (54 features)    │   │
│  │                     │  (270 features)  │                    │   │
│  └─────────────────────┴──────────────────┴────────────────────┘   │
│                         ↓                                            │
│  ┌─────────────────────┬──────────────────┬────────────────────┐   │
│  │  ECONOMIC RATIOS    │  TECHNICAL       │   INTERACTIONS     │   │
│  │  Buffett Indicator  │  RSI, MACD,      │   Cross features   │   │
│  │  Money Velocity     │  Bollinger Bands │   Derived ratios   │   │
│  │  Credit/GDP         │  (30 features)   │   (many features)  │   │
│  │  (15+ features)     │                  │                    │   │
│  └─────────────────────┴──────────────────┴────────────────────┘   │
│                         ↓                                            │
│              TOTAL: 5,416 ENGINEERED FEATURES                       │
│                         ↓                                            │
│  ┌──────────────────────────────────────────────────────┐           │
│  │            FEATURE SELECTION PROCESS                 │           │
│  │  1. Random Forest Feature Importance                 │           │
│  │  2. Select Top 50 by Importance                      │           │
│  │  3. VIF Multicollinearity Check (VIF < 10)           │           │
│  │  4. Iteratively Remove High-VIF Features             │           │
│  └──────────────────────────────────────────────────────┘           │
│                         ↓                                            │
│              FINAL: 29 SELECTED FEATURES                            │
│         Ready for Model Training (683 samples)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Feature Categories

### 1. Lag Features (135 features)

Historical values at various time lags to capture temporal patterns.

**Lags**: t-1, t-2, t-3, t-4, t-8 quarters

**Examples**:
- `GDP_lag_1` - GDP value from previous quarter
- `Unemployment_lag_4` - Unemployment from 4 quarters ago
- `FedFunds_lag_8` - Federal Funds Rate from 8 quarters ago

**Purpose**: Capture momentum and trends in economic indicators

### 2. Rolling Statistics (270 features)

Statistical aggregations over rolling windows to smooth volatility.

**Windows**: 4-quarter (1 year) and 8-quarter (2 year)

**Statistics**: mean, std, min, max, median

**Examples**:
- `GDP_rolling_4q_mean` - Average GDP over last 4 quarters
- `SPX500_rolling_8q_std` - S&P 500 volatility over 8 quarters
- `Unemployment_rolling_4q_max` - Peak unemployment in last year

**Purpose**: Capture trends, volatility, and extremes over time

### 3. Difference Features (54 features)

Rate of change calculations to capture growth/decline patterns.

**Types**:
- **Quarter-over-Quarter (QoQ)**: `(X_t - X_{t-1}) / X_{t-1} × 100`
- **Year-over-Year (YoY)**: `(X_t - X_{t-4}) / X_{t-4} × 100`

**Examples**:
- `GDP_diff_qoq` - Real GDP quarterly growth rate
- `Unemployment_diff_yoy` - Year-over-year unemployment change
- `SP500_diff_qoq` - Quarterly stock market return

**Purpose**: Identify acceleration/deceleration in economic activity

### 4. Economic Ratios (15+ features)

Domain-specific ratios with economic interpretation.

**Examples**:
- `buffett_indicator_sp500` - Market cap to GDP ratio (S&P 500 / GDP)
- `money_velocity_m1` - GDP / M1 money supply
- `credit_to_gdp` - Total credit outstanding / GDP
- `unemployment_claims_ratio` - Initial claims / total unemployment

**Purpose**: Capture fundamental economic relationships and imbalances

### 5. Technical Indicators (30 features)

Market analysis indicators adapted for economic data.

**Examples**:
- `rsi_14_SPX500` - Relative Strength Index for S&P 500
- `macd_line_NASDAQ` - Moving Average Convergence Divergence
- `bollinger_band_upper_GDP` - Bollinger Band for GDP

**Purpose**: Identify overbought/oversold conditions and momentum

### 6. Interaction Features (4,912+ features)

Cross-indicator features and complex transformations.

**Types**:
- Products of features (e.g., `GDP_growth × Unemployment_change`)
- Ratios of different indicators
- Polynomial transformations
- Composite features combining multiple transformations

**Purpose**: Capture non-linear relationships and interaction effects

---

## Feature Selection Process

### Step 1: Random Forest Importance

Train a Random Forest classifier on all 5,416 features to rank importance.

**Why Random Forest?**
- Handles non-linear relationships
- Captures interaction effects
- Robust to outliers
- Provides feature importance scores

### Step 2: Select Top 50

Keep the 50 features with highest importance scores.

**Threshold**: Reduces dimensionality while retaining most predictive power

### Step 3: VIF Multicollinearity Check

Calculate Variance Inflation Factor (VIF) for remaining features.

**VIF Threshold**: 10.0

**Formula**: `VIF_j = 1 / (1 - R²_j)`

where R²_j is from regressing feature j on all other features.

### Step 4: Iterative Removal

Iteratively remove features with VIF > 10 until all features have VIF < 10.

**Algorithm**:
1. Calculate VIF for all features
2. Remove feature with highest VIF
3. Repeat until all VIF < 10

**Result**: 29 final features (21 removed due to multicollinearity)

---

## Final Feature Set (29 Features)

### Top 15 Most Important Features

| Rank | Feature | Importance | Category | Description |
|------|---------|------------|----------|-------------|
| 1 | `LABELS_USREC` | 4.16% | Original | Historical recession indicator |
| 2 | `LABELS_USREC_rolling_4q_mean_diff_qoq` | 2.90% | Composite | QoQ change in 4Q rolling mean of recession |
| 3 | `LABELS_USREC_rolling_8q_mean_diff_qoq` | 2.13% | Composite | QoQ change in 8Q rolling mean |
| 4 | `LABELS_USREC_rolling_4q_mean` | 2.06% | Rolling | 4Q rolling average of recession |
| 5 | `LABELS_USREC_rolling_4q_min` | 1.48% | Rolling | 4Q rolling minimum |
| 6 | `LABELS_USREC_rolling_4q_median` | 1.17% | Rolling | 4Q rolling median |
| 7 | `LABELS_USREC_lag_1_rolling_4q_mean_diff_qoq` | 1.13% | Composite | Lagged rolling mean QoQ change |
| 8 | `LABELS_USREC_lag_1` | 0.98% | Lag | Previous quarter recession status |
| 9 | `LABELS_USREC_rolling_4q_max` | 0.83% | Rolling | 4Q rolling maximum |
| 10 | `LABELS_USREC_rolling_4q_mean_diff_yoy` | 0.82% | Composite | YoY change in rolling mean |
| 11 | `LABELS_USREC_diff_yoy` | 0.73% | Difference | Year-over-year change |
| 12 | `LABELS_USREC_diff_qoq` | 0.71% | Difference | Quarter-over-quarter change |
| 13 | `LABELS_USREC_rolling_8q_mean` | 0.70% | Rolling | 8Q rolling average |
| 14 | `Federal Funds Rate_lag_2_rolling_8q_median_diff_yoy` | 0.70% | Composite | Fed funds rate composite feature |
| 15 | `Real GDP_diff_qoq` | 0.68% | Difference | Real GDP quarterly growth rate |

### Key Insights

**Historical Recession Indicator Dominates**:
- 13 of 29 features are transformations of `LABELS_USREC` (historical recession indicator)
- Strong temporal autocorrelation in recession patterns
- Past recessions are highly predictive of future recessions

**Economic Fundamentals**:
- Federal Funds Rate: 3 features (monetary policy signal)
- Unemployment Level: 6 features (labor market health)
- Real GDP: 2 features (economic output)
- Unemployment Rate: 2 features (labor market slack)

**Transformation Types**:
- Composite features (combinations of lags, rolling stats, and differences) are most important
- Simple lag features alone are less predictive
- QoQ and YoY changes capture momentum

---

## Feature Distribution by Base Indicator

**Top 5 Contributing Indicators** (by number of selected features):

1. **LABELS_USREC** (Historical Recession): 13 features
2. **Unemployment Level**: 6 features
3. **Federal Funds Effective Rate**: 3 features
4. **Real Gross Domestic Product**: 2 features
5. **Unemployment Rate**: 2 features

---

## Target Variables

Four target variables created for different prediction horizons:

| Target | Description | Horizon | Class Balance |
|--------|-------------|---------|---------------|
| `recession_current` | Current quarter recession status | 0Q | 31.2% positive |
| `recession_next_1q` | Recession in next quarter | 1Q | ~30% positive |
| `recession_next_2q` | Recession 2 quarters ahead | 2Q | ~30% positive |
| `recession_within_2q` | **PRIMARY TARGET** - Recession in next 1-2 quarters | 1-2Q | ~35% positive |

**Selected Target**: `recession_within_2q`

**Rationale**: Provides early warning (1-2 quarters ahead) while maintaining reasonable class balance for training.

---

## Key Technical Decisions

### 1. No Data Leakage

**Rule**: All features look backward only; targets properly shifted forward

**Example**: When predicting recession at time t+2, features only use data up to time t.

**Implementation**: Explicit time-based train/validation/test splits with no shuffling.

### 2. Multicollinearity Handling

**Method**: VIF (Variance Inflation Factor) threshold of 10.0

**Why**: Highly correlated features cause:
- Unstable coefficient estimates
- Inflated standard errors
- Poor generalization

**Result**: 21 features removed (50 → 29)

### 3. Missing Data Strategy

**Preprocessing**: Replace ±∞ with NaN

**Imputation**: Median imputation for all features

**Training**: Models only train on samples with valid targets

**Why Median**: Robust to outliers, maintains distribution shape

### 4. Feature Selection Method

**Random Forest Importance**:
- Handles non-linear relationships
- Captures feature interactions
- More robust than linear methods (correlation, mutual information)

**Advantages**:
- No assumptions about feature-target relationship
- Works with categorical and continuous features
- Provides importance scores for ranking

### 5. Window Sizes

**4-Quarter (1 year)**: Captures annual cycles and short-term trends

**8-Quarter (2 years)**: Captures business cycle patterns (avg recession ~5 quarters)

**Lag Depths**: t-1, t-2, t-3, t-4, t-8 chosen to balance recency vs historical patterns

---

## Files Generated

```
data/processed/
├── features_full.parquet          # 13.04 MB - All 5,416 features
├── features_selected.parquet      # 0.07 MB - Top 29 features (ready for training)
└── feature_importance.csv         # Full importance rankings
```

### Feature Importance CSV

Contains all 5,416 features ranked by importance:

| Column | Description |
|--------|-------------|
| Feature name | Original feature name |
| Importance | Importance score from Random Forest |

---

## Running Feature Engineering

### Complete Pipeline

```bash
python entrypoint/engineer_features.py
```

**Output**:
- Creates 5,416 features from 27 indicators
- Selects top 29 features via Random Forest + VIF
- Saves processed data ready for model training
- Generates feature importance rankings

**Execution Time**: ~2-5 minutes (depends on hardware)

**Memory Requirements**: ~2-4 GB RAM

---

## Feature Engineering Code Structure

Located in `src/features/`:

- **`engineer.py`** - Main feature engineering logic
  - Lag feature creation
  - Rolling statistics
  - Difference calculations
  - Economic ratios
  - Technical indicators
  - Interaction features

- **`selector.py`** - Feature selection logic
  - Random Forest importance ranking
  - VIF multicollinearity filtering
  - Feature filtering and saving

- **`targets.py`** - Target variable creation
  - Recession horizon targets
  - Indicator change targets (for Stage 2)
  - Target validation

---

## Mathematical Formulations

For detailed mathematical formulations of all feature transformations, see:
- **[docs/methodology.md](../../docs/methodology.md)** - Section on Feature Engineering

---

## Best Practices

1. **Always run feature engineering before training**: Ensures features are up-to-date
2. **Check feature importance**: Understand which features drive predictions
3. **Monitor VIF**: High multicollinearity indicates redundant features
4. **Validate no leakage**: Ensure future information doesn't leak into features
5. **Use consistent preprocessing**: Apply same transformations to train and test

---

*For model training using these features, see the Usage section in the main [README.md](../../README.md)*

*For technical details and mathematical formulations, see [docs/methodology.md](../../docs/methodology.md)*
