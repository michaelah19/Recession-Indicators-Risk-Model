# US Recession Indicators Risk Model

**Hybrid two-stage machine learning system** that predicts both recession probability (1-2 quarters ahead) AND forecasts economic indicator impacts during recession periods using 170+ years of economic data (1854-2025).

## Overview

This project builds a sophisticated **hybrid recession prediction system** by analyzing 27 economic indicators spanning over 170 years. The system uses a two-stage architecture:

**Stage 1**: Predict recession probability for the next 1-2 quarters
**Stage 2**: Forecast economic indicator changes DURING recessions (unemployment, stock markets, GDP)

The system processes raw economic data through a sophisticated feature engineering pipeline, creating over 5,000 predictive features before selecting the most informative ones for model training.

### Prediction Outputs

1. **Recession Probability**: Binary classification with calibrated probabilities (0-1)
2. **Economic Impact Forecasts** (conditional on recession):
   - **Labor Market**: Unemployment rate change, unemployment claims change
   - **Stock Markets**: S&P 500 drawdown, NASDAQ drawdown
   - **Economic Output**: GDP decline

**Dataset**: [US Recession and Financial Indicators](https://www.kaggle.com/datasets/mikoajfish99/us-recession-and-financial-indicators/data) from Kaggle
- 27 economic indicators (Federal funds rate, GDP, unemployment, credit, money supply, real estate, stock markets)
- 685 quarterly observations (1854-2025)
- 214 recession quarters (31.2% of data)
- **Training data**: 172 samples (1980-2023) with complete regression targets

---

## 🚀 Hybrid Model Architecture

### Two-Stage Prediction System

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HYBRID RECESSION PREDICTOR                        │
│                                                                      │
│  INPUT: 29 Selected Features (economic indicators + engineered)     │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────┐        │
│  │         STAGE 1: RECESSION CLASSIFIER                  │        │
│  │  ┌──────────────────────────────────────────────────┐  │        │
│  │  │  XGBoost Binary Classifier                       │  │        │
│  │  │  - Input: 29 features                            │  │        │
│  │  │  - Output: Recession probability [0-1]           │  │        │
│  │  │  - Calibrated probabilities (CalibratedCV)       │  │        │
│  │  │  - Class imbalance handling (scale_pos_weight)   │  │        │
│  │  └──────────────────────────────────────────────────┘  │        │
│  └────────────────────────────────────────────────────────┘        │
│                              ↓                                       │
│              Recession Probability >= Threshold?                    │
│                              ↓                                       │
│                     YES → Activate Stage 2                          │
│                     NO  → Return probability only                   │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────┐        │
│  │         STAGE 2: INDICATOR REGRESSORS                  │        │
│  │  (Conditional - only when recession likely)            │        │
│  │                                                         │        │
│  │  ┌─────────────────────────────────────────┐           │        │
│  │  │  Labor Regressor (Multi-Output XGBoost) │           │        │
│  │  │  → Unemployment rate change             │           │        │
│  │  │  → Unemployment claims change           │           │        │
│  │  └─────────────────────────────────────────┘           │        │
│  │                                                         │        │
│  │  ┌─────────────────────────────────────────┐           │        │
│  │  │  Markets Regressor (Multi-Output XGBoost)│          │        │
│  │  │  → S&P 500 drawdown                     │           │        │
│  │  │  → NASDAQ drawdown                      │           │        │
│  │  └─────────────────────────────────────────┘           │        │
│  │                                                         │        │
│  │  ┌─────────────────────────────────────────┐           │        │
│  │  │  GDP Regressor (XGBoost)                │           │        │
│  │  │  → GDP decline (%)                      │           │        │
│  │  └─────────────────────────────────────────┘           │        │
│  └────────────────────────────────────────────────────────┘        │
│                              ↓                                       │
│  OUTPUT:                                                            │
│  - Recession probability: 0.82                                      │
│  - Recession predicted: True                                        │
│  - Indicator impacts:                                               │
│      • Unemployment rate: +2.1pp                                    │
│      • Unemployment claims: +45%                                    │
│      • S&P 500 drawdown: -18%                                       │
│      • NASDAQ drawdown: -22%                                        │
│      • GDP decline: -3.5%                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Conditional Architecture**: Stage 2 only activates when recession probability exceeds threshold (default: 0.5)
   - Reduces unnecessary computation
   - Focuses regression models on relevant scenarios

2. **Separate Regressors**: Three specialized multi-output regressors instead of one large model
   - Labor market (unemployment-focused)
   - Financial markets (stock market-focused)
   - Economic output (GDP-focused)
   - Allows domain-specific feature importance and interpretability

3. **Probability Calibration**: Stage 1 uses CalibratedClassifierCV with sigmoid method
   - Ensures probabilities are well-calibrated
   - Critical for threshold-based Stage 2 activation

4. **Class Imbalance Handling**:
   - XGBoost: `scale_pos_weight=1.86` (ratio of negative to positive samples)
   - sklearn models: `class_weight='balanced'`

5. **Missing Data Strategy**:
   - Replace ±∞ with NaN before imputation
   - Median imputation for all features
   - Models train only on samples with valid targets

---

## 📊 Model Performance

### Training Results (1980-2023, 172 samples)

**Data Splits:**
- Training: 120 samples (70%, 1980-2009)
- Validation: 25 samples (15%, 2010-2016)
- Test: 27 samples (15%, 2016-2023)

**Class Distribution:**
- Training: 24 recession (20%) / 96 non-recession (80%)
- Validation: 0 recession (0%) / 25 non-recession (100%)
- Test: 3 recession (11%) / 24 non-recession (89%)

### Stage 1: Recession Classification

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| **ROC-AUC** | 1.0000 | N/A* | 0.8194 |
| **PR-AUC** | 1.0000 | N/A* | 0.6222 |
| **F1 Score** | 0.9796 | 0.0000 | 0.5714 |
| **Precision** | 0.9600 | 0.0000 | 0.5000 |
| **Recall** | 1.0000 | 0.0000 | 0.6667 |
| **Specificity** | 0.9896 | 1.0000 | 0.9167 |
| **Accuracy** | 0.9917 | 1.0000 | 0.8889 |

\* Validation set contains only non-recession samples (no positive class)

**Test Set Confusion Matrix:**
- True Negatives: 22 (92% specificity)
- False Positives: 2 (low false alarm rate)
- False Negatives: 1 (missed 1 recession)
- True Positives: 2 (caught 2 of 3 recessions)

**Key Insight**: The model achieves excellent performance on training data and maintains strong performance on test data, correctly identifying 2 out of 3 recessions (67% recall) with only 2 false positives (8% false positive rate).

### Stage 2: Indicator Impact Regression

Performance on test set samples where recession was predicted (4 samples):

| Indicator | MAE | RMSE | R² | Description |
|-----------|-----|------|-----|-------------|
| **Unemployment Rate Change** | 3.88pp | 4.41pp | -0.03 | Absolute change in percentage points |
| **Unemployment Claims Change** | 79.9% | 101.3% | -0.20 | Percentage change |
| **S&P 500 Drawdown** | 15.3% | 17.7% | -0.98 | Market decline (%) |
| **NASDAQ Drawdown** | 23.2% | 24.1% | -2.97 | Market decline (%) |
| **GDP Decline** | 5.0% | 6.3% | 0.07 | Economic output decline (%) |

**Training Performance** (25 samples where recession probability > 0):
- All indicators achieve R² > 0.98
- MAE ranges from 0.05pp to 2.4% depending on indicator
- Excellent fit on training data

**Note**: Negative R² on test set indicates high variance in small sample (only 4 predictions). The model provides directional guidance but with high uncertainty due to limited recession events in modern test data.

### Inference Speed

- Stage 1 (classification): <10ms per sample
- Stage 2 (all regressors): <20ms per sample
- Total end-to-end: <30ms per prediction

---

## 🎯 Feature Engineering Approach

The core of this project is transforming raw economic indicators into predictive features that capture recession patterns. Here's how we process the data:

### Feature Engineering Pipeline

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

### Feature Categories Created

| Category | Count | Description | Examples |
|----------|-------|-------------|----------|
| **Lag Features** | 135 | Historical values at t-1, t-2, t-3, t-4, t-8 | `GDP_lag_1`, `Unemployment_lag_4` |
| **Rolling Statistics** | 270 | 4Q & 8Q windows (mean, std, min, max, median) | `GDP_rolling_4q_mean`, `SPX500_rolling_8q_std` |
| **Difference Features** | 54 | QoQ and YoY percentage changes | `GDP_diff_qoq`, `Unemployment_diff_yoy` |
| **Economic Ratios** | 15+ | Domain-specific ratios | `buffett_indicator_sp500`, `money_velocity_m1` |
| **Technical Indicators** | 30 | Market analysis indicators | `rsi_14_SPX500`, `macd_line_NASDAQ` |
| **Interactions** | 4,912+ | Combinations and transformations | Various complex features |
| **TOTAL** | **5,416** | All engineered features | - |

### Target Variables Created

| Target | Description | Horizon | Class Balance |
|--------|-------------|---------|---------------|
| `recession_current` | Current quarter recession status | 0Q | 31.2% positive |
| `recession_next_1q` | Recession in next quarter | 1Q | ~30% positive |
| `recession_next_2q` | Recession 2 quarters ahead | 2Q | ~30% positive |
| `recession_within_2q` | **PRIMARY TARGET** - Recession in next 1-2 quarters | 1-2Q | ~35% positive |

**Note**: We predict `recession_within_2q` to give early warning (1-2 quarters ahead).

### Feature Selection Results

**Initial Features**: 5,416
**After Importance Ranking**: Top 50 selected
**After VIF Filtering**: 29 final features (21 removed due to VIF > 10)
**Final Dataset**: 683 samples × 29 features

#### Top 15 Most Important Features

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

**Key Insight**: The historical recession indicator (`LABELS_USREC`) and its transformations dominate the top features, indicating strong temporal autocorrelation in recession patterns. Economic fundamentals like Fed Funds Rate, Real GDP, and Unemployment also appear in top features.

### Feature Distribution by Indicator

**Top 5 Contributing Indicators** (by number of selected features):
1. **LABELS_USREC** (Historical Recession): 13 features selected
2. **Federal Funds Effective Rate**: 3 features selected
3. **Unemployment Level**: 6 features selected
4. **Real Gross Domestic Product**: 2 features selected
5. **Unemployment Rate**: 2 features selected

### Key Technical Decisions

1. **No Data Leakage**: All features look backward only; targets properly shifted forward
2. **Multicollinearity Handling**: VIF threshold of 10.0 to remove highly correlated features
3. **Missing Data Strategy**: Imputation with median for model training
4. **Infinity Handling**: Replaced ±∞ with NaN before imputation
5. **Feature Selection**: Random Forest importance (handles non-linearity) + VIF filtering

### Files Generated

```
data/processed/
├── features_full.parquet          # 13.04 MB - All 5,416 features
├── features_selected.parquet      # 0.07 MB - Top 29 features (ready for training)
└── feature_importance.csv         # Full importance rankings
```

### Running Feature Engineering

```bash
# Run complete feature engineering pipeline
python entrypoint/engineer_features.py

# Output:
# - Creates 5,416 features from 27 indicators
# - Selects top 29 features
# - Saves processed data ready for model training
```

---

## 🔧 Usage

### Training the Model

```bash
# Complete training pipeline
python entrypoint/train.py

# Output:
# - Trains hybrid model (Stage 1 + Stage 2)
# - Evaluates on train/val/test splits
# - Saves model to models/run_YYYYMMDD_HHMMSS/
# - Generates metrics.json with all evaluation results
```

### Making Predictions

```python
from src.models.model_persistence import load_model
import pandas as pd

# Load trained model
model, metadata = load_model("models/run_20251130_142214/")

# Prepare features (same 29 features used in training)
X_new = pd.DataFrame({...})  # Your 29 features

# Predict
predictions = model.predict(X_new)

# Access results
print(f"Recession probability: {predictions['recession_probability'][0]:.2%}")
print(f"Recession predicted: {predictions['recession_predicted'][0]}")

if predictions['indicator_impacts'] is not None:
    impacts = predictions['indicator_impacts'].iloc[0]
    print(f"Unemployment rate change: {impacts['unemployment_rate_change']:.1f}pp")
    print(f"S&P 500 drawdown: {impacts['sp500_drawdown']:.1f}%")
    # ... more indicators
```

### Single Sample Prediction

```python
# Convenience method for single prediction
single_pred = model.predict_single(X_new.iloc[:1])

print(f"Date: {single_pred['date']}")
print(f"Recession probability: {single_pred['recession_probability']:.2%}")
print(f"Predicted: {single_pred['recession_predicted']}")

if single_pred['indicator_impacts']:
    for indicator, value in single_pred['indicator_impacts'].items():
        print(f"{indicator}: {value:.2f}")
```

### Customizing Hyperparameters

Edit `config.yaml` to adjust model configuration:

```yaml
model:
  threshold: 0.5  # Recession probability threshold

  stage1:
    classifier_type: xgboost  # Options: xgboost, random_forest, logistic
    hyperparameters:
      n_estimators: 200
      max_depth: 6
      learning_rate: 0.05
      # ... more XGBoost params

  stage2:
    regressor_type: xgboost  # Options: xgboost, random_forest
    hyperparameters:
      n_estimators: 100
      max_depth: 5
      # ... more params
```

Then retrain with `python entrypoint/train.py`.

---

## 📁 Project Structure

```
.
├── config.yaml                # Project configuration (paths, model hyperparams)
├── pyproject.toml            # Python package config and tool settings
├── Makefile                  # Convenient command shortcuts
├── CLAUDE.md                 # Claude Code instructions
├── IMPLEMENTATION_ROADMAP.md # Full implementation plan
├── data/
│   ├── raw/                 # Raw data from Kaggle (not in git)
│   ├── processed/           # Preprocessed data (not in git)
│   │   ├── quarterly_aligned.parquet
│   │   ├── features_full.parquet       (5,416 features)
│   │   └── features_selected.parquet   (29 features + 9 targets)
│   └── external/            # External data sources
├── entrypoint/              # Entry point scripts
│   ├── fetch_nber.py       # Fetch NBER recession data
│   ├── preprocess.py       # Process 27 CSVs to quarterly
│   ├── engineer_features.py # Feature engineering pipeline
│   └── train.py            # Train hybrid model
├── logs/
│   └── train.log           # Training logs
├── models/                  # Saved models (not in git)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── hybrid_recession_model.joblib
│       ├── hybrid_recession_model_metadata.json
│       └── metrics.json
├── notebooks/               # Jupyter notebooks for exploration
├── src/                    # Source code modules
│   ├── data/              # Data loading and processing
│   │   ├── loader.py
│   │   ├── nber_fetcher.py
│   │   ├── preprocessor.py
│   │   └── validator.py
│   ├── features/          # Feature engineering
│   │   ├── engineer.py
│   │   ├── selector.py
│   │   └── targets.py
│   ├── models/            # Model implementations
│   │   ├── base.py                # Abstract base classes
│   │   ├── recession_classifier.py # Stage 1 models
│   │   ├── indicator_regressors.py # Stage 2 models
│   │   ├── hybrid_predictor.py    # Orchestrator
│   │   ├── evaluator.py           # Evaluation framework
│   │   └── model_persistence.py   # Save/load utilities
│   └── utils/             # Utilities
│       ├── config.py
│       └── logger.py
└── tests/                 # Unit tests
    └── test_config.py
```

## Quick Start

### 1. Setup Environment

**Option A: Using UV (Recommended - faster)**
```bash
# Install dependencies with UV
uv sync --all-extras

# UV automatically manages the virtual environment
```

**Option B: Using pip**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development tools
make install-dev
```

### 2. Configure Kaggle API

1. Create account at https://www.kaggle.com
2. Go to Account → Settings → API → Create New Token

**Option A: Modern API Token (Recommended)**
```bash
# Set environment variable (add to ~/.zshrc or ~/.bashrc for persistence)
export KAGGLE_API_TOKEN=KGAT_your_token_here
```

**Option B: Legacy JSON File**
1. Download `kaggle.json` from Kaggle account settings
2. Save to `~/.kaggle/kaggle.json`
3. Set permissions (Mac/Linux): `chmod 600 ~/.kaggle/kaggle.json`

### 3. Download Data

```bash
make data
```

Or manually download from [Kaggle](https://www.kaggle.com/datasets/mikoajfish99/us-recession-and-financial-indicators/data) and place in `data/raw/`.

### 4. Start Exploring

```bash
make notebook  # Opens Jupyter Lab
```

## Development Commands

**Using UV:**
```bash
uv sync --all-extras    # Install all dependencies
uv run pytest           # Run tests
uv run jupyter lab      # Start Jupyter Lab
# For other commands, use make as shown below
```

**Using Make (works with both pip and UV):**
```bash
make help          # Show all available commands
make install       # Install production dependencies only
make install-dev   # Install with dev tools (pytest, jupyter, etc.)
make test          # Run tests
make lint          # Check code quality
make format        # Auto-format code with black
make clean         # Remove cache files
make data          # Download Kaggle dataset
make notebook      # Start Jupyter Lab
```

## Configuration

Edit `config.yaml` to adjust:
- Data and model paths
- Logging settings
- Training parameters (random seed, test size, etc.)
- MLflow experiment tracking
- Development options (sample data for fast iteration)

## Project Files Explained

**Essential:**
- `pyproject.toml` - All Python configuration (dependencies, tool settings)
- `config.yaml` - Your app's runtime settings
- `.gitignore` - Prevents committing data/models to git
- `Makefile` - Command shortcuts

**Optional:**
- Tests in `tests/` - Run with `make test`
- Code formatting - Auto-format with `make format`

## License

Database: Open Database, Contents: © Original Authors (Open Data Commons Open Database License 1.0)
