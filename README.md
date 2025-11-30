# US Recession Indicators Risk Model

Machine learning system for predicting US recessions 1-2 quarters ahead using 170+ years of economic data (1854-2025).

## Overview

This project builds a recession prediction model by analyzing 27 economic indicators spanning over 170 years. The system processes raw economic data through a sophisticated feature engineering pipeline, creating over 5,000 predictive features before selecting the most informative ones for model training.

**Goal**: Predict whether a recession will occur in the next 1-2 quarters (`recession_within_2q` target)

**Dataset**: [US Recession and Financial Indicators](https://www.kaggle.com/datasets/mikoajfish99/us-recession-and-financial-indicators/data) from Kaggle
- 27 economic indicators (Federal funds rate, GDP, unemployment, credit, money supply, real estate, stock markets)
- 685 quarterly observations (1854-2025)
- 214 recession quarters (31.2% of data)

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

## Project Structure

```
.
├── config.yaml         # Project configuration (paths, parameters)
├── pyproject.toml      # Python package config and all tool settings
├── Makefile            # Convenient command shortcuts
├── data/
│   ├── raw/           # Raw data from Kaggle (not in git)
│   ├── processed/     # Preprocessed data (not in git)
│   └── external/      # External data sources (not in git)
├── entrypoint/        # Main scripts (train, evaluate, predict)
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Source code modules
│   ├── data/         # Data loading and processing
│   ├── features/     # Feature engineering
│   ├── models/       # Model definitions
│   └── utils/        # Utilities (config, logging)
└── tests/            # Unit tests
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
