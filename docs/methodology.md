# Methodology

**Hybrid Two-Stage Recession Prediction System**

This document provides detailed technical methodology for the recession prediction system, including mathematical formulations, model architecture, and design decisions.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Processing](#data-processing)
3. [Feature Engineering](#feature-engineering)
4. [Target Variable Construction](#target-variable-construction)
5. [Model Architecture](#model-architecture)
6. [Training Procedure](#training-procedure)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Design Decisions & Rationale](#design-decisions--rationale)

---

## 1. System Overview

### 1.1 Problem Formulation

The system addresses two related prediction tasks:

**Task 1 (Stage 1)**: Binary classification
- **Input**: Economic indicators at time $t$
- **Output**: Probability that recession occurs in next 1-2 quarters
- **Target**: $y_1 \in \{0, 1\}$ where 1 indicates recession within 2 quarters

**Task 2 (Stage 2)**: Multi-output regression (conditional on Task 1)
- **Input**: Economic indicators at time $t$ + recession probability from Stage 1
- **Output**: Predicted changes in 5 economic indicators from $t$ to $t+2$
- **Targets**: $\mathbf{y}_2 = [y_{\text{unemp\_rate}}, y_{\text{unemp\_claims}}, y_{\text{sp500}}, y_{\text{nasdaq}}, y_{\text{gdp}}]$

### 1.2 Architectural Overview

```
Hybrid System: h(x) = (h₁(x), h₂(x, h₁(x)))

where:
- h₁: Stage 1 classifier (recession probability)
- h₂: Stage 2 regressors (indicator impacts)
- h₂ activates only if h₁(x) ≥ τ (threshold)
```

**Key Innovation**: Conditional architecture reduces unnecessary computation and focuses regression models on recession scenarios only.

---

## 2. Data Processing

### 2.1 Frequency Alignment

**Problem**: 27 economic indicators have different frequencies:
- Daily: Stock prices
- Weekly: Unemployment claims
- Monthly: Most indicators
- Quarterly: GDP
- Annual: Some financial indicators

**Solution**: Resample all to quarterly frequency using:

```python
# Upsampling (daily/weekly/monthly → quarterly)
x_q = resample(x, freq='Q', method='mean')  # Average over quarter

# Downsampling (annual → quarterly)
x_q = resample(x, freq='Q', method='forward_fill')  # Carry forward
```

**Rationale**:
- Quarterly aligns with business cycle analysis
- Reduces noise while preserving trends
- Matches typical economic reporting cycles

### 2.2 Missing Data Handling

**Approach**: Multi-stage imputation

1. **Forward-fill**: Up to 1 quarter
   ```python
   x_filled = x.fillna(method='ffill', limit=1)
   ```

2. **Median imputation**: For model training
   ```python
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(strategy='median')
   X_imputed = imputer.fit_transform(X)
   ```

3. **Infinity handling**: Replace ±∞ with NaN before imputation
   ```python
   X_clean = X.replace([np.inf, -np.inf], np.nan)
   ```

**Rationale**:
- Forward-fill preserves recent values (realistic for economic data)
- Median robust to outliers
- XGBoost handles NaN natively, but ±∞ causes errors

---

## 3. Feature Engineering

### 3.1 Feature Categories

#### A. Lag Features
Capture historical values:

$$
f_{\text{lag}}^{(k)}(x_t) = x_{t-k}, \quad k \in \{1, 2, 3, 4, 8\}
$$

**Example**: `GDP_lag_4` = GDP value 4 quarters ago

**Count**: 27 indicators × 5 lags = 135 features

#### B. Rolling Statistics
Capture recent trends over windows $w \in \{4, 8\}$ quarters:

$$
\begin{aligned}
f_{\text{mean}}^{(w)}(x_t) &= \frac{1}{w}\sum_{i=0}^{w-1} x_{t-i} \\
f_{\text{std}}^{(w)}(x_t) &= \sqrt{\frac{1}{w}\sum_{i=0}^{w-1} (x_{t-i} - \bar{x})^2} \\
f_{\text{min}}^{(w)}(x_t) &= \min_{i=0}^{w-1} x_{t-i} \\
f_{\text{max}}^{(w)}(x_t) &= \max_{i=0}^{w-1} x_{t-i} \\
f_{\text{median}}^{(w)}(x_t) &= \text{median}_{i=0}^{w-1}(x_{t-i})
\end{aligned}
$$

**Count**: 27 indicators × 2 windows × 5 statistics = 270 features

#### C. Difference Features
Capture changes:

$$
\begin{aligned}
f_{\text{QoQ}}(x_t) &= \frac{x_t - x_{t-1}}{x_{t-1}} \times 100 \quad \text{(Quarter-over-Quarter)} \\
f_{\text{YoY}}(x_t) &= \frac{x_t - x_{t-4}}{x_{t-4}} \times 100 \quad \text{(Year-over-Year)}
\end{aligned}
$$

**Count**: 27 indicators × 2 changes = 54 features

#### D. Economic Ratios
Domain-specific indicators:

$$
\begin{aligned}
\text{Buffett Indicator} &= \frac{\text{Market Cap (SPX500)}}{\text{GDP}} \times 100 \\
\text{Money Velocity (M1)} &= \frac{\text{GDP}}{\text{M1 Money Supply}} \\
\text{Credit/GDP Ratio} &= \frac{\text{Total Credit}}{\text{GDP}} \times 100 \\
\text{Yield Curve Slope} &= \text{10Y Treasury} - \text{Fed Funds Rate}
\end{aligned}
$$

**Count**: 15+ features

#### E. Technical Indicators (for market data)
$$
\begin{aligned}
\text{RSI}_{14} &= 100 - \frac{100}{1 + RS_{14}} \\
\text{MACD} &= \text{EMA}_{12} - \text{EMA}_{26} \\
\text{Bollinger Bands} &= \text{SMA}_{20} \pm 2\sigma_{20}
\end{aligned}
$$

**Count**: 30 features

#### F. Interaction Features
Combinations and transformations:
- Cross-products of related indicators
- Ratio of indicators
- Polynomial features

**Count**: 4,912+ features

### 3.2 Feature Selection

**Process**:

1. **Train Random Forest** on all 5,416 features
   ```python
   rf = RandomForestClassifier(n_estimators=100, random_state=42)
   rf.fit(X_all_features, y_recession)
   importances = rf.feature_importances_
   ```

2. **Rank by importance** and select top 50

3. **VIF (Variance Inflation Factor) filtering**
   Remove multicollinear features with VIF > 10:

   $$
   \text{VIF}_j = \frac{1}{1 - R_j^2}
   $$

   where $R_j^2$ is from regressing feature $j$ on all other features.

4. **Iterative removal**: Remove highest VIF feature, recompute, repeat until all VIF < 10

**Final Result**: 29 selected features

---

## 4. Target Variable Construction

### 4.1 Classification Targets

Four binary targets created from NBER recession indicator:

$$
\begin{aligned}
y_{\text{current}} &= r_t \\
y_{\text{next\_1q}} &= r_{t+1} \\
y_{\text{next\_2q}} &= r_{t+2} \\
y_{\text{within\_2q}} &= \max(r_{t+1}, r_{t+2}) \quad \text{(Primary Target)}
\end{aligned}
$$

where $r_t \in \{0, 1\}$ is the NBER recession indicator at quarter $t$.

**Primary Target**: $y_{\text{within\_2q}}$ provides early warning 1-2 quarters ahead.

### 4.2 Regression Targets

Five targets measuring economic impacts from $t$ to $t+2$:

1. **Unemployment Rate Change** (absolute):
   $$
   y_{\text{unemp\_rate}} = \text{UnempRate}_{t+2} - \text{UnempRate}_t \quad \text{(percentage points)}
   $$

2. **Unemployment Claims Change** (percentage):
   $$
   y_{\text{unemp\_claims}} = \frac{\text{Claims}_{t+2} - \text{Claims}_t}{\text{Claims}_t} \times 100
   $$

3. **S&P 500 Drawdown** (percentage):
   $$
   y_{\text{sp500}} = \frac{\text{SPX}_{t+2} - \text{SPX}_t}{\text{SPX}_t} \times 100
   $$

4. **NASDAQ Drawdown** (percentage):
   $$
   y_{\text{nasdaq}} = \frac{\text{NASDAQ}_{t+2} - \text{NASDAQ}_t}{\text{NASDAQ}_t} \times 100
   $$

5. **GDP Decline** (percentage):
   $$
   y_{\text{gdp}} = \frac{\text{GDP}_{t+2} - \text{GDP}_t}{\text{GDP}_t} \times 100
   $$

**Note**: All targets look forward 2 quarters (matching classification horizon).

---

## 5. Model Architecture

### 5.1 Stage 1: Recession Classifier

**Primary Model**: XGBoost with Probability Calibration

```python
# Base XGBoost
model_base = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.86,  # Class imbalance handling
    random_state=42
)

# Calibrated probabilities
model = CalibratedClassifierCV(
    estimator=model_base,
    cv=3,
    method='sigmoid'
)
```

**Output**: Calibrated probability $p = P(y_1 = 1 | \mathbf{x})$

**Class Imbalance Handling**:
$$
\text{scale\_pos\_weight} = \frac{n_{\text{negative}}}{n_{\text{positive}}} = \frac{145}{78} \approx 1.86
$$

**Alternatives Tested**:
- Random Forest (ROC-AUC: 0.889)
- Logistic Regression (ROC-AUC: 0.813)

### 5.2 Stage 2: Indicator Regressors

**Architecture**: Three specialized multi-output regressors

#### A. Labor Regressor
Predicts unemployment rate + claims (2 outputs):

```python
base = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
model = MultiOutputRegressor(base)
```

**Input**: 29 features + recession probability from Stage 1

**Output**: $[\Delta\text{UnempRate}, \Delta\text{Claims}]$

#### B. Markets Regressor
Predicts S&P 500 + NASDAQ drawdowns (2 outputs):

Same architecture as Labor Regressor.

**Output**: $[\text{SPX Drawdown}, \text{NASDAQ Drawdown}]$

#### C. GDP Regressor
Predicts GDP decline (1 output):

```python
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
```

**Output**: $\text{GDP Decline}$

**Why Separate Regressors?**
- Domain-specific interpretability
- Different scales and distributions
- Allows independent hyperparameter tuning

### 5.3 Hybrid Orchestrator

**Algorithm**:

```
Input: Features x, Threshold τ = 0.5
Output: (p, y_pred, impacts)

1. p ← h₁(x)  # Stage 1 prediction
2. y_pred ← 1 if p ≥ τ else 0
3. if p ≥ τ:
     impacts ← h₂(x, p)  # Stage 2 prediction
   else:
     impacts ← None
4. return (p, y_pred, impacts)
```

---

## 6. Training Procedure

### 6.1 Data Filtering

Original data: 685 samples (1854-2025)

**Filter**: Keep only samples with complete regression targets (all 5 indicators non-NaN)

- S&P 500 data starts: 1979
- NASDAQ data starts: 1980
- Labor/GDP data starts: ~1947

**Filtered data**: 172 samples (1980-2023)

### 6.2 Temporal Train/Val/Test Split

**Critical**: No shuffling (prevents temporal leakage)

```
Train: 70% (120 samples, 1980-2009)
Val:   15% (25 samples, 2010-2016)
Test:  15% (27 samples, 2016-2023)
```

**Class Distribution**:
- Train: 24 recession / 96 non-recession (20% positive)
- Val: 0 recession / 25 non-recession (0% positive)
- Test: 3 recession / 24 non-recession (11% positive)

### 6.3 Training Sequence

1. **Train Stage 1** on $(X_{\text{train}}, y_{\text{class, train}})$
2. **Get recession probabilities**: $p_{\text{train}} = h_1(X_{\text{train}})$
3. **Train Stage 2 Labor** on $(X_{\text{train}}, y_{\text{labor, train}}, p_{\text{train}})$
4. **Train Stage 2 Markets** on $(X_{\text{train}}, y_{\text{markets, train}}, p_{\text{train}})$
5. **Train Stage 2 GDP** on $(X_{\text{train}}, y_{\text{gdp, train}}, p_{\text{train}})$

**Note**: Stage 2 models train only on samples with valid targets (non-NaN), typically all 120 training samples.

---

## 7. Evaluation Metrics

### 7.1 Stage 1 (Classification)

**Primary Metric**: ROC-AUC (area under ROC curve)

**Additional Metrics**:
- **PR-AUC**: Precision-Recall AUC (important for imbalanced data)
- **F1 Score**: $F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ (% of recessions caught)
- **Specificity**: $\frac{TN}{TN + FP}$ (% of non-recessions correctly identified)
- **Precision**: $\frac{TP}{TP + FP}$ (% of recession predictions correct)

**Confusion Matrix**:
```
              Predicted
              0      1
Actual  0    TN     FP
        1    FN     TP
```

### 7.2 Stage 2 (Regression)

**Metrics per indicator**:

1. **MAE (Mean Absolute Error)**:
   $$
   \text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|
   $$

2. **RMSE (Root Mean Squared Error)**:
   $$
   \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}
   $$

3. **R² (Coefficient of Determination)**:
   $$
   R^2 = 1 - \frac{\sum_{i=1}^n(y_i - \hat{y}_i)^2}{\sum_{i=1}^n(y_i - \bar{y})^2}
   $$

**Interpretation**:
- MAE: Average error in same units as target
- RMSE: Penalizes large errors more heavily
- R²: Proportion of variance explained (1.0 = perfect, 0 = no better than mean)

---

## 8. Design Decisions & Rationale

### 8.1 Why Hybrid Architecture?

**Alternative**: Single-stage multi-task learning (predict recession + impacts jointly)

**Chosen**: Two-stage with conditional Stage 2

**Rationale**:
1. **Interpretability**: Separate models easier to understand and debug
2. **Efficiency**: Stage 2 only runs when needed (~20-30% of time)
3. **Modularity**: Can improve/replace stages independently
4. **Economic Logic**: Impact predictions only meaningful during recessions

### 8.2 Why XGBoost?

**Alternatives**: Random Forest, Neural Networks, Logistic Regression

**Advantages of XGBoost**:
1. **Handles missing data natively**: No imputation needed during inference
2. **Regularization**: L1/L2 prevent overfitting
3. **Tree-based**: Captures non-linear relationships
4. **Robust to outliers**: Tree splits less affected than linear models
5. **Fast inference**: <10ms per prediction
6. **Feature importance**: Built-in via gain/weight metrics

**Test Results**:
- XGBoost ROC-AUC: 0.819
- Random Forest ROC-AUC: 0.889 (but lower recall: 33% vs 67%)
- Logistic Regression ROC-AUC: 0.813

**Decision**: XGBoost selected for better recall (catches more recessions), even with slightly lower ROC-AUC.

### 8.3 Why Median Imputation?

**Alternatives**: Mean, forward-fill, KNN, iterative

**Rationale**:
1. **Robust to outliers**: Economic data has extreme values during crises
2. **Simple and fast**: No training needed
3. **Preserves scale**: Unlike normalization
4. **Works with XGBoost**: Complements native NaN handling

### 8.4 Why Temporal Splits (Not K-Fold)?

**K-Fold Cross-Validation**: Randomly split data into K folds

**Problem**: Temporal leakage - training on future to predict past

**Solution**: Temporal split
- Train on oldest data
- Validate on middle period
- Test on most recent data

**Rationale**: Mimics real-world usage (predict future from past).

### 8.5 Why 29 Features (Not More)?

**Trade-off**: More features → More information but also:
- Increased overfitting risk
- Slower training/inference
- Multicollinearity issues
- Harder to interpret

**Selection Process**:
- Started with 5,416 engineered features
- Random Forest importance → Top 50
- VIF filtering → Remove collinear (VIF > 10)
- Final: 29 features

**Result**: Excellent performance (ROC-AUC 0.82) with compact model.

### 8.6 Why 1-2 Quarter Horizon?

**Alternatives**: Current quarter, 1Q ahead, 4Q ahead

**Chosen**: 1-2 quarters ahead (`recession_within_2q`)

**Rationale**:
1. **Actionable lead time**: 3-6 months allows policy response
2. **Prediction accuracy**: Longer horizons too uncertain
3. **Economic relevance**: Matches typical planning cycles
4. **Data quality**: 2-quarter lag available for most indicators

---

## Appendix: Hyperparameters

### Stage 1 (XGBoost Classifier)
```yaml
n_estimators: 200
max_depth: 6
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
scale_pos_weight: 1.86
random_state: 42
```

### Stage 2 (XGBoost Regressors)
```yaml
n_estimators: 100
max_depth: 5
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
random_state: 42
```

### Model Selection
- Threshold: 0.5 (probability threshold for Stage 2 activation)
- Calibration: Sigmoid (CalibratedClassifierCV)
- Validation: 3-fold CV for calibration

---

**Last Updated**: December 1, 2025
