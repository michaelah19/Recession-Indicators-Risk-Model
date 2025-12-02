# Data Dictionary

Complete reference for all economic indicators in the US Recession and Financial Indicators dataset.

## Overview

**Dataset Source**: [Kaggle - US Recession and Financial Indicators](https://www.kaggle.com/datasets/mikoajfish99/us-recession-and-financial-indicators/data)

**Primary Data Source**: Federal Reserve Economic Data (FRED)

**Time Period**: 1967-01-01 to 2024-03-01 (varies by indicator)

**Total Indicators**: 27 economic and financial time series

---

## Target Variable

### USREC
- **Full Name**: NBER-Based Recession Indicator
- **Description**: Binary indicator where 1 = US economy in recession, 0 = not in recession
- **Source**: Federal Reserve Bank of St. Louis (FRED)
- **Frequency**: Monthly
- **Units**: Binary (0 or 1)
- **Date Range**: 1967-01 to 2024-03
- **Recession Periods**: Marks official NBER recession dates
- **Use in Model**: Primary target variable for Stage 1 classifier

---

## Interest Rates and Monetary Policy

### FEDFUNDS
- **Full Name**: Effective Federal Funds Rate
- **Description**: Interest rate at which depository institutions lend reserve balances overnight
- **Source**: Board of Governors of the Federal Reserve System (FRED)
- **Frequency**: Monthly
- **Units**: Percent per annum
- **Date Range**: 1954-07 to 2024-03
- **Relevance**: Central bank policy tool; inversions and rapid changes signal recession risk

### GS10
- **Full Name**: 10-Year Treasury Constant Maturity Rate
- **Description**: Yield on US Treasury securities at 10-year constant maturity
- **Source**: Board of Governors of the Federal Reserve System (FRED)
- **Frequency**: Monthly
- **Units**: Percent per annum
- **Date Range**: 1953-04 to 2024-03
- **Relevance**: Long-term interest rate benchmark; used in yield curve analysis

### GS3M
- **Full Name**: 3-Month Treasury Constant Maturity Rate
- **Description**: Yield on US Treasury securities at 3-month constant maturity
- **Source**: Board of Governors of the Federal Reserve System (FRED)
- **Frequency**: Monthly
- **Units**: Percent per annum
- **Date Range**: 1982-01 to 2024-03
- **Relevance**: Short-term interest rate; used to calculate yield curve spread (GS10 - GS3M)

### T10Y3M
- **Full Name**: 10-Year Treasury Minus 3-Month Treasury (Yield Spread)
- **Description**: Difference between 10-year and 3-month Treasury yields
- **Source**: Federal Reserve Bank of St. Louis (FRED)
- **Frequency**: Daily (aggregated to monthly)
- **Units**: Percentage points
- **Date Range**: 1982-01 to 2024-03
- **Relevance**: **Strongest recession predictor**; inversions (negative values) historically precede recessions

---

## Labor Market Indicators

### UNRATE
- **Full Name**: Unemployment Rate
- **Description**: Percentage of labor force that is jobless and actively seeking employment
- **Source**: U.S. Bureau of Labor Statistics (FRED)
- **Frequency**: Monthly
- **Units**: Percent
- **Date Range**: 1948-01 to 2024-03
- **Relevance**: Key recession indicator; rapid increases signal economic contraction

### PAYEMS
- **Full Name**: All Employees: Total Nonfarm Payrolls
- **Description**: Total number of paid employees in nonfarm establishments
- **Source**: U.S. Bureau of Labor Statistics (FRED)
- **Frequency**: Monthly
- **Units**: Thousands of persons
- **Date Range**: 1939-01 to 2024-03
- **Relevance**: Employment growth metric; declines signal weakening economy

### ICSA
- **Full Name**: Initial Claims for Unemployment Insurance
- **Description**: Number of new unemployment insurance claims filed weekly
- **Source**: U.S. Department of Labor (FRED)
- **Frequency**: Weekly (aggregated to monthly)
- **Units**: Thousands
- **Date Range**: 1967-01 to 2024-03
- **Relevance**: Leading indicator; spikes indicate labor market deterioration

---

## Economic Output and Growth

### GDP
- **Full Name**: Gross Domestic Product
- **Description**: Total market value of all finished goods and services produced
- **Source**: U.S. Bureau of Economic Analysis (FRED)
- **Frequency**: Quarterly
- **Units**: Billions of dollars (seasonally adjusted annual rate)
- **Date Range**: 1947-Q1 to 2024-Q1
- **Relevance**: Primary measure of economic health; two consecutive quarters of decline define technical recession

### INDPRO
- **Full Name**: Industrial Production Index
- **Description**: Measures real output of manufacturing, mining, and utilities
- **Source**: Board of Governors of the Federal Reserve System (FRED)
- **Frequency**: Monthly
- **Units**: Index (2017 = 100)
- **Date Range**: 1919-01 to 2024-03
- **Relevance**: Leading indicator of economic activity; declines signal manufacturing weakness

### HOUST
- **Full Name**: Housing Starts: Total New Privately Owned Housing Units
- **Description**: Number of new residential construction projects started
- **Source**: U.S. Census Bureau (FRED)
- **Frequency**: Monthly
- **Units**: Thousands of units (seasonally adjusted annual rate)
- **Date Range**: 1959-01 to 2024-03
- **Relevance**: Leading indicator; housing market weakness often precedes broader recession

### PERMIT
- **Full Name**: New Privately-Owned Housing Units Authorized (Building Permits)
- **Description**: Number of building permits issued for new housing construction
- **Source**: U.S. Census Bureau (FRED)
- **Frequency**: Monthly
- **Units**: Thousands of units (seasonally adjusted annual rate)
- **Date Range**: 1960-01 to 2024-03
- **Relevance**: Forward-looking housing indicator; declines signal future construction slowdown

---

## Consumer Indicators

### UMCSENT
- **Full Name**: University of Michigan Consumer Sentiment Index
- **Description**: Survey-based measure of consumer attitudes and expectations
- **Source**: University of Michigan (FRED)
- **Frequency**: Monthly
- **Units**: Index (1966:Q1 = 100)
- **Date Range**: 1978-01 to 2024-03
- **Relevance**: Forward-looking indicator; declining confidence often precedes reduced spending

### RETAILSL
- **Full Name**: Advance Retail Sales: Retail Trade
- **Description**: Total sales at retail and food service stores
- **Source**: U.S. Census Bureau (FRED)
- **Frequency**: Monthly
- **Units**: Millions of dollars (seasonally adjusted)
- **Date Range**: 1992-01 to 2024-03
- **Relevance**: Measures consumer spending (70% of GDP); declines indicate demand weakness

### PCE
- **Full Name**: Personal Consumption Expenditures
- **Description**: Total value of goods and services consumed by households
- **Source**: U.S. Bureau of Economic Analysis (FRED)
- **Frequency**: Monthly
- **Units**: Billions of dollars (seasonally adjusted annual rate)
- **Date Range**: 1959-01 to 2024-03
- **Relevance**: Broader measure of consumer spending than retail sales

---

## Credit and Money Supply

### TOTALSL
- **Full Name**: Total Consumer Credit Outstanding
- **Description**: Total credit extended to consumers for household, family, and personal expenses
- **Source**: Board of Governors of the Federal Reserve System (FRED)
- **Frequency**: Monthly
- **Units**: Billions of dollars (seasonally adjusted)
- **Date Range**: 1943-01 to 2024-03
- **Relevance**: Indicates borrowing capacity and consumer leverage; sharp changes signal credit conditions

### M2
- **Full Name**: M2 Money Stock
- **Description**: Sum of currency, demand deposits, savings deposits, and small time deposits
- **Source**: Board of Governors of the Federal Reserve System (FRED)
- **Frequency**: Monthly
- **Units**: Billions of dollars (seasonally adjusted)
- **Date Range**: 1959-01 to 2024-03
- **Relevance**: Broad money supply measure; contractions can signal tight monetary policy

### BUSLOANS
- **Full Name**: Commercial and Industrial Loans at All Commercial Banks
- **Description**: Total outstanding business loans from commercial banks
- **Source**: Board of Governors of the Federal Reserve System (FRED)
- **Frequency**: Monthly
- **Units**: Billions of dollars (seasonally adjusted)
- **Date Range**: 1973-01 to 2024-03
- **Relevance**: Business credit availability; declines indicate tightening credit conditions

### REALLN
- **Full Name**: Real Estate Loans at All Commercial Banks
- **Description**: Total outstanding real estate loans from commercial banks
- **Source**: Board of Governors of the Federal Reserve System (FRED)
- **Frequency**: Monthly
- **Units**: Billions of dollars (seasonally adjusted)
- **Date Range**: 1973-01 to 2024-03
- **Relevance**: Housing market credit; sharp declines preceded 2008 financial crisis

---

## Financial Markets

### SP500
- **Full Name**: S&P 500 Index
- **Description**: Stock market index tracking 500 large US companies
- **Source**: Standard & Poor's (FRED)
- **Frequency**: Daily (aggregated to monthly)
- **Units**: Index points
- **Date Range**: 1927-12 to 2024-03
- **Relevance**: Leading indicator; market declines often precede economic downturns

### NASDAQCOM
- **Full Name**: NASDAQ Composite Index
- **Description**: Stock market index of all NASDAQ-listed stocks
- **Source**: NASDAQ OMX Group (FRED)
- **Frequency**: Daily (aggregated to monthly)
- **Units**: Index points
- **Date Range**: 1971-02 to 2024-03
- **Relevance**: Tech-heavy index; captures innovation sector performance

### VIXCLS
- **Full Name**: CBOE Volatility Index (VIX)
- **Description**: Measure of market expectations of near-term volatility
- **Source**: Chicago Board Options Exchange (FRED)
- **Frequency**: Daily (aggregated to monthly)
- **Units**: Index points
- **Date Range**: 1990-01 to 2024-03
- **Relevance**: "Fear gauge"; spikes indicate market stress and recession risk

### DEXUSEU
- **Full Name**: US / Euro Foreign Exchange Rate
- **Description**: US dollars per euro
- **Source**: Board of Governors of the Federal Reserve System (FRED)
- **Frequency**: Daily (aggregated to monthly)
- **Units**: US dollars per 1 euro
- **Date Range**: 1999-01 to 2024-03
- **Relevance**: Exchange rate indicator; dollar strength/weakness affects trade and competitiveness

---

## Inflation Indicators

### CPIAUCSL
- **Full Name**: Consumer Price Index for All Urban Consumers: All Items
- **Description**: Measure of average change in prices paid by urban consumers
- **Source**: U.S. Bureau of Labor Statistics (FRED)
- **Frequency**: Monthly
- **Units**: Index (1982-84 = 100)
- **Date Range**: 1947-01 to 2024-03
- **Relevance**: Primary inflation measure; rapid increases trigger Fed tightening

### PCEPI
- **Full Name**: Personal Consumption Expenditures Price Index
- **Description**: Measure of prices paid by consumers for goods and services
- **Source**: U.S. Bureau of Economic Analysis (FRED)
- **Frequency**: Monthly
- **Units**: Index (2017 = 100)
- **Date Range**: 1959-01 to 2024-03
- **Relevance**: Fed's preferred inflation gauge; influences monetary policy decisions

### PPIFGS
- **Full Name**: Producer Price Index for Finished Goods
- **Description**: Measure of average change in selling prices by domestic producers
- **Source**: U.S. Bureau of Labor Statistics (FRED)
- **Frequency**: Monthly
- **Units**: Index (1982 = 100)
- **Date Range**: 1913-01 to 2024-03
- **Relevance**: Leading inflation indicator; producer price changes precede consumer prices

---

## Oil and Commodities

### DCOILWTICO
- **Full Name**: Crude Oil Prices: West Texas Intermediate (WTI)
- **Description**: Spot price for WTI crude oil
- **Source**: U.S. Energy Information Administration (FRED)
- **Frequency**: Daily (aggregated to monthly)
- **Units**: Dollars per barrel
- **Date Range**: 1986-01 to 2024-03
- **Relevance**: Energy cost indicator; oil shocks have historically triggered recessions

---

## Data Processing Notes

### Frequency Alignment
All indicators are resampled to monthly frequency using forward-fill:
- **Quarterly data** (GDP): Values carried forward through months
- **Weekly data** (ICSA): Aggregated to monthly mean
- **Daily data** (T10Y3M, SP500, VIX, etc.): Aggregated to monthly mean

### Missing Data Handling
- **Initial Missing Values**: Forward-filled from first valid observation
- **Internal Gaps**: Linear interpolation applied
- **Post-Processing**: Periods with >20% missing features dropped

### Feature Engineering
From these 27 base indicators, **5,416 engineered features** are created:
- **Lag features**: 1, 2, 3, 4, 8 months
- **Rolling statistics**: 3, 6, 12-month windows (mean, std, min, max)
- **Differences**: Month-over-month, quarter-over-quarter, year-over-year
- **Rate of change**: Momentum and acceleration features
- **Ratios**: Cross-indicator relationships (e.g., SP500/GDP)
- **Interaction terms**: Products of key indicators

**Final Feature Count**: 29 features (after Random Forest importance + VIF filtering)

---

## Usage in Model

### Stage 1: Recession Classification
- **Input**: All 29 selected features
- **Target**: USREC (binary recession indicator)
- **Model**: XGBoost classifier with probability calibration

### Stage 2: Indicator Forecasting
- **Input**: Same 29 selected features (conditional on recession predicted)
- **Targets**: 3-month ahead forecasts for UNRATE, FEDFUNDS, GDP, and SP500
- **Model**: Multi-output ridge regression

---

## Data Quality and Reliability

### Strengths
- **Authoritative Sources**: All data from government agencies or established financial institutions
- **Long History**: Most indicators span 40+ years, covering multiple business cycles
- **High Frequency**: Monthly or higher resolution enables granular analysis
- **Standardized Collection**: Consistent methodologies across time periods

### Limitations
- **Revisions**: GDP and employment data subject to revisions after initial release
- **Survey-Based**: Consumer sentiment relies on survey responses (sampling error)
- **Structural Changes**: Economic relationships may shift over time (non-stationarity)
- **Frequency Mismatch**: Quarterly GDP resampled to monthly (introduces assumption)

---

## References

- **FRED**: Federal Reserve Economic Data - https://fred.stlouisfed.org/
- **NBER**: National Bureau of Economic Research - https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
- **Kaggle Dataset**: https://www.kaggle.com/datasets/mikoajfish99/us-recession-and-financial-indicators/data

---

*Last Updated*: 2025-12-01
*Model Version*: Hybrid Two-Stage Architecture (XGBoost + Ridge Regression)
