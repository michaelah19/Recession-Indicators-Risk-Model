# US Recession Indicators Risk Model

Machine learning project for analyzing and predicting US recession indicators using financial and economic data.

## Dataset

This project uses the [US Recession and Financial Indicators](https://www.kaggle.com/datasets/mikoajfish99/us-recession-and-financial-indicators/data) dataset from Kaggle, which includes:

- Federal funds rate, GDP, unemployment levels
- Bank and consumer credit, money supply
- Real estate loans, stock market data (S&P 500, NASDAQ)

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

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development tools
make install-dev
```

### 2. Configure Kaggle API

1. Create account at https://www.kaggle.com
2. Go to Account → Create New API Token
3. Save `kaggle.json` to `~/.kaggle/kaggle.json`
4. Set permissions (Mac/Linux): `chmod 600 ~/.kaggle/kaggle.json`

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
