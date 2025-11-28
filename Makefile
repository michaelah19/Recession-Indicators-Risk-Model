.PHONY: help install install-dev test lint format clean data notebook

help:
	@echo "Available commands:"
	@echo "  make install       - Install package in production mode"
	@echo "  make install-dev   - Install package with development dependencies"
	@echo "  make test          - Run tests with pytest"
	@echo "  make lint          - Run code linting (flake8, mypy)"
	@echo "  make format        - Format code with black and isort"
	@echo "  make clean         - Remove generated files and caches"
	@echo "  make data          - Download Kaggle dataset"
	@echo "  make notebook      - Start Jupyter Lab"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	flake8 src/ tests/ entrypoint/
	mypy src/ tests/ entrypoint/

format:
	black src/ tests/ entrypoint/ notebooks/
	isort src/ tests/ entrypoint/ notebooks/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage build/ dist/

data:
	@echo "Downloading Kaggle dataset..."
	@echo "Make sure you have configured Kaggle API credentials (~/.kaggle/kaggle.json)"
	kaggle datasets download -d mikoajfish99/us-recession-and-financial-indicators -p data/raw --unzip

notebook:
	jupyter lab --notebook-dir=notebooks
