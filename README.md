# News Sentiment Analysis

A financial analytics pipeline that connects news sentiment to stock price movements. Built for **Nova Financial Solutions** as part of the 10 Academy KAIM Week 1 challenge.

## Project Overview

The pipeline is structured around three tasks:

- **Task 1** — Exploratory analysis of 1.4 million financial news headlines (FNSPID dataset): descriptive statistics, publisher analysis, time-series publication trends, and NLP topic modeling.
- **Task 2** — Quantitative analysis of historical stock price data using TA-Lib (SMA, EMA, RSI, MACD) and PyNance (annualised return, volatility, Sharpe ratio, max drawdown) for AAPL, MSFT, NVDA, TSLA, and GOOG.
- **Task 3** — Correlation analysis between daily news sentiment scores (VADER) and daily stock returns using Pearson correlation.

## Repository Structure

```
news-sentiment-analysis/
├── .github/workflows/    # CI/CD — GitHub Actions unit test pipeline
├── data/
│   └── raw/              # Raw datasets (not committed — see Data Setup below)
├── notebooks/            # Jupyter notebooks for each task
├── src/                  # Reusable Python modules (EDA functions)
├── tests/                # Unit tests (pytest)
├── scripts/              # Standalone utility scripts
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/KalkidanAsfaw/news-sentiment-analysis.git
cd news-sentiment-analysis
```

### 2. Install TA-Lib system dependency

TA-Lib requires a C library to be installed before the Python package:

```bash
# Ubuntu / Debian
sudo apt-get install -y libta-lib-dev

# macOS
brew install ta-lib
```

### 3. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Data Setup

The raw datasets are not committed to the repository. Place the following files in `data/raw/`:

| File | Description | Source |
|------|-------------|--------|
| `raw_analyst_ratings.csv` | 1.4M financial news headlines (FNSPID) | 10 Academy Google Drive |

Stock price data is downloaded automatically via `yfinance` when running the Task 2 notebook.

### 5. Run the notebooks

```bash
jupyter notebook
```

Open the notebooks in order:
1. `notebooks/eda_descriptive_statistics.ipynb` — Task 1 EDA
2. `notebooks/technical_indicators.ipynb` — Task 2 technical indicators

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `yfinance` | Historical stock price download |
| `TA-Lib` | Technical indicators (SMA, EMA, RSI, MACD) |
| `pynance` | Financial metrics (annualised return, Sharpe ratio) |
| `nltk`, `vaderSentiment` | Sentiment analysis |
| `textblob` | Additional NLP utilities |
| `scikit-learn` | TF-IDF and topic modeling (LDA) |
| `jupyter`, `ipykernel` | Notebook environment |
| `pytest`, `pytest-cov` | Unit testing and coverage |

## Running Tests

```bash
pytest tests/ --cov=src
```

## Data Sources

- **FNSPID** (Financial News and Stock Price Integration Dataset) — provided via 10 Academy
- **Historical stock prices** — Yahoo Finance via the `yfinance` Python library
