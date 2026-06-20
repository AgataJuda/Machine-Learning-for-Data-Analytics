# Predicting Hourly Bitcoin Return Direction

A Machine Learning for Data Analytics group project that asks a deliberately hard
question: **can the direction of the next hour's BTC/USD return be predicted from
recent market history?** We frame it as binary classification (will the next hourly
log-return be positive?) and, as a second objective, train reinforcement-learning
agents to turn any signal into a simple long/flat trading policy.

The project deliberately spans several course modules - classical time series,
tree ensembles, deep learning, reinforcement learning, uncertainty quantification,
and interpretability - wrapped in a reproducible MLflow-tracked pipeline.

## Key result

After testing classical, tree-based, deep-learning and reinforcement-learning
methods, we find **no reliable directional edge at the hourly horizon**: the best
classifier (Random Forest) reaches only **0.527 ROC-AUC** - barely above the
stratified baseline (0.519) - the forecasters score **below 50%** on direction, and
**no trading strategy beats Buy & Hold on a risk-adjusted basis**. This is consistent
with a near-efficient market and is a credible, well-supported finding rather than a
failure. The deliverable is that honest result *plus* a fully reproducible pipeline.

## Repository structure

```
.
├── data/
│   └── processed/        # train/val/test CSVs + metadata.json (produced by NB 01)
├── notebooks/
│   ├── 01_data_processing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modelling.ipynb
│   ├── 04_analysis.ipynb
│   └── 05_final_report.ipynb
├── models/              # trained models saved as plain files (produced by NB 03)
├── mlruns/              # MLflow tracking store (SQLite + artifacts)
├── reports/             # exported tables, figures, presentation.pdf
├── requirements.txt     # all libraries needed for code to work
└── README.md
```

## Dataset

Kaggle's minute-level **BTC/USD** history
([`mczielinski/bitcoin-historical-data`](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)),
downloaded automatically by notebook 01 via `kagglehub` - no manual download needed.
We restrict to the full year **2025**, resample to **hourly** bars (8,760 hours), and
engineer 25 leakage-free features (lagged returns, rolling statistics, momentum,
volatility proxies). The raw file is large and is **not** committed to the repo.

## Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo>

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

A Kaggle account is required for the automatic dataset download; see the
[kagglehub authentication guide](https://github.com/Kaggle/kagglehub) if prompted
for credentials.

## How to run

Run the notebooks **in order**, and **from inside the `notebooks/` directory** (all
paths are relative, e.g. `../data`, `../models`):

| # | Notebook | What it does |
|---|----------|--------------|
| 01 | `01_data_processing.ipynb` | download, validate, stationarity investigation, feature engineering, chronological split -> writes `data/processed/` |
| 02 | `02_eda.ipynb` | distributions, volatility, autocorrelation, stationarity, correlations |
| 03 | `03_modelling.ipynb` | trains **and saves** all models to `models/`; logs runs to MLflow |
| 04 | `04_analysis.ipynb` | evaluation, SHAP, MC-Dropout uncertainty, backtest; exports result tables |
| 05 | `05_final_report.ipynb` | standalone narrative report (loads saved artifacts) |

**Notes**
- Models use a **train-or-load** pattern: the first run of notebook 03 trains and
  saves each model; later runs load them from `models/` in seconds, so *Restart &
  Run All* always succeeds.
- A `RUN_HEAVY = False` switch at the top of notebook 03 skips the slowest
  exploratory cells (rolling LSTM forecast, hyperparameter halving search, epoch
  experiment). Set it to `True` to reproduce those in full.

## Experiment tracking (MLflow)

Every modelling run is logged to MLflow using a local SQLite backend. To open the
dashboard, from the repository root run:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

then open <http://localhost:5000>.

## Methods

| Module | Model(s) |
|--------|----------|
| Baseline | DummyClassifier (stratified) |
| Classical time series | ARIMA (returns), GARCH (volatility), Prophet |
| Tree ensembles | Random Forest, XGBoost |
| Deep learning | LSTM (sequence model) |
| Reinforcement learning | PPO and CEM trading agents |
| Uncertainty | MC-Dropout + Expected Calibration Error |
| Interpretability | SHAP (global + local) |
| MLOps | MLflow tracking, reproducible file-saved models |

## Reproducibility

- A single fixed random seed (**42**) across NumPy, Python, TensorFlow and the models.
- Relative paths throughout (run from `notebooks/`).
- Models persisted as plain files (no dependence on ephemeral MLflow run-IDs).
- Pinned dependencies in `requirements.txt` (run `pip freeze` to capture exact
  versions used).

## Team

| Name | Role |
|------|------|
| Agata Juda | Team Leader |
| Dominik Hołoś | Data Engineer |
| Weronika Jaszkiewicz | Modeler |
| Alicja Górnik | Analyst |

## Acknowledgements

Machine Learning for Data Analytics — group project. Dataset by *mczielinski* on Kaggle.
