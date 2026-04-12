# PRISM: Predictive Returns and Inferred Structure for Modern Portfolios

PRISM is a research pipeline for portfolio construction under forecast uncertainty.
It combines:
- multi-model return and correlation prediction,
- error-aware confidence scoring,
- Bayesian shrinkage,
- Sharpe-oriented allocation,
- and optimal-transport-style rebalancing.

The implementation in this repository is organized as split Python cells that run in sequence to produce tabular results and diagnostic plots.

## Repository layout

- `code/`: executable pipeline code
  - `run_prism_split.py`: main runner for split cells
  - `cell_01.py` to `cell_15.py`: pipeline stages in execution order
  - `cell_order.txt`: execution order used by the runner
  - `generate_missing_images.py`: regenerate figures from saved CSV outputs
- `outputs/`: generated tables and time series CSV artifacts
- `images/`: generated plots (P01...)
- `Latex/`: LaTeX source and compiled PRISM paper PDF
- `Final_Deliverables/`: submission-ready PDFs and Python file
- `requirements.txt`: base Python dependency list

## Quick start

### 1) Create and activate a virtual environment

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

Note: the pipeline itself also installs additional research dependencies during execution (for example `yfinance`, `optuna`, `cvxpy`, `ripser`, `persim`, `seaborn`, `tqdm`, `tabulate`, and optional `torch`).

### 3) Run the full PRISM pipeline

From the repository root:

```bash
python code/run_prism_split.py
```

This executes `code/cell_01.py` through `code/cell_15.py` in order, in one shared namespace.

## Regenerate figures from existing CSV outputs

If CSV artifacts already exist in `outputs/`, regenerate the image set with:

```bash
python code/generate_missing_images.py
```

## Main generated artifacts

Saved to `outputs/`:
- `performance_table.csv`
- `performance_table.txt`
- `results_timeseries.csv`
- `weights_timeseries.csv`
- `kappa_timeseries.csv`
- `muB_timeseries.csv`
- `sigma_timeseries.csv`

Depending on run settings, additional logs/snapshots may also be written (for example allocation, order book, and debug logs).

Saved to `images/`:
- `P01_cum_all.png` through `P31_last_trades.png` (diagnostic and performance charts)

## Papers and deliverables

- Main paper source and build: `Latex/PRISM.tex` and `Latex/PRISM.pdf`
- Final deliverables folder contains submission outputs and code snapshot.

## Notes

- Data is downloaded from Yahoo Finance during execution; internet access is required for a fresh run.
- First run can take a while due to data download, model fitting/tuning, and figure generation.
- If you are reproducing results, keep Python/package versions stable across runs.
