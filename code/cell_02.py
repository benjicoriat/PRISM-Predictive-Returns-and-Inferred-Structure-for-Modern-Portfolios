#  CELL 2 — IMPORTS & LOGGING
# ═══════════════════════════════════════════════════════════════════════
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import matplotlib.dates as mdates
import yfinance as yf, optuna, cvxpy as cp, logging, io, time, copy
from pathlib import Path
from scipy import stats
from scipy.linalg import eigh
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from ripser import ripser
from itertools import combinations
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────
LOG_BUF = io.StringIO()
def _setup_log():
    lg = logging.getLogger("PRISM"); lg.setLevel(logging.DEBUG); lg.handlers.clear()
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-5s │ %(message)s", datefmt="%H:%M:%S"))
    lg.addHandler(ch)
    bh = logging.StreamHandler(LOG_BUF); bh.setLevel(logging.DEBUG)
    bh.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-7s │ %(message)s"))
    lg.addHandler(bh)
    return lg
log = _setup_log()
log.info("✓ Logging ready — DEBUG→buffer, INFO→console")

# ═══════════════════════════════════════════════════════════════════════
