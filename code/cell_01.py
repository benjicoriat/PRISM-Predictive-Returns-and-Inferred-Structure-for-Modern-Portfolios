#  CELL 1 — INSTALLS
# ═══════════════════════════════════════════════════════════════════════
import subprocess, sys
def _pip(*p):
    for x in p:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",x])
_pip("yfinance","optuna","scipy","scikit-learn","cvxpy",
     "ripser","persim","matplotlib","seaborn","pandas","numpy","tqdm","tabulate")
try:
    import torch, torch.nn as nn; HAS_TORCH=True
except ImportError:
    try: _pip("torch"); import torch, torch.nn as nn; HAS_TORCH=True
    except: HAS_TORCH=False

# ═══════════════════════════════════════════════════════════════════════
