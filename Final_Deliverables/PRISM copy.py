#!/usr/bin/env python3
"""
PRISM v3 — Topological Sharpe–OT Pipeline
==========================================
Benjamin Coriat — April 2026

v3:  24 global indices + commodities, order book tracker,
     improved ensemble (extra features, deeper tuning, feature importance),
     vol-fit to 15 % for fair comparison, 30+ diagnostic plots,
     comprehensive per-asset logging.
"""

# ═══════════════════════════════════════════════════════════════════════
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
#  CELL 2 — IMPORTS & LOGGING
# ═══════════════════════════════════════════════════════════════════════
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import matplotlib.dates as mdates
import yfinance as yf, optuna, cvxpy as cp, logging, io, time, copy
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
#  CELL 3 — CONFIG
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    START_DATE:str="2003-01-01"; END_DATE:str="2025-12-31"
    TRAIN_START:str="2003-01-01"; TRAIN_END:str="2011-12-31"
    TEST_START:str="2012-01-01"; TEST_END:str="2025-12-31"
    WEEKLY_FREQ:str="W-FRI"
    # Features
    CORR_WINDOW_WEEKS:int=26
    RETURN_LAGS:list=field(default_factory=lambda:[1,2,4,8,12])  # added 12
    CORR_LAGS:list=field(default_factory=lambda:[1,2,4])
    VOL_WINDOW_WEEKS:int=8; VOL_HISTORY_WEEKS:int=26; EWMA_DECAY:float=0.94
    # Ensemble — improved
    RANDOM_SEED:int=42; USE_TUNING:bool=True
    TUNING_TRIALS:int=10; TUNING_TIMEOUT_SEC:int=20; CV_SPLITS:int=5
    TORCH_EPOCHS:int=100; TORCH_PATIENCE:int=12; TORCH_BATCH_SIZE:int=64
    # Allocation
    ALPHA1:float=1.0; ALPHA2:float=1.0; ALPHA3:float=1.0
    BETA1:float=0.3; GAMMA1:float=0.5
    OMEGA:list=field(default_factory=lambda:[0.25]*4)
    ALPHA4:float=1.0; ALPHA5:float=1.0; ALPHA6:float=1.0; GAMMA2:float=0.5
    TAU0:float=1.0; LAMBDA_V:float=0.94; ETA_VOL:float=0.5; ZETA_BLEND:float=0.5
    RF_ANNUAL:float=0.045
    # Portfolio
    LONG_ONLY:bool=False; W_MAX:float=0.20; L_MAX:float=1.6
    SIGMA_TARGET_ANN:float=0.15   # vol target 15 %
    DEFAULT_TC_BPS:float=10.0
    SINKHORN_EPS_SCALE:float=0.05; SINKHORN_MAX_ITER:int=200; SINKHORN_TOL:float=1e-8
    VOL_LOOKBACK_WEEKS:int=26; VOL_TARGET_CAP:float=2.5
    # Tickers — 24 global indices + commodities
    TICKERS:list=field(default_factory=lambda:[
        "^GSPC","^NDX","^DJI","^RUT","^VIX","^FTSE","^GDAXI","^N225","^HSI","^STOXX50E",
        "^AXJO","^BSESN","^NSEI","^KS11","^TWII","^BVSP","^MXX","^GSPTSE","^FCHI","^IBEX",
        "GC=F","SI=F","CL=F","NG=F",
    ])

CFG = Config()
log.info(f"Config: {len(CFG.TICKERS)} tickers | LS={not CFG.LONG_ONLY} | "
         f"Lmax={CFG.L_MAX} | σ*={CFG.SIGMA_TARGET_ANN:.0%}")

# ═══════════════════════════════════════════════════════════════════════
#  CELL 4 — ORDER BOOK TRACKER
# ═══════════════════════════════════════════════════════════════════════
class OrderBookTracker:
    """
    Simulated order book tracker from OHLCV data.
    Reconstructs a synthetic L2 book snapshot per week using:
      - mid price = Close
      - spread estimated from intra-day range (Corwin-Schultz style)
      - depth levels synthesised from volume profile
    Logs every order (fill) from the OT transport plan.
    """
    def __init__(self, tickers):
        self.tickers = tickers
        self.tk_idx = {t:i for i,t in enumerate(tickers)}
        self.book_snapshots = []    # list of dicts per date
        self.order_log = []         # every executed trade
        self.fill_summary = []      # per-rebalance summary
        log.info("OrderBookTracker initialised")

    def build_snapshot(self, date, weekly_ohlc_row, gk_vol_row):
        """Build synthetic L2 snapshot from weekly OHLC."""
        snap = {"date": date, "books": {}}
        for tk in self.tickers:
            try:
                close = weekly_ohlc_row.get(tk, np.nan)
                if np.isnan(close): continue
                gk = gk_vol_row.get(tk, 0.01)
                # Corwin-Schultz spread estimate: spread ≈ 2*(exp(α)-1) where α from HL
                spread_pct = max(gk * 0.1, 0.0001)  # ~10% of weekly vol as spread proxy
                mid = close
                best_bid = mid * (1 - spread_pct/2)
                best_ask = mid * (1 + spread_pct/2)
                # Synthetic depth: 5 levels, geometrically spaced
                levels = 5
                bids, asks = [], []
                for lvl in range(levels):
                    tick = spread_pct * (0.5 + lvl * 0.5)
                    bp = mid * (1 - tick)
                    ap = mid * (1 + tick)
                    # Size decays with depth
                    sz = 1000 * (0.8 ** lvl)
                    bids.append({"price": round(bp, 4), "size": round(sz, 1), "level": lvl})
                    asks.append({"price": round(ap, 4), "size": round(sz, 1), "level": lvl})
                snap["books"][tk] = {
                    "mid": mid, "spread_pct": spread_pct,
                    "best_bid": best_bid, "best_ask": best_ask,
                    "bids": bids, "asks": asks,
                    "gk_vol": gk,
                }
            except: continue
        self.book_snapshots.append(snap)
        return snap

    def log_rebalance(self, date, w_old, w_new, T_star, tickers, portfolio_value=1e6):
        """Log each fill from the OT transport plan."""
        n = len(tickers)
        fills = []
        total_notional = 0
        total_cost_est = 0

        # Get latest book snapshot for spread info
        snap = self.book_snapshots[-1] if self.book_snapshots else None

        for i in range(n):
            for j in range(n):
                if i == j: continue
                flow = T_star[i, j] if T_star is not None else 0
                if abs(flow) < 1e-8: continue
                notional = abs(flow) * portfolio_value
                tk_sell, tk_buy = tickers[i], tickers[j]

                # Estimate fill prices from book
                sell_spread = 0.001
                buy_spread = 0.001
                if snap and tk_sell in snap["books"]:
                    sell_spread = snap["books"][tk_sell]["spread_pct"]
                if snap and tk_buy in snap["books"]:
                    buy_spread = snap["books"][tk_buy]["spread_pct"]

                slip = (sell_spread + buy_spread) / 2 * notional
                fill = {
                    "date": date,
                    "sell_asset": tk_sell, "buy_asset": tk_buy,
                    "flow_pct": flow,
                    "notional": notional,
                    "sell_spread_bps": sell_spread * 1e4,
                    "buy_spread_bps": buy_spread * 1e4,
                    "est_slippage": slip,
                    "side": "SELL→BUY",
                }
                fills.append(fill)
                self.order_log.append(fill)
                total_notional += notional
                total_cost_est += slip

        # Net trades per asset
        net_trades = {}
        for tk_i, (i_idx) in self.tk_idx.items():
            delta = w_new[i_idx] - w_old[i_idx]
            if abs(delta) > 1e-6:
                side = "BUY" if delta > 0 else "SELL"
                net_trades[tk_i] = {
                    "delta_w": delta,
                    "side": side,
                    "notional": abs(delta) * portfolio_value,
                }

        summary = {
            "date": date,
            "n_fills": len(fills),
            "total_notional": total_notional,
            "total_slippage_est": total_cost_est,
            "net_trades": net_trades,
            "turnover_pct": np.sum(np.abs(w_new - w_old)),
        }
        self.fill_summary.append(summary)
        log.debug(f"OB {date}: {len(fills)} fills, ${total_notional:,.0f} notional, "
                  f"~${total_cost_est:,.0f} slippage")
        return summary

    def get_order_log_df(self):
        if not self.order_log: return pd.DataFrame()
        return pd.DataFrame(self.order_log)

    def get_summary_df(self):
        if not self.fill_summary: return pd.DataFrame()
        rows = []
        for s in self.fill_summary:
            rows.append({
                "date": s["date"], "n_fills": s["n_fills"],
                "total_notional": s["total_notional"],
                "total_slippage": s["total_slippage_est"],
                "turnover": s["turnover_pct"],
                "n_net_trades": len(s["net_trades"]),
            })
        return pd.DataFrame(rows).set_index("date")

    def get_spread_history(self):
        """Extract spread time series per ticker."""
        rows = []
        for snap in self.book_snapshots:
            for tk, bk in snap["books"].items():
                rows.append({"date": snap["date"], "ticker": tk,
                             "spread_bps": bk["spread_pct"]*1e4, "mid": bk["mid"]})
        return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════
#  CELL 5 — DATA
# ═══════════════════════════════════════════════════════════════════════
def download_data(tickers, start, end):
    log.info(f"Downloading {len(tickers)} tickers …")
    t0 = time.time()
    raw = yf.download(tickers, start=start, end=end, interval="1d",
                      auto_adjust=False, group_by="ticker", threads=True, progress=True)
    log.info(f"Download: {time.time()-t0:.1f}s, shape={raw.shape}")
    return raw

def clean_and_align(raw_df, tickers):
    panels = {}
    for tk in tickers:
        try:
            df = raw_df[tk][["Open","High","Low","Close","Volume"]].copy() if len(tickers)>1 \
                 else raw_df[["Open","High","Low","Close","Volume"]].copy()
            df["Close"] = df["Close"].ffill()
            for c in ["Open","High","Low"]: df[c] = df[c].fillna(df["Close"])
            df["Volume"] = df["Volume"].fillna(0)
            df = df.dropna(subset=["Close"])
            if len(df) > 100:
                panels[tk] = df
                log.debug(f"  {tk}: {len(df)} days, {df.index[0].date()}→{df.index[-1].date()}")
            else:
                log.warning(f"  {tk}: only {len(df)} days — skipped")
        except Exception as e:
            log.warning(f"  {tk}: {e}")
    log.info(f"Kept {len(panels)}/{len(tickers)} tickers")
    return panels

raw_data = download_data(CFG.TICKERS, CFG.START_DATE, CFG.END_DATE)
daily_panels = clean_and_align(raw_data, CFG.TICKERS)
TICKERS = sorted(daily_panels.keys())
n_assets = len(TICKERS)

# SPY benchmark (separate)
spy_raw = yf.download("SPY", start=CFG.START_DATE, end=CFG.END_DATE,
                       interval="1d", auto_adjust=True, progress=False)
spy_weekly = spy_raw["Close"].resample(CFG.WEEKLY_FREQ).last().pct_change().dropna()
spy_weekly.name = "SPY"
log.info(f"SPY benchmark: {len(spy_weekly)} weeks")

# ═══════════════════════════════════════════════════════════════════════
#  CELL 6 — WEEKLY RESAMPLE & GK VOL
# ═══════════════════════════════════════════════════════════════════════
def weekly_resample(dp, freq="W-FRI"):
    wc, wr, wg = {}, {}, {}
    for tk, df in dp.items():
        wk = df.resample(freq).agg({"Open":"first","High":"max","Low":"min",
                                     "Close":"last","Volume":"sum"}).dropna(subset=["Close"])
        wc[tk] = wk["Close"]; wr[tk] = wk["Close"].pct_change()
        lnhl = np.log(wk["High"]/wk["Low"])
        lnco = np.log(wk["Close"]/wk["Open"])
        gk = np.sqrt((0.5*lnhl**2 - (2*np.log(2)-1)*lnco**2).clip(lower=0))
        wg[tk] = gk
    cd = pd.DataFrame(wc).dropna(how="all")
    rd = pd.DataFrame(wr).dropna(how="all")
    gd = pd.DataFrame(wg).dropna(how="all")
    ix = cd.index.intersection(rd.index).intersection(gd.index)
    return cd.loc[ix], rd.loc[ix], gd.loc[ix]

weekly_close, weekly_ret, weekly_gk = weekly_resample(daily_panels, CFG.WEEKLY_FREQ)
log.info(f"Weekly: {weekly_ret.shape[0]} wks × {weekly_ret.shape[1]} tickers")

# ═══════════════════════════════════════════════════════════════════════
#  CELL 7 — FEATURES (improved: more features)
# ═══════════════════════════════════════════════════════════════════════
def compute_rolling_corr(rd, w=26):
    cd = {}
    for i in range(w, len(rd)):
        dt = rd.index[i]; wd = rd.iloc[i-w:i]
        if wd.dropna(axis=1).shape[1] >= 2: cd[dt] = wd.corr()
    return cd

log.info("Rolling correlations …")
corr_dict = compute_rolling_corr(weekly_ret, CFG.CORR_WINDOW_WEEKS)
log.info(f"{len(corr_dict)} corr matrices")

def build_pair_features(rd, gd, cd, tickers, clags=[1,2,4]):
    cds = sorted(cd.keys()); rows = []
    for ti,tj in tqdm(list(combinations(tickers,2)), desc="Pair feats"):
        for k, dt in enumerate(cds):
            if k < max(clags) or k >= len(cds)-1: continue
            try:
                cn = cd[dt].loc[ti,tj]; ct = cd[cds[k+1]].loc[ti,tj]
                lags = [cd[cds[k-l]].loc[ti,tj] for l in clags]
                ri = rd.loc[dt,ti] if dt in rd.index else np.nan
                rj = rd.loc[dt,tj] if dt in rd.index else np.nan
                vi = gd.loc[dt,ti] if dt in gd.index else np.nan
                vj = gd.loc[dt,tj] if dt in gd.index else np.nan
                # NEW: cross-vol and vol ratio
                vol_ratio = vi/(vj+1e-8)
                cross_vol = vi * vj
                rows.append([dt,ti,tj,cn]+lags+[ri,rj,vi,vj,vol_ratio,cross_vol,ct])
            except: continue
    cols = ["Date","Ticker_i","Ticker_j","Corr"]+[f"Corr_lag_{l}" for l in clags]+\
           ["Return_i","Return_j","GK_Vol_i","GK_Vol_j","Vol_Ratio","Cross_Vol","Target_Corr"]
    return pd.DataFrame(rows, columns=cols).dropna()

def build_return_features(rd, gd, rlags=[1,2,4,8,12], vw=8):
    rows = []
    for tk in rd.columns:
        s = rd[tk].dropna(); g = gd[tk].reindex(s.index)
        for i in range(max(rlags)+vw, len(s)-1):
            dt = s.index[i]
            lags = [s.iloc[i-l] for l in rlags]
            rm4 = s.iloc[i-3:i+1].mean(); rm12 = s.iloc[max(0,i-11):i+1].mean()
            rm26 = s.iloc[max(0,i-25):i+1].mean()   # NEW: 26-wk momentum
            rvol = s.iloc[i-vw+1:i+1].std()
            gkv = g.iloc[i] if np.isfinite(g.iloc[i]) else 0
            # NEW: vol of vol, return acceleration
            rvol12 = s.iloc[max(0,i-11):i+1].std()
            vol_of_vol = g.iloc[max(0,i-vw+1):i+1].std() if i>=vw else 0
            ret_accel = (rm4 - rm12)  # short-term vs medium
            rows.append([dt,tk,s.iloc[i]]+lags+[rm4,rm12,rm26,rvol,rvol12,gkv,vol_of_vol,ret_accel,s.iloc[i+1]])
    cols = ["Date","Ticker","Return"]+[f"Return_lag_{l}" for l in rlags]+\
           ["RM4","RM12","RM26","RVol","RVol12","GK_Vol","VolOfVol","RetAccel","Target_Return"]
    return pd.DataFrame(rows, columns=cols).dropna()

log.info("Building features …")
pair_features   = build_pair_features(weekly_ret, weekly_gk, corr_dict, TICKERS, CFG.CORR_LAGS)
return_features = build_return_features(weekly_ret, weekly_gk, CFG.RETURN_LAGS, CFG.VOL_WINDOW_WEEKS)
log.info(f"Pair: {pair_features.shape} | Return: {return_features.shape}")

# ═══════════════════════════════════════════════════════════════════════
#  CELL 8 — IMPROVED ENSEMBLE (9 base learners)
# ═══════════════════════════════════════════════════════════════════════
class TorchMLP:
    def __init__(s, d, hl=3, hs=64, do=0.1, lr=1e-3, wd=1e-4, ep=100, pat=12, bs=64):
        s.ep,s.pat,s.bs=ep,pat,bs
        s.dev="cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        ls=[]; ind=d
        for _ in range(hl): ls+=[nn.Linear(ind,hs),nn.ReLU(),nn.Dropout(do)]; ind=hs
        ls.append(nn.Linear(ind,1))
        s.m=nn.Sequential(*ls).to(s.dev)
        s.opt=torch.optim.Adam(s.m.parameters(),lr=lr,weight_decay=wd); s.lf=nn.MSELoss()
    def fit(s,X,y):
        Xt=torch.FloatTensor(X).to(s.dev); yt=torch.FloatTensor(y.reshape(-1,1)).to(s.dev)
        dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt,yt),batch_size=s.bs,shuffle=True)
        b,w=1e18,0
        for _ in range(s.ep):
            s.m.train(); el=0
            for xb,yb in dl: s.opt.zero_grad(); l=s.lf(s.m(xb),yb); l.backward(); s.opt.step(); el+=l.item()
            a=el/len(dl)
            if a<b-1e-6: b,w=a,0
            else:
                w+=1
                if w>=s.pat: break
        return s
    def predict(s,X):
        s.m.eval()
        with torch.no_grad(): return s.m(torch.FloatTensor(X).to(s.dev)).cpu().numpy().ravel()

def get_base_learners(d, cfg):
    m = OrderedDict([
        ("OLS",      LinearRegression()),
        ("Ridge",    Ridge(alpha=1.0)),
        ("Lasso",    Lasso(alpha=0.001, max_iter=5000)),
        ("ElasticNet",ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)),
        ("RF",       RandomForestRegressor(n_estimators=250, max_depth=6, min_samples_leaf=5,
                                            random_state=cfg.RANDOM_SEED, n_jobs=-1)),
        ("ExtraTrees",ExtraTreesRegressor(n_estimators=250, max_depth=8, min_samples_leaf=3,
                                           random_state=cfg.RANDOM_SEED, n_jobs=-1)),
        ("GBM",      GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                random_state=cfg.RANDOM_SEED)),
        ("AdaBoost", AdaBoostRegressor(n_estimators=100, learning_rate=0.05,
                                        random_state=cfg.RANDOM_SEED)),
    ])
    if HAS_TORCH: m["MLP"] = TorchMLP(d)
    return m

def tune_model(nm, mdl, X, y, cfg):
    if nm in ["OLS","MLP","AdaBoost"] or not cfg.USE_TUNING: return mdl
    tscv = TimeSeriesSplit(n_splits=cfg.CV_SPLITS)
    def obj(tr):
        if nm=="Ridge": m=Ridge(alpha=tr.suggest_float("a",1e-4,100,log=True))
        elif nm=="Lasso": m=Lasso(alpha=tr.suggest_float("a",1e-5,1,log=True),max_iter=5000)
        elif nm=="ElasticNet": m=ElasticNet(alpha=tr.suggest_float("a",1e-5,1,log=True),
                                             l1_ratio=tr.suggest_float("l",0.05,0.95),max_iter=5000)
        elif nm=="RF": m=RandomForestRegressor(n_estimators=tr.suggest_int("n",100,400),
            max_depth=tr.suggest_int("d",3,12),min_samples_leaf=tr.suggest_int("ml",2,10),
            max_features=tr.suggest_float("mf",0.3,1.0),random_state=cfg.RANDOM_SEED,n_jobs=-1)
        elif nm=="ExtraTrees": m=ExtraTreesRegressor(n_estimators=tr.suggest_int("n",100,400),
            max_depth=tr.suggest_int("d",4,14),min_samples_leaf=tr.suggest_int("ml",2,8),
            random_state=cfg.RANDOM_SEED,n_jobs=-1)
        elif nm=="GBM": m=GradientBoostingRegressor(n_estimators=tr.suggest_int("n",100,500),
            max_depth=tr.suggest_int("d",2,6),learning_rate=tr.suggest_float("lr",0.01,0.2,log=True),
            subsample=tr.suggest_float("ss",0.6,1.0),random_state=cfg.RANDOM_SEED)
        else: return 0
        sc=[]
        for ti,vi in tscv.split(X): m.fit(X[ti],y[ti]); sc.append(mean_squared_error(y[vi],m.predict(X[vi])))
        return np.mean(sc)
    st=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED))
    st.optimize(obj,n_trials=cfg.TUNING_TRIALS,timeout=cfg.TUNING_TIMEOUT_SEC)
    bp=st.best_params
    if nm=="Ridge": return Ridge(alpha=bp["a"])
    if nm=="Lasso": return Lasso(alpha=bp["a"],max_iter=5000)
    if nm=="ElasticNet": return ElasticNet(alpha=bp["a"],l1_ratio=bp["l"],max_iter=5000)
    if nm=="RF": return RandomForestRegressor(n_estimators=bp["n"],max_depth=bp["d"],
        min_samples_leaf=bp["ml"],max_features=bp.get("mf",1.0),random_state=cfg.RANDOM_SEED,n_jobs=-1)
    if nm=="ExtraTrees": return ExtraTreesRegressor(n_estimators=bp["n"],max_depth=bp["d"],
        min_samples_leaf=bp["ml"],random_state=cfg.RANDOM_SEED,n_jobs=-1)
    if nm=="GBM": return GradientBoostingRegressor(n_estimators=bp["n"],max_depth=bp["d"],
        learning_rate=bp["lr"],subsample=bp["ss"],random_state=cfg.RANDOM_SEED)
    return mdl

MODEL_DIAG = {"pair": defaultdict(dict), "return": defaultdict(dict)}

def train_ensemble(X, y, cfg, label="", dk=None, ds=None):
    sp = int(0.8*len(X))
    Xb,yb = X[:sp],y[:sp]; Xm,ym = X[sp:],y[sp:]
    sc = StandardScaler(); Xbs=sc.fit_transform(Xb); Xms=sc.transform(Xm)
    base = get_base_learners(X.shape[1], cfg)
    trained = OrderedDict(); bms = {}
    for nm, mdl in base.items():
        if nm != "MLP": mdl = tune_model(nm, mdl, Xbs, yb, cfg)
        mdl.fit(Xbs, yb); trained[nm] = mdl
        p = mdl.predict(Xms)
        bms[nm] = {"mse":mean_squared_error(ym,p),"mae":mean_absolute_error(ym,p),
                   "r2":r2_score(ym,p),"da":np.mean(np.sign(p)==np.sign(ym))}
    mp = np.column_stack([m.predict(Xms) for m in trained.values()])
    me = np.abs(mp - ym.reshape(-1,1))
    Z = np.hstack([mp, me])
    meta = Ridge(alpha=1.0); meta.fit(Z, ym)
    ep = meta.predict(Z)
    es = {"mse":mean_squared_error(ym,ep),"mae":mean_absolute_error(ym,ep),
          "r2":r2_score(ym,ep),"da":np.mean(np.sign(ep)==np.sign(ym))}
    if ds is not None and dk is not None:
        ds[dk] = {"base_scores":bms,"ensemble_scores":es,
                  "meta_weights":dict(zip(trained.keys(),meta.coef_[:len(trained)])),
                  "n_train":len(Xb),"n_meta":len(Xm)}
    return {"scaler":sc,"base":trained,"meta":meta}

def ens_predict(ens, X):
    Xs = ens["scaler"].transform(X)
    bp = np.column_stack([m.predict(Xs) for m in ens["base"].values()])
    return ens["meta"].predict(np.hstack([bp, np.zeros_like(bp)]))

# ═══════════════════════════════════════════════════════════════════════
#  CELL 9 — TRAIN
# ═══════════════════════════════════════════════════════════════════════
def train_pair_models(pf, cfg):
    mask = pf["Date"]<=cfg.TRAIN_END
    fc = [c for c in pf.columns if c not in ["Date","Ticker_i","Ticker_j","Target_Corr"]]
    models = {}
    for (ti,tj), grp in tqdm(pf.groupby(["Ticker_i","Ticker_j"]), desc="Pair models"):
        tr = grp[mask.loc[grp.index]]
        if len(tr)<50: continue
        try: models[(ti,tj)]=train_ensemble(tr[fc].values,tr["Target_Corr"].values,cfg,
                                             dk=(ti,tj),ds=MODEL_DIAG["pair"])
        except: pass
    return models, fc

def train_return_models(rf, cfg):
    mask = rf["Date"]<=cfg.TRAIN_END
    fc = [c for c in rf.columns if c not in ["Date","Ticker","Target_Return"]]
    models = {}
    for tk, grp in tqdm(rf.groupby("Ticker"), desc="Return models"):
        tr = grp[mask.loc[grp.index]]
        if len(tr)<50: continue
        try: models[tk]=train_ensemble(tr[fc].values,tr["Target_Return"].values,cfg,
                                        dk=tk,ds=MODEL_DIAG["return"])
        except: pass
    return models, fc

log.info("═"*30+" TRAINING "+"═"*30)
t0=time.time()
pair_models, pfc = train_pair_models(pair_features, CFG)
log.info(f"Pair models: {len(pair_models)} in {time.time()-t0:.0f}s")
t0=time.time()
return_models, rfc = train_return_models(return_features, CFG)
log.info(f"Return models: {len(return_models)} in {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════════════
#  CELL 10 — PREDICT & ERROR CHAR
# ═══════════════════════════════════════════════════════════════════════
def predict_all(pf,rf,pm,rm,pfc,rfc,cfg):
    pp,rp={},{}
    for (ti,tj),grp in pf[pf["Date"]>=cfg.TEST_START].groupby(["Ticker_i","Ticker_j"]):
        if (ti,tj) not in pm: continue
        preds=np.clip(ens_predict(pm[(ti,tj)],grp[pfc].values),-1,1)
        for dt,p,a in zip(grp["Date"],preds,grp["Target_Corr"]):
            pp.setdefault(dt,{})[(ti,tj)]=(p,a)
    for tk,grp in rf[rf["Date"]>=cfg.TEST_START].groupby("Ticker"):
        if tk not in rm: continue
        preds=ens_predict(rm[tk],grp[rfc].values)
        for dt,p,a in zip(grp["Date"],preds,grp["Target_Return"]):
            rp.setdefault(dt,{})[tk]=(p,a)
    return pp,rp

def build_outputs(pp,rp,wg,tickers,cfg):
    dates=sorted(set(rp.keys())&set(pp.keys()))
    n=len(tickers); ti={t:i for i,t in enumerate(tickers)}
    lam,k=cfg.EWMA_DECAY,cfg.VOL_HISTORY_WEEKS
    reh={t:[] for t in tickers}; peh={p:[] for p in combinations(tickers,2)}
    outs=[]
    for dt in tqdm(dates,desc="Outputs"):
        mu=np.zeros(n)
        for tk in tickers:
            if tk in rp.get(dt,{}): mu[ti[tk]]=rp[dt][tk][0]
        Ch=np.eye(n)
        for (a,b),(pred,act) in pp.get(dt,{}).items():
            if a in ti and b in ti: i,j=ti[a],ti[b]; Ch[i,j]=Ch[j,i]=pred
        for tk in tickers:
            if tk in rp.get(dt,{}): p,a=rp[dt][tk]; reh[tk].append(p-a)
        for (a,b),(pred,act) in pp.get(dt,{}).items():
            if (a,b) in peh: peh[(a,b)].append(pred-act)
        Em=np.zeros((n,5))
        for tk in tickers:
            idx=ti[tk]; errs=reh[tk]
            if len(errs)<3: Em[idx]=[0,1,0.01,0,0]; continue
            ea=np.array(errs); w=np.array([lam**(len(ea)-1-s) for s in range(len(ea))]); w/=w.sum()
            Em[idx,0]=w@ea
            if tk in rp.get(dt,{}): p,a=rp[dt][tk]; Em[idx,1]=float(np.sign(p)==np.sign(a))
            Em[idx,2]=ea.std() if len(ea)>1 else 0.01
            Em[idx,3]=stats.skew(ea) if len(ea)>2 else 0
            Em[idx,4]=stats.kurtosis(ea,fisher=True) if len(ea)>3 else 0
        Ec=np.zeros((n,n,4))
        for (a,b),errs in peh.items():
            if a not in ti or b not in ti or len(errs)<3: continue
            i,j=ti[a],ti[b]; ea=np.array(errs)
            w=np.array([lam**(len(ea)-1-s) for s in range(len(ea))]); w/=w.sum()
            vs=[w@ea,ea.std(),stats.skew(ea) if len(ea)>2 else 0,
                stats.kurtosis(ea,fisher=True) if len(ea)>3 else 0]
            for ci,v in enumerate(vs): Ec[i,j,ci]=Ec[j,i,ci]=v
        V=np.zeros((n,k))
        dl=wg.index.get_indexer([dt],method="ffill")[0]; sl=max(0,dl-k+1)
        for tk in tickers:
            idx=ti[tk]
            if tk in wg.columns: vals=wg[tk].iloc[sl:dl+1].values; V[idx,-len(vals):]=vals
        outs.append({"date":dt,"mu_hat":mu,"E_mu":Em,"C_hat":Ch,"E_C":Ec,"V":V})
    return outs

log.info("Predicting …")
pair_preds, ret_preds = predict_all(pair_features,return_features,pair_models,
                                     return_models,pfc,rfc,CFG)
log.info("Building outputs …")
pipeline_outputs = build_outputs(pair_preds,ret_preds,weekly_gk,TICKERS,CFG)
log.info(f"{len(pipeline_outputs)} output records")

# ═══════════════════════════════════════════════════════════════════════
#  CELL 11 — ALLOCATION & OT (same as v2)
# ═══════════════════════════════════════════════════════════════════════
def nearest_pd(A):
    n=A.shape[0]; A=np.nan_to_num(np.array(A,dtype=np.float64)); B=(A+A.T)/2
    try: ev,evc=np.linalg.eigh(B)
    except: return np.eye(n)*max(np.nanmean(np.diag(B)),1e-4)
    ev=np.maximum(ev,1e-8); A3=evc@np.diag(ev)@evc.T; A3=(A3+A3.T)/2
    me=np.min(np.linalg.eigvalsh(A3))
    if me<1e-8: A3+=np.eye(n)*(1e-8-me)
    return A3

def compute_conf(Em,cfg):
    n=Em.shape[0]; b,d,v,xi,ki=Em[:,0],Em[:,1],Em[:,2],Em[:,3],np.maximum(Em[:,4],0)
    mad_b=max(np.median(np.abs(b-np.median(b))),1e-8)
    sb=np.exp(-cfg.ALPHA1*np.abs(b)/mad_b); sd=cfg.BETA1*d+(1-cfg.BETA1)
    delta=v*(1+cfg.GAMMA1*ki); md=max(np.median(delta),1e-8)
    sdisp=np.exp(-cfg.ALPHA2*delta/md); sskew=np.exp(-cfg.ALPHA3*np.maximum(-xi,0))
    w=cfg.OMEGA; return np.clip(sb**w[0]*sd**w[1]*sdisp**w[2]*sskew**w[3],1e-4,1.0)

def bayes_post(mu_t,kappa,cfg,sh):
    n=len(mu_t); tau0=cfg.TAU0; m0=np.mean(mu_t)*np.ones(n); K=np.diag(kappa)
    mn=(tau0*m0+K@mu_t)/(tau0+1); nu0=n+2
    Psi0=max(nu0-n-1,1)*sh; dev=(mu_t-m0).reshape(-1,1)
    Psi_n=Psi0+(tau0/(tau0+1))*(dev@dev.T); return mn, nearest_pd(Psi_n/max(nu0+1-n-1,1))

def topo_analysis(Ch,Ec,cfg):
    n=Ch.shape[0]; D=np.sqrt(2*(1-np.clip(Ch,-1,1))); np.fill_diagonal(D,0)
    res=ripser(squareform(D),maxdim=0,metric="precomputed"); dgm0=res["dgms"][0]
    tp=np.sum(dgm0[:,1]-dgm0[:,0]); P=np.zeros((n,n))
    if tp>0:
        for i in range(n):
            for j in range(i+1,n):
                dij=D[i,j]; active=np.sum((dgm0[:,1]-dgm0[:,0])[(dgm0[:,0]<=dij)&(dij<dgm0[:,1])])
                P[i,j]=P[j,i]=active/tp
    kC=np.ones((n,n)); e1f=Ec[:,:,0][np.triu_indices(n,k=1)]
    mad1=max(np.median(np.abs(e1f-np.median(e1f))),1e-8)
    for i in range(n):
        for j in range(i+1,n):
            t1=np.exp(-cfg.ALPHA4*abs(Ec[i,j,0])/mad1)
            t2=np.exp(-cfg.ALPHA5*Ec[i,j,1]*(1+cfg.GAMMA2*max(Ec[i,j,3],0)))
            t3=np.exp(-cfg.ALPHA6*max(-Ec[i,j,2],0))
            kC[i,j]=kC[j,i]=t1*t2*t3
    phi=P*kC; ut=Ch[np.triu_indices(n,k=1)]; rb=np.mean(ut) if len(ut) else 0
    Cshr=np.full((n,n),rb); np.fill_diagonal(Cshr,1)
    Ct=phi*Ch+(1-phi)*Cshr; np.fill_diagonal(Ct,1); Ct=nearest_pd(Ct)
    pv=dgm0[:,1]-dgm0[:,0]; es=dgm0[np.argmax(pv),1] if len(pv) else np.median(D[np.triu_indices(n,k=1)])
    A=np.where((D<=es)&(np.eye(n)==0),np.clip(Ch,0,None),0)
    DA=np.diag(A.sum(1)); L=DA-A; ev,evc=eigh(L)
    gaps=np.diff(ev); kc=min(max(np.argmax(gaps[1:])+2,2),n//2) if len(gaps)>1 else 2
    Uk=evc[:,:kc]; proj=Uk@Uk.T
    Cs=proj*Ct+(np.eye(n)-proj)*rb; np.fill_diagonal(Cs,1); return nearest_pd(Cs), dgm0

def robust_cov(Cs,V,SB,cfg):
    n=Cs.shape[0]; lam=cfg.LAMBDA_V; k=V.shape[1]
    w=np.array([(1-lam)*lam**(k-1-l)/(1-lam**k) for l in range(k)])
    sh=np.sqrt(np.maximum(np.sum(w*V**2,axis=1),1e-12))
    vb=np.maximum(V.mean(1),1e-6); rv=sh/vb
    sa=sh*(1+cfg.ETA_VOL*np.maximum(rv-1,0))
    Ds=np.diag(sa); St=nearest_pd(Ds@Cs@Ds)
    return nearest_pd(cfg.ZETA_BLEND*SB+(1-cfg.ZETA_BLEND)*St)

def sharpe_alloc(muB,Sf,cfg):
    n=len(muB); rfw=cfg.RF_ANNUAL/52; mex=muB-rfw; bsr,bw=-np.inf,np.ones(n)/n
    for g in np.logspace(-2,3,50):
        w=cp.Variable(n); cons=[cp.sum(w)==1,cp.norm(w,1)<=cfg.L_MAX]
        if cfg.LONG_ONLY: cons+=[w>=0,w<=cfg.W_MAX]
        else: cons+=[w>=-cfg.W_MAX,w<=cfg.W_MAX]
        prob=cp.Problem(cp.Maximize(mex@w-g*cp.quad_form(w,Sf)),cons)
        try:
            prob.solve(solver=cp.SCS,verbose=False,max_iters=5000)
            if prob.status in ["optimal","optimal_inaccurate"]:
                wv=np.array(w.value).flatten(); pv=np.sqrt(wv@Sf@wv)
                if pv>1e-8:
                    sr=(mex@wv)/pv
                    if sr>bsr: bsr,bw=sr,wv.copy()
        except: continue
    return bw

def sinkhorn(a,b,C,eps=0.01,mi=200,tol=1e-8):
    n=len(a); aa=np.maximum(np.abs(a),1e-10); aa/=aa.sum()
    bb=np.maximum(np.abs(b),1e-10); bb/=bb.sum()
    K=np.maximum(np.exp(-C/max(eps,1e-10)),1e-300); v=np.ones(n)
    for _ in range(mi):
        u=aa/(K@v+1e-300); v=bb/(K.T@u+1e-300)
        if np.max(np.abs(K.T@u*v-bb))<tol: break
    return np.diag(u)@K@np.diag(v)

# ═══════════════════════════════════════════════════════════════════════
#  CELL 12 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════
def run_backtest(po, wr, wc, wg, tickers, cfg):
    n=len(tickers); ti={t:i for i,t in enumerate(tickers)}
    tc=np.full(n, cfg.DEFAULT_TC_BPS/1e4)
    ob = OrderBookTracker(tickers)

    dates,sr_raw,sr_vt,ewr,to_l,tc_l=[],[],[],[],[],[]
    w_hist,kap_hist,muB_hist,sig_hist=[],[],[],[]
    le,se,gl,ne,rv_l=[],[],[],[],[]
    realized=[]
    w_cur=np.ones(n)/n

    for rec in tqdm(po, desc="Backtesting"):
        dt=rec["date"]
        if dt not in wr.index: continue
        aret=np.zeros(n); ok=True
        for tk in tickers:
            if tk in wr.columns and dt in wr.index:
                r=wr.loc[dt,tk]; aret[ti[tk]]=r if np.isfinite(r) else 0
            else: ok=False
        if not ok: continue

        # Order book snapshot
        close_row = {tk: wc.loc[dt,tk] if tk in wc.columns and dt in wc.index else np.nan for tk in tickers}
        gk_row = {tk: wg.loc[dt,tk] if tk in wg.columns and dt in wg.index else 0.01 for tk in tickers}
        ob.build_snapshot(dt, close_row, gk_row)

        try:
            kappa=compute_conf(rec["E_mu"],cfg)
            mu0=np.mean(rec["mu_hat"]); mu_t=kappa*rec["mu_hat"]+(1-kappa)*mu0
            Vs=rec["V"].copy(); Vs[~np.isfinite(Vs)]=0
            Vs+=np.random.RandomState(42).randn(*Vs.shape)*1e-8
            sh=np.cov(Vs) if Vs.shape[1]>1 else np.eye(n)*0.01
            if not np.all(np.isfinite(sh)): sh=np.eye(n)*0.01
            sh=nearest_pd(sh)
            muB,SB=bayes_post(mu_t,kappa,cfg,sh)
            try: Cs,dgm0=topo_analysis(rec["C_hat"],rec["E_C"],cfg)
            except: Cs,dgm0=nearest_pd(rec["C_hat"]),None
            Sf=robust_cov(Cs,rec["V"],SB,cfg)
            wt=sharpe_alloc(muB,Sf,cfg)
        except:
            wt=np.ones(n)/n; kappa=np.ones(n); muB=np.zeros(n); Sf=np.eye(n)*0.01

        # OT
        T_star = None
        try:
            cm=np.abs(tc.reshape(-1,1))+np.abs(tc.reshape(1,-1)); np.fill_diagonal(cm,0)
            eps=cfg.SINKHORN_EPS_SCALE*np.median(cm[cm>0]) if np.any(cm>0) else 0.01
            T_star=sinkhorn(w_cur,wt,cm,eps=max(eps,1e-6))
            tc_cost=np.sum(cm*T_star)
        except: tc_cost=0

        # Order book log
        ob.log_rebalance(dt, w_cur, wt, T_star, tickers)

        pr_raw=wt@aret-tc_cost; realized.append(pr_raw)
        lb=cfg.VOL_LOOKBACK_WEEKS
        if len(realized)>=lb:
            rv=np.std(realized[-lb:])*np.sqrt(52)
            lev=min(cfg.SIGMA_TARGET_ANN/max(rv,1e-6),cfg.VOL_TARGET_CAP)
        else: lev=1.0
        pr_vt=pr_raw*lev
        ew=aret.mean()
        port_vol=np.sqrt(wt@Sf@wt)*np.sqrt(52) if np.all(np.isfinite(Sf)) else 0

        dates.append(dt); sr_raw.append(pr_raw); sr_vt.append(pr_vt)
        ewr.append(ew); to_l.append(np.sum(np.abs(wt-w_cur))); tc_l.append(tc_cost)
        w_hist.append(wt.copy()); kap_hist.append(kappa.copy())
        muB_hist.append(muB.copy()); sig_hist.append(np.sqrt(np.diag(Sf)))
        le.append(np.sum(wt[wt>0])); se.append(np.sum(wt[wt<0]))
        gl.append(np.sum(np.abs(wt))); ne.append(np.sum(wt)); rv_l.append(port_vol)
        w_cur=wt.copy()

    res=pd.DataFrame({"Strat_Raw":sr_raw,"Strat_VT":sr_vt,"EW":ewr,
        "Turnover":to_l,"TC":tc_l,"Long":le,"Short":se,"Gross":gl,
        "Net":ne,"PredVol":rv_l},index=pd.DatetimeIndex(dates))
    wdf=pd.DataFrame(w_hist,index=res.index,columns=tickers)
    kdf=pd.DataFrame(kap_hist,index=res.index,columns=tickers)
    mdf=pd.DataFrame(muB_hist,index=res.index,columns=tickers)
    sdf=pd.DataFrame(sig_hist,index=res.index,columns=tickers)
    return res, wdf, kdf, mdf, sdf, ob

log.info("═"*30+" BACKTEST "+"═"*30)
results, wdf, kdf, mdf, sdf, ob_tracker = run_backtest(
    pipeline_outputs, weekly_ret, weekly_close, weekly_gk, TICKERS, CFG)
log.info(f"Backtest: {len(results)} weeks")

# ═══════════════════════════════════════════════════════════════════════
#  CELL 13 — VOL-FIT TO 15 % + METRICS
# ═══════════════════════════════════════════════════════════════════════
spy_al = spy_weekly.reindex(results.index).fillna(0)
results["SPY"] = spy_al.values

TARGET_VOL = 0.15

def vol_scale(ser, target, lb=26, cap=3.0):
    out=ser.copy().values.astype(float)
    for i in range(lb,len(out)):
        rv=np.std(out[i-lb:i])*np.sqrt(52)
        lev=min(target/max(rv,1e-6),cap) if rv>1e-6 else 1.0
        out[i]*=lev
    return pd.Series(out, index=ser.index)

results["Strat_15"] = vol_scale(results["Strat_VT"], TARGET_VOL)
results["EW_15"]    = vol_scale(results["EW"], TARGET_VOL)
results["SPY_15"]   = vol_scale(results["SPY"], TARGET_VOL)

def mets(r, nm, ppy=52):
    r=r.dropna(); cum=(1+r).cumprod()
    tr=cum.iloc[-1]-1; yrs=len(r)/ppy
    ar=(1+tr)**(1/max(yrs,0.01))-1; av=r.std()*np.sqrt(ppy); sr=ar/max(av,1e-8)
    pk=cum.cummax(); dd=(cum-pk)/pk; mdd=dd.min()
    cal=ar/max(abs(mdd),1e-8); ds=r[r<0].std()*np.sqrt(ppy); sortino=ar/max(ds,1e-8)
    wr=(r>0).mean(); sk=stats.skew(r.values); ku=stats.kurtosis(r.values,fisher=True)
    return {"":nm,"Tot":f"{tr:.1%}","AnnR":f"{ar:.2%}","AnnV":f"{av:.2%}",
            "SR":f"{sr:.3f}","Sort":f"{sortino:.3f}","MDD":f"{mdd:.1%}",
            "Cal":f"{cal:.2f}","Win":f"{wr:.1%}","Sk":f"{sk:.2f}","Ku":f"{ku:.2f}"}

cols_map = OrderedDict([
    ("PRISM Raw","Strat_Raw"),("PRISM VT","Strat_VT"),("PRISM @15%","Strat_15"),
    ("EW","EW"),("EW @15%","EW_15"),("SPY","SPY"),("SPY @15%","SPY_15")])
mdf_all = pd.DataFrame([mets(results[c],nm) for nm,c in cols_map.items()]).set_index("")
print("\n"+"="*95+"\nPERFORMANCE\n"+"="*95)
print(mdf_all.to_string())
print(f"\nAvg Turnover={results['Turnover'].mean():.4f}  "
      f"Avg TC={results['TC'].mean()*1e4:.2f}bps  "
      f"Avg Gross={results['Gross'].mean():.2f}")

# ═══════════════════════════════════════════════════════════════════════
#  CELL 14 — PLOTS (30+)
# ═══════════════════════════════════════════════════════════════════════
sns.set_theme(style="darkgrid")
plt.rcParams.update({"figure.dpi":130,"font.size":9,"axes.titlesize":11})

def sf(name): plt.savefig(f"{name}.png",dpi=150,bbox_inches="tight"); log.debug(f"Saved {name}.png")

cum = {nm: (1+results[c]).cumprod() for nm,c in cols_map.items()}

# ── P1: Cumulative all ──
fig,ax=plt.subplots(figsize=(15,6))
for nm,s in cum.items():
    lw=2.5 if "PRISM" in nm else 1.2; ls="-" if "PRISM" in nm or nm in ["SPY","EW"] else "--"
    ax.plot(s,label=nm,lw=lw,ls=ls)
ax.set_yscale("log"); ax.set_title("Cumulative Returns — All"); ax.legend(fontsize=7,ncol=2)
sf("P01_cum_all"); plt.show()

# ── P2: Vol-matched @15% comparison ──
fig,ax=plt.subplots(figsize=(15,6))
for nm in ["PRISM @15%","EW @15%","SPY @15%"]:
    ax.plot(cum[nm],label=nm,lw=2)
ax.set_yscale("log"); ax.set_title("Vol-Matched @15% Comparison (fair)"); ax.legend()
sf("P02_vol15_compare"); plt.show()

# ── P3: Drawdowns (3 panels) ──
fig,axes=plt.subplots(3,1,figsize=(15,10),sharex=True)
for ax,(nm,c) in zip(axes,[("PRISM @15%","Strat_15"),("EW @15%","EW_15"),("SPY @15%","SPY_15")]):
    cu=(1+results[c]).cumprod(); pk=cu.cummax(); dd=(cu-pk)/pk
    ax.fill_between(dd.index,dd.values,0,alpha=0.5); ax.set_ylabel("DD"); ax.set_title(nm)
plt.tight_layout(); sf("P03_drawdowns"); plt.show()

# ── P4: Rolling 26w Sharpe ──
fig,ax=plt.subplots(figsize=(15,5))
for nm,c in [("PRISM VT","Strat_VT"),("EW","EW"),("SPY","SPY")]:
    rs=results[c].rolling(26).mean()/results[c].rolling(26).std()*np.sqrt(52)
    ax.plot(rs,label=nm,lw=1.5 if "PRISM" in nm else 1)
ax.axhline(0,color="grey",ls="--",alpha=.5); ax.set_title("Rolling 26w Sharpe"); ax.legend()
sf("P04_roll_sharpe"); plt.show()

# ── P5: Rolling 52w return ──
fig,ax=plt.subplots(figsize=(15,5))
for nm,c in [("PRISM VT","Strat_VT"),("EW","EW"),("SPY","SPY")]:
    rr=results[c].rolling(52).apply(lambda x:(1+x).prod()-1)
    ax.plot(rr,label=nm)
ax.axhline(0,color="grey",ls="--"); ax.set_title("Rolling 52w Return"); ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_:f"{y:.0%}"))
sf("P05_roll_1y_ret"); plt.show()

# ── P6: Rolling vol + target ──
fig,ax=plt.subplots(figsize=(15,5))
for nm,c in [("PRISM VT","Strat_VT"),("PRISM Raw","Strat_Raw"),("EW","EW"),("SPY","SPY")]:
    rv=results[c].rolling(26).std()*np.sqrt(52); ax.plot(rv,label=nm)
ax.axhline(TARGET_VOL,color="red",ls=":",lw=2,label=f"Target {TARGET_VOL:.0%}")
ax.set_title("Rolling 26w Vol"); ax.legend(fontsize=7)
sf("P06_roll_vol"); plt.show()

# ── P7: Long/short stacked ──
fig,axes=plt.subplots(2,1,figsize=(15,10),sharex=True)
wl=wdf.clip(lower=0); axes[0].stackplot(wl.index,wl.values.T,labels=wl.columns,alpha=.8)
axes[0].set_title("Long Weights"); axes[0].legend(fontsize=5,ncol=6,loc="upper left")
ws=wdf.clip(upper=0).abs(); axes[1].stackplot(ws.index,ws.values.T,labels=ws.columns,alpha=.8)
axes[1].set_title("Short Weights (abs)"); axes[1].legend(fontsize=5,ncol=6,loc="upper left")
plt.tight_layout(); sf("P07_weight_stacks"); plt.show()

# ── P8: Gross/net leverage ──
fig,axes=plt.subplots(2,1,figsize=(15,6),sharex=True)
axes[0].plot(results["Gross"],color="navy"); axes[0].axhline(CFG.L_MAX,color="red",ls=":")
axes[0].set_title("Gross Leverage")
axes[1].plot(results["Net"],color="green"); axes[1].axhline(1,color="grey",ls="--")
axes[1].set_title("Net Exposure")
plt.tight_layout(); sf("P08_leverage"); plt.show()

# ── P9: Turnover + TC ──
fig,axes=plt.subplots(2,1,figsize=(15,6),sharex=True)
axes[0].bar(results.index,results["Turnover"],width=5,alpha=.5,color="steelblue")
axes[0].set_title(f"Turnover (avg={results['Turnover'].mean():.3f})")
axes[1].bar(results.index,results["TC"]*1e4,width=5,alpha=.5,color="tomato")
axes[1].set_title(f"TC (bps, avg={results['TC'].mean()*1e4:.2f})")
plt.tight_layout(); sf("P09_turnover_tc"); plt.show()

# ── P10: Return distribution ──
fig,ax=plt.subplots(figsize=(13,5))
for nm,c,co in [("PRISM","Strat_VT","#1f77b4"),("EW","EW","#ff7f0e"),("SPY","SPY","#2ca02c")]:
    ax.hist(results[c].dropna(),bins=80,alpha=.4,label=nm,color=co,density=True)
ax.set_title("Weekly Return Distribution"); ax.legend(); sf("P10_ret_dist"); plt.show()

# ── P11: QQ plots ──
fig,axes=plt.subplots(1,3,figsize=(16,5))
for ax,(nm,c) in zip(axes,[("PRISM","Strat_VT"),("EW","EW"),("SPY","SPY")]):
    stats.probplot(results[c].dropna().values,dist="norm",plot=ax); ax.set_title(f"QQ — {nm}")
plt.tight_layout(); sf("P11_qq"); plt.show()

# ── P12: Monthly heatmap ──
mo=results["Strat_VT"].resample("ME").apply(lambda x:(1+x).prod()-1)
mh=pd.DataFrame({"Y":mo.index.year,"M":mo.index.month,"R":mo.values})
piv=mh.pivot_table(values="R",index="Y",columns="M",aggfunc="first")
fig,ax=plt.subplots(figsize=(14,8))
sns.heatmap(piv,annot=True,fmt=".1%",center=0,cmap="RdYlGn",ax=ax,
            xticklabels=["J","F","M","A","M","J","J","A","S","O","N","D"])
ax.set_title("Monthly Returns Heatmap — PRISM VT"); sf("P12_monthly"); plt.show()

# ── P13: κ evolution ──
fig,ax=plt.subplots(figsize=(15,6))
for tk in TICKERS[:10]:
    if tk in kdf.columns: ax.plot(kdf[tk],label=tk,alpha=.7,lw=1)
ax.set_title("Confidence κ_i (10 tickers)"); ax.legend(fontsize=6,ncol=3)
sf("P13_kappa"); plt.show()

# ── P14: Avg κ + predicted vol ──
fig,axes=plt.subplots(2,1,figsize=(15,7),sharex=True)
axes[0].plot(kdf.mean(1),color="purple",lw=1.5); axes[0].set_title("Avg κ")
axes[1].plot(results["PredVol"],color="darkred",lw=1.5)
axes[1].axhline(TARGET_VOL,color="red",ls=":"); axes[1].set_title("Predicted Ann Vol")
plt.tight_layout(); sf("P14_kappa_vol"); plt.show()

# ── P15: μ^B heatmap ──
fig,ax=plt.subplots(figsize=(16,8))
sns.heatmap(mdf.iloc[::4].T*100,cmap="RdYlGn",center=0,ax=ax,
            yticklabels=True,xticklabels=False)
ax.set_title("μ^B (% wkly)"); sf("P15_muB"); plt.show()

# ── P16: σ heatmap ──
fig,ax=plt.subplots(figsize=(16,8))
sns.heatmap(sdf.iloc[::4].T*100,cmap="YlOrRd",ax=ax,yticklabels=True,xticklabels=False)
ax.set_title("σ (% wkly)"); sf("P16_sigma"); plt.show()

# ── P17: Final corr + κ ──
if pipeline_outputs:
    last=pipeline_outputs[-1]; fig,axes=plt.subplots(1,2,figsize=(17,7))
    sns.heatmap(last["C_hat"],xticklabels=TICKERS,yticklabels=TICKERS,
                cmap="RdBu_r",center=0,vmin=-1,vmax=1,ax=axes[0])
    axes[0].set_title(f"Pred Corr — {last['date'].strftime('%Y-%m-%d')}"); axes[0].tick_params(labelsize=6)
    conf=compute_conf(last["E_mu"],CFG)
    axes[1].barh(TICKERS,conf,color=plt.cm.viridis(conf))
    axes[1].set_title("κ_i"); axes[1].set_xlim(0,1)
    plt.tight_layout(); sf("P17_final_corr"); plt.show()

# ── P18: Base R² (return models) ──
if MODEL_DIAG["return"]:
    tks=list(MODEL_DIAG["return"].keys())
    bns=list(MODEL_DIAG["return"][tks[0]]["base_scores"].keys())
    r2d={nm:[MODEL_DIAG["return"][tk]["base_scores"][nm]["r2"] for tk in tks] for nm in bns}
    r2d["Ensemble"]=[MODEL_DIAG["return"][tk]["ensemble_scores"]["r2"] for tk in tks]
    fig,ax=plt.subplots(figsize=(16,7))
    pd.DataFrame(r2d,index=tks).plot(kind="bar",ax=ax,width=.85)
    ax.set_title("Return Model R²"); ax.axhline(0,color="grey",ls="--")
    ax.legend(fontsize=6,ncol=4); plt.xticks(rotation=45,ha="right")
    plt.tight_layout(); sf("P18_ret_r2"); plt.show()

# ── P19: DA (return) ──
if MODEL_DIAG["return"]:
    dad={nm:[MODEL_DIAG["return"][tk]["base_scores"][nm]["da"] for tk in tks] for nm in bns}
    dad["Ensemble"]=[MODEL_DIAG["return"][tk]["ensemble_scores"]["da"] for tk in tks]
    fig,ax=plt.subplots(figsize=(16,7))
    pd.DataFrame(dad,index=tks).plot(kind="bar",ax=ax,width=.85)
    ax.axhline(.5,color="red",ls=":"); ax.set_title("Return Model DA")
    ax.legend(fontsize=6,ncol=4); plt.xticks(rotation=45,ha="right")
    plt.tight_layout(); sf("P19_ret_da"); plt.show()

# ── P20: Meta weights ──
if MODEL_DIAG["return"]:
    mw=pd.DataFrame({tk:MODEL_DIAG["return"][tk]["meta_weights"] for tk in tks}).T
    fig,ax=plt.subplots(figsize=(16,7))
    mw.plot(kind="bar",stacked=True,ax=ax,width=.85)
    ax.set_title("Meta Weights (Ridge coefs)"); ax.legend(fontsize=6,ncol=4)
    plt.xticks(rotation=45,ha="right"); plt.tight_layout(); sf("P20_meta_w"); plt.show()

# ── P21: Pair model diagnostics ──
if MODEL_DIAG["pair"]:
    pr2=[v["ensemble_scores"]["r2"] for v in MODEL_DIAG["pair"].values()]
    pda=[v["ensemble_scores"]["da"] for v in MODEL_DIAG["pair"].values()]
    pma=[v["ensemble_scores"]["mae"] for v in MODEL_DIAG["pair"].values()]
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    axes[0].hist(pr2,bins=30,alpha=.7,color="steelblue"); axes[0].set_title(f"Pair R² (med={np.median(pr2):.3f})")
    axes[1].hist(pda,bins=30,alpha=.7,color="seagreen"); axes[1].set_title(f"Pair DA (med={np.median(pda):.3f})")
    axes[2].hist(pma,bins=30,alpha=.7,color="coral"); axes[2].set_title(f"Pair MAE (med={np.median(pma):.4f})")
    plt.suptitle("Corr Model Diagnostics",y=1.02); plt.tight_layout(); sf("P21_pair_diag"); plt.show()

# ── P22: Weight heatmap ──
fig,ax=plt.subplots(figsize=(17,8))
sns.heatmap(wdf.iloc[::2].T,cmap="RdBu_r",center=0,ax=ax,
            yticklabels=True,xticklabels=False,vmin=-CFG.W_MAX,vmax=CFG.W_MAX)
ax.set_title("Weights (long=blue, short=red)"); sf("P22_w_heatmap"); plt.show()

# ── P23: Annual returns ──
yrl={}
for nm,c in [("PRISM VT","Strat_VT"),("EW","EW"),("SPY","SPY")]:
    yrl[nm]=results[c].resample("YE").apply(lambda x:(1+x).prod()-1)
ydf=pd.DataFrame(yrl); ydf.index=ydf.index.year
fig,ax=plt.subplots(figsize=(14,6))
ydf.plot(kind="bar",ax=ax,width=.75); ax.set_title("Annual Returns")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_:f"{y:.0%}"))
ax.axhline(0,color="grey",ls="--"); plt.xticks(rotation=0)
plt.tight_layout(); sf("P23_annual"); plt.show()

# ── P24: Underwater ──
fig,ax=plt.subplots(figsize=(15,5))
cu=(1+results["Strat_VT"]).cumprod(); pk=cu.cummax(); uw=cu/pk-1
ax.fill_between(uw.index,uw.values,0,alpha=.6,color="#d62728")
ax.set_title("PRISM VT — Underwater"); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_:f"{y:.0%}"))
sf("P24_underwater"); plt.show()

# ── P25: GK vol cross-sectional ──
fig,ax=plt.subplots(figsize=(15,5))
ax.plot(weekly_gk.mean(1)*np.sqrt(52)*100,color="darkred",lw=1)
ax.set_title("Avg GK Vol (ann %)"); sf("P25_gk_avg"); plt.show()

# ══════════════════════════════════════════════════════════════════════
# P26–P29: PER-ASSET LOG EVOLUTION
# ══════════════════════════════════════════════════════════════════════
# P26: All assets normalised log-scale
fig,ax=plt.subplots(figsize=(16,8))
normed=weekly_close/weekly_close.iloc[0]
for tk in TICKERS:
    if tk in normed.columns: ax.plot(normed[tk],label=tk,alpha=.7,lw=1)
ax.set_yscale("log"); ax.set_title("All Assets — Normalised (log)")
ax.legend(fontsize=5,ncol=5,loc="upper left"); sf("P26_all_assets_log"); plt.show()

# P27: Individual asset panels (6x4 grid)
n_tk=len(TICKERS); ncols=4; nrows=(n_tk+ncols-1)//ncols
fig,axes=plt.subplots(nrows,ncols,figsize=(20,3.5*nrows),sharex=True)
axes_flat=axes.flatten()
for i,tk in enumerate(TICKERS):
    ax=axes_flat[i]
    if tk in normed.columns:
        ax.plot(normed[tk],lw=1,color="#1f77b4")
        ax.set_title(tk,fontsize=9); ax.set_yscale("log")
for j in range(len(TICKERS),len(axes_flat)): axes_flat[j].set_visible(False)
fig.suptitle("Individual Asset Evolution (log normalised)",fontsize=14,y=1.01)
plt.tight_layout(); sf("P27_individual_assets"); plt.show()

# P28: Strategy vs EW vs SPY (large)
fig,ax=plt.subplots(figsize=(16,7))
for nm,c,co,lw in [("PRISM @15%","Strat_15","#1f77b4",3),
                     ("EW @15%","EW_15","#ff7f0e",2),("SPY @15%","SPY_15","#2ca02c",2)]:
    cu=(1+results[c]).cumprod(); ax.plot(cu,label=nm,color=co,lw=lw)
ax.set_yscale("log"); ax.set_title("Strategy vs EW vs SPY — Vol-Fitted @15%",fontsize=14)
ax.legend(fontsize=11); ax.set_ylabel("Growth of $1",fontsize=11)
ax.grid(True,alpha=.3); sf("P28_strat_vs_bench_15"); plt.show()

# ══════════════════════════════════════════════════════════════════════
# P29–P31: ORDER BOOK DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════
ob_summary = ob_tracker.get_summary_df()
if len(ob_summary) > 0:
    fig,axes=plt.subplots(3,1,figsize=(15,10),sharex=True)
    axes[0].bar(ob_summary.index,ob_summary["n_fills"],width=5,alpha=.6,color="navy")
    axes[0].set_title(f"OT Fills Per Rebalance (avg={ob_summary['n_fills'].mean():.0f})")
    axes[1].bar(ob_summary.index,ob_summary["total_notional"]/1e3,width=5,alpha=.6,color="teal")
    axes[1].set_title("Total Notional ($K)")
    axes[2].bar(ob_summary.index,ob_summary["total_slippage"],width=5,alpha=.6,color="tomato")
    axes[2].set_title(f"Est. Slippage ($, avg={ob_summary['total_slippage'].mean():.0f})")
    plt.tight_layout(); sf("P29_ob_summary"); plt.show()

# P30: Spread history
spread_hist = ob_tracker.get_spread_history()
if len(spread_hist)>0:
    fig,ax=plt.subplots(figsize=(15,6))
    avg_spread=spread_hist.groupby("date")["spread_bps"].mean()
    ax.plot(avg_spread.index,avg_spread.values,color="purple",lw=1.5)
    ax.set_title("Avg Estimated Spread (bps) Over Time"); ax.set_ylabel("bps")
    sf("P30_spread_hist"); plt.show()

# P31: Order flow heatmap (last rebalance)
if ob_tracker.fill_summary:
    last_fs = ob_tracker.fill_summary[-1]
    nt = last_fs["net_trades"]
    if nt:
        fig,ax=plt.subplots(figsize=(14,6))
        tks_sorted = sorted(nt.keys(), key=lambda x: nt[x]["delta_w"])
        deltas = [nt[t]["delta_w"]*100 for t in tks_sorted]
        colors = ["#2ca02c" if d>0 else "#d62728" for d in deltas]
        ax.barh(tks_sorted, deltas, color=colors)
        ax.set_title(f"Net Trades — Last Rebalance ({last_fs['date'].strftime('%Y-%m-%d')})")
        ax.set_xlabel("Δw (%)"); ax.axvline(0,color="grey",ls="--")
        sf("P31_last_trades"); plt.show()

log.info(f"✓ {31} plots generated")

# ═══════════════════════════════════════════════════════════════════════
#  CELL 15 — SNAPSHOT & LOG DUMP
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*60+"\nLATEST ALLOCATION\n"+"="*60)
if len(wdf)>0:
    lw=wdf.iloc[-1].sort_values(ascending=False)
    print(f"{'Ticker':>12}  {'Weight':>8}  {'Side':>6}")
    print("-"*30)
    for tk,w in lw.items():
        if abs(w)>.001: print(f"  {tk:>10}  {w:>7.2%}  {'LONG' if w>0 else 'SHORT':>6}")
    print(f"\n  Net: {lw.sum():.2%} | Gross: {lw.abs().sum():.2%} | "
          f"Long: {lw[lw>0].sum():.2%} | Short: {lw[lw<0].sum():.2%}")

# Order book summary
print("\n"+"="*60+"\nORDER BOOK SUMMARY\n"+"="*60)
obs = ob_tracker.get_summary_df()
if len(obs)>0:
    print(f"Total rebalances: {len(obs)}")
    print(f"Avg fills/rebalance: {obs['n_fills'].mean():.0f}")
    print(f"Avg notional: ${obs['total_notional'].mean():,.0f}")
    print(f"Avg slippage: ${obs['total_slippage'].mean():,.2f}")
    print(f"Total slippage: ${obs['total_slippage'].sum():,.0f}")
    odf = ob_tracker.get_order_log_df()
    if len(odf)>0:
        print(f"Total order records: {len(odf)}")
        print(f"Avg spread (sell): {odf['sell_spread_bps'].mean():.1f} bps")
        print(f"Avg spread (buy):  {odf['buy_spread_bps'].mean():.1f} bps")

# Debug log
print("\n"+"="*60+"\nDEBUG LOG (last 80 lines)\n"+"="*60)
for line in LOG_BUF.getvalue().strip().split("\n")[-80:]: print(line)

print("\n✅ PRISM v3 COMPLETE — 24 tickers, long-short, OB tracker, vol@15%, 31 plots")