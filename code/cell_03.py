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
    LONG_ONLY:bool=True; W_MAX:float=0.20; L_MAX:float=1.0
    SIGMA_TARGET_ANN:float=0.15   # vol target 15 %
    DEFAULT_TC_BPS:float=10.0
    SINKHORN_EPS_SCALE:float=0.05; SINKHORN_MAX_ITER:int=200; SINKHORN_TOL:float=1e-8
    VOL_LOOKBACK_WEEKS:int=26; VOL_TARGET_CAP:float=1.0  # no leverage
    # Tickers — 24 global indices + commodities
    TICKERS:list=field(default_factory=lambda:[
        "^GSPC","^NDX","^DJI","^RUT","^VIX","^FTSE","^GDAXI","^N225","^HSI","^STOXX50E",
        "^AXJO","^BSESN","^NSEI","^KS11","^TWII","^BVSP","^MXX","^GSPTSE","^FCHI","^IBEX",
        "GC=F","SI=F","CL=F","NG=F",
    ])

CFG = Config()
log.info(f"Config: {len(CFG.TICKERS)} tickers | LongOnly={CFG.LONG_ONLY} | "
         f"Lmax={CFG.L_MAX} | σ*={CFG.SIGMA_TARGET_ANN:.0%} | LevCap={CFG.VOL_TARGET_CAP}")

# ═══════════════════════════════════════════════════════════════════════
