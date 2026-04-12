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
