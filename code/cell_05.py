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
