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
