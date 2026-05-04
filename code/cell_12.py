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
    rng = np.random.default_rng(cfg.RANDOM_SEED)  # one stream for the whole backtest

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
            Vs+=rng.standard_normal(Vs.shape)*1e-8
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
