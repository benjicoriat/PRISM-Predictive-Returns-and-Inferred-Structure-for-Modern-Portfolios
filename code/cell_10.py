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
