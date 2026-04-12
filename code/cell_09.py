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
