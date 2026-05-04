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
    # Meta-learner was trained on [base_preds | abs(base_preds - y)].
    # At inference y is unknown, so the absolute-error block is zeroed.
    Xs = ens["scaler"].transform(X)
    bp = np.column_stack([m.predict(Xs) for m in ens["base"].values()])
    return ens["meta"].predict(np.hstack([bp, np.zeros_like(bp)]))

# ═══════════════════════════════════════════════════════════════════════
