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
