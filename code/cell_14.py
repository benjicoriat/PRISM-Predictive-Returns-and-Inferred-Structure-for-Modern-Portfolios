#  CELL 14 — PLOTS (30+)
# ═══════════════════════════════════════════════════════════════════════
sns.set_theme(style="darkgrid")
plt.rcParams.update({"figure.dpi":130,"font.size":9,"axes.titlesize":11})

def sf(name):
    out_path = IMAGES_DIR / f"{name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.debug(f"Saved {out_path}")

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
