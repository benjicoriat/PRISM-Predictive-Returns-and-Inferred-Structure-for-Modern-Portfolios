#  CELL 13 — VOL-FIT TO 15 % + METRICS
# ═══════════════════════════════════════════════════════════════════════
spy_al = spy_weekly.reindex(results.index).fillna(0)
results["SPY"] = spy_al.values

TARGET_VOL = 0.15

def vol_scale(ser, target, lb=26, cap=1.0):
    """Scale toward target vol; cap=1.0 means never lever above 1× (no leverage)."""
    out=ser.copy().values.astype(float)
    for i in range(lb,len(out)):
        rv=np.std(out[i-lb:i])*np.sqrt(52)
        lev=min(target/max(rv,1e-6),cap) if rv>1e-6 else 1.0
        out[i]*=lev
    return pd.Series(out, index=ser.index)

results["Strat_15"] = vol_scale(results["Strat_Raw"], TARGET_VOL)  # no-leverage
results["EW_15"]    = vol_scale(results["EW"], TARGET_VOL)
results["SPY_15"]   = vol_scale(results["SPY"], TARGET_VOL)

def mets(r, nm, ppy=52):
    r=r.dropna(); cum=(1+r).cumprod()
    tr=cum.iloc[-1]-1; yrs=len(r)/ppy
    ar=(1+tr)**(1/max(yrs,0.01))-1; av=r.std()*np.sqrt(ppy); sr=ar/max(av,1e-8)
    pk=cum.cummax(); dd=(cum-pk)/pk; mdd=dd.min()
    cal=ar/max(abs(mdd),1e-8); ds=r[r<0].std()*np.sqrt(ppy); sortino=ar/max(ds,1e-8)
    wr=(r>0).mean(); sk=stats.skew(r.values); ku=stats.kurtosis(r.values,fisher=True)
    return {"":nm,"Tot":f"{tr:.1%}","AnnR":f"{ar:.2%}","AnnV":f"{av:.2%}",
            "SR":f"{sr:.3f}","Sort":f"{sortino:.3f}","MDD":f"{mdd:.1%}",
            "Cal":f"{cal:.2f}","Win":f"{wr:.1%}","Sk":f"{sk:.2f}","Ku":f"{ku:.2f}"}

cols_map = OrderedDict([
    ("PRISM Raw","Strat_Raw"),("PRISM No-Lev","Strat_15"),
    ("EW","EW"),("EW No-Lev","EW_15"),("SPY","SPY"),("SPY No-Lev","SPY_15")])
mdf_all = pd.DataFrame([mets(results[c],nm) for nm,c in cols_map.items()]).set_index("")
print("\n"+"="*95+"\nPERFORMANCE\n"+"="*95)
print(mdf_all.to_string())
print(f"\nAvg Turnover={results['Turnover'].mean():.4f}  "
      f"Avg TC={results['TC'].mean()*1e4:.2f}bps  "
      f"Avg Gross={results['Gross'].mean():.2f}")

perf_txt = (
    "\n"+"="*95+"\nPERFORMANCE\n"+"="*95+"\n"
    + mdf_all.to_string()
    + "\n\n"
    + f"Avg Turnover={results['Turnover'].mean():.4f}  "
    + f"Avg TC={results['TC'].mean()*1e4:.2f}bps  "
    + f"Avg Gross={results['Gross'].mean():.2f}\n"
)
(OUTPUTS_DIR / "performance_table.txt").write_text(perf_txt, encoding="utf-8")
mdf_all.to_csv(OUTPUTS_DIR / "performance_table.csv")
results.to_csv(OUTPUTS_DIR / "results_timeseries.csv")
wdf.to_csv(OUTPUTS_DIR / "weights_timeseries.csv")
kdf.to_csv(OUTPUTS_DIR / "kappa_timeseries.csv")
mdf.to_csv(OUTPUTS_DIR / "muB_timeseries.csv")
sdf.to_csv(OUTPUTS_DIR / "sigma_timeseries.csv")
log.info(f"Saved tabular artifacts to {OUTPUTS_DIR}")

# ═══════════════════════════════════════════════════════════════════════
