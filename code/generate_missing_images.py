from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_VOL = 0.15
ONLY_MISSING = False  # regenerate all images


def load_csv(name: str) -> pd.DataFrame:
    path = OUTPUTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ── Load all CSVs ────────────────────────────────────────────────────────
results = load_csv("results_timeseries.csv")
wdf     = load_csv("weights_timeseries.csv")
kdf     = load_csv("kappa_timeseries.csv")
mdf     = load_csv("muB_timeseries.csv")
sdf     = load_csv("sigma_timeseries.csv")

for df in (results, wdf, kdf, mdf, sdf):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)


# ── No-leverage vol-scaling (cap=1.0 — never lever above 1×) ─────────────
def vol_scale(ser: pd.Series, target: float = TARGET_VOL,
              lb: int = 26, cap: float = 1.0) -> pd.Series:
    """Scale returns toward *target* annualised vol; cap prevents leverage above 1×."""
    out = ser.copy().values.astype(float)
    for i in range(lb, len(out)):
        rv = np.std(out[i - lb:i]) * np.sqrt(52)
        lev = min(target / max(rv, 1e-6), cap) if rv > 1e-6 else 1.0
        out[i] *= lev
    return pd.Series(out, index=ser.index)


results["Strat_NoLev"] = vol_scale(results["Strat_Raw"], TARGET_VOL, cap=1.0)
results["EW_NoLev"]    = vol_scale(results["EW"],        TARGET_VOL, cap=1.0)
results["SPY_NoLev"]   = vol_scale(results["SPY"],       TARGET_VOL, cap=1.0)


# ── Performance table ────────────────────────────────────────────────────
cols_map = OrderedDict([
    ("PRISM Raw",      "Strat_Raw"),
    ("PRISM No-Lev",   "Strat_NoLev"),
    ("EW",             "EW"),
    ("EW No-Lev",      "EW_NoLev"),
    ("SPY",            "SPY"),
    ("SPY No-Lev",     "SPY_NoLev"),
])


def mets(r: pd.Series, nm: str, ppy: int = 52) -> dict:
    r = r.dropna()
    cum = (1 + r).cumprod()
    tr = cum.iloc[-1] - 1
    yrs = len(r) / ppy
    ar = (1 + tr) ** (1 / max(yrs, 0.01)) - 1
    av = r.std() * np.sqrt(ppy)
    sr = ar / max(av, 1e-8)
    pk = cum.cummax()
    dd = (cum - pk) / pk
    mdd = dd.min()
    cal = ar / max(abs(mdd), 1e-8)
    ds = r[r < 0].std() * np.sqrt(ppy)
    sortino = ar / max(ds, 1e-8)
    wr = (r > 0).mean()
    sk = stats.skew(r.values)
    ku = stats.kurtosis(r.values, fisher=True)
    return {
        "": nm, "Tot": f"{tr:.1%}", "AnnR": f"{ar:.2%}", "AnnV": f"{av:.2%}",
        "SR": f"{sr:.3f}", "Sort": f"{sortino:.3f}", "MDD": f"{mdd:.1%}",
        "Cal": f"{cal:.2f}", "Win": f"{wr:.1%}", "Sk": f"{sk:.2f}", "Ku": f"{ku:.2f}",
    }


perf_df = pd.DataFrame(
    [mets(results[c], nm) for nm, c in cols_map.items()]
).set_index("")

perf_txt = (
    "\n" + "=" * 95 + "\nPERFORMANCE (No-Leverage, Vol-Targeted @15%)\n" + "=" * 95 + "\n"
    + perf_df.to_string()
    + f"\n\nAvg Turnover={results['Turnover'].mean():.4f}  "
    + f"Avg TC={results['TC'].mean()*1e4:.2f}bps  "
    + f"Avg Gross={results['Gross'].mean():.2f}\n"
)
print(perf_txt)
(OUTPUTS_DIR / "performance_table.txt").write_text(perf_txt, encoding="utf-8")
perf_df.to_csv(OUTPUTS_DIR / "performance_table.csv")


# ── Plotting helpers ─────────────────────────────────────────────────────
generated: list[str] = []
skipped:   list[str] = []


def save_fig(name: str, fig: plt.Figure) -> None:
    out = IMAGES_DIR / f"{name}.png"
    if ONLY_MISSING and out.exists():
        skipped.append(name)
        plt.close(fig)
        return
    fig.savefig(out, dpi=150, bbox_inches="tight")
    generated.append(name)
    plt.close(fig)


def save_message_plot(name: str, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.02, 0.55, message, transform=ax.transAxes,
            va="center", ha="left", fontsize=11, wrap=True)
    save_fig(name, fig)


plt.style.use("ggplot")

# Cumulative series for plotting
cum = {nm: (1 + results[c]).cumprod() for nm, c in cols_map.items()}


# ── P01: Cumulative all ──────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(15, 6))
    for nm, s in cum.items():
        lw = 2.5 if "PRISM" in nm else 1.2
        ls = "-" if "PRISM" in nm or nm in ["SPY", "EW"] else "--"
        ax.plot(s, label=nm, lw=lw, ls=ls)
    ax.set_yscale("log")
    ax.set_title("Cumulative Returns — All Strategies (No Leverage)")
    ax.legend(fontsize=7, ncol=2)
    save_fig("P01_cum_all", fig)
except Exception as exc:
    save_message_plot("P01_cum_all", "Cumulative All", f"Error: {exc}")


# ── P02: No-leverage @15% comparison ────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(15, 6))
    for nm in ["PRISM No-Lev", "EW No-Lev", "SPY No-Lev"]:
        ax.plot(cum[nm], label=nm, lw=2)
    ax.set_yscale("log")
    ax.set_title("Vol-Matched @15% Comparison — No Leverage (fair)")
    ax.legend()
    save_fig("P02_vol15_compare", fig)
except Exception as exc:
    save_message_plot("P02_vol15_compare", "No-Lev Comparison", f"Error: {exc}")


# ── P03: Drawdowns ───────────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(15, 5))
    palette = {"PRISM No-Lev": "#1f77b4", "EW No-Lev": "#ff7f0e", "SPY No-Lev": "#d62728"}
    for nm, c in [("PRISM No-Lev", "Strat_NoLev"), ("EW No-Lev", "EW_NoLev"), ("SPY No-Lev", "SPY_NoLev")]:
        cu = (1 + results[c]).cumprod()
        pk = cu.cummax()
        dd = (cu - pk) / pk
        ax.fill_between(dd.index, dd.values, 0, alpha=0.25, color=palette[nm])
        ax.plot(dd.index, dd.values, lw=1.2, label=nm, color=palette[nm])
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown Profiles — No-Leverage Strategies")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_fig("P03_drawdowns", fig)
except Exception as exc:
    save_message_plot("P03_drawdowns", "Drawdowns", f"Error: {exc}")


# ── P04: Rolling 26w Sharpe ──────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(15, 5))
    for nm, c in [("PRISM No-Lev", "Strat_NoLev"), ("EW", "EW"), ("SPY", "SPY")]:
        rs = results[c].rolling(26).mean() / results[c].rolling(26).std() * np.sqrt(52)
        ax.plot(rs, label=nm, lw=1.5 if "PRISM" in nm else 1)
    ax.axhline(0, color="grey", ls="--", alpha=0.5)
    ax.set_title("Rolling 26w Sharpe Ratio")
    ax.legend()
    save_fig("P04_roll_sharpe", fig)
except Exception as exc:
    save_message_plot("P04_roll_sharpe", "Rolling Sharpe", f"Error: {exc}")


# ── P05: Rolling 52w return ──────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(15, 5))
    for nm, c in [("PRISM No-Lev", "Strat_NoLev"), ("EW", "EW"), ("SPY", "SPY")]:
        rr = results[c].rolling(52).apply(lambda x: (1 + x).prod() - 1)
        ax.plot(rr, label=nm)
    ax.axhline(0, color="grey", ls="--")
    ax.set_title("Rolling 52w Return")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    save_fig("P05_roll_1y_ret", fig)
except Exception as exc:
    save_message_plot("P05_roll_1y_ret", "Rolling 52w Return", f"Error: {exc}")


# ── P06: Rolling vol + target ────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(15, 5))
    for nm, c in [
        ("PRISM No-Lev", "Strat_NoLev"),
        ("PRISM Raw",    "Strat_Raw"),
        ("EW",           "EW"),
        ("SPY",          "SPY"),
    ]:
        rv = results[c].rolling(26).std() * np.sqrt(52)
        ax.plot(rv, label=nm)
    ax.axhline(TARGET_VOL, color="red", ls=":", lw=2, label=f"Target {TARGET_VOL:.0%}")
    ax.set_title("Rolling 26w Annualised Volatility")
    ax.legend(fontsize=7)
    save_fig("P06_roll_vol", fig)
except Exception as exc:
    save_message_plot("P06_roll_vol", "Rolling Vol", f"Error: {exc}")


# ── P07: Long/short weight stacks ────────────────────────────────────────
try:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    wl = wdf.clip(lower=0)
    axes[0].stackplot(wl.index, wl.values.T, labels=wl.columns, alpha=0.8)
    axes[0].set_title("Long Weights")
    axes[0].legend(fontsize=5, ncol=6, loc="upper left")
    ws = wdf.clip(upper=0).abs()
    axes[1].stackplot(ws.index, ws.values.T, labels=ws.columns, alpha=0.8)
    axes[1].set_title("Short Weights (abs)")
    axes[1].legend(fontsize=5, ncol=6, loc="upper left")
    plt.tight_layout()
    save_fig("P07_weight_stacks", fig)
except Exception as exc:
    save_message_plot("P07_weight_stacks", "Weight Stacks", f"Error: {exc}")


# ── P08: Gross/net leverage ──────────────────────────────────────────────
try:
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    axes[0].plot(results["Gross"], color="navy")
    axes[0].set_title("Gross Leverage (from optimiser weights)")
    axes[1].plot(results["Net"], color="green")
    axes[1].axhline(1, color="grey", ls="--")
    axes[1].set_title("Net Exposure")
    plt.tight_layout()
    save_fig("P08_leverage", fig)
except Exception as exc:
    save_message_plot("P08_leverage", "Gross/Net Leverage", f"Error: {exc}")


# ── P09: Turnover + TC ───────────────────────────────────────────────────
try:
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    axes[0].bar(results.index, results["Turnover"], width=5, alpha=0.5, color="steelblue")
    axes[0].set_title(f"Turnover (avg={results['Turnover'].mean():.3f})")
    axes[1].bar(results.index, results["TC"] * 1e4, width=5, alpha=0.5, color="tomato")
    axes[1].set_title(f"Transaction Cost (bps, avg={results['TC'].mean()*1e4:.2f})")
    plt.tight_layout()
    save_fig("P09_turnover_tc", fig)
except Exception as exc:
    save_message_plot("P09_turnover_tc", "Turnover & TC", f"Error: {exc}")


# ── P10: Return distribution ─────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(13, 5))
    for nm, c, co in [
        ("PRISM No-Lev", "Strat_NoLev", "#1f77b4"),
        ("EW",           "EW",          "#ff7f0e"),
        ("SPY",          "SPY",         "#2ca02c"),
    ]:
        ax.hist(results[c].dropna(), bins=80, alpha=0.4, label=nm, color=co, density=True)
    ax.set_title("Weekly Return Distribution (No Leverage)")
    ax.legend()
    save_fig("P10_ret_dist", fig)
except Exception as exc:
    save_message_plot("P10_ret_dist", "Return Distribution", f"Error: {exc}")


# ── P11: QQ plots ────────────────────────────────────────────────────────
try:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (nm, c) in zip(axes, [
        ("PRISM No-Lev", "Strat_NoLev"),
        ("EW",           "EW"),
        ("SPY",          "SPY"),
    ]):
        stats.probplot(results[c].dropna().values, dist="norm", plot=ax)
        ax.set_title(f"QQ — {nm}")
    plt.tight_layout()
    save_fig("P11_qq", fig)
except Exception as exc:
    save_message_plot("P11_qq", "QQ Plots", f"Error: {exc}")


# ── P12: Monthly heatmap ─────────────────────────────────────────────────
try:
    mo = results["Strat_NoLev"].resample("M").apply(lambda x: (1 + x).prod() - 1)
    mh = pd.DataFrame({"Y": mo.index.year, "M": mo.index.month, "R": mo.values})
    piv = mh.pivot_table(values="R", index="Y", columns="M", aggfunc="first")
    piv = piv.reindex(columns=list(range(1, 13)))

    fig, ax = plt.subplots(figsize=(14, 7))
    arr = piv.values
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(arr), vmax=np.nanmax(arr))
    ax.set_title("Monthly Returns Heatmap — PRISM No-Lev")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(y) for y in piv.index])
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Monthly return")
    save_fig("P12_monthly", fig)
except Exception as exc:
    save_message_plot("P12_monthly", "Monthly Heatmap", f"Error: {exc}")


# ── P13: κ evolution ─────────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(15, 6))
    for tk in kdf.columns[:10]:
        ax.plot(kdf.index, kdf[tk], lw=1, alpha=0.8, label=tk)
    ax.set_title("Confidence κ_i (10 tickers)")
    ax.legend(fontsize=7, ncol=3)
    save_fig("P13_kappa", fig)
except Exception as exc:
    save_message_plot("P13_kappa", "Kappa Evolution", f"Error: {exc}")


# ── P14: Avg κ + predicted vol ───────────────────────────────────────────
try:
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    axes[0].plot(kdf.index, kdf.mean(axis=1), color="purple", lw=1.5)
    axes[0].set_title("Average κ")
    if "PredVol" in results.columns:
        axes[1].plot(results.index, results["PredVol"], color="darkred", lw=1.5)
    axes[1].axhline(TARGET_VOL, color="red", ls=":", lw=2,
                    label=f"Target {TARGET_VOL:.0%}")
    axes[1].set_title("Predicted Annual Volatility")
    axes[1].legend()
    plt.tight_layout()
    save_fig("P14_kappa_vol", fig)
except Exception as exc:
    save_message_plot("P14_kappa_vol", "Avg kappa + Pred Vol", f"Error: {exc}")


# ── P15: μ^B heatmap ─────────────────────────────────────────────────────
try:
    arr = (mdf.iloc[::4].T * 100).values
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanpercentile(arr, 5), vmax=np.nanpercentile(arr, 95))
    ax.set_title("μ^B (% weekly)")
    ax.set_yticks(np.arange(len(mdf.columns)))
    ax.set_yticklabels(mdf.columns, fontsize=7)
    ax.set_xticks([])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    save_fig("P15_muB", fig)
except Exception as exc:
    save_message_plot("P15_muB", "muB Heatmap", f"Error: {exc}")


# ── P16: σ heatmap ───────────────────────────────────────────────────────
try:
    arr = (sdf.iloc[::4].T * 100).values
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd",
                   vmin=np.nanpercentile(arr, 5), vmax=np.nanpercentile(arr, 95))
    ax.set_title("σ (% weekly)")
    ax.set_yticks(np.arange(len(sdf.columns)))
    ax.set_yticklabels(sdf.columns, fontsize=7)
    ax.set_xticks([])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    save_fig("P16_sigma", fig)
except Exception as exc:
    save_message_plot("P16_sigma", "Sigma Heatmap", f"Error: {exc}")


# ── P17: Predicted corr from μ^B ─────────────────────────────────────────
try:
    window = min(52, len(mdf))
    corr = mdf.tail(window).corr()
    last_kappa = kdf.iloc[-1].sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    im = axes[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_title("Predicted Corr from μ^B (last 52w)")
    axes[0].set_xticks(np.arange(len(corr.columns)))
    axes[0].set_yticks(np.arange(len(corr.index)))
    axes[0].set_xticklabels(corr.columns, rotation=90, fontsize=6)
    axes[0].set_yticklabels(corr.index, fontsize=6)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    colors = plt.cm.viridis(np.clip(last_kappa.values, 0, 1))
    axes[1].barh(last_kappa.index, last_kappa.values, color=colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Latest κ Snapshot")
    axes[1].tick_params(labelsize=7)
    save_fig("P17_final_corr", fig)
except Exception as exc:
    save_message_plot("P17_final_corr", "Final Corr + kappa", f"Error: {exc}")


def lag1_r2(s: pd.Series) -> float:
    ss = s.dropna()
    if len(ss) < 10:
        return np.nan
    x = ss.shift(1).dropna()
    y = ss.loc[x.index]
    if len(x) < 10 or x.std() == 0 or y.std() == 0:
        return np.nan
    r = x.corr(y)
    return float(r * r) if pd.notna(r) else np.nan


def lag1_da(s: pd.Series) -> float:
    ss = s.dropna()
    if len(ss) < 10:
        return np.nan
    x = np.sign(ss.shift(1).dropna())
    y = np.sign(ss.loc[x.index])
    return float((x == y).mean())


# ── P18: Return signal R² by ticker ─────────────────────────────────────
try:
    r2_vals = pd.Series(
        {tk: lag1_r2(mdf[tk]) for tk in mdf.columns}
    ).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(r2_vals.index, r2_vals.values, color="#1f77b4")
    ax.set_title("Return Signal R² by Ticker (lag-1 μ_B)")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.set_ylabel("R²")
    save_fig("P18_ret_r2", fig)
except Exception as exc:
    save_message_plot("P18_ret_r2", "Return R²", f"Error: {exc}")


# ── P19: Directional accuracy by ticker ──────────────────────────────────
try:
    da_vals = pd.Series(
        {tk: lag1_da(mdf[tk]) for tk in mdf.columns}
    ).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(da_vals.index, da_vals.values, color="#2ca02c")
    ax.axhline(0.5, color="red", ls=":", lw=1.5)
    ax.set_ylim(0, 1)
    ax.set_title("Directional Accuracy by Ticker (sign persistence in μ_B)")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.set_ylabel("DA")
    save_fig("P19_ret_da", fig)
except Exception as exc:
    save_message_plot("P19_ret_da", "Return DA", f"Error: {exc}")


# ── P20: Mean allocation weights by asset ────────────────────────────────
try:
    mean_abs    = wdf.abs().mean().sort_values(ascending=False)
    mean_signed = wdf.mean().reindex(mean_abs.index)

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(mean_abs))
    ax.bar(x - 0.2, mean_abs.values,    width=0.4, label="Mean |w|", color="#9467bd")
    ax.bar(x + 0.2, mean_signed.values, width=0.4, label="Mean w",   color="#17becf")
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_title("Mean Allocation Weights by Asset")
    ax.set_xticks(x)
    ax.set_xticklabels(mean_abs.index, rotation=75, fontsize=7)
    ax.legend()
    save_fig("P20_meta_w", fig)
except Exception as exc:
    save_message_plot("P20_meta_w", "Mean Weights", f"Error: {exc}")


# ── P21: Pair diagnostics from μ_B ───────────────────────────────────────
try:
    cols = list(mdf.columns)
    r2_list: list[float] = []
    da_list: list[float] = []
    mae_list: list[float] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair = pd.concat([mdf[cols[i]], mdf[cols[j]]], axis=1).dropna()
            if len(pair) < 10:
                continue
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            r2_list.append(float(r * r) if pd.notna(r) else np.nan)
            da_list.append(
                float((np.sign(pair.iloc[:, 0]) == np.sign(pair.iloc[:, 1])).mean())
            )
            mae_list.append(float((pair.iloc[:, 0] - pair.iloc[:, 1]).abs().mean()))

    if not r2_list:
        raise ValueError("Not enough pair history for diagnostics")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(r2_list,  bins=30, alpha=0.8, color="steelblue")
    axes[0].set_title(f"Pair R² (med={np.nanmedian(r2_list):.3f})")
    axes[1].hist(da_list,  bins=30, alpha=0.8, color="seagreen")
    axes[1].set_title(f"Pair DA (med={np.nanmedian(da_list):.3f})")
    axes[2].hist(mae_list, bins=30, alpha=0.8, color="coral")
    axes[2].set_title(f"Pair MAE (med={np.nanmedian(mae_list):.4f})")
    fig.suptitle("Pair Diagnostics from μ_B")
    plt.tight_layout()
    save_fig("P21_pair_diag", fig)
except Exception as exc:
    save_message_plot("P21_pair_diag", "Pair Diagnostics", f"Error: {exc}")


# ── P22: Weight heatmap ──────────────────────────────────────────────────
try:
    arr = wdf.iloc[::2].T.values
    lim = max(abs(np.nanpercentile(arr, 5)), abs(np.nanpercentile(arr, 95)))
    fig, ax = plt.subplots(figsize=(17, 8))
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title("Weights Heatmap (long=blue, short=red)")
    ax.set_yticks(np.arange(len(wdf.columns)))
    ax.set_yticklabels(wdf.columns, fontsize=7)
    ax.set_xticks([])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    save_fig("P22_w_heatmap", fig)
except Exception as exc:
    save_message_plot("P22_w_heatmap", "Weight Heatmap", f"Error: {exc}")


# ── P23: Annual returns ───────────────────────────────────────────────────
try:
    ydf = pd.DataFrame({
        "PRISM No-Lev": results["Strat_NoLev"].resample("A").apply(
            lambda x: (1 + x).prod() - 1),
        "EW":  results["EW"].resample("A").apply(lambda x: (1 + x).prod() - 1),
        "SPY": results["SPY"].resample("A").apply(lambda x: (1 + x).prod() - 1),
    })
    ydf.index = ydf.index.year
    fig, ax = plt.subplots(figsize=(14, 6))
    ydf.plot(kind="bar", ax=ax, width=0.75)
    ax.axhline(0, color="grey", ls="--")
    ax.set_title("Annual Returns")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    save_fig("P23_annual", fig)
except Exception as exc:
    save_message_plot("P23_annual", "Annual Returns", f"Error: {exc}")


# ── P24: Underwater ──────────────────────────────────────────────────────
try:
    cu = (1 + results["Strat_NoLev"]).cumprod()
    pk = cu.cummax()
    uw = cu / pk - 1
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.fill_between(uw.index, uw.values, 0, alpha=0.6, color="#d62728")
    ax.set_title("PRISM No-Lev — Underwater (Drawdown)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    save_fig("P24_underwater", fig)
except Exception as exc:
    save_message_plot("P24_underwater", "Underwater", f"Error: {exc}")


# ── P25: Average σ across assets ─────────────────────────────────────────
try:
    avg_sigma = sdf.mean(axis=1) * np.sqrt(52) * 100
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(avg_sigma.index, avg_sigma.values, color="darkred", lw=1.2)
    ax.set_title("Average σ Across Assets (annualised, %)")
    ax.set_ylabel("Percent")
    save_fig("P25_gk_avg", fig)
except Exception as exc:
    save_message_plot("P25_gk_avg", "Avg Sigma", f"Error: {exc}")


# ── P26: All assets normalised cumulative from μ^B ───────────────────────
try:
    cum_assets = (1 + mdf.fillna(0)).cumprod()
    normed = cum_assets / cum_assets.iloc[0]
    fig, ax = plt.subplots(figsize=(16, 8))
    for tk in normed.columns:
        ax.plot(normed.index, normed[tk], lw=1, alpha=0.8, label=tk)
    ax.set_yscale("log")
    ax.set_title("All Assets — Normalised Cumulative Signal from μ^B (log)")
    ax.legend(fontsize=6, ncol=5, loc="upper left")
    save_fig("P26_all_assets_log", fig)
except Exception as exc:
    save_message_plot("P26_all_assets_log", "All Assets", f"Error: {exc}")


# ── P27: Individual asset signal panels ──────────────────────────────────
try:
    normed = (1 + mdf.fillna(0)).cumprod()
    normed = normed / normed.iloc[0]
    tickers = list(normed.columns)
    ncols = 4
    nrows = int(np.ceil(len(tickers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.5 * nrows), sharex=True)
    axes_flat = np.array(axes).reshape(-1)

    for i, tk in enumerate(tickers):
        ax = axes_flat[i]
        ax.plot(normed.index, normed[tk], lw=1, color="#1f77b4")
        ax.set_title(tk, fontsize=9)
        ax.set_yscale("log")

    for j in range(len(tickers), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Individual Asset Signal Evolution from μ^B (log normalised)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig("P27_individual_assets", fig)
except Exception as exc:
    save_message_plot("P27_individual_assets", "Asset Panels", f"Error: {exc}")


# ── P28: Strategy vs EW vs SPY — No-Lev @15% ────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(16, 7))
    for nm, c, co, lw in [
        ("PRISM No-Lev", "Strat_NoLev", "#1f77b4", 3),
        ("EW No-Lev",    "EW_NoLev",   "#ff7f0e", 2),
        ("SPY No-Lev",   "SPY_NoLev",  "#2ca02c", 2),
    ]:
        cu = (1 + results[c]).cumprod()
        ax.plot(cu.index, cu.values, label=nm, color=co, lw=lw)
    ax.set_yscale("log")
    ax.set_title("Strategy vs EW vs SPY — Vol-Fitted @15%, No Leverage", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylabel("Growth of $1", fontsize=11)
    ax.grid(True, alpha=0.3)
    save_fig("P28_strat_vs_bench_15", fig)
except Exception as exc:
    save_message_plot("P28_strat_vs_bench_15", "Strat vs Bench", f"Error: {exc}")


# ── P29: Execution summary ───────────────────────────────────────────────
try:
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].bar(results.index, results["Turnover"], width=5, alpha=0.6, color="navy")
    axes[0].set_title(f"Turnover per Rebalance (avg={results['Turnover'].mean():.3f})")

    tc_bps = results["TC"] * 1e4
    axes[1].bar(results.index, tc_bps, width=5, alpha=0.6, color="tomato")
    axes[1].set_title(f"Transaction Cost (bps, avg={tc_bps.mean():.2f})")

    axes[2].plot(results.index, results["Gross"], color="teal", lw=1.2)
    axes[2].set_title("Gross Leverage")

    fig.tight_layout()
    save_fig("P29_ob_summary", fig)
except Exception as exc:
    save_message_plot("P29_ob_summary", "Execution Summary", f"Error: {exc}")


# ── P30: Estimated spread history ────────────────────────────────────────
try:
    spread_bps = np.where(
        results["Turnover"].values > 1e-10,
        (results["TC"].values / results["Turnover"].values) * 1e4,
        np.nan,
    )
    spread_s = pd.Series(spread_bps, index=results.index).replace(
        [np.inf, -np.inf], np.nan
    )
    smooth = spread_s.rolling(8, min_periods=1).median()

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(spread_s.index, spread_s.values,
            color="purple", lw=0.8, alpha=0.5, label="Raw")
    ax.plot(smooth.index, smooth.values,
            color="black", lw=1.5, label="8w median")
    ax.set_title("Estimated Spread (bps)")
    ax.set_ylabel("bps")
    ax.legend()
    save_fig("P30_spread_hist", fig)
except Exception as exc:
    save_message_plot("P30_spread_hist", "Spread History", f"Error: {exc}")


# ── P31: Net trades — last rebalance ─────────────────────────────────────
try:
    if len(wdf) >= 2:
        delta = (wdf.iloc[-1] - wdf.iloc[-2]).sort_values()
        label_date = f"{wdf.index[-2].date()} to {wdf.index[-1].date()}"
    elif len(wdf) == 1:
        delta = wdf.iloc[-1].sort_values()
        label_date = f"snapshot {wdf.index[-1].date()}"
    else:
        raise ValueError("No weights available")

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in delta.values]
    ax.barh(delta.index, delta.values * 100, color=colors)
    ax.axvline(0, color="grey", ls="--")
    ax.set_title(f"Net Trades — Last Rebalance ({label_date})")
    ax.set_xlabel("Δw (%)")
    save_fig("P31_last_trades", fig)
except Exception as exc:
    save_message_plot("P31_last_trades", "Last Trades", f"Error: {exc}")


print(f"\nGenerated : {len(generated)}")
if generated:
    print("  " + ", ".join(generated))
print(f"Skipped   : {len(skipped)}")
if skipped:
    print("  " + ", ".join(skipped))
print("\nDone -- 31 plots, no-leverage normalisation, performance table updated")
