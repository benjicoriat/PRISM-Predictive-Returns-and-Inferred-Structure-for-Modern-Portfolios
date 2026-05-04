#  CELL 15 — SNAPSHOT & LOG DUMP
# ═══════════════════════════════════════════════════════════════════════
latest_lines = ["", "="*60, "LATEST ALLOCATION", "="*60]
if len(wdf)>0:
    lw=wdf.iloc[-1].sort_values(ascending=False)
    latest_lines.append(f"{'Ticker':>12}  {'Weight':>8}  {'Side':>6}")
    latest_lines.append("-"*30)
    for tk,w in lw.items():
        if abs(w)>.001:
            latest_lines.append(f"  {tk:>10}  {w:>7.2%}  {'LONG' if w>0 else 'SHORT':>6}")
    latest_lines.append(
        f"\n  Net: {lw.sum():.2%} | Gross: {lw.abs().sum():.2%} | "
        f"Long: {lw[lw>0].sum():.2%} | Short: {lw[lw<0].sum():.2%}"
    )
latest_txt = "\n".join(latest_lines)
print(latest_txt)
(OUTPUTS_DIR / "latest_allocation.txt").write_text(latest_txt + "\n", encoding="utf-8")

obs_lines = ["", "="*60, "ORDER BOOK SUMMARY", "="*60]
obs = ob_tracker.get_summary_df()
if len(obs)>0:
    obs_lines.append(f"Total rebalances: {len(obs)}")
    obs_lines.append(f"Avg fills/rebalance: {obs['n_fills'].mean():.0f}")
    obs_lines.append(f"Avg notional: ${obs['total_notional'].mean():,.0f}")
    obs_lines.append(f"Avg slippage: ${obs['total_slippage'].mean():,.2f}")
    obs_lines.append(f"Total slippage: ${obs['total_slippage'].sum():,.0f}")
    odf = ob_tracker.get_order_log_df()
    if len(odf)>0:
        obs_lines.append(f"Total order records: {len(odf)}")
        obs_lines.append(f"Avg spread (sell): {odf['sell_spread_bps'].mean():.1f} bps")
        obs_lines.append(f"Avg spread (buy):  {odf['buy_spread_bps'].mean():.1f} bps")
        odf.to_csv(OUTPUTS_DIR / "order_log.csv", index=False)
    obs.to_csv(OUTPUTS_DIR / "order_book_summary.csv")
obs_txt = "\n".join(obs_lines)
print(obs_txt)
(OUTPUTS_DIR / "order_book_summary.txt").write_text(obs_txt + "\n", encoding="utf-8")

dbg_lines = ["", "="*60, "DEBUG LOG (last 80 lines)", "="*60]
dbg_tail = LOG_BUF.getvalue().strip().split("\n")[-80:]
dbg_lines.extend(dbg_tail)
dbg_txt = "\n".join(dbg_lines)
print(dbg_txt)
(OUTPUTS_DIR / "debug_log_last_80.txt").write_text(dbg_txt + "\n", encoding="utf-8")
(OUTPUTS_DIR / "debug_log_full.txt").write_text(LOG_BUF.getvalue(), encoding="utf-8")

done_msg = (
    f"\n✅ PRISM v3 COMPLETE — {len(TICKERS)} tickers, "
    f"{'long-only' if CFG.LONG_ONLY else 'long/short'}, "
    f"{'no-leverage' if CFG.VOL_TARGET_CAP <= 1.0 else f'lev≤{CFG.VOL_TARGET_CAP:g}x'}, "
    f"vol@{CFG.SIGMA_TARGET_ANN:.0%}, 31 plots"
)
print(done_msg)
(OUTPUTS_DIR / "run_complete.txt").write_text(done_msg.strip() + "\n", encoding="utf-8")
