#  CELL 4 — ORDER BOOK TRACKER
# ═══════════════════════════════════════════════════════════════════════
class OrderBookTracker:
    """
    Simulated order book tracker from OHLCV data.
    Reconstructs a synthetic L2 book snapshot per week using:
      - mid price = Close
      - spread estimated from intra-day range (Corwin-Schultz style)
      - depth levels synthesised from volume profile
    Logs every order (fill) from the OT transport plan.
    """
    def __init__(self, tickers):
        self.tickers = tickers
        self.tk_idx = {t:i for i,t in enumerate(tickers)}
        self.book_snapshots = []    # list of dicts per date
        self.order_log = []         # every executed trade
        self.fill_summary = []      # per-rebalance summary
        log.info("OrderBookTracker initialised")

    def build_snapshot(self, date, weekly_ohlc_row, gk_vol_row):
        """Build synthetic L2 snapshot from weekly OHLC."""
        snap = {"date": date, "books": {}}
        for tk in self.tickers:
            try:
                close = weekly_ohlc_row.get(tk, np.nan)
                if np.isnan(close): continue
                gk = gk_vol_row.get(tk, 0.01)
                # Corwin-Schultz spread estimate: spread ≈ 2*(exp(α)-1) where α from HL
                spread_pct = max(gk * 0.1, 0.0001)  # ~10% of weekly vol as spread proxy
                mid = close
                best_bid = mid * (1 - spread_pct/2)
                best_ask = mid * (1 + spread_pct/2)
                # Synthetic depth: 5 levels, geometrically spaced
                levels = 5
                bids, asks = [], []
                for lvl in range(levels):
                    tick = spread_pct * (0.5 + lvl * 0.5)
                    bp = mid * (1 - tick)
                    ap = mid * (1 + tick)
                    # Size decays with depth
                    sz = 1000 * (0.8 ** lvl)
                    bids.append({"price": round(bp, 4), "size": round(sz, 1), "level": lvl})
                    asks.append({"price": round(ap, 4), "size": round(sz, 1), "level": lvl})
                snap["books"][tk] = {
                    "mid": mid, "spread_pct": spread_pct,
                    "best_bid": best_bid, "best_ask": best_ask,
                    "bids": bids, "asks": asks,
                    "gk_vol": gk,
                }
            except (KeyError, ValueError, TypeError) as e:
                log.debug(f"OB snapshot skip {tk} @ {date}: {e}")
                continue
        self.book_snapshots.append(snap)
        return snap

    def log_rebalance(self, date, w_old, w_new, T_star, tickers, portfolio_value=1e6):
        """Log each fill from the OT transport plan."""
        n = len(tickers)
        fills = []
        total_notional = 0
        total_cost_est = 0

        # Get latest book snapshot for spread info
        snap = self.book_snapshots[-1] if self.book_snapshots else None

        for i in range(n):
            for j in range(n):
                if i == j: continue
                flow = T_star[i, j] if T_star is not None else 0
                if abs(flow) < 1e-8: continue
                notional = abs(flow) * portfolio_value
                tk_sell, tk_buy = tickers[i], tickers[j]

                # Estimate fill prices from book
                sell_spread = 0.001
                buy_spread = 0.001
                if snap and tk_sell in snap["books"]:
                    sell_spread = snap["books"][tk_sell]["spread_pct"]
                if snap and tk_buy in snap["books"]:
                    buy_spread = snap["books"][tk_buy]["spread_pct"]

                slip = (sell_spread + buy_spread) / 2 * notional
                fill = {
                    "date": date,
                    "sell_asset": tk_sell, "buy_asset": tk_buy,
                    "flow_pct": flow,
                    "notional": notional,
                    "sell_spread_bps": sell_spread * 1e4,
                    "buy_spread_bps": buy_spread * 1e4,
                    "est_slippage": slip,
                    "side": "SELL→BUY",
                }
                fills.append(fill)
                self.order_log.append(fill)
                total_notional += notional
                total_cost_est += slip

        # Net trades per asset
        net_trades = {}
        for tk_i, (i_idx) in self.tk_idx.items():
            delta = w_new[i_idx] - w_old[i_idx]
            if abs(delta) > 1e-6:
                side = "BUY" if delta > 0 else "SELL"
                net_trades[tk_i] = {
                    "delta_w": delta,
                    "side": side,
                    "notional": abs(delta) * portfolio_value,
                }

        summary = {
            "date": date,
            "n_fills": len(fills),
            "total_notional": total_notional,
            "total_slippage_est": total_cost_est,
            "net_trades": net_trades,
            "turnover_pct": np.sum(np.abs(w_new - w_old)),
        }
        self.fill_summary.append(summary)
        log.debug(f"OB {date}: {len(fills)} fills, ${total_notional:,.0f} notional, "
                  f"~${total_cost_est:,.0f} slippage")
        return summary

    def get_order_log_df(self):
        if not self.order_log: return pd.DataFrame()
        return pd.DataFrame(self.order_log)

    def get_summary_df(self):
        if not self.fill_summary: return pd.DataFrame()
        rows = []
        for s in self.fill_summary:
            rows.append({
                "date": s["date"], "n_fills": s["n_fills"],
                "total_notional": s["total_notional"],
                "total_slippage": s["total_slippage_est"],
                "turnover": s["turnover_pct"],
                "n_net_trades": len(s["net_trades"]),
            })
        return pd.DataFrame(rows).set_index("date")

    def get_spread_history(self):
        """Extract spread time series per ticker."""
        rows = []
        for snap in self.book_snapshots:
            for tk, bk in snap["books"].items():
                rows.append({"date": snap["date"], "ticker": tk,
                             "spread_bps": bk["spread_pct"]*1e4, "mid": bk["mid"]})
        return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════
