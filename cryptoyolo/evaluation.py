"""Did the engine actually work? — the feedback loop the rest of the system needs.

Every run writes its scores, proposals and orders to SQLite. This module is
what closes the loop on that data:

  forward_returns    join each stored score to the asset's realised return over
                     the next 1 / 7 / 30 days
  information_coeff   rank-correlation of each feature family with those returns,
                     per run and pooled, with a t-stat so you can tell signal
                     from noise
  quantile_spread     top-N minus bottom-N composite: does the ranking separate
                     winners from losers at all?
  realized_trades     reconstruct round-trip trades from the order log, net of
                     fees — hit rate, avg win / avg loss, profit factor
  benchmark_returns   BTC buy-and-hold and an equal-weight basket over the same
                     window: the bar every strategy has to clear
  equity_stats        the mark-to-market equity curve — total return, Sharpe,
                     max drawdown — vs that benchmark
  recommend_weights   turn the measured ICs into a data-driven ScoreWeights to
                     replace the hand-set priors

None of this fits weights automatically. It gives you the numbers; you decide.
With only a couple of weeks of data every estimate here is noisy — treat a
t-stat under ~2 as "no evidence either way", not as a finding.

Forward returns and the BTC/basket benchmark are computed from the local
`prices_daily` table (written each cycle by `engine.build_scores`, seeded by
`backfill_prices.py`), not from a fresh API pull. That makes every number here
reproducible run-to-run and lets the module run with no network. If the table
has no history for a symbol it transparently falls back to a live fetch and
stores the result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import prices as prices_mod
from .config import CONFIG, Config, ScoreWeights
from .store import Store

FAMILIES = ("technical", "social", "catalyst", "positioning", "events",
           "kalshi_prediction", "composite")


# --------------------------------------------------------------------------
# Price history (cached per process)
# --------------------------------------------------------------------------
# Keyed on whether a Store was available, because the two paths can return
# different series (local table vs live API) and must not alias each other.
_PX_CACHE: dict[tuple[str, str, bool], pd.Series | None] = {}
_MIN_SERIES = 5


def _as_ns(s: pd.Series) -> pd.Series:
    """Pin to a ns DatetimeIndex, dedupe, sort.

    Price sources build their index from unix ms, so it can land at ms/us
    resolution; score timestamps are ns and pandas 3 refuses a lossy asof
    across resolutions.
    """
    out = pd.Series(s.astype(float).to_numpy(),
                    index=pd.DatetimeIndex(pd.to_datetime(s.index, utc=True)).as_unit("ns"))
    return out[~out.index.duplicated(keep="last")].sort_index()


def _series_from_store(store: Store | None, symbol: str) -> pd.Series | None:
    if store is None:
        return None
    try:
        df = store.daily_prices(symbol)
    except Exception:  # noqa: BLE001 - a missing/locked table is just "no local data"
        return None
    if df is None or df.empty:
        return None
    s = _as_ns(pd.Series(df["close"].to_numpy(), index=df["date"]))
    return s if len(s) >= _MIN_SERIES else None


def price_series(symbol: str, cfg: Config = CONFIG, timeframe: str = "1y",
                 store: Store | None = None) -> pd.Series | None:
    """Daily close series for one asset, or None if it can't be sourced.

    Prefers the local `prices_daily` table when a `store` is given and holds
    enough history — that path is deterministic and needs no network. Otherwise
    it fetches from the live price sources and, if a store was supplied,
    persists what it got so the next call is served locally.

    Cached on (symbol, timeframe, db path) — keyed by the store's OWN identity,
    not just whether one was passed, so two Store instances pointing at
    different databases (e.g. an interactive session switched to a different
    db_path) never alias each other's cached series.
    """
    key = (symbol, timeframe, str(store.db_path) if store is not None else None)
    if key in _PX_CACHE:
        return _PX_CACHE[key]

    s = _series_from_store(store, symbol)
    if s is None:
        try:
            df, src = prices_mod.get_ohlcv(symbol, timeframe, cfg)
        except Exception:  # noqa: BLE001 - an asset with no history is just skipped
            _PX_CACHE[key] = None
            return None
        s = _as_ns(df["close"])
        if store is not None:
            try:
                store.upsert_daily_prices_from_series(symbol, s, src)
            except Exception:  # noqa: BLE001 - persistence is best-effort
                pass
    _PX_CACHE[key] = s
    return s


def clear_cache() -> None:
    _PX_CACHE.clear()


def _fwd_return(s: pd.Series | None, t0: pd.Timestamp, days: float) -> float:
    """Return from the first close AFTER t0 to the first close >= entry + days.

    The entry deliberately snaps *forward* to the next bar, not back to the last
    one: a daily close stamped 00:00 UTC is the price ~11h after a midday score,
    so `asof(t0)` would fold nearly a day of hindsight into the entry leg and
    bias every IC upward. NaN when there is no bar after t0, or the exit bar
    hasn't happened yet.
    """
    if s is None or s.empty:
        return np.nan
    t0 = pd.Timestamp(t0).tz_convert("UTC").as_unit("ns")
    i0 = s.index.searchsorted(t0, side="right")      # first bar strictly after t0
    if i0 >= len(s):
        return np.nan
    p0 = float(s.iloc[i0])
    t1 = (s.index[i0] + pd.Timedelta(days=days)).as_unit("ns")
    i1 = s.index.searchsorted(t1, side="left")       # first bar at/after entry+days
    if i1 >= len(s) or p0 <= 0:
        return np.nan
    p1 = float(s.iloc[i1])
    if not np.isfinite(p0) or not np.isfinite(p1):
        return np.nan
    return float(p1 / p0 - 1.0)


# --------------------------------------------------------------------------
# Forward returns joined to stored scores
# --------------------------------------------------------------------------
def forward_returns(store: Store, cfg: Config = CONFIG,
                    horizons: tuple[int, ...] = (1, 7, 30),
                    since: datetime | None = None,
                    refresh: bool = True) -> pd.DataFrame:
    """One row per stored score, with the realised forward return beside it.

    Columns: run_id, ts, symbol, <each family>, fwd_1d / fwd_7d / fwd_30d.
    Old rows that predate the positioning/events/kalshi_prediction columns
    have those families backfilled from the components JSON where possible.

    `refresh` (default True) drops the price cache first: in a long-lived
    notebook kernel it would otherwise be stale by days after more cycles run.
    """
    if refresh:
        clear_cache()
    sc = store.scores_history(since)
    if sc.empty:
        return pd.DataFrame()

    # Parse each row's components JSON once, then read whatever families the
    # dedicated columns don't already carry.
    backfill = ("positioning", "events", "kalshi_prediction", "xsec")
    if any(sc[f].isna().any() for f in backfill if f in sc.columns):
        parsed = sc["components"].apply(
            lambda c: json.loads(c) if isinstance(c, str) and c else {})
        for fam in backfill:
            col = sc[fam] if fam in sc.columns else pd.Series(np.nan, index=sc.index)
            from_json = parsed.apply(
                lambda d, f=fam: float(d[f][f]) if isinstance(d.get(f), dict)
                and f in d[f] else np.nan)
            sc[fam] = col.where(col.notna(), from_json)

    symbols = sorted(sc["symbol"].unique())
    series = {sym: price_series(sym, cfg, store=store) for sym in symbols}

    keep = ["run_id", "ts", "symbol", *[f for f in FAMILIES if f in sc.columns]]
    out = sc[keep].copy()
    for h in horizons:
        out[f"fwd_{h}d"] = [
            _fwd_return(series.get(sym), ts, h)
            for sym, ts in zip(sc["symbol"], sc["ts"])
        ]
    return out


# --------------------------------------------------------------------------
# Information coefficient
# --------------------------------------------------------------------------
def _spearman(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return np.nan
    av, bv = a[m], b[m]
    if av.nunique() < 2 or bv.nunique() < 2:
        return np.nan
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(av, bv)
        return float(rho)
    except Exception:  # noqa: BLE001 - fall back to pandas rank corr
        return float(av.rank().corr(bv.rank()))


def _overlap_factor(run_ids: pd.Index, run_ts: pd.Series, horizon: int) -> float:
    """How many consecutive per-run ICs share a forward window.

    Runs a day apart with a 7-day forward return are ~7x autocorrelated, so the
    naive t-stat (which assumes independent runs) is inflated by ~sqrt(7). This
    returns that inflation factor so it can be divided back out.
    """
    ts = run_ts.reindex(run_ids).dropna().sort_values()
    if len(ts) < 3:
        return 1.0
    span_days = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0
    gap = span_days / max(len(ts) - 1, 1)
    return max(1.0, horizon / max(gap, 1e-9))


def information_coefficient(fr: pd.DataFrame,
                            horizons: tuple[int, ...] = (1, 7, 30)
                            ) -> pd.DataFrame:
    """Rank-correlation of each family with forward return, per run then averaged.

    The per-run average is the honest point estimate: "within one cross-section,
    did a higher score mean a higher return?", one observation per run.
    `pooled_IC` pools every row and looks more significant than it is.

    `t_stat` is deflated for overlapping forward windows (`runs_eff` shows the
    effective independent-sample count) — consecutive daily runs sharing a 7- or
    30-day return are not independent, so the raw t-stat overstates confidence.
    """
    if fr.empty:
        return pd.DataFrame()
    fams = [f for f in FAMILIES if f in fr.columns]
    run_ts = fr.groupby("run_id")["ts"].min() if "ts" in fr.columns else pd.Series(dtype=object)
    rows: list[dict[str, Any]] = []
    for h in horizons:
        ycol = f"fwd_{h}d"
        if ycol not in fr.columns:
            continue
        for fam in fams:
            per_run = (
                fr.groupby("run_id")[[fam, ycol]]
                .apply(lambda g, f=fam: _spearman(g[f], g[ycol]))
                .dropna()
            )
            pooled = _spearman(fr[fam], fr[ycol])
            if per_run.empty:
                rows.append({"family": fam, "horizon_d": h, "runs": 0, "runs_eff": 0.0,
                             "mean_IC": np.nan, "t_stat": np.nan,
                             "pct_runs_IC_pos": np.nan,
                             "pooled_IC": round(pooled, 4) if pd.notna(pooled) else np.nan})
                continue
            n = len(per_run)
            mean_ic = float(per_run.mean())
            std_ic = float(per_run.std(ddof=1)) if n > 1 else np.nan
            overlap = _overlap_factor(per_run.index, run_ts, h) if not run_ts.empty else 1.0
            n_eff = n / overlap
            # Need at least ~2 effective independent observations before a
            # std-based t-stat means anything; below that it is left blank.
            t = (mean_ic / (std_ic / np.sqrt(n_eff))
                 if std_ic and std_ic > 0 and n_eff >= 2.0 else np.nan)
            rows.append({
                "family": fam, "horizon_d": h, "runs": n,
                "runs_eff": round(n_eff, 1),
                "mean_IC": round(mean_ic, 4),
                "t_stat": round(t, 2) if pd.notna(t) else np.nan,
                "pct_runs_IC_pos": round(float((per_run > 0).mean()) * 100, 1),
                "pooled_IC": round(pooled, 4) if pd.notna(pooled) else np.nan,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Quantile spread and hit rate
# --------------------------------------------------------------------------
def quantile_spread(fr: pd.DataFrame, horizon: int = 7, n: int = 3,
                    score_col: str = "composite") -> dict[str, float]:
    ycol = f"fwd_{horizon}d"
    if fr.empty or ycol not in fr.columns or score_col not in fr.columns:
        return {}
    tops, bots, alls = [], [], []
    for _, g in fr.groupby("run_id"):
        g = g.dropna(subset=[score_col, ycol])
        if len(g) < 2 * n:
            continue
        g = g.sort_values(score_col, ascending=False)
        tops.append(g[ycol].head(n).mean())
        bots.append(g[ycol].tail(n).mean())
        alls.append(g[ycol].mean())
    if not tops:
        return {}
    return {
        "runs": len(tops), "horizon_d": horizon, "n_per_side": n,
        "top_mean_ret": float(np.mean(tops)),
        "bottom_mean_ret": float(np.mean(bots)),
        "spread": float(np.mean(tops) - np.mean(bots)),
        "universe_mean_ret": float(np.mean(alls)),
        "top_minus_universe": float(np.mean(tops) - np.mean(alls)),
    }


def hit_rate(fr: pd.DataFrame, horizon: int = 7, floor: float = 0.03,
             score_col: str = "composite") -> dict[str, float]:
    ycol = f"fwd_{horizon}d"
    if fr.empty or ycol not in fr.columns:
        return {}
    d = fr.dropna(subset=[score_col, ycol])
    bull = d[d[score_col] > floor]
    bear = d[d[score_col] < -floor]
    return {
        "horizon_d": horizon,
        "bullish_calls": int(len(bull)),
        "bullish_hit_rate": float((bull[ycol] > 0).mean()) if len(bull) else np.nan,
        "bearish_calls": int(len(bear)),
        "bearish_hit_rate": float((bear[ycol] < 0).mean()) if len(bear) else np.nan,
    }


# --------------------------------------------------------------------------
# Realised trades from the order log
# --------------------------------------------------------------------------
def _order_fee(row: pd.Series, cfg: Config) -> float:
    """Realised fee: the dedicated column, then the response blob, then an
    estimate from the fee config (in that order of trust)."""
    fu = row.get("fee_usd")
    if fu is not None and pd.notna(fu) and float(fu) > 0:
        return abs(float(fu))
    try:
        resp = json.loads(row["response"] or "{}")
        if isinstance(resp.get("fee"), (int, float)):
            return abs(float(resp["fee"]))
    except Exception:  # noqa: BLE001
        pass
    notional = abs(float(row["qty"]) * float(row["price"]))
    return notional * cfg.fees.effective_taker_bps / 10_000.0


def realized_trades(store: Store, cfg: Config = CONFIG) -> pd.DataFrame:
    """FIFO round-trip reconstruction of every closed trade, net of fees.

    Paper and live fills are kept separate (`mode`) because their cost models
    differ. Open lots left at the end are reported with `close_ts` NaT and
    marked to the latest price.
    """
    orders = store.all_orders()
    if orders.empty:
        return pd.DataFrame()
    orders = orders[orders["status"].isin(["FILLED", "SENT"])].copy()
    if orders.empty:
        return pd.DataFrame()

    lots: dict[tuple, list[dict]] = {}
    closed: list[dict] = []
    for _, o in orders.iterrows():
        qty, px = float(o["qty"]), float(o["price"])
        if qty <= 0 or px <= 0:
            continue
        fee = _order_fee(o, cfg)
        key = (o["symbol"], "paper" if "paper" in str(o["mode"]) else "live")
        book = lots.setdefault(key, [])
        if o["side"].upper() == "BUY":
            book.append({"qty": qty, "px": px, "ts": o["ts"], "fee": fee})
            continue
        remaining = qty
        sell_fee_unit = fee / qty if qty else 0.0
        while remaining > 1e-12 and book:
            lot = book[0]
            take = min(remaining, lot["qty"])
            entry_fee_alloc = lot["fee"] * (take / lot["qty"]) if lot["qty"] else 0.0
            gross = take * (px - lot["px"])
            fees = entry_fee_alloc + sell_fee_unit * take
            closed.append({
                "symbol": o["symbol"], "mode": key[1],
                "open_ts": lot["ts"], "close_ts": o["ts"], "qty": take,
                "entry": lot["px"], "exit": px,
                "gross_pnl": gross, "fees": fees, "net_pnl": gross - fees,
                "ret_pct": (px / lot["px"] - 1) * 100,
                "hold_days": (o["ts"] - lot["ts"]).total_seconds() / 86400.0,
            })
            lot["qty"] -= take
            lot["fee"] -= entry_fee_alloc
            remaining -= take
            if lot["qty"] <= 1e-12:
                book.pop(0)

    open_syms = sorted({k[0] for k, v in lots.items()
                        if any(l["qty"] > 1e-12 for l in v)})
    last_px = prices_mod.latest_prices(open_syms, cfg) if open_syms else {}
    for (sym, mode), book in lots.items():
        for lot in book:
            if lot["qty"] <= 1e-12:
                continue
            px = last_px.get(sym)
            gross = lot["qty"] * (px - lot["px"]) if px else np.nan
            closed.append({
                "symbol": sym, "mode": mode, "open_ts": lot["ts"], "close_ts": pd.NaT,
                "qty": lot["qty"], "entry": lot["px"], "exit": px,
                "gross_pnl": gross, "fees": lot["fee"],
                "net_pnl": (gross - lot["fee"]) if px else np.nan,
                "ret_pct": (px / lot["px"] - 1) * 100 if px else np.nan,
                "hold_days": (datetime.now(timezone.utc) - lot["ts"]).total_seconds() / 86400.0,
            })
    df = pd.DataFrame(closed)
    return df.sort_values("open_ts").reset_index(drop=True) if not df.empty else df


def trade_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """Win rate, avg win/loss, profit factor and fee drag, per mode and overall."""
    if trades.empty:
        return pd.DataFrame()
    closed = trades[trades["close_ts"].notna()].copy()
    out: list[dict[str, Any]] = []
    groups = [("all", closed)] + [(m, g) for m, g in closed.groupby("mode")]
    for name, g in groups:
        g = g.dropna(subset=["net_pnl"])
        if g.empty:
            continue
        wins = g[g["net_pnl"] > 0]["net_pnl"]
        losses = g[g["net_pnl"] <= 0]["net_pnl"]
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        out.append({
            "book": name,
            "trades": int(len(g)),
            "win_rate_%": round(float((g["net_pnl"] > 0).mean()) * 100, 1),
            "avg_win": round(float(wins.mean()) if len(wins) else 0.0, 2),
            "avg_loss": round(float(losses.mean()) if len(losses) else 0.0, 2),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else np.inf,
            "expectancy": round(float(g["net_pnl"].mean()), 2),
            "net_pnl": round(float(g["net_pnl"].sum()), 2),
            "total_fees": round(float(g["fees"].sum()), 2),
            "fees_%_of_gross": (round(float(g["fees"].sum()) / abs(float(g["gross_pnl"].sum())) * 100, 1)
                                if abs(float(g["gross_pnl"].sum())) > 1e-9 else np.nan),
            "avg_hold_days": round(float(g["hold_days"].mean()), 1),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------
def benchmark_returns(cfg: Config = CONFIG, start: datetime | None = None,
                      end: datetime | None = None,
                      symbols: list[str] | None = None,
                      store: Store | None = None) -> dict[str, Any]:
    """Buy-and-hold BTC and an equal-weight basket over [start, end].

    Reads prices from `store` (the local `prices_daily` table) when given, so
    the benchmark a run is judged against is the same one every time it's
    recomputed.
    """
    symbols = symbols or list(cfg.symbols)
    btc = price_series("BTC", cfg, store=store)
    if btc is None or btc.empty:
        return {}

    def _norm(t, default):
        if t is None:
            return default
        t = pd.Timestamp(t)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return t.as_unit("ns")

    start = _norm(start, btc.index.min())
    end = _norm(end, btc.index.max())

    def _ret(sym: str) -> float:
        s = price_series(sym, cfg, store=store)
        if s is None or s.empty:
            return np.nan
        p0, p1 = s.asof(start), s.asof(end)
        return float(p1 / p0 - 1.0) if np.isfinite(p0) and np.isfinite(p1) and p0 > 0 else np.nan

    basket = [_ret(s) for s in symbols]
    basket = [b for b in basket if pd.notna(b)]
    return {
        "start": start, "end": end,
        "btc_buy_hold_%": round(_ret("BTC") * 100, 2),
        "equal_weight_basket_%": round(float(np.mean(basket)) * 100, 2) if basket else np.nan,
        "basket_n": len(basket),
    }


def _drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    return float((curve / peak - 1.0).min() * 100)


def _resolve_equity_mode(store: Store, cfg: Config, mode: str | None) -> tuple[str | None, str]:
    """Pick ONE snapshot mode. Blending paper (~$10k) and live (real account)
    rows into one curve produces a nonsense Sharpe/return, so never do it.
    """
    if mode is not None:
        return mode, ""
    all_ec = store.equity_curve(None).dropna(subset=["equity"])
    modes = list(all_ec["mode"].dropna().unique()) if not all_ec.empty else []
    if len(modes) <= 1:
        return (modes[0] if modes else None), ""
    prefer = cfg.execution.mode                      # "paper" | "binance"
    if prefer in modes and (all_ec["mode"] == prefer).sum() >= 2:
        return prefer, f"snapshots exist for {modes}; showing '{prefer}' (current mode)"
    fallback = all_ec["mode"].value_counts().idxmax()
    return fallback, f"snapshots exist for {modes}; showing '{fallback}' (most rows)"


def equity_stats(store: Store, cfg: Config = CONFIG,
                 mode: str | None = None) -> dict[str, Any]:
    """Total return, annualised Sharpe and max drawdown of the equity curve,
    with the BTC buy-and-hold return over the same window for comparison.

    Restricted to a single snapshot `mode` — paper and live equity are
    unrelated account sizes and must never share a curve.
    """
    mode, mode_note = _resolve_equity_mode(store, cfg, mode)
    ec = store.equity_curve(mode)
    ec = ec.dropna(subset=["equity"])
    if len(ec) < 2:
        base = ("need at least 2 equity snapshots for this mode — run a few more "
                "cycles (pipeline.run records one per cycle).")
        return {"note": f"{base} {mode_note}".strip(), "mode": mode or "all"}
    ec = ec.sort_values("ts").drop_duplicates(subset=["equity"], keep="first")
    if len(ec) < 2:
        return {"note": "equity has not moved between snapshots yet — nothing to "
                        "annualise. Comes alive once cycles run on different days.",
                "mode": mode or "all"}
    curve = ec.set_index("ts")["equity"].astype(float)
    rets = curve.pct_change().dropna()
    span_days = max((curve.index[-1] - curve.index[0]).total_seconds() / 86400.0, 1e-9)
    gap_days = span_days / max(len(rets), 1)
    periods_per_year = 365.0 / max(gap_days, 1e-9)
    # A Sharpe needs at least a day between observations to mean anything;
    # sub-daily snapshots just inflate periods_per_year.
    sharpe = (float(rets.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year))
              if rets.std(ddof=1) > 0 and span_days >= 1.0 else np.nan)
    total_ret = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
    bench = benchmark_returns(cfg, curve.index[0], curve.index[-1], store=store)
    return {
        "mode": mode or "all",
        "mode_note": mode_note,
        "snapshots": int(len(ec)),
        "span_days": round(span_days, 1),
        "start_equity": round(float(curve.iloc[0]), 2),
        "end_equity": round(float(curve.iloc[-1]), 2),
        "total_return_%": round(total_ret * 100, 2),
        "max_drawdown_%": round(_drawdown(curve), 2),
        "sharpe_annualised": round(sharpe, 2) if pd.notna(sharpe) else np.nan,
        "btc_buy_hold_%": bench.get("btc_buy_hold_%", np.nan),
        "vs_btc_pp": (round(total_ret * 100 - bench["btc_buy_hold_%"], 2)
                      if bench.get("btc_buy_hold_%") is not None else np.nan),
    }


# --------------------------------------------------------------------------
# Weight recommendation
# --------------------------------------------------------------------------
def recommend_weights(fr: pd.DataFrame, horizon: int = 7,
                      min_abs_t: float = 2.0) -> dict[str, Any]:
    """Turn measured ICs into a ScoreWeights suggestion.

    A family gets weight proportional to its mean IC, but only if the IC is
    positive AND its t-stat clears `min_abs_t` (otherwise it is indistinguish-
    able from noise and gets 0). If fewer than two families clear the bar the
    current weights are kept and the function says so — a one-family "portfolio"
    is an overfit to a short sample, not a recommendation.

    Also reports `blend_verdict`: whether the CURRENT composite even beats its
    own best single component. If it doesn't, no re-weighting of the others
    matters until that's fixed — that's the headline.
    """
    ic_all = information_coefficient(fr, (horizon,))
    if ic_all.empty:
        return {"note": "no IC could be computed yet"}
    at_h = ic_all[ic_all["horizon_d"] == horizon]
    comp = at_h[at_h["family"] == "composite"]
    fam = at_h[at_h["family"] != "composite"].dropna(subset=["mean_IC"])
    blend_verdict = ""
    if not comp.empty and not fam.empty:
        comp_ic = float(comp["mean_IC"].iloc[0])
        best_row = fam.loc[fam["mean_IC"].idxmax()]
        best_ic, best_fam = float(best_row["mean_IC"]), str(best_row["family"])
        if pd.notna(comp_ic) and comp_ic < best_ic - 0.01:
            blend_verdict = (f"the blended composite (IC {comp_ic:+.3f}) is WORSE than "
                             f"'{best_fam}' alone (IC {best_ic:+.3f}) at {horizon}d — the "
                             f"weighting is destroying signal, not combining it. Fix that "
                             f"before tuning the rest.")
        elif pd.notna(comp_ic):
            blend_verdict = (f"composite IC {comp_ic:+.3f} ≥ best single family "
                             f"('{best_fam}' {best_ic:+.3f}) — the blend is adding value.")

    ic = at_h[at_h["family"] != "composite"]
    raw: dict[str, float] = {}
    for _, r in ic.iterrows():
        keep = (pd.notna(r["mean_IC"]) and r["mean_IC"] > 0
                and pd.notna(r["t_stat"]) and abs(r["t_stat"]) >= min_abs_t)
        raw[r["family"]] = float(r["mean_IC"]) if keep else 0.0
    total = sum(raw.values())
    n_clear = sum(1 for v in raw.values() if v > 0)
    if total <= 0 or n_clear < 2:
        return {"note": f"{n_clear} of {len(raw)} families have a positive IC "
                        f"significant at |t|>={min_abs_t} for the {horizon}d horizon "
                        f"— not enough to re-weight on. Keep the hand-set weights "
                        f"and collect more data.",
                "ic_table": ic, "blend_verdict": blend_verdict}
    norm = {k: round(v / total, 3) for k, v in raw.items()}
    suggested = ScoreWeights(
        technical=norm.get("technical", 0.0), social=norm.get("social", 0.0),
        catalyst=norm.get("catalyst", 0.0), positioning=norm.get("positioning", 0.0),
        events=norm.get("events", 0.0),
        kalshi_prediction=norm.get("kalshi_prediction", 0.0),
    )
    return {
        "horizon_d": horizon, "ic_table": ic, "normalised": norm,
        "suggested": suggested, "blend_verdict": blend_verdict,
        "code": (f"CONFIG.weights = ScoreWeights(technical={norm.get('technical',0.0)}, "
                 f"social={norm.get('social',0.0)}, catalyst={norm.get('catalyst',0.0)}, "
                 f"positioning={norm.get('positioning',0.0)}, events={norm.get('events',0.0)}, "
                 f"kalshi_prediction={norm.get('kalshi_prediction',0.0)})"),
    }


# --------------------------------------------------------------------------
# Regime-gate effectiveness
# --------------------------------------------------------------------------
def regime_effectiveness(store: Store, cfg: Config = CONFIG,
                         horizon: int = 7) -> pd.DataFrame:
    """Did the regime verdict actually separate good runs from bad ones?

    Joins each cycle's recorded regime `state` to that cycle's realised
    cross-section: the mean forward return of the whole universe and of the
    top-`max_proposals` composite names. If `risk_off` runs really did precede
    weaker forward returns for longs, the gate is earning its keep.
    """
    reg = store.regime_history()
    if reg.empty:
        return pd.DataFrame()
    fr = forward_returns(store, cfg, (horizon,), refresh=False)
    if fr.empty:
        return pd.DataFrame()
    ycol = f"fwd_{horizon}d"
    n = cfg.risk.max_proposals

    per_run: list[dict[str, Any]] = []
    for run_id, g in fr.groupby("run_id"):
        g = g.dropna(subset=[ycol, "composite"])
        if g.empty:
            continue
        top = g.sort_values("composite", ascending=False).head(n)
        per_run.append({"run_id": run_id,
                        "universe_fwd": float(g[ycol].mean()),
                        "top_fwd": float(top[ycol].mean())})
    if not per_run:
        return pd.DataFrame()
    pr = pd.DataFrame(per_run).merge(reg[["run_id", "state"]], on="run_id", how="inner")
    if pr.empty:
        return pd.DataFrame()
    out = (pr.groupby("state")
             .agg(runs=("run_id", "count"),
                  universe_fwd_pct=("universe_fwd", lambda s: round(s.mean() * 100, 2)),
                  top_fwd_pct=("top_fwd", lambda s: round(s.mean() * 100, 2)))
             .reset_index())
    out["horizon_d"] = horizon
    # Order risk_off, neutral, risk_on for readability.
    order = {"risk_off": 0, "neutral": 1, "risk_on": 2}
    return out.sort_values("state", key=lambda s: s.map(order).fillna(9)).reset_index(drop=True)


# --------------------------------------------------------------------------
# One-call report
# --------------------------------------------------------------------------
def summary(store: Store, cfg: Config = CONFIG,
            horizons: tuple[int, ...] = (1, 7, 30),
            primary_horizon: int = 7, refresh: bool = True) -> dict[str, Any]:
    """Run the whole battery and print a readable report. Returns the frames.

    `refresh` re-fetches price history first (default on) so re-running the
    cell in a live notebook kernel doesn't score against stale candles.
    """
    fr = forward_returns(store, cfg, horizons, refresh=refresh)
    ic = information_coefficient(fr, horizons)
    spread = quantile_spread(fr, primary_horizon, cfg.risk.max_proposals)
    hits = hit_rate(fr, primary_horizon, cfg.risk.min_composite_floor)
    trades = realized_trades(store, cfg)
    tstats = trade_stats(trades)
    eq = equity_stats(store, cfg)
    rec = recommend_weights(fr, primary_horizon)
    regime_eff = regime_effectiveness(store, cfg, primary_horizon)

    rule = "═" * 78
    print(f"{rule}\nENGINE EVALUATION\n{rule}")
    n_runs = fr["run_id"].nunique() if not fr.empty else 0
    n_scored = len(fr)
    print(f"  score rows: {n_scored}   runs: {n_runs}")
    if not fr.empty:
        print(f"  window: {fr['ts'].min():%Y-%m-%d} → {fr['ts'].max():%Y-%m-%d}")
    if n_runs < 8:
        print("  ⚠ fewer than ~8 runs — every number below is dominated by noise. "
              "This is instrumentation, not a verdict yet.")

    if not ic.empty:
        print("\n  Information coefficient (Spearman, per-run mean):")
        print("  " + ic.to_string(index=False).replace("\n", "\n  "))
        print("    read: |t_stat| > 2 ≈ real; mean_IC of 0.03–0.05 is a decent "
              "crypto signal; negative mean_IC = the family is hurting.")
        print("    t_stat is deflated for overlapping forward windows — daily "
              "runs sharing a 7/30d return aren't independent (see runs_eff).")

    if spread:
        print(f"\n  Composite ranking, {spread['horizon_d']}d forward "
              f"({spread['n_per_side']} per side, {spread['runs']} runs):")
        print(f"    top {spread['n_per_side']}:    {spread['top_mean_ret']*100:+.2f}%")
        print(f"    bottom {spread['n_per_side']}: {spread['bottom_mean_ret']*100:+.2f}%")
        print(f"    spread:   {spread['spread']*100:+.2f}%  "
              f"(top vs universe: {spread['top_minus_universe']*100:+.2f}%)")

    if hits:
        print(f"\n  Directional hit rate ({hits['horizon_d']}d):")
        print(f"    bullish calls {hits['bullish_calls']:>4}  "
              f"hit {hits['bullish_hit_rate']*100:.0f}%"
              if pd.notna(hits.get("bullish_hit_rate")) else
              "    bullish calls: none")
        if pd.notna(hits.get("bearish_hit_rate")):
            print(f"    bearish calls {hits['bearish_calls']:>4}  "
                  f"hit {hits['bearish_hit_rate']*100:.0f}%")

    if not tstats.empty:
        print("\n  Realised trades (net of fees):")
        print("  " + tstats.to_string(index=False).replace("\n", "\n  "))

    if eq:
        if eq.get("note"):
            print(f"\n  Equity curve: {eq['note']}")
        else:
            print(f"\n  Equity curve ({eq['mode']}, {eq['snapshots']} snapshots, "
                  f"{eq['span_days']}d):")
            if eq.get("mode_note"):
                print(f"    note: {eq['mode_note']}")
            print(f"    total return {eq['total_return_%']:+.2f}%   "
                  f"BTC buy-hold {eq['btc_buy_hold_%']:+.2f}%   "
                  f"vs BTC {eq['vs_btc_pp']:+.2f}pp")
            print(f"    max drawdown {eq['max_drawdown_%']:.2f}%   "
                  f"Sharpe(annualised) {eq['sharpe_annualised']}")

    if not regime_eff.empty:
        print(f"\n  Regime gate — realised {primary_horizon}d forward return by state:")
        print("  " + regime_eff.to_string(index=False).replace("\n", "\n  "))
        print("    the gate earns its keep if risk_off rows show weaker "
              "universe/top-N returns than risk_on.")

    print(f"\n  Weight recommendation ({primary_horizon}d horizon):")
    if rec.get("blend_verdict"):
        print(f"    {rec['blend_verdict']}")
    if rec.get("note"):
        print(f"    {rec['note']}")
    else:
        print(f"    {rec['code']}")
    print(rule)

    return {"forward_returns": fr, "ic": ic, "quantile_spread": spread,
            "hit_rate": hits, "trades": trades, "trade_stats": tstats,
            "equity": eq, "recommendation": rec, "regime_effectiveness": regime_eff}
