"""Correlation-aware portfolio risk.

"1% risk each" on three alt longs that move together is really ~2.5% of one
bet. `engine.propose` sizes each trade so a stop-out costs `risk_per_trade_pct`
of equity, then caps the gross deployed notional — neither of which sees that
the names are the same trade. This module measures the correlation-adjusted
risk of a slate:

    heat = sqrt(r' C r)

with `r` the per-trade dollar-risk vector and `C` the trailing return
correlation matrix. `heat` sits between `sum(r)` (everything perfectly
correlated — one bet) and `sqrt(sum r_i^2)` (independent). `engine.propose`
scales the whole slate down if `heat` exceeds `risk.max_portfolio_heat_pct` of
equity.

Correlations come from the same 1y daily candles `build_scores` just fetched, so
this is cache-cheap in a normal cycle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import prices as prices_mod
from .config import CONFIG, Config

# When a pair has no overlapping history, assume this — crypto majors/alts sit
# around here, and erring high is the safe direction for a risk cap.
DEFAULT_CORR = 0.8


def daily_returns(symbols: list[str], cfg: Config = CONFIG,
                  lookback_days: int = 60) -> pd.DataFrame:
    """Aligned trailing daily returns, one column per symbol that had data."""
    cols: dict[str, pd.Series] = {}
    for sym in dict.fromkeys(symbols):           # dedupe, keep order
        try:
            df, _ = prices_mod.get_ohlcv(sym, "1y", cfg)
        except Exception:  # noqa: BLE001 - a symbol with no history is skipped
            continue
        s = df["close"].astype(float)
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True)).as_unit("ns")
        r = s.sort_index().pct_change().dropna().tail(lookback_days)
        if len(r) >= 15:
            cols[sym] = r
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).dropna(how="all")


def correlation_matrix(symbols: list[str], cfg: Config = CONFIG,
                       lookback_days: int = 60) -> pd.DataFrame:
    """Pairwise Pearson correlation of trailing daily returns.

    Empty (no usable history) is a valid return — the caller then falls back to
    a scalar `DEFAULT_CORR`.
    """
    rets = daily_returns(symbols, cfg, lookback_days)
    if rets.empty or rets.shape[1] < 2:
        return pd.DataFrame()
    # Pairwise so a symbol with a shorter history doesn't null the whole matrix.
    c = rets.corr(min_periods=15)
    return c.clip(-1.0, 1.0)


def portfolio_heat(risk_by_symbol: dict[str, float],
                   corr: pd.DataFrame | None = None,
                   default_corr: float = DEFAULT_CORR) -> dict[str, Any]:
    """Correlation-adjusted risk of a set of positions.

    `risk_by_symbol` maps symbol -> dollar risk (entry-to-stop loss). Returns
    `heat_usd` = sqrt(r' C r), `gross_usd` = sum(r) (the perfectly-correlated
    worst case), and `diversification_ratio` = heat / gross in (0, 1].
    """
    syms = [s for s, v in risk_by_symbol.items() if v and v > 0]
    r = np.array([risk_by_symbol[s] for s in syms], dtype=float)
    gross = float(r.sum())
    if len(syms) <= 1 or gross <= 0:
        return {"heat_usd": gross, "gross_usd": gross,
                "diversification_ratio": 1.0, "n": len(syms)}

    n = len(syms)
    C = np.full((n, n), default_corr, dtype=float)
    np.fill_diagonal(C, 1.0)
    if corr is not None and not corr.empty:
        for i, a in enumerate(syms):
            for j, b in enumerate(syms):
                if i == j:
                    continue
                v = corr.get(a, {}).get(b) if a in corr.columns else None
                if v is not None and np.isfinite(v):
                    C[i, j] = float(v)
    # Symmetrise (pairwise corr can be minutely asymmetric) and PSD-clip.
    C = (C + C.T) / 2.0
    var = float(r @ C @ r)
    heat = float(np.sqrt(max(var, 0.0)))
    return {
        "heat_usd": heat, "gross_usd": gross,
        "diversification_ratio": heat / gross if gross else 1.0,
        "n": n, "symbols": syms,
    }
