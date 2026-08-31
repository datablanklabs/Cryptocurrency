"""Market-regime gate: how much long exposure the tape currently justifies.

Long-only alt exposure while BTC is in a downtrend is structurally negative-EV.
The majors run 0.7-0.9 correlated with everything else, so a great *relative*
pick still loses money when BTC is bleeding. This module reads BTC's own daily
chart and returns one of three states plus a deployment multiplier that
`engine.propose` applies to the whole slate. Exits are never gated by it.

    risk_on   price above the long SMA and the SMA rising      -> full size
    neutral   price above a flat/falling SMA (just reclaimed)   -> half size
    risk_off  price below the long SMA, or a deep drawdown      -> no new buys

The thresholds live in `RegimeConfig`. This is a blunt instrument on purpose:
a single, legible, hard-to-overfit switch does more for long-only crypto
returns than any amount of signal tuning, precisely because it is not tuned.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import prices as prices_mod
from .config import CONFIG, Config


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(5, n // 3)).mean()


def _blank(rc, symbol: str, state: str, scale: float, note: str) -> dict[str, Any]:
    """A fully-populated verdict dict, so every caller can read every key."""
    return {
        "enabled": rc.enabled, "symbol": symbol, "state": state,
        "exposure_scale": float(scale), "note": note, "source": None,
        "btc_price": None, "btc_ma": None, "ma_days": rc.ma_days,
        "ma_window_used": None, "pct_vs_ma": None, "ma_slope_pct": None,
        "drawdown_from_high_pct": None,
    }


def assess(cfg: Config = CONFIG, symbol: str = "BTC") -> dict[str, Any]:
    """Classify the current regime from `symbol`'s daily candles.

    Returns a dict with `state`, `exposure_scale` (the multiplier for the
    deployment cap) and the raw numbers behind the call. Never raises: if the
    price history can't be fetched it returns a neutral verdict with a note, so
    a data outage degrades to half-size rather than crashing the cycle.

    A dead-band (`ma_band_pct`) around the MA stops the gate flipping every
    cycle while price chops across the line.
    """
    rc = cfg.regime
    if not rc.enabled:
        return _blank(rc, symbol, "risk_on", 1.0, "regime gate disabled")

    try:
        df, src = prices_mod.get_ohlcv(symbol, "1y", cfg)
        close = df["close"].astype(float).dropna()
    except Exception as exc:  # noqa: BLE001 - degrade to neutral, never crash
        return _blank(rc, symbol, "neutral", rc.neutral_exposure,
                      f"price history unavailable ({exc}); abstaining (neutral)")

    if len(close) < rc.min_candles:
        return _blank(rc, symbol, "neutral", rc.neutral_exposure,
                      f"only {len(close)} daily candles (< min_candles "
                      f"{rc.min_candles}); MA unreliable, abstaining (neutral)")

    # If history is short of ma_days, use what we have and SAY the window is
    # shorter than advertised rather than silently computing a faster MA.
    win = min(rc.ma_days, len(close))
    ma_series = _sma(close, win)
    price = float(close.iloc[-1])
    ma = float(ma_series.iloc[-1])
    if not np.isfinite(ma) or ma <= 0:
        return _blank(rc, symbol, "neutral", rc.neutral_exposure,
                      "moving average not computable yet; abstaining (neutral)")

    band = rc.ma_band_pct / 100.0
    upper, lower = ma * (1 + band), ma * (1 - band)
    pct_vs_ma = price / ma - 1.0
    lookback = min(21, len(ma_series) - 1)
    ma_prev = float(ma_series.iloc[-1 - lookback])
    ma_slope = (ma / ma_prev - 1.0) if np.isfinite(ma_prev) and ma_prev > 0 else 0.0
    trail_high = float(close.iloc[-min(len(close), 180):].max())
    dd = price / trail_high - 1.0

    if price < lower or dd <= -rc.drawdown_risk_off_pct / 100.0:
        state, scale = "risk_off", rc.risk_off_exposure
    elif price > upper and ma_slope > 0:
        state, scale = "risk_on", rc.risk_on_exposure
    else:
        state, scale = "neutral", rc.neutral_exposure

    out = _blank(rc, symbol, state, scale, "")
    out.update(
        source=src, btc_price=round(price, 2), btc_ma=round(ma, 2),
        ma_window_used=int(win),
        pct_vs_ma=round(pct_vs_ma * 100, 2),
        ma_slope_pct=round(ma_slope * 100, 2),
        drawdown_from_high_pct=round(dd * 100, 2),
        note=(f"{symbol} {pct_vs_ma * 100:+.1f}% vs its "
              f"{win}d MA{' (short history)' if win < rc.ma_days else ''} "
              f"±{rc.ma_band_pct:.0f}% band (MA slope {ma_slope * 100:+.1f}% / "
              f"{lookback}d, drawdown {dd * 100:.1f}% from trailing high)"),
    )
    return out


def describe(regime: dict[str, Any]) -> str:
    """One-line summary for the notebook / CLI."""
    if not regime.get("enabled", True):
        return "regime gate: OFF (full size regardless of trend)"
    icon = {"risk_on": "🟢", "neutral": "🟡", "risk_off": "🔴"}.get(regime["state"], "•")
    return (f"{icon} regime: {regime['state'].upper()}  "
            f"(deploy ×{regime['exposure_scale']:.2f})  —  {regime.get('note', '')}")
