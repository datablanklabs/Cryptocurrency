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

A second, independent input can dampen (never boost) the BTC-trend verdict:
`macro.py`'s blended read of Kalshi event-contract prices (Fed decisions, CPI,
government-shutdown risk, ...). It only ever restricts exposure further, the
same way the drawdown override does, and is a no-op whenever `assess()` is
called without a `store` — see `_apply_macro`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import macro as macro_mod
from . import prices as prices_mod
from .config import CONFIG, Config
from .store import Store


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
        "macro_enabled": rc.macro_enabled, "macro_score": None,
        "macro_note": "", "macro_forced_risk_off": False,
    }


def _macro_note_suffix(info: dict[str, Any]) -> str:
    if info.get("macro_forced_risk_off"):
        return f" | macro override -> risk_off: {info['macro_note']}"
    if info.get("macro_score") is not None:
        return f" | macro {info['macro_score']:+.2f}: {info['macro_note']}"
    return ""


def _apply_macro(rc, state: str, scale: float, cfg: Config,
                 store: Store | None) -> tuple[str, float, dict[str, Any]]:
    """Fold the Kalshi macro read into a trend-determined (state, scale).

    Folded in AFTER the trend call, the same way the drawdown override works:
    a supportive macro backdrop can never push exposure past what the trend
    alone earned, but a bad one can cut it further, or force risk_off outright
    once it crosses `macro_risk_off_threshold`. Any failure — no store passed,
    macro disabled, no credentials, no fresh snapshot — degrades to "no
    adjustment", the same contract every other input to this gate honours.
    """
    info: dict[str, Any] = {"macro_enabled": rc.macro_enabled, "macro_score": None,
                            "macro_note": "", "macro_forced_risk_off": False}
    if not rc.macro_enabled:
        info["macro_note"] = "macro gate off (CONFIG.regime.macro_enabled = False)"
        return state, scale, info
    if store is None:
        info["macro_note"] = "no store passed to regime.assess() — macro skipped"
        return state, scale, info

    try:
        m = macro_mod.score(store, cfg)
    except Exception as exc:  # noqa: BLE001 - macro must never crash the regime gate
        info["macro_note"] = f"macro unavailable ({exc})"
        return state, scale, info

    if not m.get("n_series"):
        info["macro_note"] = m.get("note", "macro: no data")
        return state, scale, info

    score_v = float(m["score"])
    info.update(macro_score=round(score_v, 4), macro_note=m.get("note", ""))

    if score_v <= rc.macro_risk_off_threshold:
        info["macro_forced_risk_off"] = True
        return "risk_off", rc.risk_off_exposure, info

    if score_v < 0:
        multiplier = max(rc.macro_min_multiplier, 1.0 + score_v * rc.macro_downweight)
        return state, scale * multiplier, info

    return state, scale, info


def assess(cfg: Config = CONFIG, symbol: str = "BTC",
          store: Store | None = None) -> dict[str, Any]:
    """Classify the current regime from `symbol`'s daily candles.

    Returns a dict with `state`, `exposure_scale` (the multiplier for the
    deployment cap) and the raw numbers behind the call. Never raises: if the
    price history can't be fetched it returns a neutral verdict with a note, so
    a data outage degrades to half-size rather than crashing the cycle.

    A dead-band (`ma_band_pct`) around the MA stops the gate flipping every
    cycle while price chops across the line.

    Pass `store` to also fold in the Kalshi macro read (see module docstring
    and `_apply_macro`); without it macro is silently skipped, exactly like
    every other optional input here.
    """
    rc = cfg.regime
    if not rc.enabled:
        state, scale, macro_info = _apply_macro(rc, "risk_on", 1.0, cfg, store)
        return {**_blank(rc, symbol, state, scale,
                         "regime gate disabled" + _macro_note_suffix(macro_info)),
               **macro_info}

    try:
        df, src = prices_mod.get_ohlcv(symbol, "1y", cfg)
        close = df["close"].astype(float).dropna()
    except Exception as exc:  # noqa: BLE001 - degrade to neutral, never crash
        state, scale, macro_info = _apply_macro(
            rc, "neutral", rc.neutral_exposure, cfg, store)
        note = f"price history unavailable ({exc}); abstaining (neutral)"
        return {**_blank(rc, symbol, state, scale, note + _macro_note_suffix(macro_info)),
               **macro_info}

    if len(close) < rc.min_candles:
        state, scale, macro_info = _apply_macro(
            rc, "neutral", rc.neutral_exposure, cfg, store)
        note = (f"only {len(close)} daily candles (< min_candles "
                f"{rc.min_candles}); MA unreliable, abstaining (neutral)")
        return {**_blank(rc, symbol, state, scale, note + _macro_note_suffix(macro_info)),
               **macro_info}

    # If history is short of ma_days, use what we have and SAY the window is
    # shorter than advertised rather than silently computing a faster MA.
    win = min(rc.ma_days, len(close))
    ma_series = _sma(close, win)
    price = float(close.iloc[-1])
    ma = float(ma_series.iloc[-1])
    if not np.isfinite(ma) or ma <= 0:
        state, scale, macro_info = _apply_macro(
            rc, "neutral", rc.neutral_exposure, cfg, store)
        note = "moving average not computable yet; abstaining (neutral)"
        return {**_blank(rc, symbol, state, scale, note + _macro_note_suffix(macro_info)),
               **macro_info}

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

    state, scale, macro_info = _apply_macro(rc, state, scale, cfg, store)

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
              f"{lookback}d, drawdown {dd * 100:.1f}% from trailing high)"
              + _macro_note_suffix(macro_info)),
    )
    out.update(macro_info)
    return out


def describe(regime: dict[str, Any]) -> str:
    """One-line summary for the notebook / CLI."""
    if not regime.get("enabled", True):
        return "regime gate: OFF (full size regardless of trend)"
    icon = {"risk_on": "🟢", "neutral": "🟡", "risk_off": "🔴"}.get(regime["state"], "•")
    return (f"{icon} regime: {regime['state'].upper()}  "
            f"(deploy ×{regime['exposure_scale']:.2f})  —  {regime.get('note', '')}")
