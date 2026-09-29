"""Candidate signals for the backtest's factor lab.

The point is to test an idea against a year of point-in-time history BEFORE it
goes anywhere near `engine.py`. A signal is any function

    fn(ctx: SignalContext) -> float

that reads only what `ctx` carries - bars that had closed by `ctx.t` - and
returns a number where higher means "expect this asset to do better". It needn't
be bounded or scaled: the IC is a rank correlation within each day's
cross-section, so only the ordering matters. NaN means "no opinion" and is
skipped.

    ./backtest.py --no-sync --signals reversal_1d,mymodule:my_signal

`load()` resolves a spec: a built-in name below, `module:function`, or
`name=module:function` to label it. The built-ins exist because the backtest
already showed something worth chasing: `technical_raw` had a significantly
NEGATIVE 1-day IC, the signature of short-term reversal.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class SignalContext:
    symbol: str
    t: pd.Timestamp
    short: pd.DataFrame             # ~1 day of 5m bars, closed by t
    medium: pd.DataFrame            # ~7 days of 1h bars
    daily: pd.DataFrame             # up to 365 days of 1d bars
    short_snap: dict = field(default_factory=dict)    # indicators.summarize(...)
    medium_snap: dict = field(default_factory=dict)
    funding: np.ndarray = field(default_factory=lambda: np.array([]))  # rates settled by t


Signal = Callable[[SignalContext], float]


def _ret(close: pd.Series, bars: int) -> float:
    if len(close) <= bars:
        return np.nan
    prior = float(close.iloc[-bars - 1])
    return float(close.iloc[-1]) / prior - 1.0 if prior else np.nan


def reversal_1d(ctx: SignalContext) -> float:
    """Short-term reversal: yesterday's losers bounce. Minus the last 24h return."""
    return -_ret(ctx.medium["close"], 24)


def reversal_3d(ctx: SignalContext) -> float:
    return -_ret(ctx.daily["close"], 3)


def momentum_30d(ctx: SignalContext) -> float:
    """Plain 30-day return, for comparison with the blended xsec factor."""
    return _ret(ctx.daily["close"], 30)


def low_vol(ctx: SignalContext) -> float:
    """Low-volatility anomaly: minus the 30-day stdev of daily returns."""
    r = ctx.daily["close"].pct_change().dropna().tail(30)
    return -float(r.std(ddof=1)) if len(r) >= 10 else np.nan


def funding_level(ctx: SignalContext) -> float:
    """Contrarian on the raw funding level (not its z-score): minus the mean of
    the last 9 settlements (~3 days)."""
    if len(ctx.funding) < 3:
        return np.nan
    return -float(np.mean(ctx.funding[-9:]))


BUILTIN: dict[str, Signal] = {
    "reversal_1d": reversal_1d,
    "reversal_3d": reversal_3d,
    "momentum_30d": momentum_30d,
    "low_vol": low_vol,
    "funding_level": funding_level,
}


def load(spec: str) -> tuple[str, Signal]:
    """Resolve 'name', 'module:function' or 'label=module:function'."""
    spec = spec.strip()
    label = None
    if "=" in spec:
        label, spec = (x.strip() for x in spec.split("=", 1))
    if spec in BUILTIN:
        return label or spec, BUILTIN[spec]
    if ":" not in spec:
        raise ValueError(f"unknown signal {spec!r}: not built-in "
                         f"({', '.join(BUILTIN)}) and not module:function")
    mod, fn = spec.split(":", 1)
    func = getattr(importlib.import_module(mod), fn)
    if not callable(func):
        raise ValueError(f"{spec} is not callable")
    return label or fn, func


def load_many(specs: str | None) -> dict[str, Signal]:
    """Comma-separated specs; 'builtin' (the default) expands to every built-in,
    'none' or '' to nothing."""
    if specs is None or specs.strip().lower() == "builtin":
        return dict(BUILTIN)
    out: dict[str, Signal] = {}
    for part in specs.split(","):
        if not part.strip() or part.strip().lower() == "none":
            continue
        if part.strip().lower() == "builtin":
            out.update(BUILTIN)
            continue
        name, fn = load(part)
        out[name] = fn
    return out
