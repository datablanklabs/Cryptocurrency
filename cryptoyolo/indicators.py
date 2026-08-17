"""Feature 1b: bands and technical indicators.

Plain pandas/numpy - no TA library dependency, so there's nothing to install and
nothing hiding behind a wrapper. Every function takes and returns a DataFrame
indexed by time.

"Standard bands" here means the three that are actually standard:
  Bollinger  - SMA +/- k * rolling stdev      (volatility around a mean)
  Keltner    - EMA +/- k * ATR                (volatility via true range)
  Donchian   - rolling high / low             (breakout levels)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BandConfig


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(2, window // 2)).mean()


def ema(s: pd.Series, window: int) -> pd.Series:
    return s.ewm(span=window, adjust=False, min_periods=max(2, window // 2)).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(s: pd.Series, window: int = 20,
              stds: tuple[float, ...] = (1.0, 2.0)) -> pd.DataFrame:
    mid = sma(s, window)
    sd = s.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
    out = pd.DataFrame({"bb_mid": mid})
    for k in stds:
        tag = str(k).replace(".", "_")
        out[f"bb_upper_{tag}"] = mid + k * sd
        out[f"bb_lower_{tag}"] = mid - k * sd
    widest = max(stds)
    tag = str(widest).replace(".", "_")
    span = (out[f"bb_upper_{tag}"] - out[f"bb_lower_{tag}"]).replace(0, np.nan)
    # %B: 0 = at lower band, 1 = at upper band. The single most useful
    # normalized "where are we in the range" number.
    out["bb_pctb"] = (s - out[f"bb_lower_{tag}"]) / span
    out["bb_width"] = span / mid.replace(0, np.nan)
    return out


def keltner(df: pd.DataFrame, window: int = 20, mult: float = 2.0) -> pd.DataFrame:
    mid = ema(df["close"], window)
    rng = atr(df, window) * mult
    return pd.DataFrame({"kc_mid": mid, "kc_upper": mid + rng, "kc_lower": mid - rng})


def donchian(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    upper = df["high"].rolling(window, min_periods=2).max()
    lower = df["low"].rolling(window, min_periods=2).min()
    return pd.DataFrame({"dc_upper": upper, "dc_lower": lower,
                         "dc_mid": (upper + lower) / 2})


def enrich(df: pd.DataFrame, bands: BandConfig | None = None) -> pd.DataFrame:
    """Attach every configured band/indicator to an OHLCV frame."""
    bands = bands or BandConfig()
    out = df.copy()
    close = out["close"]

    if bands.bollinger:
        out = out.join(bollinger(close, bands.bollinger_window, bands.bollinger_stds))
    if bands.keltner:
        out = out.join(keltner(out, bands.keltner_window, bands.keltner_atr_mult))
    if bands.donchian:
        out = out.join(donchian(out, bands.donchian_window))

    for w in bands.moving_averages:
        if len(out) >= 3:
            out[f"sma_{w}"] = sma(close, w)

    out["rsi_14"] = rsi(close, 14)
    out["atr_14"] = atr(out, 14)
    out = out.join(macd(close))
    out["ret_1"] = close.pct_change()
    out["vol_zscore"] = _zscore(out["volume"], 20)
    return out


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=max(2, window // 2)).mean()
    sd = s.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
    return ((s - mu) / sd.replace(0, np.nan)).fillna(0)


def squeeze_flag(df: pd.DataFrame, lookback: int = 100) -> bool:
    """True when Bollinger width is in the bottom quintile of its recent range.

    Low volatility tends to precede expansion; it says nothing about direction.
    """
    if "bb_width" not in df or df["bb_width"].notna().sum() < 20:
        return False
    w = df["bb_width"].dropna().tail(lookback)
    return bool(w.iloc[-1] <= w.quantile(0.20))


def summarize(df: pd.DataFrame) -> dict[str, float]:
    """Point-in-time snapshot of the latest bar, used by the decision engine."""
    if df.empty:
        return {}
    last = df.iloc[-1]
    close = float(last["close"])

    def val(col: str, default: float = np.nan) -> float:
        v = last.get(col, default)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return default if pd.isna(f) else f

    out: dict[str, float] = {
        "close": close,
        "rsi_14": val("rsi_14", 50.0),
        "atr_14": val("atr_14", close * 0.02),
        "atr_pct": val("atr_14", close * 0.02) / close if close else 0.0,
        "bb_pctb": val("bb_pctb", 0.5),
        "bb_width": val("bb_width", 0.0),
        "macd_hist": val("macd_hist", 0.0),
        "vol_zscore": val("vol_zscore", 0.0),
        "squeeze": float(squeeze_flag(df)),
    }
    for w in (20, 50, 200):
        col = f"sma_{w}"
        if col in df:
            ma = val(col)
            out[f"dist_sma_{w}"] = (close - ma) / ma if ma and not np.isnan(ma) else 0.0

    for bars, tag in ((6, "6b"), (24, "24b"), (72, "72b")):
        if len(df) > bars:
            prior = float(df["close"].iloc[-bars - 1])
            out[f"ret_{tag}"] = (close - prior) / prior if prior else 0.0
        else:
            out[f"ret_{tag}"] = 0.0
    return out
