"""Feature 1a: OHLCV price data.

Four independent sources, tried in order, because any single crypto data
endpoint will eventually rate-limit you or be unreachable from your IP:

  binance   api.binance.us / api.binance.com   (best granularity; .com is
            HTTP 451 from US IPs, .us works)
  coinbase  api.exchange.coinbase.com          (no key, US-friendly, 300/req)
  kraken    api.kraken.com                     (no key, 720 candles/req)
  yfinance  via the yfinance package           (slowest, but very reliable)

All sources are normalized to the same DataFrame: a UTC DatetimeIndex named
`time` with float columns open/high/low/close/volume, sorted ascending, with
duplicate timestamps dropped.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd
import requests

from .config import CONFIG, Config

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# Canonical interval -> minutes
INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

_session: requests.Session | None = None
_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}


def session(cfg: Config = CONFIG) -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": cfg.user_agent})
    return _session


def _get(url: str, params: dict | None = None, cfg: Config = CONFIG,
         headers: dict | None = None) -> Any:
    """GET with bounded retries and exponential backoff."""
    last: Exception | None = None
    for attempt in range(cfg.http_retries):
        try:
            r = session(cfg).get(url, params=params, timeout=cfg.http_timeout,
                                 headers=headers)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001 - surface the last error to caller
            last = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {cfg.http_retries} tries: {last}")


def parse_lookback(lookback: str) -> timedelta:
    unit = lookback[-1].lower()
    n = float(lookback[:-1])
    return {"m": timedelta(minutes=n), "h": timedelta(hours=n),
            "d": timedelta(days=n), "w": timedelta(weeks=n)}[unit]


def _finalize(df: pd.DataFrame, start: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index >= pd.Timestamp(start)]
    for c in OHLCV_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"])


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["close"])


# --------------------------------------------------------------------------
# Binance (.us / .com / testnet)
# --------------------------------------------------------------------------
def fetch_binance(symbol: str, interval: str, start: datetime,
                  cfg: Config = CONFIG) -> pd.DataFrame:
    base = cfg.execution.base_url
    quote_candidates = [cfg.execution.quote_asset, "USDT", "USD", "USDC"]
    seen: set[str] = set()
    last_err: Exception | None = None

    for quote in quote_candidates:
        if quote in seen:
            continue
        seen.add(quote)
        pair = f"{symbol}{quote}"
        frames: list[pd.DataFrame] = []
        cursor = int(start.timestamp() * 1000)
        try:
            while True:
                rows = _get(f"{base}/api/v3/klines", {
                    "symbol": pair, "interval": interval,
                    "startTime": cursor, "limit": 1000,
                }, cfg)
                if not rows:
                    break
                frames.append(pd.DataFrame(rows).iloc[:, :6])
                if len(rows) < 1000:
                    break
                cursor = int(rows[-1][0]) + 1
                if len(frames) > 12:      # hard stop: ~12k candles
                    break
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
            df.columns = ["time", *OHLCV_COLS]
            df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
            return _finalize(df.set_index("time"), start)
        except Exception as exc:  # noqa: BLE001 - try the next quote asset
            last_err = exc
            continue
    raise RuntimeError(f"binance: no candles for {symbol} ({last_err})")


# --------------------------------------------------------------------------
# Coinbase Exchange
# --------------------------------------------------------------------------
_CB_GRAN = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 3600, "1d": 86400}


def fetch_coinbase(symbol: str, interval: str, start: datetime,
                   cfg: Config = CONFIG) -> pd.DataFrame:
    gran = _CB_GRAN[interval]
    product = f"{symbol}-USD"
    frames, cursor = [], start
    now = datetime.now(timezone.utc)

    while cursor < now and len(frames) < 12:      # 300 candles per request
        end = min(cursor + timedelta(seconds=gran * 300), now)
        rows = _get("https://api.exchange.coinbase.com/products/"
                    f"{product}/candles",
                    {"granularity": gran, "start": cursor.isoformat(),
                     "end": end.isoformat()}, cfg)
        if rows:
            frames.append(pd.DataFrame(rows,
                          columns=["time", "low", "high", "open", "close", "volume"]))
        cursor = end
        time.sleep(0.12)                          # stay under the public rate limit

    if not frames:
        raise RuntimeError(f"coinbase: no candles for {product}")
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="s", utc=True)
    df = _finalize(df.set_index("time")[OHLCV_COLS], start)
    return _resample(df, "4h") if interval == "4h" else df


# --------------------------------------------------------------------------
# Kraken
# --------------------------------------------------------------------------
_KRAKEN_INT = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
_KRAKEN_ALIAS = {"BTC": "XBT", "DOGE": "XDG"}


def fetch_kraken(symbol: str, interval: str, start: datetime,
                 cfg: Config = CONFIG) -> pd.DataFrame:
    pair = f"{_KRAKEN_ALIAS.get(symbol, symbol)}USD"
    payload = _get("https://api.kraken.com/0/public/OHLC", {
        "pair": pair, "interval": _KRAKEN_INT[interval],
        "since": int(start.timestamp()),
    }, cfg)
    if payload.get("error"):
        raise RuntimeError(f"kraken: {payload['error']}")
    result = {k: v for k, v in payload["result"].items() if k != "last"}
    if not result:
        raise RuntimeError(f"kraken: empty result for {pair}")
    rows = next(iter(result.values()))
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close",
                                     "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="s", utc=True)
    return _finalize(df.set_index("time")[OHLCV_COLS], start)


# --------------------------------------------------------------------------
# yfinance
# --------------------------------------------------------------------------
_YF_INT = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}


def fetch_yfinance(symbol: str, interval: str, start: datetime,
                   cfg: Config = CONFIG) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(f"{symbol}-USD").history(
        start=start, interval=_YF_INT[interval], auto_adjust=False
    )
    if df.empty:
        raise RuntimeError(f"yfinance: no data for {symbol}-USD")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "time"
    df = _finalize(df, start)
    return _resample(df, "4h") if interval == "4h" else df


FETCHERS: dict[str, Callable[..., pd.DataFrame]] = {
    "binance": fetch_binance,
    "coinbase": fetch_coinbase,
    "kraken": fetch_kraken,
    "yfinance": fetch_yfinance,
}


def get_ohlcv(symbol: str, timeframe: str = "1w", cfg: Config = CONFIG,
              use_cache: bool = True) -> tuple[pd.DataFrame, str]:
    """Fetch candles for one asset/timeframe. Returns (df, source_used).

    Sources are tried in `cfg.price_source_order`; the first that returns a
    usable frame wins. Raises only if every source fails.
    """
    from .config import TIMEFRAMES

    spec = TIMEFRAMES[timeframe]
    interval = spec["interval"]
    start = datetime.now(timezone.utc) - parse_lookback(spec["lookback"])

    key = (symbol, timeframe, cfg.execution.venue)
    if use_cache and key in _cache:
        ts, cached = _cache[key]
        if time.time() - ts < cfg.cache_ttl_seconds:
            return cached.copy(), "cache"

    errors: list[str] = []
    for source in cfg.price_source_order:
        try:
            df = FETCHERS[source](symbol, interval, start, cfg)
            if df is not None and len(df) >= 5:
                _cache[key] = (time.time(), df.copy())
                return df, source
            errors.append(f"{source}: only {0 if df is None else len(df)} candles")
        except Exception as exc:  # noqa: BLE001 - fall through to next source
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"All price sources failed for {symbol} [{timeframe}]:\n  "
                       + "\n  ".join(errors))


def get_many(symbols: list[str], timeframe: str = "1w", cfg: Config = CONFIG,
             quiet: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch several assets, skipping (and reporting) any that fail."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df, src = get_ohlcv(sym, timeframe, cfg)
            out[sym] = df
            if not quiet:
                print(f"  {sym:<6} {len(df):>5} candles  [{src}]")
        except Exception as exc:  # noqa: BLE001 - one bad symbol shouldn't stop the run
            if not quiet:
                print(f"  {sym:<6} FAILED: {exc}")
    return out


def latest_prices(symbols: list[str], cfg: Config = CONFIG) -> dict[str, float]:
    """Spot prices in one request where possible, falling back per-symbol."""
    prices: dict[str, float] = {}
    try:
        rows = _get(f"{cfg.execution.base_url}/api/v3/ticker/price", cfg=cfg)
        table = {r["symbol"]: float(r["price"]) for r in rows}
        for sym in symbols:
            for quote in (cfg.execution.quote_asset, "USDT", "USD", "USDC"):
                if f"{sym}{quote}" in table:
                    prices[sym] = table[f"{sym}{quote}"]
                    break
    except Exception:  # noqa: BLE001 - fall back to per-symbol candles below
        pass

    for sym in symbols:
        if sym not in prices:
            try:
                df, _ = get_ohlcv(sym, "1d", cfg)
                prices[sym] = float(df["close"].iloc[-1])
            except Exception:  # noqa: BLE001 - leave the symbol out entirely
                continue
    return prices
