"""Point-in-time replay of the price-derived families over a year+ of history.

`evaluation` can only grade what the live engine has already scored, and a
7-day IC needs months of daily runs before its t-stat means anything. But three
of the families - `technical`, `xsec` (cross-sectional momentum) and
`positioning` (funding) - are pure functions of candles and funding rates, both
of which the exchanges publish historically. This module replays them:

  sync       download 5m / 1h / 1d candles (Binance.US) and funding history
             (OKX) into a separate cache DB, incrementally
  replay     at each daily step t, rebuild the three views the live engine
             sees from bars that had CLOSED by t, score them with the live
             engine's own functions, and attach realised forward returns
  simulate   hold the top-N by composite, rebalanced every `hold_days`, net of
             fees + slippage, against BTC buy-and-hold and an equal-weight basket
  report     IC per family (whole window and each half), quantile spreads,
             blend verdict, and the simulation

Beyond the live families, each replayed row also carries:
  * the five pieces of `technical_score` (`tc_trend`, `tc_momentum`,
    `tc_position`, `tc_rsi`, `tc_volume`), so a bad blend can be traced to
    the piece that causes it;
  * any candidate signals passed in (`signals.py`: built-ins, or your own
    `module:function`), IC'd alongside - an idea gets tested here before it
    touches engine.py;
  * the BTC-trend regime verdict (`regime.classify`, the live gate's own
    logic) at each step, so the simulation can be run with and without it.
    The Kalshi macro dampener has no history and is not replayed.

`replay` returns a frame shaped like `evaluation.forward_returns`, so every
tool there (information_coefficient, quantile_spread, hit_rate) works on it.

What it cannot do: `catalyst` has no point-in-time history, and `social` has
one only if you backfill it (`./backfill_social.py`, see social_backfill.py).
`composite` here is the PRICE-ONLY composite (technical + positioning at their
configured relative weights). With a social backfill, rows also carry `social`
(Reddit-only, scored by the live function) and `composite_social` (price +
social at their configured relative weights).

Known biases, stated so nobody mistakes this for more than it is:
  * survivorship - the universe is today's list; coins that died or were
    delisted in the window aren't in it. This flatters every long-only result.
  * funding history - OKX serves only ~3 months, so `positioning` is NaN before
    that (IC skips it; the composite treats it as 0, as a live run would).
  * the in-progress bar - a live run's last candle is still forming; here only
    closed bars are used. Slightly stricter than live, never looser.

The cache lives in its own SQLite file (default `data/backtest_cache.sqlite`),
never in the main database: a year of 5m bars is ~100 MB per 20 assets.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import requests

from . import engine, evaluation, indicators, positioning
from . import prices as prices_mod
from . import regime as regime_mod
from .config import CONFIG, DATA_DIR, STABLECOINS, TIMEFRAMES, Config
from .signals import Signal, SignalContext

DEFAULT_CACHE = DATA_DIR / "backtest_cache.sqlite"

INTERVAL_MS = {"5m": 300_000, "1h": 3_600_000, "1d": 86_400_000}

# The live engine's three views, taken from TIMEFRAMES so they can't drift:
# technical_score's short (1d of 5m) and medium (7d of 1h) frames, plus the
# daily frame (365d of 1d) that cross-sectional momentum reads.
VIEWS: dict[str, tuple[str, timedelta]] = {
    name: (TIMEFRAMES[tf]["interval"], prices_mod.parse_lookback(TIMEFRAMES[tf]["lookback"]))
    for name, tf in (("short", "1d"), ("medium", "1w"), ("daily", "1y"))
}

FAMILIES = ("technical_raw", "xsec", "technical", "positioning", "composite")
# technical_score's own parts, prefixed so they can't collide with a family.
COMPONENTS = ("tc_trend", "tc_momentum", "tc_position", "tc_rsi", "tc_volume")

# A view with fewer than this share of its expected bars (a gap in the venue's
# history, or a coin listed mid-window) is skipped rather than scored thin.
_MIN_COVERAGE = 0.5


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
class BarCache:
    """Candles and funding rates in their own SQLite file."""

    def __init__(self, path: Path | str = DEFAULT_CACHE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS bars (
                    symbol TEXT NOT NULL, interval TEXT NOT NULL, ts INTEGER NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    PRIMARY KEY (symbol, interval, ts)) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS funding (
                    symbol TEXT NOT NULL, funding_time INTEGER NOT NULL, rate REAL,
                    PRIMARY KEY (symbol, funding_time)) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS pairs (
                    symbol TEXT PRIMARY KEY, pair TEXT);
            """)

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def bar_range(self, symbol: str, interval: str) -> tuple[int | None, int | None]:
        with self.conn() as con:
            lo, hi = con.execute("SELECT MIN(ts), MAX(ts) FROM bars WHERE symbol=? AND interval=?",
                                 (symbol, interval)).fetchone()
        return lo, hi

    def upsert_bars(self, symbol: str, interval: str, rows: list[list]) -> int:
        if not rows:
            return 0
        with self.conn() as con:
            con.executemany(
                "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)",
                [(symbol, interval, int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                  float(r[4]), float(r[5])) for r in rows])
        return len(rows)

    def bars(self, symbol: str, interval: str) -> pd.DataFrame:
        """OHLCV indexed by bar OPEN time (UTC), oldest first."""
        with self.conn() as con:
            df = pd.read_sql_query(
                "SELECT ts, open, high, low, close, volume FROM bars "
                "WHERE symbol=? AND interval=? ORDER BY ts", con, params=(symbol, interval))
        # ns, not ms: replay slices on .asi8 against pd.Timestamp.value (ns).
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), unit="ms", utc=True)).as_unit("ns")
        return df

    def upsert_funding(self, symbol: str, rows: list[tuple[int, float]]) -> int:
        """Store settlements; returns how many were NEW (pages overlap)."""
        if not rows:
            return 0
        with self.conn() as con:
            before = con.total_changes
            con.executemany("INSERT OR IGNORE INTO funding VALUES (?,?,?)",
                            [(symbol, int(t), float(r)) for t, r in rows])
            return con.total_changes - before

    def funding(self, symbol: str) -> pd.DataFrame:
        with self.conn() as con:
            return pd.read_sql_query(
                "SELECT funding_time, rate FROM funding WHERE symbol=? ORDER BY funding_time",
                con, params=(symbol,))

    def pair(self, symbol: str) -> str | None:
        with self.conn() as con:
            row = con.execute("SELECT pair FROM pairs WHERE symbol=?", (symbol,)).fetchone()
        return row[0] if row else None

    def set_pair(self, symbol: str, pair: str) -> None:
        with self.conn() as con:
            con.execute("INSERT OR REPLACE INTO pairs VALUES (?,?)", (symbol, pair))


# --------------------------------------------------------------------------
# Sync (network)
# --------------------------------------------------------------------------
def _resolve_pair(symbol: str, cfg: Config) -> str | None:
    """First quote whose pair is actually trading.

    Binance.US keeps some dead pairs alive (BTCUSD prints flat, zero-volume
    candles at a stale price), so "the pair returns candles" isn't enough: the
    last week of daily bars must show real volume.
    """
    base = cfg.execution.base_url
    for quote in dict.fromkeys([cfg.execution.quote_asset, "USDT", "USD", "USDC"]):
        pair = f"{symbol}{quote}"
        try:
            rows = prices_mod._get(f"{base}/api/v3/klines",
                                   {"symbol": pair, "interval": "1d", "limit": 7}, cfg)
        except Exception:  # noqa: BLE001 - try the next quote
            continue
        if rows and sum(float(r[5]) for r in rows) > 0:
            return pair
    return None


def _fetch_klines(pair: str, interval: str, start_ms: int, end_ms: int,
                  cfg: Config) -> list[list]:
    base = cfg.execution.base_url
    out: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        rows = prices_mod._get(f"{base}/api/v3/klines", {
            "symbol": pair, "interval": interval,
            "startTime": cursor, "endTime": end_ms - 1, "limit": 1000}, cfg)
        if not rows:
            break
        out.extend(r[:6] for r in rows)
        if len(rows) < 1000:
            break
        cursor = int(rows[-1][0]) + 1
    return out


def _sync_bars(cache: BarCache, symbol: str, pair: str, interval: str,
               start_ms: int, end_ms: int, cfg: Config) -> int:
    """Fetch only the part of [start, end) the cache doesn't already hold."""
    iv = INTERVAL_MS[interval]
    lo, hi = cache.bar_range(symbol, interval)
    # The tail restarts AT the newest cached bar, not after it: that bar was
    # probably still forming when it was fetched, and REPLACE settles it.
    spans = ([(start_ms, end_ms)] if lo is None else
             [(start_ms, lo), (hi, end_ms)])
    n = 0
    for a, b in spans:
        if b - a >= iv:
            n += cache.upsert_bars(symbol, interval, _fetch_klines(pair, interval, a, b, cfg))
    return n


def _sync_funding(cache: BarCache, symbol: str, cfg: Config) -> int:
    """Page back through OKX funding history until it runs out or meets the cache."""
    pc = cfg.positioning
    have = cache.funding(symbol)
    newest_cached = int(have["funding_time"].max()) if not have.empty else 0
    sess = requests.Session()
    sess.headers.update({"User-Agent": cfg.user_agent})
    rows: list[tuple[int, float]] = []
    after: str | None = None
    for _ in range(100):                      # OKX keeps ~3 months: ~3 pages
        params: dict[str, Any] = {"instId": pc.inst_template.format(symbol=symbol), "limit": 100}
        if after:
            params["after"] = after
        try:
            r = sess.get(pc.venue_url, params=params, timeout=cfg.http_timeout)
            r.raise_for_status()
            data = r.json().get("data") or []
        except Exception:  # noqa: BLE001 - no perp for this asset, or a blip
            break
        page = [(int(d["fundingTime"]), float(d.get("realizedRate") or d.get("fundingRate")))
                for d in data
                if d.get("fundingTime") and (d.get("realizedRate") or d.get("fundingRate"))]
        rows.extend(page)
        if not data or min(t for t, _ in page or [(0, 0)]) <= newest_cached:
            break
        after = data[-1]["fundingTime"]
        time.sleep(pc.request_delay)
    return cache.upsert_funding(symbol, rows)


def sync(cache: BarCache, cfg: Config = CONFIG, days: int = 365,
         end: datetime | None = None, symbols: list[str] | None = None,
         verbose: bool = True) -> pd.DataFrame:
    """Bring the cache up to date for `days` of replay ending at `end`.

    Each view needs its own lookback before the first step (7d of 1h, 365d of
    1d), so the daily bars reach back `days + 365`.
    """
    end = end or datetime.now(timezone.utc)
    end_ms = int(end.timestamp() * 1000)
    report = []
    for sym in symbols or cfg.symbols:
        if sym in STABLECOINS:
            continue
        pair = cache.pair(sym) or _resolve_pair(sym, cfg)
        if not pair:
            if verbose:
                print(f"  {sym:<6} no live pair on {cfg.execution.base_url}; skipped")
            report.append({"symbol": sym, "pair": None})
            continue
        cache.set_pair(sym, pair)
        row: dict[str, Any] = {"symbol": sym, "pair": pair}
        for name, (interval, lookback) in VIEWS.items():
            start_ms = int((end - timedelta(days=days) - lookback).timestamp() * 1000)
            row[interval] = _sync_bars(cache, sym, pair, interval, start_ms, end_ms, cfg)
        row["funding"] = _sync_funding(cache, sym, cfg) if cfg.positioning.enabled else 0
        if verbose:
            print(f"  {sym:<6} {pair:<9} +{row['5m']:>6} 5m  +{row['1h']:>5} 1h  "
                  f"+{row['1d']:>4} 1d  +{row['funding']:>4} funding")
        report.append(row)
    return pd.DataFrame(report)


# --------------------------------------------------------------------------
# Replay (offline)
# --------------------------------------------------------------------------
@dataclass
class _Frames:
    """One symbol's cached bars, with close times precomputed for slicing."""
    views: dict[str, tuple[pd.DataFrame, np.ndarray]]    # name -> (bars, close_ts ns)
    fwd_px: pd.Series                                    # 1h closes at close time
    funding_t: np.ndarray                                # ms
    funding_r: np.ndarray


def _load(cache: BarCache, symbol: str) -> _Frames | None:
    views = {}
    now_ns = pd.Timestamp.now(tz="UTC").value
    for name, (interval, _) in VIEWS.items():
        df = cache.bars(symbol, interval)
        close_ts = (df.index + pd.Timedelta(milliseconds=INTERVAL_MS[interval])).asi8
        # A still-forming bar would leak a provisional close into fwd returns.
        done = close_ts <= now_ns
        df, close_ts = df[done], close_ts[done]
        if df.empty:
            return None
        views[name] = (df, close_ts)
    hourly, h_close = views["medium"]
    fwd = pd.Series(hourly["close"].to_numpy(), index=pd.DatetimeIndex(h_close, tz="UTC"))
    fund = cache.funding(symbol)
    return _Frames(views, evaluation._as_ns(fwd),
                   fund["funding_time"].to_numpy(dtype="int64"),
                   fund["rate"].to_numpy(dtype=float))


def _view_at(frames: _Frames, name: str, t: pd.Timestamp) -> pd.DataFrame | None:
    """The bars a live run at `t` would see for this view, restricted to bars
    that had closed by `t`."""
    interval, lookback = VIEWS[name]
    df, close_ts = frames.views[name]
    t_ns = t.value
    i1 = int(np.searchsorted(close_ts, t_ns, side="right"))
    i0 = int(np.searchsorted(df.index.asi8, (t - lookback).value, side="left"))
    n = i1 - i0
    if n < 5:
        return None
    # The daily view is allowed to be short (a young coin), as it is live;
    # the intraday views feed technical_score and must be mostly there.
    expected = lookback / pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    if name != "daily" and n < _MIN_COVERAGE * expected:
        return None
    return df.iloc[i0:i1]


def _positioning_at(frames: _Frames, t: pd.Timestamp, cfg: Config) -> float:
    pc = cfg.positioning
    t_ms = t.value // 1_000_000
    i1 = int(np.searchsorted(frames.funding_t, t_ms, side="right"))
    # The live store's reference window: the last `history_periods` settlements.
    rates = frames.funding_r[max(0, i1 - pc.history_periods):i1]
    if len(rates) < pc.min_periods:
        return np.nan
    return positioning.score_rates(rates, pc)[3]


def _funding_until(frames: _Frames, t: pd.Timestamp) -> np.ndarray:
    return frames.funding_r[:int(np.searchsorted(frames.funding_t, t.value // 1_000_000,
                                                 side="right"))]


def _regime_at(frames: dict[str, _Frames], t: pd.Timestamp,
               cfg: Config) -> tuple[str | None, float]:
    """The live gate's BTC-trend verdict from BTC's daily bars closed by t."""
    btc = frames.get("BTC")
    daily = _view_at(btc, "daily", t) if btc is not None else None
    if daily is None or not cfg.regime.enabled:
        return None, np.nan
    v = regime_mod.classify(daily["close"], cfg.regime)
    return v["state"], float(v["exposure_scale"])


def replay(cache: BarCache, cfg: Config = CONFIG, days: int = 365,
           end: datetime | None = None, symbols: list[str] | None = None,
           hour_utc: int = 13, horizons: tuple[int, ...] = (1, 7, 30),
           verbose: bool = True,
           signals: dict[str, Signal] | None = None,
           social_history=None) -> pd.DataFrame:
    """Score every symbol once per day at `hour_utc`, from cached bars only.

    One row per (step, symbol): run_id, ts, the FAMILIES, the technical
    COMPONENTS, one column per candidate signal, the step's regime verdict
    (`regime_state` / `regime_scale`, NaN without BTC in the cache), and
    fwd_<h>d for each horizon (NaN where the exit bar is past the end of the
    cache). A signal that raises scores NaN for that row.

    `social_history` (a `social_backfill.SocialHistory`) adds `social` and
    `composite_social`, NaN at steps the archive doesn't cover.
    """
    signals = signals or {}
    use_social = social_history is not None and not social_history.empty
    end = pd.Timestamp(end or datetime.now(timezone.utc)).tz_convert("UTC")
    first = (end - pd.Timedelta(days=days)).normalize() + pd.Timedelta(hours=hour_utc)
    steps = pd.date_range(first, end, freq="1D")

    frames = {}
    for sym in symbols or cfg.symbols:
        if sym in STABLECOINS:
            continue
        f = _load(cache, sym)
        if f is None:
            if verbose:
                print(f"  {sym:<6} not in cache; skipped (run sync first)")
            continue
        frames[sym] = f

    weights = cfg.weights.normalized()
    blend = max(0.0, min(1.0, cfg.weights.xsec_momentum_blend))
    w_price = weights.technical + weights.positioning
    w_all = w_price + weights.social
    out: list[dict[str, Any]] = []
    t_start = time.time()
    for k, t in enumerate(steps):
        rows = []
        for sym, f in frames.items():
            short, medium, daily = (_view_at(f, v, t) for v in ("short", "medium", "daily"))
            if short is None or medium is None or daily is None:
                continue
            s_snap = indicators.summarize(indicators.enrich(short, cfg.bands))
            m_snap = indicators.summarize(indicators.enrich(medium, cfg.bands))
            # xsec only reads ret_24b / ret_72b, which summarize computes from
            # the raw closes; skipping enrich here saves most of the replay time.
            d_snap = indicators.summarize(daily)
            tech_raw, parts = engine.technical_score(s_snap, m_snap)
            extra = {}
            if signals:
                ctx = SignalContext(sym, t, short, medium, daily, s_snap, m_snap,
                                    _funding_until(f, t))
                for name, fn in signals.items():
                    try:
                        v = float(fn(ctx))
                    except Exception:  # noqa: BLE001 - a buggy idea scores NaN, not a crash
                        v = np.nan
                    extra[name] = v if np.isfinite(v) else np.nan
            rows.append({"symbol": sym, "technical_raw": tech_raw,
                         "positioning": _positioning_at(f, t, cfg),
                         "parts": parts, "extra": extra,
                         "components": {"snapshot_daily": d_snap}})
        if len(rows) < 3:
            continue
        xsec = engine._xsec_momentum(rows)
        state, scale = _regime_at(frames, t, cfg)
        soc = social_history.score_at(t, cfg) if use_social else {}
        run_id = f"bt-{t:%Y%m%dT%H}"
        for r in rows:
            xz = xsec.get(r["symbol"], 0.0)
            tech = engine._clip((1.0 - blend) * r["technical_raw"] + blend * xz)
            pos = r["positioning"]
            pos0 = 0.0 if np.isnan(pos) else pos
            comp = ((weights.technical * tech + weights.positioning * pos0) / w_price
                    if w_price > 0 else tech)
            row = {"run_id": run_id, "ts": t, "symbol": r["symbol"],
                   "technical_raw": round(r["technical_raw"], 4), "xsec": round(xz, 4),
                   "technical": round(tech, 4),
                   "positioning": round(pos, 4) if not np.isnan(pos) else np.nan,
                   "composite": round(comp, 4),
                   **{c: round(float(r["parts"].get(c[3:], 0.0)), 4) for c in COMPONENTS},
                   **r["extra"],
                   "regime_state": state, "regime_scale": scale}
            if use_social:
                sv = soc.get(r["symbol"], np.nan) if soc else np.nan
                row["social"] = sv
                row["composite_social"] = (
                    round((weights.technical * tech + weights.positioning * pos0
                           + weights.social * sv) / w_all, 4)
                    if w_all > 0 and not np.isnan(sv) else np.nan)
            for h in horizons:
                row[f"fwd_{h}d"] = evaluation._fwd_return(frames[r["symbol"]].fwd_px, t, h)
            out.append(row)
        if verbose and (k + 1) % 60 == 0:
            print(f"  replayed {k + 1}/{len(steps)} days  ({time.time() - t_start:.0f}s)")
    df = pd.DataFrame(out)
    df.attrs["signals"] = list(signals)
    return df


# --------------------------------------------------------------------------
# Portfolio simulation
# --------------------------------------------------------------------------
def simulate(fr: pd.DataFrame, cfg: Config = CONFIG, hold_days: int = 7,
             top_n: int | None = None, floor: float | None = None,
             cost_bps: float | None = None, regime: bool = False,
             score_col: str = "composite", buffer: int = 0) -> dict[str, Any]:
    """Long the top-N by `score_col`, equal weight, rebalanced every `hold_days`.

    Names must also clear `floor` (default `risk.min_composite_floor`; only
    applied when ranking by the composite - a candidate signal is unscaled);
    empty slots sit in cash. Each rebalance pays `cost_bps` per side on the
    weight that changed (default taker fee + modelled slippage). Periods don't
    overlap, so the Sharpe is honest. Benchmarks: BTC and an equal-weight basket
    of the whole scored universe, both cost-free - the bar a strategy has to
    clear.

    `regime=True` applies the live BTC-trend gate as replayed: each pick's
    weight is `exposure_scale / top_n` (neutral = half size), and a risk_off
    step holds cash. Live, risk_off only blocks NEW buys and existing positions
    exit on their own terms; a rebalancing sim has no "existing" book, so this
    is the closest honest analogue, not an exact one.

    `buffer` is a rank buffer (hysteresis): a name already held is kept while
    it still ranks within the top `top_n + buffer`, and only the freed slots
    go to the best new names. At 0 every rebalance re-forms the top-N from
    scratch, paying round-trip costs for rank noise around the cut.
    """
    ycol = f"fwd_{hold_days}d"
    if fr.empty or ycol not in fr.columns or score_col not in fr.columns:
        return {}
    top_n = top_n or cfg.risk.max_proposals
    floor = cfg.risk.min_composite_floor if floor is None else floor
    if score_col != "composite":
        floor = -np.inf
    cost_bps = (cfg.fees.effective_taker_bps + cfg.fees.slippage_bps
                if cost_bps is None else cost_bps)
    gated = regime and "regime_scale" in fr.columns

    runs = sorted(fr.dropna(subset=[ycol])["run_id"].unique())
    periods = []
    prev: dict[str, float] = {}
    for run_id in runs[::hold_days]:
        g = fr[(fr["run_id"] == run_id)].dropna(subset=[ycol, score_col])
        if g.empty:
            continue
        eligible = g[g[score_col] > floor].sort_values(score_col, ascending=False)
        if buffer > 0 and prev:
            zone = list(eligible["symbol"].head(top_n + buffer))
            keep = [s for s in zone if s in prev][:top_n]
            fresh = [s for s in eligible["symbol"] if s not in keep][:top_n - len(keep)]
            picks = eligible[eligible["symbol"].isin(keep + fresh)]
        else:
            picks = eligible.head(top_n)
        scale, state = 1.0, None
        if gated:
            state = g["regime_state"].iloc[0]
            sc = g["regime_scale"].iloc[0]
            scale = float(sc) if pd.notna(sc) else 1.0
            if state == "risk_off" and cfg.regime.block_new_entries_when_risk_off:
                scale = 0.0
        w = {s: scale / top_n for s in picks["symbol"]}
        turnover = sum(abs(w.get(s, 0.0) - prev.get(s, 0.0)) for s in set(w) | set(prev))
        cost = turnover * cost_bps / 10_000.0
        gross = float(sum(w[s] * r for s, r in zip(picks["symbol"], picks[ycol])))
        btc = g.loc[g["symbol"] == "BTC", ycol]
        periods.append({"ts": g["ts"].iloc[0], "picks": ",".join(picks["symbol"]),
                        "gross": gross, "cost": cost, "net": gross - cost,
                        "turnover": turnover, "exposure": sum(w.values()),
                        "regime": state,
                        "btc": float(btc.iloc[0]) if not btc.empty else np.nan,
                        "basket": float(g[ycol].mean())})
        prev = w
    p = pd.DataFrame(periods)
    if p.empty:
        return {}

    per_year = 365.0 / hold_days

    def _stats(r: pd.Series) -> dict[str, float]:
        r = r.dropna()
        if r.empty:                       # e.g. no BTC in the replayed universe
            return {"total_%": np.nan, "cagr_%": np.nan, "sharpe": np.nan, "max_dd_%": np.nan}
        curve = (1 + r).cumprod()
        sd = r.std(ddof=1)
        return {"total_%": round((curve.iloc[-1] - 1) * 100, 2),
                "cagr_%": round((curve.iloc[-1] ** (per_year / len(r)) - 1) * 100, 2),
                "sharpe": round(float(r.mean() / sd * np.sqrt(per_year)), 2) if sd > 0 else np.nan,
                "max_dd_%": round(evaluation._drawdown(curve), 2)}

    table = pd.DataFrame({name: _stats(p[col]) for name, col in
                          (("strategy", "net"), ("strategy_gross", "gross"),
                           ("btc", "btc"), ("basket", "basket"))}).T
    return {"periods": p, "stats": table, "hold_days": hold_days, "top_n": top_n,
            "floor": floor, "cost_bps": cost_bps, "regime": gated,
            "score_col": score_col, "buffer": buffer,
            "avg_exposure": round(float(p["exposure"].mean()), 3),
            "avg_turnover": round(float(p["turnover"].mean()), 3),
            "fee_drag_%": round(float(p["cost"].sum()) * 100, 2),
            "hit_vs_basket_%": round(float((p["net"] > p["basket"]).mean()) * 100, 1)}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def report(fr: pd.DataFrame, cfg: Config = CONFIG, primary_horizon: int = 7,
           hold_days: int = 7, buffer: int | None = None) -> dict[str, Any]:
    """Print the battery and return the frames."""
    rule = "═" * 78
    print(f"{rule}\nPRICE-FAMILY BACKTEST (point-in-time replay)\n{rule}")
    if fr.empty:
        print("  nothing replayed - is the cache populated?")
        return {}
    horizons = tuple(int(c[4:-1]) for c in fr.columns if c.startswith("fwd_"))
    print(f"  {fr['run_id'].nunique()} daily steps × {fr['symbol'].nunique()} symbols   "
          f"window {fr['ts'].min():%Y-%m-%d} → {fr['ts'].max():%Y-%m-%d}")
    print("  composite here = technical + positioning only (social/catalyst have no history)")

    fams = _families(fr)
    if "social" in fams:
        cov = fr["social"].notna().groupby(fr["run_id"]).any().mean()
        print(f"  social: Reddit backfill, covers {cov:.0%} of steps "
              f"(composite_social = price + social)")
    ic = evaluation.information_coefficient(fr, horizons, fams)
    print("\n  Information coefficient, whole window:")
    print("  " + ic.to_string(index=False).replace("\n", "\n  "))

    # Out-of-sample sanity: a real signal should keep its sign in both halves.
    runs = sorted(fr["run_id"].unique())
    mid = runs[len(runs) // 2]
    halves = []
    for label, part in (("first half", fr[fr["run_id"] < mid]),
                        ("second half", fr[fr["run_id"] >= mid])):
        h = evaluation.information_coefficient(part, (primary_horizon,), fams)
        if not h.empty:
            halves.append(h.set_index("family")["mean_IC"].rename(label))
    if halves:
        print(f"\n  {primary_horizon}d mean IC by half (a real signal keeps its sign):")
        split = pd.concat(halves, axis=1)
        print("  " + split.to_string().replace("\n", "\n  "))

    n = cfg.risk.max_proposals
    spreads = {fam: evaluation.quantile_spread(fr.dropna(subset=[fam]), primary_horizon, n, fam)
               for fam in fams}
    spread_tbl = pd.DataFrame({
        fam: {"top_%": s["top_mean_ret"] * 100, "bottom_%": s["bottom_mean_ret"] * 100,
              "spread_%": s["spread"] * 100, "top_vs_universe_%": s["top_minus_universe"] * 100}
        for fam, s in spreads.items() if s}).T.round(2)
    if not spread_tbl.empty:
        print(f"\n  Top-{n} vs bottom-{n}, {primary_horizon}d forward (mean per step):")
        print("  " + spread_tbl.to_string().replace("\n", "\n  "))

    at_h = ic[ic["horizon_d"] == primary_horizon].set_index("family")["mean_IC"].dropna()
    comps = at_h.drop(["composite", "technical", "composite_social"], errors="ignore")
    if "composite" in at_h and not comps.empty:
        best = comps.idxmax()
        verdict = ("ADDS value over" if at_h["composite"] >= comps.max() - 0.01
                   else "is WORSE than")
        print(f"\n  Blend: composite IC {at_h['composite']:+.4f} {verdict} its best "
              f"component '{best}' ({comps.max():+.4f}) at {primary_horizon}d.")

    lab = _report_lab(fr, horizons, primary_horizon)
    regime_tbl = _report_regime(fr, cfg, primary_horizon)

    buffer = cfg.risk.max_proposals if buffer is None else buffer
    has_regime = "regime_state" in fr.columns and fr["regime_state"].notna().any()
    sims = {"top-N": simulate(fr, cfg, hold_days)}
    if buffer > 0:
        sims[f"top-N, buffer {buffer}"] = simulate(fr, cfg, hold_days, buffer=buffer)
    if has_regime:
        sims["top-N + regime gate"] = simulate(fr, cfg, hold_days, regime=True)
        if buffer > 0:
            sims[f"+ gate, buffer {buffer}"] = simulate(fr, cfg, hold_days, regime=True,
                                                        buffer=buffer)
    sims = {k: v for k, v in sims.items() if v}
    if sims:
        first = next(iter(sims.values()))
        print(f"\n  Portfolio: top-{first['top_n']} by composite (> {first['floor']:+.2f}), "
              f"rebalanced every {hold_days}d, {first['cost_bps']:.0f} bps/side, "
              f"{len(first['periods'])} periods:")
        rows = {name: sim["stats"].loc["strategy"] for name, sim in sims.items()}
        rows["(gross, no gate)"] = first["stats"].loc["strategy_gross"]
        rows["btc"] = first["stats"].loc["btc"]
        rows["basket"] = first["stats"].loc["basket"]
        print("  " + pd.DataFrame(rows).T.to_string().replace("\n", "\n  "))
        for name, sim in sims.items():
            print(f"    {name:<22} turnover {sim['avg_turnover']:.2f}/rebalance   "
                  f"exposure {sim['avg_exposure']:.2f}   fee drag {sim['fee_drag_%']:.2f}% "
                  f"(simple sum)   beat the basket {sim['hit_vs_basket_%']:.0f}% of periods")
    print("  caveats: survivorship (today's universe), funding only ~3 months deep, "
          "macro dampener not replayed.")
    print(rule)
    return {"forward_returns": fr, "ic": ic, "spreads": spread_tbl,
            "simulation": sims.get("top-N", {}), "simulations": sims,
            "lab": lab, "regime": regime_tbl}


def _families(fr: pd.DataFrame) -> tuple[str, ...]:
    extra = tuple(c for c in ("social", "composite_social")
                  if c in fr.columns and fr[c].notna().any())
    return FAMILIES + extra


def _ic_grid(ic: pd.DataFrame) -> pd.DataFrame:
    """family x horizon, each cell 'mean_IC (t)'."""
    if ic.empty:
        return ic
    cell = ic.apply(lambda r: (f"{r['mean_IC']:+.4f} ({r['t_stat']:+.1f})"
                               if pd.notna(r["mean_IC"]) and pd.notna(r["t_stat"])
                               else (f"{r['mean_IC']:+.4f}" if pd.notna(r["mean_IC"]) else "—")),
                    axis=1)
    grid = ic.assign(cell=cell).pivot(index="family", columns="horizon_d", values="cell")
    grid.columns = [f"{h}d IC (t)" for h in grid.columns]
    return grid.reindex([f for f in ic["family"].drop_duplicates()])


def _report_lab(fr: pd.DataFrame, horizons: tuple[int, ...],
                primary_horizon: int) -> pd.DataFrame:
    """IC of technical_score's pieces and of any candidate signals."""
    cols = ([c for c in COMPONENTS if c in fr.columns]
            + [c for c in fr.attrs.get("signals", []) if c in fr.columns])
    if not cols:
        return pd.DataFrame()
    fams = ("technical_raw", *cols)
    ic = evaluation.information_coefficient(fr, horizons, fams)
    if ic.empty:
        return ic
    print("\n  Technical score piece by piece, and candidate signals — mean IC (t):")
    print("  " + _ic_grid(ic).to_string().replace("\n", "\n  "))
    runs = sorted(fr["run_id"].unique())
    mid = runs[len(runs) // 2]
    halves = []
    for h in sorted({1, primary_horizon}):
        for label, part in (("1st", fr[fr["run_id"] < mid]), ("2nd", fr[fr["run_id"] >= mid])):
            x = evaluation.information_coefficient(part, (h,), fams)
            if not x.empty:
                halves.append(x.set_index("family")["mean_IC"].rename(f"{h}d {label} half"))
    if halves:
        print("  by half (a real signal keeps its sign):")
        print("  " + pd.concat(halves, axis=1).round(4).to_string().replace("\n", "\n  "))
    print("    tc_* are technical_score's parts before weighting; a part with a "
          "significant NEGATIVE IC is one the blend should flip or drop.")
    return ic


def _report_regime(fr: pd.DataFrame, cfg: Config, horizon: int) -> pd.DataFrame:
    """Forward returns by replayed regime state — does the gate separate anything?"""
    ycol = f"fwd_{horizon}d"
    if "regime_state" not in fr.columns or ycol not in fr.columns \
            or not fr["regime_state"].notna().any():
        return pd.DataFrame()
    n = cfg.risk.max_proposals
    per_step = []
    for run_id, g in fr.dropna(subset=[ycol]).groupby("run_id"):
        top = g.nlargest(n, "composite")[ycol].mean()
        btc = g.loc[g["symbol"] == "BTC", ycol]
        per_step.append({"state": g["regime_state"].iloc[0], "universe": g[ycol].mean(),
                         "top": top, "btc": btc.iloc[0] if not btc.empty else np.nan})
    ps = pd.DataFrame(per_step).dropna(subset=["state"])
    if ps.empty:
        return ps
    tbl = ps.groupby("state").agg(steps=("universe", "size"),
                                  universe_fwd_pct=("universe", "mean"),
                                  top_fwd_pct=("top", "mean"),
                                  btc_fwd_pct=("btc", "mean"))
    for c in ("universe_fwd_pct", "top_fwd_pct", "btc_fwd_pct"):
        tbl[c] = (tbl[c] * 100).round(2)
    tbl = tbl.reindex([s for s in ("risk_on", "neutral", "risk_off") if s in tbl.index])
    print(f"\n  Regime gate (BTC trend, replayed) — mean {horizon}d forward return by state:")
    print("  " + tbl.to_string().replace("\n", "\n  "))
    print("    the gate earns its keep if risk_off rows are weaker than risk_on.")
    return tbl


def run(cfg: Config = CONFIG, days: int = 365, cache_path: Path | str = DEFAULT_CACHE,
        do_sync: bool = True, symbols: list[str] | None = None,
        hold_days: int = 7, signals: dict[str, Signal] | None = None,
        social_path: Path | str | None = None, buffer: int | None = None) -> dict[str, Any]:
    """sync → replay → report, the notebook one-liner. `signals` defaults to
    every built-in candidate in signals.py; pass {} for none. `social_path`
    defaults to the social backfill DB when it exists."""
    from . import signals as signals_mod
    social_hist = load_social_history(social_path)
    cache = BarCache(cache_path)
    end = datetime.now(timezone.utc)
    if do_sync:
        print(f"[sync] {cache.path}")
        sync(cache, cfg, days, end, symbols)
    print("[replay]")
    horizons = tuple(sorted({1, 7, 30, hold_days}))
    fr = replay(cache, cfg, days, end, symbols, horizons=horizons,
                signals=signals_mod.BUILTIN if signals is None else signals,
                social_history=social_hist)
    return report(fr, cfg, hold_days=hold_days, primary_horizon=hold_days, buffer=buffer)


def load_social_history(path: Path | str | None = None):
    """The social backfill as a SocialHistory, or None if there isn't one.

    Opened read-only in spirit: SocialHistory only reads. Default path is
    social_backfill.DEFAULT_PATH, used only if the file already exists.
    """
    from . import social_backfill
    from .store import Store
    p = Path(path) if path else social_backfill.DEFAULT_PATH
    if not p.exists():
        return None
    hist = social_backfill.SocialHistory(Store(p))
    return None if hist.empty else hist
