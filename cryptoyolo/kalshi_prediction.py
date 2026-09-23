"""Feature 7: per-asset directional signal from Kalshi's own crypto price
markets.

`macro.py` reads Kalshi's MACRO event contracts (Fed decisions, CPI prints,
government-shutdown risk) to dampen the regime gate. This module reads the
OTHER thing Kalshi lists: markets on the price of a specific asset itself —
"will BTC be above $110,000 at Friday 5pm ET?" and similar, one ladder of
strikes per expiration, all sharing one series ticker
(`config.KALSHI_CRYPTO_SERIES`).

A ladder of (strike, P(YES)) pairs is a set of points on the market's own
implied survival curve for the future price: P(price > strike), decreasing in
strike. Interpolating that curve AT THE CURRENT SPOT PRICE gives P(price ends
above where it is right now) — a genuine, if short-dated, directional read
priced by people with money on the line, independent of everything else this
book already looks at. Unlike `MacroSeries`, no hand-set `direction` prior is
needed: "YES" here always and only means "price ends higher", so the
probability itself IS the signal.

Two honest caveats, matching macro.py's:

  Short-dated. Kalshi's crypto markets are typically same-day or same-week
  expiries, while this book generally holds 1-30 days
  (`exits.horizon_days`). Treat this as a near-term tilt, not a horizon
  match — hence `ScoreWeights.kalshi_prediction` defaulting low.

  Only "above a single strike" markets are read (`strike_type` "greater" /
  "greater_or_equal"). Kalshi also lists "between $X and $Y" range buckets on
  some series; those aren't a simple point on a survival curve and are
  skipped rather than mis-modelled.

Auth reuses `macro.KalshiClient` — the same signed-request scheme, the same
API key pair, no separate credential to provision.
"""

from __future__ import annotations

import re
import time
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from . import macro as macro_mod
from .config import CONFIG, Config, KALSHI_CRYPTO_SERIES
from .store import Store, iso, utcnow

_STRIKE_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")

# Range ("between") buckets don't reduce to one (strike, P(price > strike))
# point, so only single-threshold markets are read. An empty/missing
# strike_type is kept rather than dropped — some series omit the field on
# what is, in practice, a plain "above" market.
_SUPPORTED_STRIKE_TYPES = {"", "greater", "greater_or_equal"}


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------
def _strike(m: dict[str, Any]) -> float | None:
    """The price threshold a market resolves against, or None if unreadable.

    Prefers Kalshi's own `floor_strike` field. Falls back to parsing the
    first dollar figure out of the market's subtitle/title when the field is
    absent or unusable — this repo can't verify a given series' exact market
    schema without live credentials, the same caveat `macro.py` carries about
    series tickers themselves.
    """
    raw = m.get("floor_strike")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    text = m.get("subtitle") or m.get("title") or ""
    match = _STRIKE_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def fetch(store: Store, cfg: Config = CONFIG, verbose: bool = True) -> int:
    """Pull the latest open-strike ladder for every configured series and
    persist it. Best-effort per series, same contract as macro.fetch(): one
    bad series or a rate limit must not sink the others."""
    kc = cfg.kalshi_prediction
    if not kc.enabled:
        return 0
    if not KALSHI_CRYPTO_SERIES:
        if verbose:
            print("  kalshi_prediction: KALSHI_CRYPTO_SERIES is empty (config.py) — nothing to fetch")
        return 0

    client = macro_mod.KalshiClient(cfg)
    if not client.configured:
        if verbose:
            print("  kalshi_prediction: KALSHI_API_KEY_ID / private key not set — skipping")
        return 0

    fetched = iso()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for series in KALSHI_CRYPTO_SERIES:
        try:
            markets = macro_mod._open_markets(client, series.series_ticker)
        except Exception as exc:  # noqa: BLE001 - one series failing isn't fatal
            missing.append(series.series_ticker)
            if verbose:
                print(f"  {series.symbol} ({series.series_ticker}): failed ({exc})")
            time.sleep(kc.request_delay)
            continue

        # One expiration only: the ladder is a single survival curve, and
        # interpolating across strikes from different expiries is meaningless.
        event_markets = macro_mod._nearest_event(markets)
        kept = 0
        for m in event_markets:
            if str(m.get("strike_type") or "").lower() not in _SUPPORTED_STRIKE_TYPES:
                continue        # range bucket — not modelled, see module docstring
            strike = _strike(m)
            prob = macro_mod._market_probability(m)
            vol = macro_mod._market_volume(m)
            if strike is None or prob is None or vol < kc.min_volume:
                continue
            rows.append({
                "ticker": m.get("ticker"), "series_ticker": series.series_ticker,
                "symbol": series.symbol, "strike": strike,
                "probability": round(prob, 4), "volume": vol,
                "close_time": m.get("close_time"), "fetched_at": fetched,
            })
            kept += 1
        if verbose:
            event = (event_markets[0].get("event_ticker") or "") if event_markets else "no open event"
            print(f"  {series.symbol:<6} {series.series_ticker:<12} "
                  f"{len(markets):>3} open market(s), {event} -> {kept} kept")
        time.sleep(kc.request_delay)

    if rows:
        store.upsert_kalshi_prediction(rows)
    if verbose:
        print(f"  -> {len(rows)} Kalshi price-market snapshot(s) stored"
              + (f" (unreachable: {', '.join(missing)})" if missing else ""))
    return len(rows)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def latest_ladder(store: Store, cfg: Config = CONFIG) -> pd.DataFrame:
    """This run's usable strike ladders, one row per (symbol, strike),
    dropping any ticker whose latest snapshot is older than
    `stale_after_hours`. Fetched once per `engine.build_scores` call, ahead
    of the per-symbol loop — `score_one` then just interpolates it.

    Keeps only each series' most recent fetch — one event, one survival
    curve — and drops markets whose close_time has already passed: an
    earlier fetch's (possibly expired) event must never be interpolated
    together with the current one.
    """
    kc = cfg.kalshi_prediction
    since = utcnow() - timedelta(hours=kc.stale_after_hours)
    df = store.kalshi_prediction_latest(since)
    if df.empty:
        return df
    latest = df.groupby("series_ticker")["fetched_at"].transform("max")
    df = df[df["fetched_at"] == latest]
    closes = pd.to_datetime(df["close_time"], utc=True, errors="coerce", format="ISO8601")
    df = df[~(closes <= utcnow())]      # unparseable close_time (NaT) is kept
    return df.reset_index(drop=True)


def score_one(symbol: str, spot: float | None, ladder: pd.DataFrame,
             cfg: Config = CONFIG) -> dict[str, Any]:
    """Kalshi-implied directional score for one asset, in [-1, 1].

    Interpolates this asset's strike ladder at `spot` to get
    `p_up` = P(price at expiry > spot right now). A coin-flip reading (50%)
    scores 0; a near-certain one scores near its full ±1. `spot` outside the
    observed strike range scores 0.0 ("no opinion"): a ladder entirely above
    spot only bounds P(up) from below, so reading its edge probability as
    the answer would be a confident number the market never priced.

    Needs >= 2 open strikes to interpolate anything. A single-strike or empty
    ladder, or a missing spot price, scores 0.0 with a note explaining why —
    not a number derived from too little to mean anything.
    """
    out: dict[str, Any] = {
        "symbol": symbol, "n_strikes": 0, "implied_p_up": None,
        "kalshi_prediction": 0.0, "note": "",
    }
    kc = cfg.kalshi_prediction
    if not kc.enabled:
        out["note"] = "disabled (CONFIG.kalshi_prediction.enabled = False)"
        return out

    sub = ladder[ladder["symbol"] == symbol] if (ladder is not None and not ladder.empty) else pd.DataFrame()
    if sub.empty:
        out["note"] = f"no Kalshi price-market snapshot within {kc.stale_after_hours:.0f}h"
        return out
    if spot is None or not np.isfinite(spot) or spot <= 0:
        out["note"] = "no spot price to interpolate against"
        return out

    # Average any duplicate strikes so np.interp gets strictly increasing x.
    sub = sub.groupby("strike", as_index=False)["probability"].mean()
    strikes = sub["strike"].to_numpy(dtype=float)
    probs = sub["probability"].to_numpy(dtype=float)
    out["n_strikes"] = int(len(strikes))
    if len(strikes) < 2:
        out["note"] = "need >= 2 open strikes to interpolate a ladder"
        return out
    if not strikes[0] <= spot <= strikes[-1]:
        out["n_strikes"] = 0        # no usable read; describe() shows n/a
        out["note"] = (f"spot {spot:,.0f} outside the open strikes "
                       f"{strikes[0]:,.0f}-{strikes[-1]:,.0f} — no opinion")
        return out

    p_up = float(np.interp(float(spot), strikes, probs))
    deviation = (p_up - 0.5) * 2.0
    score = max(-1.0, min(1.0, deviation))

    out.update(
        n_strikes=int(len(strikes)), implied_p_up=round(p_up, 4),
        kalshi_prediction=round(score, 4),
        note=f"Kalshi-implied P(up)={p_up * 100:.0f}% across {len(strikes)} open strike(s)",
    )
    return out


def score_symbols(store: Store, cfg: Config = CONFIG,
                  spot_prices: dict[str, float] | None = None) -> pd.DataFrame:
    """Bulk convenience wrapper over `score_one`, for the notebook / ad-hoc
    use. `engine.build_scores` does NOT call this — it calls `score_one`
    directly inside its per-symbol loop, where the spot price it just fetched
    is already sitting in scope, rather than fetching prices twice."""
    ladder = latest_ladder(store, cfg)
    spot_prices = spot_prices or {}
    rows = [score_one(s, spot_prices.get(s), ladder, cfg) for s in cfg.symbols]
    return pd.DataFrame(rows)


def describe(m: dict[str, Any]) -> str:
    """One-line summary for the notebook / CLI."""
    if not m.get("n_strikes"):
        return f"kalshi_prediction: n/a — {m.get('note', 'no data')}"
    score = float(m.get("kalshi_prediction", 0.0))
    icon = "🔴" if score <= -0.3 else ("🟢" if score >= 0.3 else "🟡")
    return f"{icon} kalshi_prediction: {score:+.2f}  —  {m.get('note', '')}"


# --------------------------------------------------------------------------
# Discovery — not used by the pipeline
# --------------------------------------------------------------------------
def list_series(cfg: Config = CONFIG, category: str = "", query: str = "") -> pd.DataFrame:
    """Look up real Kalshi series tickers to populate `config.KALSHI_CRYPTO_SERIES`.

    Same tool as `macro.list_series` (Kalshi's catalog changes over time and
    this repo can't know today's exact tickers) — run this once with working
    credentials, e.g. `kalshi_prediction.list_series(query="bitcoin")`, and
    hand-write the real tickers into config.py.
    """
    return macro_mod.list_series(cfg, category=category, query=query)
