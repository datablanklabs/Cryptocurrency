"""Feature 6: macro backdrop via Kalshi event-contract prices.

Kalshi is a CFTC-regulated prediction market: each contract settles at $1 if a
YES proposition resolves true, $0 otherwise, so its live price IS a market-
implied probability — continuously updated by people with money on the line.
This module reads a small, hand-maintained set of MACRO contracts
(`config.MACRO_SERIES`: Fed rate decisions, CPI prints, government-shutdown
risk, recession odds — the kind of broad macro uncertainty that moves risk
assets generally, crypto included, in a way nothing else in this system can
see coming) and blends them into ONE macro score in [-1, 1] that `regime.py`
uses to dampen — never boost — the deployment scale its own BTC-trend read
already produced.

Two honest caveats, matching catalysts.py's:

  The `direction` on each configured series (does a YES resolution favor or
  hurt risk assets?) is a prior I wrote down, not a fitted coefficient. "Fed
  cuts rates" -> bullish and "government shutdown" -> bearish are defensible,
  but they are still judgement calls, all listed in MACRO_SERIES so you can
  argue with them.

  Kalshi's crypto-specific contracts (BTC/ETH price-range markets) are thin
  and short-dated — not modelled here. The genuine edge is on the macro side:
  event contracts on things a crypto-only book has no other way to see coming.

Auth is Kalshi's signed-request scheme: every request, even a read, carries an
RSA-PSS signature over `timestamp + method + path`, made with the private key
half of an API key pair generated in Kalshi account settings. See
.env.example for how to provision KALSHI_API_KEY_ID and the private key.
"""

from __future__ import annotations

import base64
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from .config import CONFIG, Config, MACRO_SERIES, get_secret
from .store import Store, iso, utcnow


class KalshiError(RuntimeError):
    """Kalshi rejected the request, or the API key isn't usable."""


# --------------------------------------------------------------------------
# Signed REST client
# --------------------------------------------------------------------------
class KalshiClient:
    """Thin signed-REST client for Kalshi's market-data endpoints.

    Only what this module needs: GET /markets and GET /series. Every request
    is signed with RSA-PSS(SHA256) over `timestamp + "GET" + path`, per
    Kalshi's documented auth flow.
    """

    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg
        self.mc = cfg.macro
        self.key_id = get_secret("kalshi_key_id")
        self._private_key: Any = None
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": cfg.user_agent})

    @property
    def configured(self) -> bool:
        return bool(self.key_id) and self._load_private_key() is not None

    def _load_private_key(self) -> Any:
        if self._private_key is not None:
            return self._private_key
        pem = get_secret("kalshi_private_key")
        if not pem:
            path = get_secret("kalshi_private_key_path")
            if not path:
                return None
            try:
                pem = Path(path).expanduser().read_text()
            except OSError:
                return None
        try:
            self._private_key = serialization.load_pem_private_key(
                pem.encode("utf-8"), password=None)
        except (ValueError, TypeError):
            return None
        return self._private_key

    def _headers(self, method: str, path: str) -> dict[str, str]:
        key = self._load_private_key()
        ts = str(int(time.time() * 1000))
        message = f"{ts}{method.upper()}{path}".encode("utf-8")
        sig = key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                       salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("ascii"),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    def get(self, endpoint: str, params: dict | None = None) -> Any:
        """GET `{api_prefix}{endpoint}` (e.g. endpoint="/markets")."""
        if not self.configured:
            raise KalshiError(
                "KALSHI_API_KEY_ID and a private key (KALSHI_PRIVATE_KEY_PATH "
                "or KALSHI_PRIVATE_KEY) are required — see .env.example.")
        path = f"{self.mc.api_prefix}{endpoint}"
        url = f"{self.mc.base_url}{path}"
        last: Exception | None = None
        for attempt in range(self.cfg.http_retries):
            try:
                r = self.session.get(url, params=params, headers=self._headers("GET", path),
                                     timeout=self.cfg.http_timeout)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as exc:  # noqa: BLE001 - surface the last error to caller
                last = exc
                time.sleep(0.4 * (attempt + 1))
        raise KalshiError(f"GET {path} failed after {self.cfg.http_retries} tries: {last}")


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------
def _market_probability(m: dict[str, Any]) -> float | None:
    """Implied P(YES) for a Kalshi market, 0..1.

    Prefers the yes bid/ask midpoint — what you could actually transact at
    right now — over `last_price`, which can be stale on a thin market.
    Untraded markets are filtered downstream by `min_volume`, not here.
    """
    bid, ask = m.get("yes_bid"), m.get("yes_ask")
    if bid is not None and ask is not None and (bid > 0 or ask > 0):
        return ((bid + ask) / 2.0) / 100.0
    last = m.get("last_price")
    return (last / 100.0) if last is not None else None


def fetch(store: Store, cfg: Config = CONFIG, verbose: bool = True) -> int:
    """Pull the latest snapshot of every configured series and persist it.

    Best-effort per series, same as catalysts/positioning: one bad ticker or a
    rate limit must not sink the others. No credentials or an empty
    MACRO_SERIES both degrade to "nothing fetched", not an error.
    """
    mc = cfg.macro
    if not mc.enabled:
        return 0
    if not MACRO_SERIES:
        if verbose:
            print("  macro: MACRO_SERIES is empty (config.py) — nothing to fetch")
        return 0

    client = KalshiClient(cfg)
    if not client.configured:
        if verbose:
            print("  macro: KALSHI_API_KEY_ID / private key not set — skipping")
        return 0

    fetched = iso()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for series in MACRO_SERIES:
        try:
            data = client.get("/markets", params={
                "series_ticker": series.series_ticker, "status": "open",
                "limit": mc.max_markets_per_series,
            })
            markets = data.get("markets") or []
        except Exception as exc:  # noqa: BLE001 - one series failing isn't fatal
            missing.append(series.series_ticker)
            if verbose:
                print(f"  {series.label}: failed ({exc})")
            time.sleep(mc.request_delay)
            continue

        markets = sorted(markets, key=lambda m: m.get("close_time") or "")
        kept = 0
        for m in markets[: mc.max_markets_per_series]:
            prob = _market_probability(m)
            vol = int(m.get("volume") or 0)
            if prob is None or vol < mc.min_volume:
                continue
            rows.append({
                "ticker": m.get("ticker"), "series_ticker": series.series_ticker,
                "label": series.label, "title": m.get("title") or m.get("subtitle") or "",
                "probability": round(prob, 4), "volume": vol,
                "close_time": m.get("close_time"), "fetched_at": fetched,
            })
            kept += 1
        if verbose:
            print(f"  {series.label:<42} {len(markets):>2} open market(s) -> {kept} kept")
        time.sleep(mc.request_delay)

    if rows:
        store.upsert_macro(rows)
    if verbose:
        print(f"  -> {len(rows)} Kalshi market snapshot(s) stored"
              + (f" (unreachable: {', '.join(missing)})" if missing else ""))
    return len(rows)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score(store: Store, cfg: Config = CONFIG) -> dict[str, Any]:
    """Blend the latest Kalshi snapshots into one macro score in [-1, 1].

    Each series contributes `direction * (probability - 0.5) * 2 * weight` — a
    market pricing its proposition at a coin flip (50%) contributes nothing;
    pricing it near-certain contributes its full weight. `direction=0` series
    (no inherent direction) contribute `-abs(deviation) * weight` instead: any
    extreme reading is read as elevated uncertainty, whichever way it points.

    The blend is a weight-normalised mean, not a saturating tanh sum like
    catalysts/events — Kalshi prices are already bounded probabilities, there
    is nothing left to saturate.
    """
    mc = cfg.macro
    out: dict[str, Any] = {
        "enabled": mc.enabled, "score": 0.0, "n_series": 0, "n_markets": 0,
        "top_label": "", "top_probability": None, "top_contribution": 0.0,
        "note": "", "detail": pd.DataFrame(),
    }
    if not mc.enabled:
        out["note"] = "macro disabled (CONFIG.macro.enabled = False)"
        return out
    if not MACRO_SERIES:
        out["note"] = "MACRO_SERIES is empty (config.py)"
        return out

    since = utcnow() - timedelta(hours=mc.stale_after_hours)
    snap = store.macro_latest(since)
    if snap.empty:
        out["note"] = f"no Kalshi snapshot within {mc.stale_after_hours:.0f}h — run macro.fetch()"
        return out

    by_series = {s.series_ticker: s for s in MACRO_SERIES}
    rows = []
    for series_ticker, group in snap.groupby("series_ticker"):
        cs = by_series.get(series_ticker)
        if cs is None:
            continue
        prob = float(group["probability"].mean())
        deviation = (prob - 0.5) * 2.0     # -1..+1: how far from a coin flip
        if cs.direction == 0:
            contribution = -abs(deviation) * cs.weight
        else:
            contribution = cs.direction * deviation * cs.weight
        rows.append({
            "series_ticker": series_ticker, "label": cs.label,
            "probability": round(prob, 4), "direction": cs.direction,
            "weight": cs.weight, "contribution": round(contribution, 4),
            "n_markets": int(len(group)),
        })

    if not rows:
        out["note"] = "no stored snapshot matches a configured series"
        return out

    detail = (pd.DataFrame(rows)
              .sort_values("contribution", key=abs, ascending=False)
              .reset_index(drop=True))
    total_weight = float(detail["weight"].sum())
    net = float(detail["contribution"].sum() / total_weight) if total_weight > 0 else 0.0
    net = max(-1.0, min(1.0, net))
    top = detail.iloc[0]

    out.update(
        score=round(net, 4), n_series=int(len(detail)), n_markets=int(detail["n_markets"].sum()),
        top_label=str(top["label"]), top_probability=round(float(top["probability"]), 4),
        top_contribution=round(float(top["contribution"]), 4),
        note=(f"{top['label']} {top['probability'] * 100:.0f}% "
              f"(net {net:+.2f} across {len(detail)} series)"),
        detail=detail,
    )
    return out


def describe(m: dict[str, Any]) -> str:
    """One-line summary for the notebook / CLI."""
    if not m.get("enabled", True):
        return "macro: OFF (CONFIG.macro.enabled = False)"
    if not m.get("n_series"):
        return f"macro: n/a — {m.get('note', 'no data')}"
    icon = "🔴" if m["score"] <= -0.3 else ("🟢" if m["score"] >= 0.3 else "🟡")
    return f"{icon} macro: {m['score']:+.2f}  —  {m['note']}"


# --------------------------------------------------------------------------
# Discovery — not used by the pipeline
# --------------------------------------------------------------------------
def list_series(cfg: Config = CONFIG, category: str = "", query: str = "") -> pd.DataFrame:
    """Look up real Kalshi series tickers to populate `config.MACRO_SERIES`.

    Kalshi's series catalog changes over time and this repo can't know today's
    exact tickers, so MACRO_SERIES ships with placeholders. Run this once with
    working credentials — e.g. `macro.list_series(query="fed")` — read off the
    real tickers, and hand-write them into config.py, the same way
    SCHEDULED_EVENTS is hand-maintained.
    """
    client = KalshiClient(cfg)
    params: dict[str, Any] = {}
    if category:
        params["category"] = category
    data = client.get("/series", params=params)
    rows = data.get("series") or []
    df = pd.DataFrame([
        {"ticker": r.get("ticker"), "title": r.get("title"), "category": r.get("category")}
        for r in rows
    ])
    if query and not df.empty:
        needle = query.lower()
        df = df[df["title"].str.lower().str.contains(needle, na=False)
                | df["ticker"].str.lower().str.contains(needle, na=False)]
    return df.reset_index(drop=True)
