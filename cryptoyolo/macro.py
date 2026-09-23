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
already produced. Each series is scored against its own normal level
(`MacroSeries.baseline`), not a 50% coin flip, and only readings worse than
normal count.

Two honest caveats, matching catalysts.py's:

  The `direction` on each configured series (does a YES resolution favor or
  hurt risk assets?) and its `baseline` (what probability is normal for it)
  are priors I wrote down, not fitted coefficients. "Fed
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

from .config import CONFIG, Config, MACRO_SERIES, MacroSeries, get_secret
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
        # The file path first (see config.ENV_KEYS: inline PEM in .env is easy
        # to mangle), then the inline PEM — either one failing falls through
        # to the other rather than disabling Kalshi outright.
        candidates: list[str] = []
        path = get_secret("kalshi_private_key_path")
        if path:
            try:
                candidates.append(Path(path).expanduser().read_text())
            except (OSError, RuntimeError):
                # RuntimeError: expanduser() on an unresolvable "~name/" prefix
                # (e.g. a "~~/" typo) — treat like a missing file, don't crash.
                pass
        inline = get_secret("kalshi_private_key")
        if inline:
            # A one-line .env value can only carry literal "\n" escapes.
            candidates.append(inline.replace("\\n", "\n"))
        for pem in candidates:
            try:
                self._private_key = serialization.load_pem_private_key(
                    pem.encode("utf-8"), password=None)
                return self._private_key
            except (ValueError, TypeError):
                continue
        return None

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
                    last = KalshiError(f"HTTP 429 (rate limited) on attempt {attempt + 1}")
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
def _num(raw: Any) -> float | None:
    """Kalshi sends prices/volumes as decimal strings ("0.2200"); tolerate
    numbers and blanks too."""
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _market_probability(m: dict[str, Any]) -> float | None:
    """Implied P(YES) for a Kalshi market, 0..1.

    Prefers the yes bid/ask midpoint — what you could actually transact at
    right now — over the last trade, which can be stale on a thin market.
    Untraded markets are filtered downstream by `min_volume`, not here.

    Reads the current `*_dollars` fields (decimal strings, already 0..1) and
    falls back to the legacy integer-cent fields Kalshi used to send.
    """
    bid, ask = _num(m.get("yes_bid_dollars")), _num(m.get("yes_ask_dollars"))
    if bid is not None and ask is not None and (bid > 0 or ask > 0):
        return (bid + ask) / 2.0
    last = _num(m.get("last_price_dollars"))
    if last is not None:
        return last
    bid, ask = _num(m.get("yes_bid")), _num(m.get("yes_ask"))
    if bid is not None and ask is not None and (bid > 0 or ask > 0):
        return ((bid + ask) / 2.0) / 100.0
    last = _num(m.get("last_price"))
    return (last / 100.0) if last is not None else None


def _market_volume(m: dict[str, Any]) -> int:
    """Lifetime contracts traded: `volume_fp` (current, fractional string),
    else the legacy integer `volume`."""
    vol = _num(m.get("volume_fp"))
    if vol is None:
        vol = _num(m.get("volume"))
    return int(vol or 0)


def _open_markets(client: KalshiClient, series_ticker: str, max_pages: int = 10) -> list[dict]:
    """Every OPEN market in a series, following Kalshi's cursor pagination.

    A single page isn't enough: Kalshi doesn't return markets in expiry
    order, so a short `limit` can miss the nearest event entirely (a 100-row
    KXFED page started at the April meeting, skipping October).
    """
    markets: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        params: dict[str, Any] = {"series_ticker": series_ticker, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/markets", params=params)
        markets += data.get("markets") or []
        cursor = data.get("cursor")
        if not cursor:
            break
    return markets


def _nearest_event(markets: list[dict]) -> list[dict]:
    """The markets of the soonest-closing event in a series (e.g. the next
    FOMC meeting, the next CPI print), sorted by ticker."""
    events: dict[str, list[dict]] = {}
    for m in markets:
        events.setdefault(m.get("event_ticker") or m.get("ticker") or "", []).append(m)
    if not events:
        return []
    nearest = min(events, key=lambda e: min(m.get("close_time") or "~" for m in events[e]))
    return sorted(events[nearest], key=lambda m: m.get("ticker") or "")


def _select(series: MacroSeries, event_markets: list[dict], limit: int) -> list[dict]:
    """Which of the nearest event's markets represent this series' proposition.

    With `series.outcomes`, only the markets whose ticker ends in one of those
    suffixes (e.g. "-H25" / "-H26" on KXFEDDECISION, "-T0.3" on a strike
    ladder); `score()` sums them, so they must be mutually exclusive outcomes
    of the same event. Without, up to `limit` markets, averaged — right for a
    single yes/no market per event.
    """
    if series.outcomes:
        return [m for m in event_markets
                if any((m.get("ticker") or "").endswith(f"-{o}") for o in series.outcomes)]
    return event_markets[:limit]


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
            markets = _open_markets(client, series.series_ticker)
        except Exception as exc:  # noqa: BLE001 - one series failing isn't fatal
            missing.append(series.series_ticker)
            if verbose:
                print(f"  {series.label}: failed ({exc})")
            time.sleep(mc.request_delay)
            continue

        event_markets = _nearest_event(markets)
        selected = _select(series, event_markets, mc.max_markets_per_series)
        kept = 0
        for m in selected:
            prob = _market_probability(m)
            vol = _market_volume(m)
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
            event = (event_markets[0].get("event_ticker") or "") if event_markets else "no open event"
            print(f"  {series.label:<42} {len(markets):>3} open market(s), "
                  f"{event} -> {kept} kept")
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
def _combined_probability(series: MacroSeries, probs: pd.Series) -> float:
    """One fetch's markets for a series -> the series' P(YES). `outcomes` are
    mutually exclusive, so P(any of them) is their sum; otherwise average."""
    if series.outcomes:
        return min(1.0, float(probs.sum()))
    return float(probs.mean())


def _contribution(series: MacroSeries, prob: float) -> float:
    """How far worse than normal a reading is, as -weight..0.

    The "bad" side is YES for direction=-1 and NO for direction=+1; the
    excess of P(bad) over its baseline is scaled by the room above baseline,
    so certainty of the bad outcome scores the full -weight wherever the
    baseline sits. direction=0 (no inherent direction) reads a move either
    way from baseline as elevated uncertainty, scaled the same way.
    """
    b = series.baseline
    if series.direction == 0:
        excess = (prob - b) / (1.0 - b) if prob >= b else (b - prob) / b
    else:
        p_bad = prob if series.direction < 0 else 1.0 - prob
        b_bad = b if series.direction < 0 else 1.0 - b
        excess = max(0.0, (p_bad - b_bad) / (1.0 - b_bad))
    return -min(1.0, excess) * series.weight + 0.0     # + 0.0: no "-0.0" in the detail


def series_history(history: pd.DataFrame) -> pd.DataFrame:
    """`store.macro_history()` collapsed to one row per (series, fetch), with
    the same combined probability `score()` uses — what the history chart
    plots. Rows for series no longer in MACRO_SERIES are averaged."""
    cols = ["series_ticker", "label", "fetched_at", "probability"]
    if history is None or history.empty:
        return pd.DataFrame(columns=cols)
    by_series = {s.series_ticker: s for s in MACRO_SERIES}
    rows = []
    for (ticker, fetched), group in history.groupby(["series_ticker", "fetched_at"]):
        cs = by_series.get(ticker) or MacroSeries(str(group["label"].iloc[0]), ticker)
        rows.append({"series_ticker": ticker, "label": cs.label, "fetched_at": fetched,
                     "probability": _combined_probability(cs, group["probability"])})
    return pd.DataFrame(rows, columns=cols)


def score(store: Store, cfg: Config = CONFIG) -> dict[str, Any]:
    """Blend the latest Kalshi snapshots into one macro score in [-1, 0].

    Each series is measured against its own `baseline` — the probability that
    counts as normal for that event — not a 50% coin flip: a 6% recession
    chance is an ordinary reading, not good news. See `_contribution`: only a
    reading WORSE than baseline contributes (negatively, up to its full
    weight at certainty); at or better than baseline it contributes 0.

    Never positive, by design: the regime gate this feeds can only dampen,
    so a calm series has nothing to add — and if it could score positive it
    would cancel out an alarming series in the blend.

    The blend is a weight-normalised mean, not a saturating tanh sum like
    catalysts/events — Kalshi prices are already bounded probabilities, there
    is nothing left to saturate. Calm series still count in the denominator:
    one series at its worst moves the score by its share of the total weight,
    so forcing risk_off takes broad stress, not a single alarming market.
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
        # Only the latest fetch: once an event settles and the series rolls
        # to the next one, the old event's rows are still inside the
        # staleness window and must not be blended with the new event's.
        group = group[group["fetched_at"] == group["fetched_at"].max()]
        prob = _combined_probability(cs, group["probability"])
        rows.append({
            "series_ticker": series_ticker, "label": cs.label,
            "probability": round(prob, 4), "baseline": cs.baseline,
            "direction": cs.direction, "weight": cs.weight,
            "contribution": round(_contribution(cs, prob), 4),
            "n_markets": int(len(group)),
        })

    if not rows:
        out["note"] = "no stored snapshot matches a configured series"
        return out

    detail = (pd.DataFrame(rows)
              .sort_values("contribution", kind="stable")     # most alarming first
              .reset_index(drop=True))
    total_weight = float(detail["weight"].sum())
    net = float(detail["contribution"].sum() / total_weight) if total_weight > 0 else 0.0
    net = max(-1.0, min(0.0, net))
    top = detail.iloc[0]

    if top["contribution"] < 0:
        note = (f"{top['label']} {top['probability'] * 100:.0f}% vs normal "
                f"{top['baseline'] * 100:.0f}% (net {net:+.2f} across {len(detail)} series)")
    else:
        note = f"all {len(detail)} series at or better than normal (net {net:+.2f})"
    out.update(
        score=round(net, 4), n_series=int(len(detail)), n_markets=int(detail["n_markets"].sum()),
        top_label=str(top["label"]), top_probability=round(float(top["probability"]), 4),
        top_contribution=round(float(top["contribution"]), 4),
        note=note, detail=detail,
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

    Kalshi's series catalog changes over time, so MACRO_SERIES is
    hand-maintained, the same way SCHEDULED_EVENTS is. Run this with working
    credentials — e.g. `macro.list_series(query="fed")` — to find current
    tickers when a configured series stops returning open markets.
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
