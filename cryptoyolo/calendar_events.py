"""Feature 5: scheduled, dated catalysts.

The `catalyst` family scores news that has already been published — and, as the
evaluation module tends to confirm, already moved the price. This family scores
the opposite: events with a KNOWN FUTURE DATE. A cliff unlock on the 14th is
knowable on the 1st; an ETF decision deadline, a mainnet launch and an
exchange-listing effective date all are too. That is the one kind of edge that
is genuinely forward-looking rather than a reaction.

Each event contributes on a tent function of how close its date is:

      weight
       1.0 |        ┌─────────┐
           |       /           \\
       0.0 |______/             \\________
           +----|----|----|----|----|----  days to event
            +lookahead  +peak  0  -lookback

ramping up as the date approaches, flat through the `peak_window_days` before
it, then decaying for a few days after (the market often keeps digesting a big
unlock). The contribution is `direction * magnitude_scaled * weight`, and a
symbol's events combine as a signed, saturating sum — many small ones can't
outweigh one large one of the opposite sign.

The schedule is hand-maintained in `config.SCHEDULED_EVENTS`. Keep it current:
past events drop out automatically `lookback_days` after their date, but a
schedule you stopped updating just goes quiet, it doesn't warn you.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from .config import CONFIG, Config, SCHEDULED_EVENTS, ScheduledEvent
from .store import Store, utcnow

# Fallback direction when an event's `direction` is left at 0.
_KIND_DIRECTION = {
    "token_unlock": -1, "unlock": -1, "emission": -1,
    "mainnet": +1, "upgrade": +1, "hard_fork": +1, "listing": +1,
    "etf_decision": +1, "airdrop": +1, "halving": +1,
}


def _parse_date(raw: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001 - unparseable date = skip the event
        return None


def _fetch_unlocks(cfg: Config, verbose: bool) -> list[ScheduledEvent]:
    """Best-effort pull from a public unlock feed. Never fatal."""
    ec = cfg.events
    if not ec.fetch_unlocks or not ec.unlocks_url:
        return []
    try:
        r = requests.get(ec.unlocks_url, timeout=cfg.http_timeout,
                         headers={"User-Agent": cfg.user_agent})
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  events: unlock feed unavailable ({exc})")
        return []
    known = {a.symbol for a in cfg.universe}
    out: list[ScheduledEvent] = []
    rows = data if isinstance(data, list) else data.get("data", data.get("results", []))
    for row in rows or []:
        try:
            sym = str(row.get("symbol") or row.get("ticker") or "").upper()
            if sym not in known:
                continue
            date = row.get("date") or row.get("unlock_date") or row.get("nextUnlock")
            if not date:
                continue
            pct = float(row.get("pct_of_circulating")
                        or row.get("percent") or row.get("value") or 0.0)
            out.append(ScheduledEvent(sym, str(date)[:10], "token_unlock", pct, -1,
                                      str(row.get("note") or "from unlock feed")))
        except Exception:  # noqa: BLE001 - one malformed row shouldn't sink the feed
            continue
    if verbose:
        print(f"  events: {len(out)} unlock(s) from feed")
    return out


def upcoming(store: Store | None = None, cfg: Config = CONFIG,
             verbose: bool = False) -> pd.DataFrame:
    """All in-window events with their computed proximity weight, newest date first."""
    ec = cfg.events
    now = utcnow()
    events = list(SCHEDULED_EVENTS) + _fetch_unlocks(cfg, verbose)
    rows: list[dict[str, Any]] = []
    for ev in events:
        dt = _parse_date(ev.date)
        if dt is None:
            continue
        days_to = (dt - now).total_seconds() / 86400.0
        if days_to > ec.lookahead_days or days_to < -ec.lookback_days:
            continue

        if days_to > ec.peak_window_days:
            denom = max(ec.lookahead_days - ec.peak_window_days, 1e-9)
            weight = max(0.0, min(1.0, (ec.lookahead_days - days_to) / denom))
        elif days_to >= 0:
            weight = 1.0
        else:
            weight = max(0.0, min(1.0, 1.0 + days_to / max(ec.lookback_days, 1e-9)))

        if ev.kind in ("token_unlock", "unlock", "emission"):
            mag_scaled = min(1.0, ev.magnitude / max(ec.unlock_pct_full_weight, 1e-9))
        else:
            mag_scaled = max(0.0, min(1.0, ev.magnitude))
        direction = ev.direction or _KIND_DIRECTION.get(ev.kind, 0)

        rows.append({
            "symbol": ev.symbol, "kind": ev.kind, "date": dt.date().isoformat(),
            "days_to_event": round(days_to, 1), "magnitude": round(ev.magnitude, 3),
            "direction": direction, "weight": round(weight, 3),
            "contribution": round(direction * mag_scaled * weight, 4),
            "note": ev.note,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("days_to_event").reset_index(drop=True) if not df.empty else df


def score_symbols(store: Store | None = None, cfg: Config = CONFIG,
                  verbose: bool = False) -> pd.DataFrame:
    """Per-symbol events score in [-1, 1], plus the nearest event's details."""
    ev_df = upcoming(store, cfg, verbose)
    rows: list[dict[str, Any]] = []
    for symbol in cfg.symbols:
        sub = ev_df[ev_df["symbol"] == symbol] if not ev_df.empty else pd.DataFrame()
        if sub.empty:
            rows.append({"symbol": symbol, "events": 0.0, "n_events": 0,
                         "next_event": "", "days_to_event": None, "magnitude": 0.0})
            continue
        net = math.tanh(float(sub["contribution"].sum()) / 1.0)
        # "next" = the nearest event still ahead of us; only fall back to a
        # just-passed one if there is nothing upcoming in the window.
        ahead = sub[sub["days_to_event"] >= 0]
        nearest = (ahead.iloc[0] if not ahead.empty
                   else sub.sort_values("days_to_event", ascending=False).iloc[0])
        rows.append({
            "symbol": symbol,
            "events": round(max(-1.0, min(1.0, net)), 4),
            "n_events": int(len(sub)),
            "next_event": f"{nearest['kind']} {nearest['date']}",
            "days_to_event": float(nearest["days_to_event"]),
            "magnitude": float(nearest["magnitude"]),
        })
    return pd.DataFrame(rows).sort_values("events", key=abs, ascending=False).reset_index(drop=True)
