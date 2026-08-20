"""Exit management: when to close an existing position.

The buy-side scoreboard is the wrong instrument for this. It ranks assets by how
attractive they look *now*, which means a position quietly deteriorating below
its stop only surfaces if it happens to out-rank every buy candidate that day.
It usually doesn't, and the position sits there.

So exits are evaluated separately, before the ranking, against each position's
own recorded entry terms. Five independent triggers, each switchable in
ExitConfig:

  stop        price traded through the stop set at entry
  target      price reached the take-profit set at entry
  trailing    gave back more than trail_pct from the high-water mark
  horizon     held past cfg.exits.horizon_days, the thesis it was approved on
  reversal    the composite score that opened it has inverted

Precedence is the order in TRIGGERS below. Stop and target come first because
they are level events that have already happened - the position is at a price
you previously said you would act on. Horizon and reversal are judgement calls
and yield to them.

Positions bought outside this dashboard have no recorded stop, target or open
date. Rather than invent them, only the metadata-free triggers (score reversal)
apply, and the exit reason says so.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import CONFIG, Config
from .store import Store, iso, utcnow

# (key, config flag, human label) in precedence order.
TRIGGERS = [
    ("stop",     "stop_loss",      "stop hit"),
    ("target",   "take_profit",    "target hit"),
    ("trailing", "trailing_stop",  "trailing stop"),
    ("horizon",  "horizon_expiry", "horizon expired"),
    ("reversal", "score_reversal", "score reversed"),
]
_PRECEDENCE = {k: i for i, (k, _, _) in enumerate(TRIGGERS)}
_LABELS = {k: lbl for k, _, lbl in TRIGGERS}


def _age_days(opened_at: str | None) -> float | None:
    if not opened_at:
        return None
    try:
        opened = datetime.fromisoformat(opened_at)
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        return (utcnow() - opened).total_seconds() / 86400.0
    except Exception:  # noqa: BLE001 - unparseable timestamp = unknown age
        return None


def evaluate(store: Store, cfg: Config = CONFIG,
             holdings: dict[str, float] | None = None,
             scores: pd.DataFrame | None = None,
             prices: dict[str, float] | None = None) -> pd.DataFrame:
    """Check every open position against the enabled triggers.

    Returns one row per position that should be exited, with the trigger that
    fired, a human explanation, and how much to sell. Also refreshes each
    position's high-water mark as a side effect, which is what makes the
    trailing stop work across runs.
    """
    ex = cfg.exits
    # The quote asset is cash, not a position. On Binance `holdings()` returns
    # every balance including USDT, and iterating it here can emit a "sell your
    # cash" signal in edge cases (a stablecoin depeg trips the stop check).
    quote = cfg.execution.quote_asset
    holdings = {s: q for s, q in (holdings or {}).items()
                if q > 0 and s != quote}
    if not ex.enabled or not holdings:
        return pd.DataFrame()

    meta_df = store.position_meta()
    meta = {r["symbol"]: dict(r) for _, r in meta_df.iterrows()} if not meta_df.empty else {}

    score_map: dict[str, float] = {}
    if scores is not None and not scores.empty:
        score_map = dict(zip(scores["symbol"], scores["composite"]))

    if prices is None:
        from . import prices as prices_mod
        try:
            prices = prices_mod.latest_prices(list(holdings), cfg)
        except Exception:  # noqa: BLE001 - without prices only reversal can fire
            prices = {}

    rows: list[dict[str, Any]] = []

    for symbol, qty in sorted(holdings.items()):
        last = prices.get(symbol)
        m = meta.get(symbol, {})
        has_meta = bool(m)

        # Refresh the high-water mark first so the trailing stop sees this run's
        # price. Without persisting this, a trailing stop can only ever measure
        # from the entry, which is not a trailing stop.
        high_water = None
        if last and has_meta:
            high_water = store.bump_high_water(symbol, last)

        entry = float(m.get("entry_price") or 0) or None
        stop = float(m.get("stop") or 0) or None
        target = float(m.get("target") or 0) or None
        horizon_days = float(m.get("horizon_days") or 0) or ex.horizon_days
        age = _age_days(m.get("opened_at"))
        composite = score_map.get(symbol)
        fired: list[dict[str, Any]] = []

        if last is not None:
            if ex.stop_loss and stop and last <= stop:
                fired.append({
                    "trigger": "stop", "fraction": 100.0,
                    "detail": f"last {last:,.6g} at or below stop {stop:,.6g}"
                              f"{f' ({(last/entry-1)*100:+.1f}% from entry)' if entry else ''}",
                })
            if ex.take_profit and target and last >= target:
                fired.append({
                    "trigger": "target", "fraction": ex.take_profit_fraction,
                    "detail": f"last {last:,.6g} at or above target {target:,.6g}"
                              f"{f' ({(last/entry-1)*100:+.1f}% from entry)' if entry else ''}",
                })
            if ex.trailing_stop and high_water and entry:
                gain_from_entry = (high_water / entry - 1) * 100
                giveback = (1 - last / high_water) * 100
                # Only armed once the trade actually ran - otherwise this is
                # just a tighter stop firing on entry noise.
                if gain_from_entry >= ex.trail_activate_pct and giveback >= ex.trail_pct:
                    fired.append({
                        "trigger": "trailing", "fraction": 100.0,
                        "detail": f"gave back {giveback:.1f}% from high "
                                  f"{high_water:,.6g} (limit {ex.trail_pct:.1f}%; "
                                  f"peak was {gain_from_entry:+.1f}% from entry)",
                    })

        if ex.horizon_expiry and age is not None and age >= horizon_days:
            fired.append({
                "trigger": "horizon", "fraction": 100.0,
                "detail": f"held {age:.1f}d, past the {horizon_days:.0f}d horizon "
                          f"the trade was approved on",
            })

        if ex.score_reversal and composite is not None and composite <= ex.score_reversal_threshold:
            fired.append({
                "trigger": "reversal", "fraction": 100.0,
                "detail": f"composite {composite:+.3f} at or below "
                          f"{ex.score_reversal_threshold:+.3f}",
            })

        if not fired:
            continue
        if not has_meta and not ex.allow_exits_without_metadata:
            continue

        fired.sort(key=lambda f: _PRECEDENCE[f["trigger"]])
        primary = fired[0]
        label = _LABELS[primary["trigger"]]
        also = [_LABELS[f["trigger"]] for f in fired[1:]]

        sell_qty = qty * min(100.0, max(0.0, primary["fraction"])) / 100.0
        rows.append({
            "symbol": symbol,
            "trigger": primary["trigger"],
            "trigger_label": label,
            "reason": primary["detail"],
            "also_fired": ", ".join(also),
            "qty_held": qty,
            "qty": sell_qty,
            "fraction_pct": primary["fraction"],
            "last": last,
            "entry": entry,
            "stop": stop,
            "target": target,
            "high_water": high_water,
            "age_days": age,
            "composite": composite,
            "has_metadata": has_meta,
            "pnl_pct": ((last / entry - 1) * 100) if (last and entry) else None,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["precedence"] = df["trigger"].map(_PRECEDENCE)
    df = df.sort_values(["precedence", "symbol"]).reset_index(drop=True)
    return df.drop(columns=["precedence"])


def describe(signal: pd.Series, cfg: Config = CONFIG) -> str:
    """Human rationale for one exit, used in the approval prompt."""
    bits = [f"EXIT — {signal['trigger_label']}: {signal['reason']}."]
    if signal.get("also_fired"):
        bits.append(f"Also firing: {signal['also_fired']}.")
    if signal.get("age_days") is not None:
        bits.append(f"Held {signal['age_days']:.1f} days.")
    if signal.get("pnl_pct") is not None:
        bits.append(f"Unrealised {signal['pnl_pct']:+.1f}% vs entry.")
    if signal.get("composite") is not None:
        bits.append(f"Current composite {signal['composite']:+.3f}.")
    if not signal.get("has_metadata"):
        bits.append("No entry terms recorded for this position (opened outside "
                    "this dashboard), so only score-based exits could apply.")
    if signal.get("fraction_pct", 100.0) < 100.0:
        bits.append(f"Partial exit: selling {signal['fraction_pct']:.0f}% of the position.")
    return " ".join(bits)


def active_triggers(cfg: Config = CONFIG) -> pd.DataFrame:
    """Which triggers are switched on, for display in the notebook."""
    ex = cfg.exits
    settings = {
        "stop_loss": "at the stop recorded at entry",
        "take_profit": f"at the target ({ex.take_profit_fraction:.0f}% of position)",
        "trailing_stop": f"give back {ex.trail_pct:.1f}% from high, "
                         f"armed above {ex.trail_activate_pct:.1f}% profit",
        "horizon_expiry": f"after {ex.horizon_days:.0f} days",
        "score_reversal": f"composite <= {ex.score_reversal_threshold:+.2f}",
    }
    return pd.DataFrame([
        {"trigger": label, "enabled": "on" if getattr(ex, flag) else "off",
         "fires": settings[flag]}
        for _, flag, label in TRIGGERS
    ])
