"""Loss halt: stop opening new positions once the book has lost too much.

`notify.drawdown_alert_pct` tells you the account is down; nothing stopped the
next cycle from buying anyway. This is the circuit breaker. It reads the same
per-cycle equity snapshots the evaluation uses, plus this cycle's live equity,
and trips when either

  * equity sits `risk.halt_drawdown_pct` below its peak, or
  * equity fell `risk.halt_window_loss_pct` over the last `risk.halt_window_days`.

A tripped halt blocks new BUYs in `engine.propose`. Exits, stops and the
protective orders keep working - a halt stops adding risk, it never traps you
in a position.

It does not reset itself: a book that stops buying can only recover through
the positions it already holds, so "wait until the drawdown heals" could mean
never. Resuming is a decision - set `risk.halt_reset_after` to a timestamp and
only snapshots after it count. The same applies after a withdrawal, which the
equity curve cannot tell apart from a loss.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

from .config import CONFIG, Config
from .store import Store, utcnow


def assess(store: Store, cfg: Config = CONFIG, mode: str = "paper",
           equity_now: float | None = None) -> dict[str, Any]:
    """Verdict for this cycle. Never raises; unknown history means not tripped."""
    rc = cfg.risk
    out: dict[str, Any] = {"enabled": rc.halt_enabled, "tripped": False, "reasons": [],
                           "drawdown_pct": None, "window_loss_pct": None,
                           "peak": None, "equity": equity_now, "note": ""}
    if not rc.halt_enabled:
        out["note"] = "loss halt disabled (CONFIG.risk.halt_enabled = False)"
        return out

    ec = store.equity_curve(mode).dropna(subset=["equity"])
    curve = (ec.sort_values("ts").set_index("ts")["equity"].astype(float)
             if not ec.empty else pd.Series(dtype=float))
    now = pd.Timestamp(utcnow())
    if equity_now is not None:
        curve = pd.concat([curve, pd.Series([float(equity_now)], index=[now])])
    if rc.halt_reset_after:
        try:
            reset = pd.Timestamp(rc.halt_reset_after)
            reset = reset.tz_localize("UTC") if reset.tzinfo is None else reset
            curve = curve[curve.index >= reset]
        except (ValueError, TypeError):
            out["note"] = f"halt_reset_after {rc.halt_reset_after!r} unparseable; ignored. "
    if len(curve) < 2 or curve.max() <= 0:
        out["note"] += "not enough equity history to judge"
        return out

    last = float(curve.iloc[-1])
    peak = float(curve.max())
    out["peak"] = round(peak, 2)
    if peak > 0:
        out["drawdown_pct"] = round((last / peak - 1.0) * 100, 2)

    # The window's baseline is the last snapshot at or before its start; with a
    # shorter history, the oldest one we have (so a young book is still judged).
    start = curve.index[-1] - timedelta(days=rc.halt_window_days)
    before = curve[curve.index <= start]
    base = float(before.iloc[-1]) if not before.empty else float(curve.iloc[0])
    if base > 0:
        out["window_loss_pct"] = round((last / base - 1.0) * 100, 2)

    dd, wl = out["drawdown_pct"], out["window_loss_pct"]
    if rc.halt_drawdown_pct > 0 and dd is not None and dd <= -rc.halt_drawdown_pct:
        out["reasons"].append(f"equity {dd:.1f}% below its peak ${peak:,.2f} "
                              f"(limit {rc.halt_drawdown_pct:.0f}%)")
    if rc.halt_window_loss_pct > 0 and wl is not None and wl <= -rc.halt_window_loss_pct:
        out["reasons"].append(f"equity {wl:.1f}% over the last "
                              f"{rc.halt_window_days:g}d (limit {rc.halt_window_loss_pct:.0f}%)")
    out["tripped"] = bool(out["reasons"])
    out["note"] += ("; ".join(out["reasons"]) if out["tripped"] else
                    f"clear — drawdown {dd if dd is not None else 0:.1f}%, "
                    f"{rc.halt_window_days:g}d change {wl if wl is not None else 0:+.1f}%")
    return out


def describe(verdict: dict[str, Any]) -> str:
    if not verdict.get("enabled", True):
        return "loss halt: OFF"
    if verdict.get("tripped"):
        return (f"🛑 LOSS HALT — new BUYs blocked: {verdict['note']}. Exits still run. "
                f"Resume by setting CONFIG.risk.halt_reset_after.")
    return f"loss halt: {verdict.get('note', '')}"
