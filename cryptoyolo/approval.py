"""Approval gate.

By default nothing executes without an explicit per-trade answer typed here: an
empty answer is a rejection, and there is no timeout that proceeds on silence.
In live mode the prompt escalates - you must type the symbol back, so muscle
memory on the `y` key cannot spend real money.

`auto_approve` (opt-in, never the default) skips the prompt entirely and accepts
everything. Those decisions are recorded as `auto-approved` rather than
`approved` so the audit trail always separates machine choices from yours, and
combining it with a live account additionally requires
CRYPTO_YOLO_ALLOW_AUTO_LIVE=1.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd

from .broker import execution_banner, is_rejected, is_validate_only
from .config import CONFIG, Config
from .engine import fmt_price, fmt_qty
from .logsetup import get_logger
from .notify import notify
from .store import Store, iso

_log = get_logger("approval")

APPROVE = {"y", "yes", "a", "approve"}
REJECT = {"n", "no", "r", "reject", ""}
# Both count as approved for execution, but stay distinct in the audit trail.
APPROVED_STATES = {"approved", "auto-approved"}


class NoStdin(RuntimeError):
    """Raised when a prompt is impossible (nbconvert, papermill, cron)."""


def _default_input(prompt: str) -> str:
    """input(), but a headless kernel raises instead of crashing the run.

    Executing this notebook with nbconvert/papermill has no stdin, and the
    right behaviour there is to reject everything and say so - never to
    fall through to executing unapproved trades.
    """
    try:
        return input(prompt)
    except Exception as exc:  # noqa: BLE001 - StdinNotImplementedError, EOFError, ...
        raise NoStdin(str(exc)) from exc


def request_approval(proposals: pd.DataFrame, cfg: Config = CONFIG,
                     input_fn: Callable[[str], str] | None = None,
                     preflight: dict[str, list[str]] | None = None,
                     auto_approve: bool | None = None) -> pd.DataFrame:
    """Prompt for each proposal. Returns the frame with a `decision` column set.

    Answers: y/yes to approve, n/no/Enter to reject, `q` to stop reviewing and
    reject everything remaining.

    `auto_approve` (default False, or CONFIG.execution.auto_approve) accepts
    every proposal without prompting. Decisions are recorded as `auto-approved`
    rather than `approved`, so the audit trail always distinguishes a machine
    decision from a human one - which matters when you later ask whether the
    engine is actually any good.

    Auto-approving into a LIVE account is unattended real-money trading and
    needs a third switch, CRYPTO_YOLO_ALLOW_AUTO_LIVE=1, on top of the two that
    already gate live mode. Without it, auto-approve refuses rather than
    quietly falling back to prompting (a cron job has no one to prompt).
    """
    ask = input_fn or _default_input
    if proposals.empty:
        return proposals

    if auto_approve is None:
        auto_approve = cfg.execution.auto_approve

    if auto_approve:
        return _auto_approve(proposals, cfg, preflight)

    banner = execution_banner(cfg)
    live = cfg.execution.live_enabled
    rule = "═" * 78

    print(f"\n{rule}\n{banner}\n{rule}")
    if live:
        print("Approving below sends REAL orders. Ctrl-C now to abort everything.\n")

    decisions: list[str] = []
    stopped = False

    for _, p in proposals.iterrows():
        if stopped:
            decisions.append("rejected")
            continue

        is_exit = p.get("kind") == "exit"
        print(f"\n{'─' * 78}")
        if is_exit:
            print(f"#{p['rank']}  ⟵ EXIT {p['symbol']}   [{p['trigger']}]   "
                  f"horizon {p['horizon']}")
            print(f"{'─' * 78}")
            print(f"  price   {fmt_price(p['entry']):>18}")
            print(f"  size    {fmt_qty(p['qty']):>18} {p['symbol']}   "
                  f"≈ ${p['notional']:,.2f}")
        else:
            move_pct = (p["target"] - p["entry"]) / p["entry"] * 100
            stop_pct = (p["stop"] - p["entry"]) / p["entry"] * 100
            rr_net = p.get("reward_risk_net", p["reward_risk"])
            fee = p.get("fee_est", 0.0)
            print(f"#{p['rank']}  {p['side']} {p['symbol']}   "
                  f"composite {p['composite']:+.3f}   horizon {p['horizon']}")
            print(f"{'─' * 78}")
            print(f"  entry   {fmt_price(p['entry']):>18}")
            print(f"  stop    {fmt_price(p['stop']):>18}   ({stop_pct:+.2f}%)")
            print(f"  target  {fmt_price(p['target']):>18}   ({move_pct:+.2f}%)   "
                  f"R:R {p['reward_risk']:.2f} gross / {rr_net:.2f} net")
            print(f"  size    {fmt_qty(p['qty']):>18} {p['symbol']}   ≈ ${p['notional']:,.2f}"
                  f"   risk ≈ ${p['risk_usd']:,.2f}   fee ≈ ${fee:,.2f}/side")
        print(f"\n  {p['rationale']}")

        warns = (preflight or {}).get(p["proposal_id"], [])
        if warns:
            print("\n  ⚠ preflight warnings:")
            for w in warns:
                print(f"      · {w}")

        try:
            if live:
                answer = ask(f"\n  LIVE ORDER — type '{p['symbol']}' to approve "
                             f"(anything else rejects, 'q' stops): ").strip()
                if answer.lower() == "q":
                    stopped = True
                    decisions.append("rejected")
                    continue
                approved = answer == p["symbol"]
            else:
                answer = ask("\n  Approve? [y/N, 'q' to stop] ").strip().lower()
                if answer == "q":
                    stopped = True
                    decisions.append("rejected")
                    continue
                approved = answer in APPROVE
        except NoStdin:
            print("\n  ⚠ No stdin available (headless execution). Rejecting all "
                  "remaining proposals — nothing will be executed.")
            stopped = True
            decisions.append("rejected")
            continue
        except KeyboardInterrupt:
            print("\n  ⚠ Interrupted — rejecting all remaining proposals.")
            stopped = True
            decisions.append("rejected")
            continue

        decisions.append("approved" if approved else "rejected")
        print(f"  → {'APPROVED' if approved else 'rejected'}")

    out = proposals.copy()
    out["decision"] = decisions
    n_ok = (out["decision"] == "approved").sum()
    print(f"\n{rule}\n{n_ok} of {len(out)} approved.\n{rule}")
    return out


def _auto_approve(proposals: pd.DataFrame, cfg: Config = CONFIG,
                  preflight: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Accept every proposal, with no prompt. Never the default."""
    rule = "═" * 78
    out = proposals.copy()
    print(f"\n{rule}\n{execution_banner(cfg)}")

    if cfg.execution.live_enabled and not cfg.execution.auto_live_enabled:
        print("AUTO-APPROVE REFUSED — live mode without CRYPTO_YOLO_ALLOW_AUTO_LIVE=1.\n"
              "Unattended real-money trading needs that switch set explicitly.\n"
              "Nothing was approved; run interactively, or set the variable.")
        print(rule)
        out["decision"] = "rejected"
        return out

    label = ("*** AUTO-APPROVING LIVE ORDERS — REAL MONEY, NO PROMPT ***"
             if cfg.execution.auto_live_enabled
             else "AUTO-APPROVE — accepting all proposals without prompting")
    print(f"{label}\n{rule}")

    for _, p in proposals.iterrows():
        kind = f"EXIT [{p['trigger']}]" if p.get("kind") == "exit" else p["side"]
        print(f"  ✓ auto-approved  #{p['rank']} {kind:<22} {p['symbol']:<6} "
              f"qty {fmt_qty(p['qty']):>18}  ≈ ${p['notional']:,.2f}")
        for w in (preflight or {}).get(p["proposal_id"], []):
            print(f"      ⚠ {w}")

    out["decision"] = "auto-approved"
    print(f"\n{rule}\n{len(out)} of {len(out)} auto-approved.\n{rule}")
    return out


def _sync_position_meta(proposal, record: dict, broker, store: Store,
                        cfg: Config = CONFIG) -> None:
    """Give a filled position the memory its exits depend on.

    A BUY records where the stop, target and clock were set at entry - without
    this, level- and time-based exits have nothing to evaluate against later. A
    SELL that flattens the position clears the row so a future re-entry starts
    a fresh clock and high-water mark rather than inheriting stale terms.
    """
    if is_rejected(record.get("status")):
        return
    symbol = record["symbol"]

    if record["side"].upper() == "BUY":
        fill = float(record.get("price") or proposal["entry"])
        store.upsert_position_meta({
            "symbol": symbol,
            "opened_at": iso(),
            "entry_price": fill,
            "stop": float(proposal["stop"]),
            "target": float(proposal["target"]),
            "horizon_days": float(cfg.exits.horizon_days),
            "high_water": fill,
            "proposal_id": proposal["proposal_id"],
            "mode": record.get("mode", cfg.execution.mode),
        })
        return

    try:
        remaining = float(broker.holdings().get(symbol, 0.0))
    except Exception:  # noqa: BLE001 - keep the row rather than lose the terms
        return
    if remaining <= 1e-9:
        store.delete_position_meta(symbol)
    elif str(proposal.get("trigger", "")) == "target hit":
        # A partial take-profit that leaves a remainder: retire the target so
        # it cannot fire again next run and halve the position repeatedly.
        store.clear_position_target(symbol)


def execute_approved(proposals: pd.DataFrame, broker, store: Store, run_id: str,
                     cfg: Config = CONFIG) -> pd.DataFrame:
    """Execute every proposal marked approved. Records all outcomes."""
    if proposals.empty:
        return pd.DataFrame()

    results: list[dict[str, Any]] = []
    for _, p in proposals.iterrows():
        store.set_decision(p["proposal_id"], p["decision"])
        if p["decision"] not in APPROVED_STATES:
            continue
        # Resting protective orders lock the base asset, so they must be
        # cancelled BEFORE a sell or the order fails on free balance.
        if p["side"].upper() == "SELL" and hasattr(broker, "cancel_protection"):
            broker.cancel_protection(p["symbol"])

        record = broker.execute(p.to_dict(), run_id)
        _sync_position_meta(p, record, broker, store, cfg)

        rejected = is_rejected(record["status"])
        validated = is_validate_only(record["status"])   # dry run: nothing sent
        is_exit = p.get("kind") == "exit"
        mark = "✗" if rejected else "✓"
        print(f"  {mark} {record['side']:<4} {record['symbol']:<6} "
              f"qty={record['qty']:<14.6f} @ {record['price']:<12,.4f} "
              f"[{record['status']} · {record['mode']}]")
        _log.info("%s %s qty=%.6f @ %.6f [%s · %s]", record["side"],
                  record["symbol"], record["qty"], record["price"],
                  record["status"], record["mode"])

        if rejected and cfg.notify.on_rejection:
            notify(f"{record['symbol']} order rejected",
                   f"{record['side']} {record['symbol']} — {record['status']} "
                   f"[{record['mode']}]", cfg, tag="rejected")
        elif validated:
            pass    # the ✓ line above already says VALIDATED; no "filled" alert
        elif not rejected and is_exit and cfg.notify.on_exit_trigger:
            notify(f"{record['symbol']} exit filled — {p.get('trigger', 'exit')}",
                   f"sold {record['qty']:g} {record['symbol']} @ "
                   f"{record['price']:,.4f} [{record['mode']}]", cfg, tag="warning")
        elif not rejected and cfg.notify.on_execution:
            notify(f"{record['symbol']} {record['side']} filled",
                   f"{record['qty']:g} @ {record['price']:,.4f} [{record['mode']}]",
                   cfg, tag="info")

        if (p["side"].upper() == "BUY"
                and not rejected
                and hasattr(broker, "place_protection")):
            try:
                broker.place_protection(p.to_dict(), record)
            except Exception as exc:  # noqa: BLE001 - entry stands, protection didn't
                _log.exception("%s: protective order failed after entry fill",
                               p["symbol"])
                print(f"    ⚠ {p['symbol']}: entry filled but protective order failed "
                      f"({exc}). Position is UNPROTECTED between runs.")
        results.append({
            "symbol": record["symbol"], "side": record["side"],
            "qty": record["qty"], "price": record["price"],
            "status": record["status"], "mode": record["mode"],
            "venue": record["venue"], "order_id": record["order_id"],
        })
    if not results:
        print("  Nothing approved — no orders sent.")
    return pd.DataFrame(results)
