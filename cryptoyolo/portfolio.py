"""Feature 0: account balance and holdings.

One question answered honestly: what do you actually own right now, and what is
it worth? Everything downstream (position sizing, the cash cap, whether a SELL
is even possible) depends on this, so it is worth seeing before the charts.

Cost basis is the awkward part. In paper mode the book is ours, so average cost
and unrealised P&L are exact. On a live Binance account we only know the cost of
fills *this system* made - anything you bought elsewhere has no basis we can
honestly report, and it is marked unknown rather than guessed at.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import prices as prices_mod
from .broker import get_broker
from .config import CONFIG, Config
from .store import Store

STABLES = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USD"}
DUST_USD = 1.0          # ignore balances worth less than this


def _cost_basis_from_orders(store: Store) -> dict[str, dict[str, float]]:
    """Average cost per symbol, derived from this system's own filled orders.

    Only covers trades placed through this dashboard. Positions opened
    elsewhere legitimately have no basis here.
    """
    with store.conn() as con:
        df = pd.read_sql_query(
            "SELECT symbol, side, qty, price FROM orders "
            "WHERE status IN ('FILLED','SENT') AND qty > 0 AND price > 0 "
            "ORDER BY ts ASC", con,
        )
    basis: dict[str, dict[str, float]] = {}
    for _, r in df.iterrows():
        b = basis.setdefault(r["symbol"], {"qty": 0.0, "avg": 0.0})
        if r["side"].upper() == "BUY":
            new_qty = b["qty"] + r["qty"]
            b["avg"] = ((b["qty"] * b["avg"]) + (r["qty"] * r["price"])) / new_qty if new_qty > 0 else 0.0
            b["qty"] = new_qty
        else:
            b["qty"] = max(0.0, b["qty"] - r["qty"])
            if b["qty"] <= 1e-12:
                b["avg"] = 0.0
    return basis


def snapshot(store: Store, cfg: Config = CONFIG) -> dict[str, Any]:
    """Current cash, holdings, and totals for whichever broker is configured."""
    broker = get_broker(store, cfg)
    mode = getattr(broker, "mode", "paper")
    venue = getattr(broker, "venue", "paper")
    quote = cfg.execution.quote_asset
    warnings: list[str] = []

    cash = broker.available_cash()
    if cash is None:
        warnings.append(
            "Balance unavailable — no Binance credentials, or the account call failed. "
            "Holdings and the cash cap can't be applied this run."
        )

    raw = broker.holdings() or {}
    # In paper mode the book only ever holds traded assets. On Binance the
    # account also carries the quote asset itself, which is cash, not a position.
    positions = {s: q for s, q in raw.items() if s != quote and abs(q) > 0}

    if mode == "paper":
        book = store.paper_positions()
        basis = {r["symbol"]: {"qty": r["qty"], "avg": r["avg_price"]}
                 for _, r in book.iterrows()} if not book.empty else {}
        basis_source = "paper book"
    else:
        basis = _cost_basis_from_orders(store)
        basis_source = "local order log"

    live: dict[str, float] = {}
    if positions:
        try:
            live = prices_mod.latest_prices(list(positions), cfg)
        except Exception as exc:  # noqa: BLE001 - show quantities even if pricing fails
            warnings.append(f"Could not price holdings: {exc}")

    meta_df = store.position_meta()
    pmeta = {r["symbol"]: dict(r) for _, r in meta_df.iterrows()} if not meta_df.empty else {}

    rows: list[dict[str, Any]] = []
    for sym, qty in sorted(positions.items()):
        px = live.get(sym)
        if px is None and sym in STABLES:
            px = 1.0
        value = qty * px if px else None
        avg = basis.get(sym, {}).get("avg") or None
        # Only claim a cost basis if it covers the size actually held.
        covered = basis.get(sym, {}).get("qty", 0.0) >= qty * 0.999 if avg else False
        pnl = (px - avg) * qty if (px and avg and covered) else None
        # Exit terms recorded at entry, so you can see how close a position is
        # to being closed without waiting for the exit pass to say so.
        from .exits import _age_days
        pm = pmeta.get(sym, {})
        stop, target = pm.get("stop"), pm.get("target")
        rows.append({
            "symbol": sym, "qty": qty,
            "avg_cost": avg if covered else None,
            "last": px, "value": value,
            "pnl": pnl,
            "pnl_pct": ((px / avg - 1) * 100) if (px and avg and covered) else None,
            "age_days": _age_days(pm.get("opened_at")),
            "stop": stop or None,
            "target": target or None,
            "to_stop_pct": ((px / stop - 1) * 100) if (px and stop) else None,
            "to_target_pct": ((target / px - 1) * 100) if (px and target) else None,
            "high_water": pm.get("high_water") or None,
        })

    pos_df = pd.DataFrame(rows)
    positions_value = float(pos_df["value"].fillna(0).sum()) if not pos_df.empty else 0.0
    if not pos_df.empty and positions_value > 0:
        pos_df["weight_pct"] = pos_df["value"].fillna(0) / positions_value * 100
        pos_df = pos_df.sort_values("value", ascending=False, na_position="last").reset_index(drop=True)

    unpriced = int(pos_df["last"].isna().sum()) if not pos_df.empty else 0
    if unpriced:
        warnings.append(f"{unpriced} holding(s) could not be priced; excluded from totals.")
    if not pos_df.empty and pos_df["age_days"].isna().any():
        n = int(pos_df["age_days"].isna().sum())
        warnings.append(
            f"{n} position(s) have no recorded entry terms (opened outside this "
            f"dashboard). Level- and time-based exits can't evaluate them; only "
            f"score-reversal applies."
        )
    if not pos_df.empty and pos_df["avg_cost"].isna().any() and mode != "paper":
        warnings.append(
            f"Cost basis shown only for positions opened through this dashboard "
            f"({basis_source}); others show '—' rather than a guess."
        )

    total_equity = (cash or 0.0) + positions_value
    configured = cfg.risk.account_equity_usd
    if cash is not None and configured > 0 and abs(total_equity - configured) / configured > 0.20:
        warnings.append(
            f"Risk basis (CONFIG.risk.account_equity_usd = ${configured:,.2f}) is more "
            f"than 20% away from real equity (${total_equity:,.2f}). Position sizes are "
            f"computed from the risk basis — consider syncing it."
        )

    return {
        "mode": mode, "venue": venue, "quote": quote,
        "cash": cash, "positions": pos_df,
        "positions_value": positions_value,
        "total_equity": total_equity if cash is not None else None,
        "configured_equity": configured,
        "warnings": warnings,
        "basis_source": basis_source,
    }


def format_summary(snap: dict[str, Any]) -> str:
    """Plain-text account header."""
    from .engine import fmt_price, fmt_qty

    q = snap["quote"]
    lines = [
        "═" * 78,
        f"ACCOUNT — {snap['mode']} · {snap['venue']}",
        "═" * 78,
    ]
    cash = snap["cash"]
    lines.append(f"  cash ({q}){'':<8} " +
                 (f"${cash:>16,.2f}" if cash is not None else f"{'unavailable':>17}"))
    lines.append(f"  positions{'':<11} ${snap['positions_value']:>16,.2f}")
    if snap["total_equity"] is not None:
        lines.append(f"  {'─' * 40}")
        lines.append(f"  total equity{'':<8} ${snap['total_equity']:>16,.2f}")
    lines.append(f"  risk basis{'':<10} ${snap['configured_equity']:>16,.2f}"
                 f"   (used for position sizing)")

    df = snap["positions"]
    if df.empty:
        lines.append("\n  No open positions.")
    else:
        lines.append(f"\n  {'symbol':<8}{'qty':>18}{'avg cost':>15}{'last':>15}"
                     f"{'value':>14}{'unrealised P&L':>22}{'wt%':>7}"
                     f"{'age':>7}{'→stop':>9}{'→target':>10}")
        lines.append("  " + "─" * 125)
        for _, r in df.iterrows():
            avg = fmt_price(r["avg_cost"]) if pd.notna(r["avg_cost"]) else "—"
            last = fmt_price(r["last"]) if pd.notna(r["last"]) else "—"
            val = f"${r['value']:,.2f}" if pd.notna(r["value"]) else "—"
            if pd.notna(r["pnl"]):
                pnl = f"${r['pnl']:+,.2f} ({r['pnl_pct']:+.1f}%)"
            else:
                pnl = "—"
            wt = f"{r['weight_pct']:.1f}" if "weight_pct" in df and pd.notna(r.get("weight_pct")) else "—"
            age = f"{r['age_days']:.1f}d" if pd.notna(r.get("age_days")) else "—"
            to_stop = f"{r['to_stop_pct']:+.1f}%" if pd.notna(r.get("to_stop_pct")) else "—"
            to_tgt = f"{r['to_target_pct']:+.1f}%" if pd.notna(r.get("to_target_pct")) else "—"
            lines.append(f"  {r['symbol']:<8}{fmt_qty(r['qty']):>18}{avg:>15}{last:>15}"
                         f"{val:>14}{pnl:>22}{wt:>7}{age:>7}{to_stop:>9}{to_tgt:>10}")

    for w in snap["warnings"]:
        lines.append(f"\n  ⚠ {w}")
    lines.append("═" * 78)
    return "\n".join(lines)
