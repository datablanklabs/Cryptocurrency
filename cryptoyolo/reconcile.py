"""Bring the local book back in line with what actually happened at the exchange.

The local database only learns about fills this code sent itself. Two things
happen at Binance without it:

  * a resting protective order (stop / OCO) fills while nobody is running a
    cycle, and
  * you trade a pair by hand in the Binance app.

Neither reaches `orders`, so `evaluation.realized_trades` keeps a closed
position open forever, and `position_meta` keeps the old entry terms - the next
entry on that symbol would inherit a stale `opened_at` (an instant horizon exit)
and a stale high-water mark (a trailing stop armed on a previous trade's peak).

`reconcile()` runs at the start of every live cycle and repairs that:

  1. protection  ask the exchange what became of every 'resting' protective
                 order, and mark it filled or cancelled
  2. fills       import every fill on a tracked pair that no local order
                 accounts for (OCO legs, manual trades) as its own order row
  3. positions   drop `position_meta` for positions the exchange no longer holds
  4. drift       report where the local order log and the exchange disagree

`protect_open_positions()` runs at the end of a cycle that is allowed to
execute: a live position with recorded terms but no resting protection (opened
before protection was switched on, or re-sized by a partial exit) gets one.

Paper and validate-only modes have nothing to reconcile - the local book is the
only book - so both functions are no-ops there.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .broker import BinanceError
from .config import CONFIG, Config
from .logsetup import get_logger
from .store import Store, iso

_log = get_logger("reconcile")

# Order ids of rows this module wrote. Engine-placed orders are "bnc-<hex>";
# an imported fill is keyed on the exchange's own orderId, so re-importing the
# same order (e.g. a stop that filled in two parts across two cycles) replaces
# the row instead of duplicating it.
IMPORTED_PREFIX = "bnc-x-"
DUST_USD = 1.0
_OPEN_STATES = {"NEW", "PARTIALLY_FILLED", "PENDING_CANCEL", "PENDING_NEW"}
_FILLED_STATES = ("FILLED", "PARTIALLY_FILLED")


def _is_live(broker) -> bool:
    client = getattr(broker, "client", None)
    return bool(getattr(broker, "live", False) and getattr(client, "configured", False))


def _ms_to_iso(ms: int | float) -> str:
    return iso(datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc))


# --------------------------------------------------------------------------
# 1. protective orders
# --------------------------------------------------------------------------
def _refresh_protection(store: Store, broker) -> list[dict[str, Any]]:
    """Mark resting protective orders the exchange has finished with."""
    rows = store.resting_orders()
    if rows.empty:
        return []
    rows = rows[rows["mode"] == "binance-live"]
    client = broker.client
    out: list[dict[str, Any]] = []
    for _, r in rows.iterrows():
        pair = broker.pair(r["symbol"])
        try:
            if r["kind"] == "oco" and r.get("order_list_id"):
                ol = client.get_order_list(r["order_list_id"])
                if ol.get("listOrderStatus") not in ("ALL_DONE", "REJECT"):
                    continue
                legs = [client.get_order(pair, leg["orderId"])
                        for leg in ol.get("orders", [])]
            elif r.get("exchange_ref"):
                legs = [client.get_order(pair, r["exchange_ref"])]
                if legs[0].get("status") in _OPEN_STATES:
                    continue
            else:
                continue
        except BinanceError as exc:
            _log.warning("protective order %s on %s: status unknown (%s)",
                         r.get("exchange_ref"), r["symbol"], exc)
            continue
        filled = [leg for leg in legs if float(leg.get("executedQty") or 0) > 0]
        status = "filled" if filled else "cancelled"
        store.set_protective_status(int(r["id"]), status)
        out.append({"symbol": r["symbol"], "kind": r["kind"], "status": status,
                    "legs": [f"{leg.get('type')}:{leg.get('status')}" for leg in legs]})
    return out


# --------------------------------------------------------------------------
# 2. fills the local log doesn't know about
# --------------------------------------------------------------------------
def _tracked_since(store: Store) -> dict[str, pd.Timestamp]:
    """Symbols worth checking, each with the time local tracking began.

    Fills before that are pre-dashboard history (the account's own past), not
    something this system's accounting should absorb.
    """
    since: dict[str, pd.Timestamp] = {}
    orders = store.live_orders()
    if not orders.empty:
        since.update(orders.groupby("symbol")["ts"].min().to_dict())
    meta = store.position_meta(book="binance")
    if not meta.empty:
        for _, m in meta.iterrows():
            t = pd.to_datetime(m["opened_at"], utc=True, format="ISO8601")
            since[m["symbol"]] = min(since.get(m["symbol"], t), t)
    return since


def _import_fills(store: Store, broker, cfg: Config) -> list[dict[str, Any]]:
    orders = store.live_orders()
    engine_refs: set[str] = set()
    imported_qty: dict[str, float] = {}
    if not orders.empty:
        ours = ~orders["order_id"].str.startswith(IMPORTED_PREFIX)
        engine_refs = set(orders.loc[ours, "exchange_ref"].dropna().astype(str)) - {""}
        imp = orders[~ours]
        imported_qty = dict(zip(imp["exchange_ref"].astype(str), imp["qty"].astype(float)))

    prot = store.protective_orders()
    prot = prot[prot["mode"] == "binance-live"] if not prot.empty else prot
    prot_lists = set(prot["order_list_id"].dropna().astype(str)) - {""} if not prot.empty else set()
    prot_refs = set(prot["exchange_ref"].dropna().astype(str)) - {""} if not prot.empty else set()

    out: list[dict[str, Any]] = []
    for symbol, t0 in sorted(_tracked_since(store).items()):
        pair = broker.pair(symbol)
        try:
            trades = broker.client.my_trades(pair)
        except BinanceError as exc:
            _log.warning("myTrades %s failed: %s", pair, exc)
            continue
        by_order: dict[str, list[dict]] = {}
        for t in trades or []:
            by_order.setdefault(str(t["orderId"]), []).append(t)

        for oid, fills in by_order.items():
            if oid in engine_refs:
                continue                          # an order this code sent
            first_ms = min(int(f["time"]) for f in fills)
            if pd.Timestamp(first_ms, unit="ms", tz="UTC") < t0:
                continue                          # before this system tracked the pair
            qty = sum(float(f["qty"]) for f in fills)
            if qty <= 0:
                continue
            if abs(imported_qty.get(oid, -1.0) - qty) <= 1e-12:
                continue                          # already imported, unchanged
            quote = sum(float(f.get("quoteQty") or float(f["qty"]) * float(f["price"]))
                        for f in fills)
            price = quote / qty
            side = "BUY" if fills[0].get("isBuyer") else "SELL"
            fee_usd, fee_detail = broker._fills_fee_usd(fills, price, symbol)
            list_id = str(fills[0].get("orderListId", -1))
            origin = ("protective" if (list_id in prot_lists or oid in prot_refs)
                      else "external")
            row = {
                "order_id": f"{IMPORTED_PREFIX}{oid}", "proposal_id": None,
                "run_id": "reconcile", "ts": _ms_to_iso(first_ms),
                "venue": broker.venue, "mode": "binance-live", "symbol": symbol,
                "side": side,
                "order_type": "PROTECTIVE" if origin == "protective" else "EXTERNAL",
                "qty": qty, "price": price, "status": "FILLED", "exchange_ref": oid,
                "fee_usd": round(fee_usd, 6) if fee_usd is not None else None,
                "response": json.dumps({"source": "reconcile", "origin": origin,
                                        "order_list_id": list_id,
                                        "fee_detail": fee_detail, "fills": fills},
                                       default=str),
            }
            store.save_order(row)
            out.append({"symbol": symbol, "side": side, "qty": qty, "price": price,
                        "origin": origin, "ts": row["ts"]})
    return out


# --------------------------------------------------------------------------
# 3 + 4. positions and drift
# --------------------------------------------------------------------------
def _local_net_qty(store: Store) -> dict[str, float]:
    o = store.live_orders()
    if o.empty:
        return {}
    o = o[o["status"].isin(_FILLED_STATES)]
    signed = o["qty"].astype(float).where(o["side"].str.upper() == "BUY",
                                          -o["qty"].astype(float))
    return signed.groupby(o["symbol"]).sum().to_dict()


def _value(broker, symbol: str, qty: float) -> float | None:
    if qty <= 0:
        return 0.0
    try:
        return qty * float(broker.client.price(broker.pair(symbol)))
    except Exception:  # noqa: BLE001 - unknown value, not zero
        return None


def _close_stale_meta(store: Store, broker, holdings: dict[str, float]) -> list[str]:
    # The whole Binance book, including rows a validate-only run left behind
    # before that stopped writing terms: none of them is held, so all go.
    meta = store.position_meta(book="binance")
    if meta.empty:
        return []
    closed = []
    for _, m in meta.iterrows():
        sym = m["symbol"]
        value = _value(broker, sym, float(holdings.get(sym, 0.0)))
        if value is not None and value < DUST_USD:
            store.delete_position_meta(sym, "binance")
            store.mark_protective_cancelled(sym)
            closed.append(sym)
    return closed


def _drift(store: Store, holdings: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for sym, local in sorted(_local_net_qty(store).items()):
        venue = float(holdings.get(sym, 0.0))
        gap = venue - max(local, 0.0)
        if abs(gap) > max(1e-8, 0.01 * max(abs(venue), abs(local))):
            out.append({"symbol": sym, "local": round(local, 8),
                        "exchange": round(venue, 8), "gap": round(gap, 8)})
    return out


def reconcile(store: Store, broker, cfg: Config = CONFIG,
              verbose: bool = True) -> dict[str, Any]:
    """Sync local records with the exchange. No-op outside live mode.

    Never raises: a reconciliation failure is logged and reported, and the
    cycle carries on with the book as it was.
    """
    if not _is_live(broker):
        return {"skipped": "not live"}
    result: dict[str, Any] = {"protection": [], "imported": [], "closed": [], "drift": []}
    try:
        result["protection"] = _refresh_protection(store, broker)
        result["imported"] = _import_fills(store, broker, cfg)
        # Balances straight from the client: broker.holdings() maps an API
        # failure to {}, which here would read as "everything was sold" and
        # wipe every position's entry terms.
        holdings = broker.client.balances()
        result["closed"] = _close_stale_meta(store, broker, holdings)
        result["drift"] = _drift(store, holdings)
    except Exception as exc:  # noqa: BLE001 - never fail a cycle on bookkeeping
        _log.exception("reconciliation failed")
        result["error"] = str(exc)

    if verbose:
        print("\n[reconcile] syncing local records with the exchange...")
        for p in result["protection"]:
            print(f"  · {p['symbol']}: protective {p['kind']} {p['status']} "
                  f"({', '.join(p['legs'])})")
        for f in result["imported"]:
            print(f"  + imported {f['origin']} fill: {f['side']} {f['qty']:g} "
                  f"{f['symbol']} @ {f['price']:,.6g} ({f['ts'][:16]})")
        for sym in result["closed"]:
            print(f"  · {sym}: no longer held at the exchange — entry terms cleared")
        for d in result["drift"]:
            print(f"  ⚠ {d['symbol']}: local log nets {d['local']:g}, exchange holds "
                  f"{d['exchange']:g} (gap {d['gap']:+g}) — traded outside tracked "
                  f"history, or a fill this log never saw")
        if result.get("error"):
            print(f"  ! reconciliation failed: {result['error']}")
        if not any(result[k] for k in ("protection", "imported", "closed", "drift")):
            print("  in sync")
    for f in result["imported"]:
        _log.info("imported %s fill %s %s %.8g @ %.8g", f["origin"], f["side"],
                  f["symbol"], f["qty"], f["price"])
    return result


# --------------------------------------------------------------------------
# protection for positions that have none
# --------------------------------------------------------------------------
def protect_open_positions(store: Store, broker, cfg: Config = CONFIG,
                           verbose: bool = True) -> list[dict[str, Any]]:
    """Rest a stop (and target, as one OCO) on every live position lacking one.

    Uses the terms recorded at entry. Skips a position already through its
    stop or target - the exits pass owns that - and one without a recorded
    stop, rather than inventing a level.
    """
    if not _is_live(broker) or not cfg.execution.place_stop_orders:
        return []
    meta = store.position_meta(book="binance")
    if meta.empty:
        return []
    resting = store.resting_orders()
    covered = set(resting["symbol"]) if not resting.empty else set()
    try:
        free = broker.client.free_balances()
    except Exception as exc:  # noqa: BLE001 - try again next cycle
        _log.warning("protect_open_positions: balances unreadable (%s)", exc)
        return []

    placed: list[dict[str, Any]] = []
    header = False
    for _, m in meta.iterrows():
        sym = m["symbol"]
        if sym in covered:
            continue
        qty = float(free.get(sym, 0.0))
        stop = float(m["stop"]) if pd.notna(m.get("stop")) and m["stop"] else None
        target = float(m["target"]) if pd.notna(m.get("target")) and m["target"] else None
        try:
            last = float(broker.client.price(broker.pair(sym)))
        except Exception:  # noqa: BLE001
            continue
        why = None
        if qty * last < cfg.risk.min_notional_usd:
            why = f"free balance ${qty * last:,.2f} is under the minimum order"
        elif stop is None:
            why = "no stop recorded at entry"
        elif last <= stop:
            why = f"already through the stop ({last:,.6g} ≤ {stop:,.6g}) — the exits pass handles it"
        elif target is not None and last >= target:
            why = f"already past the target ({last:,.6g} ≥ {target:,.6g}) — the exits pass handles it"
        if verbose and not header:
            print("\n[protection] live positions without a resting stop:")
            header = True
        if why:
            if verbose:
                print(f"  · {sym}: not protected — {why}")
            continue
        row = broker.place_protection({"symbol": sym, "stop": stop, "target": target,
                                       "qty": qty}, {"qty": qty})
        if row:
            placed.append(row)
    return placed
