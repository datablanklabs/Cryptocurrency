"""reconcile: exchange-side fills reach the local book; protection is kept on.

A real BinanceBroker with a fake client, so the broker's own protection and
fee code runs but nothing touches the network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cryptoyolo import evaluation, reconcile
from cryptoyolo.broker import BinanceBroker, BinanceError, PaperBroker
from cryptoyolo.store import iso

T0 = pd.Timestamp("2026-09-01T12:00:00Z")


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


class FakeClient:
    configured = True

    def __init__(self):
        self.trades: dict[str, list[dict]] = {}
        self.orders: dict[int, dict] = {}
        self.order_lists: dict[int, dict] = {}
        self.bal: dict[str, float] = {"USDT": 1_000.0}
        self.free: dict[str, float] | None = None
        self.px: dict[str, float] = {}
        self.placed: list[tuple] = []
        self.cancelled: list[str] = []
        self.fail_balances = False

    # reads
    def my_trades(self, pair, limit=1000):
        return list(self.trades.get(pair, []))

    def get_order(self, pair, order_id):
        return self.orders[int(order_id)]

    def get_order_list(self, order_list_id):
        return self.order_lists[int(order_list_id)]

    def balances(self):
        if self.fail_balances:
            raise BinanceError(500, -1000, "boom")
        return dict(self.bal)

    def free_balances(self):
        return dict(self.free if self.free is not None else self.bal)

    def price(self, pair):
        return self.px[pair]

    # writes (protection)
    def normalize_order(self, pair, qty, price=None):
        return (f"{qty:.8f}".rstrip("0").rstrip("."),
                None if price is None else f"{price:.8f}".rstrip("0").rstrip("."), [])

    def place_oco(self, pair, qty, tp, stop, stop_limit, **kw):
        self.placed.append(("oco", pair, float(qty), float(tp), float(stop)))
        return {"orderListId": 900 + len(self.placed)}

    def place_stop_limit(self, pair, qty, stop, stop_limit, **kw):
        self.placed.append(("stop", pair, float(qty), None, float(stop)))
        return {"orderId": 800 + len(self.placed)}

    def cancel_open_orders(self, pair):
        self.cancelled.append(pair)


def _trade(order_id, ts, qty, price, buyer, order_list_id=-1, fee=None):
    return {"symbol": "SOLUSDT", "id": order_id * 10, "orderId": order_id,
            "orderListId": order_list_id, "price": str(price), "qty": str(qty),
            "quoteQty": str(qty * price),
            "commission": str(fee if fee is not None else qty * price * 0.001),
            "commissionAsset": "USDT", "time": _ms(ts), "isBuyer": buyer}


@pytest.fixture
def live(store, cfg):
    cfg.execution.mode = "binance"
    b = BinanceBroker(store, cfg)
    b.client = FakeClient()
    b.live = True
    return b


def _open_sol_position(store, qty=2.0, entry=100.0, oco_list_id="7"):
    """What approval.execute_approved leaves behind after a live SOL entry."""
    store.save_order({
        "order_id": "bnc-aaa", "proposal_id": "p1", "run_id": "r1", "ts": iso(T0),
        "venue": "binance-us", "mode": "binance-live", "symbol": "SOL", "side": "BUY",
        "order_type": "LIMIT", "qty": qty, "price": entry, "status": "FILLED",
        "exchange_ref": "100", "fee_usd": 0.2, "response": "{}"})
    store.upsert_position_meta({
        "symbol": "SOL", "opened_at": iso(T0), "entry_price": entry, "stop": 90.0,
        "target": 125.0, "horizon_days": 30.0, "high_water": entry,
        "proposal_id": "p1", "mode": "binance-live"})
    if oco_list_id:
        store.save_protective_order({
            "symbol": "SOL", "kind": "oco", "order_type": "OCO",
            "exchange_ref": oco_list_id, "order_list_id": oco_list_id, "qty": qty,
            "stop_price": 90.0, "limit_price": 89.8, "target_price": 125.0,
            "trailing_delta": None, "status": "resting", "mode": "binance-live",
            "venue": "binance-us", "placed_at": iso(T0), "response": "{}"})


def _stop_fired_overnight(client, qty=2.0, fill=90.0):
    client.order_lists[7] = {"listOrderStatus": "ALL_DONE",
                             "orders": [{"orderId": 201}, {"orderId": 202}]}
    client.orders[201] = {"type": "STOP_LOSS_LIMIT", "status": "FILLED",
                          "executedQty": str(qty)}
    client.orders[202] = {"type": "LIMIT_MAKER", "status": "EXPIRED", "executedQty": "0"}
    client.trades["SOLUSDT"] = [
        _trade(100, T0, qty, 100.0, True, fee=0.2),                 # our entry
        _trade(201, T0 + pd.Timedelta(hours=9), qty, fill, False, 7, fee=0.18),
    ]
    client.bal = {"USDT": 1_180.0}                                   # SOL gone
    client.px["SOLUSDT"] = fill


def test_noop_outside_live_mode(store, cfg):
    assert reconcile.reconcile(store, PaperBroker(store, cfg), cfg,
                               verbose=False) == {"skipped": "not live"}


def test_overnight_stop_fill_is_imported_and_the_position_closed(store, cfg, live):
    _open_sol_position(store)
    _stop_fired_overnight(live.client)

    res = reconcile.reconcile(store, live, cfg, verbose=False)

    assert [p["status"] for p in res["protection"]] == ["filled"]
    assert store.resting_orders().empty
    imported = store.live_orders("SOL")
    sell = imported[imported["side"] == "SELL"].iloc[0]
    assert sell["order_id"] == "bnc-x-201"
    assert sell["order_type"] == "PROTECTIVE"
    assert (sell["qty"], sell["price"], sell["fee_usd"]) == (2.0, 90.0, 0.18)
    assert store.position_meta("SOL").empty          # stale terms cleared
    assert res["closed"] == ["SOL"] and res["drift"] == []

    trades = evaluation.realized_trades(store, cfg)
    closed = trades[trades["close_ts"].notna()]
    assert len(closed) == 1
    assert closed["net_pnl"].iloc[0] == pytest.approx(2 * (90 - 100) - 0.2 - 0.18)


def test_reconcile_is_idempotent(store, cfg, live):
    _open_sol_position(store)
    _stop_fired_overnight(live.client)
    reconcile.reconcile(store, live, cfg, verbose=False)
    again = reconcile.reconcile(store, live, cfg, verbose=False)
    assert again["imported"] == [] and again["protection"] == []
    assert (store.live_orders("SOL")["side"] == "SELL").sum() == 1


def test_a_stop_filling_across_two_cycles_updates_one_row(store, cfg, live):
    _open_sol_position(store, oco_list_id=None)
    c = live.client
    c.px["SOLUSDT"] = 90.0
    c.bal = {"USDT": 1_000.0, "SOL": 1.0}
    c.trades["SOLUSDT"] = [_trade(100, T0, 2.0, 100.0, True),
                           _trade(300, T0 + pd.Timedelta(hours=1), 1.0, 90.0, False)]
    reconcile.reconcile(store, live, cfg, verbose=False)
    c.trades["SOLUSDT"].append(_trade(300, T0 + pd.Timedelta(hours=2), 1.0, 89.0, False))
    reconcile.reconcile(store, live, cfg, verbose=False)

    sells = store.live_orders("SOL").query("side == 'SELL'")
    assert len(sells) == 1
    assert sells["qty"].iloc[0] == 2.0
    assert sells["price"].iloc[0] == pytest.approx(89.5)
    assert sells["order_type"].iloc[0] == "EXTERNAL"      # not one of our OCOs


def test_fills_before_local_tracking_are_not_imported(store, cfg, live):
    _open_sol_position(store, oco_list_id=None)
    c = live.client
    c.px["SOLUSDT"] = 100.0
    c.bal = {"USDT": 1_000.0, "SOL": 2.0}
    c.trades["SOLUSDT"] = [_trade(50, T0 - pd.Timedelta(days=30), 5.0, 60.0, True),
                           _trade(100, T0, 2.0, 100.0, True)]
    res = reconcile.reconcile(store, live, cfg, verbose=False)
    assert res["imported"] == []
    assert len(store.live_orders("SOL")) == 1


def test_unreadable_balances_never_wipe_entry_terms(store, cfg, live):
    _open_sol_position(store, oco_list_id=None)
    live.client.fail_balances = True
    res = reconcile.reconcile(store, live, cfg, verbose=False)
    assert "error" in res
    assert not store.position_meta("SOL").empty


def test_drift_is_reported_when_the_exchange_holds_more(store, cfg, live):
    _open_sol_position(store, oco_list_id=None)
    c = live.client
    c.px["SOLUSDT"] = 100.0
    c.bal = {"USDT": 1_000.0, "SOL": 3.0}
    c.trades["SOLUSDT"] = [_trade(100, T0, 2.0, 100.0, True)]
    res = reconcile.reconcile(store, live, cfg, verbose=False)
    assert res["drift"] == [{"symbol": "SOL", "local": 2.0, "exchange": 3.0, "gap": 1.0}]


# -- protect_open_positions ---------------------------------------------------
def test_unprotected_position_gets_an_oco_for_the_free_balance(store, cfg, live):
    _open_sol_position(store, oco_list_id=None)
    c = live.client
    c.bal = {"USDT": 1_000.0, "SOL": 2.0}
    c.px["SOLUSDT"] = 105.0
    placed = reconcile.protect_open_positions(store, live, cfg, verbose=False)
    assert len(placed) == 1
    assert c.placed == [("oco", "SOLUSDT", 2.0, 125.0, 90.0)]
    assert store.resting_orders("SOL")["kind"].tolist() == ["oco"]
    # already covered next cycle: nothing new
    assert reconcile.protect_open_positions(store, live, cfg, verbose=False) == []


def test_position_without_a_target_gets_a_stop_only(store, cfg, live):
    _open_sol_position(store, oco_list_id=None)
    store.clear_position_target("SOL", "binance")
    live.client.bal = {"USDT": 1_000.0, "SOL": 2.0}
    live.client.px["SOLUSDT"] = 105.0
    reconcile.protect_open_positions(store, live, cfg, verbose=False)
    assert live.client.placed == [("stop", "SOLUSDT", 2.0, None, 90.0)]


@pytest.mark.parametrize("last", [89.0, 130.0])
def test_position_already_through_a_level_is_left_to_the_exits_pass(store, cfg, live, last):
    _open_sol_position(store, oco_list_id=None)
    live.client.bal = {"USDT": 1_000.0, "SOL": 2.0}
    live.client.px["SOLUSDT"] = last
    assert reconcile.protect_open_positions(store, live, cfg, verbose=False) == []
    assert live.client.placed == []


def test_protection_is_off_when_disabled_or_not_live(store, cfg, live):
    _open_sol_position(store, oco_list_id=None)
    live.client.bal = {"USDT": 1_000.0, "SOL": 2.0}
    live.client.px["SOLUSDT"] = 105.0
    cfg.execution.place_stop_orders = False
    assert reconcile.protect_open_positions(store, live, cfg, verbose=False) == []
    cfg.execution.place_stop_orders = True
    live.live = False
    assert reconcile.protect_open_positions(store, live, cfg, verbose=False) == []


def test_adding_to_a_protected_position_replaces_its_protection(store, cfg, live):
    _open_sol_position(store)                  # resting OCO for 2 SOL
    c = live.client
    c.bal = {"USDT": 700.0, "SOL": 3.0}        # just bought 1 more
    live.place_protection({"symbol": "SOL", "stop": 95.0, "target": 130.0, "qty": 1.0},
                          {"qty": 1.0})
    assert c.cancelled == ["SOLUSDT"]
    assert c.placed == [("oco", "SOLUSDT", 3.0, 130.0, 95.0)]      # whole holding
    rest = store.resting_orders("SOL")
    assert len(rest) == 1 and rest["qty"].iloc[0] == 3.0


def test_paper_terms_for_a_live_symbol_are_left_alone(store, cfg, live):
    _open_sol_position(store, oco_list_id=None)
    store.upsert_position_meta({"symbol": "SOL", "opened_at": iso(T0), "entry_price": 120.0,
                                "stop": 96.0, "target": 170.0, "horizon_days": 30.0,
                                "proposal_id": "paper-p", "mode": "paper"})
    _stop_fired_overnight(live.client)           # the LIVE position closed
    reconcile.reconcile(store, live, cfg, verbose=False)
    assert store.position_meta("SOL", book="binance").empty
    assert store.position_meta("SOL", book="paper")["stop"].iloc[0] == 96.0
