"""portfolio.snapshot: what you own, what it cost, what it's worth - honest
about what it can't know (unreadable balances, positions opened elsewhere)."""

from __future__ import annotations

import pytest

from cryptoyolo import portfolio
from cryptoyolo.store import iso


@pytest.fixture
def px(monkeypatch):
    prices = {"SOL": 110.0, "ETH": 2_000.0}
    monkeypatch.setattr(portfolio.prices_mod, "latest_prices",
                        lambda syms, cfg: {s: prices[s] for s in syms if s in prices})
    return prices


def test_paper_snapshot_values_the_book_exactly(store, cfg, px):
    store.apply_paper_fill("SOL", "BUY", 10.0, 100.0)            # $1,000 of cash spent
    store.upsert_position_meta({"symbol": "SOL", "opened_at": iso(), "entry_price": 100.0,
                                "stop": 90.0, "target": 130.0, "horizon_days": 30.0,
                                "proposal_id": "p", "mode": "paper"})
    snap = portfolio.snapshot(store, cfg)
    assert (snap["mode"], snap["cash"]) == ("paper", 9_000.0)
    row = snap["positions"].iloc[0]
    assert (row["symbol"], row["avg_cost"], row["value"], row["pnl"]) == ("SOL", 100.0, 1_100.0, 100.0)
    assert row["stop"] == 90.0 and row["to_stop_pct"] == pytest.approx((110 / 90 - 1) * 100)
    assert snap["total_equity"] == 10_100.0 and not snap["warnings"]


def test_paper_snapshot_ignores_the_live_books_terms(store, cfg, px):
    store.apply_paper_fill("SOL", "BUY", 1.0, 100.0)
    store.upsert_position_meta({"symbol": "SOL", "opened_at": iso(), "entry_price": 50.0,
                                "stop": 40.0, "target": 80.0, "horizon_days": 30.0,
                                "proposal_id": "live", "mode": "binance-live"})
    row = portfolio.snapshot(store, cfg)["positions"].iloc[0]
    assert row["stop"] is None
    assert any("no recorded entry terms" in w for w in portfolio.snapshot(store, cfg)["warnings"])


def _live_order(store, oid, side, qty, price, status="FILLED", mode="binance-live"):
    store.save_order({"order_id": oid, "proposal_id": None, "run_id": "r", "ts": iso(),
                      "venue": "binance-us", "mode": mode, "symbol": "SOL", "side": side,
                      "order_type": "MARKET", "qty": qty, "price": price, "status": status,
                      "exchange_ref": oid, "response": "{}"})


def test_live_cost_basis_uses_only_live_fills_including_partials(store):
    _live_order(store, "a", "BUY", 2.0, 100.0)
    _live_order(store, "b", "BUY", 2.0, 110.0, status="PARTIALLY_FILLED")
    _live_order(store, "c", "BUY", 50.0, 1.0, mode="paper")          # another book
    _live_order(store, "d", "BUY", 50.0, 1.0, status="EXPIRED_NO_FILL")
    _live_order(store, "e", "SELL", 1.0, 120.0)
    basis = portfolio._cost_basis_from_orders(store)["SOL"]
    assert basis["qty"] == pytest.approx(3.0)
    assert basis["avg"] == pytest.approx(105.0)


class _LiveBroker:
    mode, venue = "binance", "binance-us"

    def __init__(self, cash, holdings):
        self._cash, self._holdings = cash, holdings

    def available_cash(self):
        return self._cash

    def holdings(self):
        return self._holdings


def test_live_snapshot_claims_no_basis_it_cannot_support(store, cfg, px, monkeypatch):
    cfg.execution.mode = "binance"
    _live_order(store, "a", "BUY", 1.0, 100.0)         # we only bought 1 of the 3 held
    monkeypatch.setattr(portfolio, "get_broker",
                        lambda s, c: _LiveBroker(500.0, {"USDT": 500.0, "SOL": 3.0}))
    snap = portfolio.snapshot(store, cfg)
    row = snap["positions"].iloc[0]
    assert row["symbol"] == "SOL"                       # USDT is cash, not a position
    assert row["avg_cost"] is None and row["pnl"] is None
    assert any("Cost basis shown only" in w for w in snap["warnings"])
    assert "—" in portfolio.format_summary(snap)


def test_unreadable_balance_is_unknown_not_zero(store, cfg, px, monkeypatch):
    cfg.execution.mode = "binance"
    monkeypatch.setattr(portfolio, "get_broker", lambda s, c: _LiveBroker(None, {}))
    snap = portfolio.snapshot(store, cfg)
    assert snap["cash"] is None and snap["total_equity"] is None
    assert any("Balance unavailable" in w for w in snap["warnings"])
    assert "unavailable" in portfolio.format_summary(snap)


def test_risk_basis_drift_is_flagged(store, cfg, px):
    cfg.risk.account_equity_usd = 50_000.0             # paper equity is ~$10,000
    assert any("more than 20% away" in w for w in portfolio.snapshot(store, cfg)["warnings"])
