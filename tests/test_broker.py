"""Execution-cost maths and filter rounding — the parts that must be exact
because they decide the price and quantity that get sent."""

from __future__ import annotations

import math

import pytest

from cryptoyolo import broker
from cryptoyolo.broker import BinanceClient, is_rejected
from cryptoyolo.config import Config


def test_round_step_floors_to_the_tick():
    r = BinanceClient._round_step
    assert r(1.23456, "0.001") == "1.234"          # floored, never rounded up
    assert r(1.2, "0.5") == "1"
    assert r(1.7, "0.5") == "1.5"
    assert float(r(0.000000217, "0.00000001")) == pytest.approx(2.1e-7)
    assert r(5.0, "0") == "5"                       # zero step -> passthrough


def test_commission_scales_with_notional_and_respects_enabled(cfg: Config):
    assert broker.commission_usd(10_000, cfg) == pytest.approx(40.0)   # 40 bps
    assert broker.commission_usd(10_000, cfg, maker=True) == pytest.approx(40.0)
    cfg.fees.enabled = False
    assert broker.commission_usd(10_000, cfg) == 0.0


def test_apply_slippage_direction():
    assert broker.apply_slippage(100.0, "BUY", 10) == pytest.approx(100.1)
    assert broker.apply_slippage(100.0, "SELL", 10) == pytest.approx(99.9)


def test_slippage_bps_capped_by_limit_cross(cfg: Config):
    cfg.fees.slippage_bps = 50.0
    cfg.execution.entry_limit_cross_bps = 15.0
    assert broker.slippage_bps(1_000, cfg, order_type="MARKET") == 50.0
    assert broker.slippage_bps(1_000, cfg, order_type="LIMIT") == 15.0   # min(50, 15)


def test_rejected_state_membership():
    assert is_rejected("REJECTED_INSUFFICIENT_CASH")
    assert is_rejected("EXPIRED_NO_FILL")
    assert not is_rejected("FILLED")
    assert not is_rejected(None)


def test_paper_broker_models_fees_and_slippage_on_a_fill(store, cfg: Config,
                                                         monkeypatch):
    """A paper BUY must debit cash for notional + commission and fill through
    the modelled slippage, or the paper track record reads better than live."""
    monkeypatch.setattr("cryptoyolo.prices.latest_prices",
                        lambda syms, c: {"BTC": 100.0})
    pb = broker.PaperBroker(store, cfg)
    start_cash = store.paper_cash()
    proposal = {"proposal_id": "p1", "symbol": "BTC", "side": "BUY",
                "qty": 1.0, "entry": 100.0, "stop": 97.0, "target": 106.0,
                "notional": 100.0}
    rec = pb.execute(proposal, run_id="r1")

    assert rec["status"] == "FILLED"
    assert rec["price"] > 100.0                       # BUY paid up through slippage
    assert rec["fee_usd"] > 0.0
    spent = start_cash - store.paper_cash()
    assert spent == pytest.approx(rec["qty"] * rec["price"] + rec["fee_usd"], abs=1e-4)
