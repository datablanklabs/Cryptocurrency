"""Config maths: weight normalisation, fee model, the live-mode switches."""

from __future__ import annotations

import math

from cryptoyolo.config import Config, ExecutionConfig, FeeConfig, ScoreWeights


def test_normalized_weights_sum_to_one():
    n = ScoreWeights(technical=0.5, social=0.2, catalyst=0.3,
                     positioning=0.1, events=0.0).normalized()
    total = n.technical + n.social + n.catalyst + n.positioning + n.events
    assert math.isclose(total, 1.0, rel_tol=1e-9)
    # ratios preserved: technical was 5x social
    assert math.isclose(n.technical / n.social, 2.5, rel_tol=1e-9)


def test_normalized_all_zero_falls_back_to_uniform():
    w = ScoreWeights(0, 0, 0, 0, 0, 0, xsec_momentum_blend=0.42).normalized()
    assert (w.technical, w.social, w.catalyst, w.positioning, w.events,
           w.kalshi_prediction) == ((1 / 6,) * 6)
    assert w.xsec_momentum_blend == 0.42          # blend is not a family, kept as-is


def test_fee_bnb_discount():
    base = FeeConfig(taker_bps=40.0)
    assert base.effective_taker_bps == 40.0
    disc = FeeConfig(taker_bps=40.0, use_bnb_discount=True, bnb_discount_pct=25.0)
    assert math.isclose(disc.effective_taker_bps, 30.0, rel_tol=1e-9)
    assert FeeConfig(enabled=False, taker_bps=40.0).effective_taker_bps == 0.0


def test_round_trip_cost_is_two_legs_of_fee_plus_slippage():
    f = FeeConfig(taker_bps=40.0, maker_bps=40.0, slippage_bps=6.0)
    assert f.round_trip_bps == 80.0                     # fees only
    assert f.round_trip_cost_bps == 2 * 40.0 + 2 * 6.0  # + spread, both legs


def test_entry_slippage_guard_never_below_the_limit_cross():
    ex = ExecutionConfig(entry_order_type="LIMIT", entry_limit_cross_bps=15.0,
                         max_entry_slippage_bps=10.0)
    # raw guard (10) is below the cross (15) -> lifted to cross + margin
    assert ex.entry_slippage_guard_bps == 15.0 + 20.0
    # a sane raw guard is passed through
    ex2 = ExecutionConfig(entry_order_type="LIMIT", entry_limit_cross_bps=15.0,
                          max_entry_slippage_bps=60.0)
    assert ex2.entry_slippage_guard_bps == 60.0
    # 0 disables and is honoured as-is
    assert ExecutionConfig(max_entry_slippage_bps=0.0).entry_slippage_guard_bps == 0.0
    # MARKET entry: raw value used directly
    assert ExecutionConfig(entry_order_type="MARKET",
                           max_entry_slippage_bps=30.0).entry_slippage_guard_bps == 30.0


def test_live_disabled_without_env(monkeypatch):
    monkeypatch.delenv("CRYPTO_YOLO_ALLOW_LIVE", raising=False)
    c = Config()
    c.execution.mode = "binance"
    c.execution.dry_run = False
    assert c.execution.live_enabled is False        # env var still missing
