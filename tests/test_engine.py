"""Scoring helpers and — the important one — risk-first position sizing in
`engine.propose`: a stop-out must cost exactly `risk_per_trade_pct` of the risk
basis, the fee-adjusted target must land net R:R on the configured ratio, and
`target_pct_cap` must clip an unreachable target rather than lie about R:R."""

from __future__ import annotations

import pytest

from cryptoyolo import engine
from cryptoyolo.config import Config


def test_clip_and_squash_bounds():
    assert engine._clip(5.0) == 1.0
    assert engine._clip(-5.0) == -1.0
    assert engine._squash(0.0, 1.0) == 0.0
    assert engine._squash(10.0, 0.0) == 0.0            # zero scale -> zero, no div
    assert -1.0 <= engine._squash(3.0, 1.0) <= 1.0


def test_technical_score_is_bounded_and_labelled():
    short = {"close": 100.0, "atr_pct": 0.03, "macd_hist": 0.4, "vol_zscore": 1.0}
    medium = {"close": 100.0, "atr_pct": 0.03, "dist_sma_20": 0.05,
              "dist_sma_50": 0.08, "bb_pctb": 0.7, "rsi_14": 55.0,
              "macd_hist": 0.3, "squeeze": 0.0}
    score, parts = engine.technical_score(short, medium)
    assert -1.0 <= score <= 1.0
    assert {"trend", "momentum", "position", "rsi", "volume"} <= set(parts)


def _scores_row(**over):
    row = {
        "symbol": "AAA", "composite": 0.5, "price": 100.0,
        "atr_pct_1w": 0.04, "atr_pct_daily": 0.04,
        "technical": 0.5, "technical_raw": 0.5, "xsec": 0.0,
        "social": 0.0, "catalyst": 0.0, "positioning": 0.0, "events": 0.0,
        "w_technical": 0.25, "w_social": 0.0, "w_catalyst": 0.0,
        "w_positioning": 0.0, "w_events": 0.0,
        "components": {},
    }
    row.update(over)
    return row


def _one_row_frame(**over):
    import pandas as pd
    return pd.DataFrame([_scores_row(**over)])


def _sizing_cfg() -> Config:
    c = Config()
    c.notify.enabled = False
    c.risk.size_off_live_equity = False
    c.risk.account_equity_usd = 10_000.0
    c.risk.risk_per_trade_pct = 1.0
    c.risk.stop_scaling = "legacy"          # stop_pct = atr_stop_mult * atr_daily
    c.risk.atr_stop_mult = 1.5
    c.risk.max_position_pct = 50.0
    c.risk.max_total_deployed_pct = 90.0
    c.risk.correlation_sizing = False
    c.risk.fee_adjust_targets = False
    return c


def test_propose_sizes_so_a_stop_out_costs_exactly_risk_per_trade(store):
    cfg = _sizing_cfg()
    props = engine.propose(_one_row_frame(atr_pct_daily=0.04), store, "r1", cfg,
                           holdings={}, cash_available=None)
    assert len(props) == 1
    p = props.iloc[0]
    assert p["side"] == "BUY"
    # stop_pct = 1.5 * 0.04 = 0.06  ->  stop_dist = 6.0 on a $100 entry
    assert p["entry"] - p["stop"] == pytest.approx(6.0, rel=1e-9)
    # the invariant: (entry - stop) * qty == 1% of $10,000 == $100
    assert (p["entry"] - p["stop"]) * p["qty"] == pytest.approx(100.0, rel=1e-9)
    assert p["risk_usd"] == pytest.approx(100.0, abs=0.01)
    # plain 2R target, fees off
    assert p["reward_risk"] == pytest.approx(2.0, rel=1e-9)


def test_propose_fee_adjusted_target_hits_net_reward_risk(store):
    cfg = _sizing_cfg()
    cfg.risk.fee_adjust_targets = True
    cfg.risk.reward_risk_target = 2.0
    cfg.fees.taker_bps = 40.0
    cfg.fees.slippage_bps = 6.0            # round_trip_cost_bps = 92 -> 0.92 on $100
    props = engine.propose(_one_row_frame(atr_pct_daily=0.04), store, "r2", cfg,
                           holdings={}, cash_available=None)
    p = props.iloc[0]
    assert p["reward_risk"] > 2.0                       # widened gross
    assert p["reward_risk_net"] == pytest.approx(2.0, abs=0.02)


def test_propose_target_pct_cap_clips_an_unreachable_target(store):
    cfg = _sizing_cfg()
    cfg.risk.reward_risk_target = 2.0
    cfg.risk.target_pct_cap = 10.0        # target may sit at most 10% from entry
    props = engine.propose(_one_row_frame(atr_pct_daily=0.10), store, "r3", cfg,
                           holdings={}, cash_available=None)
    p = props.iloc[0]
    # stop_dist = 1.5 * 0.10 * 100 = 15; a 2R target would be 30 away, but the
    # cap holds it to 10 -> honest R:R is well under 2.
    assert (p["target"] - p["entry"]) == pytest.approx(10.0, rel=1e-9)
    assert p["reward_risk"] < 2.0


def test_propose_empty_when_nothing_clears_the_floor(store):
    cfg = _sizing_cfg()
    props = engine.propose(_one_row_frame(composite=0.001), store, "r4", cfg,
                           holdings={}, cash_available=None)
    assert props.empty
    assert "skipped" in props.attrs
