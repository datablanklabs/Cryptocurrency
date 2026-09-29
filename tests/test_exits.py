"""exits.evaluate: each trigger fires on its own condition, precedence holds,
the trailing stop's memory persists, and each book is judged on its own terms.
Prices are passed in, so nothing touches the network."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from cryptoyolo import exits
from cryptoyolo.store import iso, utcnow


def _meta(store, symbol="SOL", entry=100.0, stop=90.0, target=130.0, age_days=1.0,
          high_water=None, mode="paper"):
    store.upsert_position_meta({
        "symbol": symbol, "opened_at": iso(utcnow() - timedelta(days=age_days)),
        "entry_price": entry, "stop": stop, "target": target, "horizon_days": 30.0,
        "high_water": high_water if high_water is not None else entry,
        "proposal_id": "p", "mode": mode})


def _eval(store, cfg, last, holdings=None, composite=None, book="paper", symbol="SOL"):
    scores = (pd.DataFrame({"symbol": [symbol], "composite": [composite]})
              if composite is not None else None)
    return exits.evaluate(store, cfg, holdings or {symbol: 10.0}, scores,
                          prices={symbol: last}, book=book)


def test_nothing_fires_inside_the_levels(store, cfg):
    _meta(store)
    assert _eval(store, cfg, 105.0, composite=0.2).empty


def test_disabled_or_flat_returns_empty(store, cfg):
    _meta(store)
    assert exits.evaluate(store, cfg, {}, None, prices={}).empty
    cfg.exits.enabled = False
    assert _eval(store, cfg, 50.0).empty


def test_the_quote_asset_is_never_a_position(store, cfg):
    sig = exits.evaluate(store, cfg, {"USDT": 500.0}, None, prices={"USDT": 0.5})
    assert sig.empty


def test_stop_hit_sells_everything(store, cfg):
    _meta(store)
    sig = _eval(store, cfg, 89.0).iloc[0]
    assert (sig["trigger"], sig["qty"], sig["fraction_pct"]) == ("stop", 10.0, 100.0)
    assert sig["pnl_pct"] == pytest.approx(-11.0)


def test_target_hit_sells_the_configured_fraction(store, cfg):
    _meta(store)
    cfg.exits.take_profit_fraction = 50.0
    sig = _eval(store, cfg, 131.0).iloc[0]
    assert (sig["trigger"], sig["qty"]) == ("target", 5.0)
    assert "Partial exit" in exits.describe(sig, cfg)


def test_trailing_stop_needs_the_trade_to_have_run_first(store, cfg):
    cfg.exits.trail_pct, cfg.exits.trail_activate_pct = 8.0, 3.0
    _meta(store, high_water=102.0)                  # peaked only +2%: not armed
    assert _eval(store, cfg, 93.0).empty            # -8.8% from high, above the stop
    _meta(store, symbol="ETH", high_water=120.0)    # peaked +20%: armed
    sig = _eval(store, cfg, 110.0, symbol="ETH").iloc[0]
    assert sig["trigger"] == "trailing" and sig["high_water"] == 120.0


def test_high_water_persists_across_runs(store, cfg):
    _meta(store)
    _eval(store, cfg, 125.0)                        # a new high, nothing fires
    assert store.position_meta("SOL", book="paper")["high_water"].iloc[0] == 125.0
    sig = _eval(store, cfg, 114.0).iloc[0]          # -8.8% from the remembered high
    assert sig["trigger"] == "trailing"


def test_horizon_counts_from_the_recorded_open(store, cfg):
    _meta(store, age_days=31)
    sig = _eval(store, cfg, 105.0).iloc[0]
    assert sig["trigger"] == "horizon" and sig["age_days"] == pytest.approx(31, abs=0.01)


def test_score_reversal(store, cfg):
    _meta(store)
    sig = _eval(store, cfg, 105.0, composite=-0.2).iloc[0]
    assert sig["trigger"] == "reversal"


def test_level_events_take_precedence_and_the_rest_are_listed(store, cfg):
    _meta(store, age_days=40)
    sig = _eval(store, cfg, 85.0, composite=-0.5).iloc[0]
    assert sig["trigger"] == "stop"
    assert sig["also_fired"] == "horizon expired, score reversed"


def test_positions_without_terms_only_exit_on_reversal(store, cfg):
    assert _eval(store, cfg, 1.0).empty             # would be far through any stop
    sig = _eval(store, cfg, 1.0, composite=-0.5).iloc[0]
    assert sig["trigger"] == "reversal" and not sig["has_metadata"]
    assert "opened outside" in exits.describe(sig, cfg)
    cfg.exits.allow_exits_without_metadata = False
    assert _eval(store, cfg, 1.0, composite=-0.5).empty


def test_each_book_is_judged_on_its_own_terms(store, cfg):
    _meta(store, stop=90.0, mode="binance-live")
    _meta(store, stop=110.0, mode="paper")
    assert _eval(store, cfg, 100.0, book="binance").empty
    assert _eval(store, cfg, 100.0, book="paper").iloc[0]["trigger"] == "stop"


def test_book_defaults_to_the_configured_mode(store, cfg):
    _meta(store, stop=110.0, mode="binance-live")
    assert exits.evaluate(store, cfg, {"SOL": 1.0}, None, prices={"SOL": 100.0}).empty
    cfg.execution.mode = "binance"
    sig = exits.evaluate(store, cfg, {"SOL": 1.0}, None, prices={"SOL": 100.0})
    assert sig.iloc[0]["trigger"] == "stop"


def test_disabled_triggers_do_not_fire(store, cfg):
    _meta(store, age_days=40)
    cfg.exits.stop_loss = cfg.exits.horizon_expiry = False
    assert _eval(store, cfg, 85.0).empty
    tbl = exits.active_triggers(cfg).set_index("trigger")["enabled"]
    assert (tbl["stop hit"], tbl["horizon expired"], tbl["target hit"]) == ("off", "off", "on")
