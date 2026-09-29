"""loss_halt: trips on drawdown or a trailing-window loss, blocks BUYs only,
and resumes only when told to."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from cryptoyolo import engine, loss_halt
from cryptoyolo.store import iso, utcnow
from tests.test_engine import _scores_row, _sizing_cfg


def _snap(store, days_ago: float, equity: float, mode: str = "paper") -> None:
    """An equity snapshot at a chosen time (record_equity always stamps now)."""
    with store.conn() as con:
        con.execute("INSERT INTO equity_snapshots (run_id, ts, mode, cash, "
                    "positions_value, equity) VALUES (?,?,?,?,?,?)",
                    (f"r{days_ago}", iso(utcnow() - timedelta(days=days_ago)), mode,
                     equity, 0.0, equity))


def test_disabled_or_without_history_never_trips(store, cfg):
    cfg.risk.halt_enabled = False
    assert loss_halt.assess(store, cfg, "paper", 1.0)["tripped"] is False
    cfg.risk.halt_enabled = True
    v = loss_halt.assess(store, cfg, "paper", 1_000.0)
    assert v["tripped"] is False and "not enough" in v["note"]


def test_drawdown_from_peak_trips(store, cfg):
    _snap(store, 3, 100.0)
    _snap(store, 2, 120.0)                       # peak
    v = loss_halt.assess(store, cfg, "paper", 100.0)
    assert v["drawdown_pct"] == -16.67
    assert v["tripped"] and len(v["reasons"]) == 1 and "peak" in v["reasons"][0]


def test_window_loss_trips_against_the_snapshot_before_the_window(store, cfg):
    _snap(store, 10, 100.0)                      # baseline: last one before -7d
    _snap(store, 6, 100.0)
    v = loss_halt.assess(store, cfg, "paper", 91.0)
    assert v["window_loss_pct"] == -9.0
    assert v["tripped"] and "7d" in v["reasons"][0]


def test_modes_are_separate_books(store, cfg):
    _snap(store, 2, 10_000.0, mode="paper")
    _snap(store, 1, 500.0, mode="binance")
    assert loss_halt.assess(store, cfg, "binance", 500.0)["tripped"] is False


def test_reset_after_ignores_the_old_peak(store, cfg):
    _snap(store, 5, 120.0)
    _snap(store, 2, 100.0)
    assert loss_halt.assess(store, cfg, "paper", 100.0)["tripped"]
    cfg.risk.halt_reset_after = iso(utcnow() - timedelta(days=3))
    assert loss_halt.assess(store, cfg, "paper", 100.0)["tripped"] is False


def test_unparseable_reset_is_ignored_not_fatal(store, cfg):
    _snap(store, 5, 120.0)
    cfg.risk.halt_reset_after = "next tuesday"
    v = loss_halt.assess(store, cfg, "paper", 100.0)
    assert v["tripped"] and "unparseable" in v["note"]


def test_halt_blocks_buys_but_not_exits(store):
    cfg = _sizing_cfg()
    scores = pd.DataFrame([_scores_row(symbol="AAA", composite=0.5),
                           _scores_row(symbol="BBB", composite=-0.5)])
    halt = {"tripped": True, "note": "equity -20% below its peak"}
    props = engine.propose(scores, store, "r1", cfg, holdings={"BBB": 50.0},
                           cash_available=5_000.0, halt=halt)
    assert props["side"].tolist() == ["SELL"]            # the trim still runs

    empty = engine.propose(scores.iloc[:1], store, "r2", cfg, cash_available=5_000.0,
                           halt=halt)
    assert empty.empty and empty.attrs["skipped"]["halted"] == 1
    assert "LOSS HALT" in engine.format_proposals(empty)


def test_halt_log_round_trip(store):
    assert store.last_halt_state("paper") is None
    store.record_halt("r1", "paper", {"tripped": True, "reasons": ["x"]})
    store.record_halt("r2", "binance", {"tripped": False})
    assert store.last_halt_state("paper") is True
    assert store.last_halt_state("binance") is False
