"""Pipeline monitoring: the cycle summary's counts and edge-triggered alerts."""

from __future__ import annotations

import pandas as pd

from cryptoyolo import pipeline
from cryptoyolo.store import Store


def test_cycle_summary_does_not_count_validated_as_executed(cfg):
    execs = pd.DataFrame({"status": ["FILLED", "VALIDATED", "REJECTED"]})
    s = pipeline._cycle_summary(cfg, "r1", regime=None, proposals=None,
                                executions=execs, equity=None)
    assert (s["executed"], s["validated"], s["rejected"]) == (1, 1, 1)


class _Broker:
    mode = "paper"


def test_drawdown_alert_fires_once_on_the_crossing(store: Store, cfg, monkeypatch):
    sent = []
    monkeypatch.setattr(pipeline, "notify", lambda title, *a, **k: sent.append(title))
    monkeypatch.setattr(pipeline, "_snapshot_equity", lambda *a, **k: None)
    cfg.notify.on_drawdown = True
    cfg.notify.drawdown_alert_pct = 10.0

    for i, eq in enumerate([100.0, 100.0, 85.0]):       # crosses -10% here
        store.record_equity(f"r{i}", eq, 0.0, eq, mode="paper")
    pipeline._finalize_equity(store, _Broker(), None, cfg, "r2")
    assert len(sent) == 1

    store.record_equity("r3", 80.0, 0.0, 80.0, mode="paper")   # still below
    pipeline._finalize_equity(store, _Broker(), None, cfg, "r3")
    assert len(sent) == 1                                 # no repeat


def test_last_regime_state_tracks_the_latest_cycle(store: Store):
    assert store.last_regime_state() is None
    store.record_regime("r1", {"state": "risk_on"})
    store.record_regime("r2", {"state": "risk_off"})
    assert store.last_regime_state() == "risk_off"
