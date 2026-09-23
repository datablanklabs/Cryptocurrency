"""execute_approved: a validate-only (dry-run) order is not a fill and must
not be announced as one."""

from __future__ import annotations

import pandas as pd

from cryptoyolo import approval
from cryptoyolo.store import Store


class _ValidateOnlyBroker:
    mode = "binance-test"

    def execute(self, proposal, run_id):
        return {"symbol": proposal["symbol"], "side": proposal["side"],
                "qty": proposal["qty"], "price": 100.0, "status": "VALIDATED",
                "mode": self.mode, "venue": "binance", "order_id": "x"}

    def holdings(self):
        return {"BTC": 1.0}             # nothing was sold


def test_validated_exit_sends_no_fill_alert(store: Store, cfg, monkeypatch):
    sent = []
    monkeypatch.setattr(approval, "notify", lambda title, *a, **k: sent.append(title))
    cfg.notify.on_exit_trigger = True
    cfg.notify.on_execution = True
    proposals = pd.DataFrame([{
        "proposal_id": "p1", "decision": "approved", "symbol": "BTC",
        "side": "SELL", "kind": "exit", "trigger": "stop", "qty": 1.0,
        "entry": 100.0, "stop": 90.0, "target": 120.0,
    }])
    out = approval.execute_approved(proposals, _ValidateOnlyBroker(), store, "r1", cfg)
    assert list(out["status"]) == ["VALIDATED"]
    assert not any("filled" in t for t in sent)
