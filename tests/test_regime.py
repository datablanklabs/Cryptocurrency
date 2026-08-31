"""The BTC-trend gate: uptrend -> risk_on/full size, downtrend -> risk_off/no
buys, thin history -> abstain, disabled -> full size."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoyolo import regime
from cryptoyolo.config import Config


def _btc_frame(values):
    idx = pd.date_range("2025-06-01", periods=len(values), freq="D", tz="UTC")
    return pd.DataFrame({"open": values, "high": values, "low": values,
                         "close": values, "volume": 1.0}, index=idx)


def _patch_prices(monkeypatch, values):
    monkeypatch.setattr(regime.prices_mod, "get_ohlcv",
                        lambda sym, tf, cfg: (_btc_frame(values), "test"))


def test_uptrend_is_risk_on(monkeypatch):
    _patch_prices(monkeypatch, list(np.linspace(100, 360, 260)))
    v = regime.assess(Config())
    assert v["state"] == "risk_on"
    assert v["exposure_scale"] == 1.0


def test_downtrend_is_risk_off(monkeypatch):
    _patch_prices(monkeypatch, list(np.linspace(360, 100, 260)))
    v = regime.assess(Config())
    assert v["state"] == "risk_off"
    assert v["exposure_scale"] == 0.0


def test_thin_history_abstains_to_neutral(monkeypatch):
    _patch_prices(monkeypatch, list(np.linspace(100, 120, 50)))
    v = regime.assess(Config())
    assert v["state"] == "neutral"
    assert "min_candles" in v["note"]


def test_disabled_gate_returns_full_size(monkeypatch):
    _patch_prices(monkeypatch, list(np.linspace(360, 100, 260)))   # would be risk_off
    cfg = Config()
    cfg.regime.enabled = False
    v = regime.assess(cfg)
    assert v["state"] == "risk_on"
    assert v["exposure_scale"] == 1.0


def test_price_fetch_failure_degrades_to_neutral(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("all sources down")
    monkeypatch.setattr(regime.prices_mod, "get_ohlcv", _boom)
    v = regime.assess(Config())
    assert v["state"] == "neutral"
    assert v["exposure_scale"] == Config().regime.neutral_exposure
