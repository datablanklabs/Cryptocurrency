"""The BTC-trend gate: uptrend -> risk_on/full size, downtrend -> risk_off/no
buys, thin history -> abstain, disabled -> full size."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_disabled_gate_ignores_macro_even_with_a_store_and_a_bad_reading(monkeypatch):
    """`enabled=False` must mean full size unconditionally - `describe()` prints
    'full size regardless of trend', not 'unless macro disagrees'. A regression
    guard: assess() used to route through _apply_macro() even when disabled, so
    a macro score at/below macro_risk_off_threshold could still force risk_off
    on a gate the operator explicitly turned off."""
    _patch_prices(monkeypatch, list(np.linspace(360, 100, 260)))   # would be risk_off
    monkeypatch.setattr(regime.macro_mod, "score",
                        lambda store, cfg: {"n_series": 1, "score": -0.99, "note": "shutdown"})
    cfg = Config()
    cfg.regime.enabled = False
    v = regime.assess(cfg, store=object())
    assert v["state"] == "risk_on"
    assert v["exposure_scale"] == 1.0
    assert v["macro_score"] is None


def test_price_fetch_failure_degrades_to_neutral(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("all sources down")
    monkeypatch.setattr(regime.prices_mod, "get_ohlcv", _boom)
    v = regime.assess(Config())
    assert v["state"] == "neutral"
    assert v["exposure_scale"] == Config().regime.neutral_exposure


# --------------------------------------------------------------------------
# Macro fold-in (Kalshi, via cryptoyolo.macro) — regime.macro_mod.score is
# monkeypatched directly, the same way _patch_prices stands in for the price
# fetch; macro.py's own scoring math is covered in test_macro.py.
# --------------------------------------------------------------------------
def _patch_macro(monkeypatch, result: dict):
    monkeypatch.setattr(regime.macro_mod, "score", lambda store, cfg: result)


def test_macro_skipped_when_no_store_is_passed(monkeypatch):
    _patch_prices(monkeypatch, list(np.linspace(100, 360, 260)))
    v = regime.assess(Config())          # no store kwarg
    assert v["state"] == "risk_on"
    assert v["exposure_scale"] == 1.0
    assert v["macro_score"] is None


def test_macro_dampens_exposure_without_flipping_state(monkeypatch):
    _patch_prices(monkeypatch, list(np.linspace(100, 360, 260)))   # would be risk_on
    _patch_macro(monkeypatch, {"n_series": 1, "score": -0.3, "note": "CPI hot 80%"})
    cfg = Config()
    v = regime.assess(cfg, store=object())      # macro_mod.score is stubbed; any non-None store
    assert v["state"] == "risk_on"
    expected_mult = max(cfg.regime.macro_min_multiplier,
                        1.0 + (-0.3) * cfg.regime.macro_downweight)
    assert v["exposure_scale"] == pytest.approx(cfg.regime.risk_on_exposure * expected_mult)
    assert v["macro_score"] == -0.3
    assert "macro" in v["note"]


def test_macro_forces_risk_off_past_threshold(monkeypatch):
    _patch_prices(monkeypatch, list(np.linspace(100, 360, 260)))   # would be risk_on
    _patch_macro(monkeypatch, {"n_series": 1, "score": -0.9, "note": "shutdown 95%"})
    v = regime.assess(Config(), store=object())
    assert v["state"] == "risk_off"
    assert v["exposure_scale"] == 0.0
    assert v["macro_forced_risk_off"] is True
    assert "override" in v["note"]


def test_macro_never_boosts_exposure_past_the_trend(monkeypatch):
    _patch_prices(monkeypatch, [200.0] * 260)                     # flat -> neutral trend
    _patch_macro(monkeypatch, {"n_series": 1, "score": 0.8, "note": "calm"})
    cfg = Config()
    v = regime.assess(cfg, store=object())
    assert v["state"] == "neutral"
    assert v["exposure_scale"] == cfg.regime.neutral_exposure     # unchanged, not boosted


def test_macro_disabled_via_config_is_ignored_even_with_a_store(monkeypatch):
    _patch_prices(monkeypatch, list(np.linspace(100, 360, 260)))
    _patch_macro(monkeypatch, {"n_series": 1, "score": -0.9, "note": "x"})
    cfg = Config()
    cfg.regime.macro_enabled = False
    v = regime.assess(cfg, store=object())
    assert v["state"] == "risk_on"
    assert v["exposure_scale"] == 1.0
    assert v["macro_score"] is None
