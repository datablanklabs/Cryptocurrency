"""portfolio_risk: heat = sqrt(r' C r) sits between the independent and the
perfectly-correlated case, and the correlation inputs degrade safely."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoyolo import portfolio_risk as prisk


def _corr(value: float, syms=("A", "B")) -> pd.DataFrame:
    m = np.full((len(syms), len(syms)), value, dtype=float)
    np.fill_diagonal(m, 1.0)
    return pd.DataFrame(m, index=list(syms), columns=list(syms))


def test_one_position_is_its_own_heat():
    h = prisk.portfolio_heat({"A": 100.0})
    assert (h["heat_usd"], h["gross_usd"], h["diversification_ratio"]) == (100.0, 100.0, 1.0)


def test_perfect_correlation_is_one_bet_and_independence_diversifies():
    same = prisk.portfolio_heat({"A": 100.0, "B": 100.0}, _corr(1.0))
    assert same["heat_usd"] == pytest.approx(200.0)
    indep = prisk.portfolio_heat({"A": 100.0, "B": 100.0}, _corr(0.0))
    assert indep["heat_usd"] == pytest.approx(np.sqrt(2) * 100.0)
    assert indep["diversification_ratio"] == pytest.approx(1 / np.sqrt(2))


def test_a_hedge_lowers_heat_below_the_independent_case():
    h = prisk.portfolio_heat({"A": 100.0, "B": 100.0}, _corr(-0.5))
    assert h["heat_usd"] == pytest.approx(100.0)          # sqrt(2 * 100^2 * (1 - 0.5))


def test_missing_pairs_fall_back_to_the_conservative_default():
    h = prisk.portfolio_heat({"A": 100.0, "B": 100.0}, None, default_corr=0.8)
    assert h["heat_usd"] == pytest.approx(np.sqrt(2 * 100**2 * 1.8))
    partial = prisk.portfolio_heat({"A": 100.0, "C": 100.0}, _corr(0.0))   # C not in matrix
    assert partial["heat_usd"] == pytest.approx(h["heat_usd"])


def test_zero_and_negative_risks_are_ignored():
    h = prisk.portfolio_heat({"A": 100.0, "B": 0.0, "C": -5.0})
    assert (h["heat_usd"], h["n"]) == (100.0, 1)


def _fake_ohlcv(series: dict[str, pd.Series]):
    def get(sym, timeframe, cfg):
        if sym not in series:
            raise RuntimeError("no data")
        return pd.DataFrame({"close": series[sym]}), "fake"
    return get


def test_correlation_matrix_from_trailing_daily_returns(monkeypatch, cfg):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=120, freq="D", tz="UTC")
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))
    other = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))
    series = {"A": pd.Series(base, idx), "B": pd.Series(base * 2, idx),     # identical returns
              "C": pd.Series(other, idx), "D": pd.Series(base[:10], idx[:10])}
    monkeypatch.setattr(prisk.prices_mod, "get_ohlcv", _fake_ohlcv(series))

    rets = prisk.daily_returns(["A", "B", "C", "D", "ZZZ"], cfg, lookback_days=60)
    assert list(rets.columns) == ["A", "B", "C"]      # D too short, ZZZ unavailable
    assert len(rets) == 60

    c = prisk.correlation_matrix(["A", "B", "C"], cfg, 60)
    assert c.loc["A", "B"] == pytest.approx(1.0)
    assert abs(c.loc["A", "C"]) < 0.5


def test_correlation_matrix_is_empty_without_two_usable_series(monkeypatch, cfg):
    monkeypatch.setattr(prisk.prices_mod, "get_ohlcv", _fake_ohlcv({}))
    assert prisk.correlation_matrix(["A", "B"], cfg).empty
