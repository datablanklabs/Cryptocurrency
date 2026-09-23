"""get_ohlcv's cache must report the source that produced the frame."""

from __future__ import annotations

import pandas as pd

from cryptoyolo import prices


def test_cache_hit_reports_the_original_source(cfg, monkeypatch):
    calls = []

    def _fake(symbol, interval, start, cfg):
        calls.append(symbol)
        idx = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
        return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                             "close": 1.0, "volume": 1.0}, index=idx)

    monkeypatch.setattr(prices, "_cache", {})
    monkeypatch.setitem(prices.FETCHERS, "yfinance", _fake)
    cfg.price_source_order = ("yfinance",)
    _, first = prices.get_ohlcv("BTC", "1d", cfg)
    _, second = prices.get_ohlcv("BTC", "1d", cfg)
    assert first == second == "yfinance"
    assert len(calls) == 1                  # the second call really was a cache hit
