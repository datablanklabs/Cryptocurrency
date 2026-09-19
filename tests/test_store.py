"""SQLite round-trips: the new prices_daily table, paper-book accounting, the
equity-snapshot key, and the trailing high-water mark."""

from __future__ import annotations

import pandas as pd
import pytest

from cryptoyolo.store import Store, iso


def test_prices_daily_round_trip_and_upsert(store: Store, make_series):
    s = make_series([100, 101, 102, 103, 104], start="2026-02-01")
    n = store.upsert_daily_prices_from_series("BTC", s, "kraken")
    assert n == 5

    got = store.daily_prices("BTC")
    assert list(got["close"]) == [100, 101, 102, 103, 104]
    assert str(got["date"].dt.tz) == "UTC"

    # re-upsert with changed values -> updated in place, not duplicated
    store.upsert_daily_prices_from_series("BTC", make_series([9, 9, 9, 9, 9],
                                                            start="2026-02-01"), "coinbase")
    got2 = store.daily_prices("BTC")
    assert len(got2) == 5
    assert set(got2["close"]) == {9.0}

    cov = store.daily_price_coverage()
    assert cov.loc[cov["symbol"] == "BTC", "days"].iloc[0] == 5


def test_prices_daily_collapses_intraday_to_one_row_per_date(store: Store):
    idx = pd.to_datetime(["2026-03-01T01:00", "2026-03-01T23:00",
                          "2026-03-02T05:00"], utc=True)
    s = pd.Series([10.0, 11.0, 12.0], index=idx)
    store.upsert_daily_prices_from_series("ETH", s, "binance")
    got = store.daily_prices("ETH")
    assert list(got["close"]) == [11.0, 12.0]        # last close of Mar 1 wins


def test_paper_fill_accounting(store: Store):
    start = store.paper_cash()
    store.apply_paper_fill("BTC", "BUY", 2.0, 100.0, commission=0.5)
    assert store.paper_cash() == pytest.approx(start - 200.0 - 0.5)
    pos = store.paper_positions()
    assert pos.loc[pos["symbol"] == "BTC", "qty"].iloc[0] == 2.0
    assert pos.loc[pos["symbol"] == "BTC", "avg_price"].iloc[0] == 100.0

    store.apply_paper_fill("BTC", "SELL", 2.0, 120.0, commission=0.6)
    assert store.paper_positions().empty                       # flat -> row removed
    assert store.paper_cash() == pytest.approx(start - 200.0 - 0.5 + 240.0 - 0.6)


def test_equity_snapshot_key_is_ts_plus_mode(store: Store, monkeypatch):
    # pin the timestamp so both writes collide on the (ts, mode) primary key
    monkeypatch.setattr("cryptoyolo.store.iso",
                        lambda *a, **k: "2026-08-30T12:00:00+00:00")
    store.record_equity("r1", 10_000.0, 0.0, 10_000.0, mode="paper")
    store.record_equity("r2", 9_000.0, 500.0, 9_500.0, mode="paper")
    curve = store.equity_curve("paper")
    assert len(curve) == 1                          # same (ts, mode) -> replaced
    assert curve["equity"].iloc[-1] == 9_500.0
    store.record_equity("r3", 1.0, 0.0, 1.0, mode="binance")
    assert len(store.equity_curve()) == 2           # a different mode is its own row


def test_high_water_only_ratchets_up(store: Store):
    store.upsert_position_meta({
        "symbol": "SOL", "opened_at": iso(), "entry_price": 100.0,
        "stop": 90.0, "target": 130.0, "horizon_days": 30.0,
        "high_water": 100.0, "proposal_id": "p", "mode": "paper",
    })
    assert store.bump_high_water("SOL", 115.0) == 115.0
    assert store.bump_high_water("SOL", 108.0) == 115.0        # a dip never lowers it
    assert store.bump_high_water("SOL", 120.0) == 120.0


def test_macro_markets_latest_is_newest_row_per_ticker(store: Store):
    store.upsert_macro([
        {"ticker": "KXFED-1", "series_ticker": "KXFED", "label": "Fed cuts",
         "title": "t1", "probability": 0.5, "volume": 10, "close_time": "c1",
         "fetched_at": "2026-01-01T00:00:00+00:00"},
        {"ticker": "KXFED-1", "series_ticker": "KXFED", "label": "Fed cuts",
         "title": "t1", "probability": 0.6, "volume": 20, "close_time": "c1",
         "fetched_at": "2026-01-02T00:00:00+00:00"},
    ])
    latest = store.macro_latest()
    assert len(latest) == 1
    assert latest["probability"].iloc[0] == 0.6      # newest fetch wins

    hist = store.macro_history("KXFED")
    assert len(hist) == 2                            # full history is kept, not overwritten


def test_macro_latest_drops_rows_older_than_since(store: Store):
    from datetime import datetime, timezone

    store.upsert_macro([
        {"ticker": "KXOLD-1", "series_ticker": "KXOLD", "label": "old",
         "title": "", "probability": 0.5, "volume": 5, "close_time": "",
         "fetched_at": "2020-01-01T00:00:00+00:00"},
    ])
    assert store.macro_latest(datetime(2025, 1, 1, tzinfo=timezone.utc)).empty
    assert not store.macro_latest().empty
