"""Feature 7: Kalshi crypto price-market fetch/filter and the per-asset
interpolated score. No network — `fetch()` is exercised against a fake
client, same shape as test_macro.py."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from cryptoyolo import kalshi_prediction
from cryptoyolo.config import Config, KalshiCryptoSeries
from cryptoyolo.store import Store, iso, utcnow


# --------------------------------------------------------------------------
# _strike
# --------------------------------------------------------------------------
def test_strike_prefers_floor_strike_field():
    assert kalshi_prediction._strike({"floor_strike": 110000}) == pytest.approx(110000.0)


def test_strike_falls_back_to_parsing_subtitle():
    m = {"subtitle": "$110,000 or above", "title": "ignored"}
    assert kalshi_prediction._strike(m) == pytest.approx(110000.0)


def test_strike_none_when_unparseable():
    assert kalshi_prediction._strike({"title": "no dollar figure here"}) is None
    assert kalshi_prediction._strike({}) is None


# --------------------------------------------------------------------------
# fetch()
# --------------------------------------------------------------------------
class _FakeClient:
    """Stands in for macro.KalshiClient: same .get(endpoint, params) shape."""

    def __init__(self, markets_by_series):
        self._markets = markets_by_series
        self.configured = True

    def get(self, endpoint, params=None):
        assert endpoint == "/markets"
        return {"markets": self._markets.get(params["series_ticker"], [])}


def test_fetch_stores_ladder_and_drops_thin_and_range_markets(store: Store, monkeypatch):
    monkeypatch.setattr(kalshi_prediction, "KALSHI_CRYPTO_SERIES",
                        (KalshiCryptoSeries("BTC", "KXBTCD"),))
    fake = _FakeClient({
        "KXBTCD": [
            {"ticker": "KXBTCD-100K", "event_ticker": "KXBTCD-26OCT01", "floor_strike": 100000, "yes_bid": 70,
             "yes_ask": 74, "volume": 500, "close_time": "2026-10-01T00:00:00Z"},
            {"ticker": "KXBTCD-120K", "event_ticker": "KXBTCD-26OCT01", "floor_strike": 120000, "yes_bid": 20,
             "yes_ask": 24, "volume": 300, "close_time": "2026-10-01T00:00:00Z"},
            # zero volume -> filtered by min_volume, not a probability bug
            {"ticker": "KXBTCD-THIN", "event_ticker": "KXBTCD-26OCT01", "floor_strike": 150000, "last_price": 5,
             "volume": 0, "close_time": "2026-10-01T00:00:00Z"},
            # a "between" range bucket -> not modelled, must be skipped
            {"ticker": "KXBTCD-RANGE", "event_ticker": "KXBTCD-26OCT01", "strike_type": "between",
             "floor_strike": 100000, "cap_strike": 110000, "yes_bid": 40,
             "yes_ask": 44, "volume": 200, "close_time": "2026-10-01T00:00:00Z"},
        ],
    })
    monkeypatch.setattr(kalshi_prediction.macro_mod, "KalshiClient", lambda cfg: fake)

    n = kalshi_prediction.fetch(store, Config(), verbose=False)
    assert n == 2
    ladder = store.kalshi_prediction_latest()
    assert set(ladder["ticker"]) == {"KXBTCD-100K", "KXBTCD-120K"}
    assert set(ladder["symbol"]) == {"BTC"}


def test_fetch_keeps_only_the_nearest_expiry_and_null_strike_type(store: Store, monkeypatch):
    monkeypatch.setattr(kalshi_prediction, "KALSHI_CRYPTO_SERIES",
                        (KalshiCryptoSeries("BTC", "KXBTCD"),))
    later = {"event_ticker": "KXBTCD-26OCT08", "close_time": "2026-10-08T00:00:00Z"}
    soon = {"event_ticker": "KXBTCD-26OCT01", "close_time": "2026-10-01T00:00:00Z"}
    fake = _FakeClient({"KXBTCD": [
        {"ticker": "KXBTCD-26OCT08-100K", "floor_strike": 100000, "yes_bid": 60,
         "yes_ask": 64, "volume": 10, **later},
        # JSON null strike_type on a plain "above" market must be kept
        {"ticker": "KXBTCD-26OCT01-100K", "strike_type": None, "floor_strike": 100000,
         "yes_bid": 70, "yes_ask": 74, "volume": 10, **soon},
        {"ticker": "KXBTCD-26OCT01-120K", "floor_strike": 120000, "yes_bid": 20,
         "yes_ask": 24, "volume": 10, **soon},
    ]})
    monkeypatch.setattr(kalshi_prediction.macro_mod, "KalshiClient", lambda cfg: fake)

    assert kalshi_prediction.fetch(store, Config(), verbose=False) == 2
    assert set(store.kalshi_prediction_latest()["ticker"]) == {
        "KXBTCD-26OCT01-100K", "KXBTCD-26OCT01-120K"}


def test_fetch_noop_when_disabled(store: Store):
    cfg = Config()
    cfg.kalshi_prediction.enabled = False
    assert kalshi_prediction.fetch(store, cfg, verbose=False) == 0


def test_fetch_noop_when_series_empty(store: Store, monkeypatch):
    monkeypatch.setattr(kalshi_prediction, "KALSHI_CRYPTO_SERIES", ())
    assert kalshi_prediction.fetch(store, Config(), verbose=False) == 0


def test_fetch_noop_without_credentials(store: Store, monkeypatch):
    monkeypatch.setattr(kalshi_prediction, "KALSHI_CRYPTO_SERIES",
                        (KalshiCryptoSeries("BTC", "KXBTCD"),))
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY", raising=False)
    assert kalshi_prediction.fetch(store, Config(), verbose=False) == 0


def test_fetch_one_series_failing_does_not_sink_the_others(store: Store, monkeypatch):
    monkeypatch.setattr(kalshi_prediction, "KALSHI_CRYPTO_SERIES", (
        KalshiCryptoSeries("BTC", "KXBTCD"),
        KalshiCryptoSeries("ETH", "KXETHD"),
    ))

    class _PartiallyBrokenClient:
        configured = True

        def get(self, endpoint, params=None):
            if params["series_ticker"] == "KXBTCD":
                raise RuntimeError("rate limited")
            return {"markets": [{"ticker": "KXETHD-4K", "floor_strike": 4000,
                                 "yes_bid": 50, "yes_ask": 54, "volume": 50,
                                 "close_time": "2026-10-10"}]}

    monkeypatch.setattr(kalshi_prediction.macro_mod, "KalshiClient",
                        lambda cfg: _PartiallyBrokenClient())
    n = kalshi_prediction.fetch(store, Config(), verbose=False)
    assert n == 1
    assert list(store.kalshi_prediction_latest()["ticker"]) == ["KXETHD-4K"]


# --------------------------------------------------------------------------
# score_one()
# --------------------------------------------------------------------------
def _ladder(rows):
    return pd.DataFrame(rows)


def test_score_one_interpolates_p_up_at_spot():
    # 100k strike -> P(above)=0.70 ; 120k strike -> P(above)=0.20
    ladder = _ladder([
        {"symbol": "BTC", "strike": 100_000.0, "probability": 0.70},
        {"symbol": "BTC", "strike": 120_000.0, "probability": 0.20},
    ])
    # halfway between the strikes -> halfway between the probabilities
    out = kalshi_prediction.score_one("BTC", 110_000.0, ladder, Config())
    assert out["implied_p_up"] == pytest.approx(0.45, abs=1e-6)
    # deviation = (0.45 - 0.5) * 2 = -0.10
    assert out["kalshi_prediction"] == pytest.approx(-0.10, abs=1e-6)
    assert out["n_strikes"] == 2


def test_score_one_has_no_opinion_when_spot_is_outside_the_strikes():
    """A ladder entirely above spot only bounds P(up) from below — reading its
    edge probability as the answer would be a confident, unpriced number."""
    ladder = _ladder([
        {"symbol": "BTC", "strike": 130_000.0, "probability": 0.10},
        {"symbol": "BTC", "strike": 140_000.0, "probability": 0.03},
    ])
    for spot in (110_000.0, 500_000.0):
        out = kalshi_prediction.score_one("BTC", spot, ladder, Config())
        assert out["kalshi_prediction"] == 0.0
        assert out["implied_p_up"] is None
        assert "outside the open strikes" in out["note"]


def test_score_one_nan_spot_is_no_spot_not_max_bullish():
    ladder = _ladder([
        {"symbol": "BTC", "strike": 100_000.0, "probability": 0.70},
        {"symbol": "BTC", "strike": 120_000.0, "probability": 0.20},
    ])
    out = kalshi_prediction.score_one("BTC", float("nan"), ladder, Config())
    assert out["kalshi_prediction"] == 0.0
    assert "no spot price" in out["note"]


def test_score_one_needs_at_least_two_strikes():
    ladder = _ladder([{"symbol": "BTC", "strike": 100_000.0, "probability": 0.70}])
    out = kalshi_prediction.score_one("BTC", 100_000.0, ladder, Config())
    assert out["kalshi_prediction"] == 0.0
    assert out["n_strikes"] == 1
    assert "need >= 2" in out["note"]


def test_score_one_no_ladder_for_symbol():
    ladder = _ladder([{"symbol": "ETH", "strike": 4_000.0, "probability": 0.5}])
    out = kalshi_prediction.score_one("BTC", 100_000.0, ladder, Config())
    assert out["kalshi_prediction"] == 0.0
    assert "no Kalshi price-market snapshot" in out["note"]


def test_score_one_no_spot_price():
    ladder = _ladder([
        {"symbol": "BTC", "strike": 100_000.0, "probability": 0.70},
        {"symbol": "BTC", "strike": 120_000.0, "probability": 0.20},
    ])
    out = kalshi_prediction.score_one("BTC", None, ladder, Config())
    assert out["kalshi_prediction"] == 0.0
    assert "no spot price" in out["note"]


def test_score_one_disabled():
    cfg = Config()
    cfg.kalshi_prediction.enabled = False
    out = kalshi_prediction.score_one("BTC", 100_000.0, pd.DataFrame(), cfg)
    assert out["kalshi_prediction"] == 0.0
    assert "disabled" in out["note"]


# --------------------------------------------------------------------------
# latest_ladder() / score_symbols()
# --------------------------------------------------------------------------
def test_latest_ladder_drops_stale_snapshots(store: Store):
    cfg = Config()
    cfg.kalshi_prediction.stale_after_hours = 1.0
    old = iso(utcnow() - timedelta(hours=5))
    store.upsert_kalshi_prediction([
        {"ticker": "KXBTCD-100K", "event_ticker": "KXBTCD-26OCT01", "series_ticker": "KXBTCD", "symbol": "BTC",
         "strike": 100_000.0, "probability": 0.7, "volume": 10,
         "close_time": "x", "fetched_at": old},
    ])
    assert kalshi_prediction.latest_ladder(store, cfg).empty


def _row(ticker, strike, prob, fetched_at, close_time="x"):
    return {"ticker": ticker, "series_ticker": "KXBTCD", "symbol": "BTC",
            "strike": strike, "probability": prob, "volume": 10,
            "close_time": close_time, "fetched_at": fetched_at}


def test_latest_ladder_keeps_only_the_latest_fetch(store: Store):
    """An earlier fetch's event (still inside the staleness window) must not
    be interpolated together with the current event's strikes."""
    old, new = iso(utcnow() - timedelta(hours=2)), iso()
    store.upsert_kalshi_prediction([
        _row("KXBTCD-A-110K", 110_000.0, 0.9, old),
        _row("KXBTCD-B-100K", 100_000.0, 0.7, new),
        _row("KXBTCD-B-120K", 120_000.0, 0.2, new),
    ])
    ladder = kalshi_prediction.latest_ladder(store, Config())
    assert set(ladder["ticker"]) == {"KXBTCD-B-100K", "KXBTCD-B-120K"}


def test_latest_ladder_drops_markets_already_closed(store: Store):
    now = iso()
    past = iso(utcnow() - timedelta(minutes=5))
    future = iso(utcnow() + timedelta(hours=3))
    store.upsert_kalshi_prediction([
        _row("KXBTCD-A-100K", 100_000.0, 0.7, now, close_time=past),
        _row("KXBTCD-A-120K", 120_000.0, 0.2, now, close_time=future),
    ])
    ladder = kalshi_prediction.latest_ladder(store, Config())
    assert list(ladder["ticker"]) == ["KXBTCD-A-120K"]


def test_score_symbols_wraps_latest_ladder_and_score_one(store: Store):
    fetched = iso()
    store.upsert_kalshi_prediction([
        {"ticker": "KXBTCD-100K", "event_ticker": "KXBTCD-26OCT01", "series_ticker": "KXBTCD", "symbol": "BTC",
         "strike": 100_000.0, "probability": 0.70, "volume": 10,
         "close_time": "x", "fetched_at": fetched},
        {"ticker": "KXBTCD-120K", "event_ticker": "KXBTCD-26OCT01", "series_ticker": "KXBTCD", "symbol": "BTC",
         "strike": 120_000.0, "probability": 0.20, "volume": 10,
         "close_time": "x", "fetched_at": fetched},
    ])
    df = kalshi_prediction.score_symbols(store, Config(), spot_prices={"BTC": 110_000.0})
    row = df[df["symbol"] == "BTC"].iloc[0]
    assert row["kalshi_prediction"] == pytest.approx(-0.10, abs=1e-6)
    # an asset with no ladder and no spot still gets a (neutral) row
    assert (df["symbol"] == "ETH").any()


def test_describe_smoke():
    assert "n/a" in kalshi_prediction.describe({"n_strikes": 0, "note": "x"})
    assert "+0.50" in kalshi_prediction.describe(
        {"n_strikes": 2, "kalshi_prediction": 0.5, "note": "x"})
