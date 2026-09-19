"""Feature 6: Kalshi request signing, snapshot fetch/filter, and the blended
macro score. No network — the signed client is exercised against a throwaway
RSA keypair, and `fetch()` is exercised against a fake client."""

from __future__ import annotations

import base64
from datetime import timedelta

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from cryptoyolo import macro
from cryptoyolo.config import Config, MacroSeries
from cryptoyolo.store import Store, iso, utcnow


# --------------------------------------------------------------------------
# Signed client
# --------------------------------------------------------------------------
def _write_keypair(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path = tmp_path / "kalshi_test_key.pem"
    path.write_bytes(pem)
    return path, key.public_key()


def test_client_signs_the_documented_message(tmp_path, monkeypatch):
    key_path, public_key = _write_keypair(tmp_path)
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key_path))

    client = macro.KalshiClient(Config())
    assert client.configured

    headers = client._headers("GET", "/trade-api/v2/markets")
    assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"].isdigit()

    message = (headers["KALSHI-ACCESS-TIMESTAMP"] + "GET"
              + "/trade-api/v2/markets").encode("utf-8")
    sig = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    # Raises if this isn't a valid signature over exactly the documented
    # message — proves _headers signs what Kalshi's auth scheme expects.
    public_key.verify(
        sig, message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                   salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_client_unconfigured_without_credentials(monkeypatch):
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY", raising=False)
    client = macro.KalshiClient(Config())
    assert not client.configured
    with pytest.raises(macro.KalshiError):
        client.get("/markets")


# --------------------------------------------------------------------------
# _market_probability
# --------------------------------------------------------------------------
def test_probability_prefers_bid_ask_midpoint():
    m = {"yes_bid": 40, "yes_ask": 44, "last_price": 10}
    assert macro._market_probability(m) == pytest.approx(0.42)


def test_probability_falls_back_to_last_price():
    m = {"yes_bid": None, "yes_ask": None, "last_price": 63}
    assert macro._market_probability(m) == pytest.approx(0.63)


def test_probability_none_when_nothing_available():
    assert macro._market_probability({}) is None


# --------------------------------------------------------------------------
# fetch()
# --------------------------------------------------------------------------
class _FakeClient:
    """Stands in for KalshiClient: same .get(endpoint, params) shape."""

    def __init__(self, markets_by_series):
        self._markets = markets_by_series
        self.configured = True

    def get(self, endpoint, params=None):
        assert endpoint == "/markets"
        return {"markets": self._markets.get(params["series_ticker"], [])}


def test_fetch_stores_snapshots_and_drops_thin_markets(store: Store, monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES",
                        (MacroSeries("Fed cuts", "KXFED", +1, 1.0),))
    fake = _FakeClient({
        "KXFED": [
            {"ticker": "KXFED-1", "yes_bid": 60, "yes_ask": 64, "volume": 500,
             "close_time": "2026-10-01T00:00:00Z", "title": "Fed cuts in Oct"},
            # no bid/ask, zero volume -> filtered out by min_volume, not a probability bug
            {"ticker": "KXFED-2", "last_price": 0, "volume": 0,
             "close_time": "2026-11-01T00:00:00Z"},
        ],
    })
    monkeypatch.setattr(macro, "KalshiClient", lambda cfg: fake)

    n = macro.fetch(store, Config(), verbose=False)
    assert n == 1
    snap = store.macro_latest()
    assert list(snap["ticker"]) == ["KXFED-1"]
    assert snap["probability"].iloc[0] == pytest.approx(0.62)


def test_fetch_noop_when_disabled(store: Store):
    cfg = Config()
    cfg.macro.enabled = False
    assert macro.fetch(store, cfg, verbose=False) == 0


def test_fetch_noop_when_series_empty(store: Store, monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES", ())
    assert macro.fetch(store, Config(), verbose=False) == 0


def test_fetch_noop_without_credentials(store: Store, monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES", (MacroSeries("x", "X"),))
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY", raising=False)
    assert macro.fetch(store, Config(), verbose=False) == 0


def test_fetch_one_series_failing_does_not_sink_the_others(store: Store, monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES", (
        MacroSeries("Fed cuts", "KXFED", +1, 1.0),
        MacroSeries("CPI hot", "KXCPI", -1, 1.0),
    ))

    class _PartiallyBrokenClient:
        configured = True

        def get(self, endpoint, params=None):
            if params["series_ticker"] == "KXFED":
                raise RuntimeError("rate limited")
            return {"markets": [{"ticker": "KXCPI-1", "yes_bid": 20, "yes_ask": 24,
                                 "volume": 50, "close_time": "2026-10-10"}]}

    monkeypatch.setattr(macro, "KalshiClient", lambda cfg: _PartiallyBrokenClient())
    n = macro.fetch(store, Config(), verbose=False)
    assert n == 1
    assert list(store.macro_latest()["ticker"]) == ["KXCPI-1"]


# --------------------------------------------------------------------------
# score()
# --------------------------------------------------------------------------
def test_score_is_the_weight_normalised_blend(store: Store, monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES", (
        MacroSeries("Fed cuts", "KXFED", +1, 1.0),
        MacroSeries("CPI hot", "KXCPI", -1, 2.0),
    ))
    fetched = iso()
    store.upsert_macro([
        {"ticker": "KXFED-1", "series_ticker": "KXFED", "label": "Fed cuts",
         "title": "", "probability": 0.75, "volume": 100,
         "close_time": "2026-10-01", "fetched_at": fetched},
        {"ticker": "KXCPI-1", "series_ticker": "KXCPI", "label": "CPI hot",
         "title": "", "probability": 0.20, "volume": 100,
         "close_time": "2026-10-10", "fetched_at": fetched},
    ])
    result = macro.score(store, Config())
    # Fed:  +1 * (0.75-0.5)*2 * 1.0 =  0.5
    # CPI:  -1 * (0.20-0.5)*2 * 2.0 =  1.2
    # blend: (0.5 + 1.2) / (1.0 + 2.0)
    assert result["score"] == pytest.approx((0.5 + 1.2) / 3.0, abs=1e-4)
    assert result["n_series"] == 2


def test_score_direction_zero_reads_any_extreme_as_bearish(store: Store, monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES",
                        (MacroSeries("Uncertainty index", "KXVOL", 0, 1.0),))
    store.upsert_macro([{"ticker": "KXVOL-1", "series_ticker": "KXVOL",
                         "label": "Uncertainty index", "title": "",
                         "probability": 0.95, "volume": 10,
                         "close_time": "x", "fetched_at": iso()}])
    assert macro.score(store, Config())["score"] < 0


def test_score_ignores_stale_snapshot(store: Store, monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES",
                        (MacroSeries("Fed cuts", "KXFED", +1, 1.0),))
    cfg = Config()
    cfg.macro.stale_after_hours = 1.0
    old = iso(utcnow() - timedelta(hours=5))
    store.upsert_macro([{"ticker": "KXFED-1", "series_ticker": "KXFED",
                         "label": "Fed cuts", "title": "", "probability": 0.9,
                         "volume": 10, "close_time": "x", "fetched_at": old}])
    result = macro.score(store, cfg)
    assert result["n_series"] == 0
    assert "no Kalshi snapshot" in result["note"]


def test_score_reports_no_data_when_series_empty(store: Store, monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES", ())
    result = macro.score(store, Config())
    assert result["n_series"] == 0
    assert "MACRO_SERIES is empty" in result["note"]


def test_describe_smoke():
    assert "OFF" in macro.describe({"enabled": False})
    assert "n/a" in macro.describe({"enabled": True, "n_series": 0, "note": "x"})
    assert "+0.50" in macro.describe(
        {"enabled": True, "n_series": 1, "score": 0.5, "note": "x"})
