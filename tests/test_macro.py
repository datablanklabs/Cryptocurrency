"""Feature 6: Kalshi request signing, snapshot fetch/filter, and the blended
macro score. No network — the signed client is exercised against a throwaway
RSA keypair, and `fetch()` is exercised against a fake client."""

from __future__ import annotations

import base64
from datetime import timedelta

import pandas as pd
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


def test_client_unconfigured_when_key_path_has_bad_tilde(monkeypatch):
    """A "~~/" typo makes expanduser() raise RuntimeError, not OSError — it
    must read as "not configured", not crash every fetch."""
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", "~~/nowhere/kalshi.key")
    monkeypatch.delenv("KALSHI_PRIVATE_KEY", raising=False)
    assert not macro.KalshiClient(Config()).configured


def test_client_prefers_key_file_and_survives_a_bad_inline_key(tmp_path, monkeypatch):
    key_path, _ = _write_keypair(tmp_path)
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key_path))
    monkeypatch.setenv("KALSHI_PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----\\ngarbage")
    assert macro.KalshiClient(Config()).configured


def test_client_accepts_inline_key_with_escaped_newlines(tmp_path, monkeypatch):
    key_path, _ = _write_keypair(tmp_path)
    one_line = key_path.read_text().strip().replace("\n", "\\n")
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY", one_line)
    assert macro.KalshiClient(Config()).configured


def test_credential_status_reports_unloadable_kalshi_key_as_unconfigured(monkeypatch):
    from cryptoyolo.config import credential_status
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY", "not a pem")
    assert credential_status()["kalshi"] is False


def test_client_unconfigured_without_credentials(monkeypatch):
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY", raising=False)
    client = macro.KalshiClient(Config())
    assert not client.configured
    with pytest.raises(macro.KalshiError):
        client.get("/markets")


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return {}


class _AlwaysRateLimitedSession:
    def get(self, *a, **k):
        return _FakeResponse(429)


def test_client_surfaces_rate_limit_reason_when_every_retry_is_429(tmp_path, monkeypatch):
    """Regression guard: the 429 branch used to `continue` without recording
    `last`, so exhausting every retry on a rate limit raised KalshiError with
    the real cause silently swapped for "None"."""
    key_path, _ = _write_keypair(tmp_path)
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key_path))
    monkeypatch.setattr(macro.time, "sleep", lambda *_a: None)   # skip real backoff delays

    cfg = Config()
    cfg.http_retries = 2
    client = macro.KalshiClient(cfg)
    client.session = _AlwaysRateLimitedSession()

    with pytest.raises(macro.KalshiError, match="429"):
        client.get("/markets")


# --------------------------------------------------------------------------
# _market_probability
# --------------------------------------------------------------------------
def test_probability_prefers_bid_ask_midpoint():
    m = {"yes_bid_dollars": "0.4000", "yes_ask_dollars": "0.4400",
         "last_price_dollars": "0.1000"}
    assert macro._market_probability(m) == pytest.approx(0.42)


def test_probability_falls_back_to_last_price():
    m = {"yes_bid_dollars": "0.0000", "yes_ask_dollars": "0.0000",
         "last_price_dollars": "0.6300"}
    assert macro._market_probability(m) == pytest.approx(0.63)


def test_probability_reads_legacy_cent_fields():
    assert macro._market_probability({"yes_bid": 40, "yes_ask": 44}) == pytest.approx(0.42)
    assert macro._market_probability({"last_price": 63}) == pytest.approx(0.63)


def test_volume_reads_current_then_legacy_field():
    assert macro._market_volume({"volume_fp": "473118.39"}) == 473118
    assert macro._market_volume({"volume": 50}) == 50
    assert macro._market_volume({}) == 0


def test_probability_none_when_nothing_available():
    assert macro._market_probability({}) is None


# --------------------------------------------------------------------------
# fetch()
# --------------------------------------------------------------------------
def _mkt(ticker, bid, ask, volume, close, event=None, last="0.0000"):
    """A market dict in the shape Kalshi's API returns today."""
    return {"ticker": ticker, "event_ticker": event or ticker.rsplit("-", 1)[0],
            "yes_bid_dollars": bid, "yes_ask_dollars": ask,
            "last_price_dollars": last, "volume_fp": volume,
            "close_time": close, "title": ticker}


class _FakeClient:
    """Stands in for KalshiClient: same .get(endpoint, params) shape, serving
    each series `page_size` markets at a time behind a cursor."""

    def __init__(self, markets_by_series, page_size=1000):
        self._markets = markets_by_series
        self._page = page_size
        self.configured = True

    def get(self, endpoint, params=None):
        assert endpoint == "/markets"
        rows = self._markets.get(params["series_ticker"], [])
        start = int(params.get("cursor") or 0)
        end = start + self._page
        return {"markets": rows[start:end],
                "cursor": str(end) if end < len(rows) else ""}


def test_fetch_stores_snapshots_and_drops_thin_markets(store: Store, monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES",
                        (MacroSeries("Fed cuts", "KXFED", +1, 1.0),))
    fake = _FakeClient({
        "KXFED": [
            _mkt("KXFED-26OCT-1", "0.6000", "0.6400", "500.00", "2026-10-01T00:00:00Z"),
            # no bid/ask, zero volume -> filtered out by min_volume, not a probability bug
            _mkt("KXFED-26OCT-2", "0.0000", "0.0000", "0.00", "2026-10-01T00:00:00Z"),
        ],
    })
    monkeypatch.setattr(macro, "KalshiClient", lambda cfg: fake)

    n = macro.fetch(store, Config(), verbose=False)
    assert n == 1
    snap = store.macro_latest()
    assert list(snap["ticker"]) == ["KXFED-26OCT-1"]
    assert snap["probability"].iloc[0] == pytest.approx(0.62)


def test_fetch_reads_only_the_nearest_event_across_pages(store: Store, monkeypatch):
    """Kalshi doesn't return markets in expiry order: the nearest event can sit
    on a later page. fetch() must page through and pick it, not whatever came
    back first."""
    monkeypatch.setattr(macro, "MACRO_SERIES",
                        (MacroSeries("Recession", "KXREC", -1, 1.0),))
    fake = _FakeClient({"KXREC": [
        _mkt("KXREC-27", event="KXREC-27", bid="0.2200", ask="0.2300", volume="1000", close="2028-01-31T00:00:00Z"),
        _mkt("KXREC-28", event="KXREC-28", bid="0.3000", ask="0.3200", volume="1000", close="2029-01-31T00:00:00Z"),
        _mkt("KXREC-26", event="KXREC-26", bid="0.0500", ask="0.0600", volume="1000", close="2027-01-31T00:00:00Z"),
    ]}, page_size=2)
    monkeypatch.setattr(macro, "KalshiClient", lambda cfg: fake)

    assert macro.fetch(store, Config(), verbose=False) == 1
    snap = store.macro_latest()
    assert list(snap["ticker"]) == ["KXREC-26"]
    assert snap["probability"].iloc[0] == pytest.approx(0.055)


def test_fetch_with_outcomes_keeps_only_the_named_markets(store: Store, monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES", (
        MacroSeries("Fed hikes", "KXFEDDECISION", -1, 1.0, outcomes=("H25", "H26")),))
    close = "2026-10-28T17:59:00Z"
    fake = _FakeClient({"KXFEDDECISION": [
        _mkt("KXFEDDECISION-26OCT-H0", "0.4700", "0.4800", "700000", close),
        _mkt("KXFEDDECISION-26OCT-H25", "0.5100", "0.5200", "570000", close),
        _mkt("KXFEDDECISION-26OCT-H26", "0.0000", "0.0100", "170000", close),
        _mkt("KXFEDDECISION-26OCT-C25", "0.0000", "0.0100", "510000", close),
        # a later meeting's hike market must not leak in
        _mkt("KXFEDDECISION-26DEC-H25", "0.3000", "0.3100", "90000", "2026-12-09T19:00:00Z"),
    ]})
    monkeypatch.setattr(macro, "KalshiClient", lambda cfg: fake)

    assert macro.fetch(store, Config(), verbose=False) == 2
    assert sorted(store.macro_latest()["ticker"]) == [
        "KXFEDDECISION-26OCT-H25", "KXFEDDECISION-26OCT-H26"]


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
            return {"markets": [_mkt("KXCPI-1", "0.2000", "0.2400", "50", "2026-10-10")]}

    monkeypatch.setattr(macro, "KalshiClient", lambda cfg: _PartiallyBrokenClient())
    n = macro.fetch(store, Config(), verbose=False)
    assert n == 1
    assert list(store.macro_latest()["ticker"]) == ["KXCPI-1"]


# --------------------------------------------------------------------------
# score()
# --------------------------------------------------------------------------
def _snap(store, readings):
    """Store one fetch: {series_ticker: probability}."""
    fetched = iso()
    store.upsert_macro([
        {"ticker": f"{t}-1", "series_ticker": t, "label": t, "title": "",
         "probability": p, "volume": 100, "close_time": "2026-10-01",
         "fetched_at": fetched}
        for t, p in readings.items()
    ])


def test_score_measures_each_series_against_its_baseline(store: Store, monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES", (
        MacroSeries("Fed hikes", "KXFED", -1, 1.0, baseline=0.15),
        MacroSeries("CPI hot", "KXCPI", -1, 2.0, baseline=0.20),
    ))
    _snap(store, {"KXFED": 0.575, "KXCPI": 0.10})
    result = macro.score(store, Config())
    # Fed:  (0.575-0.15)/(1-0.15) = 0.5 above normal -> -0.5 * 1.0
    # CPI:  below its 20% normal -> 0 (no credit for being calm)
    # blend: -0.5 / (1.0 + 2.0)
    assert result["score"] == pytest.approx(-0.5 / 3.0, abs=1e-4)
    assert result["n_series"] == 2
    assert result["top_label"] == "Fed hikes"
    assert "vs normal 15%" in result["note"]


def test_score_ordinary_low_readings_are_neutral_not_bullish(store: Store, monkeypatch):
    """The old coin-flip baseline scored a 5.5% recession chance as +0.71."""
    monkeypatch.setattr(macro, "MACRO_SERIES", (
        MacroSeries("Recession", "KXREC", -1, 0.8, baseline=0.15),
        MacroSeries("CPI hot", "KXCPI", -1, 0.8, baseline=0.20),
    ))
    _snap(store, {"KXREC": 0.055, "KXCPI": 0.075})
    result = macro.score(store, Config())
    assert result["score"] == 0.0
    assert "at or better than normal" in result["note"]


def test_score_calm_series_cannot_cancel_an_alarming_one(store: Store, monkeypatch):
    """The review's scenario: scored +0.05 against a 50% baseline."""
    monkeypatch.setattr(macro, "MACRO_SERIES", (
        MacroSeries("Fed hikes", "KXFED", -1, 1.0, baseline=0.15),
        MacroSeries("CPI hot", "KXCPI", -1, 0.8, baseline=0.20),
        MacroSeries("Recession", "KXREC", -1, 0.8, baseline=0.15),
    ))
    _snap(store, {"KXFED": 0.03, "KXCPI": 0.90, "KXREC": 0.60})
    result = macro.score(store, Config())
    cpi = (0.90 - 0.20) / 0.80 * 0.8
    rec = (0.60 - 0.15) / 0.85 * 0.8
    assert result["score"] == pytest.approx(-(cpi + rec) / 2.6, abs=1e-4)
    assert result["score"] < -0.3


def test_score_direction_plus_one_counts_the_no_side(store: Store, monkeypatch):
    # "Fed cuts" is good news; normally 60% likely. 24% means NO is at 76%
    # vs a normal 40%: (0.76-0.40)/(1-0.40) = 0.6 worse than normal.
    monkeypatch.setattr(macro, "MACRO_SERIES",
                        (MacroSeries("Fed cuts", "KXCUT", +1, 1.0, baseline=0.60),))
    _snap(store, {"KXCUT": 0.24})
    assert macro.score(store, Config())["score"] == pytest.approx(-0.6, abs=1e-4)


def test_macro_series_rejects_a_degenerate_baseline():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            MacroSeries("x", "X", baseline=bad)


def test_score_sums_outcome_markets(store: Store, monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES", (
        MacroSeries("Fed hikes", "KXFEDDECISION", -1, 1.0, outcomes=("H25", "H26")),))
    fetched = iso()
    store.upsert_macro([
        {"ticker": t, "series_ticker": "KXFEDDECISION", "label": "Fed hikes",
         "title": "", "probability": p, "volume": 100,
         "close_time": "2026-10-28", "fetched_at": fetched}
        for t, p in (("KXFEDDECISION-26OCT-H25", 0.60), ("KXFEDDECISION-26OCT-H26", 0.15))
    ])
    result = macro.score(store, Config())
    # P(any hike) = 0.75 -> -1 * (0.75-0.5)*2 = -0.5
    assert result["top_probability"] == pytest.approx(0.75)
    assert result["score"] == pytest.approx(-0.5)


def test_score_uses_only_the_latest_fetch_after_an_event_rolls(store: Store, monkeypatch):
    """Once the October meeting settles, its rows are still inside the
    staleness window — they must not be averaged with December's."""
    monkeypatch.setattr(macro, "MACRO_SERIES",
                        (MacroSeries("Fed hikes", "KXFEDDECISION", -1, 1.0, outcomes=("H25",)),))
    old, new = iso(utcnow() - timedelta(hours=2)), iso()
    store.upsert_macro([
        {"ticker": "KXFEDDECISION-26OCT-H25", "series_ticker": "KXFEDDECISION",
         "label": "Fed hikes", "title": "", "probability": 0.90, "volume": 100,
         "close_time": "2026-10-28", "fetched_at": old},
        {"ticker": "KXFEDDECISION-26DEC-H25", "series_ticker": "KXFEDDECISION",
         "label": "Fed hikes", "title": "", "probability": 0.20, "volume": 100,
         "close_time": "2026-12-09", "fetched_at": new},
    ])
    result = macro.score(store, Config())
    assert result["top_probability"] == pytest.approx(0.20)
    assert result["n_markets"] == 1


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


def test_series_history_combines_like_score(monkeypatch):
    monkeypatch.setattr(macro, "MACRO_SERIES", (
        MacroSeries("Fed hikes", "KXFEDDECISION", -1, 1.0, outcomes=("H25", "H26")),))
    hist = pd.DataFrame([
        {"series_ticker": "KXFEDDECISION", "label": "Fed hikes", "fetched_at": "t1",
         "ticker": "KXFEDDECISION-26OCT-H25", "probability": 0.51},
        {"series_ticker": "KXFEDDECISION", "label": "Fed hikes", "fetched_at": "t1",
         "ticker": "KXFEDDECISION-26OCT-H26", "probability": 0.01},
    ])
    out = macro.series_history(hist)
    assert len(out) == 1
    assert out["probability"].iloc[0] == pytest.approx(0.52)
