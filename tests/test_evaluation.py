"""The feedback loop: forward-return snapping, FIFO trade reconstruction, the
information coefficient, and — the point of item 4 — that `forward_returns`
reads prices from the local table and needs no network."""

from __future__ import annotations

import uuid

import pandas as pd
import pytest

from cryptoyolo import evaluation


@pytest.fixture(autouse=True)
def _clear_price_cache():
    evaluation.clear_cache()
    yield
    evaluation.clear_cache()


# --------------------------------------------------------------------------
# _fwd_return: entry snaps FORWARD to the next bar, never back (no hindsight)
# --------------------------------------------------------------------------
def test_fwd_return_snaps_to_the_next_bar_after_t0(make_series):
    s = make_series([100, 101, 102, 103, 104, 105], start="2026-01-01")
    # t0 sits exactly on the Jan-2 bar; the entry must be the NEXT close (Jan 3,
    # 102), not Jan 2's 101 that `asof(t0)` would fold in as hindsight.
    t0 = pd.Timestamp("2026-01-02T00:00", tz="UTC")
    got = evaluation._fwd_return(s, t0, 1)
    assert got == pytest.approx(103 / 102 - 1)
    assert got != pytest.approx(103 / 101 - 1)


def test_fwd_return_nan_when_exit_bar_has_not_printed(make_series):
    s = make_series([100, 101, 102, 103, 104], start="2026-01-01")
    assert pd.isna(evaluation._fwd_return(s, pd.Timestamp("2026-01-04T12:00", tz="UTC"), 30))
    assert pd.isna(evaluation._fwd_return(s, pd.Timestamp("2031-01-01", tz="UTC"), 1))
    assert pd.isna(evaluation._fwd_return(None, pd.Timestamp("2026-01-01", tz="UTC"), 1))


# --------------------------------------------------------------------------
# realized_trades: FIFO, fees, partials, paper/live separation
# --------------------------------------------------------------------------
def _order(store, *, side, qty, price, fee, ts, mode="paper", status="FILLED"):
    store.save_order({
        "order_id": f"o-{uuid.uuid4().hex[:10]}", "proposal_id": None,
        "run_id": "r", "ts": ts, "venue": mode, "mode": mode,
        "symbol": "AAA", "side": side, "order_type": "MARKET",
        "qty": qty, "price": price, "status": status,
        "exchange_ref": None, "fee_usd": fee, "response": None,
    })


def test_realized_trades_simple_round_trip_net_of_fees(store, cfg):
    _order(store, side="BUY", qty=10, price=100.0, fee=4.0, ts="2026-01-01T00:00:00+00:00")
    _order(store, side="SELL", qty=10, price=120.0, fee=4.8, ts="2026-01-08T00:00:00+00:00")
    trades = evaluation.realized_trades(store, cfg)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["gross_pnl"] == pytest.approx(200.0)
    assert t["fees"] == pytest.approx(4.0 + 4.8)
    assert t["net_pnl"] == pytest.approx(191.2)
    assert t["ret_pct"] == pytest.approx(20.0)
    assert t["hold_days"] == pytest.approx(7.0, abs=0.01)

    stats = evaluation.trade_stats(trades)
    assert set(stats["book"]) == {"all", "paper"}
    assert stats.loc[stats["book"] == "all", "net_pnl"].iloc[0] == pytest.approx(191.2)


def test_realized_trades_fifo_partial_and_live_bucket(store, cfg, monkeypatch):
    monkeypatch.setattr("cryptoyolo.prices.latest_prices",
                        lambda syms, c: {"AAA": 125.0})
    _order(store, side="BUY", qty=10, price=100.0, fee=4.0, ts="2026-01-01T00:00:00+00:00")
    _order(store, side="BUY", qty=10, price=110.0, fee=4.4, ts="2026-01-02T00:00:00+00:00")
    _order(store, side="SELL", qty=15, price=130.0, fee=7.8, ts="2026-01-05T00:00:00+00:00")
    _order(store, side="BUY", qty=2, price=100.0, fee=0.8,
           ts="2026-01-03T00:00:00+00:00", mode="binance-live")

    trades = evaluation.realized_trades(store, cfg)
    closed = trades[trades["close_ts"].notna()].sort_values("entry")
    assert len(closed) == 2                                   # FIFO split across two lots
    assert list(closed["gross_pnl"].round(2)) == [300.0, 100.0]
    assert list(closed["qty"]) == [10, 5]
    # the untouched paper remainder and the live buy are open lots
    assert set(trades["mode"]) == {"paper", "live"}
    assert (trades["close_ts"].isna()).sum() == 2


# --------------------------------------------------------------------------
# information_coefficient / recommend_weights
# --------------------------------------------------------------------------
def _perfect_fr():
    return pd.DataFrame({
        "run_id": ["a"] * 5 + ["b"] * 5,
        "ts": pd.to_datetime(["2026-01-01"] * 5 + ["2026-01-02"] * 5, utc=True),
        "symbol": list("VWXYZ") * 2,
        "composite": [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1],
        "fwd_7d": [0.01, 0.02, 0.03, 0.04, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01],
    })


def test_information_coefficient_perfect_rank_correlation():
    ic = evaluation.information_coefficient(_perfect_fr(), (7,))
    comp = ic[(ic["family"] == "composite") & (ic["horizon_d"] == 7)].iloc[0]
    assert comp["mean_IC"] == pytest.approx(1.0, abs=1e-9)
    assert comp["pct_runs_IC_pos"] == 100.0


def test_recommend_weights_keeps_hand_set_when_evidence_is_thin():
    rec = evaluation.recommend_weights(_perfect_fr(), horizon=7)
    assert "suggested" not in rec           # not enough families cleared |t| >= 2
    assert rec.get("note")


# --------------------------------------------------------------------------
# item 4: forward_returns is deterministic and runs with no network
# --------------------------------------------------------------------------
def test_forward_returns_reads_stored_prices_and_never_hits_the_network(
        store, cfg, make_series, monkeypatch):
    series = make_series([100.0 + i for i in range(400)], start="2026-05-01")
    store.upsert_daily_prices_from_series("BTC", series, "test")

    def _no_network(*a, **k):
        raise AssertionError("evaluation must not fetch prices when the table is populated")
    monkeypatch.setattr("cryptoyolo.prices.get_ohlcv", _no_network)

    run_id = "20260830T120000-t"
    store.start_run(run_id, "paper")
    store.save_scores(run_id, [{
        "symbol": "BTC", "technical": 0.2, "social": 0.0, "catalyst": 0.0,
        "composite": 0.2, "positioning": 0.0, "events": 0.0, "xsec": 0.0,
        "components": {},
    }])

    fr = evaluation.forward_returns(store, cfg, horizons=(1, 7, 30))
    assert not fr.empty
    assert fr["fwd_1d"].notna().all()
    assert fr["fwd_7d"].notna().all()
    assert fr["fwd_30d"].notna().all()
    assert (fr["fwd_7d"] > 0).all()        # a rising ramp -> positive forward return
