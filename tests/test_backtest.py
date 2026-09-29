"""The price-family backtest: point-in-time slicing (no lookahead), that replay
scores with the live engine's own functions, and the portfolio simulation's
cost accounting. All offline, against a synthetic cache under tmp_path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoyolo import backtest, engine, indicators, positioning, signals
from cryptoyolo.backtest import INTERVAL_MS, BarCache

END = pd.Timestamp("2026-06-01T00:00", tz="UTC")


def _bars(n: int, interval: str, end: pd.Timestamp, seed: int) -> list[list]:
    """`n` random-walk candles ending at `end`, as kline rows (open-time ms)."""
    rng = np.random.default_rng(seed)
    iv = INTERVAL_MS[interval]
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    t0 = end.value // 1_000_000 - n * iv
    return [[t0 + i * iv, c, c * 1.002, c * 0.998, c, 10 + rng.random()]
            for i, c in enumerate(closes)]


@pytest.fixture
def cache(tmp_path) -> BarCache:
    c = BarCache(tmp_path / "bt.sqlite")
    for k, sym in enumerate(("AAA", "BBB", "CCC", "DDD")):
        c.upsert_bars(sym, "5m", _bars(288 * 30, "5m", END, k))
        c.upsert_bars(sym, "1h", _bars(24 * 60, "1h", END, 10 + k))
        c.upsert_bars(sym, "1d", _bars(400, "1d", END, 20 + k))
        t0 = END.value // 1_000_000 - 90 * 8 * 3_600_000
        c.upsert_funding(sym, [(t0 + i * 8 * 3_600_000, 0.0001 * np.sin(i + k))
                               for i in range(90 * 3)])
    return c


def _replay(cache, cfg, **kw):
    kw.setdefault("days", 20)
    kw.setdefault("symbols", ["AAA", "BBB", "CCC", "DDD"])
    return backtest.replay(cache, cfg, end=END.to_pydatetime(), verbose=False, **kw)


def test_view_uses_only_bars_closed_by_t(cache):
    f = backtest._load(cache, "AAA")
    t = pd.Timestamp("2026-05-20T13:00", tz="UTC")
    for name, (interval, lookback) in backtest.VIEWS.items():
        v = backtest._view_at(f, name, t)
        last_close = v.index[-1] + pd.Timedelta(milliseconds=INTERVAL_MS[interval])
        assert last_close <= t                       # nothing still forming at t
        assert v.index[0] >= t - lookback


def test_replay_has_no_lookahead(cache, cfg, tmp_path):
    """Rewriting every candle after t must not change any score AT t."""
    t = pd.Timestamp("2026-05-25T13:00", tz="UTC")
    before = _replay(cache, cfg, signals=signals.BUILTIN)
    with cache.conn() as con:
        con.execute("UPDATE bars SET close = close * 3, open = open * 3 WHERE ts >= ?",
                    (t.value // 1_000_000,))
        con.execute("UPDATE funding SET rate = rate + 0.01 WHERE funding_time > ?",
                    (t.value // 1_000_000,))
    after = _replay(cache, cfg, signals=signals.BUILTIN)
    cols = [*backtest.FAMILIES, *backtest.COMPONENTS, *signals.BUILTIN]
    b = before[before["ts"] <= t].set_index(["run_id", "symbol"])[cols]
    a = after[after["ts"] <= t].set_index(["run_id", "symbol"])[cols]
    assert len(b) > 0
    pd.testing.assert_frame_equal(a, b)
    # ...while the forward returns straddling t DO see the change.
    assert not before["fwd_1d"].equals(after["fwd_1d"])


def test_replay_scores_with_the_live_engine_functions(cache, cfg):
    t = pd.Timestamp("2026-05-25T13:00", tz="UTC")
    fr = _replay(cache, cfg)
    row = fr[(fr["ts"] == t) & (fr["symbol"] == "AAA")].iloc[0]

    f = backtest._load(cache, "AAA")
    short = indicators.summarize(indicators.enrich(backtest._view_at(f, "short", t), cfg.bands))
    medium = indicators.summarize(indicators.enrich(backtest._view_at(f, "medium", t), cfg.bands))
    tech_raw, parts = engine.technical_score(short, medium)
    assert row["technical_raw"] == pytest.approx(tech_raw, abs=1e-4)
    for c in backtest.COMPONENTS:                   # the pieces, unweighted
        assert row[c] == pytest.approx(parts[c[3:]], abs=1e-4)

    rates = f.funding_r[f.funding_t <= t.value // 1_000_000][-cfg.positioning.history_periods:]
    assert row["positioning"] == pytest.approx(positioning.score_rates(rates, cfg.positioning)[3],
                                               abs=1e-4)
    assert set(fr.columns) >= {"run_id", "ts", "symbol", *backtest.FAMILIES, "fwd_7d"}


def test_positioning_is_nan_before_funding_history_begins(cache, cfg):
    with cache.conn() as con:
        con.execute("DELETE FROM funding")
    fr = _replay(cache, cfg)
    assert fr["positioning"].isna().all()
    assert fr["composite"].notna().all()          # treated as 0, as a live run would


def test_score_rates_matches_the_live_positioning_score(store, cfg):
    rates = [0.0001, 0.0002, -0.0001] * 10 + [0.0006]
    store.upsert_funding([{"symbol": "BTC", "funding_time": i, "rate": r, "venue": "okx",
                           "fetched_at": "x"} for i, r in enumerate(rates)])
    live = positioning.score_symbols(store, cfg).set_index("symbol").loc["BTC", "positioning"]
    assert live == pytest.approx(positioning.score_rates(np.array(rates), cfg.positioning)[3],
                                 abs=1e-4)


def test_stablecoins_are_never_synced_or_replayed(cache, cfg):
    fr = backtest.replay(cache, cfg, days=5, end=END.to_pydatetime(),
                         symbols=["AAA", "BBB", "CCC", "USDC"], verbose=False)
    assert "USDC" not in set(fr["symbol"])


# --------------------------------------------------------------------------
# simulate
# --------------------------------------------------------------------------
def _fr(scores: dict[str, list[float]], rets: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    n = len(next(iter(scores.values())))
    for i in range(n):
        for sym in scores:
            rows.append({"run_id": f"bt-{i:03d}", "ts": pd.Timestamp("2026-01-01", tz="UTC")
                         + pd.Timedelta(days=i), "symbol": sym,
                         "composite": scores[sym][i], "fwd_1d": rets[sym][i]})
    return pd.DataFrame(rows)


def test_simulate_charges_cost_on_turnover_only(cfg):
    fr = _fr({"BTC": [0.5, 0.5, 0.5], "X": [0.1, 0.1, -0.9]},
             {"BTC": [0.01, 0.01, 0.01], "X": [0.0, 0.0, 0.0]})
    sim = backtest.simulate(fr, cfg, hold_days=1, top_n=1, floor=0.0, cost_bps=50)
    p = sim["periods"]
    assert list(p["picks"]) == ["BTC", "BTC", "BTC"]
    assert list(p["turnover"]) == [1.0, 0.0, 0.0]         # bought once, then held
    assert p["cost"].iloc[0] == pytest.approx(0.005)
    assert p["net"].iloc[1] == pytest.approx(0.01)


def test_simulate_leaves_slots_in_cash_below_the_floor(cfg):
    fr = _fr({"BTC": [-0.5], "X": [0.2]}, {"BTC": [0.05], "X": [0.10]})
    sim = backtest.simulate(fr, cfg, hold_days=1, top_n=2, floor=0.0, cost_bps=0)
    p = sim["periods"].iloc[0]
    assert p["picks"] == "X"
    assert p["gross"] == pytest.approx(0.05)              # half in X, half in cash
    assert p["basket"] == pytest.approx(0.075)
    assert p["btc"] == pytest.approx(0.05)


# --------------------------------------------------------------------------
# factor lab: candidate signals
# --------------------------------------------------------------------------
def test_candidate_signals_become_columns_and_a_bad_one_scores_nan(cache, cfg):
    def boom(ctx):
        raise RuntimeError("buggy idea")

    fr = _replay(cache, cfg, days=5,
                 signals={"rev": signals.reversal_1d, "boom": boom,
                          "sym_len": lambda ctx: len(ctx.symbol)})
    assert fr.attrs["signals"] == ["rev", "boom", "sym_len"]
    assert fr["rev"].notna().all()
    assert fr["boom"].isna().all()
    assert (fr["sym_len"] == 3).all()


def test_signal_context_holds_only_bars_closed_by_t(cache, cfg):
    seen = []

    def spy(ctx):
        seen.append((ctx.t, ctx.short.index[-1], ctx.medium.index[-1], ctx.daily.index[-1]))
        return 0.0

    _replay(cache, cfg, days=3, signals={"spy": spy})
    for t, *last_opens in seen:
        for last_open, interval in zip(last_opens, ("5m", "1h", "1d")):
            assert last_open + pd.Timedelta(milliseconds=INTERVAL_MS[interval]) <= t


def test_signal_specs_resolve():
    assert signals.load("low_vol") == ("low_vol", signals.low_vol)
    assert signals.load("vol=cryptoyolo.signals:low_vol") == ("vol", signals.low_vol)
    assert signals.load("cryptoyolo.signals:reversal_3d")[0] == "reversal_3d"
    with pytest.raises(ValueError):
        signals.load("not_a_signal")
    assert set(signals.load_many("builtin")) == set(signals.BUILTIN)
    assert signals.load_many("none") == {}
    assert list(signals.load_many("low_vol, reversal_1d")) == ["low_vol", "reversal_1d"]


def test_reversal_is_minus_the_last_24h_return():
    closes = pd.Series(np.linspace(100, 110, 48))
    ctx = signals.SignalContext("X", END, short=pd.DataFrame(), daily=pd.DataFrame(),
                                medium=pd.DataFrame({"close": closes}))
    assert signals.reversal_1d(ctx) == pytest.approx(-(110 / closes.iloc[-25] - 1))


# --------------------------------------------------------------------------
# regime, replayed
# --------------------------------------------------------------------------
@pytest.fixture
def bear_cache(cache):
    """Adds a BTC whose daily chart falls 60% over 400 days: risk_off."""
    iv = INTERVAL_MS["1d"]
    t0 = END.value // 1_000_000 - 400 * iv
    closes = np.linspace(100_000, 40_000, 400)
    cache.upsert_bars("BTC", "1d", [[t0 + i * iv, c, c, c, c, 1.0]
                                    for i, c in enumerate(closes)])
    for interval, n in (("5m", 288 * 30), ("1h", 24 * 60)):
        cache.upsert_bars("BTC", interval, _bars(n, interval, END, 99))
    return cache


def test_replay_attaches_the_live_gates_verdict(bear_cache, cfg):
    fr = _replay(bear_cache, cfg, days=5, symbols=["AAA", "BBB", "CCC", "BTC"])
    assert set(fr["regime_state"]) == {"risk_off"}
    assert (fr["regime_scale"] == cfg.regime.risk_off_exposure).all()


def test_regime_columns_are_empty_without_btc(cache, cfg):
    fr = _replay(cache, cfg, days=3)
    assert fr["regime_state"].isna().all()


def test_simulate_with_the_gate_scales_and_sits_out_risk_off(cfg):
    fr = _fr({"BTC": [0.5, 0.5, 0.5], "X": [0.4, 0.4, 0.4]},
             {"BTC": [0.10, 0.10, 0.10], "X": [0.10, 0.10, 0.10]})
    fr["regime_state"] = np.repeat(["risk_on", "neutral", "risk_off"], 2)
    fr["regime_scale"] = np.repeat([1.0, 0.5, 0.0], 2)
    sim = backtest.simulate(fr, cfg, hold_days=1, top_n=2, floor=0.0, cost_bps=0,
                            regime=True)
    p = sim["periods"]
    assert list(p["exposure"]) == [1.0, 0.5, 0.0]
    assert list(p["gross"].round(6)) == [0.10, 0.05, 0.0]
    assert sim["regime"] is True
    ungated = backtest.simulate(fr, cfg, hold_days=1, top_n=2, floor=0.0, cost_bps=0)
    assert list(ungated["periods"]["exposure"]) == [1.0, 1.0, 1.0]


def test_simulate_can_rank_by_a_candidate_signal(cfg):
    fr = _fr({"A": [0.9], "B": [0.1]}, {"A": [0.0], "B": [0.2]})
    fr["low_vol"] = [-5.0, -1.0]                 # B is the calmer name
    sim = backtest.simulate(fr, cfg, hold_days=1, top_n=1, cost_bps=0, score_col="low_vol")
    assert sim["periods"]["picks"].iloc[0] == "B"


def test_rank_buffer_keeps_an_incumbent_inside_top_n_plus_k(cfg):
    # A leads day 0; on day 1 it slips to 2nd behind B. top_n=1.
    fr = _fr({"A": [0.9, 0.5, 0.5], "B": [0.1, 0.6, 0.6], "C": [0.0, 0.0, 0.0]},
             {"A": [0.0] * 3, "B": [0.0] * 3, "C": [0.0] * 3})
    plain = backtest.simulate(fr, cfg, hold_days=1, top_n=1, floor=-1, cost_bps=0)
    kept = backtest.simulate(fr, cfg, hold_days=1, top_n=1, floor=-1, cost_bps=0, buffer=1)
    assert list(plain["periods"]["picks"]) == ["A", "B", "B"]
    assert list(kept["periods"]["picks"]) == ["A", "A", "A"]        # still within top 2
    assert kept["avg_turnover"] < plain["avg_turnover"]


def test_rank_buffer_drops_an_incumbent_outside_the_zone(cfg):
    fr = _fr({"A": [0.9, 0.1], "B": [0.1, 0.6], "C": [0.0, 0.5]},
             {"A": [0.0] * 2, "B": [0.0] * 2, "C": [0.0] * 2})
    kept = backtest.simulate(fr, cfg, hold_days=1, top_n=1, floor=-1, cost_bps=0, buffer=1)
    assert list(kept["periods"]["picks"]) == ["A", "B"]             # A fell to 3rd
