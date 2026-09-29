"""social_backfill: paging, resumability, and point-in-time scoring that matches
the live social score. The archive is faked; nothing touches the network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from cryptoyolo import backtest, social, social_backfill
from cryptoyolo.store import Store, utcnow

END = datetime(2026, 6, 1, tzinfo=timezone.utc)
_row = social.RedditClient._row          # before any test monkeypatches the class


class FakeArchive:
    """Arctic Shift, in memory: a post every hour for 10 days on r/Bitcoin."""
    posts: dict[str, list[float]] = {}
    calls = 0

    def __init__(self, cfg, sources=None):
        pass

    def archive(self, subreddit, kind, after, before, limit=100, newest_first=False):
        FakeArchive.calls += 1
        ts = [t for t in self.posts.get(subreddit, []) if after < t < before]
        ts = sorted(ts, reverse=newest_first)[:limit]
        return [_row(f"t3_{int(t)}", subreddit, "post", f"u{int(t) % 7}",
                                         t, 3, 1, "Bitcoin looks bullish, buying more", "", "")
                for t in ts]


@pytest.fixture
def archive(monkeypatch):
    start = (END - timedelta(days=10)).timestamp()
    FakeArchive.posts = {"Bitcoin": [start + 1800 + 3600 * i for i in range(240)]}
    FakeArchive.calls = 0
    monkeypatch.setattr(social_backfill.social, "RedditClient", FakeArchive)
    monkeypatch.setattr(social_backfill.time, "sleep", lambda s: None)
    return FakeArchive


def _held(store):
    with store.conn() as con:
        return sorted(r[0] for r in con.execute("SELECT created_utc FROM reddit_posts"))


def test_backfill_pages_the_whole_window_and_extracts_mentions(store, cfg, archive):
    rep = social_backfill.backfill(store, cfg, days=10, end=END, subreddits=["Bitcoin"],
                                   verbose=False)
    assert rep["complete"].all()
    assert _held(store) == archive.posts["Bitcoin"]            # all 240, none twice
    m = store.mentions_since(END - timedelta(days=30))
    assert len(m) == 240 and set(m["symbol"]) == {"BTC"}
    assert archive.calls >= 3                                 # >100 per page => paged


def test_a_capped_run_resumes_without_gaps(store, cfg, archive):
    rep = social_backfill.backfill(store, cfg, days=10, end=END, subreddits=["Bitcoin"],
                                   max_requests=1, verbose=False)
    assert not rep["complete"].all() and len(_held(store)) == 100
    social_backfill.backfill(store, cfg, days=10, end=END, subreddits=["Bitcoin"],
                             verbose=False)
    assert _held(store) == archive.posts["Bitcoin"]


def test_extending_the_window_backwards_fills_downward_and_contiguously(store, cfg, archive):
    social_backfill.backfill(store, cfg, days=3, end=END, subreddits=["Bitcoin"],
                             verbose=False)
    three = len(_held(store))
    # one call finds nothing newer than what's held, one fetches the next page back
    rep = social_backfill.backfill(store, cfg, days=10, end=END, subreddits=["Bitcoin"],
                                   max_requests=2, verbose=False)
    held = _held(store)
    assert not rep["complete"].all() and len(held) == three + 100
    # the 100 new ones are the ones just below what was held — no hole
    gaps = np.diff(held)
    assert gaps.max() == 3600
    social_backfill.backfill(store, cfg, days=10, end=END, subreddits=["Bitcoin"],
                             verbose=False)
    assert _held(store) == archive.posts["Bitcoin"]


def test_status_reports_what_is_held(store, cfg, archive):
    social_backfill.backfill(store, cfg, days=10, end=END, subreddits=["Bitcoin"],
                             verbose=False)
    st = social_backfill.status(store)
    assert st.loc[0, "items"] == 240 and st.loc[0, "mentions"] == 240


# -- point-in-time scoring ----------------------------------------------------
def _seed_recent(store: Store, now: datetime) -> None:
    """A week of chatter: BTC steady and gloomy, SOL upbeat and spiking in the
    last day. Tones differ on purpose: per-source de-biasing centres a platform
    whose every post sounds the same to zero."""
    rows, mentions = [], []
    for i in range(7 * 24):
        t = (now - timedelta(hours=7 * 24 - i) + timedelta(minutes=30)).timestamp()
        for sym, text in (("BTC", "Bitcoin dumping, bearish"),
                          ("SOL", "Solana pumping, bullish")):
            if sym == "SOL" and i < 6 * 24 and i % 6:
                continue                              # quiet until the last day
            pid = f"t3_{sym}{i}"
            rows.append(_row(pid, "CryptoCurrency", "post", f"a{i % 11}", t, 2, 0,
                             text, "", ""))
    patterns = social._build_patterns(social.CONFIG)
    mentions = social.mentions_for_rows(rows, patterns)
    store.upsert_posts(rows)
    store.upsert_mentions(mentions)


def test_score_at_matches_the_live_score(store, cfg):
    now = utcnow()
    _seed_recent(store, now)
    live = social.score_symbols(store, cfg).set_index("symbol")["social"]
    hist = social_backfill.SocialHistory(store)
    pit = hist.score_at(pd.Timestamp(now), cfg)
    assert pit["SOL"] > 0                            # the spike registers
    for sym in cfg.symbols:
        assert pit[sym] == pytest.approx(live[sym], abs=1e-9)


def test_score_at_sees_nothing_after_t_and_is_empty_outside_coverage(store, cfg):
    now = utcnow()
    _seed_recent(store, now)
    hist = social_backfill.SocialHistory(store)
    assert hist.score_at(pd.Timestamp(now - timedelta(days=30)), cfg) == {}
    # two days ago: SOL's spike (last 24h) hasn't happened yet
    early = hist.score_at(pd.Timestamp(now - timedelta(days=2)), cfg)
    assert early["SOL"] < hist.score_at(pd.Timestamp(now), cfg)["SOL"]


def test_replay_carries_social_where_the_archive_covers(tmp_path, cfg):
    from tests.test_backtest import END as BT_END, _bars
    cache = backtest.BarCache(tmp_path / "bt.sqlite")
    for k, sym in enumerate(("BTC", "ETH", "SOL")):
        cache.upsert_bars(sym, "5m", _bars(288 * 30, "5m", BT_END, k))
        cache.upsert_bars(sym, "1h", _bars(24 * 60, "1h", BT_END, 10 + k))
        cache.upsert_bars(sym, "1d", _bars(400, "1d", BT_END, 20 + k))
    sstore = Store(tmp_path / "soc.sqlite")
    _seed_recent(sstore, BT_END.to_pydatetime() - timedelta(days=2))
    fr = backtest.replay(cache, cfg, days=10, end=BT_END.to_pydatetime(),
                         symbols=["BTC", "ETH", "SOL"], verbose=False,
                         social_history=social_backfill.SocialHistory(sstore))
    covered = fr[fr["social"].notna()]
    assert 0 < covered["run_id"].nunique() < fr["run_id"].nunique()
    assert covered["composite_social"].notna().all()
    assert fr.loc[fr["social"].isna(), "composite_social"].isna().all()
    assert "social" in backtest._families(fr)
