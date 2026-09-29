"""Backfill a year of Reddit history so the social family can be backtested.

`social` is scored from mention *velocity* against each asset's own baseline,
and live collection only started in August - so it has weeks of history, and
its IC has almost no independent samples behind it. Arctic Shift is an archive:
it serves any subreddit's posts (and comments) for any date range. This module
pages through that archive into a SEPARATE database (default
`data/social_backfill.sqlite`, same schema as the main store) and
`backtest.replay(social_store=...)` scores social at each historical step with
`social.score_mentions` - the live scoring function - from mentions created
before that step.

Never writes the main database: backfilled history would change live velocity
baselines, and that database is the accrued record.

Resumable and incremental: each (subreddit, kind) fetches only the part of the
requested window it doesn't already hold, oldest first, so an interrupted run
picks up where it stopped.

What the replayed family is NOT, stated so it isn't mistaken for more:
  * Reddit only. Live social also reads StockTwits, Mastodon and /biz/, which
    have no archive here.
  * Posts only by default. Live scraping reads comments too; `--comments` adds
    them, at roughly ten times the requests.
  * Vote scores are as archived (final), not as they stood at the step, so the
    engagement weighting has a little lookahead. Mention counts, authors and
    sentiment are point-in-time.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
import pandas as pd

from . import social
from .config import CONFIG, DATA_DIR, Config
from .store import Store

DEFAULT_PATH = DATA_DIR / "social_backfill.sqlite"


def _held_range(store: Store, subreddit: str, kind: str) -> tuple[float | None, float | None]:
    with store.conn() as con:
        row = con.execute("SELECT MIN(created_utc), MAX(created_utc) FROM reddit_posts "
                          "WHERE subreddit=? AND kind=?", (subreddit, kind)).fetchone()
    return (row[0], row[1]) if row else (None, None)


def _fetch_span(client: social.RedditClient, store: Store, patterns: dict,
                subreddit: str, kind: str, start: float, stop: float,
                delay: float, budget: list[int],
                newest_first: bool = False) -> tuple[int, int, bool]:
    """Page the open interval (start, stop). Returns (items, mentions, complete).

    The direction keeps what is held contiguous if a run stops part-way: a span
    NEWER than the held range fills oldest-first (upward from it), a span OLDER
    than it fills newest-first (downward from it). Either way the next run's
    "what's missing" is exactly the unfetched remainder.
    """
    lo, hi, n_items, n_mentions = start, stop, 0, 0
    while hi - lo > 1:
        if budget[0] <= 0:
            return n_items, n_mentions, False
        budget[0] -= 1
        rows = client.archive(subreddit, kind, lo, hi, newest_first=newest_first)
        if rows is None:
            return n_items, n_mentions, False          # rate-limited out; resume later
        if not rows:
            break
        store.upsert_posts(rows)
        mentions = social.mentions_for_rows(rows, patterns)
        store.upsert_mentions(mentions)
        n_items += len(rows)
        n_mentions += len(mentions)
        if len(rows) < 100:
            break                                     # the span is exhausted
        # Bounds are exclusive; overlap by a second so items sharing the edge
        # timestamp aren't skipped (upserts make the overlap harmless), but
        # always make progress.
        if newest_first:
            hi = min(hi - 1, min(r["created_utc"] for r in rows) + 1)
        else:
            lo = max(lo + 1, max(r["created_utc"] for r in rows) - 1)
        time.sleep(delay)
    return n_items, n_mentions, True


def backfill(store: Store, cfg: Config = CONFIG, days: int = 365,
             end: datetime | None = None, subreddits: list[str] | None = None,
             comments: bool = False, delay: float = 0.6,
             max_requests: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Fill `store` with `days` of archived Reddit history ending at `end`.

    `max_requests` caps the total API calls this invocation makes (the run is
    resumable, so a cap just means "continue next time").
    """
    end = end or datetime.now(timezone.utc)
    t_end = end.timestamp()
    t_start = (end - timedelta(days=days)).timestamp()
    client = social.RedditClient(cfg, sources=("arctic",))
    patterns = social._build_patterns(cfg)
    kinds = ("post", "comment") if comments else ("post",)
    budget = [max_requests if max_requests is not None else 10**9]
    report = []
    for sub in subreddits or list(cfg.reddit.subreddits):
        for kind in kinds:
            lo, hi = _held_range(store, sub, kind)
            # Only the parts of the window not already held: older than the
            # oldest row, and newer than the newest.
            spans = ([(t_start, t_end, False)] if lo is None else
                     [(max(hi, t_start), t_end, False),         # newer: upward
                      (t_start, min(lo, t_end), True)])          # older: downward
            items = ments = 0
            complete = True
            for a, b, newest_first in spans:
                if b - a < 60:
                    continue
                i, m, ok = _fetch_span(client, store, patterns, sub, kind, a, b,
                                       delay, budget, newest_first)
                items, ments, complete = items + i, ments + m, complete and ok
            report.append({"subreddit": sub, "kind": kind, "new_items": items,
                           "new_mentions": ments, "complete": complete})
            if verbose:
                print(f"  r/{sub:<22} {kind:<7} +{items:>6} items  +{ments:>5} mentions"
                      f"{'' if complete else '   (incomplete — rerun to resume)'}")
    return pd.DataFrame(report)


def status(store: Store) -> pd.DataFrame:
    """What the backfill DB holds, per subreddit and kind. No network."""
    with store.conn() as con:
        df = pd.read_sql_query(
            "SELECT subreddit, kind, COUNT(*) AS items, MIN(created_utc) AS first, "
            "MAX(created_utc) AS last FROM reddit_posts GROUP BY subreddit, kind "
            "ORDER BY subreddit, kind", con)
        m = pd.read_sql_query("SELECT subreddit, COUNT(*) AS mentions FROM mentions "
                              "GROUP BY subreddit", con)
    if df.empty:
        return df
    for c in ("first", "last"):
        df[c] = pd.to_datetime(df[c], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    return df.merge(m, on="subreddit", how="left")


class SocialHistory:
    """All backfilled mentions in memory, sliced point-in-time per step."""

    def __init__(self, store: Store):
        m = store.mentions_since(datetime.fromtimestamp(0, tz=timezone.utc))
        self.mentions = (m.sort_values("created_utc").reset_index(drop=True)
                         if not m.empty else m)
        with store.conn() as con:
            posts = pd.read_sql_query("SELECT created_utc FROM reddit_posts "
                                      "ORDER BY created_utc", con)
        self.post_t = posts["created_utc"].to_numpy(dtype=float)
        self._m_t = (self.mentions["created_utc"].to_numpy(dtype=float)
                     if not self.mentions.empty else self.post_t[:0])

    @property
    def empty(self) -> bool:
        return self.mentions.empty

    def score_at(self, t: pd.Timestamp, cfg: Config) -> dict[str, float]:
        """social per symbol as a live run at `t` would have scored it.

        Empty when the archive holds no posts in the baseline window before `t`
        (outside the backfilled range): that is "no data", which the replay
        records as NaN, not the 0.0 a live cold start would show.
        """
        rc = cfg.reddit
        hi = t.timestamp()
        lo = hi - rc.baseline_days * 86400.0
        i0, i1 = self._m_t.searchsorted(lo, "left"), self._m_t.searchsorted(hi, "left")
        p0, p1 = self.post_t.searchsorted(lo, "left"), self.post_t.searchsorted(hi, "left")
        if p1 - p0 < 2:
            return {}
        span = (self.post_t[p1 - 1] - self.post_t[p0]) / 3600.0
        hist = self.mentions.iloc[i0:i1]
        df = social.score_mentions(hist, span, t.to_pydatetime(), cfg)
        return dict(zip(df["symbol"], df["social"]))


def summary_line(store: Store) -> str:
    st = status(store)
    if st.empty:
        return "social backfill: empty"
    return (f"social backfill: {int(st['items'].sum()):,} items, "
            f"{int(st['mentions'].fillna(0).sum()):,} mentions, "
            f"{st['first'].min()} → {st['last'].max()}, {st['subreddit'].nunique()} subreddits")

