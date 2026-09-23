#!/usr/bin/env python3
"""Standalone collector entry point for launchd / cron.

Why this exists rather than `python3 -m cryptoyolo.scheduler`: that form only
resolves when the current directory happens to be the project root, and neither
launchd nor cron guarantees a working directory. This script bootstraps its own
location onto sys.path, so it can be invoked by absolute path from anywhere.

    ./collect.py                 # Reddit only
    ./collect.py --catalysts     # Reddit + news + GitHub  (what the 8h job runs)
    ./collect.py --status        # what has been collected, no network calls

A file lock keeps runs from overlapping. If one pass stalls on a slow feed, the
next scheduled trigger exits immediately instead of stacking a second writer
onto the same SQLite file.
"""

from __future__ import annotations

import argparse
import fcntl
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from cryptoyolo.config import CONFIG, load_dotenv  # noqa: E402
from cryptoyolo.logsetup import configure, get_logger  # noqa: E402
from cryptoyolo.store import Store  # noqa: E402

LOCK_PATH = PROJECT / "data" / "collector.lock"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def show_status(store: Store) -> None:
    import pandas as pd

    span = store.history_span_hours()
    with store.conn() as con:
        posts = con.execute("SELECT COUNT(*) n FROM reddit_posts").fetchone()["n"]
        mentions = con.execute("SELECT COUNT(*) n FROM mentions").fetchone()["n"]
        cats = con.execute("SELECT COUNT(*) n FROM catalysts").fetchone()["n"]
        last = con.execute(
            "SELECT MAX(fetched_at) t FROM reddit_posts").fetchone()["t"]
    print(f"database      : {CONFIG.db_path}")
    print(f"reddit posts  : {posts:,}")
    print(f"mentions      : {mentions:,}")
    print(f"catalysts     : {cats:,}")
    print(f"history span  : {span:.1f}h")
    print(f"last collected: {last or 'never'}")
    ready = span >= CONFIG.reddit.velocity_window_hours * 1.5
    print(f"velocity ready: {'yes' if ready else 'no'} "
          f"(needs ~{CONFIG.reddit.velocity_window_hours * 1.5:.0f}h)")

    rank = store.source_rank()
    if not rank.empty:
        age = store.source_rank_age_hours()
        print(f"\nsource ranking  (recomputed every {CONFIG.reddit.rank_refresh_hours:.0f}h; "
              f"age {age:.1f}h)")
        keys = {s.lower() for s in CONFIG.reddit.subreddits}
        shown = rank[rank["subreddit"].str.lower().isin(keys)]
        print(f"  {'#':>2}  {'subreddit':<24}{'value':>8}{'mention%':>10}"
              f"{'assets':>8}{'tone':>7}{'items/h':>9}")
        for i, (_, r) in enumerate(shown.iterrows(), 1):
            print(f"  {i:>2}. {r['subreddit']:<24}{r['value']:>8.3f}"
                  f"{r['mention_rate']*100:>9.1f}%{int(r['assets']):>8}"
                  f"{r['tone']:>7.2f}{r['items_per_hour']:>9.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="crypto-yolo collector")
    ap.add_argument("--catalysts", action="store_true",
                    help="also scan news RSS and GitHub releases")
    ap.add_argument("--prices", action="store_true",
                    help="also sweep 1y daily closes into prices_daily "
                         "(keeps evaluation's price record fresh between cycles)")
    ap.add_argument("--status", action="store_true",
                    help="report what has been collected, then exit")
    args = ap.parse_args()

    load_dotenv()
    configure()
    log = get_logger("collect")
    store = Store(CONFIG.db_path)

    if args.status:
        show_status(store)
        return 0

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[{stamp()}] another collection is still running — skipping this pass")
            return 0

        print(f"[{stamp()}] collecting...")
        log.info("collection pass started (catalysts=%s prices=%s)",
                 args.catalysts, args.prices)
        from cryptoyolo import social
        try:
            stats = social.scrape(store, CONFIG, verbose=True)
            print(f"  reddit: {stats}")
        except Exception as exc:  # noqa: BLE001 - never let one source kill the job
            log.exception("reddit collection failed")
            print(f"  ! reddit failed: {exc}")

        from cryptoyolo import feeds
        try:
            fstats = feeds.scrape(store, CONFIG, verbose=True)
            print(f"  feeds: {fstats}")
        except Exception as exc:  # noqa: BLE001
            log.exception("feeds collection failed")
            print(f"  ! feeds failed: {exc}")

        from cryptoyolo import positioning
        try:
            positioning.fetch(store, CONFIG, verbose=True)
        except Exception as exc:  # noqa: BLE001
            log.exception("funding fetch failed")
            print(f"  ! funding fetch failed: {exc}")

        from cryptoyolo import macro
        try:
            macro.fetch(store, CONFIG, verbose=True)
        except Exception as exc:  # noqa: BLE001
            log.exception("macro fetch failed")
            print(f"  ! macro fetch failed: {exc}")

        from cryptoyolo import kalshi_prediction
        try:
            kalshi_prediction.fetch(store, CONFIG, verbose=True)
        except Exception as exc:  # noqa: BLE001
            log.exception("Kalshi price-market fetch failed")
            print(f"  ! Kalshi price-market fetch failed: {exc}")

        if args.catalysts:
            from cryptoyolo import catalysts
            try:
                n = catalysts.scan(store, CONFIG, verbose=True)
                print(f"  catalysts: {n} events")
            except Exception as exc:  # noqa: BLE001
                log.exception("catalyst scan failed")
                print(f"  ! catalysts failed: {exc}")

        if args.prices:
            from cryptoyolo import prices as prices_mod
            ok = 0
            for sym in CONFIG.symbols:
                try:
                    df, src = prices_mod.get_ohlcv(sym, "1y", CONFIG)
                    store.upsert_daily_prices_from_series(sym, df["close"], src)
                    ok += 1
                except Exception as exc:  # noqa: BLE001 - one bad symbol is fine
                    log.warning("price sweep %s failed: %s", sym, exc)
            print(f"  prices: {ok}/{len(CONFIG.symbols)} symbols → prices_daily")

        span = store.history_span_hours()
        log.info("collection pass done — history span %.1fh", span)
        print(f"[{stamp()}] done — history span now {span:.1f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
