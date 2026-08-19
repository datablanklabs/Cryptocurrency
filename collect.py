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


def main() -> int:
    ap = argparse.ArgumentParser(description="crypto-yolo collector")
    ap.add_argument("--catalysts", action="store_true",
                    help="also scan news RSS and GitHub releases")
    ap.add_argument("--status", action="store_true",
                    help="report what has been collected, then exit")
    args = ap.parse_args()

    load_dotenv()
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
        from cryptoyolo import social
        try:
            stats = social.scrape(store, CONFIG, verbose=True)
            print(f"  reddit: {stats}")
        except Exception as exc:  # noqa: BLE001 - never let one source kill the job
            print(f"  ! reddit failed: {exc}")

        if args.catalysts:
            from cryptoyolo import catalysts
            try:
                n = catalysts.scan(store, CONFIG, verbose=True)
                print(f"  catalysts: {n} events")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! catalysts failed: {exc}")

        print(f"[{stamp()}] done — history span now {store.history_span_hours():.1f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
