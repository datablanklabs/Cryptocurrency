#!/usr/bin/env python3
"""Backfill archived Reddit history (Arctic Shift) for the social backtest.

    ./backfill_social.py                    # 365 days of posts, every configured sub
    ./backfill_social.py --days 90 --subs Bitcoin,CryptoCurrency
    ./backfill_social.py --comments         # comments too (~10x the requests)
    ./backfill_social.py --max-requests 500 # stop after 500 calls; rerun to resume
    ./backfill_social.py --status           # what's held, no network

Writes data/social_backfill.sqlite, never the main database. Resumable: each
run fetches only what the file doesn't already hold. Then:

    ./backtest.py --no-sync                 # picks the file up automatically

See cryptoyolo/social_backfill.py for what the replayed family is and isn't.
"""

from __future__ import annotations

import argparse
import fcntl
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from cryptoyolo import social_backfill                 # noqa: E402
from cryptoyolo.config import CONFIG, load_dotenv      # noqa: E402
from cryptoyolo.store import Store                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="backfill archived Reddit history")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--subs", help="comma-separated subreddits (default: CONFIG.reddit.subreddits)")
    ap.add_argument("--comments", action="store_true", help="also backfill comments")
    ap.add_argument("--max-requests", type=int, help="cap API calls this run")
    ap.add_argument("--delay", type=float, default=0.6, help="seconds between requests")
    ap.add_argument("--db", default=str(social_backfill.DEFAULT_PATH))
    ap.add_argument("--status", action="store_true", help="show what's held and exit")
    args = ap.parse_args()

    load_dotenv()
    store = Store(args.db)
    if args.status:
        st = social_backfill.status(store)
        print(st.to_string(index=False) if not st.empty else "empty")
        return 0

    lock_path = Path(args.db).with_suffix(".lock")
    with open(lock_path, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another backfill is running on this file — exiting.")
            return 0
        subs = [s.strip() for s in args.subs.split(",")] if args.subs else None
        print(f"[backfill] {args.days}d of {'posts + comments' if args.comments else 'posts'} "
              f"→ {args.db}")
        rep = social_backfill.backfill(store, CONFIG, days=args.days, subreddits=subs,
                                       comments=args.comments, delay=args.delay,
                                       max_requests=args.max_requests)
        print(social_backfill.summary_line(store))
        return 0 if rep["complete"].all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
