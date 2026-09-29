#!/usr/bin/env python3
"""Replay the price-derived families (technical, xsec, positioning) over history.

    ./backtest.py                     # sync the cache, replay 365 days, report
    ./backtest.py --days 540          # a longer window
    ./backtest.py --no-sync           # offline: replay whatever is cached
    ./backtest.py --csv out.csv       # also write the per-(day, symbol) frame
    ./backtest.py --signals reversal_1d,mymod:my_signal   # candidate signals to IC
    ./backtest.py --signals none      # skip the candidate signals
    ./backtest.py --no-social         # ignore data/social_backfill.sqlite

Candles and funding go into their own cache DB (default
data/backtest_cache.sqlite), never the main database. The first sync of a year
of 5m bars takes a few minutes; later runs fetch only the new tail.
See cryptoyolo/backtest.py for what this can and can't tell you.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from cryptoyolo import backtest, signals          # noqa: E402
from cryptoyolo.config import CONFIG, load_dotenv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="point-in-time backtest of the price families")
    ap.add_argument("--days", type=int, default=365, help="replay window (default 365)")
    ap.add_argument("--hold", type=int, default=7,
                    help="rebalance / primary IC horizon in days (default 7)")
    ap.add_argument("--symbols", help="comma-separated subset (default: the universe)")
    ap.add_argument("--cache", default=str(backtest.DEFAULT_CACHE), help="cache DB path")
    ap.add_argument("--no-sync", action="store_true", help="don't touch the network")
    ap.add_argument("--csv", help="write the replayed frame here")
    ap.add_argument("--social-db", help="social backfill DB (default: "
                    "data/social_backfill.sqlite if it exists)")
    ap.add_argument("--no-social", action="store_true", help="skip the social replay")
    ap.add_argument("--buffer", type=int, help="rank buffer for the extra simulation rows "
                    "(default: max_proposals; 0 = none)")
    ap.add_argument("--signals", default="builtin",
                    help="candidate signals to IC alongside the families: 'builtin' "
                         f"(default: {', '.join(signals.BUILTIN)}), 'none', or a "
                         "comma list of built-in names / module:function / "
                         "label=module:function")
    args = ap.parse_args()
    try:
        extra = signals.load_many(args.signals)
    except (ValueError, ImportError, AttributeError) as exc:
        print(f"--signals: {exc}")
        return 2

    load_dotenv()
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    cache = backtest.BarCache(args.cache)
    end = datetime.now(timezone.utc)
    if not args.no_sync:
        print(f"[sync] {cache.path}")
        backtest.sync(cache, CONFIG, args.days, end, symbols)
    print("[replay]")
    horizons = tuple(sorted({1, 7, 30, args.hold}))
    social_hist = None if args.no_social else backtest.load_social_history(args.social_db)
    if social_hist is not None:
        print(f"[social] {len(social_hist.mentions):,} backfilled mentions")
    fr = backtest.replay(cache, CONFIG, args.days, end, symbols, horizons=horizons,
                         signals=extra, social_history=social_hist)
    if args.csv and not fr.empty:
        fr.to_csv(args.csv, index=False)
        print(f"wrote {len(fr):,} rows to {args.csv}")
    backtest.report(fr, CONFIG, primary_horizon=args.hold, hold_days=args.hold,
                    buffer=args.buffer)
    return 0 if not fr.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
