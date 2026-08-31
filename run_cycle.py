#!/usr/bin/env python3
"""Run one full trading cycle from the command line.

    ./run_cycle.py                    # propose only, execute nothing
    ./run_cycle.py --approve          # prompt for each trade (same as the notebook)
    ./run_cycle.py --auto-approve     # accept every proposal, no prompting
    ./run_cycle.py --auto-approve --collect   # refresh data first, then trade

Default is propose-only: with no flags this prints the slate and exits without
touching anything, so running it by accident cannot trade.

--auto-approve removes the human from the loop. In paper mode that is exactly
what you want for unattended testing - it accumulates a decision record you can
later score the engine against. Against a live account it is unattended
real-money trading, and needs CRYPTO_YOLO_ALLOW_AUTO_LIVE=1 as well as the two
switches that already gate live mode.
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

LOCK_PATH = PROJECT / "data" / "run_cycle.lock"


def main() -> int:
    ap = argparse.ArgumentParser(description="crypto-yolo trading cycle")
    ap.add_argument("--approve", action="store_true",
                    help="prompt for each proposal (needs a terminal)")
    ap.add_argument("--auto-approve", action="store_true",
                    help="approve every proposal without prompting (NOT the default)")
    ap.add_argument("--collect", action="store_true",
                    help="refresh Reddit and catalysts before scoring")
    ap.add_argument("--paper", action="store_true",
                    help="force paper mode regardless of config")
    args = ap.parse_args()

    if args.approve and args.auto_approve:
        print("--approve and --auto-approve are mutually exclusive.")
        return 2

    load_dotenv()
    configure()
    log = get_logger("run_cycle")
    if args.paper:
        CONFIG.execution.mode = "paper"

    ex = CONFIG.execution
    if args.auto_approve and ex.live_enabled and not ex.auto_live_enabled:
        print("Refusing to auto-approve into a LIVE account.\n"
              "  mode=binance, dry_run=False and CRYPTO_YOLO_ALLOW_LIVE=1 are set,\n"
              "  but CRYPTO_YOLO_ALLOW_AUTO_LIVE=1 is not.\n"
              "Set it in .env if you genuinely want unattended real-money trading.")
        return 3

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another cycle is still running — skipping.")
            return 0

        store = Store(CONFIG.db_path)
        from cryptoyolo import pipeline

        started = datetime.now(timezone.utc)
        print(f"[{started:%Y-%m-%d %H:%M:%S UTC}] starting cycle "
              f"(approve={args.approve}, auto_approve={args.auto_approve})")
        log.info("cycle start (mode=%s approve=%s auto_approve=%s collect=%s)",
                 CONFIG.execution.mode, args.approve, args.auto_approve, args.collect)

        result = pipeline.run(
            store, CONFIG,
            scrape_reddit=args.collect,
            scan_catalysts=args.collect,
            # Propose-only unless one of the approval flags is given.
            interactive=args.approve,
            auto_approve=args.auto_approve,
        )

        execs = result.get("executions")
        n = 0 if execs is None or execs.empty else len(execs)
        # Timestamp the END, not a copy of the start - reusing one `stamp` made
        # every cycle look instantaneous in the logs.
        done = datetime.now(timezone.utc)
        print(f"[{done:%Y-%m-%d %H:%M:%S UTC}] cycle complete in "
              f"{(done - started).total_seconds():.1f}s — {n} order(s) executed")
        log.info("cycle finished in %.1fs — %d order(s); summary=%s",
                 (done - started).total_seconds(), n, result.get("summary"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
