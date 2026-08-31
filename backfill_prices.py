#!/usr/bin/env python3
"""Populate `prices_daily` so `evaluation` runs deterministically and offline.

`evaluation` computes forward returns and the BTC/basket benchmark from the
local `prices_daily` table. From now on `engine.build_scores` fills that table
every cycle, but the runs already in the database predate it — this one-off
fetches 1y of daily closes for every asset in the universe (plus anything that
already appears in the `scores` table) and stores them.

    ./backfill_prices.py             # fetch + store
    ./backfill_prices.py --status    # show coverage, no network
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from cryptoyolo.config import CONFIG, load_dotenv          # noqa: E402
from cryptoyolo.logsetup import configure, get_logger      # noqa: E402
from cryptoyolo.store import Store                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="backfill the prices_daily table")
    ap.add_argument("--status", action="store_true",
                    help="print per-symbol coverage and exit (no network)")
    args = ap.parse_args()

    load_dotenv()
    configure()
    log = get_logger("backfill")
    store = Store(CONFIG.db_path)

    cov = store.daily_price_coverage()
    if args.status:
        if cov.empty:
            print("prices_daily is empty — run ./backfill_prices.py")
        else:
            print(cov.to_string(index=False))
            print(f"\n{len(cov)} symbols, {int(cov['days'].sum()):,} rows total")
        return 0

    from cryptoyolo import prices as prices_mod

    with store.conn() as con:
        seen = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM scores")]
    symbols = list(dict.fromkeys([*CONFIG.symbols, *seen]))
    log.info("backfilling %d symbols", len(symbols))
    print(f"backfilling {len(symbols)} symbols into {CONFIG.db_path} ...")

    ok = 0
    for sym in symbols:
        try:
            df, src = prices_mod.get_ohlcv(sym, "1y", CONFIG)
            n = store.upsert_daily_prices_from_series(sym, df["close"], src)
            print(f"  {sym:<6} {n:>4} days  [{src}]")
            ok += 1
        except Exception as exc:  # noqa: BLE001 - one bad symbol shouldn't stop the backfill
            log.warning("%s failed: %s", sym, exc)
            print(f"  {sym:<6} FAILED: {exc}")

    total = int(store.daily_price_coverage()["days"].sum())
    log.info("backfill done — %d/%d symbols, %d rows", ok, len(symbols), total)
    print(f"\ndone — {ok}/{len(symbols)} symbols, {total:,} rows in prices_daily")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
