"""Recurring background scraping.

The Reddit signal needs history to be worth anything: velocity is defined
against a baseline, and on a cold database there is no baseline, so the first
run's social scores are all ~0 by construction. Leaving a collector running
between sessions is what makes the second and later runs meaningful.

Two ways to run it:

  in-notebook   start(store) spawns a daemon thread. Dies with the kernel, so
                it only accumulates history while the notebook is open.

  scheduled     Use ./collect.py (launchd or cron) - it bootstraps its own
                sys.path so it runs from any working directory. Do NOT schedule
                `python -m cryptoyolo.scheduler`: that form only resolves when
                cwd happens to be the project root, which neither launchd nor
                cron guarantees. See the README.
"""

from __future__ import annotations

import argparse
import threading
import time
from datetime import datetime, timezone
from typing import Any

from .config import CONFIG, Config, load_dotenv
from .store import Store


class Collector:
    """A daemon thread that re-scrapes on a fixed interval."""

    def __init__(self, store: Store, cfg: Config = CONFIG,
                 interval_minutes: int = 30, scan_catalysts: bool = False):
        self.store, self.cfg = store, cfg
        self.interval = max(5, interval_minutes) * 60
        self.scan_catalysts = scan_catalysts
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.history: list[dict[str, Any]] = []

    def _loop(self) -> None:
        from . import catalysts as catalysts_mod
        from . import social

        while not self._stop.is_set():
            entry: dict[str, Any] = {"at": datetime.now(timezone.utc).isoformat()}
            try:
                entry["reddit"] = social.scrape(self.store, self.cfg, verbose=False)
                if self.scan_catalysts:
                    entry["catalysts"] = catalysts_mod.scan(self.store, self.cfg,
                                                           verbose=False)
            except Exception as exc:  # noqa: BLE001 - never kill the collector
                entry["error"] = str(exc)
            self.history.append(entry)
            self._stop.wait(self.interval)

    def start(self) -> "Collector":
        if self._thread and self._thread.is_alive():
            print("Collector already running.")
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="crypto-yolo-collector")
        self._thread.start()
        print(f"Collector started · every {self.interval // 60} min · "
              f"stops when the kernel does.")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("Collector stopped.")

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "cycles": len(self.history),
            "history_hours": round(self.store.history_span_hours(), 2),
            "last": self.history[-1] if self.history else None,
        }


_ACTIVE: Collector | None = None


def start(store: Store, cfg: Config = CONFIG, interval_minutes: int = 30,
          scan_catalysts: bool = False) -> Collector:
    global _ACTIVE
    if _ACTIVE and _ACTIVE.running:
        print("Reusing the collector already running.")
        return _ACTIVE
    _ACTIVE = Collector(store, cfg, interval_minutes, scan_catalysts).start()
    return _ACTIVE


def stop() -> None:
    if _ACTIVE:
        _ACTIVE.stop()


def main() -> None:
    """CLI entry point for cron."""
    parser = argparse.ArgumentParser(description="crypto-yolo background collector")
    parser.add_argument("--once", action="store_true", help="single pass, then exit")
    parser.add_argument("--interval", type=int, default=30, help="minutes between passes")
    parser.add_argument("--catalysts", action="store_true", help="also scan news/GitHub")
    args = parser.parse_args()

    load_dotenv()
    store = Store(CONFIG.db_path)

    from . import catalysts as catalysts_mod
    from . import social

    if args.once:
        social.scrape(store, CONFIG, verbose=True)
        if args.catalysts:
            catalysts_mod.scan(store, CONFIG, verbose=True)
        print(f"History span: {store.history_span_hours():.1f}h")
        return

    collector = Collector(store, CONFIG, args.interval, args.catalysts)
    collector.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        collector.stop()


if __name__ == "__main__":
    main()
