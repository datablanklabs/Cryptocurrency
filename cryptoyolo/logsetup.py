"""Central logging setup: a rotating file in `data/` plus stderr.

Everything user-facing in this project is `print()` — right for the notebook,
wrong for the unattended launchd jobs, where there are no levels, no timestamps,
no rotation, and a traceback vanishes into a redirected stream. `configure()`
wires a `cryptoyolo` logger tree that the CLI entry points and the pipeline log
through, *on top of* the existing prints (which stay as the human transcript).

Nothing is configured on import — a library import must not attach handlers.
Call `configure()` once from an entry point (`run_cycle.py`, `collect.py`,
`backfill_prices.py`, or a notebook cell); `get_logger()` is a no-op accessor.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import CONFIG

DEFAULT_LOG_PATH = CONFIG.db_path.parent / "crypto-yolo.log"
_ROOT_NAME = "cryptoyolo"
_configured = False


def _resolve_level(level: int | str | None) -> int:
    if isinstance(level, int):
        return level
    name = (level or os.environ.get("CRYPTO_YOLO_LOG_LEVEL") or "INFO").upper()
    return getattr(logging, name, logging.INFO)


def configure(level: int | str | None = None,
              logfile: str | Path | None = None,
              *, console: bool = True, force: bool = False) -> logging.Logger:
    """Idempotently set up the `cryptoyolo` logger. Safe to call repeatedly.

    `level` falls back to env `CRYPTO_YOLO_LOG_LEVEL`, then INFO. `force=True`
    rebuilds the handlers (used by tests).
    """
    global _configured
    root = logging.getLogger(_ROOT_NAME)
    if _configured and not force:
        return root

    lvl = _resolve_level(level)
    root.setLevel(lvl)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    path = Path(logfile or DEFAULT_LOG_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=5,
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError as exc:  # read-only fs, un-creatable dir, ...
        root.warning("file logging disabled (%s)", exc)

    _configured = True
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """A child of the `cryptoyolo` logger. No side effects."""
    root = logging.getLogger(_ROOT_NAME)
    return root if not name else root.getChild(name)
