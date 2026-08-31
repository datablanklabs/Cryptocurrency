"""Shared fixtures.

Every test that touches SQLite gets a throwaway database under `tmp_path` —
the real `data/` directory holds days of accrued history and must never be
opened by the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cryptoyolo.config import Config  # noqa: E402
from cryptoyolo.store import Store  # noqa: E402


@pytest.fixture
def cfg() -> Config:
    """A fresh default Config (paper mode, notifications effectively inert)."""
    c = Config()
    c.notify.enabled = False          # no banners / network from the suite
    return c


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "test.sqlite")


@pytest.fixture
def make_series():
    """Build a daily close Series: `make_series([100, 101, ...], start=...)`."""
    def _make(values, start="2026-01-01"):
        idx = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
        return pd.Series(np.asarray(values, dtype=float), index=idx)
    return _make


@pytest.fixture
def ohlcv_from_close():
    """Wrap a close Series into an OHLCV DataFrame like `prices.get_ohlcv`."""
    def _make(close: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": 1_000.0,
        }, index=close.index)
    return _make
