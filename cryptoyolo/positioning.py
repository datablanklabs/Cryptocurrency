"""Feature 4: positioning — perpetual funding rates as a crowding measure.

Deliberately a separate family from social sentiment, because it measures a
different thing. Social tells you what people are *saying*; funding tells you
what leveraged traders are *paying* to hold a side. Positive funding means longs
pay shorts (crowded long); negative means the reverse.

Why it earns its own component: it is the only input in the system that goes
genuinely negative on its own. Measured across 100 periods (~33 days), SOL was
negative 27% of the time and ETH 25%, while every social source measured is
positive nearly always. Without something like this the composite can drift
toward "everything is a buy" — which was exactly the failure mode the per-source
sentiment de-biasing had to correct for.

Read contrarian by default: crowded positioning is fragile positioning, so
unusually high funding scores bearish. Flip `contrarian=False` to read it as
momentum confirmation instead — both are defensible and neither is tested here.

Scored as a z-value against each asset's OWN funding history, not an absolute
threshold. 0.01% means something different for BTC than for a thin altcoin, and
the same relative-to-own-baseline logic already governs mention velocity and
sentiment de-biasing.
"""

from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd
import requests

from .config import CONFIG, Config
from .store import Store, iso


def fetch(store: Store, cfg: Config = CONFIG, verbose: bool = True) -> int:
    """Pull realised funding history for the universe and persist it."""
    pc = cfg.positioning
    if not pc.enabled:
        return 0
    sess = requests.Session()
    sess.headers.update({"User-Agent": cfg.user_agent})
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for asset in cfg.universe:
        inst = pc.inst_template.format(symbol=asset.symbol)
        try:
            r = sess.get(pc.venue_url,
                         params={"instId": inst, "limit": pc.history_periods},
                         timeout=cfg.http_timeout)
            r.raise_for_status()
            data = r.json().get("data") or []
        except Exception:  # noqa: BLE001 - one missing perp isn't fatal
            missing.append(asset.symbol)
            time.sleep(pc.request_delay)
            continue

        for d in data:
            rate = d.get("realizedRate") or d.get("fundingRate")
            if rate is None or not d.get("fundingTime"):
                continue
            rows.append({
                "symbol": asset.symbol,
                "funding_time": int(d["fundingTime"]),
                "rate": float(rate),
                "venue": "okx",
                "fetched_at": iso(),
            })
        time.sleep(pc.request_delay)

    store.upsert_funding(rows)
    if verbose:
        print(f"  funding: {len(rows)} periods across "
              f"{len(cfg.universe) - len(missing)}/{len(cfg.universe)} assets"
              + (f" (no perp: {', '.join(missing)})" if missing else ""))
    return len(rows)


def score_symbols(store: Store, cfg: Config = CONFIG) -> pd.DataFrame:
    """Per-asset positioning score in [-1, 1].

    z      = (latest funding - mean) / stdev, over the asset's own history
    score  = -tanh(z / z_scale)      contrarian: crowded long -> bearish

    Assets with too little history score 0 rather than a number derived from a
    handful of points - the same thin-sample caution used for sentiment
    de-biasing.
    """
    pc = cfg.positioning
    hist = store.funding_history()
    rows = []

    for symbol in cfg.symbols:
        sub = hist[hist["symbol"] == symbol] if not hist.empty else pd.DataFrame()
        if sub.empty or len(sub) < pc.min_periods:
            rows.append({"symbol": symbol, "funding_now": None, "funding_mean": None,
                         "funding_z": 0.0, "n_periods": len(sub), "positioning": 0.0})
            continue

        rates = sub.sort_values("funding_time")["rate"].to_numpy()
        latest = float(rates[-1])
        # The latest point is included in its own reference distribution. At
        # n>=20 the distortion is tiny (measured 0.04 z on ETH at n=100) and
        # excluding it would make the score jump around as the window rolls,
        # so it stays in deliberately.
        mean = float(rates.mean())
        std = float(rates.std(ddof=0))
        z = (latest - mean) / std if std > 1e-12 else 0.0
        raw = math.tanh(z / pc.z_scale) if pc.z_scale else 0.0
        score = -raw if pc.contrarian else raw

        rows.append({
            "symbol": symbol,
            "funding_now": round(latest * 100, 6),      # as a percentage
            "funding_mean": round(mean * 100, 6),
            "funding_z": round(z, 4),
            "n_periods": int(len(rates)),
            "positioning": round(float(max(-1.0, min(1.0, score))), 4),
        })

    return pd.DataFrame(rows).sort_values("positioning", key=abs, ascending=False
                                          ).reset_index(drop=True)
