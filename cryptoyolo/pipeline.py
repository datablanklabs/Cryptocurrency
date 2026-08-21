"""End-to-end orchestration: scrape -> score -> propose -> approve -> execute.

`run()` is what the notebook's final cell calls. Every stage is individually
callable too, so you can re-run scoring without re-scraping.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from . import (approval, broker as broker_mod, catalysts, engine, exits, feeds,
               positioning, social)
from .config import CONFIG, Config, credential_status, load_dotenv
from .store import Store, iso, utcnow


def new_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:4]}"


def collect(store: Store, cfg: Config = CONFIG, scrape_reddit: bool = True,
            scan_catalysts: bool = True, verbose: bool = True,
            scrape_feeds: bool | None = None) -> dict[str, Any]:
    """Refresh the social and catalyst data feeding the engine.

    `scrape_feeds` (StockTwits + Mastodon) defaults to whatever `scrape_reddit`
    is, preserving existing behaviour, but is separable: previously they were
    nested under the Reddit flag, so `scrape_reddit=False` silently disabled two
    unrelated platforms.
    """
    if scrape_feeds is None:
        scrape_feeds = scrape_reddit
    stats: dict[str, Any] = {}
    if scrape_reddit:
        if verbose:
            print("\n[1/2] Reddit")
        try:
            stats["reddit"] = social.scrape(store, cfg, verbose)
        except Exception as exc:  # noqa: BLE001 - engine still runs without social
            print(f"  ! Reddit collection failed: {exc}")
            stats["reddit"] = {"posts": 0, "mentions": 0, "error": str(exc)}

    if scrape_feeds:
        if verbose:
            print("\n[1b] StockTwits + Mastodon")
        try:
            stats["feeds"] = feeds.scrape(store, cfg, verbose)
        except Exception as exc:  # noqa: BLE001 - engine still runs without them
            print(f"  ! Extra feeds failed: {exc}")
            stats["feeds"] = {}

    if scrape_feeds and cfg.positioning.enabled:
        if verbose:
            print("\n[1c] Positioning (funding rates)")
        try:
            stats["positioning"] = positioning.fetch(store, cfg, verbose)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Funding fetch failed: {exc}")
            stats["positioning"] = 0

    if scan_catalysts:
        if verbose:
            print("\n[2/2] Catalysts (GitHub + news)")
        try:
            stats["catalysts"] = catalysts.scan(store, cfg, verbose)
        except Exception as exc:  # noqa: BLE001 - engine still runs without catalysts
            print(f"  ! Catalyst scan failed: {exc}")
            stats["catalysts"] = 0
    return stats


def run(store: Store, cfg: Config = CONFIG, scrape_reddit: bool = True,
        scan_catalysts: bool = True, interactive: bool = True,
        input_fn: Callable[[str], str] | None = None,
        auto_approve: bool | None = None) -> dict[str, Any]:
    """Full cycle. Returns every intermediate artifact for inspection."""
    run_id = new_run_id()
    store.start_run(run_id, cfg.execution.mode)
    rule = "═" * 78

    print(f"{rule}\nRUN {run_id}   ·   {utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{broker_mod.execution_banner(cfg)}\n{rule}")

    stats = collect(store, cfg, scrape_reddit, scan_catalysts)

    # Holdings first: assets you own must be scored even when they sit outside
    # the configured universe, or they can never produce an exit signal.
    pre_broker = broker_mod.get_broker(store, cfg)
    held_symbols = [s for s, q in (pre_broker.holdings() or {}).items()
                    if q > 0 and s != cfg.execution.quote_asset]

    print("\n[scoring] building composite scores...")
    if held_symbols:
        outside = [s for s in held_symbols if s not in set(cfg.symbols)]
        if outside:
            print(f"  including {len(outside)} held asset(s) outside the universe: "
                  f"{', '.join(outside)}")
    scores = engine.build_scores(store, cfg, extra_symbols=held_symbols)
    if scores.empty:
        print("  No scoreable assets — aborting.")
        store.finish_run(run_id)
        return {"run_id": run_id, "scores": scores, "proposals": pd.DataFrame(),
                "executions": pd.DataFrame(), "stats": stats}

    store.save_scores(run_id, scores.to_dict("records"))

    broker = pre_broker
    holdings = broker.holdings()
    cash = broker.available_cash()

    print("\n[exits] reviewing open positions...")
    exit_signals = exits.evaluate(store, cfg, holdings, scores)
    if not cfg.exits.enabled:
        print("  exit management disabled (CONFIG.exits.enabled = False)")
    elif exit_signals.empty:
        n_open = len([s for s, q in holdings.items()
                      if q > 0 and s != cfg.execution.quote_asset])
        print(f"  {n_open} open position(s); no exit trigger fired")
    else:
        for _, sig in exit_signals.iterrows():
            print(f"  ⟵ {sig['symbol']:<6} {sig['trigger_label']:<16} {sig['reason']}")

    print("\n[proposals] ranking candidates...")
    if cash is None:
        print("  spendable balance: unknown (no credentials) — cash cap not applied; "
              "sizing falls back to risk.account_equity_usd")
    else:
        deploy_cap = cfg.risk.account_equity_usd * (cfg.risk.max_total_deployed_pct / 100.0)
        print(f"  spendable balance: ${cash:,.2f}   ·   deployment cap: ${deploy_cap:,.2f}"
              f"   ·   binding: {'cash' if cash < deploy_cap else 'deployment'}")

    proposals = engine.propose(scores, store, run_id, cfg, holdings,
                               cash_available=cash, exit_signals=exit_signals)

    if proposals.empty:
        print(engine.format_proposals(proposals))
        store.finish_run(run_id)
        return {"run_id": run_id, "scores": scores, "proposals": proposals,
                "executions": pd.DataFrame(), "stats": stats,
                "holdings": holdings, "cash": cash, "exit_signals": exit_signals}

    preflight = {p["proposal_id"]: broker.preflight(p.to_dict())
                 for _, p in proposals.iterrows()}

    if auto_approve is None:
        auto_approve = cfg.execution.auto_approve
    if not interactive and not auto_approve:
        print(engine.format_proposals(proposals))
        print("\n[non-interactive] no approval requested; nothing executed.")
        store.finish_run(run_id)
        return {"run_id": run_id, "scores": scores, "proposals": proposals,
                "executions": pd.DataFrame(), "stats": stats,
                "preflight": preflight, "holdings": holdings, "cash": cash,
                "exit_signals": exit_signals}

    reviewed = approval.request_approval(proposals, cfg, input_fn, preflight,
                                         auto_approve=auto_approve)

    print("\n[execution]")
    executions = approval.execute_approved(reviewed, broker, store, run_id, cfg)

    store.finish_run(run_id)
    return {"run_id": run_id, "scores": scores, "proposals": reviewed,
            "executions": executions, "stats": stats,
            "preflight": preflight, "holdings": holdings, "cash": cash,
            "exit_signals": exit_signals}


def status(cfg: Config = CONFIG) -> pd.DataFrame:
    """Which integrations are wired up. Shows no secret values."""
    creds = credential_status()
    rows = [
        ("Binance API keys", creds["binance"],
         "trade execution + best price granularity"),
        ("Reddit OAuth", creds["reddit_oauth"],
         "optional — keyless fallbacks (Arctic Shift, Atom feeds) cover it"),
        ("GitHub token", creds["github"],
         "optional — raises rate limit from 60/hr to 5000/hr"),
        ("CRYPTO_YOLO_ALLOW_LIVE", creds["live_trading_env"],
         "second switch required for real orders"),
        ("CRYPTO_YOLO_ALLOW_AUTO_LIVE", creds["auto_live_env"],
         "third switch — only for auto-approve against a live account"),
    ]
    return pd.DataFrame(
        [{"integration": n, "configured": "yes" if ok else "no", "why": why}
         for n, ok, why in rows]
    )
