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
               macro, positioning, regime as regime_mod, social)
from .config import CONFIG, Config, credential_status, load_dotenv
from .logsetup import get_logger
from .notify import notify
from .store import Store, iso, utcnow

_log = get_logger("pipeline")


def _mark_to_market(holdings: dict[str, float], scores, cfg: Config,
                    quote: str) -> float:
    """Dollar value of open positions, priced from the score frame where
    possible (no extra network) and from a spot lookup otherwise."""
    px: dict[str, float] = {}
    if scores is not None and not scores.empty:
        px = dict(zip(scores["symbol"], scores["price"]))
    missing = [s for s, q in holdings.items()
               if q > 0 and s != quote and s not in px]
    if missing:
        try:
            from . import prices as prices_mod
            px.update(prices_mod.latest_prices(missing, cfg))
        except Exception:  # noqa: BLE001 - value what we can
            pass
    return float(sum(q * px.get(s, 0.0) for s, q in holdings.items()
                     if q > 0 and s != quote))


def _snapshot_equity(store: Store, broker, scores, cfg: Config, run_id: str) -> None:
    """Record this cycle's mark-to-market equity for the curve / benchmark.

    Re-reads holdings and cash so it reflects any fills this cycle just made.
    Best-effort: a pricing hiccup must not fail an otherwise-complete run.
    """
    try:
        quote = cfg.execution.quote_asset
        holdings = broker.holdings() or {}
        cash = broker.available_cash()
        pos_val = _mark_to_market(holdings, scores, cfg, quote)
        equity = (cash + pos_val) if cash is not None else None
        store.record_equity(run_id, cash, pos_val, equity,
                            mode=getattr(broker, "mode", cfg.execution.mode))
    except Exception as exc:  # noqa: BLE001
        _log.exception("equity snapshot skipped")
        print(f"  ! equity snapshot skipped ({exc})")


def _finalize_equity(store: Store, broker, scores, cfg: Config, run_id: str) -> None:
    """Snapshot equity for the curve, then alert if the book has drawn down past
    `notify.drawdown_alert_pct` from its own trailing peak."""
    _snapshot_equity(store, broker, scores, cfg, run_id)
    if not cfg.notify.on_drawdown:
        return
    try:
        mode = getattr(broker, "mode", cfg.execution.mode)
        ec = store.equity_curve(mode).dropna(subset=["equity"])
        if len(ec) < 3:
            return
        curve = ec.sort_values("ts")["equity"].astype(float)
        peak = float(curve.cummax().iloc[-1])
        last = float(curve.iloc[-1])
        dd = (last / peak - 1.0) * 100 if peak > 0 else 0.0
        if dd <= -cfg.notify.drawdown_alert_pct:
            _log.warning("equity drawdown %.1f%% (%.2f vs peak %.2f) [%s]",
                         dd, last, peak, mode)
            notify(f"{mode}: equity drawdown {dd:.1f}%",
                   f"equity ${last:,.2f}, down {dd:.1f}% from peak ${peak:,.2f}",
                   cfg, tag="drawdown")
    except Exception:  # noqa: BLE001 - a monitoring check must never fail a run
        _log.exception("drawdown check failed")


def _cycle_summary(cfg: Config, run_id: str, *, regime: dict | None,
                   proposals: pd.DataFrame | None,
                   executions: pd.DataFrame | None,
                   equity: float | None) -> dict[str, Any]:
    props = proposals if proposals is not None else pd.DataFrame()
    execs = executions if executions is not None else pd.DataFrame()
    has_side = not props.empty and "side" in props.columns
    has_kind = not props.empty and "kind" in props.columns
    rej = (execs["status"].map(broker_mod.is_rejected)
           if not execs.empty and "status" in execs.columns else pd.Series(dtype=bool))
    return {
        "run_id": run_id,
        "mode": cfg.execution.mode,
        "regime": (regime or {}).get("state", "n/a"),
        "proposals": int(len(props)),
        "buys": int((props["side"] == "BUY").sum()) if has_side else 0,
        "exits": int((props["kind"] == "exit").sum()) if has_kind else 0,
        "executed": int((~rej).sum()) if len(rej) else 0,
        "rejected": int(rej.sum()) if len(rej) else 0,
        "equity": round(float(equity), 2) if equity else None,
    }


def _log_summary(s: dict[str, Any]) -> None:
    _log.info("cycle %s done: mode=%s regime=%s proposals=%d (buy=%d exit=%d) "
              "executed=%d rejected=%d equity=%s",
              s["run_id"], s["mode"], s["regime"], s["proposals"], s["buys"],
              s["exits"], s["executed"], s["rejected"],
              f"${s['equity']:,.2f}" if s["equity"] is not None else "n/a")


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
            _log.exception("Reddit collection failed")
            print(f"  ! Reddit collection failed: {exc}")
            stats["reddit"] = {"posts": 0, "mentions": 0, "error": str(exc)}

    if scrape_feeds:
        if verbose:
            print("\n[1b] StockTwits + Mastodon")
        try:
            stats["feeds"] = feeds.scrape(store, cfg, verbose)
        except Exception as exc:  # noqa: BLE001 - engine still runs without them
            _log.exception("extra feeds failed")
            print(f"  ! Extra feeds failed: {exc}")
            stats["feeds"] = {}

    if scrape_feeds and cfg.positioning.enabled:
        if verbose:
            print("\n[1c] Positioning (funding rates)")
        try:
            stats["positioning"] = positioning.fetch(store, cfg, verbose)
        except Exception as exc:  # noqa: BLE001
            _log.exception("funding fetch failed")
            print(f"  ! Funding fetch failed: {exc}")
            stats["positioning"] = 0

    if scrape_feeds and cfg.macro.enabled:
        if verbose:
            print("\n[1d] Macro (Kalshi event contracts)")
        try:
            stats["macro"] = macro.fetch(store, cfg, verbose)
        except Exception as exc:  # noqa: BLE001 - engine still runs without it
            _log.exception("macro fetch failed")
            print(f"  ! Macro fetch failed: {exc}")
            stats["macro"] = 0

    if scan_catalysts:
        if verbose:
            print("\n[2/2] Catalysts (GitHub + news)")
        try:
            stats["catalysts"] = catalysts.scan(store, cfg, verbose)
        except Exception as exc:  # noqa: BLE001 - engine still runs without catalysts
            _log.exception("catalyst scan failed")
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
        _log.error("run %s aborted: no scoreable assets (price fetch failed?)", run_id)
        print("  No scoreable assets — aborting.")
        store.finish_run(run_id)
        return {"run_id": run_id, "scores": scores, "proposals": pd.DataFrame(),
                "executions": pd.DataFrame(), "stats": stats, "summary": None}

    store.save_scores(run_id, scores.to_dict("records"))

    # If scoring only got prices out of the yfinance backstop, the other three
    # feeds were unreachable this run — worth knowing before trusting the slate.
    px_sources = scores.attrs.get("price_sources", [])
    if px_sources == ["yfinance"] and cfg.notify.on_price_fallback:
        _log.warning("price feed fell through to yfinance only")
        notify("price feed fell back to yfinance",
               "binance / coinbase / kraken were all unreachable this run; "
               "prices came from the yfinance backstop.", cfg, tag="warning")
    elif px_sources:
        _log.info("price sources this run: %s", ", ".join(px_sources))

    broker = pre_broker
    holdings = broker.holdings()
    cash = broker.available_cash()

    quote = cfg.execution.quote_asset
    positions_value = _mark_to_market(holdings or {}, scores, cfg, quote)
    live_equity = (cash + positions_value) if cash is not None else None

    print("\n[exits] reviewing open positions...")
    exit_signals = exits.evaluate(store, cfg, holdings, scores)
    if not cfg.exits.enabled:
        print("  exit management disabled (CONFIG.exits.enabled = False)")
    elif exit_signals.empty:
        n_open = len([s for s, q in holdings.items()
                      if q > 0 and s != quote])
        print(f"  {n_open} open position(s); no exit trigger fired")
    else:
        for _, sig in exit_signals.iterrows():
            print(f"  ⟵ {sig['symbol']:<6} {sig['trigger_label']:<16} {sig['reason']}")

    print("\n[regime] reading BTC trend...")
    regime = regime_mod.assess(cfg, store=store)
    print(f"  {regime_mod.describe(regime)}")
    store.record_regime(run_id, regime)   # so its own effect can be measured later
    _log.info("run %s regime=%s scale=%.2f", run_id, regime.get("state"),
              float(regime.get("exposure_scale", 1.0)))
    if regime.get("state") == "risk_off" and cfg.notify.on_regime_risk_off:
        notify(f"{cfg.execution.mode}: regime risk_off",
               regime.get("note", "BTC-trend gate is blocking new long entries."),
               cfg, tag="warning")

    print("\n[proposals] ranking candidates...")
    risk_basis = (live_equity if (cfg.risk.size_off_live_equity and live_equity)
                  else cfg.risk.account_equity_usd)
    if cfg.risk.size_off_live_equity and live_equity:
        print(f"  risk basis: ${risk_basis:,.2f} (live equity: ${cash:,.2f} cash "
              f"+ ${positions_value:,.2f} positions)")
    else:
        print(f"  risk basis: ${risk_basis:,.2f} (static account_equity_usd; "
              f"{'live equity unreadable' if cfg.risk.size_off_live_equity else 'size_off_live_equity=False'})")
    if cash is None:
        print("  spendable balance: unknown (no credentials) — cash cap not applied")
    else:
        deploy_cap = risk_basis * (cfg.risk.max_total_deployed_pct / 100.0) * regime.get("exposure_scale", 1.0)
        print(f"  spendable balance: ${cash:,.2f}   ·   deployment cap: ${deploy_cap:,.2f}"
              f"   ·   binding: {'cash' if cash < deploy_cap else 'deployment'}")

    proposals = engine.propose(scores, store, run_id, cfg, holdings,
                               cash_available=cash, exit_signals=exit_signals,
                               equity_override=live_equity, regime=regime)

    if proposals.empty:
        print(engine.format_proposals(proposals))
        _finalize_equity(store, broker, scores, cfg, run_id)
        store.finish_run(run_id)
        summary = _cycle_summary(cfg, run_id, regime=regime, proposals=proposals,
                                 executions=None, equity=live_equity)
        _log_summary(summary)
        return {"run_id": run_id, "scores": scores, "proposals": proposals,
                "executions": pd.DataFrame(), "stats": stats,
                "holdings": holdings, "cash": cash, "exit_signals": exit_signals,
                "summary": summary}

    preflight = {p["proposal_id"]: broker.preflight(p.to_dict())
                 for _, p in proposals.iterrows()}

    if auto_approve is None:
        auto_approve = cfg.execution.auto_approve
    if not interactive and not auto_approve:
        print(engine.format_proposals(proposals))
        print("\n[non-interactive] no approval requested; nothing executed.")
        _finalize_equity(store, broker, scores, cfg, run_id)
        store.finish_run(run_id)
        summary = _cycle_summary(cfg, run_id, regime=regime, proposals=proposals,
                                 executions=None, equity=live_equity)
        _log_summary(summary)
        return {"run_id": run_id, "scores": scores, "proposals": proposals,
                "executions": pd.DataFrame(), "stats": stats,
                "preflight": preflight, "holdings": holdings, "cash": cash,
                "exit_signals": exit_signals, "summary": summary}

    reviewed = approval.request_approval(proposals, cfg, input_fn, preflight,
                                         auto_approve=auto_approve)

    print("\n[execution]")
    executions = approval.execute_approved(reviewed, broker, store, run_id, cfg)

    _finalize_equity(store, broker, scores, cfg, run_id)
    store.finish_run(run_id)
    summary = _cycle_summary(cfg, run_id, regime=regime, proposals=reviewed,
                             executions=executions, equity=live_equity)
    _log_summary(summary)
    return {"run_id": run_id, "scores": scores, "proposals": reviewed,
            "executions": executions, "stats": stats,
            "preflight": preflight, "holdings": holdings, "cash": cash,
            "exit_signals": exit_signals, "summary": summary}


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
        ("Kalshi API key", creds["kalshi"],
         "optional — macro regime dampener (Feature 6); no-op without it"),
        ("CRYPTO_YOLO_ALLOW_LIVE", creds["live_trading_env"],
         "second switch required for real orders"),
        ("CRYPTO_YOLO_ALLOW_AUTO_LIVE", creds["auto_live_env"],
         "third switch — only for auto-approve against a live account"),
    ]
    return pd.DataFrame(
        [{"integration": n, "configured": "yes" if ok else "no", "why": why}
         for n, ok, why in rows]
    )
