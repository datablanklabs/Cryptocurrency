"""The decision engine: three feature families -> ranked trade candidates.

What this does and does not claim
---------------------------------
It ranks the universe by a transparent scoring function and sizes the top names
against your stated risk budget. It does NOT know which trades will be the most
profitable over the next day-to-week, and neither does anything else. Every
number below is a heuristic with hand-set weights; the value here is that the
weights are visible, the components are logged per run, and you can argue with
any of them.

Read `components` in the output before acting on `composite`. If a proposal is
driven by a social score built on 40 mentions from 6 accounts, that is worth
knowing, and the table tells you.

Spot-only constraint
--------------------
Binance spot cannot short. A bearish signal on an asset you do not hold is not a
trade, it is an avoid - so it is reported in the ranking but never turned into a
proposal. Bearish signals on assets you DO hold become SELL (exit) proposals.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import catalysts as catalysts_mod
from . import indicators, prices
from . import social as social_mod
from .config import CONFIG, Config
from .store import Store, iso, utcnow


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _squash(x: float, scale: float) -> float:
    """Map an unbounded quantity into [-1, 1] with a soft knee at `scale`."""
    return float(np.tanh(x / scale)) if scale else 0.0


# --------------------------------------------------------------------------
# Technical scoring
# --------------------------------------------------------------------------
def technical_score(short: dict[str, float], medium: dict[str, float]) -> tuple[float, dict]:
    """Score one asset from its 1d (short) and 1w (medium) indicator snapshots.

    Components, each in [-1, 1]:
      trend       where price sits relative to its 20/50 SMAs on the weekly view
      momentum    MACD histogram, normalized by ATR so it compares across assets
      position    %B - band position, interpreted differently by regime
      rsi         classic overbought/oversold, deliberately low-weighted
      volume      volume z-score, as confirmation of the other components

    The regime split on `position` is the one genuinely non-obvious piece: in a
    trending market, price riding the upper band is strength (momentum), while
    in a range-bound market the same reading is stretched (mean reversion). One
    fixed interpretation is wrong half the time, so trend strength decides.
    """
    dist20 = medium.get("dist_sma_20", 0.0)
    dist50 = medium.get("dist_sma_50", 0.0)
    atr_pct = max(medium.get("atr_pct", 0.02), 1e-6)

    # Normalize distances by the asset's own volatility - 3% above the mean
    # means something very different for BTC than for SHIB.
    trend = _clip(0.6 * _squash(dist20 / atr_pct, 3.0) + 0.4 * _squash(dist50 / atr_pct, 5.0))

    mom_short = _squash(short.get("macd_hist", 0.0) / (short.get("close", 1.0) * atr_pct), 0.8)
    mom_med = _squash(medium.get("macd_hist", 0.0) / (medium.get("close", 1.0) * atr_pct), 0.8)
    momentum = _clip(0.45 * mom_short + 0.55 * mom_med)

    trend_strength = min(1.0, abs(trend) / 0.5)
    pctb = medium.get("bb_pctb", 0.5)
    centered = _clip((pctb - 0.5) * 2.0)          # -1 at lower band, +1 at upper
    # Trending -> band position confirms direction. Ranging -> it fades.
    position = _clip(centered * trend_strength - centered * (1 - trend_strength) * 0.8)

    rsi_val = medium.get("rsi_14", 50.0)
    rsi_score = _clip((50.0 - rsi_val) / 25.0) * 0.6   # oversold = mildly bullish

    volume = _clip(_squash(short.get("vol_zscore", 0.0), 2.0)) * np.sign(momentum or 1.0)

    parts = {
        "trend": round(trend, 4),
        "momentum": round(momentum, 4),
        "position": round(position, 4),
        "rsi": round(rsi_score, 4),
        "volume": round(float(volume), 4),
    }
    weights = {"trend": 0.32, "momentum": 0.30, "position": 0.18, "rsi": 0.10, "volume": 0.10}
    score = sum(parts[k] * w for k, w in weights.items())

    # A volatility squeeze doesn't say which way, but it does say a move is
    # more likely - so it amplifies whatever direction the rest of the
    # components already agree on.
    if medium.get("squeeze", 0.0) > 0:
        score *= 1.15
        parts["squeeze"] = 1.0

    return _clip(score), parts


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_scores(store: Store, cfg: Config = CONFIG, verbose: bool = True,
                 extra_symbols: list[str] | None = None) -> pd.DataFrame:
    """Compute technical/social/catalyst/composite scores.

    `extra_symbols` covers assets you hold that are not in the configured
    universe. Without it those positions are never scored, so no exit signal can
    ever be produced for them - they would be silently unsellable by the engine.
    """
    social_df = social_mod.score_symbols(store, cfg).set_index("symbol")
    catalyst_df = catalysts_mod.score_symbols(store, cfg).set_index("symbol")
    weights = cfg.weights.normalized()

    universe = set(cfg.symbols)
    targets = list(cfg.symbols) + [s for s in (extra_symbols or []) if s not in universe]

    rows: list[dict[str, Any]] = []
    for symbol in targets:
        try:
            short_raw, _ = prices.get_ohlcv(symbol, "1d", cfg)
            med_raw, _ = prices.get_ohlcv(symbol, "1w", cfg)
            # Daily candles. Needed for stop sizing: ATR on the 1w view is the
            # range of a *1-hour* bar, which for a 1-7 day hold produces stops
            # under 1% that get taken out by ordinary intraday noise.
            long_raw, _ = prices.get_ohlcv(symbol, "1y", cfg)
        except Exception as exc:  # noqa: BLE001 - drop the asset, note why
            if verbose:
                print(f"  {symbol}: no price data ({exc})")
            continue

        short = indicators.summarize(indicators.enrich(short_raw, cfg.bands))
        medium = indicators.summarize(indicators.enrich(med_raw, cfg.bands))
        daily = indicators.summarize(indicators.enrich(long_raw, cfg.bands))
        if not short or not medium:
            continue

        tech, tech_parts = technical_score(short, medium)
        soc = float(social_df.loc[symbol, "social"]) if symbol in social_df.index else 0.0
        cat = float(catalyst_df.loc[symbol, "catalyst"]) if symbol in catalyst_df.index else 0.0
        composite = weights.technical * tech + weights.social * soc + weights.catalyst * cat

        components = {
            "technical_parts": tech_parts,
            "social": social_df.loc[symbol].to_dict() if symbol in social_df.index else {},
            "catalyst": catalyst_df.loc[symbol].to_dict() if symbol in catalyst_df.index else {},
            "snapshot_1d": short,
            "snapshot_1w": medium,
            "snapshot_daily": daily,
        }

        rows.append({
            "symbol": symbol,
            "held_only": symbol not in universe,
            "technical": round(tech, 4),
            "social": round(soc, 4),
            "catalyst": round(cat, 4),
            "composite": round(composite, 4),
            "w_technical": round(weights.technical * tech, 4),
            "w_social": round(weights.social * soc, 4),
            "w_catalyst": round(weights.catalyst * cat, 4),
            "price": round(medium["close"], 6),
            "atr_pct_1w": round(medium.get("atr_pct", 0.0), 5),
            "atr_pct_daily": round(daily.get("atr_pct", medium.get("atr_pct", 0.02)), 5),
            "rsi_1w": round(medium.get("rsi_14", 50.0), 1),
            "bb_pctb_1w": round(medium.get("bb_pctb", 0.5), 3),
            "mentions_24h": int(social_df.loc[symbol, "mentions_24h"]) if symbol in social_df.index else 0,
            "n_events": int(catalyst_df.loc[symbol, "n_events"]) if symbol in catalyst_df.index else 0,
            "top_event": catalyst_df.loc[symbol, "top_event"] if symbol in catalyst_df.index else "",
            "components": components,
        })

    df = pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)
    df.attrs["social_ready"] = bool(social_df.attrs.get("baseline_ready", False)) if hasattr(social_df, "attrs") else False
    return df


# --------------------------------------------------------------------------
# Rationale
# --------------------------------------------------------------------------
def _rationale(row: pd.Series, side: str, cfg: Config) -> str:
    c = row["components"]
    tp = c.get("technical_parts", {})
    sc = c.get("social", {})
    ct = c.get("catalyst", {})
    snap = c.get("snapshot_1w", {})
    bits: list[str] = []

    drivers = sorted(
        [("technical", row["w_technical"]), ("social", row["w_social"]), ("catalyst", row["w_catalyst"])],
        key=lambda kv: abs(kv[1]), reverse=True,
    )
    lead, lead_val = drivers[0]
    bits.append(f"Driven mainly by {lead} ({lead_val:+.3f} of {row['composite']:+.3f} composite).")

    if abs(tp.get("trend", 0)) > 0.15:
        bits.append(f"Trend {tp['trend']:+.2f} (price {snap.get('dist_sma_20', 0)*100:+.1f}% vs SMA20).")
    if abs(tp.get("momentum", 0)) > 0.15:
        bits.append(f"Momentum {tp['momentum']:+.2f}.")
    bits.append(f"RSI(1w) {snap.get('rsi_14', 50):.0f}, %B {snap.get('bb_pctb', 0.5):.2f}.")
    if tp.get("squeeze"):
        bits.append("Bollinger squeeze active — expansion more likely, direction not implied by it.")

    n = sc.get("mentions_24h", 0)
    if n:
        bits.append(
            f"Reddit: {int(n)} mentions/24h from {int(sc.get('unique_authors', 0))} accounts, "
            f"velocity {sc.get('velocity', 0):+.2f} (log2 vs baseline), "
            f"tone {sc.get('avg_sentiment', 0):+.2f}."
        )
    else:
        bits.append("Reddit: no mentions in window.")

    if ct.get("n_events"):
        bits.append(
            f"Catalysts: {int(ct['n_events'])} events "
            f"({int(ct.get('n_github', 0))} dev / {int(ct.get('n_news', 0))} news), "
            f"strongest '{ct.get('top_event', '')}' at {ct.get('top_impact', 0):.2f}."
        )
    else:
        bits.append("Catalysts: none found in window.")

    if side == "SELL":
        bits.append("Bearish score on an existing holding — proposed as an exit, not a short.")
    return " ".join(bits)


# --------------------------------------------------------------------------
# Proposals
# --------------------------------------------------------------------------
def propose(scores: pd.DataFrame, store: Store, run_id: str, cfg: Config = CONFIG,
            holdings: dict[str, float] | None = None,
            cash_available: float | None = None,
            exit_signals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Turn scores into at most `cfg.risk.max_proposals` sized trade candidates.

    Sizing is risk-first: the stop distance comes from ATR, and quantity is set
    so that being stopped out costs `risk_per_trade_pct` of equity - never a
    fixed dollar amount per trade, which silently takes far more risk on
    volatile names.

    `exit_signals` (from exits.evaluate) are emitted first, into their own
    reserved slots, and are NOT subject to `min_composite_score` - a stop that
    has already been hit is a fact about the position, not an opinion about its
    ranking. Entries then fill the remaining `max_proposals` budget, so exits and
    entries never compete for the same slots.

    `cash_available` is the real spendable balance (paper cash, or the free
    quote-asset balance on Binance). Total BUY notional is capped to it, so the
    slate can never propose spending money that isn't there - `account_equity_usd`
    is only the *risk basis* and may be larger than what's actually liquid.
    Pass None when the balance can't be read; the cap is then skipped rather
    than guessed, and preflight still warns per-trade.

    SELLs are exempt from both the cash cap and the deployment cap: an exit
    releases capital rather than consuming it.
    """
    risk = cfg.risk
    holdings = holdings or {}

    def _empty(reason: dict) -> pd.DataFrame:
        df = pd.DataFrame()
        df.attrs["skipped"] = reason
        df.attrs["cash_available"] = cash_available
        df.attrs["min_notional_usd"] = risk.min_notional_usd
        df.attrs["min_composite_score"] = risk.min_composite_score
        return df

    if scores.empty:
        return _empty({"no_scores": 1})

    candidates: list[dict[str, Any]] = []
    exiting: set[str] = set()
    # Why candidates were dropped. An empty slate is a legitimate outcome, but
    # "no proposals" is only useful if it says which constraint bound - the
    # earlier version blamed the score threshold even when the real blocker was
    # an empty wallet, which is a misdiagnosis, not a shortcut.
    skipped: dict[str, int] = {"below_score": 0, "bearish_unheld": 0,
                               "below_min_notional": 0, "flagged_for_exit": 0}
    # Every symbol under an exit signal, whether or not the exit itself makes it
    # into a slot. A dust-sized position can fail the minimum-notional check and
    # so produce no sellable ticket - but it must still never come back as a BUY
    # in the same run. Proposing an entry on something the exit logic just voted
    # to close is incoherent, and the earlier version did exactly that.
    flagged_for_exit: set[str] = set()
    if exit_signals is not None and not exit_signals.empty:
        flagged_for_exit = set(exit_signals["symbol"])

    # ---- exits first, in their own slots -------------------------------
    if exit_signals is not None and not exit_signals.empty and cfg.exits.max_exit_proposals > 0:
        by_symbol = {r["symbol"]: r for _, r in scores.iterrows()} if not scores.empty else {}
        for _, sig in exit_signals.head(cfg.exits.max_exit_proposals).iterrows():
            symbol = sig["symbol"]
            qty = min(float(sig["qty"]), float(holdings.get(symbol, 0.0)))
            last = sig.get("last")
            row = by_symbol.get(symbol)
            price = float(last) if last else (float(row["price"]) if row is not None else 0.0)
            if qty <= 0 or price <= 0:
                continue
            notional = qty * price
            if notional < risk.min_notional_usd:
                print(f"  [exit] {symbol}: {sig['trigger_label']} fired but the position "
                      f"is worth ${notional:,.2f}, under the ${risk.min_notional_usd:,.2f} "
                      f"minimum — cannot be sold. Left open; excluded from new entries.")
                continue
            exiting.add(symbol)
            candidates.append({
                "symbol": symbol, "side": "SELL",
                "composite": float(sig["composite"]) if pd.notna(sig.get("composite")) else 0.0,
                "entry": price,
                # An exit already met its condition; the stop/target on the
                # ticket are the levels that fired, not new ones to defend.
                "stop": float(sig["stop"]) if pd.notna(sig.get("stop")) else price,
                "target": float(sig["target"]) if pd.notna(sig.get("target")) else price,
                "qty": qty, "notional": notional, "risk_usd": 0.0,
                "row": row, "exit_signal": sig,
            })

    # ---- capital released by the exits ---------------------------------
    # Exits are ranked first AND executed first, so their proceeds are actually
    # spendable by the entries in this same run. Sizing buys off the pre-sale
    # balance would leave capital idle for no reason. The haircut absorbs
    # slippage and fees; if a sell is then rejected at the approval prompt, the
    # execution-time cash check still blocks the buy it was funding.
    exit_proceeds = 0.0
    if cash_available is not None:
        gross = sum(c["notional"] for c in candidates if c["side"] == "SELL")
        exit_proceeds = gross * (1.0 - risk.exit_proceeds_haircut_pct / 100.0)
        if gross > 0:
            print(f"  [capital] {len(exiting)} exit(s) release ≈${exit_proceeds:,.2f} "
                  f"(${gross:,.2f} less {risk.exit_proceeds_haircut_pct:.1f}% haircut); "
                  f"buying power ${cash_available:,.2f} → ${cash_available + exit_proceeds:,.2f}")
    buying_power = None if cash_available is None else cash_available + exit_proceeds

    # ---- then entries, from the score ranking ---------------------------
    ranked = scores.sort_values("composite", key=abs, ascending=False)

    for _, row in ranked.iterrows():
        if len(candidates) - len(exiting) >= risk.max_proposals:
            break
        composite = float(row["composite"])
        symbol = row["symbol"]
        if symbol in flagged_for_exit:
            skipped["flagged_for_exit"] += 1
            continue            # closing (or flagged to close) — never re-enter
        if abs(composite) < risk.min_composite_score:
            skipped["below_score"] += 1
            continue

        held = float(holdings.get(symbol, 0.0))
        if composite > 0:
            side = "BUY"
        elif held > 0:
            side = "SELL"
        else:
            skipped["bearish_unheld"] += 1
            continue        # bearish with no position: spot can't short — skip

        entry = float(row["price"])
        # Stop distance is scaled to DAILY volatility, matching the 1-7 day
        # horizon. The floor keeps a freakishly quiet asset from producing a
        # stop so tight that position size explodes.
        atr_daily = float(row.get("atr_pct_daily") or row["atr_pct_1w"])
        atr_abs = max(atr_daily * entry, entry * 0.01)
        stop_dist = risk.atr_stop_mult * atr_abs

        if side == "BUY":
            stop = entry - stop_dist
            target = entry + stop_dist * risk.reward_risk_target
        else:
            stop = entry + stop_dist
            target = entry - stop_dist * risk.reward_risk_target

        risk_budget = risk.account_equity_usd * (risk.risk_per_trade_pct / 100.0)
        qty = risk_budget / stop_dist if stop_dist > 0 else 0.0

        max_notional = risk.account_equity_usd * (risk.max_position_pct / 100.0)
        if side == "BUY" and buying_power is not None:
            # No single purchase may exceed what will actually be spendable
            # once this run's exits have settled.
            max_notional = min(max_notional, buying_power)
        if qty * entry > max_notional:
            qty = max_notional / entry

        if side == "SELL":
            qty = min(qty, held)          # never sell more than you hold

        notional = qty * entry
        if notional < risk.min_notional_usd or qty <= 0:
            skipped["below_min_notional"] += 1
            continue

        candidates.append({
            "symbol": symbol, "side": side, "composite": composite,
            "entry": entry, "stop": stop, "target": target,
            "qty": qty, "notional": notional,
            "risk_usd": min(risk_budget, notional),
            "row": row, "exit_signal": None,
        })

    if not candidates:
        return _empty(skipped)

    # Slate-level caps. Both apply to BUYs only - a SELL frees capital, so
    # scaling exits down to respect a *deployment* limit is backwards.
    #
    # Three individually-affordable buys can still collectively overdraw the
    # account, so the binding constraint is the sum, not the max.
    buys = [c for c in candidates if c["side"] == "BUY"]
    buy_total = sum(c["notional"] for c in buys)
    limits = [("deployment", risk.account_equity_usd * (risk.max_total_deployed_pct / 100.0))]
    if buying_power is not None:
        limits.append(("cash", float(buying_power)))

    binding, cap = min(limits, key=lambda kv: kv[1])
    if buy_total > cap > 0:
        scale = cap / buy_total
        for c in buys:
            c["qty"] *= scale
            c["notional"] *= scale
            c["risk_usd"] *= scale
        candidates = [c for c in candidates if c["notional"] >= risk.min_notional_usd]
        print(f"  [{binding} cap] BUY total ${buy_total:,.2f} → ${cap:,.2f} "
              f"(×{scale:.3f}); {len(candidates)} proposal(s) remain above the "
              f"${risk.min_notional_usd:,.2f} minimum.")
    elif cap <= 0 and buys:
        candidates = [c for c in candidates if c["side"] == "SELL"]
        print(f"  [{binding} cap] no spendable balance — all BUY proposals dropped.")

    if not candidates:
        return _empty(skipped)

    from . import exits as exits_mod

    ts = iso()
    out: list[dict[str, Any]] = []
    for rank, c in enumerate(candidates, start=1):
        row = c["row"]
        sig = c.get("exit_signal")
        is_exit = sig is not None
        rr = abs(c["target"] - c["entry"]) / max(abs(c["entry"] - c["stop"]), 1e-9)

        if row is not None:
            payload = {
                "components": row["components"],
                "scores": {"technical": row["technical"], "social": row["social"],
                           "catalyst": row["catalyst"], "composite": row["composite"]},
                "weights": cfg.weights.normalized().__dict__,
            }
        else:
            # Held asset with no score row (e.g. price fetch failed). The exit
            # still stands - it fired on recorded entry terms, not on a score.
            payload = {"components": {}, "scores": {}, "note": "no score row"}
        if is_exit:
            payload["exit"] = {k: (None if pd.isna(v) else v)
                               for k, v in sig.to_dict().items()}

        out.append({
            "proposal_id": f"{run_id}-{rank}-{uuid.uuid4().hex[:6]}",
            "run_id": run_id, "ts": ts, "symbol": c["symbol"], "side": c["side"],
            "rank": rank, "composite": round(c["composite"], 4),
            "entry": round(c["entry"], 8), "stop": round(c["stop"], 8),
            "target": round(c["target"], 8),
            "qty": float(c["qty"]), "notional": round(c["notional"], 2),
            "risk_usd": round(c["risk_usd"], 2),
            "reward_risk": 0.0 if is_exit else round(rr, 2),
            "kind": "exit" if is_exit else "entry",
            "trigger": sig["trigger_label"] if is_exit else "",
            "horizon": "close now" if is_exit else "1-7 days",
            "rationale": (exits_mod.describe(sig, cfg) if is_exit
                          else _rationale(row, c["side"], cfg)),
            "payload": json.dumps(payload, default=str),
            "decision": "pending",
        })

    df = pd.DataFrame(out)
    store.save_proposals(
        df.drop(columns=["risk_usd", "reward_risk", "kind", "trigger"]).to_dict("records"))
    return df


def fmt_price(x: float) -> str:
    """Format a price with enough precision to actually be reviewable.

    A fixed 6dp renders SHIB as 0.000005, which is useless for approving a
    trade — decimals scale with magnitude instead.
    """
    x = float(x)
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:,.4f}"
    if ax >= 0.01:
        return f"{x:,.6f}"
    if ax >= 0.0001:
        return f"{x:,.8f}"
    return f"{x:.10f}"


def fmt_qty(x: float) -> str:
    x = float(x)
    return f"{x:,.2f}" if abs(x) >= 1000 else f"{x:,.6f}"


def format_proposals(proposals: pd.DataFrame) -> str:
    """Readable summary for the approval prompt."""
    if proposals.empty:
        sk = proposals.attrs.get("skipped", {}) or {}
        cash = proposals.attrs.get("cash_available")
        min_notional = proposals.attrs.get("min_notional_usd", 0.0)
        min_score = proposals.attrs.get("min_composite_score", 0.0)

        lines = ["No proposals this run."]
        # Lead with the constraint that actually bound. An empty wallet and a
        # flat scoreboard are very different problems and need different fixes.
        if sk.get("below_min_notional"):
            n = sk["below_min_notional"]
            lines.append(
                f"  · {n} candidate(s) scored well enough but sized below the "
                f"${min_notional:,.2f} minimum order."
            )
            if cash is not None and cash < min_notional:
                lines.append(
                    f"    Spendable balance is ${cash:,.2f} — that is the binding "
                    f"constraint, not the scores. Add quote-asset funds, or lower "
                    f"CONFIG.risk.min_notional_usd if your venue allows smaller orders."
                )
        if sk.get("below_score"):
            lines.append(f"  · {sk['below_score']} scored under "
                         f"min_composite_score ({min_score:+.2f}).")
        if sk.get("bearish_unheld"):
            lines.append(f"  · {sk['bearish_unheld']} were bearish on assets you "
                         f"don't hold — spot can't short, so that's an avoid.")
        if sk.get("flagged_for_exit"):
            lines.append(f"  · {sk['flagged_for_exit']} are already flagged for exit.")
        if sk.get("no_scores"):
            lines.append("  · No assets could be scored at all (price fetch failed?).")
        if len(lines) == 1:
            lines.append("  · Nothing cleared the entry filters.")
        lines.append("An empty slate is a valid output; the engine is not required "
                     "to manufacture three trades every run.")
        return "\n".join(lines)
    lines: list[str] = []
    for _, p in proposals.iterrows():
        is_exit = p.get("kind") == "exit"
        header = (f"#{p['rank']}  ⟵ EXIT {p['symbol']}   [{p['trigger']}]"
                  if is_exit else
                  f"#{p['rank']}  {p['side']} {p['symbol']}   "
                  f"composite {p['composite']:+.3f}")
        lines.append(f"\n{'─' * 78}\n{header}   horizon {p['horizon']}\n{'─' * 78}\n")
        if is_exit:
            lines.append(
                f"  price   {fmt_price(p['entry']):>16}\n"
                f"  size    {fmt_qty(p['qty']):>16} {p['symbol']}  "
                f"≈ ${p['notional']:,.2f}\n"
            )
        else:
            move_pct = (p["target"] - p["entry"]) / p["entry"] * 100
            stop_pct = (p["stop"] - p["entry"]) / p["entry"] * 100
            lines.append(
                f"  entry   {fmt_price(p['entry']):>16}\n"
                f"  stop    {fmt_price(p['stop']):>16}   ({stop_pct:+.2f}%)\n"
                f"  target  {fmt_price(p['target']):>16}   ({move_pct:+.2f}%)   "
                f"R:R {p['reward_risk']:.2f}\n"
                f"  size    {fmt_qty(p['qty']):>16} {p['symbol']}  ≈ ${p['notional']:,.2f}   "
                f"risk ≈ ${p['risk_usd']:,.2f}\n"
            )
        lines.append(f"\n  {p['rationale']}\n")
    return "".join(lines)
