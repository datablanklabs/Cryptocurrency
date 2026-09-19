#!/usr/bin/env python3
"""Generate crypto_yolo_dashboard.ipynb.

The notebook is generated rather than hand-edited so it stays diffable and
regenerable: change a cell here, re-run this script, get a clean notebook with
no stale execution counts or embedded output.

    python3 build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).parent / "crypto_yolo_dashboard.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


CELLS = [
md(r"""
# crypto-yolo — trading dashboard

> **This notebook is generated from `build_notebook.py` and ships without saved
> outputs.** If it looks empty, that's expected — **Run → Run All Cells**. Edits
> belong in `build_notebook.py`; re-run it to regenerate this file.

Run all cells. The notebook refreshes five feature families, scores the
universe, proposes up to **3 ranked trade candidates**, asks you to approve each
one (auto-approval is opt-in, never the default), and executes the approved ones
through the Binance API.

Signals are short-horizon (1d/1w technicals, ~5-day news lookback). How long a
position may be *held* is separate: `CONFIG.exits.horizon_days`, default 30 —
a backstop, since stop, target, trailing and score-reversal usually fire first.

| Feature | Source | Feeds |
|---|---|---|
| **1 · Price & bands** | Binance.US → Coinbase → Kraken → yfinance | technical score (+ cross-sectional momentum) |
| **2 · Social sentiment** | 13 subreddits + StockTwits + Mastodon + /biz/ | social score |
| **3 · Dev & news catalysts** | GitHub releases, CoinDesk/Cointelegraph/Decrypt/Defiant | catalyst score |
| **4 · Positioning** | OKX perpetual funding rates | positioning score |
| **5 · Scheduled events** | hand-maintained calendar (unlocks, mainnets, ETF dates) | events score |
| **6 · Macro backdrop** | Kalshi event-contract prices (Fed, CPI, shutdown risk) | dampens the regime gate |

Sizing is against **live equity**, the stop is scaled to the holding horizon,
targets are widened to clear round-trip fees, and a **BTC-trend regime gate**
scales the whole slate (blocking new buys in a downtrend). The
**Evaluation & benchmark** section near the end is the feedback loop: it scores
the engine against realised returns and against buy-and-hold BTC.

---

### Read this once

**The engine ranks; it does not predict.** No system can identify the three
trades that *will* maximise profit over the next week. What this does is score
every asset with a transparent, hand-tuned formula, show you every component
that went into each score, and size positions against a stated risk budget. The
weights in `ScoreWeights` are priors chosen by hand — **they have not been
fitted to realised returns.** The Evaluation section measures whether they hold
up; until it says so, treat the output as a research shortlist that shows its
work, not as a forecast.

**Signal quality, honestly.** Reddit mention volume is trivially gamed and
mostly lagging — by the time a coin trends on r/wsb, the move usually already
happened. That is why social carries a low weight. The genuinely forward-looking
family is **events**: a token unlock dated next Tuesday is knowable now in a way
a price move is not (it ships empty — you maintain the schedule).
**Positioning** is the only family that goes reliably negative on its own —
without it the composite drifts toward rating everything a buy.

**Execution is gated.** Default mode is `paper` — simulated fills, nothing
leaves your machine. Real orders need `mode="binance"`, `dry_run=False`, **and**
the environment variable `CRYPTO_YOLO_ALLOW_LIVE=1`. Two switches in two places
that must agree, so re-running a cell can never place a live order by itself.

This is not financial advice. You approve every trade; you own every outcome.
"""),

md("## Setup"),

code(r"""
import sys, warnings
from pathlib import Path

import pandas as pd
import plotly.io as pio
from IPython.display import clear_output, display

sys.path.insert(0, str(Path.cwd()))
warnings.filterwarnings("ignore")

# Drop any previously-imported cryptoyolo modules before importing.
#
# Python caches modules in sys.modules, so if the package changed on disk after
# this kernel first imported it, re-running this cell would silently keep the
# OLD code — and you'd get confusing failures like
# "'Config' object has no attribute 'exits'" from a config cell that looks
# correct. Purging here means re-running Setup always picks up the current
# source, no kernel restart needed.
for _stale in [m for m in list(sys.modules)
               if m == "cryptoyolo" or m.startswith("cryptoyolo.")]:
    del sys.modules[_stale]
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pio.renderers.default = "notebook"

from cryptoyolo import (charts, engine, evaluation, indicators, pipeline, prices,
                        regime, scheduler)
from cryptoyolo import calendar_events as events_mod
from cryptoyolo import catalysts as catalysts_mod
from cryptoyolo import feeds as feeds_mod
from cryptoyolo import macro as macro_mod
from cryptoyolo import social as social_mod
from cryptoyolo.config import CONFIG, TIMEFRAMES, load_dotenv
from cryptoyolo.store import Store

load_dotenv()                      # reads ./.env if present; never overwrites real env vars
store = Store(CONFIG.db_path)

# Fail loudly and legibly if the loaded code is still older than this notebook,
# rather than letting a later cell die on a missing attribute.
_required = {"exits": "exit management", "risk": "risk sizing",
             "execution": "order execution", "fees": "fee model",
             "regime": "regime gate", "events": "scheduled events",
             "macro": "macro backdrop (Kalshi)"}
_missing = [f"CONFIG.{a} ({why})" for a, why in _required.items()
            if not hasattr(CONFIG, a)]
if _missing:
    raise RuntimeError(
        "Loaded cryptoyolo is out of date — missing: " + ", ".join(_missing)
        + ".\nRestart the kernel (Kernel → Restart Kernel and Run All Cells)."
    )

print(f"database   : {CONFIG.db_path}")
print(f"universe   : {len(CONFIG.symbols)} assets")
print(f"reddit hist: {store.history_span_hours():.1f}h accumulated\n")
pipeline.status(CONFIG)
"""),

md(r"""
### Credentials

Nothing is hard-coded. Create a `.env` next to this notebook (see `.env.example`):

| Variable | Needed for | Without it |
|---|---|---|
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Feature 2 | falls back to keyless sources — still works |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Execution | paper mode still works fully |
| `GITHUB_TOKEN` | Feature 3 | works, but 60 req/hr caps the universe scan |
| `KALSHI_API_KEY_ID` / `KALSHI_PRIVATE_KEY_PATH` | Feature 6 | macro fetch skips; regime gate runs on BTC trend alone |
| `CRYPTO_YOLO_ALLOW_LIVE=1` | Live orders | orders stay validate-only |

Reddit credentials are **optional** — the scraper falls back to Arctic Shift (a
public Pushshift successor) and Reddit's own Atom feeds when they're absent. They
are free if you want the better feed: <https://www.reddit.com/prefs/apps> →
*create app* → type **script** → copy the id under the app name and the secret.
"""),

md("## Configuration"),

code(r"""
# ── Risk ────────────────────────────────────────────────────────────────
# account_equity_usd is the FALLBACK risk basis — used only when live equity
# can't be read. With size_off_live_equity=True (default) the risk budget is
# recomputed each cycle from cash + marked-to-market positions, so the book
# compounds after a good run and de-risks in a drawdown on its own.
CONFIG.risk.account_equity_usd    = 10_000.0
CONFIG.risk.size_off_live_equity  = True
CONFIG.risk.risk_per_trade_pct    = 1.0        # % of equity lost if stopped out
CONFIG.risk.max_position_pct      = 20.0       # cap per position
CONFIG.risk.max_total_deployed_pct = 60.0     # cap across all proposals (× regime scale)
CONFIG.risk.reward_risk_target    = 2.0        # target = 2R (net of fees if fee_adjust_targets)
CONFIG.risk.fee_adjust_targets    = True       # widen the target so NET R:R = 2
CONFIG.risk.max_proposals         = 3

# Stop distance vs the HOLDING horizon. "horizon" scales 1.5×daily-ATR by
# sqrt(stop_horizon_days) so a multi-week thesis isn't stopped by one day's
# noise; "legacy" is the old 1.5×daily-ATR stop. Wider stop ⇒ smaller size for
# the same 1% $ risk. Tune stop_horizon_days / atr_stop_mult from the eval loop.
CONFIG.risk.stop_scaling          = "horizon"  # "horizon" | "legacy"
CONFIG.risk.stop_horizon_days     = 10.0
CONFIG.risk.atr_stop_mult         = 1.5
CONFIG.risk.stop_min_pct          = 3.0        # clamp: floor on stop distance
CONFIG.risk.stop_max_pct          = 40.0       # clamp: cap on stop distance
CONFIG.risk.target_pct_cap        = 60.0       # a clamped-wide stop can't imply an unreachable target

# Candidate selection. "relative" takes the top max_proposals of the
# cross-section each run (so a broad down day still surfaces the best relative
# longs, a broad up day doesn't wave everything through); "absolute" keeps the
# old fixed |composite| ≥ min_composite_score gate.
CONFIG.risk.selection_mode        = "relative"   # "relative" | "absolute"
CONFIG.risk.min_composite_floor   = 0.03         # relative mode: never trade pure noise
CONFIG.risk.min_composite_score   = 0.10         # absolute mode only
CONFIG.risk.trim_threshold        = 0.10         # trim a holding only once this bearish (both modes)

# Correlation-aware sizing. Caps sqrt(r' C r) over the new BUY slate — three
# 1%-risk alt longs that move together are really one ~2.5% bet. Scales the
# whole slate down if the correlation-adjusted risk exceeds this % of equity.
CONFIG.risk.correlation_sizing    = True
CONFIG.risk.max_portfolio_heat_pct = 2.5
CONFIG.risk.corr_lookback_days    = 60

# ── Fees & slippage (the return you give away on every trade) ───────────
# Binance.US base tier is ~40 bps taker/maker — a ~0.8% round trip. Enable the
# BNB discount for a 25% cut, earn a lower tier with volume, and prefer the
# marketable-limit entry below over MARKET. slippage_bps is applied to PAPER
# fills so the paper record isn't optimistic vs live.
CONFIG.fees.enabled           = True
CONFIG.fees.taker_bps         = 40.0
CONFIG.fees.maker_bps         = 40.0
CONFIG.fees.use_bnb_discount  = False       # set True if you hold BNB for fees
CONFIG.fees.slippage_bps      = 6.0

# ── Market-regime gate (BTC trend) ─────────────────────────────────────
# Scales the whole slate's deployment by where BTC sits vs its long MA, and
# blocks new BUYs outright in a hard downtrend. Exits are never gated.
CONFIG.regime.enabled                        = True
CONFIG.regime.ma_days                        = 200
CONFIG.regime.ma_band_pct                    = 2.0   # dead-band so an MA chop doesn't flip the gate
CONFIG.regime.neutral_exposure               = 0.5
CONFIG.regime.risk_off_exposure              = 0.0
CONFIG.regime.block_new_entries_when_risk_off = True

# ── Macro backdrop (Feature 6 — Kalshi, dampens the regime gate above) ──
# Only ever DAMPENS exposure_scale, never boosts it. A macro score at/below
# macro_risk_off_threshold forces risk_off outright; anything less extreme
# scales exposure down by up to (1 - macro_downweight), floored at
# macro_min_multiplier. Ships a no-op: config.py::MACRO_SERIES is empty
# until you populate it with real Kalshi series tickers (see Feature 6).
CONFIG.regime.macro_enabled            = True
CONFIG.regime.macro_risk_off_threshold = -0.6
CONFIG.regime.macro_downweight         = 0.5
CONFIG.macro.enabled                   = True
CONFIG.macro.min_volume                = 1       # ignore untraded (meaningless-price) markets
CONFIG.macro.stale_after_hours         = 36.0    # older snapshot = treated as no-data

# ── Scheduled events (Feature 5 — the one forward-looking family) ───────
# Token unlocks, mainnet dates, ETF decisions, listing effective dates. The
# schedule is hand-maintained in cryptoyolo/config.py::SCHEDULED_EVENTS — it
# ships EMPTY, so this family scores 0 until you populate it. Weight is set
# below alongside the others.
CONFIG.events.enabled          = True
CONFIG.events.lookahead_days   = 30.0
CONFIG.events.peak_window_days = 7.0

# ── Exits: when to close an existing position ───────────────────────────
# Five independent triggers, each switchable. Evaluated every run against live
# prices and the terms recorded when the position was opened — so an exit no
# longer depends on the asset happening to rank well on the buy-side scoreboard.
CONFIG.exits.enabled                  = True
CONFIG.exits.stop_loss                = True    # price through the stop set at entry
CONFIG.exits.take_profit              = True    # price reached the target
CONFIG.exits.take_profit_fraction     = 100.0   # % of position to sell on a target hit
CONFIG.exits.trailing_stop            = True
CONFIG.exits.trail_pct                = 8.0     # max giveback from the high-water mark
CONFIG.exits.trail_activate_pct       = 3.0     # only arm once this far in profit
CONFIG.exits.horizon_expiry           = True    # the trade thesis ran out of time
CONFIG.exits.horizon_days             = 30.0
CONFIG.exits.score_reversal           = True    # the thesis inverted
CONFIG.exits.score_reversal_threshold = -0.15
CONFIG.exits.max_exit_proposals       = 3       # exits get their OWN slots

# ── Score weights (must be defensible to you, not to me) ────────────────
# Hand-set priors. Once the DB holds a few weeks of runs, the Evaluation
# section near the end prints a data-driven ScoreWeights from measured ICs —
# use that instead of trusting these. normalized() divides by the sum, so any
# weight can go to 0.0 to drop that family.
CONFIG.weights.technical   = 0.50
CONFIG.weights.social      = 0.20
CONFIG.weights.catalyst    = 0.30
CONFIG.weights.positioning = 0.10
CONFIG.weights.events      = 0.0    # scheduled dated catalysts — raise to ~0.15 once SCHEDULED_EVENTS is filled
# Cross-sectional momentum blended into the technical score (not a family):
# technical_final = 0.7·technical + 0.3·xsec_rank. 0.0 restores the isolated score.
CONFIG.weights.xsec_momentum_blend = 0.30

# ── Positioning: perpetual funding as a crowding measure ────────────────
# Contrarian by default — crowded positioning is fragile positioning, so
# unusually high funding scores bearish. contrarian=False reads it as momentum.
CONFIG.positioning.enabled     = True
CONFIG.positioning.contrarian  = True
CONFIG.positioning.z_scale     = 1.5    # tanh knee, in standard deviations
CONFIG.positioning.min_periods = 20     # below this, score 0 rather than noise

# ── Chart bands ─────────────────────────────────────────────────────────
CONFIG.bands.bollinger        = True
CONFIG.bands.bollinger_window = 20
CONFIG.bands.bollinger_stds   = (1.0, 2.0)
CONFIG.bands.keltner          = False
CONFIG.bands.donchian         = False
CONFIG.bands.moving_averages  = (20, 50)
CONFIG.bands.show_volume      = True
CONFIG.bands.show_rsi         = True

# ── Execution ───────────────────────────────────────────────────────────
# "paper"   → simulated fills, nothing sent anywhere            (default)
# "binance" → real API; still validate-only unless BOTH
#             dry_run=False AND env CRYPTO_YOLO_ALLOW_LIVE=1
CONFIG.execution.mode        = "paper"
CONFIG.execution.venue       = "binance-us"   # binance-com / binance-test are geo-blocked from US IPs
CONFIG.execution.dry_run     = True
CONFIG.execution.order_type  = "MARKET"       # EXITS must fill, so they stay MARKET
CONFIG.execution.quote_asset = "USDT"
# Entries go as a marketable IOC LIMIT: crosses the book by at most
# entry_limit_cross_bps (fills now, usually as taker), and the unfilled part is
# cancelled rather than left resting — a resting entry would desync the position
# book. A partial fill is kept and the position is sized to it. Set
# entry_order_type="MARKET" for the old unbounded-slippage behaviour.
CONFIG.execution.entry_order_type      = "LIMIT"
CONFIG.execution.entry_limit_cross_bps = 15.0
# Refuse an entry if the live price has gapped this far above the proposal
# price. Read via CONFIG.execution.entry_slippage_guard_bps, which keeps it a
# margin above entry_limit_cross_bps so the two can't drift out of sync.
CONFIG.execution.max_entry_slippage_bps = 60.0

# ── Protective resting orders (continuous, venue-side protection) ────────
# The exits pass only sees the market when you run the notebook. An order
# resting AT Binance is watched by Binance continuously — that is the only way
# an overnight stop breach gets acted on at the time it happens.
# Binance.US supports STOP_LOSS_LIMIT and TAKE_PROFIT_LIMIT but NOT market
# STOP_LOSS, so a protective stop is always a stop-LIMIT.
CONFIG.execution.place_stop_orders       = False   # rest a protective stop after entry
CONFIG.execution.place_limit_orders      = False   # rest a take-profit at the target
CONFIG.execution.place_stop_limit_orders = True    # stop leg is STOP_LOSS_LIMIT (required)
CONFIG.execution.use_oco                 = True    # send both as one OCO (see below)
CONFIG.execution.stop_limit_offset_bps   = 25.0    # limit sits this far through the trigger
CONFIG.execution.use_trailing_delta      = False   # let Binance trail the stop itself
CONFIG.execution.trailing_delta_bps      = 0       # 0 = derive from CONFIG.exits.trail_pct

w = CONFIG.weights.normalized()
print(f"weights    : technical {w.technical:.0%} · social {w.social:.0%} · catalyst {w.catalyst:.0%} "
      f"· positioning {w.positioning:.0%} · events {w.events:.0%}")
print(f"risk/trade : {CONFIG.risk.risk_per_trade_pct:.1f}% of "
      f"{'live equity' if CONFIG.risk.size_off_live_equity else f'${CONFIG.risk.account_equity_usd:,.0f}'}")
print(f"stop       : {CONFIG.risk.stop_scaling} "
      f"(≈{CONFIG.risk.atr_stop_mult}×dailyATR×√{CONFIG.risk.stop_horizon_days:.0f}d, "
      f"clamp {CONFIG.risk.stop_min_pct:.0f}–{CONFIG.risk.stop_max_pct:.0f}%)")
print(f"selection  : {CONFIG.risk.selection_mode}   ·   fees {CONFIG.fees.effective_taker_bps:.0f} bps taker "
      f"(round trip {CONFIG.fees.round_trip_bps:.0f} bps)")
print(f"heat cap   : {'on' if CONFIG.risk.correlation_sizing else 'OFF'}, "
      f"{CONFIG.risk.max_portfolio_heat_pct:.1f}% of equity (corr over {CONFIG.risk.corr_lookback_days}d)")

from cryptoyolo.broker import execution_banner
from cryptoyolo import regime as _regime
print(f"execution  : {execution_banner(CONFIG)}")
# store=store folds in the last Kalshi macro snapshot (if any) — see
# Feature 6 below. Without it this line would only ever show the BTC trend.
print(f"regime     : {_regime.describe(_regime.assess(CONFIG, store=store))}")
"""),

md(r"""
---
## Feature 0 · Account balance & holdings

What you actually own right now. Everything downstream depends on it — position
sizing, the cash cap on purchases, and whether a SELL is even possible (spot
can't short, so exits only exist for assets you hold).

Balance comes from paper cash in paper mode, and from your **free** quote-asset
balance on Binance otherwise. Cost basis is exact in paper mode; on a live
account only fills placed *through this dashboard* have a basis we can honestly
report, so anything bought elsewhere shows `—` rather than a guess.
"""),

code(r"""
from cryptoyolo import portfolio

snap = portfolio.snapshot(store, CONFIG)
print(portfolio.format_summary(snap))
"""),

code(r"""
# Same data as a frame, for sorting/filtering or feeding a chart.
positions = snap["positions"]
if positions.empty:
    print(f"No open positions. Cash: "
          + (f"${snap['cash']:,.2f}" if snap["cash"] is not None else "unavailable"))
else:
    display(positions.style.format({
        "qty": "{:,.6f}", "avg_cost": "{:,.6f}", "last": "{:,.6f}",
        "value": "${:,.2f}", "pnl": "${:+,.2f}", "pnl_pct": "{:+.2f}%",
        "weight_pct": "{:.1f}%",
    }, na_rep="—").hide(axis="index"))
"""),

md(r"""
### Exit rules in force

Exits are evaluated **before** the buy-side ranking, against each position's own
recorded entry terms — not against how attractive the asset looks today. That
separation is the point: a position drifting below its stop would otherwise only
surface if it out-ranked every buy candidate, which it rarely does.

When several fire at once, precedence is the order below. Stop and target come
first because they are level events that have *already happened* — the position
is at a price you previously said you would act on. Horizon and score reversal
are judgement calls and yield to them.

Positions opened outside this dashboard have no recorded stop, target or open
date. Rather than invent them, only score-reversal applies, and the exit says so.
"""),

code(r"""
from cryptoyolo import exits

display(exits.active_triggers(CONFIG).style.hide(axis="index"))

signals = exits.evaluate(store, CONFIG, snap["positions"].set_index("symbol")["qty"].to_dict()
                         if not snap["positions"].empty else {})
if signals.empty:
    print("\nNo exit trigger fired on current positions.")
else:
    print(f"\n{len(signals)} exit signal(s):\n")
    for _, sig in signals.iterrows():
        print(f"  {sig['symbol']:<6} [{sig['trigger_label']}] {sig['reason']}")
"""),

md(r"""
Sizing uses `CONFIG.risk.account_equity_usd` as the **risk basis** — how much
you're willing to lose per trade — which is deliberately separate from your
balance. If the two drift far apart the summary above says so. To sync it to
real equity:

```python
if snap["total_equity"]:
    CONFIG.risk.account_equity_usd = snap["total_equity"]
```

Purchases are capped at actual cash regardless, so leaving them out of sync
can't cause an overdraft — it only makes position sizes larger or smaller than
you probably intend.
"""),

md(r"""
### Continuous protection between runs

Everything in the exits section runs **only when you run the notebook**. A stop
breached at 3am is acted on at your next run, not when it happened. Closing that
gap needs an order resting at the venue, which is what
`place_stop_orders` / `place_limit_orders` do.

**Binance.US has no market `STOP_LOSS`** — its order types are `LIMIT`,
`LIMIT_MAKER`, `MARKET`, `STOP_LOSS_LIMIT` and `TAKE_PROFIT_LIMIT`. A protective
stop is therefore always a stop-*limit*, and that carries a real risk worth
naming: **in a gap or a fast flush the limit may not fill and the stop simply
does not protect you.** `stop_limit_offset_bps` places the limit below the
trigger to make a fill likelier; it cannot guarantee one.

**Both legs go as one OCO.** Two independent resting sells for the same quantity
would let both fill, or the second be rejected for insufficient balance — a
double-sell hazard, not protection. With `use_oco=False` and both flags on, only
the stop is placed and the notebook says so.

**Resting sells lock the asset.** An exit that tries to market-sell while a stop
rests would fail on free balance, so protection is cancelled automatically
before every SELL.

Two limits to keep in mind. Only *stop* and *target* can be delegated to the
venue — horizon expiry and score reversal are judgements this code makes, so
they still need a run. And protection is only real in **live** mode: paper and
validate-only record what would rest but place nothing (Binance offers no test
endpoint for OCO, so it cannot be dry-run validated at all).
"""),

code(r"""
ex = CONFIG.execution
print(f"protective orders : stop={ex.place_stop_orders}  target={ex.place_limit_orders}"
      f"  oco={ex.use_oco}  trailing_delta={ex.use_trailing_delta}")
print(f"stop leg type     : {'STOP_LOSS_LIMIT' if ex.place_stop_limit_orders else 'STOP_LOSS (unsupported on binance-us)'}")
print(f"limit offset      : {ex.stop_limit_offset_bps:.0f} bps through the trigger")
if (ex.place_stop_orders or ex.place_limit_orders) and not ex.live_enabled:
    print("\n⚠ Not in live mode — protective orders will be RECORDED but NOT placed. "
          "Nothing actually rests at the exchange.")

resting = store.resting_orders()
if resting.empty:
    print("\nNo protective orders currently resting.")
else:
    display(resting[["symbol", "kind", "order_type", "qty", "stop_price",
                     "limit_price", "target_price", "status", "placed_at"]]
            .style.hide(axis="index"))
"""),

md(r"""
---
## Feature 1 · Price charts with bands

Candles, Bollinger/Keltner/Donchian overlays, moving averages, volume and RSI
across **1 hour / 1 day / 1 week / 1 month / 1 year**.

Sources are tried in order and the first that answers wins, so a rate-limit or
a regional block on one venue doesn't stop the run. Note `api.binance.com` and
the Binance testnet return **HTTP 451 from US IPs** — `api.binance.us` is the
working default.
"""),

code(r"""
# Interactive: pick asset, timeframe, and overlays.
import ipywidgets as widgets

symbol_w = widgets.Dropdown(options=CONFIG.symbols, value="BTC", description="Asset:")
tf_w = widgets.ToggleButtons(
    options=[(TIMEFRAMES[k]["label"], k) for k in TIMEFRAMES],
    value="1w", description="Window:",
)
bb_w = widgets.Checkbox(value=True, description="Bollinger")
kc_w = widgets.Checkbox(value=False, description="Keltner")
dc_w = widgets.Checkbox(value=False, description="Donchian")
vol_w = widgets.Checkbox(value=True, description="Volume")
rsi_w = widgets.Checkbox(value=True, description="RSI")
out = widgets.Output()

def _draw(*_):
    CONFIG.bands.bollinger = bb_w.value
    CONFIG.bands.keltner = kc_w.value
    CONFIG.bands.donchian = dc_w.value
    CONFIG.bands.show_volume = vol_w.value
    CONFIG.bands.show_rsi = rsi_w.value
    with out:
        clear_output(wait=True)
        try:
            charts.price_chart(symbol_w.value, tf_w.value, CONFIG).show()
        except Exception as exc:
            print(f"Chart failed for {symbol_w.value} [{tf_w.value}]: {exc}")

for wdg in (symbol_w, tf_w, bb_w, kc_w, dc_w, vol_w, rsi_w):
    wdg.observe(_draw, names="value")

display(widgets.VBox([
    widgets.HBox([symbol_w, tf_w]),
    widgets.HBox([bb_w, kc_w, dc_w, vol_w, rsi_w]),
    out,
]))
_draw()
"""),

code(r"""
# Static equivalent — works in any nbconvert/HTML export where widgets don't.
charts.price_chart("ETH", "1mo", CONFIG).show()
"""),

code(r"""
# Whole universe at a glance, with the outer Bollinger envelope.
charts.grid(CONFIG.symbols[:12], timeframe="1w", cfg=CONFIG, cols=3).show()
"""),

md(r"""
---
## Feature 2 · Social sentiment & mention velocity

Pulls posts *and* comments across 13 subreddits, StockTwits, Mastodon and /biz/, extracts
asset mentions, scores sentiment with a crypto-native lexicon (`moon`, `rug`,
`rekt`, 🚀 — generic English sentiment models are useless here), and stores
everything so mention **velocity** can be measured against each asset's own
baseline.

**Two things to know about how this is scored:**

*Precision over recall.* `DOT`, `OP`, `NEAR`, `LINK`, `ATOM`, `ADA` and friends
are ordinary English words. Counting them naively yields a signal made almost
entirely of noise, so assets flagged `ambiguous` in the config match only as
`$DOT` or by full name (`polkadot`) — never as the bare word.

*Velocity, not volume.* Raw counts rank BTC and ETH first every single run.
The score uses the log-ratio of current chatter to that asset's own baseline,
damped by author diversity (so one person posting 40 times counts once-ish) and
by how much of the text carried actual directional language.

**Sources.** Reddit 403s anonymous `.json`, so the scraper falls through a
ladder: official OAuth API → **Arctic Shift** (public Pushshift successor,
keyless, carries scores) → **Reddit Atom feeds** (keyless, no scores) → legacy
`.json`. It prints which one served. No credentials required; they just give a
better feed. On the RSS tier vote scores are unavailable and engagement
weighting goes flat — you'll see a notice when that happens.

**Cold-start:** on a fresh database there is no baseline, so the first run's
social scores are ~0 by construction. That is correct behaviour, not a bug —
run the collector in the next cell for a day to make this feature meaningful.
"""),

code(r"""
stats = social_mod.scrape(store, CONFIG, verbose=True)

social_scores = social_mod.score_symbols(store, CONFIG)   # all sources
print(f"\nhistory: {social_scores.attrs['history_hours']:.1f}h · "
      f"baseline ready: {social_scores.attrs['baseline_ready']}")
if not social_scores.attrs["baseline_ready"]:
    print("→ Not enough history for velocity yet; social scores stay near 0 until "
          "the collector has run for ~1.5x the velocity window.")

social_scores.head(15)
"""),

md(r"""
### StockTwits & Mastodon

Two more keyless feeds, both measured before being wired in. Together they took
asset coverage from **10 to 18 of 20** in a single pass.

**StockTwits** cashtag streams arrive *pre-attributed to a symbol*, so the
mention matcher is bypassed entirely — no false positives to control for. About
57% carry a **user-declared** Bullish/Bearish label: a stated opinion rather than
one inferred from a word list.

**Mastodon** uses broad hashtags, not per-asset ones. Only wide tags carry
traffic (#crypto 3.6 posts/h); per-asset tags are effectively dead — #dot
returns one post per 33 hours — so a request per asset would buy month-old
content.

**Sentiment is de-biased per source.** StockTwits users self-label ~88% Bullish
(mean tone +0.55 vs Reddit's +0.05). Raw, that rated 18 of 20 assets strongly
bullish — and a signal that calls everything a buy cannot *rank* anything. Each
source's mean tone is subtracted before assets are compared, so what counts is
being bullish *relative to how bullish that platform always is*.
"""),

code(r"""
feed_stats = feeds_mod.scrape(store, CONFIG, verbose=True)

_sql = (
    "SELECT source, COUNT(*) AS mentions, COUNT(DISTINCT symbol) AS assets, "
    "ROUND(AVG(sentiment), 3) AS raw_tone "
    "FROM mentions WHERE created_utc >= strftime('%s','now') - 86400 "
    "GROUP BY source ORDER BY mentions DESC"
)
with store.conn() as _con:
    by_source = pd.read_sql_query(_sql, _con)
print("\nlast 24h by source (raw_tone is BEFORE de-biasing):")
display(by_source)
"""),

code(r"""
# Optional: keep collecting in the background while the kernel lives.
# For history that survives restarts, use cron instead (see the last section).
# collector = scheduler.start(store, CONFIG, interval_minutes=30)
# collector.status()
# scheduler.stop()
"""),

md(r"""
---
## Feature 3 · Developer activity & news catalysts

Two forward-looking sources: **GitHub** releases per asset repo (a chain that
just tagged a mainnet upgrade is objectively different from one whose repo has
been quiet), and **news RSS** classified against an event taxonomy — upgrade,
ETF, token unlock, exploit, lawsuit, listing, delay.

**Direction is not taken from keywords alone.** "ETF" is a *topic*, not a
direction: `record ETF inflows` and `ETF outflows accelerate` are both ETF news
pointing opposite ways. Topic-type events (`etf`, `regulation`, `listing`,
`partnership`, `institutional`) take their sign from the headline's own tone,
while intrinsically directional events (an exploit is never good news) keep
their prior and are merely discounted when the tone disagrees.

The impact weights in `EVENT_TYPES` are hand-set priors, not fitted
coefficients. They encode a defensible *ordering* — an exploit outweighs a
delay — but the magnitudes are judgement calls, collected in one place so you
can argue with them.
"""),

code(r"""
n_events = catalysts_mod.scan(store, CONFIG, verbose=True, include_github=True)

catalyst_scores = catalysts_mod.score_symbols(store, CONFIG)
catalyst_scores[catalyst_scores["n_events"] > 0].head(15)
"""),

code(r"""
# The individual events behind those scores, strongest first.
feed = catalysts_mod.recent_events(store, limit=20, cfg=CONFIG)
if feed.empty:
    print("No catalysts in the lookback window.")
else:
    display(feed.style.format({"signed": "{:+.3f}"}).hide(axis="index"))
"""),

md(r"""
---
## Feature 4 · Positioning (funding rates)

Not social sentiment, which is why it is a separate family. Social tells you what
people are *saying*; funding tells you what leveraged traders are *paying* to hold
a side. Positive funding means longs pay shorts — a crowded long.

It earns its own component because it is **the only input here that goes
genuinely negative on its own**. Over 100 periods (~33 days) SOL funding was
negative 27% of the time and ETH 25%, while every social source measured is
positive nearly always. Without it the composite drifts toward "everything is a
buy" — the same failure the per-source sentiment de-biasing had to correct.

Scored as a z-value against each asset's **own** funding history, not an
absolute threshold: 0.01% means something different for BTC than for a thin
altcoin. Same relative-to-own-baseline logic as mention velocity.
"""),

code(r"""
from cryptoyolo import positioning as positioning_mod

pos_scores = positioning_mod.score_symbols(store, CONFIG)
display(pos_scores.head(12).style.format({
    "funding_now": "{:+.4f}%", "funding_mean": "{:+.4f}%",
    "funding_z": "{:+.2f}", "positioning": "{:+.3f}",
}, na_rep="—").hide(axis="index"))

_neg = (pos_scores["positioning"] < -0.01).sum()
_pos = (pos_scores["positioning"] > 0.01).sum()
print(f"\ncrowded long (scores bearish): {_neg}   crowded short (scores bullish): {_pos}")
print("A uniform tilt shifts every composite together; the DISCRIMINATION comes")
print("from the z-magnitude, so compare assets to each other, not to zero.")
"""),

md(r"""
---
## Feature 6 · Macro backdrop (Kalshi event contracts)

Not a scored family like the five above — this one feeds the **regime gate**,
not the composite. Kalshi is a CFTC-regulated prediction market: a contract
settles at \$1 if a YES proposition resolves true, \$0 otherwise, so its live
price *is* a market-implied probability. `CONFIG` reads a hand-maintained list
of series (`config.py::MACRO_SERIES` — Fed decisions, CPI prints,
government-shutdown risk) and blends them into one score that can only ever
**dampen** the BTC-trend deployment scale, never boost it.

`MACRO_SERIES` ships empty — until you populate it with real Kalshi series
tickers (run `macro_mod.list_series(CONFIG, query="fed")` with working
credentials to find them), this section reports "no data" and the regime gate
runs exactly as it did before this feature existed.
"""),

code(r"""
n_macro = macro_mod.fetch(store, CONFIG, verbose=True)

macro_read = macro_mod.score(store, CONFIG)
print(f"\n{macro_mod.describe(macro_read)}")
if not macro_read["detail"].empty:
    display(macro_read["detail"].style.format({
        "probability": "{:.0%}", "contribution": "{:+.3f}", "weight": "{:.2f}",
    }).hide(axis="index"))
"""),

code(r"""
# Snapshot: where each configured series stands right now.
charts.macro_chart(macro_read["detail"]).show()

# History: how it got there. Builds up one point per macro.fetch() call (this
# cell, plus the 8-hourly collector job if it's running) — a single point is
# expected right after first setting this feature up.
charts.macro_history_chart(store.macro_history()).show()
"""),

md(r"""
---
## Decision engine

Composite = `w_technical × technical + w_social × social + w_catalyst × catalyst
+ w_positioning × positioning + w_events × events` — the normalised form of the
weights set in Configuration, each component in [-1, +1]. `technical` is itself
`0.7 × (per-asset technical) + 0.3 × (cross-sectional momentum rank)`, so a
column `technical_raw` shows the isolated score and `xsec` the universe-relative
part.

The one genuinely non-obvious piece of the per-asset technical score is how band
position is read: in a trending market, price riding the upper Bollinger band is
*strength*; in a range-bound market the identical reading is *stretched*. A
single fixed interpretation is wrong half the time, so trend strength decides.
"""),

code(r"""
scores = engine.build_scores(store, CONFIG, verbose=True)

display(scores[[
    "symbol", "composite", "technical", "technical_raw", "xsec", "social",
    "catalyst", "positioning", "events", "price", "rsi_1w", "atr_pct_daily",
    "mentions_24h", "n_events", "top_event",
]].style.format({
    "composite": "{:+.3f}", "technical": "{:+.3f}", "technical_raw": "{:+.3f}",
    "xsec": "{:+.2f}", "social": "{:+.3f}", "catalyst": "{:+.3f}",
    "positioning": "{:+.3f}", "events": "{:+.3f}", "price": "{:,.4f}",
    "atr_pct_daily": "{:.2%}",
}).background_gradient(subset=["composite"], cmap="RdYlGn", vmin=-0.6, vmax=0.6)
  .hide(axis="index"))
"""),

code(r"""
charts.score_chart(scores, template="plotly_white").show()
"""),

md(r"""
---
## Proposals → approval → execution

The cell below runs the full cycle and **prompts for each trade**. Answer `y` to
approve, `n` or Enter to reject, `q` to stop reviewing.

**Exits come first, and fund the entries.** Positions triggering an exit rule
are proposed *and executed* ahead of any new entry, and do **not** consume the
`max_proposals` budget. Their expected proceeds (less
`risk.exit_proceeds_haircut_pct` for slippage and fees) are added to buying
power, so the three buys can draw on capital the sells are about to release
instead of leaving it idle. If you then reject a sell, the execution-time cash
check still blocks the buy it was funding.
Exits are also exempt from `min_composite_score` — a stop that has been hit is a
fact about the position, not an opinion about its ranking. An asset under an
exit signal is never proposed as a buy in the same run, even if the exit itself
is too small to execute.

**Spot markets cannot short.** A bearish score on an asset you don't hold is not
a trade, it's an avoid — such assets appear in the ranking above but are never
proposed. Held assets are always scored even when they sit outside the
configured universe; otherwise they would be silently unsellable.

**Purchases can never exceed your cash.** Three checks, because each catches
something the others miss:

1. *Per trade* — no single BUY exceeds the spendable balance.
2. *Per slate* — total BUY notional is capped at the balance, since three
   individually-affordable buys can still collectively overdraw. When the cap
   binds, all buys scale down proportionally and anything falling under the
   exchange minimum is dropped.
3. *At execution* — checked again per fill, because approvals happen one at a
   time and fills happen at live prices. A buy that no longer fits is rejected
   and logged, never partially filled into a negative balance.

Balance comes from paper cash in paper mode, and from your **free** USDT on
Binance otherwise (free, not total — funds locked in open orders can't be
spent). If it can't be read, the cap is skipped rather than guessed, and
preflight warns per trade instead.

In live mode the prompt escalates: you must type the ticker back, so muscle
memory on `y` cannot spend real money.
"""),

code(r"""
result = pipeline.run(
    store, CONFIG,
    scrape_reddit=False,     # already scraped above; set True for a fully self-contained run
    scan_catalysts=False,    # ditto
    interactive=True,        # False = show proposals, execute nothing
    # auto_approve=True,     # accept everything without prompting — NOT the default.
    #                        # Live accounts additionally need CRYPTO_YOLO_ALLOW_AUTO_LIVE=1.
    #                        # Same thing from a shell: ./run_cycle.py --auto-approve
)

# An empty slate is a normal outcome, so guard before subscripting columns —
# propose() returns a bare DataFrame with no columns when nothing qualifies.
_props = result["proposals"]
if _props.empty:
    print("No proposals to review this run — see the diagnosis above.")
else:
    _cols = ["rank", "symbol", "side", "kind", "trigger", "composite", "entry",
             "stop", "target", "qty", "notional", "fee_est", "reward_risk",
             "reward_risk_net", "decision"]
    display(_props[[c for c in _cols if c in _props.columns]])
"""),

md(r"""
## Did it work? — Evaluation & benchmark

The weights above are hand-set priors. This section is the feedback loop that
tells you whether they mean anything: it joins every stored score to the asset's
**realised** forward return, measures each family's information coefficient
(rank-correlation with return), reconstructs the actual round-trip trades net of
fees, and compares the equity curve to simply holding BTC.

With only a couple of weeks of data every number here is noisy — a t-stat under
~2 is "no evidence", not a finding. Re-run this weekly; act on it after a month.
"""),

code(r"""
from cryptoyolo import evaluation as ev

report = ev.summary(store, CONFIG, horizons=(1, 7, 30), primary_horizon=7)
"""),

code(r"""
# The full information-coefficient table (per-run mean IC, its t-stat, and the
# pooled IC) for every family at every horizon. Negative mean_IC = the family
# is pointing the wrong way over that horizon.
display(report["ic"])
"""),

code(r"""
# Realised trades, newest first — every round trip the order log implies, with
# P&L net of fees and the fee drag as a share of gross P&L.
_tr = report["trades"]
if _tr.empty:
    print("No closed trades yet.")
else:
    display(_tr.tail(20))
    display(report["trade_stats"])
"""),

code(r"""
# Data-driven weights from the measured ICs. This does NOT change anything —
# it prints a ScoreWeights(...) you can paste into the Configuration cell once
# there is enough history for the t-stats to clear |t| ≥ 2. The blend_verdict
# is the headline: does the composite even beat its own best component?
_rec = report["recommendation"]
if _rec.get("blend_verdict"):
    print(_rec["blend_verdict"], "\n")
print(_rec.get("code") or _rec.get("note"))
"""),

code(r"""
# Regime-gate self-check: realised forward return per regime state. The gate is
# working if risk_off rows precede weaker universe / top-N returns than risk_on.
# Empty until a few cycles have run with the gate on (pipeline.run logs it).
_re = report["regime_effectiveness"]
if _re is None or _re.empty:
    print("No regime history yet — runs before this build didn't log it.")
else:
    display(_re)
"""),

md("## Portfolio & audit trail"),

code(r"""
positions = store.paper_positions()
if positions.empty:
    print("No paper positions.")
else:
    live = prices.latest_prices(list(positions["symbol"]), CONFIG)
    positions["last"] = positions["symbol"].map(live)
    positions["value"] = positions["qty"] * positions["last"]
    positions["pnl"] = (positions["last"] - positions["avg_price"]) * positions["qty"]
    positions["pnl_pct"] = (positions["last"] / positions["avg_price"] - 1) * 100
    display(positions[["symbol", "qty", "avg_price", "last", "value", "pnl", "pnl_pct"]]
            .style.format({"qty": "{:.6f}", "avg_price": "{:,.4f}", "last": "{:,.4f}",
                           "value": "${:,.2f}", "pnl": "${:+,.2f}", "pnl_pct": "{:+.2f}%"})
            .hide(axis="index"))

print(f"\npaper cash: ${store.paper_cash():,.2f}")
"""),

code(r"""
print("Recent orders")
display(store.recent_orders(15))

print("\nProposal history — every candidate ever surfaced, and what you decided")
display(store.proposal_history(20))
"""),

md(r"""
---
## Enabling live Binance trading

Only do this once you've watched paper mode long enough to trust the ranking.

**1 · Keys.** Binance → API Management → create a key with **Spot Trading**
enabled and **withdrawals disabled**. IP-restrict it. Put them in `.env`:

```
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
```

**2 · Validate first.** Leave `dry_run=True` and set `CONFIG.execution.mode =
"binance"`. Orders go to `/api/v3/order/test`: Binance checks symbol, lot size,
tick size, min-notional and your permissions, then discards the order. Nothing
fills. Run a few cycles here — this catches filter and precision errors without
risking anything.

**3 · Go live.** Set `dry_run=False` **and** export `CRYPTO_YOLO_ALLOW_LIVE=1`
before starting Jupyter. Both are required; either alone stays validate-only.

```bash
CRYPTO_YOLO_ALLOW_LIVE=1 jupyter lab
```

**Venue.** `api.binance.com` and `testnet.binance.vision` return **HTTP 451**
from US IP addresses. `binance-us` is the default and works. Switch with
`CONFIG.execution.venue`.

### Keep collecting while you trade manually

Feature 2's velocity signal needs history that accrues in real time, so run the
collector on a schedule and use this notebook manually. They share the SQLite
database safely — WAL mode, verified with a collector writing while a notebook
reader held an open connection: no lock errors, and the reader sees new commits.

Install the 8-hourly job (00:00 / 08:00 / 16:00 local, plus once at load):

```bash
cd /Users/socrates/Documents/notebooks/crypto-yolo
cp com.crypto-yolo.collector.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.crypto-yolo.collector.plist
```

Check on it:

```bash
launchctl list | grep crypto-yolo     # 2nd column = last exit code (0 = fine)
./collect.py --status                 # history span, and whether velocity is ready
tail -f data/collector.log            # watch a pass run
```

`collect.py` bootstraps its own path so it runs from any directory, and an
`flock` guard means a stalled pass can never stack a second writer on the
database. To stop it: `launchctl unload ~/Library/LaunchAgents/com.crypto-yolo.collector.plist`.

The in-notebook `scheduler.start(...)` thread is fine for a single session but
dies with the kernel — use launchd for the baseline.

### Worth doing before trusting the scores

The weights are priors, not findings. The **Evaluation & benchmark** section
above is the check: it joins every stored score to the realised forward return
and reports each family's information coefficient, the top-vs-bottom spread, the
realised trade record net of fees, and the equity curve against BTC
buy-and-hold. Run it weekly (`evaluation.summary(store, CONFIG)`); once the
t-stats clear |t| ≥ 2, replace the hand-set weights with
`evaluation.recommend_weights(...)`'s output.
"""),
]


def build() -> Path:
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    nbf.validate(nb)
    OUT.write_text(nbf.writes(nb))
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({len(CELLS)} cells, {path.stat().st_size:,} bytes)")
