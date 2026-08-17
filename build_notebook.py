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

Run all cells. The notebook refreshes three feature families, scores the
universe, proposes up to **3 ranked trade candidates** for a 1-day-to-1-week
horizon, asks you to approve each one, and executes the approved ones through
the Binance API.

| Feature | Source | Feeds |
|---|---|---|
| **1 · Price & bands** | Binance.US → Coinbase → Kraken → yfinance | technical score |
| **2 · Reddit sentiment** | r/wallstreetbets, r/cryptocurrency | social score |
| **3 · Dev & news catalysts** | GitHub releases, CoinDesk/Cointelegraph/Decrypt/Defiant | catalyst score |

---

### Read this once

**The engine ranks; it does not predict.** No system can identify the three
trades that *will* maximise profit over the next week. What this does is score
every asset with a transparent, hand-tuned formula, show you every component
that went into each score, and size positions against a stated risk budget. The
weights in `ScoreWeights` are priors chosen by hand — **they have not been
fitted to realised returns and this has not been backtested.** Treat the output
as a research shortlist that shows its work, not as a forecast.

**Signal quality, honestly.** Reddit mention volume is trivially gamed and
mostly lagging — by the time a coin trends on r/wsb, the move usually already
happened. That is why social is weighted lowest (0.20). The most genuinely
forward-looking input here is the *scheduled* catalyst: a token unlock dated
next Tuesday is knowable in advance in a way that a price move is not.

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

sys.path.insert(0, str(Path.cwd()))
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pio.renderers.default = "notebook"

from cryptoyolo import charts, engine, indicators, pipeline, prices, scheduler
from cryptoyolo import catalysts as catalysts_mod
from cryptoyolo import social as social_mod
from cryptoyolo.config import CONFIG, TIMEFRAMES, load_dotenv
from cryptoyolo.store import Store

load_dotenv()                      # reads ./.env if present; never overwrites real env vars
store = Store(CONFIG.db_path)

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
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Feature 2 | **Reddit returns 403** — social score is flat 0 |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Execution | paper mode still works fully |
| `GITHUB_TOKEN` | Feature 3 | works, but 60 req/hr caps the universe scan |
| `CRYPTO_YOLO_ALLOW_LIVE=1` | Live orders | orders stay validate-only |

Reddit credentials are free: <https://www.reddit.com/prefs/apps> → *create app* →
type **script** → copy the id under the app name and the secret.
"""),

md("## Configuration"),

code(r"""
# ── Risk ────────────────────────────────────────────────────────────────
# account_equity_usd is the RISK BASIS only — it sets how much you're willing
# to lose per trade. It is NOT a spending limit. Actual purchases are separately
# capped at your real spendable balance (paper cash, or free USDT on Binance),
# so this number being larger than your balance can't cause an overdraft.
CONFIG.risk.account_equity_usd   = 10_000.0
CONFIG.risk.risk_per_trade_pct   = 1.0        # % of equity lost if stopped out
CONFIG.risk.max_position_pct     = 20.0       # cap per position
CONFIG.risk.max_total_deployed_pct = 60.0     # cap across all proposals
CONFIG.risk.atr_stop_mult        = 1.5        # stop = 1.5 x daily ATR(14)
CONFIG.risk.reward_risk_target   = 2.0        # target = 2R
CONFIG.risk.max_proposals        = 3
CONFIG.risk.min_composite_score  = 0.05       # 0.0 = always fill all 3 slots

# ── Score weights (must be defensible to you, not to me) ────────────────
CONFIG.weights.technical = 0.50
CONFIG.weights.social    = 0.20
CONFIG.weights.catalyst  = 0.30

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
CONFIG.execution.order_type  = "MARKET"
CONFIG.execution.quote_asset = "USDT"

w = CONFIG.weights.normalized()
print(f"weights    : technical {w.technical:.0%} · social {w.social:.0%} · catalyst {w.catalyst:.0%}")
print(f"risk/trade : ${CONFIG.risk.account_equity_usd * CONFIG.risk.risk_per_trade_pct / 100:,.2f}")

from cryptoyolo.broker import execution_banner
print(f"execution  : {execution_banner(CONFIG)}")
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
from IPython.display import clear_output, display

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
## Feature 2 · Reddit sentiment & mention velocity

Pulls posts *and* comments from r/wallstreetbets and r/cryptocurrency, extracts
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

**Cold-start:** on a fresh database there is no baseline, so the first run's
social scores are ~0 by construction. That is correct behaviour, not a bug —
run the collector in the next cell for a day to make this feature meaningful.
"""),

code(r"""
stats = social_mod.scrape(store, CONFIG, verbose=True)

social_scores = social_mod.score_symbols(store, CONFIG)
print(f"\nhistory: {social_scores.attrs['history_hours']:.1f}h · "
      f"baseline ready: {social_scores.attrs['baseline_ready']}")
if not social_scores.attrs["baseline_ready"]:
    print("→ Not enough history for velocity yet; social scores stay near 0 until "
          "the collector has run for ~1.5x the velocity window.")

social_scores.head(15)
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
## Decision engine

Composite = `0.50 × technical + 0.20 × social + 0.30 × catalyst`, each component
in [-1, +1].

The one genuinely non-obvious piece is how band position is read: in a trending
market, price riding the upper Bollinger band is *strength*; in a range-bound
market the identical reading is *stretched*. A single fixed interpretation is
wrong half the time, so trend strength decides which reading applies.
"""),

code(r"""
scores = engine.build_scores(store, CONFIG, verbose=True)

display(scores[[
    "symbol", "composite", "technical", "social", "catalyst",
    "price", "rsi_1w", "bb_pctb_1w", "atr_pct_daily",
    "mentions_24h", "n_events", "top_event",
]].style.format({
    "composite": "{:+.3f}", "technical": "{:+.3f}", "social": "{:+.3f}",
    "catalyst": "{:+.3f}", "price": "{:,.4f}", "bb_pctb_1w": "{:.2f}",
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

**Spot markets cannot short.** A bearish score on an asset you don't hold is not
a trade, it's an avoid — such assets appear in the ranking above but are never
proposed. Bearish scores on assets you *do* hold become SELL (exit) proposals.

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
)

result["proposals"][["rank", "symbol", "side", "composite", "entry", "stop",
                     "target", "qty", "notional", "reward_risk", "decision"]]
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

### Keep the Reddit history warm

Feature 2 is only as good as its baseline. A cron entry collecting every 30
minutes is what turns social from noise into signal:

```bash
*/30 * * * * cd /Users/socrates/Documents/notebooks/crypto-yolo && \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m cryptoyolo.scheduler --once >> data/collector.log 2>&1
```

### Worth doing before trusting the scores

The weights are priors, not findings. Every run writes its scores and
proposals to SQLite, so after a few weeks you can check the only question that
matters — whether high composite scores actually preceded higher returns:

```python
import pandas as pd
with store.conn() as con:
    hist = pd.read_sql_query("SELECT ts, symbol, composite FROM scores", con)
# join each row to the asset's forward 1d / 7d return and correlate.
# If the correlation is ~0, the weights are decoration. Change them.
```
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
