# crypto-yolo

A Jupyter dashboard that pulls crypto price data, Reddit sentiment, and
developer/news catalysts; scores the universe with a transparent formula;
proposes up to three ranked trade candidates for a 1-day-to-1-week horizon;
asks for per-trade approval; and executes approved trades through the Binance
API.

```bash
jupyter lab crypto_yolo_dashboard.ipynb
```

Runs in paper mode out of the box with no credentials at all — though Feature 2
needs free Reddit keys to return anything (see below).

---

## What this is, and what it isn't

**It ranks; it does not predict.** Nothing can identify the three trades that
*will* maximise profit over the next week. What this does is score every asset
with a visible, hand-tuned formula, expose every component behind each score,
and size positions against a stated risk budget.

The weights in `ScoreWeights` and the impact priors in `EVENT_TYPES` were
chosen by hand. **They have not been fitted to realised returns, and this has
not been backtested.** The value on offer is legibility — you can see exactly
why an asset ranked where it did, and change any weight you disagree with.

**On signal quality.** Reddit mention volume is trivially gamed and mostly
lagging; by the time a coin trends on r/wallstreetbets the move that caused the
trend has usually happened. That's why social carries the lowest weight (0.20).
The genuinely forward-looking input is the *scheduled* catalyst — a token unlock
dated next Tuesday is knowable in advance in a way a price move is not.

Every run writes its scores and proposals to SQLite specifically so you can
answer the only question that matters: did high composite scores actually
precede higher returns? If the correlation is ~0, the weights are decoration.

This is not financial advice. You approve every trade; you own every outcome.

---

## Layout

```
crypto_yolo_dashboard.ipynb   the dashboard (generated — edit build_notebook.py)
build_notebook.py             regenerates the notebook
cryptoyolo/
  config.py       universe, weights, risk, execution settings, env loading
  store.py        SQLite: posts, mentions, catalysts, scores, proposals, orders
  prices.py       OHLCV from Binance.US / Coinbase / Kraken / yfinance
  indicators.py   Bollinger, Keltner, Donchian, RSI, MACD, ATR
  charts.py       plotly candlesticks, universe grid, score attribution
  social.py       Feature 2 — Reddit OAuth, mention extraction, sentiment
  catalysts.py    Feature 3 — GitHub releases, news RSS, event taxonomy
  engine.py       scoring, ranking, risk-based position sizing
  broker.py       signed Binance REST client + paper broker
  approval.py     per-trade approval gate
  pipeline.py     end-to-end orchestration
  scheduler.py    recurring collector (thread or cron)
data/             SQLite database (gitignored)
```

---

## The three features

### 1 · Price charts with bands

Candles with Bollinger (configurable σ), Keltner, Donchian, moving averages,
volume and RSI, over **1h / 1d / 1w / 1mo / 1y**.

Four price sources are tried in order — `binance` → `coinbase` → `kraken` →
`yfinance` — so a rate limit or regional block on one doesn't stop the run.

> `api.binance.com` and `testnet.binance.vision` return **HTTP 451** from US IP
> addresses. `api.binance.us` works and is the default. Change with
> `CONFIG.execution.venue`.

### 2 · Reddit sentiment and mention velocity

Posts *and* comments from r/wallstreetbets and r/cryptocurrency, with a
crypto-native sentiment lexicon (`moon`, `rug`, `rekt`, 🚀) — generic English
sentiment models score crypto slang badly.

Two decisions worth knowing about:

**Precision over recall.** `DOT`, `OP`, `NEAR`, `LINK`, `ATOM`, `ADA` are
ordinary English words; counting them naively produces a signal that is almost
entirely noise. Assets flagged `ambiguous` match only as `$DOT` or by full name
(`polkadot`), never as the bare word.

**Velocity, not volume.** Raw counts rank BTC and ETH first every run. The score
uses the log-ratio of current chatter against each asset's *own* baseline,
damped by author diversity (one person posting 40 times counts roughly once) and
by how much text carried actual directional language.

**Reddit requires OAuth.** Anonymous `.json` access now returns 403. Create a
free script app at <https://www.reddit.com/prefs/apps> and set
`REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET`.

**Cold start:** velocity needs history. On a fresh database the first run's
social scores are ~0 by construction — correct behaviour, not a bug. Run the
collector (below) to build a baseline.

### 3 · Developer activity and news catalysts

GitHub releases per asset repo, plus news RSS classified against an event
taxonomy (upgrade, ETF, unlock, exploit, lawsuit, listing, delay…).

**Direction isn't taken from keywords alone.** "ETF" is a topic, not a
direction — `record ETF inflows` and `ETF outflows accelerate` point opposite
ways. Topic-type events take their sign from the headline's own tone;
intrinsically directional events (an exploit is never good news) keep their
prior and are discounted when tone disagrees.

---

## Execution safety

| Mode | `mode` | `dry_run` | `CRYPTO_YOLO_ALLOW_LIVE` | Effect |
|---|---|---|---|---|
| Paper *(default)* | `paper` | — | — | Simulated fills in local SQLite |
| Validate | `binance` | `True` | — | `POST /api/v3/order/test` — validated, never fills |
| Validate | `binance` | `False` | unset | Still validate-only |
| **Live** | `binance` | `False` | `1` | **Real orders, real money** |

Live requires two switches in two different places that must agree — a config
field and an environment variable — so re-running a notebook cell can never
place a live order by itself. In live mode the approval prompt also requires
typing the ticker back, so muscle memory on `y` can't spend money.

Quantities and prices are rounded to each pair's `LOT_SIZE` / `PRICE_FILTER` /
`NOTIONAL` filters before submission, and the client syncs to Binance server
time to avoid `-1021` timestamp rejections.

**Spot cannot short.** A bearish score on an asset you don't hold is an avoid,
not a trade — it appears in the ranking but is never proposed. Bearish scores on
assets you *do* hold become SELL (exit) proposals.

### Going live

1. Binance → API Management → new key, **Spot Trading** enabled, **withdrawals
   disabled**, IP-restricted. Put it in `.env`.
2. Set `CONFIG.execution.mode = "binance"`, leave `dry_run = True`. Run several
   cycles — this catches filter/precision errors at zero risk.
3. Set `dry_run = False` and start Jupyter with the env var:
   ```bash
   CRYPTO_YOLO_ALLOW_LIVE=1 jupyter lab
   ```

---

## Position sizing

Risk-first, not fixed-dollar. Stop distance is `1.5 × ATR(14)` on **daily**
candles — matching the 1-to-7 day horizon — and quantity is set so that being
stopped out costs exactly `risk_per_trade_pct` of equity.

Sizing off hourly ATR (the obvious mistake) produces sub-1% stops that get taken
out by ordinary intraday noise. Positions are capped per-trade
(`max_position_pct`) and in aggregate (`max_total_deployed_pct`); when a cap
binds, buys scale down proportionally so the ranking is preserved.

### Purchases never exceed available cash

`account_equity_usd` is the **risk basis**, not a spending limit — it sets how
much you're willing to lose per trade. Spending is capped separately against
your real balance, in three places, because each catches something the others
miss:

1. **Per trade** — no single BUY exceeds the spendable balance.
2. **Per slate** — total BUY notional is capped at the balance. Three
   individually-affordable buys can still collectively overdraw, so the binding
   constraint is the sum. Anything falling below the exchange minimum after
   scaling is dropped rather than proposed as an invalid order.
3. **At execution** — re-checked per fill, since approvals happen one at a time
   and fills use live prices. A buy that no longer fits is rejected and logged
   (`REJECTED_INSUFFICIENT_CASH`), never partially filled into a negative
   balance.

Balance is paper cash in paper mode, and **free** quote-asset balance on Binance
otherwise — free rather than total, because funds locked in open orders can't be
spent. If the balance can't be read (no credentials, API error) the cap is
skipped rather than guessed: unknown is treated as unknown, not as zero, and
preflight warns per trade instead.

SELLs are exempt from both caps — an exit releases capital rather than consuming
it — and are separately capped at the quantity you actually hold.

---

## Keeping the Reddit baseline warm

Feature 2 is only as good as its history. A cron entry turns social from noise
into signal:

```bash
*/30 * * * * cd /Users/socrates/Documents/notebooks/crypto-yolo && \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m cryptoyolo.scheduler --once >> data/collector.log 2>&1
```

Or in-notebook, for as long as the kernel lives:

```python
collector = scheduler.start(store, CONFIG, interval_minutes=30)
```

---

## Credentials

Copy `.env.example` to `.env`. Nothing is hard-coded; every value is read from
the environment at runtime, and `.env` is gitignored.

| Variable | For | Without it |
|---|---|---|
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Feature 2 | 403 — social score flat 0 |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Execution | paper mode works fully |
| `GITHUB_TOKEN` | Feature 3 | works; 60 req/hr caps the scan |
| `CRYPTO_YOLO_ALLOW_LIVE` | Live orders | orders stay validate-only |

---

## Dependencies

Everything needed is already installed in this environment. `ccxt`, `praw`,
`feedparser` and `vaderSentiment` are deliberately *not* used — see
`requirements.txt` for why.
