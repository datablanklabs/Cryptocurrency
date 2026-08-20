# crypto-yolo

A Jupyter dashboard that pulls crypto price data, Reddit sentiment, and
developer/news catalysts; scores the universe with a transparent formula;
proposes up to three ranked trade candidates on short-horizon signals (held up
to `exits.horizon_days`, default 30);
asks for per-trade approval (or `--auto-approve`, opt-in); and executes approved
trades through the Binance API.

```bash
jupyter lab crypto_yolo_dashboard.ipynb
```

Runs in paper mode out of the box with **no credentials at all** — all three
features work keyless, including Reddit sentiment (it falls back to Arctic Shift
and Reddit's own Atom feeds). Credentials improve data quality and are needed
only to place orders.

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
collect.py                    scheduled collector entry point (path-independent)
run_cycle.py                  one trading cycle from the CLI (--auto-approve lives here)
com.crypto-yolo.collector.plist   launchd job, runs collect.py every 8h
cryptoyolo/
  config.py       universe, weights, risk, execution settings, env loading
  store.py        SQLite: posts, mentions, catalysts, scores, proposals, orders,
                  position_meta (entry terms that exits evaluate against)
  prices.py       OHLCV from Binance.US / Coinbase / Kraken / yfinance
  indicators.py   Bollinger, Keltner, Donchian, RSI, MACD, ATR
  charts.py       plotly candlesticks, universe grid, score attribution
  portfolio.py    Feature 0 — account balance, holdings, cost basis, P&L
  social.py       Feature 2 — Reddit source ladder, mention extraction, sentiment
  feeds.py        Feature 2b — StockTwits cashtags, Mastodon hashtags
  catalysts.py    Feature 3 — GitHub releases, news RSS, event taxonomy
  exits.py        five configurable exit triggers for open positions
  engine.py       scoring, ranking, risk-based position sizing
  broker.py       signed Binance REST client + paper broker
  approval.py     per-trade approval gate
  pipeline.py     end-to-end orchestration
  scheduler.py    in-kernel collector thread (dies with the kernel)
data/             SQLite database (gitignored)
```

---

## The features

### 0 · Account balance and holdings

Cash, open positions, market value, unrealised P&L and portfolio weights — shown
before anything else, because position sizing, the cash cap and whether a SELL is
even possible all depend on it.

Cost basis is exact in paper mode (the book is ours). On a live Binance account
only fills placed through this dashboard have a basis that can be honestly
reported; anything bought elsewhere shows `—` rather than a guess. When the
balance can't be read it reports *unavailable* rather than `$0.00`, so a missing
credential never looks like an empty account.

It also flags when `account_equity_usd` (the risk basis) has drifted more than
20% from real equity, since sizing is computed from the former.

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

**Sources, in fallback order.** Reddit 403s anonymous `.json` clients, so the
scraper tries four sources and uses the first that answers:

| Source | Key? | Scores? | Notes |
|---|---|---|---|
| `oauth` | yes | yes | Official API. Real `hot` vs `new`, pagination, best limits |
| `arctic` | **no** | **yes** | [Arctic Shift](https://arctic-shift.photon-reddit.com), a public Pushshift successor. 100/req, current |
| `rss` | **no** | no | Reddit's own Atom feeds. Full comment bodies, ~25/req, rate-limited |
| `public_json` | no | yes | Legacy anonymous `.json`. Usually 403; last resort |

**Credentials are optional.** Without them the ladder falls through to Arctic
Shift and the feature works — verified pulling 360 items and 36 mentions with no
keys set. Credentials still give the best data, so set them if you have them.

Two caveats on the keyless path. The RSS feed carries **no vote scores**, so
engagement weighting goes flat (mention counts and sentiment are unaffected);
the scraper prints a notice when it degrades to that. And the keyless sources
are chronological only — they have no `hot`, so the scraper fetches one listing
instead of two rather than burning duplicate requests on a shared public
service.

PullPush.io, the other well-known Pushshift successor, sits behind a Cloudflare
challenge and isn't usable from a script — it's deliberately not in the ladder.

**Sources measured, not guessed.** All 13 were sampled (200 items each) via
Arctic Shift on 2026-08-20 and ranked by mention-rate x asset-breadth x tone.
`span` is how much wall-clock time 200 items covers — low means active:

| Subreddit | mention % | assets | tone | span |
|---|---|---|---|---|
| CryptoMarkets | 18.5% | 7 | 0.20 | 65h |
| CryptoCurrency | 11.5% | 6 | 0.25 | 13h |
| ethtrader | 33.0% | 4 | 0.13 | 326h |
| Bitcoin | 29.0% | 2 | 0.21 | 10h |
| defi | 13.0% | 5 | 0.16 | 200h |
| SatoshiStreetBets | 12.5% | 8 | 0.10 | 2349h |
| solana | 46.5% | 4 | 0.05 | 91h |
| CryptoMoonShots | 9.0% | 4 | 0.22 | 239h |
| altcoin | 12.5% | 10 | 0.06 | 1554h |
| BitcoinMarkets | 19.0% | 2 | 0.16 | 721h |
| CryptoCurrencyTrading | 4.5% | 3 | 0.13 | 482h |
| CryptoTechnology | 8.0% | 3 | 0.06 | 340h |
| binance | 5.0% | 4 | 0.02 | 207h |

**The order re-ranks itself daily.** The table above is only the seed. Once
history exists, `social.rank_subreddits` re-derives the order from what each
source has actually delivered *to us* — no re-probing, the data is already in
SQLite — and caches it for `rank_refresh_hours` (24h). Ordering is an
expected-value estimate, not a weighted sum:

```
expected_new = min(page_size, items_per_hour * collection_cadence_hours)
usefulness   = mention_rate * breadth * (0.4 + 0.6 * tone)
value        = expected_new * usefulness
```

`expected_new` is what matters under a rate limit: a source earns an early slot
only if it has produced new material since the last pass. That is why
r/SatoshiStreetBets (0.15 items/hour) sinks despite a decent mention rate, while
r/Bitcoin (17 items/hour, 35% mentions) leads despite covering only 2 assets.
Breadth is log-scaled, so single-asset subs still earn a place for per-asset
sentiment without dominating.

Subreddits with no stats yet are spliced in after the top `probation_after_top`
(3) rather than appended last — otherwise a rate limit could starve a new source
forever and it could never earn a rank. See `./collect.py --status`. Two things the measurement settled:

**r/wallstreetbets was dropped.** It measured 0.5% crypto mentions, 1 asset and
0.00 tone confidence — it is an equities sub, and was half the original config.

**Single-asset subs can't rank.** r/solana's 46.5% mention rate looks excellent,
but a source that only ever discusses one asset contributes nothing to ordering
twenty against each other. They are included for per-asset sentiment, not breadth.

**Rate limiting:** Arctic Shift signals it with HTTP **422** (`"Timeout. Maybe
slow down a bit"`), not 429. The client treats 422 as a rate limit with real
backoff, and paces `inter_subreddit_delay` between subs — without both, 13 subs
x 2 calls fails constantly. A full pass takes ~50s.

**Cold start:** velocity needs history. On a fresh database the first run's
social scores are ~0 by construction — correct behaviour, not a bug. Run the
collector (below) to build a baseline.

### 2b · StockTwits and Mastodon

Two additional keyless social feeds, both measured before being wired in
(`cryptoyolo/feeds.py`). Adding them took assets-with-chatter from **10 to 18 of
20** in a single pass.

| Feed | Yield | Why it earns a slot |
|---|---|---|
| **StockTwits** | 600 msgs, 600 mentions | Cashtag streams arrive **pre-attributed to a symbol** — the mention matcher is bypassed entirely, so there are no false positives to control. ~57% carry a **user-declared** Bullish/Bearish label: a stated opinion, not one inferred from a word list. All 20 assets resolve, 30 msgs each. |
| **Mastodon** | 280 statuses, 304 mentions | Public hashtag timelines, no auth, 300 req/window. |

**Mastodon uses broad tags, not per-asset ones.** Measured: only wide tags are
alive (#crypto 3.6 posts/h, #bitcoin 3.5/h) while per-asset tags are effectively
dead — #dot returns one post per 33 hours. Spending a request per asset would
buy month-old content, so we pull seven broad tags and run the normal extractor.

**Sentiment is de-biased per source, and this matters a lot.** StockTwits users
self-label ~88% Bullish (mean tone **+0.55**, against Reddit's **+0.05**). Fed in
raw, that rated 18 of 20 assets strongly bullish — and a signal that calls
everything a buy cannot *rank* anything. Each source's mean tone is now
subtracted before assets are compared, so the score measures "bullish relative to
how bullish that platform always is" — the same relative-to-own-baseline logic
already used for mention velocity. It compressed ADA from 0.96 to 0.46 and
restored a usable spread.

Both feeds are individually switchable (`CONFIG.stocktwits.enabled`,
`CONFIG.mastodon.enabled`), collection is separately gated by
`pipeline.collect(scrape_feeds=...)`, and a failure in one never blocks the
others.

**Social alone cannot clear the proposal bar.** With three sources the social
component reaches ~0.34, which at weight 0.20 contributes ~0.068. The
`min_composite_score` default is **0.10**, deliberately above that ceiling: no
trade can be proposed on social sentiment alone — the weakest and most gameable
of the three families — without corroboration from technicals or a catalyst.
Lowering it below ~0.07 re-opens that door.

### 3 · Developer activity and news catalysts

GitHub releases per asset repo, plus news RSS classified against an event
taxonomy (upgrade, ETF, unlock, exploit, lawsuit, listing, delay…).

**Direction isn't taken from keywords alone.** "ETF" is a topic, not a
direction — `record ETF inflows` and `ETF outflows accelerate` point opposite
ways. Topic-type events take their sign from the headline's own tone;
intrinsically directional events (an exploit is never good news) keep their
prior and are discounted when tone disagrees.

### Exit management

Exits are evaluated **before** the buy-side ranking, against each position's own
recorded entry terms rather than how attractive the asset looks today. Without
that separation a deteriorating position only surfaces if it out-ranks every buy
candidate — which it rarely does, so it just sits there.

| Trigger | Fires when | Config |
|---|---|---|
| Stop hit | price ≤ the stop set at entry | `stop_loss` |
| Target hit | price ≥ the target set at entry | `take_profit`, `take_profit_fraction` |
| Trailing stop | gave back > `trail_pct` from the high-water mark | `trailing_stop`, `trail_pct`, `trail_activate_pct` |
| Horizon expiry | held past `horizon_days` (default 30) | `horizon_expiry`, `horizon_days` |
| Score reversal | composite ≤ threshold | `score_reversal`, `score_reversal_threshold` |

Each is independently switchable; `CONFIG.exits.enabled = False` turns the whole
pass off. Precedence when several fire is the table order — stop and target are
level events that already happened, so they outrank the judgement calls.

Four behaviours worth knowing:

- **Exits get their own slots.** `max_exit_proposals` is separate from
  `max_proposals`, so closing a position never costs you an entry slot.
- **Exits ignore `min_composite_score`.** A stop that has been hit is a fact
  about the position, not an opinion about its ranking.
- **Held assets are always scored**, even outside the configured universe —
  otherwise they are silently unsellable.
- **An asset under an exit signal is never proposed as a buy** in the same run,
  even when the exit itself is too small to meet the exchange minimum (the
  position is left open, with a printed explanation).

### Continuous protection between runs

Everything above runs **only when you run the notebook** — a stop breached at 3am
is acted on at your next run. Closing that gap needs an order resting at the
venue: `place_stop_orders` and `place_limit_orders` submit one after each entry
fills.

| Setting | Effect |
|---|---|
| `place_stop_orders` | rest a protective stop after entry |
| `place_limit_orders` | rest a take-profit at the target |
| `place_stop_limit_orders` | stop leg uses `STOP_LOSS_LIMIT` (required on Binance.US) |
| `use_oco` | send both legs as one OCO so filling one cancels the other |
| `stop_limit_offset_bps` | how far through the trigger the limit sits (default 25) |
| `use_trailing_delta` / `trailing_delta_bps` | let Binance trail the stop natively (10–2000 bps) |

Five things that shaped this implementation:

- **Binance.US has no market `STOP_LOSS`.** Its order types are `LIMIT`,
  `LIMIT_MAKER`, `MARKET`, `STOP_LOSS_LIMIT`, `TAKE_PROFIT_LIMIT`. A protective
  stop is therefore always a stop-*limit* — and **in a gap or fast flush the
  limit may not fill, so the stop does not protect you.** The offset makes a fill
  likelier; it cannot guarantee one. This is a property of the venue, not a bug.
- **Both legs go as one OCO.** Two independent resting sells for the same
  quantity could both fill, or the second be rejected for insufficient balance.
  With `use_oco=False` and both flags on, only the stop is placed, with a warning.
- **Resting sells lock the base asset,** so protection is cancelled automatically
  before every SELL — otherwise the exit fails on free balance.
- **Only stop and target can be delegated.** Horizon expiry and score reversal are
  judgements this code makes, so they still require a run.
- **Protection is only real in live mode.** Paper and validate-only record what
  *would* rest but place nothing — Binance has no test endpoint for OCO, so it
  cannot be dry-run validated at all. Binance also caps resting algo orders per
  symbol (`MAX_NUM_ALGO_ORDERS`, currently 5).

### Exits fund the entries

Exits are proposed *and executed* before entries, so their expected proceeds
(less `risk.exit_proceeds_haircut_pct` for slippage and fees) are added to buying
power. With $1,000 cash and a $9,000 exit pending, the entries size against
~$9,910 rather than leaving that capital idle. Reject the sell at the prompt and
the execution-time cash check still blocks the buy it was funding.

The trailing stop needs memory across runs, so each position carries a
`position_meta` row (opened-at, stop, target, horizon, high-water mark) written
on the entry fill and cleared when the position closes. Positions opened outside
this dashboard have no such row: only score-reversal can act on them, and both
the exit reason and the Feature 0 table say so rather than inventing terms.

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

## Running a cycle from the CLI

```bash
./run_cycle.py                  # propose only — executes nothing (default)
./run_cycle.py --approve        # prompt per trade, same as the notebook
./run_cycle.py --auto-approve   # accept every proposal, no prompting
./run_cycle.py --auto-approve --collect --paper
```

With no flags it prints the slate and exits, so running it by accident cannot
trade. An `flock` guard prevents overlapping cycles.

### --auto-approve

Off by default, and it takes a deliberate flag or `CONFIG.execution.auto_approve
= True`. It removes the human from the loop, so what it means depends entirely
on the mode:

| Mode | Effect |
|---|---|
| paper | simulated fills, unattended. The intended use — builds a decision record you can score the engine against |
| binance + `dry_run=True` | orders validated against Binance filters, never filled |
| binance live | **unattended real-money trading** — requires `CRYPTO_YOLO_ALLOW_AUTO_LIVE=1` |

That last row is a third switch on top of the two that already gate live mode.
Without it, auto-approve **refuses and approves nothing** rather than quietly
falling back to prompting — a cron job has nobody to prompt, and silently
skipping execution would be just as surprising as silently trading.

Auto-approved trades are recorded as `auto-approved`, not `approved`, so the
audit trail always distinguishes a machine decision from yours. That distinction
is the point: it lets you ask later whether the engine's unattended picks
actually performed, separately from the ones you chose.

A caveat worth stating plainly: the scoring weights have never been fitted to
realised returns, so auto-approving into a live account is betting real money on
untested heuristics with nobody watching. Paper mode exists precisely so you can
gather that evidence first.

## Running the collector on a schedule

Feature 2 is only as good as its baseline: mention *velocity* is measured against
each asset's own history, so the data has to accrue in real time. Run the
collector on a schedule and use the notebook manually — they share the SQLite
database safely (WAL mode, verified with a collector writing while a notebook
reader held an open connection: no lock errors, and the reader sees new commits).

`collect.py` is the entry point. It bootstraps its own path, so it works from
any directory — `python3 -m cryptoyolo.scheduler` only resolves when cwd happens
to be the project root, which neither launchd nor cron guarantees.

```bash
./collect.py --catalysts    # Reddit + news + GitHub — what the scheduled job runs
./collect.py                # Reddit only (faster)
./collect.py --status       # what's collected so far, no network calls
```

An `flock` guard means overlapping runs are impossible: if one pass stalls on a
slow feed, the next trigger exits immediately rather than stacking a second
writer onto the database.

### Install the 8-hourly job (macOS launchd)

```bash
cp com.crypto-yolo.collector.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.crypto-yolo.collector.plist
```

Runs at 00:00, 08:00 and 16:00 local, plus once immediately at load so you get
confirmation it works. Calendar intervals rather than a raw 28800s timer, because
launchd runs a missed calendar job when the Mac wakes — a plain interval can
drift across sleep.

```bash
launchctl list | grep crypto-yolo          # 2nd column is last exit code (0 = fine)
tail -f data/collector.log                 # watch it work
./collect.py --status                      # is the baseline ready?
launchctl unload ~/Library/LaunchAgents/com.crypto-yolo.collector.plist   # stop
```

The job runs only while you are logged in (a LaunchAgent, not a daemon), and not
while the Mac is fully powered off — both fine for this, since a missed window
just means slightly less history.

### cron alternative

```bash
0 */8 * * * /Users/socrates/Documents/notebooks/crypto-yolo/collect.py --catalysts >> /Users/socrates/Documents/notebooks/crypto-yolo/data/collector.log 2>&1
```

Works, but on modern macOS `cron` needs Full Disk Access granted to `/usr/sbin/cron`
in System Settings → Privacy & Security, or it fails silently. launchd is the
better default here.

### In-notebook alternative

`scheduler.start(store, CONFIG, interval_minutes=480)` runs a thread inside the
kernel. Convenient, but it dies with the kernel — use it for a session, not for
the baseline.

---

## Credentials

Copy `.env.example` to `.env`. Nothing is hard-coded; every value is read from
the environment at runtime, and `.env` is gitignored.

| Variable | For | Without it |
|---|---|---|
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Feature 2 | falls back to keyless sources; still works |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Execution | paper mode works fully |
| `GITHUB_TOKEN` | Feature 3 | works; 60 req/hr caps the scan |
| `CRYPTO_YOLO_ALLOW_LIVE` | Live orders | orders stay validate-only |

---

## Dependencies

Everything needed is already installed in this environment. `ccxt`, `praw`,
`feedparser` and `vaderSentiment` are deliberately *not* used — see
`requirements.txt` for why.
