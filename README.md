# crypto-yolo

A Jupyter dashboard that pulls crypto price data, Reddit sentiment,
developer/news catalysts and a scheduled-events calendar; scores the universe
with a transparent formula; proposes up to three ranked trade candidates on
short-horizon signals (held up to `exits.horizon_days`, default 30); asks for
per-trade approval (or `--auto-approve`, opt-in); and executes approved trades
through the Binance API.

```bash
python3 build_notebook.py            # the notebook is generated & gitignored
jupyter lab crypto_yolo_dashboard.ipynb
```

Runs in paper mode out of the box with **no credentials at all** — every
feature works keyless, including Reddit sentiment (it falls back to Arctic Shift
and Reddit's own Atom feeds). Credentials improve data quality and are needed
only to place orders.

---

## The return-focused pass (what changed, and why)

The engine was rebuilt around one gap: **there was no feedback loop.** Scores
were written to SQLite for years' worth of "did high scores precede higher
returns?" and nobody had ever answered it. Working in the order that matters:

1. **`evaluation.py` — the feedback loop.** Joins every stored score to the
   asset's realised 1d/7d/30d forward return; reports each family's information
   coefficient with a t-stat; reconstructs the real round-trip trades net of
   fees; compares the equity curve to buy-and-hold BTC; and prints a
   data-driven `ScoreWeights` to replace the hand-set priors. Paper fills now
   model **fees + slippage** so the paper record isn't optimistic vs live.
2. **Execution cost.** A `FeeConfig` (Binance.US ~40 bps taker, BNB-discount
   aware); entries go as a **marketable IOC limit** — crosses the book by at
   most `entry_limit_cross_bps`, fills now, and the unfilled remainder is
   cancelled rather than left resting (a partial fill is kept and the position
   sized to it); targets are widened so the *net* reward:risk after the full
   round trip (fees **and** slippage) equals `reward_risk_target`, then capped
   at `target_pct_cap` so a clamped-wide stop can't imply an unreachable
   target.
3. **Sizing & risk.** Position size is off **live equity** (cash + MTM
   positions), so the book compounds and de-risks on its own. The stop is
   **scaled to the holding horizon** (`1.5·dailyATR·√stop_horizon_days`) instead
   of 1.5·dailyATR, which for a multi-week hold sat inside one day's noise. A
   **BTC-trend regime gate** (with a `ma_band_pct` dead-band so an MA chop
   doesn't flip it) scales the whole slate and blocks new buys in a downtrend.
4. **Signal.** A fifth family, **`events`** (scheduled unlocks / mainnets / ETF
   dates — genuinely forward-looking; weight defaults to **0** until you fill
   `SCHEDULED_EVENTS`); **cross-sectional momentum** blended into the technical
   score; **relative selection** (take the top of the cross-section, not
   everything over a fixed threshold), with a separate `trim_threshold` before
   a holding is sold on the buy-side scoreboard.
5. **Benchmark.** Every evaluation compares to BTC and an equal-weight basket;
   `pipeline.run` now records an equity snapshot per cycle so the curve exists.
6. **Portfolio risk.** `portfolio_risk.py` caps the **correlation-adjusted**
   risk of the BUY slate — `sqrt(r' C r)` against `max_portfolio_heat_pct` of
   equity — so three 1%-risk alt longs that move together aren't quietly one
   2.5% bet. The regime gate is the macro switch; this is the micro one.

Every one of these is behind a config flag with the old behaviour still
reachable — see `build_notebook.py`'s Configuration cell.

Also from the review: the regime verdict is **persisted per cycle**
(`regime_log`) and `evaluation.regime_effectiveness()` measures whether
`risk_off` runs actually preceded weaker returns; realised fees are a
first-class `orders.fee_usd` column (not JSON spelunking); `recommend_weights`
now leads with a **`blend_verdict`** — does the composite even beat its own best
single component?; and the slippage guard is read through
`execution.entry_slippage_guard_bps`, which can never sit below the limit
cross.

---

## What this is, and what it isn't

**It ranks; it does not predict.** Nothing can identify the three trades that
*will* maximise profit over the next week. What this does is score every asset
with a visible, hand-tuned formula, expose every component behind each score,
size positions against a stated risk budget, and **now measure itself against
realised returns** so you can tell whether any of it works.

The weights in `ScoreWeights` and the impact priors in `EVENT_TYPES` were
chosen by hand. **They have not been fitted to realised returns.** Run
`evaluation.summary(store, CONFIG)` after a few weeks; once a family's IC clears
|t| ≥ 2, `evaluation.recommend_weights()` prints the weight vector to use
instead. Until then the value on offer is legibility — you can see exactly why
an asset ranked where it did, and change any weight you disagree with.

**On signal quality.** Reddit mention volume is trivially gamed and mostly
lagging; by the time a coin trends on r/wallstreetbets the move that caused the
trend has usually happened. That's why social carries a low weight. The
genuinely forward-looking input is the *scheduled* catalyst — a token unlock
dated next Tuesday is knowable in advance in a way a price move is not. And
**positioning** is the only family that goes reliably negative; without it the
composite drifts toward rating everything a buy.

Every run writes its scores and proposals to SQLite specifically so you can
answer the only question that matters: did high composite scores actually
precede higher returns? If the correlation is ~0, the weights are decoration.

This is not financial advice. You approve every trade; you own every outcome.

---

## Layout

```
crypto_yolo_dashboard.ipynb   the dashboard — GENERATED and gitignored; run
                              `python3 build_notebook.py` to (re)create it
build_notebook.py             the notebook's real source
collect.py                    scheduled collector entry point (path-independent)
run_cycle.py                  one trading cycle from the CLI (--auto-approve lives here)
backfill_prices.py            one-off: fill prices_daily for the runs already logged
com.crypto-yolo.collector.plist   launchd job, runs collect.py every 8h
com.crypto-yolo.trader.plist      launchd job, runs one paper cycle a day
cryptoyolo/
  config.py       universe, weights, risk, fees, regime, events, execution,
                  notifications; env loading
  store.py        SQLite: posts, mentions, catalysts, scores, proposals, orders,
                  position_meta, equity_snapshots (the equity curve), prices_daily
                  (point-in-time closes for a reproducible evaluation)
  prices.py       OHLCV from Binance.US / Coinbase / Kraken / yfinance
  indicators.py   Bollinger, Keltner, Donchian, RSI, MACD, ATR
  charts.py       plotly candlesticks, universe grid, score attribution
  portfolio.py    Feature 0 — account balance, holdings, cost basis, P&L
  social.py       Feature 2 — Reddit source ladder, mention extraction, sentiment,
                  per-source tone de-biasing
  feeds.py        Feature 2b — StockTwits cashtags, Mastodon hashtags, 4chan /biz/
  positioning.py  Feature 4 — perpetual funding rates as a crowding measure
  catalysts.py    Feature 3 — GitHub releases, news RSS, event taxonomy
  calendar_events.py  Feature 5 — scheduled dated events (unlocks, mainnets, ETF dates)
  macro.py        Feature 6 — Kalshi event-contract prices (Fed, CPI, shutdown risk, ...)
  kalshi_prediction.py  Feature 7 — Kalshi's own crypto price markets, interpolated
                  at spot into a per-asset directional score
  regime.py       BTC-trend regime gate: how much long exposure the tape justifies,
                  dampened (never boosted) by macro.py's read
  portfolio_risk.py  correlation matrix + sqrt(r' C r) heat of a candidate slate
  exits.py        five configurable exit triggers for open positions
  engine.py       scoring (6 families + xsec momentum), ranking, risk-based sizing
  broker.py       signed Binance REST client + paper broker + the fee/slippage model
  approval.py     per-trade approval gate
  pipeline.py     end-to-end orchestration (equity snapshot + structured summary per cycle)
  evaluation.py   the feedback loop: forward-return IC, realised trades, benchmark
  logsetup.py     rotating log file in data/ + stderr, for the unattended jobs
  notify.py       best-effort alerts (ntfy / webhook / macOS banner)
  scheduler.py    in-kernel collector thread (dies with the kernel)
tests/            pytest suite — `pip install -r requirements-dev.txt && pytest`
data/             SQLite database + logs (gitignored)
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

### 2 · Social sentiment and mention velocity

Posts *and* comments across 13 subreddits plus StockTwits, Mastodon and /biz/,
with a crypto-native sentiment lexicon (`moon`, `rug`, `rekt`, 🚀) — generic English
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
| **4chan /biz/** | ~920 posts, ~77 mentions | Willing to be negative — ~24% of its mentions score bearish. Anonymous, so weights are flat and damped to 0.6, and it is excluded from author-diversity. |

**Mastodon uses broad tags, not per-asset ones.** Measured: only wide tags are
alive (#crypto 3.6 posts/h, #bitcoin 3.5/h) while per-asset tags are effectively
dead — #dot returns one post per 33 hours. Spending a request per asset would
buy month-old content, so we pull seven broad tags and run the normal extractor.

**What /biz/ actually adds, precisely.** Because de-biasing centres every source
on its own mean, a platform being more bearish *overall* is centred out — so
/biz/ does not pull the level down. Its contribution is **cross-asset dispersion
and coverage** within a differently-minded population. The aggregate positivity
problem is solved by the de-biasing below, not by adding a bearish venue.

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

### 4 · Positioning (perpetual funding)

Not social sentiment, which is why it is a **separate family** rather than
another feed. Social tells you what people are *saying*; funding tells you what
leveraged traders are *paying* to hold a side. Positive funding = longs pay
shorts = crowded long.

It earns its own component because it is **the only input that goes genuinely
negative on its own**. Over 100 periods (~33 days):

| Asset | negative periods | range |
|---|---|---|
| SOL | 27% | −0.0103% .. +0.0100% |
| ETH | 25% | −0.0053% .. +0.0100% |
| BTC | 10% | −0.0039% .. +0.0100% |

Every social source measured is positive nearly always, so without this the
composite drifts toward "everything is a buy".

Scored as a z-value against each asset's **own** funding history — 0.01% means
something different for BTC than for a thin altcoin. Read **contrarian** by
default (crowded positioning is fragile positioning); `contrarian=False` reads it
as momentum confirmation instead. Both are defensible; neither is tested here.

**Weights:** families are added without rescaling the others — `normalized()`
divides by the sum, so `positioning = 0.10` alongside `0.50/0.20/0.30` just
makes room and every prior ratio holds. `events` and `kalshi_prediction` both
ship at **0.0** because their inputs are empty (`SCHEDULED_EVENTS` /
`KALSHI_CRYPTO_SERIES`) — a non-zero weight on an all-zero family only dilutes
the rest; raise `events` to ~0.15 once you populate `SCHEDULED_EVENTS`, and
`kalshi_prediction` to ~0.10 (deliberately modest — see Feature 7's
short-dated caveat) once you populate `KALSHI_CRYPTO_SERIES`. Set any weight
to `0.0` to drop that family. Don't trust these numbers — run `evaluation` and
use `recommend_weights()`.

A caveat visible today: all 20 assets currently sit at the funding cap, so
positioning applies a near-uniform bearish tilt. The **discrimination** comes
from the z-magnitude (ETH −0.84 vs LINK −0.49), so compare assets to each other
rather than to zero.

### 3 · Developer activity and news catalysts

GitHub releases per asset repo, plus news RSS classified against an event
taxonomy (upgrade, ETF, unlock, exploit, lawsuit, listing, delay…).

**Direction isn't taken from keywords alone.** "ETF" is a topic, not a
direction — `record ETF inflows` and `ETF outflows accelerate` point opposite
ways. Topic-type events take their sign from the headline's own tone;
intrinsically directional events (an exploit is never good news) keep their
prior and are discounted when tone disagrees.

This family scores news that has *already been published* — which
`evaluation` tends to show is a reaction, not a lead. For the forward-looking
version, see Feature 5.

### 5 · Scheduled events (the forward-looking family)

`calendar_events.py`. Events with a **known future date** — token unlocks,
mainnet launches, ETF decision deadlines, listing effective dates — scored by
how close the date is and how big the event is, on a tent function that ramps up
as the date approaches, holds through `peak_window_days`, then decays after.
Token-unlock magnitude is `% of circulating supply`; everything else is a 0–1
size. Events combine as a signed saturating sum.

The schedule is **hand-maintained in `config.py::SCHEDULED_EVENTS` and ships
empty** — this family scores 0 until you populate it. Optionally set
`events.fetch_unlocks` + `unlocks_url` to pull from a public unlock feed
(best-effort, fails quietly). This is the one input that is genuinely knowable
in advance rather than a reaction to a move that already happened.

### 6 · Macro backdrop (Kalshi event contracts)

`macro.py`. Not a scored family like the five above — it feeds `regime.py`
instead. Kalshi is a CFTC-regulated prediction market: a contract settles at
$1 if a YES proposition resolves true, $0 otherwise, so its live price *is* a
market-implied probability. `config.py::MACRO_SERIES` is a hand-maintained
list of series to read — Fed rate decisions, CPI prints, government-shutdown
risk, recession odds — the kind of broad macro uncertainty that moves risk
assets generally, crypto included, in a way nothing else here can see coming.

Each series is scored against its own **`baseline`** — the probability that
is normal for that event (e.g. ~15% for a recession starting this year) — not
a 50% coin flip. Only a reading *worse* than normal counts:
`−weight × (P(bad) − baseline) / (1 − baseline)`, where P(bad) is P(YES) for a
bad event (`direction −1`) and P(NO) for a good one (`+1`). A reading at or
better than normal contributes **0**, never a positive amount — the gate can
only dampen, so a calm series has nothing to add and must not cancel out an
alarming one. The blend is the weight-normalised mean, in [-1, 0]; calm series
still count in the denominator, so forcing risk_off takes broad stress rather
than one alarming market. `direction` and `baseline` are hand-set priors, same
spirit as `catalysts.py`'s `EVENT_TYPES` — argue with them in `MACRO_SERIES`.

**Ships populated** with series verified against the live API on 2026-09-22:
P(Fed hike at the next FOMC, `KXFEDDECISION`), P(core CPI m/m above 0.3%,
`KXCPICORE`), P(US recession starts this year, `KXRECSSNBER`) and government
shutdown (`KXGOVSHUT`, only listed around funding deadlines). `fetch()` pages
through every open market in a series and reads only the soonest-closing
event. Series with several markets per event name the ones to read via
`outcomes` (ticker suffixes, summed — so they must be mutually exclusive, e.g.
hike-25 + hike->25). Kalshi's catalog changes; if a series stops returning
markets, `cryptoyolo.macro.list_series(query="fed")` finds current tickers.
Needs a Kalshi API key (RSA key pair, see `.env.example`); without one,
`fetch()` skips and the regime fold-in is a no-op.

**Only ever dampens, never boosts.** See the regime section below.

### 7 · Kalshi crypto price-prediction markets

`kalshi_prediction.py`. Unlike Feature 6, this IS a scored family — it feeds
the composite, not the regime gate. A Kalshi crypto series
(`config.py::KALSHI_CRYPTO_SERIES`, e.g. mapping `"BTC"` to a series ticker)
lists a ladder of "will price be above \$X at close" markets sharing one
expiration. Interpolating that ladder's (strike, P(YES)) points at the
asset's **current spot price** gives the market's own estimate of
`P(price ends above where it is right now)` — a directional read priced by
people with money on the line.

No hand-set `direction` prior is needed here, unlike `MACRO_SERIES`: "YES"
always and only means "price ends higher", so the interpolated probability
itself *is* the signal. `deviation = (p_up − 0.5) × 2`, clamped to [-1, 1] —
a coin-flip reading contributes nothing, a near-certain one contributes its
full value. Only the soonest-closing event's ladder is read (all open
markets are paged through; each series' latest fetch only, expired markets
dropped). A spot price outside the observed strikes scores **0 (no opinion)**
— a ladder entirely above spot only bounds P(up), it doesn't price it.

Two honest caveats, matching Feature 6's:

- **Short-dated.** These markets are typically same-day or same-week
  expiries, while this book generally holds 1–30 days (`exits.horizon_days`).
  Treat this as a near-term tilt, not a horizon match — `ScoreWeights.kalshi_prediction`
  defaults low for that reason.
- **Only single-strike "above" markets are read.** Kalshi also lists "between
  \$X and \$Y" range buckets on some series; those aren't a simple point on a
  survival curve and are skipped rather than mis-modelled.

**Ships empty, like `SCHEDULED_EVENTS`.** `KALSHI_CRYPTO_SERIES` tickers are
placeholders, not verified. Run `cryptoyolo.kalshi_prediction.list_series(query="bitcoin")`
(needs working credentials) to find the real ones. Reuses the same Kalshi API
key as Feature 6 — no separate credential.

### Cross-sectional momentum

Not a family — a blend coefficient. `technical_final = (1 −
xsec_momentum_blend)·technical + xsec_momentum_blend·xsec`, where `xsec` is the
asset's trailing-return rank (≈1-month and 3-month) z-scored **across the
universe** and squashed to [−1, 1]. The isolated technical score only ever sees
an asset against its own history; ranking it against the other 19 is one of the
few crypto factors that survives out-of-sample. Set the blend to `0.0` to
restore the isolated score. `technical_raw` and `xsec` are both exposed in the
score table.

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

Risk-first, not fixed-dollar. Quantity is set so that being stopped out costs
exactly `risk_per_trade_pct` of the **risk basis**.

**Risk basis is live equity** (`risk.size_off_live_equity`, default on): cash
plus marked-to-market positions, recomputed every cycle. So the budget grows
after a good stretch and shrinks in a drawdown without a config edit.
`account_equity_usd` is only the fallback when live equity can't be read.

**Stop distance is scaled to the holding horizon** (`risk.stop_scaling =
"horizon"`): `atr_stop_mult · dailyATR(14) · √stop_horizon_days`, clamped to
`[stop_min_pct, stop_max_pct]`. The old `1.5 · dailyATR` stop was ~3-5% for a
major — inside one day's range, so a multi-week thesis got whipsawed out on
noise. `√10 ≈ 3.2×` wider fixes that; position size shrinks to hold the dollar
risk at `risk_per_trade_pct`. Set `stop_scaling = "legacy"` for the old stop.
Sizing off *hourly* ATR (the original mistake, now impossible) produced sub-1%
stops.

**Targets clear fees.** With `risk.fee_adjust_targets` the take-profit distance
is solved so the *net* reward:risk — the full round-trip cost (fees **and**
slippage, both legs) taken off the reward and added to the risk — comes out to
exactly `reward_risk_target`, then clamped at `target_pct_cap` of entry. The
ticket shows both gross and net R:R; if the cap or the regime scaler bit, net
R:R prints below target and that's the honest number.

**The regime gate scales the slate.** `regime.assess()` reads BTC's daily chart
with a `ma_band_pct` (default 2%) dead-band around the MA: `risk_on` (price
> MA·1.02 and the MA rising) → full size; `neutral` (inside the band, or above a
flat/falling MA) → `×neutral_exposure` (0.5); `risk_off` (price < MA·0.98, or a
deep drawdown) → new BUYs dropped entirely. If fewer than `regime.min_candles`
daily bars are available the gate abstains (neutral). The deployment cap is
multiplied by this. Exits are never gated.

**Macro can dampen the gate further, never loosen it.** When `regime.assess()`
is called with a `store` (as the pipeline does), it folds in `macro.py`'s
Kalshi-derived score: a bad-but-not-extreme reading multiplies exposure by
`1 + score × macro_downweight` (default 1.0, so a −0.2 score cuts 20%),
floored at `macro_min_multiplier`; a reading
at or below `macro_risk_off_threshold` (default -0.6) forces `risk_off`
outright, regardless of what the BTC trend alone said. A calm or supportive
macro reading never increases exposure past what the trend earned — this
input only ever removes risk. Silently skipped (no adjustment) whenever
`MACRO_SERIES` is empty, Kalshi isn't configured, or the snapshot is older
than `macro.stale_after_hours`.

**Correlation heat cap.** After the deployment/regime cap, `portfolio_risk`
computes `heat = sqrt(r' C r)` over the new BUY slate — `r` the per-trade dollar
risk, `C` the trailing `corr_lookback_days` return-correlation matrix (missing
pairs default to 0.8). If `heat` exceeds `max_portfolio_heat_pct` of equity the
whole slate scales down, and the run prints the heat, the perfectly-correlated
`gross`, and the `diversification_ratio` (heat / gross). Set
`correlation_sizing = False` to fall back to the gross deployment cap only.

Positions are capped per-trade (`max_position_pct`) and in aggregate
(`max_total_deployed_pct × regime scale`); when a cap binds, buys scale down
proportionally so the ranking is preserved.

### Purchases never exceed available cash

Spending is capped separately against your real balance, in three places,
because each catches something the others miss:

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

SELLs are exempt from the cash cap, the deployment cap **and the regime gate** —
an exit releases capital and reduces risk — and are separately capped at the
quantity you actually hold.

---

## Fees and execution cost

`CONFIG.fees` (`FeeConfig`). Binance.US charges ~40 bps taker/maker at the base
tier — a ~0.8% round trip, an order of magnitude above Binance.com. Against a 2R
target of ~8% that is a large slice of the edge, so it is now modelled
everywhere:

| Lever | Effect |
|---|---|
| `use_bnb_discount` | pay fees in BNB for a 25% cut (`bnb_discount_pct`) |
| lower the `taker_bps` / `maker_bps` | as you earn a lower fee tier with volume |
| `execution.entry_order_type = "LIMIT"` | entries cross the book by at most `entry_limit_cross_bps` — slippage is **capped**, not open-ended; exits stay MARKET so they always fill |
| `execution.max_entry_slippage_bps` | refuse an entry outright if the market has gapped past this vs the proposal price |
| `fees.slippage_bps` | modelled cost applied to **paper** fills, so the paper record isn't better than a live account can be |

Every proposal ticket shows `fee ≈ $x/side` and `R:R gross / net`. Paper fills
record the modelled fee and slippage in the order row, which is what lets
`evaluation` report fee drag as a share of realised P&L.

---

## Evaluation — did any of it work?

`evaluation.py`. Run it from the notebook (the **Evaluation & benchmark**
section) or directly:

```python
from cryptoyolo import evaluation
report = evaluation.summary(store, CONFIG)          # prints the full battery
evaluation.recommend_weights(report["forward_returns"])   # data-driven ScoreWeights
```

| Function | Answers |
|---|---|
| `forward_returns` | for every stored score, the asset's realised 1d/7d/30d return |
| `information_coefficient` | per-family Spearman IC, **per-run mean with an overlap-deflated t-stat** (`runs_eff` shows the effective independent-sample count — daily runs sharing a 7/30d window are not independent) plus a pooled IC |
| `quantile_spread` | mean forward return of the top-N composite minus the bottom-N — does the ranking separate anything? |
| `hit_rate` | share of bullish (bearish) calls that went up (down) |
| `realized_trades` / `trade_stats` | FIFO round-trip reconstruction from the order log (realised fee from the `orders.fee_usd` column, then the response blob, then an estimate): win rate, avg win/loss, profit factor, **fee drag as % of gross P&L** |
| `benchmark_returns` | buy-and-hold BTC and an equal-weight basket over the same window |
| `equity_stats` | total return, annualised Sharpe, max drawdown of the equity curve, and the gap vs BTC — restricted to **one** snapshot mode (paper and live never share a curve) |
| `regime_effectiveness` | realised forward return of the universe / top-N per recorded regime state — did `risk_off` runs actually precede weaker returns? |
| `recommend_weights` | leads with **`blend_verdict`** (does the composite beat its own best single component?); then weights ∝ measured IC, zeroed for any family with \|t\| < 2, keeping the hand-set weights if fewer than two families clear |

**With a few weeks of data every number is noisy.** A t-stat under ~2 is "no
evidence", not a finding. The equity curve needs `pipeline.run` to have recorded
snapshots on different days — it writes one per cycle to `equity_snapshots`.

**Prices are read from a local table, not re-fetched.** Forward returns and the
BTC / basket benchmark come from `prices_daily` — point-in-time daily closes
written every cycle by `build_scores` (and by `collect.py --prices`). So the IC,
Sharpe and benchmark numbers are the same each time you recompute them, and
`evaluation` runs with no network. Runs logged before this table existed have no
prices yet — run `./backfill_prices.py` once to fill them in
(`./backfill_prices.py --status` shows coverage). If a symbol still has no local
history the module falls back to a live fetch and stores the result.

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

### Run one paper cycle a day (macOS launchd)

The evaluation module is only as good as the record it scores. `collect.py` on a
schedule keeps the *inputs* fresh, but scores, proposals, equity snapshots and
regime verdicts only accrue when a cycle actually runs — so the equity curve,
the BTC benchmark and `regime_effectiveness()` never fill in if you only run the
notebook occasionally. `com.crypto-yolo.trader.plist` runs one unattended cycle
a day to close that gap:

```bash
cp com.crypto-yolo.trader.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.crypto-yolo.trader.plist
./run_cycle.py --auto-approve --paper      # run once by hand to confirm
```

It runs `run_cycle.py --auto-approve --paper` at 13:05 local — after the 08:00
collector pass, and clear of it so the two don't write SQLite at the same
instant. `--paper` is a hard floor: this job **cannot** place a live order even
if `CONFIG.execution.mode` is switched to `binance` for your manual runs. Output
goes to `data/trader.log` (raw) and `data/crypto-yolo.log` (structured, rotated).
`launchctl unload ~/Library/LaunchAgents/com.crypto-yolo.trader.plist` stops it.

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
./collect.py --catalysts --prices   # Reddit + news + GitHub + daily closes — the scheduled job
./collect.py                        # Reddit only (faster)
./collect.py --status               # what's collected so far, no network calls
```

`--prices` sweeps 1y of daily closes for the universe into the `prices_daily`
table, so `evaluation` always has a local price record even on days no trading
cycle runs. `build_scores` writes the same rows every cycle, so this is a
freshness top-up, not a requirement.

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
| `KALSHI_API_KEY_ID` / `KALSHI_PRIVATE_KEY_PATH` | Feature 6 (macro regime dampener) + Feature 7 (crypto price-prediction family) | both fetches skip; regime fold-in and the kalshi_prediction family are no-ops |
| `CRYPTO_YOLO_ALLOW_LIVE` | Live orders | orders stay validate-only |
| `CRYPTO_YOLO_LOG_LEVEL` | log verbosity (`INFO` default) | INFO |
| `CRYPTO_YOLO_NTFY_URL` | push alerts via [ntfy](https://ntfy.sh) | alerts still logged + macOS banner |
| `CRYPTO_YOLO_ALERT_WEBHOOK` | Slack/Discord alert webhook | as above |

---

## Logging and alerts

The unattended jobs log through `cryptoyolo/logsetup.py`: a rotating
`data/crypto-yolo.log` (2 MB × 5) plus stderr, with timestamps and levels. The
human `print()` transcript is unchanged — `data/collector.log` and
`data/trader.log` still hold that. Every cycle also logs one structured summary
line (`cycle … regime=… proposals=… executed=… rejected=… equity=…`), and
`pipeline.run` returns it under `result["summary"]`.

`cryptoyolo/notify.py` sends best-effort alerts — via an ntfy topic, a
Slack/Discord webhook, and/or a macOS Notification Center banner — for the
events in `CONFIG.notify`: an order rejected, a stop/target/trailing/horizon
exit filling, the regime gate going `risk_off`, the price feed falling through
to yfinance, and equity drawing down past `notify.drawdown_alert_pct` (10%).
Routine fills are off by default. With nothing configured an alert is just a log
line. `CONFIG.notify.enabled = False` turns it all off.

## Development

```bash
pip install -r requirements-dev.txt
pytest                          # ~84 tests, no network, throwaway SQLite
python3 build_notebook.py       # regenerate the (gitignored) notebook
```

The repo is under git; `data/` (history, logs, the SQLite file) and the
generated notebook are ignored. Tests never open `data/` — they run against a
`tmp_path` database.

---

## Dependencies

Everything needed is already installed in this environment. `ccxt`, `praw`,
`feedparser` and `vaderSentiment` are deliberately *not* used — see
`requirements.txt` for why. `cryptography` is a real dependency, not
optional-in-practice like the others above: `macro.py` imports it
unconditionally to RSA-PSS-sign Kalshi requests, even though `macro.fetch()`
itself is a no-op without `KALSHI_API_KEY_ID` set.
