"""Configuration for the crypto-yolo dashboard.

Everything tunable lives here. The notebook imports `CONFIG` and can override any
field at runtime before running the pipeline.

Secrets are NEVER stored in this file. They are read from environment variables
(or a local .env file, which should be gitignored). See .env.example.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def load_dotenv(path: Path | str | None = None) -> dict[str, str]:
    """Minimal .env loader. Does not overwrite already-set env vars."""
    path = Path(path) if path else PROJECT_DIR / ".env"
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# Asset universe
# --------------------------------------------------------------------------
# `symbol` is the canonical ticker used everywhere internally.
# `aliases` feed the Reddit mention matcher (case-insensitive whole-word match).
# `github` is owner/repo for developer-activity scanning (None = skip).
# `ambiguous` marks tickers that are also common English words; for those the
# bare ticker only counts when written as $TICKER, to avoid absurd false
# positives ("ONE", "GAS", "SUI", "NEAR" etc. appear constantly in normal text).


@dataclass(frozen=True)
class Asset:
    symbol: str
    name: str
    aliases: tuple[str, ...] = ()
    github: str | None = None
    ambiguous: bool = False
    coingecko_id: str | None = None


UNIVERSE: tuple[Asset, ...] = (
    Asset("BTC", "Bitcoin", ("bitcoin", "xbt"), "bitcoin/bitcoin", coingecko_id="bitcoin"),
    Asset("ETH", "Ethereum", ("ethereum", "ether"), "ethereum/go-ethereum", coingecko_id="ethereum"),
    Asset("SOL", "Solana", ("solana",), "anza-xyz/agave", coingecko_id="solana"),
    Asset("XRP", "XRP", ("ripple",), "XRPLF/rippled", coingecko_id="ripple"),
    Asset("ADA", "Cardano", ("cardano",), "IntersectMBO/cardano-node", ambiguous=True, coingecko_id="cardano"),
    Asset("DOGE", "Dogecoin", ("dogecoin",), "dogecoin/dogecoin", coingecko_id="dogecoin"),
    Asset("AVAX", "Avalanche", ("avalanche",), "ava-labs/avalanchego", coingecko_id="avalanche-2"),
    Asset("LINK", "Chainlink", ("chainlink",), "smartcontractkit/chainlink", ambiguous=True, coingecko_id="chainlink"),
    Asset("DOT", "Polkadot", ("polkadot",), "paritytech/polkadot-sdk", ambiguous=True, coingecko_id="polkadot"),
    Asset("LTC", "Litecoin", ("litecoin",), "litecoin-project/litecoin", coingecko_id="litecoin"),
    Asset("BCH", "Bitcoin Cash", ("bitcoin cash", "bcash"), "bitcoin-cash-node/bitcoin-cash-node", coingecko_id="bitcoin-cash"),
    Asset("ATOM", "Cosmos", ("cosmos",), "cosmos/gaia", ambiguous=True, coingecko_id="cosmos"),
    Asset("UNI", "Uniswap", ("uniswap",), "Uniswap/v3-core", ambiguous=True, coingecko_id="uniswap"),
    Asset("SHIB", "Shiba Inu", ("shiba inu", "shibainu"), None, coingecko_id="shiba-inu"),
    Asset("NEAR", "NEAR Protocol", ("near protocol",), "near/nearcore", ambiguous=True, coingecko_id="near"),
    Asset("APT", "Aptos", ("aptos",), "aptos-labs/aptos-core", ambiguous=True, coingecko_id="aptos"),
    Asset("ARB", "Arbitrum", ("arbitrum",), "OffchainLabs/nitro", ambiguous=True, coingecko_id="arbitrum"),
    Asset("OP", "Optimism", ("optimism",), "ethereum-optimism/optimism", ambiguous=True, coingecko_id="optimism"),
    # INJ: the chain repo is not public under any InjectiveLabs path that
    # resolves (injective-core / injective-chain both 404). Only the TS SDK is
    # public, and an SDK's release cadence is a poor proxy for chain upgrades —
    # a misleading dev signal is worse than none, so this stays None.
    Asset("INJ", "Injective", ("injective",), None, ambiguous=True, coingecko_id="injective-protocol"),
    Asset("SUI", "Sui", ("sui network",), "MystenLabs/sui", ambiguous=True, coingecko_id="sui"),
)

ASSET_BY_SYMBOL: dict[str, Asset] = {a.symbol: a for a in UNIVERSE}


# --------------------------------------------------------------------------
# Scheduled dated events (Feature 5)
# --------------------------------------------------------------------------
# Hand-maintained schedule of KNOWN FUTURE dates. This is the one genuinely
# forward-looking input in the system: unlike a news headline, a cliff unlock
# on the 14th is knowable on the 1st. Keep it current — a stale schedule is
# worse than an empty one because it scores confidently on events that already
# passed (they drop out automatically `lookback_days` after the date).
#
#   date        ISO date of the event (UTC)
#   kind        token_unlock | mainnet | upgrade | etf_decision | listing | undefined
#   magnitude   for token_unlock: % of circulating supply unlocking (e.g. 4.2).
#               for everything else: a 0..1 subjective size (0.3 minor, 1.0 major).
#   direction   -1 bearish / +1 bullish / 0 let magnitude+kind decide
#   note        free text, shown in the rationale
#
# EDIT THIS. The entries below are illustrative placeholders, not researched
# facts — replace them with real dates before trusting the `events` family.

@dataclass(frozen=True)
class ScheduledEvent:
    symbol: str
    date: str
    kind: str = "unlock"
    magnitude: float = 0.5
    direction: int = 0
    note: str = ""


SCHEDULED_EVENTS: tuple[ScheduledEvent, ...] = (
    # e.g. ScheduledEvent("ARB", "2026-09-16", "token_unlock", 2.1, -1,
    #                      "monthly cliff unlock, ~2% of circulating"),
    # e.g. ScheduledEvent("OP",  "2026-09-30", "token_unlock", 5.4, -1,
    #                      "large scheduled unlock"),
    # e.g. ScheduledEvent("SUI", "2026-09-01", "token_unlock", 1.3, -1, ""),
)


# --------------------------------------------------------------------------
# Macro event contracts (Feature 6 — Kalshi)
# --------------------------------------------------------------------------
# Hand-maintained list of Kalshi SERIES to read as macro signals (see
# cryptoyolo/macro.py). A Kalshi series is a recurring family of event
# contracts — e.g. one market per FOMC meeting — so unlike SCHEDULED_EVENTS
# this does not need a date: `macro.fetch()` always picks the soonest-closing
# OPEN event in the series (the next meeting, the next print).
#
#   label           human-readable, shown in the regime note
#   series_ticker   Kalshi's series ticker. Verified against the live API on
#                   2026-09-22, but Kalshi's catalog changes — re-check with
#                   `macro.list_series(query="fed")` if a series stops
#                   returning open markets.
#   direction       does a YES resolution favor (+1) or hurt (-1) risk assets
#                   like crypto? 0 = no inherent direction — any move away
#                   from `baseline` is scored as elevated uncertainty
#                   (bearish), whichever way it points.
#   weight          relative importance within the blended macro score.
#   baseline        the P(YES) that counts as NORMAL for this event — the
#                   reference each reading is scored against (not 50%). Only
#                   readings worse than baseline dampen the gate; certainty of
#                   the bad outcome costs the full weight. Default 0.5 suits a
#                   genuine coin-flip proposition only.
#   outcomes        market-ticker suffixes to read within the nearest event,
#                   for series that list several markets per event. Their
#                   probabilities are SUMMED, so they must be mutually
#                   exclusive: e.g. ("H25", "H26") on KXFEDDECISION = P(any
#                   hike), or a single strike ("T0.3",) on a threshold ladder
#                   like KXCPICORE. Empty = one yes/no market per event
#                   (averaged if there are several).
#
# The directions, weights, strikes and baselines below are judgement calls,
# not fitted coefficients — argue with them.

@dataclass(frozen=True)
class MacroSeries:
    label: str
    series_ticker: str
    direction: int = -1
    weight: float = 1.0
    outcomes: tuple[str, ...] = ()
    baseline: float = 0.5

    def __post_init__(self):
        if not 0.0 < self.baseline < 1.0:
            raise ValueError(f"MacroSeries {self.series_ticker}: baseline must be "
                             f"strictly between 0 and 1, got {self.baseline}")


MACRO_SERIES: tuple[MacroSeries, ...] = (
    # Categorical per-meeting event: H0 hold, H25/H26 hike 25/>25bp,
    # C25/C26 cut 25/>25bp. KXFED (the rate-level ladder) carries the same
    # information in a less direct shape.
    # Baseline ~15%: most meetings are holds or cuts; a hike is the exception.
    MacroSeries("Fed hikes at the next FOMC meeting", "KXFEDDECISION", -1, 1.0,
                outcomes=("H25", "H26"), baseline=0.15),
    # Monthly core-CPI m/m ladder; >0.3% m/m (~3.7%+ annualised) is a hot print.
    # Baseline ~20%: hot prints happen, but roughly one month in five.
    MacroSeries("Core CPI m/m prints above 0.3%", "KXCPICORE", -1, 0.8,
                outcomes=("T0.3",), baseline=0.20),
    # One yes/no market per calendar year; the nearest is the current year.
    # Baseline ~15%: roughly the long-run odds of a recession starting in any
    # given year.
    MacroSeries("US recession starts this year (NBER)", "KXRECSSNBER", -1, 0.8,
                baseline=0.15),
    # No open markets as of 2026-09-22 — Kalshi lists them around funding
    # deadlines; contributes nothing until then.
    # Baseline ~25%: markets only list near funding deadlines, when some
    # shutdown risk is the norm.
    MacroSeries("Government shutdown", "KXGOVSHUT", -1, 0.6, baseline=0.25),
)


# --------------------------------------------------------------------------
# Kalshi crypto price markets (Feature 7)
# --------------------------------------------------------------------------
# Hand-maintained mapping of universe SYMBOL -> the Kalshi series that lists
# that asset's own "will price be above $X" markets (see
# cryptoyolo/kalshi_prediction.py). Distinct from MACRO_SERIES above, which
# reads Kalshi's MACRO event contracts (Fed, CPI, ...) to dampen the regime
# gate -- this reads Kalshi's CRYPTO-specific markets and turns them into a
# per-asset directional score in the composite, the same way positioning or
# events do.
#
#   symbol          a symbol in UNIVERSE this series predicts, e.g. "BTC".
#   series_ticker   Kalshi's series ticker. THESE ARE PLACEHOLDERS, NOT
#                   VERIFIED TICKERS, Kalshi's catalog changes -- run
#                   `kalshi_prediction.list_series(query="bitcoin")` (needs
#                   working credentials) to find the real ones.
#
# Ships EMPTY, same as SCHEDULED_EVENTS -- this family scores
# 0.0 for every asset until populated with real tickers.

@dataclass(frozen=True)
class KalshiCryptoSeries:
    symbol: str
    series_ticker: str


KALSHI_CRYPTO_SERIES: tuple[KalshiCryptoSeries, ...] = (
    # e.g. KalshiCryptoSeries("BTC", "KXBTCD"),
    # e.g. KalshiCryptoSeries("ETH", "KXETHD"),
)


# --------------------------------------------------------------------------
# Chart timeframes -> (lookback, candle interval)
# --------------------------------------------------------------------------
# Interval strings are canonical; each price source maps them to its own vocab.
TIMEFRAMES: dict[str, dict[str, str]] = {
    "1h":  {"lookback": "1h",  "interval": "1m",  "label": "Past hour"},
    "1d":  {"lookback": "1d",  "interval": "5m",  "label": "Past day"},
    "1w":  {"lookback": "7d",  "interval": "1h",  "label": "Past week"},
    "1mo": {"lookback": "30d", "interval": "4h",  "label": "Past month"},
    "1y":  {"lookback": "365d", "interval": "1d", "label": "Past year"},
}


@dataclass
class BandConfig:
    """Overlay bands drawn on price charts."""
    bollinger: bool = True
    bollinger_window: int = 20
    bollinger_stds: tuple[float, ...] = (1.0, 2.0)
    keltner: bool = False
    keltner_window: int = 20
    keltner_atr_mult: float = 2.0
    donchian: bool = False
    donchian_window: int = 20
    moving_averages: tuple[int, ...] = (20, 50, 200)
    show_volume: bool = True
    show_rsi: bool = True


@dataclass
class ScoreWeights:
    """How the feature families combine into one composite score.

    These are hand-set priors, not fitted parameters. Once the database holds a
    few weeks of runs, `evaluation.recommend_weights()` measures each family's
    information coefficient against realised forward returns and prints a
    data-driven weight vector to replace these guesses. Do that before trusting
    them.

    `positioning` and `events` were added after the fact and deliberately do NOT
    rescale the others: normalized() divides by the sum, so adding a family
    alongside the existing weights preserves every prior ratio and simply makes
    room. Set any weight to 0.0 to drop that family entirely.
    """
    technical: float = 0.50
    social: float = 0.20
    catalyst: float = 0.30
    positioning: float = 0.10
    # Scheduled dated events (token unlocks, mainnet dates, ETF decisions,
    # listing effective dates). The only genuinely forward-looking family here.
    # Defaults to 0.0 because config.SCHEDULED_EVENTS ships EMPTY - a non-zero
    # weight on an all-zero family would just dilute every other family's real
    # contribution. Raise this (0.15 is a sensible start) once you populate the
    # schedule. See calendar_events.py.
    events: float = 0.0
    # Feature 7: per-asset directional read from Kalshi's own crypto price
    # markets (config.KALSHI_CRYPTO_SERIES) -- see kalshi_prediction.py.
    # Defaults to 0.0 for the same reason `events` does: KALSHI_CRYPTO_SERIES
    # ships EMPTY, and a non-zero weight on an all-zero family only dilutes
    # every other family's real contribution. Raise it (0.10-0.15 is a
    # reasonable start) once you populate the series list -- and keep it
    # modest even then: these markets are short-dated (same-day/week) versus
    # the 1-30 day horizon the rest of this book trades on.
    kalshi_prediction: float = 0.0
    # Cross-sectional momentum is NOT a family — it is a blend coefficient that
    # folds an asset's trailing-return rank *relative to the universe* into its
    # technical score. `technical_final = (1 - b) * technical + b * xsec_z`.
    # 0.0 restores the isolated per-asset technical score. Cross-sectional
    # momentum is one of the few robust crypto factors, and the isolated
    # technical score cannot see it.
    xsec_momentum_blend: float = 0.30

    def normalized(self) -> "ScoreWeights":
        fam = (self.technical, self.social, self.catalyst,
               self.positioning, self.events, self.kalshi_prediction)
        total = sum(fam)
        if total <= 0:
            return ScoreWeights(1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6,
                                xsec_momentum_blend=self.xsec_momentum_blend)
        return ScoreWeights(
            self.technical / total, self.social / total, self.catalyst / total,
            self.positioning / total, self.events / total,
            self.kalshi_prediction / total,
            xsec_momentum_blend=self.xsec_momentum_blend,
        )


@dataclass
class RiskConfig:
    """Position sizing and guardrails."""
    account_equity_usd: float = 10_000.0
    # Size against live equity (cash + marked-to-market positions) instead of
    # the static `account_equity_usd` above. This is what makes the book
    # compound: after a good run the risk budget grows, after a drawdown it
    # shrinks, without you editing a config field every week. Falls back to
    # `account_equity_usd` whenever live equity can't be read (no credentials,
    # API error) - unknown is not treated as zero.
    size_off_live_equity: bool = True
    risk_per_trade_pct: float = 1.0          # % of equity risked between entry and stop
    max_position_pct: float = 20.0           # cap on any single position as % of equity
    max_total_deployed_pct: float = 60.0     # cap across all proposed trades
    atr_stop_mult: float = 1.5               # stop distance = mult * ATR(14) on DAILY candles
    reward_risk_target: float = 2.0          # take-profit distance = R:R * stop distance

    # ---- Correlation-aware sizing --------------------------------------
    # "1% risk each" on three alt longs that move together is really ~2.5% of
    # one bet. This caps the CORRELATION-ADJUSTED risk of the new BUY slate:
    # portfolio_heat = sqrt(r' C r) with r the per-trade $ risk vector and C the
    # trailing return-correlation matrix. If it exceeds `max_portfolio_heat_pct`
    # of equity the whole slate scales down. The regime gate is a blunt macro
    # switch; this is the micro one. Set False to disable (fall back to the
    # gross deployment cap only).
    correlation_sizing: bool = True
    max_portfolio_heat_pct: float = 2.5     # cap on sqrt(r' C r) as % of equity
    corr_lookback_days: int = 60            # trailing window for the correlation matrix

    # ---- Stop distance vs holding horizon --------------------------------
    # "horizon" scales the stop by sqrt(stop_horizon_days) so a multi-week hold
    # is not stopped out by a single day's noise. 1.5x daily ATR is ~3-5% for a
    # major, which for a 10-30 day thesis sits *inside* the normal daily range
    # and guarantees whipsaw. "legacy" keeps the old 1.5x-daily-ATR stop.
    stop_scaling: str = "horizon"           # "horizon" | "legacy"
    stop_horizon_days: float = 10.0         # expected trade duration used for stop sizing
    stop_min_pct: float = 3.0              # floor on stop distance, % of entry
    stop_max_pct: float = 40.0            # cap, so a wild alt can't imply a 90% stop
    # A clamped-wide stop plus a fixed reward:risk can imply a target 80%+ away
    # that no 30-day hold will ever reach - the trade then only ever exits via
    # stop / trailing / horizon and the printed R:R is fiction. Cap the target
    # distance at this % of entry so the R:R is honestly reduced instead.
    target_pct_cap: float = 60.0

    # ---- Fee-aware targets ----------------------------------------------
    # Widen the take-profit so that the *net* reward:risk after a round-trip
    # (fees + modelled slippage, both legs) still equals `reward_risk_target`.
    # Without it a nominal 2:1 trade is really ~1.8:1 once Binance.US costs are
    # paid twice.
    fee_adjust_targets: bool = True

    # ---- Candidate selection ------------------------------------------
    # "relative" ranks the cross-section and takes the top `max_proposals`
    # names whose composite clears `min_composite_floor` - so a broad down day
    # can still surface the best *relative* longs, and a broad up day doesn't
    # wave everything through. "absolute" keeps the old fixed-threshold gate.
    selection_mode: str = "relative"        # "relative" | "absolute"
    min_composite_floor: float = 0.03       # relative mode: never trade pure noise
    # A held asset is only *trimmed* on the buy-side scoreboard once its
    # composite is this bearish. Deliberately wider than min_composite_floor:
    # trimming on a -0.04 blip just churns fees, and exits.py already forces a
    # full exit at score_reversal_threshold (-0.15). Applies in both selection
    # modes.
    trim_threshold: float = 0.10

    # Skip dust trades. Deliberately well above the venue floor: Binance.US
    # MIN_NOTIONAL on BTCUSDT is $1.00, but sub-$15 positions are mostly fees.
    min_notional_usd: float = 15.0
    max_proposals: int = 3
    # Minimum |composite| to be proposable in "absolute" selection mode. Raise
    # it to be pickier; set it to 0.0 to always surface a full slate. It is
    # deliberately not 0: on a genuinely directionless day the honest output is
    # one candidate, or none, rather than three manufactured ones. Ignored when
    # `selection_mode == "relative"` (which uses `min_composite_floor`).
    min_composite_score: float = 0.10
    # Exits are proposed and executed before entries, so their proceeds are
    # spendable by the buys in the same run. That projection is discounted by
    # this much to absorb slippage and fees - the sale rarely nets exactly the
    # quoted notional, and sizing buys off an optimistic figure is how you end
    # up with a rejected final order.
    exit_proceeds_haircut_pct: float = 1.0


@dataclass
class FeeConfig:
    """Trading costs — the return the engine gives away on every trade.

    Binance.US charges roughly 0.4% taker / 0.4% maker at the base fee tier, an
    order of magnitude above Binance.com's 0.1%. A round trip is therefore
    ~0.8%+, which against a nominal 2R target of ~8% is a large slice of the
    edge. Three levers here:

      * `use_bnb_discount`   pay fees in BNB for a 25% cut (must hold BNB).
      * a lower fee tier      earned with 30-day volume; drop the bps below.
      * maker over taker      post a limit that rests instead of crossing.

    `slippage_bps` is the *modelled* execution cost applied to PAPER fills so
    that the paper track record is not systematically better than live. Once
    live fills accumulate, compare the recorded per-fill `slippage_bps` /
    `fee_bps` in the order log against these assumptions and adjust.
    """
    enabled: bool = True
    taker_bps: float = 40.0            # Binance.US base tier ~= 0.40%
    maker_bps: float = 40.0           # ~= 0.40%; earn a lower tier with volume
    use_bnb_discount: bool = False
    bnb_discount_pct: float = 25.0

    # Paper-fill realism. A market order pays the spread plus some impact; a
    # marketable limit caps that. This is a flat estimate — refine from the
    # per-fill slippage recorded in the order log.
    slippage_bps: float = 6.0
    # Extra impact for orders that are large relative to the asset. Applied as
    # `impact_bps_per_1pct_adv * (notional / est_daily_dollar_volume * 100)`.
    # Left at 0 by default (daily volume isn't always available); set it if you
    # trade thin alts in size.
    impact_bps_per_1pct_adv: float = 0.0

    def _apply_bnb(self, bps: float) -> float:
        return bps * (1 - self.bnb_discount_pct / 100.0) if self.use_bnb_discount else bps

    @property
    def effective_taker_bps(self) -> float:
        return self._apply_bnb(self.taker_bps) if self.enabled else 0.0

    @property
    def effective_maker_bps(self) -> float:
        return self._apply_bnb(self.maker_bps) if self.enabled else 0.0

    @property
    def round_trip_bps(self) -> float:
        """Fees only: entry taker + exit taker. Used for the 'fee ≈ $x' label."""
        return 2 * self.effective_taker_bps

    @property
    def round_trip_cost_bps(self) -> float:
        """Full round-trip drag a target must clear: fees + slippage, both legs.

        `round_trip_bps` is fees alone; a real round trip also pays the spread
        twice, so target-widening uses this larger figure.
        """
        slip = self.slippage_bps if self.enabled else 0.0
        return 2 * self.effective_taker_bps + 2 * slip


@dataclass
class RegimeConfig:
    """Market-regime gate: how much long exposure the tape currently justifies.

    Long-only alt exposure while BTC is in a downtrend is structurally
    negative-EV — the majors drag everything with them (cross-correlation runs
    0.7-0.9), so a great relative pick still loses money. This scales the whole
    slate's deployment by where BTC sits against its long moving average, and
    can block new entries outright in a hard downtrend. Exits are never gated.

    State is decided on BTC daily candles, with a dead-band around the MA so
    that price chopping across it does not flip the gate every cycle:
      risk_on   price > MA * (1 + ma_band_pct/100) and the MA rising
      risk_off  price < MA * (1 - ma_band_pct/100), or a deep drawdown
      neutral   inside the band, or above a flat/falling MA
    """
    enabled: bool = True
    ma_days: int = 200
    # Dead-band around the MA, in %. Price must be this far past the MA to flip
    # the gate; inside the band the state is "neutral". 0 = a bare MA cross
    # (noisy). 2% is a reasonable default for BTC.
    ma_band_pct: float = 2.0
    # If fewer than this many daily candles are available the MA is unreliable
    # and the gate abstains (returns neutral with a note).
    min_candles: int = 150
    # Deployment multiplier applied to max_total_deployed_pct per state.
    risk_on_exposure: float = 1.0
    neutral_exposure: float = 0.5
    risk_off_exposure: float = 0.0
    # In risk_off, drop new BUY proposals entirely rather than just shrinking
    # them. Exits and existing positions are unaffected.
    block_new_entries_when_risk_off: bool = True
    # A drawdown from the trailing high past this also forces risk_off even if
    # price is still inside the MA band.
    drawdown_risk_off_pct: float = 25.0

    # ---- Macro backdrop (Kalshi event contracts, see macro.py) -----------
    # Folds macro.score()'s blended read into this gate the same way the
    # drawdown override works: it can only ever DAMPEN the deployment scale
    # the BTC trend already computed, or force risk_off outright when it
    # crosses macro_risk_off_threshold — it never boosts past what the trend
    # alone earned. A no-op whenever assess() is called without a `store`,
    # macro is disabled, MACRO_SERIES is empty, or there is no fresh Kalshi
    # snapshot — the same graceful-degradation contract as every other input
    # to this gate.
    macro_enabled: bool = True
    macro_risk_off_threshold: float = -0.6   # macro score at/below this forces risk_off
    macro_downweight: float = 1.0            # exposure_scale x (1 + score * this), floored below
    macro_min_multiplier: float = 0.5        # floor for the dampening multiplier


@dataclass
class EventsConfig:
    """Feature 5: scheduled, dated catalysts.

    Distinct from the `catalyst` family, which scores news that has *already*
    been published (and usually already moved the price). This family scores
    events with a KNOWN FUTURE DATE — token unlocks, mainnet launches, ETF
    decision deadlines, exchange-listing effective dates — by how close the
    date is and how big the event is. A cliff unlock worth 8% of circulating
    supply nine days out is a knowable headwind in a way a price move is not.

    The schedule is maintained by hand in `SCHEDULED_EVENTS` (below) and/or
    pulled from a public unlock feed when `fetch_unlocks` is set and the source
    is reachable. Each event scores on a tent function: it ramps up as the date
    approaches, peaks in the `peak_window_days` before it, then decays after.
    """
    enabled: bool = True
    lookahead_days: float = 30.0        # ignore events further out than this
    lookback_days: float = 3.0          # keep scoring briefly after the date
    peak_window_days: float = 7.0       # full weight inside this many days of the event
    # Magnitude -> score scale. An unlock of `unlock_pct_full_weight` of
    # circulating supply (or an event with magnitude 1.0) maps to a full-
    # strength signal; smaller ones scale down linearly.
    unlock_pct_full_weight: float = 5.0
    fetch_unlocks: bool = False
    unlocks_url: str = ""              # optional public JSON feed; failquietly if unset


@dataclass
class MacroConfig:
    """Feature 6: macro backdrop via Kalshi event-contract prices.

    Kalshi is a CFTC-regulated prediction market: a contract settles at $1 if
    a YES proposition resolves true, $0 otherwise, so its live price IS a
    market-implied probability. This pulls a small, hand-maintained set of
    MACRO series (`config.MACRO_SERIES`) — Fed decisions, CPI prints,
    government-shutdown risk — the kind of broad macro uncertainty that moves
    risk assets generally, crypto included, in a way no per-asset signal here
    can see coming. See `cryptoyolo/macro.py`.

    Requires an authenticated Kalshi API key (RSA key pair) — see
    .env.example. Fails soft with no credentials: `fetch()` skips, `score()`
    reports zero series, and `regime.py`'s fold-in is a no-op.
    """
    enabled: bool = True
    base_url: str = "https://api.elections.kalshi.com"
    api_prefix: str = "/trade-api/v2"
    request_delay: float = 0.25
    # For a series without `outcomes`: how many of the nearest event's
    # markets to average (a plain yes/no series has just one).
    max_markets_per_series: int = 3
    # Ignore markets with less than this much lifetime volume — an untraded
    # market's price is not a meaningful probability, it's just wherever the
    # order book happened to open.
    min_volume: int = 1
    # A snapshot older than this many hours is treated as no-data, not stale
    # data — the regime gate should not act on a reading from before the last
    # `macro.fetch()` had a chance to run (e.g. the 8-hourly collector job).
    stale_after_hours: float = 36.0


@dataclass
class KalshiPredictionConfig:
    """Feature 7: per-asset directional signal from Kalshi's own crypto price
    markets — distinct from `MacroConfig` above, which reads Kalshi's MACRO
    event contracts to dampen the regime gate.

    A Kalshi crypto series (`config.KALSHI_CRYPTO_SERIES`) lists a ladder of
    "will price be above $X at close" markets sharing one expiration. Given
    that ladder's (strike, P(YES)) pairs, interpolating at the CURRENT spot
    price gives the market's own estimate of P(price ends above where it is
    right now) — a directional read priced by people with money on the line,
    genuinely independent of this book's own technical/social/catalyst
    inputs. No hand-set `direction` prior is needed here, unlike
    `MacroSeries`: "YES" always and only means "price ends higher", so the
    probability itself IS the signal.

    Two honest caveats, matching macro.py's:

      Short-dated. These markets are typically same-day or same-week
      expiries, while this book generally holds 1-30 days
      (`exits.horizon_days`). Treat it as a near-term tilt, not a horizon
      match — hence `ScoreWeights.kalshi_prediction` defaulting low.

      Only "above/below a single strike" markets are read. Kalshi also lists
      "between $X and $Y" range buckets on some series; those aren't a
      straightforward (strike, P(YES>strike)) point and are skipped rather
      than mis-modelled — see `kalshi_prediction._strike`.

    Requires the same Kalshi API key as `MacroConfig` (RSA key pair) — see
    .env.example. Fails soft with no credentials, no configured series, or
    fewer than 2 open strikes to interpolate: the family scores 0.0.
    """
    enabled: bool = True
    request_delay: float = 0.25
    # Same reasoning as MacroConfig.min_volume: an untraded strike's price is
    # not a meaningful probability.
    min_volume: int = 1
    # Short-dated markets go stale fast — much sooner than macro's 36h, since
    # a same-day contract's remaining life can be measured in hours.
    stale_after_hours: float = 6.0


@dataclass
class ExitConfig:
    """When to close an existing position.

    Five independent triggers, each individually switchable. They are evaluated
    every run against live prices and the position's own recorded entry terms,
    so an exit no longer depends on the asset happening to rank well on the
    buy-side scoreboard.

    Precedence when several fire at once is the order listed in exits.TRIGGERS:
    stop and target are level events that already happened, so they outrank
    horizon and score-based reasons.
    """
    enabled: bool = True

    # 1. Hard stop: price traded through the stop recorded at entry.
    stop_loss: bool = True

    # 2. Take profit: price reached the target recorded at entry.
    take_profit: bool = True
    take_profit_fraction: float = 100.0     # % of the position to sell on a hit

    # 3. Horizon expiry: the trade thesis has run out of time. A position held
    #    past its horizon is no longer the trade that was approved.
    horizon_expiry: bool = True
    horizon_days: float = 30.0

    # 4. Trailing stop: give back at most `trail_pct` from the high-water mark,
    #    but only once the position is `trail_activate_pct` in profit - otherwise
    #    it is just a tighter stop that fires on entry noise.
    trailing_stop: bool = True
    trail_pct: float = 8.0
    trail_activate_pct: float = 3.0

    # 5. Score reversal: the thesis that opened the position has inverted.
    score_reversal: bool = True
    score_reversal_threshold: float = -0.15

    # Exits get their own slots so they never compete with new entries for the
    # `max_proposals` budget. Set to 0 to disable exit proposals entirely.
    max_exit_proposals: int = 3

    # Positions with no recorded entry terms (bought outside this dashboard)
    # can still exit on score reversal, which needs no entry metadata.
    allow_exits_without_metadata: bool = True


@dataclass
class RedditConfig:
    # Data sources, tried in order until one returns rows. Reddit's anonymous
    # .json endpoints are 403 from most hosts, so without credentials the
    # keyless fallbacks are what actually work:
    #   oauth       official API (needs a free script app) - best quality
    #   arctic      Arctic Shift, a public Pushshift successor - has scores
    #   rss         Reddit's own Atom feeds - works keyless, but NO scores
    #   public_json legacy anonymous .json - usually 403, kept as a last resort
    sources: tuple[str, ...] = ("oauth", "arctic", "rss", "public_json")
    # All 14 candidates measured 2026-08-20 against Arctic Shift (200 items
    # each). Ordered best-first by mention-rate x asset-breadth x tone, so if a
    # run is cut short by rate limits the most informative sources are already
    # collected. Measured stats are in the README table.
    #
    # One carries a known caveat, kept because breadth was requested:
    # r/CryptoMoonShots is a pump-promotion venue (high volume, low information).
    # Watch whether it degrades the social score.
    # r/wallstreetbets was measured and dropped: 0.5% crypto mentions, 1 asset,
    # 0.00 tone confidence - it is an equities sub, not a crypto one.
    subreddits: tuple[str, ...] = (
        "CryptoMarkets",          # 18.5% mentions, 7 assets, tone 0.20
        "CryptoCurrency",         # 11.5%, 6 assets, tone 0.25 - freshest (13h/200)
        "ethtrader",              # 33.0%, 4 assets - ETH-dominated
        "Bitcoin",                # 29.0%, 2 assets - single-asset
        "defi",                   # 13.0%, 5 assets
        "SatoshiStreetBets",      # 12.5%, 8 assets - low activity (2349h/200)
        "solana",                 # 46.5%, 4 assets - single-asset
        "CryptoMoonShots",        #  9.0%, 4 assets - pump-shill venue
        "altcoin",                # 12.5%, 10 assets - best breadth, low activity
        "BitcoinMarkets",         # 19.0%, 2 assets
        "CryptoCurrencyTrading",  #  4.5%, 3 assets
        "CryptoTechnology",       #  8.0%, 3 assets
        "binance",                #  5.0%, 4 assets
    )
    # Seconds to wait between subreddits. The keyless sources are shared public
    # services; 14 subs x 2 calls per cycle will trip Arctic Shift's limiter
    # without pacing.
    inter_subreddit_delay: float = 2.5

    # Adaptive fetch order. Subreddits are re-ranked from what they have
    # actually delivered (see social.rank_subreddits) so the highest-yield
    # sources are fetched before a rate limiter can cut the pass short.
    adaptive_ranking: bool = True
    rank_refresh_hours: float = 24.0    # recompute once a day
    rank_window_days: float = 7.0       # look back this far when measuring
    collection_cadence_hours: float = 8.0   # matches the launchd schedule
    # Where to slot a subreddit that has no stats yet: after this many proven
    # leaders. Appending new sources last means a rate limit can starve them
    # forever, so they could never earn a rank.
    probation_after_top: int = 3
    listings: tuple[str, ...] = ("new", "hot")
    posts_per_listing: int = 100
    include_comments: bool = True
    comments_per_post: int = 40
    max_posts_for_comments: int = 15
    baseline_days: int = 7        # window used to compute the "normal" mention rate
    # Below this many mentions, a source's measured tone bias is mostly noise,
    # so the de-biasing correction is shrunk toward zero.
    min_samples_for_debias: int = 30
    velocity_window_hours: int = 24


@dataclass
class StockTwitsConfig:
    """StockTwits cashtag streams — https://api.stocktwits.com/api/2 (no key).

    The most information-dense social source measured: every message arrives
    already attributed to one symbol, so the mention matcher is bypassed
    entirely and there are no false positives to control for. Better still,
    ~63% carry a user-declared Bullish/Bearish label, which is a stated opinion
    rather than one inferred from a word list.
    """
    enabled: bool = True
    symbol_suffix: str = ".X"          # crypto namespace: BTC.X, ETH.X, ...
    request_delay: float = 1.2         # unauthenticated; no advertised limit
    # Request budget: one call per symbol. 0 = no cap (follow the universe).
    # A fixed number equal to the universe size silently drops any asset added
    # later, so this defaults to uncapped.
    max_symbols: int = 0
    # A user-declared label is far stronger evidence than lexicon inference.
    label_confidence: float = 0.9


@dataclass
class MastodonConfig:
    """Mastodon public hashtag timelines (no auth, 300 req/window).

    Measured 2026-08-20: only BROAD tags carry real traffic (#crypto 3.6
    posts/h, #bitcoin 3.5/h) while per-asset tags are effectively dead (#dot is
    one post per 33 hours). So we pull a few wide tags and run the normal
    extractor over them, rather than spending a request per asset on tags that
    return month-old content.
    """
    enabled: bool = True
    instance: str = "https://mastodon.social"
    hashtags: tuple[str, ...] = ("crypto", "bitcoin", "btc", "cryptocurrency",
                                 "ethereum", "defi", "altcoin")
    limit: int = 40
    request_delay: float = 0.8


@dataclass
class BizConfig:
    """4chan /biz/ — https://a.4cdn.org (read-only JSON, no key, no auth).

    Included for one specific reason: it is the only social venue measured that
    is genuinely willing to be negative. 43% of its scored posts are bearish,
    against a mean tone of +0.077 - versus StockTwits at +0.48 and Reddit at
    +0.04. Every other source is a place where people talk their own book, so
    without something like this the social score can only ever say "buy".

    Two caveats. The board is anonymous, so author diversity cannot damp
    brigading the way it does elsewhere - weights are deliberately flat. And the
    prose is crude enough that the lexicon picks up hostility unrelated to any
    asset, which is part of why this carries a reduced weight.
    """
    enabled: bool = True
    board: str = "biz"
    include_replies: bool = True    # catalog embeds last_replies: volume, no extra calls
    request_delay: float = 1.0
    # Damped relative to other sources: high noise, anonymous, no engagement signal.
    weight: float = 0.6


@dataclass
class PositioningConfig:
    """Perpetual funding rates as a crowding measure — OKX public API, no key.

    Not social sentiment, which is why it is a separate family. Funding is what
    leveraged traders are actually *paying* to hold a side: positive means longs
    pay shorts (crowded long), negative means the reverse. It is the only input
    here that goes genuinely negative on its own - measured over 100 periods,
    SOL was negative 27% of the time and ETH 25%, against social sources that
    are positive almost always.

    Read contrarian by default: crowded positioning is fragile positioning, so
    unusually high funding scores bearish. Set `contrarian=False` to read it as
    momentum confirmation instead.

    Scored against each asset's OWN funding history rather than an absolute
    threshold, for the same reason mention velocity is: 0.01% means something
    different for BTC than for a thin altcoin.
    """
    enabled: bool = True
    venue_url: str = "https://www.okx.com/api/v5/public/funding-rate-history"
    inst_template: str = "{symbol}-USDT-SWAP"
    history_periods: int = 100      # ~33 days at 8h funding intervals
    request_delay: float = 0.35
    contrarian: bool = True
    z_scale: float = 1.5            # tanh knee, in standard deviations
    min_periods: int = 20           # below this the z-score is noise; score 0


@dataclass
class CatalystConfig:
    news_feeds: tuple[tuple[str, str], ...] = (
        ("CoinDesk", "https://feeds.feedburner.com/CoinDesk"),
        ("Cointelegraph", "https://cointelegraph.com/rss"),
        ("Decrypt", "https://decrypt.co/feed"),
        ("TheDefiant", "https://thedefiant.io/api/feed"),
    )
    github_lookback_days: int = 14
    news_lookback_days: int = 5
    max_items_per_feed: int = 120


@dataclass
class ExecutionConfig:
    """Binance execution settings.

    venue:
      "binance-us"   -> https://api.binance.us      (works from US IPs)
      "binance-com"  -> https://api.binance.com     (blocked from US IPs, HTTP 451)
      "binance-test" -> https://testnet.binance.vision (Spot testnet, also geo-gated)

    Safety ladder, from safest to riskiest:
      1. mode="paper"                       -> nothing touches Binance's order books
      2. mode="binance" + dry_run=True      -> POST /api/v3/order/test (validates, no fill)
      3. mode="binance" + dry_run=False     -> POST /api/v3/order (REAL MONEY)

    Step 3 additionally requires env CRYPTO_YOLO_ALLOW_LIVE=1. Both switches must
    agree, so a stray config edit alone can never place a real order.
    """
    mode: str = "paper"                 # "paper" | "binance"
    venue: str = "binance-us"
    dry_run: bool = True
    quote_asset: str = "USDT"
    order_type: str = "MARKET"          # "MARKET" | "LIMIT" — used for EXITS (must fill)
    limit_offset_bps: float = 5.0       # for LIMIT: how far through the mid to place
    recv_window_ms: int = 5_000

    # ---- Entry execution: bound the slippage a market order can't ----------
    # A plain MARKET entry on a thin alt pays the full spread plus impact, and
    # you find out the cost only after the fill. A *marketable* limit crosses
    # the book (so it still fills promptly, usually as taker) but never worse
    # than `entry_limit_cross_bps` through the reference price — the slippage is
    # capped instead of open-ended. Set `entry_order_type="MARKET"` to restore
    # the old behaviour. Exits always use `order_type` above.
    entry_order_type: str = "LIMIT"          # "LIMIT" (marketable) | "MARKET"
    entry_limit_cross_bps: float = 15.0      # max adverse cross for an entry limit
    # Refuse an entry whose price has drifted this far above the proposal price
    # since it was sized. A guard against sending size into a market that has
    # gapped away. 0 disables the check. Read it through
    # `entry_slippage_guard_bps`, not directly — see that property.
    max_entry_slippage_bps: float = 60.0

    # ---- Protective orders resting on the exchange after an entry fills ----
    # These are what give protection *between* notebook runs. The exits pass
    # only sees the market when you run it; an order resting at Binance is
    # watched by Binance continuously.
    #
    # Binance.US supports LIMIT, LIMIT_MAKER, MARKET, STOP_LOSS_LIMIT and
    # TAKE_PROFIT_LIMIT. It does NOT support market STOP_LOSS, so a protective
    # stop is always a stop-LIMIT: `stop_limit_offset_bps` sets how far through
    # the trigger the limit sits, since a limit exactly at the stop may not fill
    # in a fast move.
    place_stop_orders: bool = False        # rest a protective stop after entry
    place_limit_orders: bool = False       # rest a take-profit limit at the target
    place_stop_limit_orders: bool = True   # stop leg uses STOP_LOSS_LIMIT (required here)
    stop_limit_offset_bps: float = 25.0

    # When both a stop and a target are wanted, send them as one OCO so that
    # filling one cancels the other. Two independent resting sells for the same
    # quantity is a double-sell hazard, not protection.
    use_oco: bool = True

    # Binance can trail the stop itself via trailingDelta (10-2000 bps).
    # 0 = derive from CONFIG.exits.trail_pct.
    use_trailing_delta: bool = False
    trailing_delta_bps: int = 0

    # Binance caps resting algo (stop/OCO) orders per symbol; see the
    # MAX_NUM_ALGO_ORDERS filter, currently 5 on BTCUSDT.
    max_algo_orders_per_symbol: int = 5

    BASE_URLS = {
        "binance-us": "https://api.binance.us",
        "binance-com": "https://api.binance.com",
        "binance-test": "https://testnet.binance.vision",
    }

    @property
    def base_url(self) -> str:
        return self.BASE_URLS[self.venue]

    @property
    def entry_slippage_guard_bps(self) -> float:
        """The drift that actually refuses an entry.

        `max_entry_slippage_bps` on its own can be set below
        `entry_limit_cross_bps` by accident, which is incoherent: you'd reject
        an entry for drifting less than the limit is already willing to cross.
        This keeps the guard at least a margin above the cross for a LIMIT
        entry. 0 (disabled) is honoured as-is.
        """
        raw = self.max_entry_slippage_bps
        if raw <= 0:
            return 0.0
        if self.entry_order_type.upper() == "LIMIT":
            return max(raw, self.entry_limit_cross_bps + 20.0)
        return raw

    # Approve every proposal without prompting. Off by default - the human in
    # the loop is the thing standing between the scoring heuristics and your
    # money, so removing it has to be a deliberate act.
    auto_approve: bool = False

    @property
    def live_enabled(self) -> bool:
        """Real orders require BOTH the config switch and the env var."""
        return (
            self.mode == "binance"
            and not self.dry_run
            and _env_flag("CRYPTO_YOLO_ALLOW_LIVE", False)
        )

    @property
    def auto_live_enabled(self) -> bool:
        """Unattended REAL-money trading: auto-approve combined with live.

        This is the highest-risk configuration the system can be in - an
        untested scoring heuristic spending real money with nobody watching -
        so it takes a third switch of its own rather than falling out of two
        settings that were each reasonable alone.
        """
        return self.auto_approve and self.live_enabled and _env_flag(
            "CRYPTO_YOLO_ALLOW_AUTO_LIVE", False)


@dataclass
class NotifyConfig:
    """Out-of-band alerts — see `cryptoyolo/notify.py`. Every channel optional;
    with none set, alerts degrade to a log line.

    Routine fills are OFF by default: a daily auto-approve paper job would
    otherwise ping several times a day. The events left on are the ones you
    actually want to interrupt you.
    """
    enabled: bool = True
    macos_banner: bool = True
    ntfy_url: str = ""       # e.g. https://ntfy.sh/your-private-topic  (or env CRYPTO_YOLO_NTFY_URL)
    webhook_url: str = ""    # Slack/Discord incoming webhook           (or env CRYPTO_YOLO_ALERT_WEBHOOK)

    on_execution: bool = False          # every fill
    on_rejection: bool = True           # an order the engine sent was rejected
    on_exit_trigger: bool = True        # a stop / target / trailing / horizon exit filled
    on_regime_risk_off: bool = True     # the BTC-trend gate flipped to risk_off
    on_price_fallback: bool = True      # scoring fell through to the yfinance backstop
    on_drawdown: bool = True
    drawdown_alert_pct: float = 10.0    # alert when equity FIRST falls this far below its own peak


@dataclass
class Config:
    universe: tuple[Asset, ...] = UNIVERSE
    bands: BandConfig = field(default_factory=BandConfig)
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    risk: RiskConfig = field(default_factory=RiskConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    reddit: RedditConfig = field(default_factory=RedditConfig)
    stocktwits: StockTwitsConfig = field(default_factory=StockTwitsConfig)
    mastodon: MastodonConfig = field(default_factory=MastodonConfig)
    biz: BizConfig = field(default_factory=BizConfig)
    positioning: PositioningConfig = field(default_factory=PositioningConfig)
    catalysts: CatalystConfig = field(default_factory=CatalystConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)
    kalshi_prediction: KalshiPredictionConfig = field(default_factory=KalshiPredictionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)

    db_path: Path = DATA_DIR / "crypto_yolo.sqlite"
    http_timeout: int = 20
    http_retries: int = 3
    user_agent: str = "crypto-yolo-dashboard/1.0 (personal research)"
    price_source_order: tuple[str, ...] = ("binance", "coinbase", "kraken", "yfinance")
    cache_ttl_seconds: int = 120

    @property
    def symbols(self) -> list[str]:
        return [a.symbol for a in self.universe]

    def with_(self, **kwargs) -> "Config":
        return replace(self, **kwargs)


CONFIG = Config()

# Credentials, read lazily so the notebook can call load_dotenv() first.
ENV_KEYS = {
    "binance_key": "BINANCE_API_KEY",
    "binance_secret": "BINANCE_API_SECRET",
    "reddit_client_id": "REDDIT_CLIENT_ID",
    "reddit_client_secret": "REDDIT_CLIENT_SECRET",
    "reddit_user_agent": "REDDIT_USER_AGENT",
    "github_token": "GITHUB_TOKEN",
    "kalshi_key_id": "KALSHI_API_KEY_ID",
    # Prefer a file path over inline PEM — private key material with embedded
    # newlines is awkward (and easy to mis-copy) as a single .env line.
    "kalshi_private_key_path": "KALSHI_PRIVATE_KEY_PATH",
    "kalshi_private_key": "KALSHI_PRIVATE_KEY",
}


def get_secret(name: str) -> str | None:
    """Fetch a credential by logical name. Returns None if unset."""
    return os.environ.get(ENV_KEYS.get(name, name)) or None


def _kalshi_configured() -> bool:
    """True only if the key ID is set AND a private key actually loads — a
    set-but-unreadable key must not show as configured."""
    if not get_secret("kalshi_key_id"):
        return False
    from .macro import KalshiClient     # lazy: macro imports this module
    try:
        return KalshiClient().configured
    except Exception:  # noqa: BLE001 - a status check must never raise
        return False


def credential_status() -> dict[str, bool]:
    """Which integrations are configured. Never returns the values themselves."""
    return {
        "binance": bool(get_secret("binance_key") and get_secret("binance_secret")),
        "reddit_oauth": bool(get_secret("reddit_client_id") and get_secret("reddit_client_secret")),
        "github": bool(get_secret("github_token")),
        "kalshi": _kalshi_configured(),
        "live_trading_env": _env_flag("CRYPTO_YOLO_ALLOW_LIVE", False),
        "auto_live_env": _env_flag("CRYPTO_YOLO_ALLOW_AUTO_LIVE", False),
    }
