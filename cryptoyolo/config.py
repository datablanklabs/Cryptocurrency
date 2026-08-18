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
    """How the three feature families combine into one composite score.

    These are hand-set priors, not fitted parameters. Retune them from the
    backtest-ish attribution table in the notebook rather than trusting them.
    """
    technical: float = 0.50
    social: float = 0.20
    catalyst: float = 0.30

    def normalized(self) -> "ScoreWeights":
        total = self.technical + self.social + self.catalyst
        if total <= 0:
            return ScoreWeights(1 / 3, 1 / 3, 1 / 3)
        return ScoreWeights(self.technical / total, self.social / total, self.catalyst / total)


@dataclass
class RiskConfig:
    """Position sizing and guardrails."""
    account_equity_usd: float = 10_000.0
    risk_per_trade_pct: float = 1.0          # % of equity risked between entry and stop
    max_position_pct: float = 20.0           # cap on any single position as % of equity
    max_total_deployed_pct: float = 60.0     # cap across all proposed trades
    atr_stop_mult: float = 1.5               # stop distance = mult * ATR(14) on the 1h chart
    reward_risk_target: float = 2.0          # take-profit distance = R:R * stop distance
    min_notional_usd: float = 15.0           # skip dust trades (Binance minNotional is ~$10)
    max_proposals: int = 3
    # Minimum |composite| to be proposable. Raise it to be pickier; set it to
    # 0.0 to always surface a full slate of `max_proposals`. It is deliberately
    # not 0 by default: on a genuinely directionless day the honest output is
    # one candidate, or none, rather than three manufactured ones.
    min_composite_score: float = 0.05
    # Exits are proposed and executed before entries, so their proceeds are
    # spendable by the buys in the same run. That projection is discounted by
    # this much to absorb slippage and fees - the sale rarely nets exactly the
    # quoted notional, and sizing buys off an optimistic figure is how you end
    # up with a rejected final order.
    exit_proceeds_haircut_pct: float = 1.0


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

    # 3. Horizon expiry: the 1-7 day thesis has run out of time. A trade held
    #    past its horizon is no longer the trade that was approved.
    horizon_expiry: bool = True
    horizon_days: float = 7.0

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
    subreddits: tuple[str, ...] = ("wallstreetbets", "cryptocurrency")
    listings: tuple[str, ...] = ("new", "hot")
    posts_per_listing: int = 100
    include_comments: bool = True
    comments_per_post: int = 40
    max_posts_for_comments: int = 15
    baseline_days: int = 7        # window used to compute the "normal" mention rate
    velocity_window_hours: int = 24


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
    order_type: str = "MARKET"          # "MARKET" | "LIMIT"
    limit_offset_bps: float = 5.0       # for LIMIT: how far through the mid to place
    recv_window_ms: int = 5_000

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
    def live_enabled(self) -> bool:
        """Real orders require BOTH the config switch and the env var."""
        return (
            self.mode == "binance"
            and not self.dry_run
            and _env_flag("CRYPTO_YOLO_ALLOW_LIVE", False)
        )


@dataclass
class Config:
    universe: tuple[Asset, ...] = UNIVERSE
    bands: BandConfig = field(default_factory=BandConfig)
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    risk: RiskConfig = field(default_factory=RiskConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    reddit: RedditConfig = field(default_factory=RedditConfig)
    catalysts: CatalystConfig = field(default_factory=CatalystConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

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
}


def get_secret(name: str) -> str | None:
    """Fetch a credential by logical name. Returns None if unset."""
    return os.environ.get(ENV_KEYS.get(name, name)) or None


def credential_status() -> dict[str, bool]:
    """Which integrations are configured. Never returns the values themselves."""
    return {
        "binance": bool(get_secret("binance_key") and get_secret("binance_secret")),
        "reddit_oauth": bool(get_secret("reddit_client_id") and get_secret("reddit_client_secret")),
        "github": bool(get_secret("github_token")),
        "live_trading_env": _env_flag("CRYPTO_YOLO_ALLOW_LIVE", False),
    }
