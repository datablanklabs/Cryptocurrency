"""Feature 2: Reddit scraping, mention extraction, and sentiment.

Three parts:

  RedditClient    Pulls posts and comments from the configured subreddits
                  (see RedditConfig) through a ladder of sources: the official
                  OAuth API first, then two keyless fallbacks (Arctic Shift and
                  Reddit's own Atom feeds). Reddit 403s anonymous .json now, so
                  without the fallbacks this feature would need credentials to
                  return anything at all.

  extract         Finds which assets a piece of text is about. The hard part is
                  precision, not recall: DOT, OP, NEAR, LINK, ATOM and friends
                  are ordinary English words, and counting those naively gives
                  you a sentiment signal made almost entirely of noise. Assets
                  flagged `ambiguous` in config only match as $DOT or by full
                  name ("polkadot"), never as the bare word.

  score_symbols   Turns mentions into a per-asset signal. The headline number is
                  *velocity* - how far current chatter deviates from that
                  asset's own baseline - not raw volume, because raw volume just
                  ranks BTC and ETH first every single time.

A caveat worth keeping in mind while reading the output: subreddit mention
volume is trivially manipulated, and by the time something trends the move that
caused the trend has usually already happened. This is why social carries a low
weight, and why `positioning` exists as a separate family - it is the only input
that goes reliably negative.

Sources are mixed (Reddit, StockTwits, Mastodon, /biz/) and each has its own tone
baseline, so sentiment is de-biased per source before assets are compared. One
consequence worth understanding: because each source is centred on its own mean,
a platform being more bearish OVERALL is centred out. What a source like /biz/
contributes is cross-asset dispersion and coverage, not a downward pull on the
level.
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd
import requests

from .config import CONFIG, Config, get_secret
from .store import Store, iso, utcnow

# --------------------------------------------------------------------------
# Sentiment lexicon (crypto-native, deliberately not generic English)
# --------------------------------------------------------------------------
BULLISH = {
    "moon": 2.0, "mooning": 2.2, "moonshot": 1.8, "bullish": 2.0, "bull": 1.4,
    "pump": 1.6, "pumping": 1.8, "pumped": 1.2, "long": 1.0, "longing": 1.2,
    "buy": 1.2, "buying": 1.3, "bought": 0.9, "accumulate": 1.4,
    "accumulating": 1.4, "hodl": 1.5, "hodling": 1.5, "breakout": 1.8,
    "rally": 1.7, "rallying": 1.8, "surge": 1.8, "surging": 1.9, "ath": 1.6,
    "gains": 1.4, "green": 1.0, "ripping": 1.7, "lambo": 1.5, "printing": 1.4,
    "undervalued": 1.6, "oversold": 1.2, "bottomed": 1.3, "reversal": 0.9,
    "adoption": 1.3, "partnership": 1.4, "upgrade": 1.2, "listing": 1.5,
    "institutional": 1.1, "squeeze": 1.3, "lfg": 1.8, "wagmi": 1.7,
    "gem": 1.5, "parabolic": 2.0, "exploding": 1.7, "sending": 1.4,
    "10x": 1.8, "100x": 2.0, "diamond": 1.2, "based": 0.8, "alpha": 0.9,
    "golden cross": 1.9, "cup and handle": 1.3, "higher low": 1.2,
    # Headline verbs — these carry the direction in news copy, where nobody
    # writes "moon" but everyone writes "jumps" / "soars".
    "jumps": 1.3, "soars": 1.7, "climbs": 1.1, "spikes": 1.3, "gains": 1.2,
    "inflows": 1.2, "record high": 1.6, "outperform": 1.2, "recovers": 1.0,
}
BEARISH = {
    "dump": -1.7, "dumping": -1.9, "dumped": -1.4, "bearish": -2.0, "bear": -1.3,
    "crash": -2.1, "crashing": -2.2, "rug": -2.4, "rugpull": -2.6,
    "rugged": -2.4, "scam": -2.3, "ponzi": -2.4, "short": -1.2,
    "shorting": -1.4, "sell": -1.2, "selling": -1.3, "sold": -1.0,
    "dead": -1.8, "dying": -1.7, "rekt": -2.0, "liquidated": -2.0,
    "liquidation": -1.6, "bleeding": -1.7, "red": -1.0, "tanking": -1.9,
    "collapse": -2.1, "collapsing": -2.2, "capitulation": -1.8,
    "overvalued": -1.5, "overbought": -1.1, "topped": -1.3,
    "death cross": -1.9, "hack": -2.2, "hacked": -2.4, "exploit": -2.1,
    "exploited": -2.3, "delisted": -2.3, "lawsuit": -1.8, "fud": -0.9,
    "bagholder": -1.6, "bags": -1.0, "ngmi": -1.7, "worthless": -2.2,
    "avoid": -1.4, "plummet": -2.0, "plunging": -2.0, "lower high": -1.2,
    "paper hands": -1.0, "exit liquidity": -2.0,
    # Headline verbs (bearish side).
    "drops": -1.3, "falls": -1.3, "slips": -1.1, "slides": -1.2,
    "tumbles": -1.6, "sinks": -1.5, "retreats": -1.0, "outflows": -1.3,
    "plunges": -1.8, "sell-off": -1.7, "selloff": -1.7, "headwinds": -1.1,
    "weigh on": -1.2, "struggles": -1.2, "stalls": -1.1, "falling": -1.2,
    "fell": -1.2, "declining": -1.3, "decline": -1.2, "fails": -1.2,
}
EMOJI = {"🚀": 2.0, "🌙": 1.6, "📈": 1.3, "💎": 1.3, "🙌": 0.8, "🐂": 1.4,
         "🤑": 1.2, "💰": 0.9, "📉": -1.4, "🐻": -1.5, "💀": -1.6,
         "🩸": -1.5, "🤡": -1.4, "⚰️": -1.7}
NEGATORS = {"not", "no", "never", "isnt", "isn't", "wasnt", "wasn't", "dont",
            "don't", "doesnt", "doesn't", "didnt", "didn't", "wont", "won't",
            "cant", "can't", "cannot", "aint", "ain't", "without", "hardly",
            "barely", "stop", "stopped", "avoid"}
INTENSIFIERS = {"very": 1.4, "super": 1.4, "extremely": 1.6, "massively": 1.6,
                "huge": 1.4, "insane": 1.5, "absolutely": 1.4, "so": 1.15,
                "really": 1.2, "fucking": 1.4, "literally": 1.1}
LEXICON = {**BULLISH, **BEARISH}
MULTIWORD = [p for p in LEXICON if " " in p]

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_URL_RE = re.compile(r"https?://\S+")
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(text: str) -> str:
    """Plain text out of the HTML fragment Reddit's Atom feed puts in <content>."""
    import html as _html
    return " ".join(_html.unescape(_TAG_RE.sub(" ", text or "")).split())


def sentiment(text: str) -> tuple[float, float]:
    """Return (polarity in [-1, 1], confidence in [0, 1]).

    Confidence rises with the number of sentiment-bearing terms found, so a post
    that merely names a coin scores ~0 confidence and contributes almost nothing
    downstream. This matters more than the polarity itself - most Reddit posts
    carry no directional opinion at all.
    """
    if not text:
        return 0.0, 0.0
    low = _URL_RE.sub(" ", text.lower())

    total, hits = 0.0, 0
    for phrase in MULTIWORD:
        n = low.count(phrase)
        if n:
            total += LEXICON[phrase] * n
            hits += n
            low = low.replace(phrase, " ")

    for ch, val in EMOJI.items():
        n = text.count(ch)
        if n:
            total += val * min(n, 4)      # cap emoji spam
            hits += min(n, 4)

    tokens = _TOKEN_RE.findall(low)
    for i, tok in enumerate(tokens):
        val = LEXICON.get(tok)
        if val is None:
            continue
        window = tokens[max(0, i - 3):i]
        if any(w in NEGATORS for w in window):
            val = -val * 0.75             # negation flips, slightly damped
        for w in window:
            if w in INTENSIFIERS:
                val *= INTENSIFIERS[w]
                break
        total += val
        hits += 1

    if hits == 0:
        return 0.0, 0.0
    polarity = math.tanh(total / (2.5 * math.sqrt(hits)))
    confidence = min(1.0, hits / 6.0)
    return round(polarity, 4), round(confidence, 4)


# --------------------------------------------------------------------------
# Mention extraction
# --------------------------------------------------------------------------
def _build_patterns(cfg: Config) -> dict[str, list[tuple[str, re.Pattern]]]:
    pats: dict[str, list[tuple[str, re.Pattern]]] = {}
    for asset in cfg.universe:
        rules = [("cashtag", re.compile(rf"\${re.escape(asset.symbol)}\b", re.I))]
        for alias in (asset.name.lower(), *[a.lower() for a in asset.aliases]):
            rules.append(("name", re.compile(rf"\b{re.escape(alias)}\b", re.I)))
        if not asset.ambiguous:
            # Case-SENSITIVE: "BTC" is a ticker, "btc" in prose usually is too,
            # but "Sol"/"Near"/"Link" as ordinary words are not.
            rules.append(("ticker", re.compile(rf"\b{re.escape(asset.symbol)}\b")))
        pats[asset.symbol] = rules
    return pats


def extract_symbols(text: str, patterns: dict) -> dict[str, str]:
    """Map symbol -> which rule matched ('cashtag' | 'name' | 'ticker')."""
    found: dict[str, str] = {}
    for symbol, rules in patterns.items():
        for kind, pat in rules:
            if pat.search(text):
                # cashtag is the strongest evidence; keep the best rule seen
                if kind == "cashtag" or symbol not in found:
                    found[symbol] = kind
                if kind == "cashtag":
                    break
    return found


# --------------------------------------------------------------------------
# Reddit client
# --------------------------------------------------------------------------
class RedditClient:
    """Fetches posts and comments, degrading through a ladder of sources.

    Reddit's anonymous .json endpoints return 403 from most hosts now, so
    without OAuth credentials the official API is simply closed. Two public
    alternatives still work and are tried in order:

      oauth        Official API via a free "script" app. Best: full listings
                   ('hot' vs 'new' are actually different), pagination, real
                   scores. Needs REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET.

      arctic       Arctic Shift (arctic-shift.photon-reddit.com), a public
                   Pushshift successor. No auth, 100 items/request, and it
                   carries `score` - the only keyless source that does.
                   Chronological only, so 'hot' and 'new' return the same set.

      rss          Reddit's own Atom feeds. Still served anonymously where
                   .json is blocked. Carries id/author/timestamp/full body but
                   *no score*, and caps around 25 items with aggressive rate
                   limiting.

      public_json  The legacy anonymous .json path. Kept last because it works
                   from a few hosts, but expect 403.

    PullPush.io, the other well-known Pushshift successor, sits behind a
    Cloudflare challenge and is not usable from a script - it is deliberately
    not in the ladder.
    """

    OAUTH = "https://oauth.reddit.com"
    PUBLIC = "https://www.reddit.com"
    ARCTIC = "https://arctic-shift.photon-reddit.com/api"

    def __init__(self, cfg: Config = CONFIG, sources: tuple[str, ...] | None = None):
        self.cfg = cfg
        self.session = requests.Session()
        self.ua = get_secret("reddit_user_agent") or cfg.user_agent
        self.session.headers.update({"User-Agent": self.ua})
        self._token: str | None = None
        self._token_expiry: float = 0.0
        self.mode = "none"
        self.sources = sources or cfg.reddit.sources
        self.degraded: list[str] = []

    # -- auth ---------------------------------------------------------------
    def authenticate(self) -> bool:
        cid, secret = get_secret("reddit_client_id"), get_secret("reddit_client_secret")
        if not (cid and secret):
            return False
        if self._token and time.time() < self._token_expiry - 60:
            return True
        try:
            r = self.session.post(
                f"{self.PUBLIC}/api/v1/access_token",
                auth=(cid, secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": self.ua},
                timeout=self.cfg.http_timeout,
            )
            r.raise_for_status()
            payload = r.json()
            self._token = payload["access_token"]
            self._token_expiry = time.time() + float(payload.get("expires_in", 3600))
            return True
        except Exception as exc:  # noqa: BLE001 - fall down the source ladder
            print(f"  ! Reddit OAuth failed ({exc}); trying keyless sources")
            return False

    # Arctic Shift signals rate limiting with HTTP 422 and the body
    # {"error": "Timeout. Maybe slow down a bit"} - NOT 429. Treating that as a
    # generic error (short backoff, 3 tries) makes it look like the subreddit is
    # unavailable, which is exactly what happens once the universe grows past a
    # couple of subs.
    RATE_LIMIT_CODES = (422, 429, 503)

    def _get(self, url: str, params: dict, headers: dict | None = None,
             tries: int | None = None) -> Any:
        tries = tries or max(self.cfg.http_retries, 5)
        for attempt in range(tries):
            try:
                r = self.session.get(url, params=params,
                                     headers=headers or {"User-Agent": self.ua},
                                     timeout=self.cfg.http_timeout)
                if r.status_code in self.RATE_LIMIT_CODES:
                    # Linear-ish backoff; the limiter clears in a few seconds.
                    time.sleep(3.0 + 2.5 * attempt)
                    continue
                if r.status_code == 403:
                    raise PermissionError("403")
                r.raise_for_status()
                return r.json()
            except PermissionError:
                raise
            except Exception:  # noqa: BLE001 - retry, then give up on this source
                time.sleep(0.8 * (attempt + 1))
        return None

    # -- normalizers --------------------------------------------------------
    @staticmethod
    def _row(post_id, subreddit, kind, author, created, score, ncomments,
             title, body, permalink) -> dict[str, Any]:
        return {
            "post_id": post_id, "subreddit": subreddit, "source": "reddit",
            "kind": kind, "author": author, "created_utc": float(created or 0),
            "score": int(score or 0), "num_comments": int(ncomments or 0),
            "title": title or "", "body": body or "",
            "permalink": permalink or "", "fetched_at": iso(),
        }

    # -- source: official API ----------------------------------------------
    def _oauth(self, subreddit: str, path: str, limit: int, kind: str) -> list[dict]:
        if not self.authenticate():
            return []
        rows, after, fetched = [], None, 0
        while fetched < limit:
            params = {"limit": min(100, limit - fetched), "raw_json": 1}
            if after:
                params["after"] = after
            payload = self._get(f"{self.OAUTH}/r/{subreddit}/{path}", params,
                                headers={"User-Agent": self.ua,
                                         "Authorization": f"Bearer {self._token}"})
            if not payload or "data" not in payload:
                break
            children = payload["data"].get("children", [])
            if not children:
                break
            for ch in children:
                d = ch.get("data", {})
                if kind == "post":
                    rows.append(self._row(f"t3_{d.get('id')}", subreddit, "post",
                                          d.get("author"), d.get("created_utc"),
                                          d.get("score"), d.get("num_comments"),
                                          d.get("title"), d.get("selftext"),
                                          f"https://reddit.com{d.get('permalink','')}"))
                else:
                    rows.append(self._row(f"t1_{d.get('id')}", subreddit, "comment",
                                          d.get("author"), d.get("created_utc"),
                                          d.get("score"), 0, "", d.get("body"),
                                          f"https://reddit.com{d.get('permalink','')}"))
            fetched += len(children)
            after = payload["data"].get("after")
            if not after:
                break
            time.sleep(0.6)
        return rows

    # -- source: Arctic Shift ----------------------------------------------
    def _arctic(self, subreddit: str, limit: int, kind: str) -> list[dict]:
        endpoint = "posts" if kind == "post" else "comments"
        payload = self._get(f"{self.ARCTIC}/{endpoint}/search",
                            {"subreddit": subreddit, "limit": min(100, limit),
                             "sort": "desc"})
        if not payload or "data" not in payload:
            return []
        rows = []
        for d in payload["data"]:
            pid = d.get("id", "")
            if kind == "post":
                rows.append(self._row(f"t3_{pid}", subreddit, "post", d.get("author"),
                                      d.get("created_utc"), d.get("score"),
                                      d.get("num_comments"), d.get("title"),
                                      d.get("selftext"),
                                      f"https://reddit.com{d.get('permalink','')}"))
            else:
                rows.append(self._row(f"t1_{pid}", subreddit, "comment", d.get("author"),
                                      d.get("created_utc"), d.get("score"), 0, "",
                                      d.get("body"),
                                      f"https://reddit.com{d.get('permalink','')}"))
        return rows

    # -- source: Reddit Atom feeds -----------------------------------------
    def _rss(self, subreddit: str, limit: int, kind: str) -> list[dict]:
        import xml.etree.ElementTree as ET
        from datetime import datetime as _dt

        path = "comments" if kind == "comment" else "new"
        url = f"{self.PUBLIC}/r/{subreddit}/{path}/.rss"
        for attempt in range(4):
            try:
                r = self.session.get(url, params={"limit": min(100, limit)},
                                     headers={"User-Agent": self.ua},
                                     timeout=self.cfg.http_timeout)
                if r.status_code in (429, 503):
                    time.sleep(4 * (attempt + 1))
                    continue
                r.raise_for_status()
                text = r.text
                break
            except Exception:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
        else:
            return []

        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return []

        ns = {"a": "http://www.w3.org/2005/Atom"}
        rows = []
        for e in root.findall(".//a:entry", ns):
            raw_id = e.findtext("a:id", "", ns) or ""
            author_el = e.find("a:author/a:name", ns)
            author = (author_el.text or "").lstrip("/u/") if author_el is not None else None
            updated = e.findtext("a:updated", "", ns)
            try:
                created = _dt.fromisoformat(updated.replace("Z", "+00:00")).timestamp()
            except Exception:  # noqa: BLE001 - undated entry
                created = 0.0
            content = e.findtext("a:content", "", ns) or ""
            body = _strip_tags(content)
            title = e.findtext("a:title", "", ns) or ""
            link_el = e.find("a:link", ns)
            link = link_el.get("href") if link_el is not None else ""
            is_comment = raw_id.startswith("t1_")
            rows.append(self._row(
                raw_id or link, subreddit,
                "comment" if is_comment else "post", author, created,
                0,          # RSS carries no score - engagement weighting degrades
                0,
                "" if is_comment else title,
                body, link,
            ))
        return rows

    # -- source: legacy anonymous JSON --------------------------------------
    def _public_json(self, subreddit: str, path: str, limit: int, kind: str) -> list[dict]:
        payload = self._get(f"{self.PUBLIC}/r/{subreddit}/{path}.json",
                            {"limit": min(100, limit), "raw_json": 1}, tries=1)
        if not payload or "data" not in payload:
            return []
        rows = []
        for ch in payload["data"].get("children", []):
            d = ch.get("data", {})
            if kind == "post":
                rows.append(self._row(f"t3_{d.get('id')}", subreddit, "post",
                                      d.get("author"), d.get("created_utc"),
                                      d.get("score"), d.get("num_comments"),
                                      d.get("title"), d.get("selftext"),
                                      f"https://reddit.com{d.get('permalink','')}"))
            else:
                rows.append(self._row(f"t1_{d.get('id')}", subreddit, "comment",
                                      d.get("author"), d.get("created_utc"),
                                      d.get("score"), 0, "", d.get("body"),
                                      f"https://reddit.com{d.get('permalink','')}"))
        return rows

    # -- public API ---------------------------------------------------------
    def _try_sources(self, subreddit: str, listing: str, limit: int,
                     kind: str) -> list[dict]:
        for source in self.sources:
            try:
                if source == "oauth":
                    rows = self._oauth(subreddit, listing if kind == "post" else "comments",
                                       limit, kind)
                elif source == "arctic":
                    rows = self._arctic(subreddit, limit, kind)
                elif source == "rss":
                    rows = self._rss(subreddit, limit, kind)
                elif source == "public_json":
                    rows = self._public_json(subreddit,
                                             listing if kind == "post" else "comments",
                                             limit, kind)
                else:
                    continue
            except PermissionError:
                continue
            except Exception:  # noqa: BLE001 - try the next source
                continue
            if rows:
                self.mode = source
                if source in ("rss",) and "no-scores" not in self.degraded:
                    self.degraded.append("no-scores")
                return rows
        return []

    def listing(self, subreddit: str, listing: str = "new",
                limit: int = 100) -> list[dict[str, Any]]:
        return self._try_sources(subreddit, listing, limit, "post")

    def comments(self, subreddit: str, limit: int = 100) -> list[dict[str, Any]]:
        """Recent comments across the whole subreddit.

        Much cheaper than walking each post's comment tree, and comments are
        where the actual opinions live - post titles are mostly memes.
        """
        return self._try_sources(subreddit, "comments", limit, "comment")


# --------------------------------------------------------------------------
# Adaptive source ranking
# --------------------------------------------------------------------------
def source_stats(store: Store, cfg: Config = CONFIG,
                 days: float | None = None,
                 source: str | None = "reddit") -> pd.DataFrame:
    """Per-subreddit delivery stats, measured from what we already collected.

    Uses our own tables rather than re-probing the API: the data is free,
    already fetched, and reflects what each source actually gives *us* under
    *our* matcher - which is the thing we care about, not its general activity.
    """
    days = days or cfg.reddit.rank_window_days
    cutoff = (utcnow() - timedelta(days=days)).timestamp()
    # Scope to one platform. Without this the subreddit ranking is computed over
    # StockTwits cashtags and Mastodon hashtags too - they live in the same
    # tables now - which pollutes source_rank with ~26 rows that are not
    # subreddits and can never be fetched as one.
    where, params = "created_utc >= ?", [cutoff]
    if source:
        where += " AND source = ?"
        params.append(source)
    with store.conn() as con:
        posts = pd.read_sql_query(
            f"SELECT post_id, subreddit, created_utc FROM reddit_posts WHERE {where}",
            con, params=params)
        mentions = pd.read_sql_query(
            f"SELECT post_id, subreddit, symbol, confidence FROM mentions WHERE {where}",
            con, params=params)
    if posts.empty:
        return pd.DataFrame()

    # Reddit subreddit names are case-insensitive, but we store whatever string
    # was configured - so renaming "cryptocurrency" to "CryptoCurrency" split
    # that sub's history across two keys and halved its apparent yield. Group on
    # a folded key so history survives a change of spelling.
    posts["_key"] = posts["subreddit"].str.lower()
    if not mentions.empty:
        mentions["_key"] = mentions["subreddit"].str.lower()
    else:
        mentions["_key"] = pd.Series(dtype=str)

    rows = []
    for key, grp in posts.groupby("_key"):
        m = mentions[mentions["_key"] == key]
        sub = key
        n_items = len(grp)
        span_h = (grp["created_utc"].max() - grp["created_utc"].min()) / 3600.0
        rows.append({
            "subreddit": sub,          # folded key; matched case-insensitively
            "n_items": n_items,
            # Items per hour is the honest freshness measure. A sub whose 200
            # items span 2349h yields <1 new item per 8h cycle - fetching it
            # costs a request and returns almost nothing.
            "items_per_hour": n_items / span_h if span_h > 0.5 else float(n_items),
            "mention_rate": m["post_id"].nunique() / n_items if n_items else 0.0,
            "assets": int(m["symbol"].nunique()),
            "tone": float(m["confidence"].mean()) if not m.empty else 0.0,
        })
    return pd.DataFrame(rows)


def _splice_unranked(ranked: list[str], configured: list[str],
                     cfg: Config = CONFIG) -> list[str]:
    """Insert never-yet-collected subreddits just behind the proven leaders.

    Appending them last is self-defeating: under a rate limit the tail may
    never be reached, so a new source can never gather the stats it needs to
    earn a ranking - it stays last forever. Splicing after the top few gives it
    a real chance to prove itself without displacing known-good sources.
    """
    unranked = [s for s in configured if s not in set(ranked)]
    if not unranked:
        return ranked
    head = max(1, int(getattr(cfg.reddit, "probation_after_top", 3)))
    return ranked[:head] + unranked + ranked[head:]


def rank_subreddits(store: Store, cfg: Config = CONFIG,
                    force: bool = False) -> list[str]:
    """Order subreddits by expected useful mentions per fetch.

    Not an arbitrary weighted sum - an expected-value estimate:

        expected_new = min(page_size, items_per_hour * cadence_hours)
        usefulness   = mention_rate * breadth * (0.4 + 0.6 * tone)
        value        = expected_new * usefulness

    `expected_new` is what matters for ordering under a rate limit: a source is
    only worth an early slot if it has actually produced new material since the
    last pass. Breadth is log-scaled because a sub covering 10 assets is more
    useful than one covering 1, but not ten times more - and single-asset subs
    still earn a place for per-asset sentiment.

    Recomputed at most every `rank_refresh_hours`; falls back to the configured
    order (the hand-measured one) until enough history exists to beat it.
    """
    configured = list(cfg.reddit.subreddits)
    if not cfg.reddit.adaptive_ranking:
        return configured

    by_key = {s.lower(): s for s in configured}      # folded -> configured spelling

    age = store.source_rank_age_hours()
    if not force and age is not None and age < cfg.reddit.rank_refresh_hours:
        cached = store.source_rank()
        if not cached.empty:
            ranked = [by_key[k] for k in cached["subreddit"].str.lower() if k in by_key]
            return _splice_unranked(ranked, configured, cfg)

    stats = source_stats(store, cfg)
    if stats.empty or len(stats) < 2:
        return configured

    page = float(cfg.reddit.posts_per_listing)
    universe_n = max(len(cfg.symbols), 2)
    scored = []
    for _, r in stats.iterrows():
        expected_new = min(page, r["items_per_hour"] * cfg.reddit.collection_cadence_hours)
        breadth = math.log1p(r["assets"]) / math.log1p(universe_n)
        usefulness = r["mention_rate"] * breadth * (0.4 + 0.6 * min(r["tone"], 1.0))
        scored.append({**r.to_dict(), "value": expected_new * usefulness})

    scored.sort(key=lambda d: -d["value"])
    store.save_source_rank([
        {"subreddit": d["subreddit"], "computed_at": iso(), "rank": i,
         "value": round(d["value"], 5), "mention_rate": round(d["mention_rate"], 5),
         "assets": int(d["assets"]), "tone": round(d["tone"], 5),
         "items_per_hour": round(d["items_per_hour"], 4), "n_items": int(d["n_items"])}
        for i, d in enumerate(scored)
    ])
    ranked = [by_key[d["subreddit"].lower()] for d in scored
              if d["subreddit"].lower() in by_key]
    return _splice_unranked(ranked, configured, cfg)


# --------------------------------------------------------------------------
# Scrape pipeline
# --------------------------------------------------------------------------
def scrape(store: Store, cfg: Config = CONFIG, verbose: bool = True) -> dict[str, int]:
    """One scrape pass. Persists posts + mentions and returns counters.

    Safe to run repeatedly - posts are upserted by id, so re-running only adds
    what's new and refreshes scores on what it already had.
    """
    client = RedditClient(cfg)
    patterns = _build_patterns(cfg)
    rc = cfg.reddit
    all_rows: list[dict] = []

    order = rank_subreddits(store, cfg)
    if verbose and order != list(rc.subreddits):
        print(f"  fetch order (adaptive): {', '.join(order[:5])} ...")
    for sub_idx, sub in enumerate(order):
        if sub_idx:
            time.sleep(getattr(rc, "inter_subreddit_delay", 2.5))
        for listing in rc.listings:
            try:
                rows = client.listing(sub, listing, rc.posts_per_listing)
                all_rows.extend(rows)
                if verbose:
                    print(f"  r/{sub}/{listing:<4} {len(rows):>4} posts   [{client.mode}]")
                # Keyless sources are chronological only - they have no notion of
                # 'hot' vs 'new' and return the identical rows, so asking twice
                # just burns requests against a shared public service.
                if client.mode in ("arctic", "rss") and len(rc.listings) > 1:
                    if verbose:
                        print(f"       (source is chronological; skipping "
                              f"{', '.join(rc.listings[1:])})")
                    break
            except PermissionError as exc:
                if verbose:
                    print(f"  r/{sub}/{listing}: {exc}")
                return {"posts": 0, "mentions": 0, "blocked": 1}
            except Exception as exc:  # noqa: BLE001 - keep going with other listings
                if verbose:
                    print(f"  r/{sub}/{listing}: failed ({exc})")

        if rc.include_comments:
            try:
                rows = client.comments(sub, rc.comments_per_post * 2)
                all_rows.extend(rows)
                if verbose:
                    print(f"  r/{sub}/comments {len(rows):>4} comments")
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  r/{sub}/comments: failed ({exc})")

    if not all_rows:
        if verbose:
            print("  ! No source returned data. With no Reddit credentials this "
                  "usually means the keyless fallbacks are down or rate-limited; "
                  "set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET for a reliable feed.")
        return {"posts": 0, "mentions": 0, "blocked": 1, "source": client.mode}

    if verbose and "no-scores" in client.degraded:
        print("  ~ Source carries no vote scores; engagement weighting is flat. "
              "Mention counts and sentiment are unaffected.")

    store.upsert_posts(all_rows)

    mentions: list[dict] = []
    for row in all_rows:
        text = f"{row['title']}\n{row['body']}".strip()
        if not text:
            continue
        hits = extract_symbols(text, patterns)
        if not hits:
            continue
        pol, conf = sentiment(text)
        # Engagement weight: a +400 post is worth more than a +1 post, but
        # sub-linearly - log keeps one viral thread from owning the signal.
        engagement = math.log1p(max(0, row["score"])) + 0.5 * math.log1p(max(0, row["num_comments"]))
        kind_w = 1.0 if row["kind"] == "post" else 0.6
        for symbol, matched_on in hits.items():
            # A $-cashtag is a deliberate reference; a name match is weaker.
            rule_w = {"cashtag": 1.0, "ticker": 0.85, "name": 0.8}[matched_on]
            mentions.append({
                "post_id": row["post_id"], "symbol": symbol,
                "subreddit": row["subreddit"], "source": "reddit",
                "author": row["author"],
                "created_utc": row["created_utc"], "sentiment": pol,
                "confidence": conf,
                "weight": round((1.0 + engagement) * kind_w * rule_w, 4),
                "matched_on": matched_on,
            })

    store.upsert_mentions(mentions)
    if verbose:
        print(f"  -> {len(all_rows)} items, {len(mentions)} symbol mentions stored "
              f"[source: {client.mode}]")
    return {"posts": len(all_rows), "mentions": len(mentions), "blocked": 0,
            "source": client.mode, "degraded": list(client.degraded)}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_symbols(store: Store, cfg: Config = CONFIG) -> pd.DataFrame:
    """Aggregate stored mentions into a per-symbol social score in [-1, 1].

    social = tanh(velocity) * weighted_sentiment * diversity * confidence

    velocity     recent mention rate vs. this asset's own baseline (a ratio, so
                 BTC's huge constant volume doesn't dominate)
    diversity    unique authors / mentions - suppresses one person posting 40x
    confidence   share of mentions that carried actual directional language
    """
    rc = cfg.reddit
    now = utcnow()
    recent_start = now - timedelta(hours=rc.velocity_window_hours)
    baseline_start = now - timedelta(days=rc.baseline_days)

    hist = store.mentions_since(baseline_start)
    # Measure over the SAME window the baseline uses, or the two silently
    # disagree the moment baseline_days is changed from its default.
    span_hours = store.history_span_hours(within_days=rc.baseline_days)

    # Per-source mean tone across ALL assets in the window. Subtracting it makes
    # the score measure deviation from a platform's norm rather than its
    # absolute cheerfulness.
    source_bias: dict[str, float] = {}
    if not hist.empty and "source" in hist.columns:
        grouped = hist.groupby("source")["sentiment"]
        min_n = getattr(rc, "min_samples_for_debias", 30)
        for src, series in grouped:
            n = len(series)
            # Shrink the correction toward zero on thin samples: with a handful
            # of mentions the "platform mean" is mostly noise, and subtracting
            # noise is worse than subtracting nothing.
            shrink = min(1.0, n / float(min_n)) if min_n else 1.0
            source_bias[src] = float(series.mean()) * shrink

    rows = []

    for symbol in cfg.symbols:
        sub = hist[hist["symbol"] == symbol] if not hist.empty else pd.DataFrame()
        if sub.empty:
            rows.append({"symbol": symbol, "mentions_24h": 0, "unique_authors": 0,
                         "avg_sentiment": 0.0, "velocity": 0.0, "diversity": 0.0,
                         "confidence": 0.0, "social": 0.0})
            continue

        recent = sub[sub["created_at"] >= pd.Timestamp(recent_start)]
        older = sub[sub["created_at"] < pd.Timestamp(recent_start)]

        n_recent = len(recent)
        recent_rate = n_recent / max(1.0, rc.velocity_window_hours)
        older_hours = max(1.0, min(span_hours, rc.baseline_days * 24) - rc.velocity_window_hours)
        base_rate = len(older) / older_hours if len(older) else 0.0

        # Ratio vs. own baseline, log-scaled. +1 means "double the usual chatter".
        if base_rate > 0:
            velocity = math.log2((recent_rate + 1e-9) / (base_rate + 1e-9))
        else:
            velocity = 0.5 if n_recent >= 3 else 0.0

        if n_recent:
            w = recent["weight"].to_numpy()
            # Centre sentiment on each SOURCE's own baseline before comparing
            # assets. StockTwits users self-label ~88% Bullish (mean +0.55 vs
            # Reddit's +0.05), so raw tone made 18 of 20 assets look strongly
            # positive - a signal that rates everything a buy cannot rank
            # anything. What carries information is being bullish *relative to
            # how bullish that platform always is*, the same relative-to-own-
            # baseline logic already used for mention velocity.
            s = (recent["sentiment"] - recent["source"].map(source_bias).fillna(0.0)
                 ).clip(-1.0, 1.0).to_numpy()
            c = recent["confidence"].to_numpy()
            eff = w * c                        # opinion-free posts barely count
            avg_sent = float((s * eff).sum() / eff.sum()) if eff.sum() > 0 else 0.0
            # Diversity is measured only over mentions that HAVE an author.
            # /biz/ is anonymous, so its rows carry author=None; pandas
            # nunique() skips those, and counting them in the denominator
            # anyway silently depressed diversity for every asset /biz/
            # discussed - penalising breadth instead of measuring it.
            attributed = recent[recent["author"].notna()]
            authors = attributed["author"].nunique()
            n_attributed = len(attributed)
            diversity = (min(1.0, authors / max(3.0, n_attributed * 0.5))
                         if n_attributed else 0.0)
            avg_conf = float(c.mean())
        else:
            avg_sent, authors, diversity, avg_conf = 0.0, 0, 0.0, 0.0

        # Velocity sets magnitude, sentiment sets sign. A quiet coin with a
        # positive tone shouldn't outrank a coin whose chatter just tripled.
        magnitude = math.tanh(max(0.0, velocity) / 2.0)
        social = magnitude * avg_sent * (0.4 + 0.6 * diversity) * (0.3 + 0.7 * avg_conf)

        rows.append({
            "symbol": symbol,
            "mentions_24h": n_recent,
            "unique_authors": int(authors),
            "avg_sentiment": round(avg_sent, 4),
            "velocity": round(velocity, 4),
            "diversity": round(diversity, 4),
            "confidence": round(avg_conf, 4),
            "social": round(float(max(-1.0, min(1.0, social))), 4),
        })

    df = pd.DataFrame(rows).sort_values("mentions_24h", ascending=False).reset_index(drop=True)
    df.attrs["history_hours"] = span_hours
    df.attrs["baseline_ready"] = span_hours >= rc.velocity_window_hours * 1.5
    return df
