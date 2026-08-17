"""Feature 2: Reddit scraping, mention extraction, and sentiment.

Three parts:

  RedditClient    Pulls posts and comments from r/wallstreetbets and
                  r/cryptocurrency. Reddit blocks anonymous .json access with
                  HTTP 403 from most hosts now, so the OAuth path (a free
                  "script" app) is the real one; the public path stays as a
                  fallback for hosts where it still works.

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
volume is trivially manipulated, and by the time something trends on r/wsb the
move that caused the trend has usually already happened. This is why social is
weighted lowest of the three families by default.
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
    OAUTH = "https://oauth.reddit.com"
    PUBLIC = "https://www.reddit.com"

    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg
        self.session = requests.Session()
        self.ua = get_secret("reddit_user_agent") or cfg.user_agent
        self.session.headers.update({"User-Agent": self.ua})
        self._token: str | None = None
        self._token_expiry: float = 0.0
        self.mode = "unauthenticated"

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
            self.mode = "oauth"
            return True
        except Exception as exc:  # noqa: BLE001 - degrade to the public endpoint
            print(f"  ! Reddit OAuth failed ({exc}); falling back to public JSON")
            return False

    def _request(self, path: str, params: dict) -> dict | None:
        authed = self.authenticate()
        base = self.OAUTH if authed else self.PUBLIC
        url = f"{base}{path}" if authed else f"{base}{path}.json"
        headers = {"User-Agent": self.ua}
        if authed:
            headers["Authorization"] = f"Bearer {self._token}"

        for attempt in range(self.cfg.http_retries):
            try:
                r = self.session.get(url, params=params, headers=headers,
                                     timeout=self.cfg.http_timeout)
                if r.status_code in (429, 503):
                    time.sleep(2 ** attempt + 1)
                    continue
                if r.status_code == 403:
                    raise PermissionError(
                        "Reddit returned 403. Anonymous access is blocked - set "
                        "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET (free script app "
                        "at https://www.reddit.com/prefs/apps)."
                    )
                r.raise_for_status()
                return r.json()
            except PermissionError:
                raise
            except Exception:  # noqa: BLE001 - retry with backoff
                time.sleep(0.6 * (attempt + 1))
        return None

    # -- fetching -----------------------------------------------------------
    def listing(self, subreddit: str, listing: str = "new",
                limit: int = 100) -> list[dict[str, Any]]:
        rows, after, fetched = [], None, 0
        while fetched < limit:
            params = {"limit": min(100, limit - fetched), "raw_json": 1}
            if after:
                params["after"] = after
            payload = self._request(f"/r/{subreddit}/{listing}", params)
            if not payload or "data" not in payload:
                break
            children = payload["data"].get("children", [])
            if not children:
                break
            for ch in children:
                d = ch.get("data", {})
                rows.append({
                    "post_id": f"t3_{d.get('id')}",
                    "subreddit": subreddit,
                    "kind": "post",
                    "author": d.get("author"),
                    "created_utc": float(d.get("created_utc") or 0),
                    "score": int(d.get("score") or 0),
                    "num_comments": int(d.get("num_comments") or 0),
                    "title": d.get("title") or "",
                    "body": d.get("selftext") or "",
                    "permalink": f"https://reddit.com{d.get('permalink', '')}",
                    "fetched_at": iso(),
                })
            fetched += len(children)
            after = payload["data"].get("after")
            if not after:
                break
            time.sleep(0.6)
        return rows

    def comments(self, subreddit: str, limit: int = 100) -> list[dict[str, Any]]:
        """Recent comments across the whole subreddit.

        Much cheaper than walking each post's comment tree, and comments are
        where the actual opinions live - post titles are mostly memes.
        """
        payload = self._request(f"/r/{subreddit}/comments",
                                {"limit": min(100, limit), "raw_json": 1})
        if not payload or "data" not in payload:
            return []
        rows = []
        for ch in payload["data"].get("children", []):
            d = ch.get("data", {})
            rows.append({
                "post_id": f"t1_{d.get('id')}",
                "subreddit": subreddit,
                "kind": "comment",
                "author": d.get("author"),
                "created_utc": float(d.get("created_utc") or 0),
                "score": int(d.get("score") or 0),
                "num_comments": 0,
                "title": "",
                "body": d.get("body") or "",
                "permalink": f"https://reddit.com{d.get('permalink', '')}",
                "fetched_at": iso(),
            })
        return rows


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

    for sub in rc.subreddits:
        for listing in rc.listings:
            try:
                rows = client.listing(sub, listing, rc.posts_per_listing)
                all_rows.extend(rows)
                if verbose:
                    print(f"  r/{sub}/{listing:<4} {len(rows):>4} posts   [{client.mode}]")
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
        return {"posts": 0, "mentions": 0, "blocked": 0}

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
                "subreddit": row["subreddit"], "author": row["author"],
                "created_utc": row["created_utc"], "sentiment": pol,
                "confidence": conf,
                "weight": round((1.0 + engagement) * kind_w * rule_w, 4),
                "matched_on": matched_on,
            })

    store.upsert_mentions(mentions)
    if verbose:
        print(f"  -> {len(all_rows)} items, {len(mentions)} symbol mentions stored")
    return {"posts": len(all_rows), "mentions": len(mentions), "blocked": 0}


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
    span_hours = store.history_span_hours()
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
            s = recent["sentiment"].to_numpy()
            c = recent["confidence"].to_numpy()
            eff = w * c                        # opinion-free posts barely count
            avg_sent = float((s * eff).sum() / eff.sum()) if eff.sum() > 0 else 0.0
            authors = recent["author"].nunique()
            diversity = min(1.0, authors / max(3.0, n_recent * 0.5))
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
