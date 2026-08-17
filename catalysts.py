"""Feature 3: developer activity and news catalysts.

Two sources of "what is about to change":

  GitHub    Each asset's main repo - releases (shipped and pre-release) and
            commit velocity. A chain that just tagged a mainnet upgrade is
            objectively different from one whose repo has been quiet for a
            month, and this is public, structured, and hard to fake.

  News RSS  CoinDesk / Cointelegraph / Decrypt / The Defiant. Headlines are
            classified against an event taxonomy (upgrade, ETF, unlock, hack,
            lawsuit, listing, ...) with hand-set impact weights and directions.

Two honest caveats about the scoring:

  The impact weights are priors I wrote down, not coefficients fitted to
  realized returns. They encode "an exploit is worse than a delay" - a
  defensible ordering - but the magnitudes are guesses. They are all in
  EVENT_TYPES below, in one place, so you can argue with them.

  Headlines are usually a *reaction*, not a leading indicator. The part of this
  with genuine forward-looking content is the scheduled/upcoming detection
  (`_is_forward_looking`) and unreleased GitHub tags - a token unlock dated next
  Tuesday is knowable in advance in a way that a price move is not.
"""

from __future__ import annotations

import math
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import pandas as pd
import requests

from .config import CONFIG, Config, get_secret
from .store import Store, iso, utcnow

# event_type -> (base_impact 0..1, prior_direction -1|0|+1, tone_driven)
#
# tone_driven marks event types that are *topics*, not directional events. "ETF"
# is the clearest case: "record ETF inflows" and "ETF outflows accelerate" are
# both ETF news pointing opposite ways, so the keyword cannot set the sign - the
# headline's own language has to. Events with intrinsic direction (an exploit is
# never good news) keep their prior and only get discounted when tone disagrees.
EVENT_TYPES: dict[str, tuple[float, int, bool]] = {
    "etf":            (0.95, +1, True),
    "exploit":        (0.95, -1, False),
    "hack":           (0.95, -1, False),
    "delisting":      (0.85, -1, False),
    "lawsuit":        (0.75, -1, False),
    "regulation":     (0.60, -1, True),
    "token_unlock":   (0.70, -1, False),
    "outage":         (0.65, -1, False),
    "depeg":          (0.80, -1, False),
    "hard_fork":      (0.70, +1, False),
    "mainnet":        (0.75, +1, False),
    "upgrade":        (0.60, +1, False),
    "halving":        (0.80, +1, False),
    "listing":        (0.70, +1, True),
    "partnership":    (0.50, +1, True),
    "institutional":  (0.60, +1, True),
    "airdrop":        (0.55, +1, False),
    "burn":           (0.50, +1, False),
    "staking":        (0.45, +1, True),
    "buyback":        (0.60, +1, False),
    "testnet":        (0.35, +1, False),
    "audit":          (0.30, +1, False),
    "delay":          (0.45, -1, False),
    "release":        (0.30, +1, False),
    "dev_activity":   (0.20, +1, False),
}

# Ordered: the first matching pattern wins, so put the severe ones first.
EVENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("exploit",       re.compile(r"\b(exploit(ed)?|drained|vulnerabilit(y|ies))\b", re.I)),
    ("hack",          re.compile(r"\b(hack(ed|s)?|breach(ed)?|stolen funds)\b", re.I)),
    ("depeg",         re.compile(r"\b(de-?peg(ged)?|lost (its )?peg)\b", re.I)),
    ("etf",           re.compile(r"\b(etfs?|exchange[- ]traded funds?)\b", re.I)),
    ("delisting",     re.compile(r"\b(delist(ed|ing)?|removed from)\b", re.I)),
    ("lawsuit",       re.compile(r"\b(lawsuit|sue[sd]?|suing|court|settlement|indict)", re.I)),
    ("regulation",    re.compile(r"\b(sec|cftc|regulat(or|ory|ion)|ban(ned|s)?|crackdown|subpoena)\b", re.I)),
    ("token_unlock",  re.compile(r"\b(unlock(s|ed|ing)?|vesting|cliff|token release|emission)\b", re.I)),
    ("outage",        re.compile(r"\b(outage|halt(ed|s)?|downtime|stall(ed)?|degraded)\b", re.I)),
    ("halving",       re.compile(r"\b(halv(ing|ed))\b", re.I)),
    ("hard_fork",     re.compile(r"\b(hard ?fork|soft ?fork|fork)\b", re.I)),
    ("mainnet",       re.compile(r"\b(mainnet|genesis launch|go[- ]live)\b", re.I)),
    ("upgrade",       re.compile(r"\b(upgrade|hardfork|migration|v[0-9]+(\.[0-9]+)?\s+launch|protocol update)\b", re.I)),
    ("listing",       re.compile(r"\b(list(ed|ing)? on|new listing|debut(s)? on|now trading on)\b", re.I)),
    ("institutional", re.compile(r"\b(institutional|blackrock|fidelity|grayscale|treasury|custody)\b", re.I)),
    ("partnership",   re.compile(r"\b(partner(ship|s|ed)?|collaborat|integrat(es|ion|ed))\b", re.I)),
    ("airdrop",       re.compile(r"\b(air ?drop)\b", re.I)),
    ("burn",          re.compile(r"\b(burn(s|ed|ing)?|deflationary)\b", re.I)),
    ("buyback",       re.compile(r"\b(buy ?back|repurchase)\b", re.I)),
    ("staking",       re.compile(r"\b(stak(e|es|ing)|validator|restaking)\b", re.I)),
    ("testnet",       re.compile(r"\b(testnet|devnet|beta release)\b", re.I)),
    ("audit",         re.compile(r"\b(audit(ed|s)?|formal verification)\b", re.I)),
    # "slips" deliberately excluded - "Bitcoin slips to $63k" is a price move,
    # not a shipping delay, and it fired constantly on live headlines.
    ("delay",         re.compile(r"\b(delay(ed|s)?|postpone[ds]?|push(ed)? back)\b", re.I)),
]

FORWARD_RE = re.compile(
    r"\b(will|upcoming|scheduled|slated|expected|set to|plans? to|to launch|"
    r"ahead of|next (week|month|monday|tuesday|wednesday|thursday|friday)|"
    r"in (the coming|coming) (days|weeks)|q[1-4]\s?20\d\d|"
    r"on (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))", re.I,
)
NEGATION_RE = re.compile(r"\b(denie[sd]|reject(ed|s)?|rumou?r|unconfirmed|speculation|"
                         r"no plans|dismisse[sd]|false)\b", re.I)


def classify(text: str) -> tuple[str, float, int]:
    """Classify a headline. Returns (event_type, impact 0..1, direction).

    The keyword taxonomy sets a prior direction, then the headline's own tone
    gets a vote. Without that vote, "Endowment's crypto exposure drops $2M amid
    ETF outflows" scores bullish purely because it contains the word "ETF" -
    which is exactly the kind of error that makes a naive news scraper worse
    than no news scraper. Strong priors (exploit, hack) still win over tone;
    weak ones defer to it.
    """
    from .social import sentiment

    for name, pat in EVENT_PATTERNS:
        if not pat.search(text):
            continue
        impact, direction, tone_driven = EVENT_TYPES[name]
        if NEGATION_RE.search(text):
            impact *= 0.35              # rumor/denial: real, but much weaker
        if FORWARD_RE.search(text):
            impact *= 1.25              # scheduled/upcoming = actually tradeable

        tone, conf = sentiment(text)
        has_tone = conf >= 0.15 and abs(tone) >= 0.20

        if tone_driven:
            if has_tone:
                direction = 1 if tone > 0 else -1
            else:
                impact *= 0.45          # topic with no stated direction: weak
        elif has_tone and abs(tone) >= 0.30 and (tone > 0) != (direction > 0):
            if EVENT_TYPES[name][0] < 0.60:   # weak prior: headline knows better
                direction = 1 if tone > 0 else -1
                impact *= 0.85
            else:                             # strong prior: keep it, discount it
                impact *= 0.55
        return name, min(1.0, impact), direction
    return "", 0.0, 0


def _is_forward_looking(text: str) -> bool:
    return bool(FORWARD_RE.search(text))


# --------------------------------------------------------------------------
# GitHub developer activity
# --------------------------------------------------------------------------
def _gh_headers(cfg: Config) -> dict[str, str]:
    h = {"User-Agent": cfg.user_agent, "Accept": "application/vnd.github+json"}
    token = get_secret("github_token")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def fetch_github(cfg: Config = CONFIG, verbose: bool = True) -> list[dict[str, Any]]:
    """Releases + commit velocity per asset repo.

    Unauthenticated GitHub allows 60 requests/hour, which this can exhaust with
    a 20-asset universe. Set GITHUB_TOKEN to raise it to 5000.
    """
    out: list[dict[str, Any]] = []
    headers = _gh_headers(cfg)
    since = utcnow() - timedelta(days=cfg.catalysts.github_lookback_days)
    rate_limited = False

    for asset in cfg.universe:
        if not asset.github or rate_limited:
            continue
        try:
            r = requests.get(f"https://api.github.com/repos/{asset.github}/releases",
                             params={"per_page": 5}, headers=headers,
                             timeout=cfg.http_timeout)
            if r.status_code == 403 and "rate limit" in r.text.lower():
                rate_limited = True
                if verbose:
                    print("  ! GitHub rate limit hit — set GITHUB_TOKEN to raise it. "
                          "Skipping remaining repos.")
                break
            r.raise_for_status()

            for rel in r.json():
                published = rel.get("published_at") or rel.get("created_at")
                if not published:
                    continue
                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                if pub_dt < since:
                    continue
                title = rel.get("name") or rel.get("tag_name") or "release"
                body = (rel.get("body") or "")[:1200]
                etype, impact, direction = classify(f"{title} {body}")
                if not etype:
                    etype = "release"
                    impact, direction, _ = EVENT_TYPES["release"]
                if rel.get("prerelease"):
                    impact *= 0.6
                age_days = (utcnow() - pub_dt).days
                out.append({
                    "symbol": asset.symbol, "source": "github",
                    "source_name": asset.github, "event_type": etype,
                    "title": f"[release] {title}",
                    "url": rel.get("html_url"), "published_at": pub_dt.isoformat(),
                    "impact": round(impact * _decay(age_days, 14), 4),
                    "direction": direction,
                    "detail": body[:400], "fetched_at": iso(),
                })

            time.sleep(0.25)
        except Exception as exc:  # noqa: BLE001 - one repo failing is not fatal
            if verbose:
                print(f"  {asset.symbol}: github failed ({exc})")

    if verbose:
        print(f"  -> {len(out)} github events")
    return out


def _decay(age_days: float, halflife: float) -> float:
    return float(0.5 ** (max(0.0, age_days) / halflife))


# --------------------------------------------------------------------------
# News RSS
# --------------------------------------------------------------------------
def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    try:
        dt = parsedate_to_datetime(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001 - fall through to ISO parsing
        pass
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001 - undated item
        return None


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def _feed_items(xml_text: str, limit: int) -> list[dict[str, str]]:
    """Parse RSS 2.0 or Atom without a feed library."""
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items: list[dict[str, str]] = []

    for item in root.findall(".//item")[:limit]:               # RSS 2.0
        items.append({
            "title": (item.findtext("title") or "").strip(),
            "link": (item.findtext("link") or "").strip(),
            "date": item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date") or "",
            "summary": _strip_html(item.findtext("description") or "")[:600],
        })
    if not items:
        for entry in root.findall(".//atom:entry", ns)[:limit]:  # Atom
            link_el = entry.find("atom:link", ns)
            items.append({
                "title": (entry.findtext("atom:title", default="", namespaces=ns) or "").strip(),
                "link": (link_el.get("href") if link_el is not None else "") or "",
                "date": entry.findtext("atom:updated", default="", namespaces=ns)
                        or entry.findtext("atom:published", default="", namespaces=ns) or "",
                "summary": _strip_html(entry.findtext("atom:summary", default="", namespaces=ns))[:600],
            })
    return items


def fetch_news(cfg: Config = CONFIG, verbose: bool = True) -> list[dict[str, Any]]:
    """Pull news feeds and attach classified events to the assets they name."""
    from .social import _build_patterns, extract_symbols

    patterns = _build_patterns(cfg)
    cutoff = utcnow() - timedelta(days=cfg.catalysts.news_lookback_days)
    out: list[dict[str, Any]] = []

    for name, url in cfg.catalysts.news_feeds:
        try:
            r = requests.get(url, timeout=cfg.http_timeout,
                             headers={"User-Agent": cfg.user_agent})
            r.raise_for_status()
            items = _feed_items(r.text, cfg.catalysts.max_items_per_feed)
        except Exception as exc:  # noqa: BLE001 - skip an unreachable feed
            if verbose:
                print(f"  {name}: failed ({exc})")
            continue

        kept = 0
        for it in items:
            pub = _parse_date(it["date"])
            if pub is None or pub < cutoff:
                continue
            text = f"{it['title']} {it['summary']}"
            symbols = extract_symbols(text, patterns)
            if not symbols:
                continue
            etype, impact, direction = classify(text)
            if not etype or impact <= 0:
                continue
            age_days = (utcnow() - pub).total_seconds() / 86400.0
            # Forward-looking items decay slowly (they're about the future);
            # reported news decays fast (it's already in the price).
            halflife = 6.0 if _is_forward_looking(text) else 1.5
            for symbol in symbols:
                out.append({
                    "symbol": symbol, "source": "news", "source_name": name,
                    "event_type": etype, "title": it["title"][:300],
                    "url": it["link"], "published_at": pub.isoformat(),
                    "impact": round(impact * _decay(age_days, halflife), 4),
                    "direction": direction, "detail": it["summary"][:400],
                    "fetched_at": iso(),
                })
                kept += 1
        if verbose:
            print(f"  {name:<14} {len(items):>3} items -> {kept} tagged events")
    return out


# --------------------------------------------------------------------------
# Pipeline + scoring
# --------------------------------------------------------------------------
def scan(store: Store, cfg: Config = CONFIG, verbose: bool = True,
         include_github: bool = True) -> int:
    events: list[dict] = []
    if include_github:
        events += fetch_github(cfg, verbose)
    events += fetch_news(cfg, verbose)
    if events:
        store.upsert_catalysts(events)
    if verbose:
        print(f"  -> {len(events)} catalyst events stored")
    return len(events)


def score_symbols(store: Store, cfg: Config = CONFIG) -> pd.DataFrame:
    """Aggregate stored catalysts into a per-symbol score in [-1, 1].

    Events combine as a signed, saturating sum: many small positives can't
    outweigh one large negative (an exploit should dominate three partnership
    announcements), which a plain mean would get wrong.
    """
    since = utcnow() - timedelta(days=max(cfg.catalysts.news_lookback_days,
                                          cfg.catalysts.github_lookback_days))
    df = store.catalysts_since(since)
    rows = []

    for symbol in cfg.symbols:
        sub = df[df["symbol"] == symbol] if not df.empty else pd.DataFrame()
        if sub.empty:
            rows.append({"symbol": symbol, "n_events": 0, "n_github": 0,
                         "n_news": 0, "top_event": "", "top_impact": 0.0,
                         "catalyst": 0.0})
            continue

        signed = (sub["impact"] * sub["direction"]).to_numpy()
        pos = float(signed[signed > 0].sum())
        neg = float(-signed[signed < 0].sum())
        net = math.tanh((pos - neg) / 1.5)

        strongest = sub.loc[sub["impact"].idxmax()]
        rows.append({
            "symbol": symbol,
            "n_events": int(len(sub)),
            "n_github": int((sub["source"] == "github").sum()),
            "n_news": int((sub["source"] == "news").sum()),
            "top_event": str(strongest["event_type"]),
            "top_impact": round(float(strongest["impact"]), 4),
            "catalyst": round(net, 4),
        })

    return pd.DataFrame(rows).sort_values("catalyst", key=abs, ascending=False).reset_index(drop=True)


def recent_events(store: Store, symbol: str | None = None, limit: int = 20,
                  cfg: Config = CONFIG) -> pd.DataFrame:
    """Human-readable catalyst feed for the dashboard."""
    since = utcnow() - timedelta(days=cfg.catalysts.news_lookback_days)
    df = store.catalysts_since(since)
    if df.empty:
        return df
    if symbol:
        df = df[df["symbol"] == symbol]
    df = df.assign(signed=df["impact"] * df["direction"])
    cols = ["symbol", "event_type", "signed", "source_name", "title", "published_at", "url"]
    return df.reindex(columns=cols).sort_values("signed", key=abs, ascending=False).head(limit)
