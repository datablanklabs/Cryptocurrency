"""Feature 2b: non-Reddit social feeds — StockTwits and Mastodon.

Both are keyless and were measured before being wired in (2026-08-20):

  StockTwits  100% of messages are on-topic, because the cashtag stream
              attributes each one to a symbol before we ever see it. All 20
              universe assets resolve, 30 messages each. ~63% carry a
              user-declared Bullish/Bearish label - a stated opinion, which is
              much stronger evidence than anything a word list can infer.

  Mastodon    Public hashtag timelines, no auth, 300 requests per window.
              Only broad tags are alive: #crypto 3.6 posts/h, #bitcoin 3.5/h,
              while per-asset tags are near-dead (#dot is one post per 33h).
              So we pull wide tags and extract, instead of burning a request
              per asset on tags returning month-old content.

Both normalize into the same row shape the Reddit path uses, so mentions land
in one table and feed one score. The `source` column keeps them separable for
analysis - which matters, because their sentiment baselines differ sharply:
StockTwits sampled 48 bullish to 9 bearish, a long skew that would bias a naive
cross-platform average.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any

import requests

from .config import CONFIG, Config
from .social import _build_patterns, _strip_tags, extract_symbols, sentiment
from .store import Store, iso


def _session(cfg: Config) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": cfg.user_agent})
    return s


def _row(post_id, channel, source, kind, author, created, score,
         title, body, permalink) -> dict[str, Any]:
    return {
        "post_id": post_id, "subreddit": channel, "source": source,
        "kind": kind, "author": author, "created_utc": float(created or 0),
        "score": int(score or 0), "num_comments": 0,
        "title": title or "", "body": body or "",
        "permalink": permalink or "", "fetched_at": iso(),
    }


# --------------------------------------------------------------------------
# StockTwits
# --------------------------------------------------------------------------
def fetch_stocktwits(cfg: Config = CONFIG, verbose: bool = True
                     ) -> tuple[list[dict], list[dict]]:
    """Return (posts, mentions). Mentions are direct, not extracted."""
    st = cfg.stocktwits
    if not st.enabled:
        return [], []
    sess = _session(cfg)
    posts: list[dict] = []
    mentions: list[dict] = []
    labelled = 0

    targets = cfg.universe if not st.max_symbols else cfg.universe[: st.max_symbols]
    for asset in targets:
        pair = f"{asset.symbol}{st.symbol_suffix}"
        try:
            r = sess.get(f"https://api.stocktwits.com/api/2/streams/symbol/{pair}.json",
                         timeout=cfg.http_timeout)
            if r.status_code == 429:
                if verbose:
                    print(f"  ! StockTwits rate limited at {pair}; stopping early")
                break
            r.raise_for_status()
            msgs = r.json().get("messages", [])
        except Exception as exc:  # noqa: BLE001 - one symbol failing isn't fatal
            if verbose:
                print(f"  {pair}: failed ({exc})")
            time.sleep(st.request_delay)
            continue

        for m in msgs:
            body = m.get("body", "")
            created = 0.0
            try:
                created = datetime.strptime(
                    m["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc).timestamp()
            except Exception:  # noqa: BLE001 - undated message
                pass
            user = (m.get("user") or {}).get("username")
            pid = f"st_{m.get('id')}"
            posts.append(_row(pid, pair, "stocktwits", "post", user, created,
                              (m.get("likes") or {}).get("total", 0),
                              "", body,
                              f"https://stocktwits.com/message/{m.get('id')}"))

            # Prefer the user's own label over our lexicon.
            basic = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
            if basic == "Bullish":
                pol, conf = 1.0, st.label_confidence
                labelled += 1
            elif basic == "Bearish":
                pol, conf = -1.0, st.label_confidence
                labelled += 1
            else:
                pol, conf = sentiment(body)

            mentions.append({
                "post_id": pid, "symbol": asset.symbol, "subreddit": pair,
                "source": "stocktwits", "author": user, "created_utc": created,
                "sentiment": pol, "confidence": conf,
                # No engagement signal to speak of, so weight is flat; the
                # label carries the information instead.
                "weight": 1.0,
                "matched_on": "cashtag_label" if basic else "cashtag",
            })
        time.sleep(st.request_delay)

    if verbose:
        share = labelled / len(mentions) * 100 if mentions else 0
        print(f"  StockTwits: {len(posts)} messages, {len(mentions)} mentions "
              f"({share:.0f}% user-labelled)")
    return posts, mentions


# --------------------------------------------------------------------------
# Mastodon
# --------------------------------------------------------------------------
def fetch_mastodon(cfg: Config = CONFIG, verbose: bool = True
                   ) -> tuple[list[dict], list[dict]]:
    md = cfg.mastodon
    if not md.enabled:
        return [], []
    sess = _session(cfg)
    patterns = _build_patterns(cfg)
    posts: list[dict] = []
    mentions: list[dict] = []

    for tag in md.hashtags:
        try:
            r = sess.get(f"{md.instance}/api/v1/timelines/tag/{tag}",
                         params={"limit": md.limit}, timeout=cfg.http_timeout)
            if r.status_code == 429:
                if verbose:
                    print(f"  ! Mastodon rate limited at #{tag}; stopping early")
                break
            r.raise_for_status()
            statuses = r.json()
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  #{tag}: failed ({exc})")
            time.sleep(md.request_delay)
            continue

        for stt in statuses:
            text = _strip_tags(stt.get("content", ""))     # content is HTML
            if not text:
                continue
            created = 0.0
            try:
                created = datetime.fromisoformat(
                    stt["created_at"].replace("Z", "+00:00")).timestamp()
            except Exception:  # noqa: BLE001
                pass
            acct = (stt.get("account") or {}).get("acct")
            pid = f"md_{stt.get('id')}"
            posts.append(_row(pid, f"#{tag}", "mastodon", "post", acct, created,
                              stt.get("favourites_count", 0), "", text,
                              stt.get("url", "")))

            found = extract_symbols(text, patterns)
            if not found:
                continue
            pol, conf = sentiment(text)
            for symbol, matched_on in found.items():
                mentions.append({
                    "post_id": pid, "symbol": symbol, "subreddit": f"#{tag}",
                    "source": "mastodon", "author": acct, "created_utc": created,
                    "sentiment": pol, "confidence": conf, "weight": 1.0,
                    "matched_on": matched_on,
                })
        time.sleep(md.request_delay)

    if verbose:
        print(f"  Mastodon: {len(posts)} statuses, {len(mentions)} mentions")
    return posts, mentions


# --------------------------------------------------------------------------
# 4chan /biz/
# --------------------------------------------------------------------------
_BACKLINK_RE = re.compile(r">>\d+")


def _clean_biz(html_text: str) -> str:
    """Strip 4chan markup: post backlinks (>>12345) carry no sentiment."""
    txt = _BACKLINK_RE.sub(" ", _strip_tags(html_text or ""))
    return " ".join(txt.split())


def fetch_biz(cfg: Config = CONFIG, verbose: bool = True
              ) -> tuple[list[dict], list[dict]]:
    """Thread OPs plus the replies the catalog already embeds.

    `last_replies` comes free inside the catalog response, so we get the newest
    few posts per thread without a request each - which matters, since replies
    are where the actual opinions are and OPs are mostly bait.
    """
    bz = cfg.biz
    if not bz.enabled:
        return [], []
    sess = _session(cfg)
    patterns = _build_patterns(cfg)
    posts: list[dict] = []
    mentions: list[dict] = []

    try:
        r = sess.get(f"https://a.4cdn.org/{bz.board}/catalog.json",
                     timeout=cfg.http_timeout)
        r.raise_for_status()
        pages = r.json()
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  /{bz.board}/ catalog failed ({exc})")
        return [], []

    items: list[dict] = []
    for page in pages if isinstance(pages, list) else []:
        for th in page.get("threads", []):
            items.append(th)
            if bz.include_replies:
                items.extend(th.get("last_replies") or [])

    for it in items:
        text = _clean_biz(f"{it.get('sub','')} {it.get('com','')}")
        if not text:
            continue
        pid = f"4c_{it.get('no')}"
        created = float(it.get("time") or 0)
        posts.append(_row(pid, f"/{bz.board}/", "biz", "post", None, created,
                          it.get("replies", 0), "", text,
                          f"https://boards.4chan.org/{bz.board}/thread/{it.get('resto') or it.get('no')}"))

        found = extract_symbols(text, patterns)
        if not found:
            continue
        pol, conf = sentiment(text)
        for symbol, matched_on in found.items():
            mentions.append({
                "post_id": pid, "symbol": symbol, "subreddit": f"/{bz.board}/",
                "source": "biz", "author": None, "created_utc": created,
                "sentiment": pol, "confidence": conf,
                # Anonymous board: no karma, no author diversity to lean on.
                "weight": float(bz.weight),
                "matched_on": matched_on,
            })
    time.sleep(bz.request_delay)

    if verbose:
        neg = sum(1 for m in mentions if m["sentiment"] < 0)
        share = neg / len(mentions) * 100 if mentions else 0
        print(f"  /{bz.board}/: {len(posts)} posts, {len(mentions)} mentions "
              f"({share:.0f}% bearish)")
    return posts, mentions


# --------------------------------------------------------------------------
def scrape(store: Store, cfg: Config = CONFIG, verbose: bool = True) -> dict[str, int]:
    """Collect every enabled non-Reddit feed and persist it."""
    stats = {"stocktwits": 0, "mastodon": 0, "biz": 0, "mentions": 0}
    for name, fn in (("stocktwits", fetch_stocktwits), ("mastodon", fetch_mastodon),
                     ("biz", fetch_biz)):
        try:
            posts, mentions = fn(cfg, verbose)
        except Exception as exc:  # noqa: BLE001 - one platform must not sink the rest
            if verbose:
                print(f"  ! {name} failed: {exc}")
            continue
        if posts:
            store.upsert_posts(posts)
        if mentions:
            store.upsert_mentions(mentions)
        stats[name] = len(posts)
        stats["mentions"] += len(mentions)
    return stats
