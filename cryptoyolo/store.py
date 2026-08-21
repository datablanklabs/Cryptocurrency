"""SQLite persistence.

The store is what makes "recurring" scraping meaningful: mention counts are only
interesting relative to a baseline, and a baseline needs history. Every run
appends to the same database, so the second and later runs can compute velocity
(how far today's chatter deviates from this asset's own normal).

It also gives you an auditable record of every proposal made and every order
sent, which matters a lot more than the charts if you ever want to know whether
this thing actually works.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    mode        TEXT,
    notes       TEXT
);

-- Historically Reddit-only; the name is kept to avoid migrating live data.
-- `source` distinguishes platforms and `subreddit` holds the channel within it
-- (subreddit name, StockTwits cashtag, or Mastodon hashtag).
CREATE TABLE IF NOT EXISTS reddit_posts (
    post_id     TEXT PRIMARY KEY,
    subreddit   TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'reddit',
    kind        TEXT NOT NULL,          -- 'post' | 'comment'
    author      TEXT,
    created_utc REAL NOT NULL,
    score       INTEGER,
    num_comments INTEGER,
    title       TEXT,
    body        TEXT,
    permalink   TEXT,
    fetched_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_created ON reddit_posts(created_utc);
CREATE INDEX IF NOT EXISTS idx_posts_sub ON reddit_posts(subreddit);

CREATE TABLE IF NOT EXISTS mentions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id      TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    subreddit    TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'reddit',
    author       TEXT,
    created_utc  REAL NOT NULL,
    sentiment    REAL NOT NULL,
    confidence   REAL NOT NULL,
    weight       REAL NOT NULL,
    matched_on   TEXT,
    UNIQUE(post_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_mentions_symbol_time ON mentions(symbol, created_utc);

CREATE TABLE IF NOT EXISTS catalysts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT NOT NULL,
    source       TEXT NOT NULL,         -- 'github' | 'news'
    source_name  TEXT,
    event_type   TEXT,
    title        TEXT NOT NULL,
    url          TEXT,
    published_at TEXT,
    impact       REAL NOT NULL,
    direction    INTEGER NOT NULL,      -- -1 bearish, 0 neutral, +1 bullish
    detail       TEXT,
    fetched_at   TEXT NOT NULL,
    UNIQUE(symbol, source, url, title)
);
CREATE INDEX IF NOT EXISTS idx_catalysts_symbol ON catalysts(symbol, published_at);

CREATE TABLE IF NOT EXISTS scores (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    ts            TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    technical     REAL, social REAL, catalyst REAL, composite REAL,
    components    TEXT
);
CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id);

CREATE TABLE IF NOT EXISTS proposals (
    proposal_id  TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    ts           TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    rank         INTEGER,
    composite    REAL,
    entry        REAL, stop REAL, target REAL,
    qty          REAL, notional REAL,
    horizon      TEXT,
    rationale    TEXT,
    payload      TEXT,
    decision     TEXT DEFAULT 'pending' -- pending | approved | rejected | expired
);

CREATE TABLE IF NOT EXISTS orders (
    order_id      TEXT PRIMARY KEY,
    proposal_id   TEXT,
    run_id        TEXT,
    ts            TEXT NOT NULL,
    venue         TEXT,
    mode          TEXT,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    order_type    TEXT,
    qty           REAL,
    price         REAL,
    status        TEXT,
    exchange_ref  TEXT,
    response      TEXT
);

CREATE TABLE IF NOT EXISTS paper_positions (
    symbol     TEXT PRIMARY KEY,
    qty        REAL NOT NULL,
    avg_price  REAL NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_cash (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    balance REAL NOT NULL
);

-- Entry metadata for open positions, kept as a sidecar keyed by symbol.
-- Deliberately separate from paper_positions: the *quantity* comes from the
-- paper book or from Binance balances depending on mode, but the exit terms
-- (when it was opened, where the stop is, how high it has run) are ours either
-- way. Without this table a position has no memory and no exit can be time- or
-- level-based.
CREATE TABLE IF NOT EXISTS position_meta (
    symbol       TEXT PRIMARY KEY,
    opened_at    TEXT NOT NULL,
    entry_price  REAL,
    stop         REAL,
    target       REAL,
    horizon_days REAL,
    high_water   REAL,
    proposal_id  TEXT,
    mode         TEXT,
    updated_at   TEXT
);

-- Protective orders resting at the exchange (or simulated in paper mode).
-- Tracked locally because a resting sell LOCKS the asset: an exit that tries to
-- market-sell without cancelling these first fails on insufficient free balance.
CREATE TABLE IF NOT EXISTS protective_orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT NOT NULL,
    kind         TEXT NOT NULL,       -- 'oco' | 'stop' | 'target'
    order_type   TEXT,
    exchange_ref TEXT,
    order_list_id TEXT,
    qty          REAL,
    stop_price   REAL,
    limit_price  REAL,
    target_price REAL,
    trailing_delta INTEGER,
    status       TEXT,                -- 'resting' | 'cancelled' | 'simulated' | 'failed'
    mode         TEXT,
    venue        TEXT,
    placed_at    TEXT NOT NULL,
    updated_at   TEXT,
    response     TEXT
);
CREATE INDEX IF NOT EXISTS idx_protective_symbol ON protective_orders(symbol, status);

-- Daily-refreshed fetch order for subreddits, derived from what each one has
-- actually delivered (see social.rank_subreddits). Cached because recomputing
-- it every collection would be wasted work and would make the order jitter.
-- Realised perpetual funding per asset. Kept as history because the score is a
-- z-value against each asset's OWN distribution, which needs a distribution.
CREATE TABLE IF NOT EXISTS funding_rates (
    symbol       TEXT NOT NULL,
    funding_time INTEGER NOT NULL,
    rate         REAL NOT NULL,
    venue        TEXT,
    fetched_at   TEXT NOT NULL,
    PRIMARY KEY (symbol, funding_time)
);
CREATE INDEX IF NOT EXISTS idx_funding_symbol ON funding_rates(symbol, funding_time);

CREATE TABLE IF NOT EXISTS source_rank (
    subreddit      TEXT PRIMARY KEY,
    computed_at    TEXT NOT NULL,
    rank           INTEGER,
    value          REAL,
    mention_rate   REAL,
    assets         INTEGER,
    tone           REAL,
    items_per_hour REAL,
    n_items        INTEGER
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utcnow()).isoformat()


class Store:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, re-creating the database if it has gone missing.

        The directory is only guaranteed to exist at construction time, but a
        Store instance can outlive it - a cleanup script, a synced folder that
        unmounts, an external drive. Previously that surfaced mid-run as
        "OperationalError: unable to open database file", which is an unhelpful
        way to lose a trading cycle. Re-creating the directory and schema on
        demand makes the store self-healing: you lose the history, but the run
        completes and the audit trail continues.
        """
        missing = not self.db_path.exists()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL")
            if missing:
                con.executescript(SCHEMA)
                con.execute(
                    "INSERT OR IGNORE INTO paper_cash(id, balance) VALUES (1, ?)",
                    (10_000.0,),
                )
            yield con
            con.commit()
        finally:
            con.close()

    def _migrate(self, con: sqlite3.Connection) -> None:
        """Additive migrations for databases created before a column existed.

        SQLite ALTER TABLE ADD COLUMN is cheap and non-destructive, so existing
        rows keep their data and default to 'reddit' - which is what they are.
        """
        for table in ("reddit_posts", "mentions"):
            cols = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
            if "source" not in cols:
                con.execute(
                    f"ALTER TABLE {table} ADD COLUMN source TEXT NOT NULL DEFAULT 'reddit'")

    def _init_schema(self) -> None:
        with self.conn() as con:
            con.executescript(SCHEMA)
            self._migrate(con)
            con.execute(
                "INSERT OR IGNORE INTO paper_cash(id, balance) VALUES (1, ?)", (10_000.0,)
            )

    # -- runs ---------------------------------------------------------------
    def start_run(self, run_id: str, mode: str, notes: str = "") -> None:
        with self.conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO runs(run_id, started_at, mode, notes) VALUES (?,?,?,?)",
                (run_id, iso(), mode, notes),
            )

    def finish_run(self, run_id: str) -> None:
        with self.conn() as con:
            con.execute("UPDATE runs SET finished_at=? WHERE run_id=?", (iso(), run_id))

    # -- reddit -------------------------------------------------------------
    def upsert_posts(self, rows: Iterable[dict[str, Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self.conn() as con:
            cur = con.executemany(
                """INSERT INTO reddit_posts
                   (post_id, subreddit, source, kind, author, created_utc, score,
                    num_comments, title, body, permalink, fetched_at)
                   VALUES (:post_id,:subreddit,:source,:kind,:author,:created_utc,:score,
                           :num_comments,:title,:body,:permalink,:fetched_at)
                   ON CONFLICT(post_id) DO UPDATE SET
                     score=excluded.score,
                     num_comments=excluded.num_comments,
                     fetched_at=excluded.fetched_at""",
                rows,
            )
            return cur.rowcount

    def upsert_mentions(self, rows: Iterable[dict[str, Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self.conn() as con:
            cur = con.executemany(
                """INSERT INTO mentions
                   (post_id, symbol, subreddit, source, author, created_utc,
                    sentiment, confidence, weight, matched_on)
                   VALUES (:post_id,:symbol,:subreddit,:source,:author,:created_utc,
                           :sentiment,:confidence,:weight,:matched_on)
                   ON CONFLICT(post_id, symbol) DO UPDATE SET
                     sentiment=excluded.sentiment,
                     weight=excluded.weight""",
                rows,
            )
            return cur.rowcount

    def mentions_since(self, since: datetime) -> pd.DataFrame:
        with self.conn() as con:
            df = pd.read_sql_query(
                "SELECT * FROM mentions WHERE created_utc >= ?",
                con, params=(since.timestamp(),),
            )
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
        return df

    def post_count_since(self, since: datetime) -> int:
        with self.conn() as con:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM reddit_posts WHERE created_utc >= ?",
                (since.timestamp(),),
            ).fetchone()
        return int(row["n"])

    def history_span_hours(self, within_days: float = 7.0) -> float:
        """Hours of RECENT Reddit history, measured inside the baseline window.

        Deliberately not the span of the whole table. Low-activity subreddits
        return months-old posts in their first page, so a total-span measure
        jumps to ~94 days after one collection and reports "velocity ready" on
        what is effectively a cold database. Velocity only ever looks back
        `baseline_days`, so readiness has to be measured over the same window.

        Pass within_days=0 for the raw span of everything stored.
        """
        with self.conn() as con:
            if within_days and within_days > 0:
                cutoff = (utcnow() - timedelta(days=within_days)).timestamp()
                row = con.execute(
                    "SELECT MIN(created_utc) AS lo, MAX(created_utc) AS hi "
                    "FROM reddit_posts WHERE created_utc >= ?", (cutoff,),
                ).fetchone()
            else:
                row = con.execute(
                    "SELECT MIN(created_utc) AS lo, MAX(created_utc) AS hi FROM reddit_posts"
                ).fetchone()
        if not row or row["lo"] is None:
            return 0.0
        return (row["hi"] - row["lo"]) / 3600.0

    # -- catalysts ----------------------------------------------------------
    def upsert_catalysts(self, rows: Iterable[dict[str, Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self.conn() as con:
            cur = con.executemany(
                """INSERT INTO catalysts
                   (symbol, source, source_name, event_type, title, url,
                    published_at, impact, direction, detail, fetched_at)
                   VALUES (:symbol,:source,:source_name,:event_type,:title,:url,
                           :published_at,:impact,:direction,:detail,:fetched_at)
                   ON CONFLICT(symbol, source, url, title) DO UPDATE SET
                     impact=excluded.impact,
                     direction=excluded.direction,
                     fetched_at=excluded.fetched_at""",
                rows,
            )
            return cur.rowcount

    def catalysts_since(self, since: datetime) -> pd.DataFrame:
        with self.conn() as con:
            return pd.read_sql_query(
                "SELECT * FROM catalysts WHERE published_at >= ? ORDER BY published_at DESC",
                con, params=(since.isoformat(),),
            )

    # -- scores / proposals / orders ---------------------------------------
    def save_scores(self, run_id: str, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        ts = iso()
        payload = [
            {
                "run_id": run_id, "ts": ts, "symbol": r["symbol"],
                "technical": r.get("technical"), "social": r.get("social"),
                "catalyst": r.get("catalyst"), "composite": r.get("composite"),
                "components": json.dumps(r.get("components", {}), default=str),
            }
            for r in rows
        ]
        with self.conn() as con:
            con.executemany(
                """INSERT INTO scores(run_id, ts, symbol, technical, social,
                        catalyst, composite, components)
                   VALUES (:run_id,:ts,:symbol,:technical,:social,:catalyst,
                           :composite,:components)""",
                payload,
            )

    def save_proposals(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.conn() as con:
            con.executemany(
                """INSERT OR REPLACE INTO proposals
                   (proposal_id, run_id, ts, symbol, side, rank, composite,
                    entry, stop, target, qty, notional, horizon, rationale,
                    payload, decision)
                   VALUES (:proposal_id,:run_id,:ts,:symbol,:side,:rank,:composite,
                           :entry,:stop,:target,:qty,:notional,:horizon,:rationale,
                           :payload,:decision)""",
                rows,
            )

    def set_decision(self, proposal_id: str, decision: str) -> None:
        with self.conn() as con:
            con.execute(
                "UPDATE proposals SET decision=? WHERE proposal_id=?", (decision, proposal_id)
            )

    def save_order(self, row: dict[str, Any]) -> None:
        with self.conn() as con:
            con.execute(
                """INSERT OR REPLACE INTO orders
                   (order_id, proposal_id, run_id, ts, venue, mode, symbol, side,
                    order_type, qty, price, status, exchange_ref, response)
                   VALUES (:order_id,:proposal_id,:run_id,:ts,:venue,:mode,:symbol,
                           :side,:order_type,:qty,:price,:status,:exchange_ref,:response)""",
                row,
            )

    def recent_orders(self, limit: int = 25) -> pd.DataFrame:
        with self.conn() as con:
            return pd.read_sql_query(
                "SELECT ts, mode, venue, symbol, side, order_type, qty, price, status "
                "FROM orders ORDER BY ts DESC LIMIT ?", con, params=(limit,),
            )

    def proposal_history(self, limit: int = 50) -> pd.DataFrame:
        with self.conn() as con:
            return pd.read_sql_query(
                "SELECT ts, symbol, side, rank, composite, entry, stop, target, "
                "notional, decision FROM proposals ORDER BY ts DESC LIMIT ?",
                con, params=(limit,),
            )

    # -- position metadata --------------------------------------------------
    def upsert_position_meta(self, row: dict[str, Any]) -> None:
        row = {**row}
        row.setdefault("updated_at", iso())
        row.setdefault("high_water", row.get("entry_price"))
        with self.conn() as con:
            con.execute(
                """INSERT INTO position_meta
                   (symbol, opened_at, entry_price, stop, target, horizon_days,
                    high_water, proposal_id, mode, updated_at)
                   VALUES (:symbol,:opened_at,:entry_price,:stop,:target,
                           :horizon_days,:high_water,:proposal_id,:mode,:updated_at)
                   ON CONFLICT(symbol) DO UPDATE SET
                     entry_price=excluded.entry_price,
                     -- Averaging up raises entry_price; leaving high_water
                     -- behind would make gain_from_entry negative and silently
                     -- disarm the trailing stop.
                     high_water=MAX(COALESCE(position_meta.high_water, 0),
                                    excluded.entry_price),
                     stop=excluded.stop,
                     target=excluded.target,
                     horizon_days=excluded.horizon_days,
                     proposal_id=excluded.proposal_id,
                     updated_at=excluded.updated_at""",
                row,
            )

    def position_meta(self, symbol: str | None = None) -> pd.DataFrame:
        with self.conn() as con:
            if symbol:
                return pd.read_sql_query(
                    "SELECT * FROM position_meta WHERE symbol=?", con, params=(symbol,))
            return pd.read_sql_query("SELECT * FROM position_meta", con)

    def bump_high_water(self, symbol: str, price: float) -> float:
        """Raise the high-water mark; never lowers it. Returns the mark in force."""
        with self.conn() as con:
            row = con.execute(
                "SELECT high_water FROM position_meta WHERE symbol=?", (symbol,)
            ).fetchone()
            if row is None:
                return price
            current = float(row["high_water"] or 0.0)
            if price > current:
                con.execute(
                    "UPDATE position_meta SET high_water=?, updated_at=? WHERE symbol=?",
                    (price, iso(), symbol),
                )
                return price
            return current

    def clear_position_target(self, symbol: str) -> None:
        """Retire the take-profit after a PARTIAL exit has taken it.

        Without this a partial take-profit re-fires on every run: the price is
        still above the target, so it sells another `take_profit_fraction` of
        what is left, halving the position indefinitely (100 -> 50 -> 25 -> ...)
        and paying a fee each time while never actually closing. Clearing the
        target means the take-profit fires once; the remainder then rides on the
        trailing stop, the hard stop and the horizon.
        """
        with self.conn() as con:
            con.execute(
                "UPDATE position_meta SET target=NULL, updated_at=? WHERE symbol=?",
                (iso(), symbol),
            )

    def delete_position_meta(self, symbol: str) -> None:
        with self.conn() as con:
            con.execute("DELETE FROM position_meta WHERE symbol=?", (symbol,))

    # -- funding rates ------------------------------------------------------
    def upsert_funding(self, rows: Iterable[dict[str, Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self.conn() as con:
            cur = con.executemany(
                """INSERT INTO funding_rates(symbol, funding_time, rate, venue, fetched_at)
                   VALUES (:symbol,:funding_time,:rate,:venue,:fetched_at)
                   ON CONFLICT(symbol, funding_time) DO UPDATE SET
                     rate=excluded.rate, fetched_at=excluded.fetched_at""",
                rows,
            )
            return cur.rowcount

    def funding_history(self, symbol: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM funding_rates"
        params: tuple = ()
        if symbol:
            q += " WHERE symbol = ?"
            params = (symbol,)
        with self.conn() as con:
            return pd.read_sql_query(q + " ORDER BY funding_time ASC", con, params=params)

    # -- source ranking -----------------------------------------------------
    def save_source_rank(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.conn() as con:
            con.execute("DELETE FROM source_rank")
            con.executemany(
                """INSERT INTO source_rank
                   (subreddit, computed_at, rank, value, mention_rate, assets,
                    tone, items_per_hour, n_items)
                   VALUES (:subreddit,:computed_at,:rank,:value,:mention_rate,
                           :assets,:tone,:items_per_hour,:n_items)""",
                rows,
            )

    def source_rank(self) -> pd.DataFrame:
        with self.conn() as con:
            return pd.read_sql_query(
                "SELECT * FROM source_rank ORDER BY rank ASC", con)

    def source_rank_age_hours(self) -> float | None:
        """Hours since the ranking was last computed, or None if never."""
        with self.conn() as con:
            row = con.execute("SELECT MAX(computed_at) AS t FROM source_rank").fetchone()
        if not row or not row["t"]:
            return None
        try:
            ts = datetime.fromisoformat(row["t"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return (utcnow() - ts).total_seconds() / 3600.0
        except Exception:  # noqa: BLE001 - unparseable = treat as stale
            return None

    # -- protective orders --------------------------------------------------
    def save_protective_order(self, row: dict[str, Any]) -> None:
        row = {**row}
        row.setdefault("updated_at", iso())
        row.setdefault("placed_at", iso())
        with self.conn() as con:
            con.execute(
                """INSERT INTO protective_orders
                   (symbol, kind, order_type, exchange_ref, order_list_id, qty,
                    stop_price, limit_price, target_price, trailing_delta,
                    status, mode, venue, placed_at, updated_at, response)
                   VALUES (:symbol,:kind,:order_type,:exchange_ref,:order_list_id,
                           :qty,:stop_price,:limit_price,:target_price,
                           :trailing_delta,:status,:mode,:venue,:placed_at,
                           :updated_at,:response)""",
                row,
            )

    def resting_orders(self, symbol: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM protective_orders WHERE status='resting'"
        params: tuple = ()
        if symbol:
            q += " AND symbol=?"
            params = (symbol,)
        with self.conn() as con:
            return pd.read_sql_query(q + " ORDER BY placed_at DESC", con, params=params)

    def mark_protective_cancelled(self, symbol: str) -> int:
        """Retire both live ('resting') and paper ('simulated') protection.

        Simulated rows are included so the table reflects reality after a
        position closes, rather than accumulating stale protection for
        positions that no longer exist.
        """
        with self.conn() as con:
            cur = con.execute(
                "UPDATE protective_orders SET status='cancelled', updated_at=? "
                "WHERE symbol=? AND status IN ('resting','simulated')", (iso(), symbol),
            )
            return cur.rowcount

    # -- paper book ---------------------------------------------------------
    def paper_cash(self) -> float:
        with self.conn() as con:
            return float(con.execute("SELECT balance FROM paper_cash WHERE id=1").fetchone()["balance"])

    def set_paper_cash(self, balance: float) -> None:
        with self.conn() as con:
            con.execute("UPDATE paper_cash SET balance=? WHERE id=1", (balance,))

    def paper_positions(self) -> pd.DataFrame:
        with self.conn() as con:
            return pd.read_sql_query("SELECT * FROM paper_positions", con)

    def apply_paper_fill(self, symbol: str, side: str, qty: float, price: float) -> None:
        """Average-cost bookkeeping for the simulated book."""
        with self.conn() as con:
            row = con.execute(
                "SELECT qty, avg_price FROM paper_positions WHERE symbol=?", (symbol,)
            ).fetchone()
            cur_qty = float(row["qty"]) if row else 0.0
            cur_avg = float(row["avg_price"]) if row else 0.0
            signed = qty if side.upper() == "BUY" else -qty
            new_qty = cur_qty + signed

            if side.upper() == "BUY":
                new_avg = ((cur_qty * cur_avg) + (qty * price)) / new_qty if new_qty > 0 else 0.0
            else:
                new_avg = cur_avg if new_qty > 1e-12 else 0.0

            if abs(new_qty) < 1e-12:
                con.execute("DELETE FROM paper_positions WHERE symbol=?", (symbol,))
            else:
                con.execute(
                    """INSERT INTO paper_positions(symbol, qty, avg_price, updated_at)
                       VALUES (?,?,?,?)
                       ON CONFLICT(symbol) DO UPDATE SET
                         qty=excluded.qty, avg_price=excluded.avg_price,
                         updated_at=excluded.updated_at""",
                    (symbol, new_qty, new_avg, iso()),
                )
            cash = float(con.execute("SELECT balance FROM paper_cash WHERE id=1").fetchone()["balance"])
            con.execute(
                "UPDATE paper_cash SET balance=? WHERE id=1",
                (cash - signed * price,),
            )
