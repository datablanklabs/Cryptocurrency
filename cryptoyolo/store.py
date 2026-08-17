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

CREATE TABLE IF NOT EXISTS reddit_posts (
    post_id     TEXT PRIMARY KEY,
    subreddit   TEXT NOT NULL,
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
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL")
            yield con
            con.commit()
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self.conn() as con:
            con.executescript(SCHEMA)
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
                   (post_id, subreddit, kind, author, created_utc, score,
                    num_comments, title, body, permalink, fetched_at)
                   VALUES (:post_id,:subreddit,:kind,:author,:created_utc,:score,
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
                   (post_id, symbol, subreddit, author, created_utc,
                    sentiment, confidence, weight, matched_on)
                   VALUES (:post_id,:symbol,:subreddit,:author,:created_utc,
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

    def history_span_hours(self) -> float:
        """How much Reddit history we've accumulated. Velocity is meaningless
        until this is comfortably larger than the velocity window."""
        with self.conn() as con:
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
