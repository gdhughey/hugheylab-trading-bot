#!/usr/bin/env python3
"""
Database layer - owns the SQLite schema shared by the ML engine and budget tracker.
"""

import os
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = os.getenv('DB_PATH', 'data/trading_bot.db')

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    source      TEXT,                       -- which provider supplied this bar
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL,          -- BUY | SELL
    price       REAL    NOT NULL,
    shares      INTEGER NOT NULL,
    amount      REAL    NOT NULL,
    status      TEXT    NOT NULL,          -- PENDING | EXECUTED | REJECTED
    created_at  TEXT    NOT NULL,
    settled_at  TEXT,
    week_key    TEXT    NOT NULL,          -- ISO year-week, for weekly budget rollover
    realized_pnl REAL                       -- set on SELL execution; NULL for BUY
);

CREATE INDEX IF NOT EXISTS idx_trades_week   ON trades (week_key, status);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);

CREATE TABLE IF NOT EXISTS positions (
    symbol      TEXT PRIMARY KEY,
    shares      INTEGER NOT NULL DEFAULT 0,
    avg_price   REAL    NOT NULL DEFAULT 0,
    updated_at  TEXT
);
"""


def connect(db_path: str = None) -> sqlite3.Connection:
    """Open a connection with sane concurrency defaults."""
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


class Database:
    """Owns schema creation. Other components open their own connections."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self.conn = connect(self.db_path)
        self.init_schema()
        logger.info(f"Database ready at {self.db_path}")

    def init_schema(self):
        with self.conn:
            self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self):
        """Additive migrations for databases created by an earlier version."""
        cols = {r['name'] for r in self.conn.execute("PRAGMA table_info(trades)")}
        if 'realized_pnl' not in cols:
            with self.conn:
                self.conn.execute("ALTER TABLE trades ADD COLUMN realized_pnl REAL")
            logger.info("Migrated: added trades.realized_pnl")

        pcols = {r['name'] for r in self.conn.execute("PRAGMA table_info(prices)")}
        if 'source' not in pcols:
            with self.conn:
                self.conn.execute("ALTER TABLE prices ADD COLUMN source TEXT")
            logger.info("Migrated: added prices.source")

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
