#!/usr/bin/env python3
"""
Database layer - owns the SQLite schema shared by the ML engine and budget tracker.
"""

import os
import sqlite3
import logging
from datetime import datetime, timezone
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
    price       REAL    NOT NULL,          -- simulated fill (ref +/- slippage or spread)
    shares      REAL    NOT NULL,          -- fractional, rounded to 6 dp
    amount      REAL    NOT NULL,          -- net cash movement: gross + fees (BUY) | gross - fees (SELL)
    status      TEXT    NOT NULL,          -- PENDING | EXECUTED | REJECTED
    created_at  TEXT    NOT NULL,
    settled_at  TEXT,                      -- status-decided timestamp (NOT cash settlement)
    week_key    TEXT    NOT NULL,          -- legacy ISO year-week; still stamped, no longer read
    realized_pnl REAL,                     -- SELL rows: net of fees; NULL for BUY
    ref_price   REAL,                      -- the quote the caller passed, before costs
    fees        REAL    NOT NULL DEFAULT 0,
    gross_pnl   REAL,                      -- SELL rows: realized_pnl + fees
    available_at TEXT,                     -- SELL rows: UTC ISO when the proceeds settle
    trade_date  TEXT,                      -- ET calendar date of created_at
    entry_probability REAL,                -- BUY rows: model probability at entry
    exit_reason TEXT                       -- SELL rows: tp | sl | timeout | eod | manual
);

CREATE INDEX IF NOT EXISTS idx_trades_week   ON trades (week_key, status);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);

CREATE TABLE IF NOT EXISTS positions (
    symbol      TEXT PRIMARY KEY,
    shares      REAL    NOT NULL DEFAULT 0,
    avg_price   REAL    NOT NULL DEFAULT 0, -- net cost basis per share (from amount)
    entry_ref   REAL    NOT NULL DEFAULT 0, -- qty-weighted ref_price of the open lots; barriers test against this
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS account (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    opened_at     TEXT NOT NULL,           -- UTC ISO; every report filters created_at >= opened_at
    starting_cash REAL NOT NULL,
    cash          REAL NOT NULL,
    account_type  TEXT NOT NULL            -- cash | margin
);

CREATE TABLE IF NOT EXISTS day_state (
    date              TEXT PRIMARY KEY,    -- ET date
    start_equity      REAL NOT NULL,
    loss_tripped_at   TEXT,
    loss_announced_at TEXT,
    report_posted_at  TEXT
);

CREATE TABLE IF NOT EXISTS equity_history (
    date             TEXT PRIMARY KEY,     -- ET date; written by the 16:05 ET report tick only
    cash             REAL NOT NULL,
    positions_value  REAL NOT NULL,
    equity           REAL NOT NULL,
    fees_to_date     REAL NOT NULL,
    realized_to_date REAL NOT NULL,
    recorded_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_ts            TEXT NOT NULL,       -- UTC ISO of the 5m bar scored
    symbol            TEXT NOT NULL,
    asset_class       TEXT NOT NULL,
    probability       REAL NOT NULL,
    bar               REAL NOT NULL,
    above_bar         INTEGER NOT NULL,
    ref_price         REAL NOT NULL,       -- bar close
    trade_date        TEXT NOT NULL,       -- ET date
    executed_trade_id INTEGER,             -- BUY trade id when this signal was traded
    label             INTEGER,             -- triple-barrier outcome; NULL until labelled
    labeled_at        TEXT,
    UNIQUE (symbol, bar_ts)
);

CREATE INDEX IF NOT EXISTS idx_signals_label ON signals (label, symbol);
"""

# (table, column, declaration) for every column added after its table first
# shipped. Each entry is applied only when PRAGMA table_info lacks the column,
# so the list is safe to run on every start. A NOT NULL addition needs a
# DEFAULT or SQLite refuses the ALTER on a populated table.
COLUMN_MIGRATIONS = [
    ('trades', 'realized_pnl', 'REAL'),
    ('prices', 'source', 'TEXT'),
    ('trades', 'ref_price', 'REAL'),
    ('trades', 'fees', 'REAL NOT NULL DEFAULT 0'),
    ('trades', 'gross_pnl', 'REAL'),
    ('trades', 'available_at', 'TEXT'),
    ('trades', 'trade_date', 'TEXT'),
    ('trades', 'entry_probability', 'REAL'),
    ('trades', 'exit_reason', 'TEXT'),
    ('positions', 'entry_ref', 'REAL NOT NULL DEFAULT 0'),
]


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

    def __init__(self, db_path: str = None, now: datetime = None):
        self.db_path = db_path or DB_PATH
        self.conn = connect(self.db_path)
        self.init_schema(now=now)
        logger.info(f"Database ready at {self.db_path}")

    def init_schema(self, now: datetime = None):
        with self.conn:
            self.conn.executescript(SCHEMA)
        self._migrate(now=now)

    def _migrate(self, now: datetime = None):
        """Additive migrations for databases created by an earlier version.

        `now` (tz-aware UTC) is only consulted the first time the account row
        is seeded; tests pass it so opened_at is deterministic.
        """
        for table, column, decl in COLUMN_MIGRATIONS:
            cols = {r['name'] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                with self.conn:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                logger.info(f"Migrated: added {table}.{column}")

        # Lots opened before entry_ref existed have only avg_price. The barrier
        # test measures ref-to-ref against entry_ref (and divides by it), so
        # give those lots their cost basis as the reference instead of 0. Only
        # rows still at 0 are touched, so this is idempotent across restarts.
        with self.conn:
            cur = self.conn.execute(
                "UPDATE positions SET entry_ref = avg_price WHERE shares > 0 AND entry_ref = 0")
        if cur.rowcount:
            logger.info(f"Migrated: backfilled entry_ref on {cur.rowcount} open position(s)")

        # Seed the single account row. INSERT OR IGNORE means a later change to
        # STARTING_CASH / ACCOUNT_TYPE never touches an opened account: there is
        # deliberately no setter, because restating starting cash would corrupt
        # the all-time return every report is built on.
        # Normalise to UTC: every ledger filter is a lexical compare against
        # UTC ISO stamps, so a non-UTC offset here would silently exclude rows.
        opened_at = ((now or datetime.now(timezone.utc)).astimezone(timezone.utc)
                     .isoformat(timespec='seconds'))
        starting_cash = float(os.getenv('STARTING_CASH', 500))
        account_type = os.getenv('ACCOUNT_TYPE', 'cash').strip().lower()
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO account (id, opened_at, starting_cash, cash, account_type) "
                "VALUES (1, ?, ?, ?, ?)",
                (opened_at, starting_cash, starting_cash, account_type),
            )
        if cur.rowcount:
            logger.info(
                f"Opened paper account: ${starting_cash:,.2f} ({account_type}) at {opened_at}")

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
