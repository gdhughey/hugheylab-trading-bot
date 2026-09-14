"""
Shared fixtures.

Every test uses a FILE database under tmp_path. dev/ct-test.sh runs pytest with
DB_PATH=:memory:, and each connect(':memory:') would be a separate empty
database, so the same path must be passed explicitly to every component
(Database, BudgetTracker, engines).

Clock rule: Database() seeds account.opened_at from `now` or the wall clock,
and every ledger read filters created_at >= opened_at. A test that constructs
Database() itself MUST pass now= (earlier than every timestamp it uses) or
pin opened_at with an UPDATE right after - never rely on the wall clock, or
the test turns into a time bomb the day the suite runs later than its data.
"""

import sqlite3
from datetime import datetime, timezone

import pytest

from src.database import Database

# Fixed clock for the account seed so account.opened_at never depends on wall
# time. Early enough that any 2026 `now=` a test passes is >= opened_at, which
# is what the "created_at >= opened_at" report filters need.
OPENED_AT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# The schema exactly as shipped BEFORE the paper-brokerage change (git dd28b31):
# INTEGER shares, no account/day_state/equity_history/signals tables, none of
# the new trade/position columns. Kept verbatim so test_migration exercises the
# real upgrade path a production database will take.
OLD_SCHEMA = """
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

# Twenty-one legacy trades (spec section 5): nine closed round trips (BUY then
# SELL, with the realized_pnl the old code wrote) and three rejected BUYs,
# spread over two ISO weeks. Ids are assigned in list order (1..21), which the
# migration tests rely on: id 1 is the first BUY, id 2 the SELL that closed it.
LEGACY_TRADES = [
    # (symbol, side, price, shares, amount, status, created_at, settled_at, week_key, realized_pnl)
    ('AAPL', 'BUY',  150.0, 2, 300.0, 'EXECUTED', '2026-09-01T14:35:00+00:00', '2026-09-01T14:35:05+00:00', '2026-W36', None),   # 1
    ('AAPL', 'SELL', 155.0, 2, 310.0, 'EXECUTED', '2026-09-02T15:00:00+00:00', '2026-09-02T15:00:05+00:00', '2026-W36', 10.0),   # 2
    ('MSFT', 'BUY',  400.0, 1, 400.0, 'REJECTED', '2026-09-03T14:40:00+00:00', '2026-09-03T14:41:00+00:00', '2026-W36', None),   # 3
    ('NVDA', 'BUY',  120.0, 3, 360.0, 'EXECUTED', '2026-09-03T15:10:00+00:00', '2026-09-03T15:10:05+00:00', '2026-W36', None),   # 4
    ('NVDA', 'SELL', 118.0, 3, 354.0, 'EXECUTED', '2026-09-03T19:50:00+00:00', '2026-09-03T19:50:05+00:00', '2026-W36', -6.0),   # 5
    ('AMD',  'BUY',  160.0, 2, 320.0, 'EXECUTED', '2026-09-04T14:35:00+00:00', '2026-09-04T14:35:05+00:00', '2026-W36', None),   # 6
    ('AMD',  'SELL', 164.0, 2, 328.0, 'EXECUTED', '2026-09-04T17:20:00+00:00', '2026-09-04T17:20:05+00:00', '2026-W36', 8.0),    # 7
    ('GOOG', 'BUY',  200.0, 1, 200.0, 'REJECTED', '2026-09-04T18:00:00+00:00', '2026-09-04T18:05:00+00:00', '2026-W36', None),   # 8
    ('META', 'BUY',  500.0, 1, 500.0, 'EXECUTED', '2026-09-08T14:35:00+00:00', '2026-09-08T14:35:05+00:00', '2026-W37', None),   # 9
    ('META', 'SELL', 495.0, 1, 495.0, 'EXECUTED', '2026-09-08T19:55:00+00:00', '2026-09-08T19:55:05+00:00', '2026-W37', -5.0),   # 10
    ('AMZN', 'BUY',  180.0, 2, 360.0, 'EXECUTED', '2026-09-09T14:35:00+00:00', '2026-09-09T14:35:05+00:00', '2026-W37', None),   # 11
    ('AMZN', 'SELL', 183.0, 2, 366.0, 'EXECUTED', '2026-09-09T16:40:00+00:00', '2026-09-09T16:40:05+00:00', '2026-W37', 6.0),    # 12
    ('TSLA', 'BUY',  250.0, 1, 250.0, 'EXECUTED', '2026-09-09T17:00:00+00:00', '2026-09-09T17:00:05+00:00', '2026-W37', None),   # 13
    ('TSLA', 'SELL', 245.0, 1, 245.0, 'EXECUTED', '2026-09-10T14:45:00+00:00', '2026-09-10T14:45:05+00:00', '2026-W37', -5.0),   # 14
    ('NFLX', 'BUY',  700.0, 1, 700.0, 'REJECTED', '2026-09-10T15:00:00+00:00', '2026-09-10T15:05:00+00:00', '2026-W37', None),   # 15
    ('AAPL', 'BUY',  152.0, 2, 304.0, 'EXECUTED', '2026-09-10T15:30:00+00:00', '2026-09-10T15:30:05+00:00', '2026-W37', None),   # 16
    ('AAPL', 'SELL', 156.0, 2, 312.0, 'EXECUTED', '2026-09-10T19:50:00+00:00', '2026-09-10T19:50:05+00:00', '2026-W37', 8.0),    # 17
    ('MSFT', 'BUY',  405.0, 1, 405.0, 'EXECUTED', '2026-09-11T14:35:00+00:00', '2026-09-11T14:35:05+00:00', '2026-W37', None),   # 18
    ('MSFT', 'SELL', 407.0, 1, 407.0, 'EXECUTED', '2026-09-11T16:10:00+00:00', '2026-09-11T16:10:05+00:00', '2026-W37', 2.0),    # 19
    ('NVDA', 'BUY',  121.0, 3, 363.0, 'EXECUTED', '2026-09-11T16:30:00+00:00', '2026-09-11T16:30:05+00:00', '2026-W37', None),   # 20
    ('NVDA', 'SELL', 119.0, 3, 357.0, 'EXECUTED', '2026-09-11T19:55:00+00:00', '2026-09-11T19:55:05+00:00', '2026-W37', -6.0),   # 21
]

# Derived from LEGACY_TRADES; the migration tests assert these survive the upgrade.
LEGACY_TRADE_COUNT = len(LEGACY_TRADES)                                   # 21
LEGACY_EXECUTED_COUNT = sum(1 for t in LEGACY_TRADES if t[5] == 'EXECUTED')  # 18
LEGACY_REALIZED_TOTAL = sum(t[9] for t in LEGACY_TRADES if t[9] is not None)  # 12.0

# Two flat position rows - the live positions table is flat at account open.
LEGACY_POSITIONS = [
    # (symbol, shares, avg_price, updated_at)
    ('AAPL', 0, 152.0, '2026-09-10T19:50:05+00:00'),
    ('MSFT', 0, 0.0,   '2026-09-11T16:10:05+00:00'),
]


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """A fresh, fully migrated file DB with a $500 cash account opened at OPENED_AT.

    STARTING_CASH / ACCOUNT_TYPE are cleared so the seed uses the code defaults
    regardless of the shell environment on the container. A test that needs a
    margin account or a different balance sets the env and constructs its own
    Database(path, now=...) on a different path.
    """
    monkeypatch.delenv('STARTING_CASH', raising=False)
    monkeypatch.delenv('ACCOUNT_TYPE', raising=False)
    path = str(tmp_path / 'test.db')
    Database(path, now=OPENED_AT).close()
    return path


@pytest.fixture
def old_db_path(tmp_path):
    """A file DB written with the PRE-change schema and 21 legacy trades.

    Database() is deliberately NOT constructed here: the test under migration
    constructs it (with its own now=) so it can observe the upgrade.
    """
    path = str(tmp_path / 'old.db')
    conn = sqlite3.connect(path)
    try:
        conn.executescript(OLD_SCHEMA)
        conn.executemany(
            "INSERT INTO trades (symbol, side, price, shares, amount, status, "
            "created_at, settled_at, week_key, realized_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            LEGACY_TRADES,
        )
        conn.executemany(
            "INSERT INTO positions (symbol, shares, avg_price, updated_at) VALUES (?, ?, ?, ?)",
            LEGACY_POSITIONS,
        )
        conn.commit()
    finally:
        conn.close()
    return path
