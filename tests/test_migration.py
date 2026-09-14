"""
Migration tests: a file DB written with the pre-change schema (tests/conftest.py
OLD_SCHEMA) upgrades in place when Database() opens it, and a fresh DB gets the
same final shape from SCHEMA alone.

Every Database() call here passes now=NOW; nothing depends on wall time.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

from src.database import Database, connect
from tests.conftest import (LEGACY_EXECUTED_COUNT, LEGACY_REALIZED_TOTAL,
                            LEGACY_TRADE_COUNT)

# Fixed clock passed to every Database() here; nothing depends on wall time.
NOW = datetime(2026, 9, 14, 13, 30, 0, tzinfo=timezone.utc)
NOW_ISO = '2026-09-14T13:30:00+00:00'

NEW_TABLES = {'account', 'day_state', 'equity_history', 'signals'}

# column -> declared type, as PRAGMA table_info reports it
NEW_TRADE_COLS = {
    'ref_price': 'REAL',
    'fees': 'REAL',
    'gross_pnl': 'REAL',
    'available_at': 'TEXT',
    'trade_date': 'TEXT',
    'entry_probability': 'REAL',
    'exit_reason': 'TEXT',
}


def _cols(conn, table) -> dict:
    """{column name: declared type} for one table."""
    return {r['name']: r['type'] for r in conn.execute(f"PRAGMA table_info({table})")}


def _names(conn, kind) -> set:
    """Names of every table or index in sqlite_master."""
    return {r['name'] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = ?", (kind,))}


def test_old_db_gains_new_columns_and_tables(old_db_path):
    Database(old_db_path, now=NOW).close()
    conn = connect(old_db_path)
    try:
        trades = _cols(conn, 'trades')
        for col, typ in NEW_TRADE_COLS.items():
            assert trades.get(col) == typ, f"trades.{col} missing or wrong type"
        assert _cols(conn, 'positions').get('entry_ref') == 'REAL'
        assert NEW_TABLES <= _names(conn, 'table')
        assert 'idx_signals_label' in _names(conn, 'index')
        # All 21 legacy rows survive untouched and pick up the fees default.
        assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == LEGACY_TRADE_COUNT
        assert conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status = 'EXECUTED'").fetchone()[0] == LEGACY_EXECUTED_COUNT
        assert conn.execute(
            "SELECT realized_pnl FROM trades WHERE id = 2").fetchone()[0] == 10.0
        assert conn.execute(
            "SELECT SUM(realized_pnl) FROM trades").fetchone()[0] == LEGACY_REALIZED_TOTAL
        assert conn.execute(
            "SELECT COUNT(*) FROM trades WHERE fees = 0").fetchone()[0] == LEGACY_TRADE_COUNT
        assert conn.execute(
            "SELECT COUNT(*) FROM trades WHERE ref_price IS NULL AND available_at IS NULL "
            "AND trade_date IS NULL").fetchone()[0] == LEGACY_TRADE_COUNT
        assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 2
    finally:
        conn.close()


def test_fresh_db_declares_real_shares_and_full_shape(db_path):
    conn = connect(db_path)
    try:
        assert _cols(conn, 'trades')['shares'] == 'REAL'
        assert _cols(conn, 'positions')['shares'] == 'REAL'
        assert set(NEW_TRADE_COLS) <= set(_cols(conn, 'trades'))
        assert 'entry_ref' in _cols(conn, 'positions')
        assert NEW_TABLES <= _names(conn, 'table')
        assert 'idx_signals_label' in _names(conn, 'index')
    finally:
        conn.close()


def test_migrated_integer_shares_column_stores_fraction(old_db_path):
    Database(old_db_path, now=NOW).close()
    conn = connect(old_db_path)
    try:
        # Not rebuilt: still INTEGER-declared. SQLite affinity keeps 0.5 as REAL.
        assert _cols(conn, 'trades')['shares'] == 'INTEGER'
        assert _cols(conn, 'positions')['shares'] == 'INTEGER'
        # TSLA also has legacy whole-share rows (ids 13-14), so read back the
        # new trade by its id rather than by symbol.
        with conn:
            trade_id = conn.execute(
                "INSERT INTO trades (symbol, side, price, shares, amount, status, created_at, week_key) "
                "VALUES ('TSLA', 'BUY', 100.0, 0.5, 50.0, 'EXECUTED', ?, '2026-W38')",
                (NOW_ISO,)).lastrowid
            conn.execute(
                "INSERT INTO positions (symbol, shares, avg_price, updated_at) "
                "VALUES ('TSLA', 0.5, 100.0, ?)", (NOW_ISO,))
        t = conn.execute("SELECT shares FROM trades WHERE id = ?", (trade_id,)).fetchone()['shares']
        p = conn.execute("SELECT shares FROM positions WHERE symbol = 'TSLA'").fetchone()['shares']
        assert t == 0.5 and isinstance(t, float)
        assert p == 0.5 and isinstance(p, float)
    finally:
        conn.close()


def test_realized_pnl_and_source_migrations_still_run(tmp_path):
    # An even older DB: trades without realized_pnl, prices without source.
    path = str(tmp_path / 'older.db')
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE prices (symbol TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL,
            low REAL, close REAL, volume REAL, PRIMARY KEY (symbol, date));
        CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
            side TEXT NOT NULL, price REAL NOT NULL, shares INTEGER NOT NULL, amount REAL NOT NULL,
            status TEXT NOT NULL, created_at TEXT NOT NULL, settled_at TEXT, week_key TEXT NOT NULL);
        CREATE TABLE positions (symbol TEXT PRIMARY KEY, shares INTEGER NOT NULL DEFAULT 0,
            avg_price REAL NOT NULL DEFAULT 0, updated_at TEXT);
    """)
    conn.close()
    Database(path, now=NOW).close()
    conn = connect(path)
    try:
        assert _cols(conn, 'trades').get('realized_pnl') == 'REAL'
        assert _cols(conn, 'prices').get('source') == 'TEXT'
        assert set(NEW_TRADE_COLS) <= set(_cols(conn, 'trades'))
    finally:
        conn.close()


def test_account_seeded_once_from_env(old_db_path, monkeypatch):
    monkeypatch.setenv('STARTING_CASH', '500')
    monkeypatch.delenv('ACCOUNT_TYPE', raising=False)   # default must be 'cash'
    Database(old_db_path, now=NOW).close()
    conn = connect(old_db_path)
    try:
        rows = conn.execute("SELECT * FROM account").fetchall()
        assert len(rows) == 1
        acct = rows[0]
        assert acct['id'] == 1
        assert acct['opened_at'] == NOW_ISO
        assert acct['starting_cash'] == 500.0
        assert acct['cash'] == 500.0
        assert acct['account_type'] == 'cash'
        # Every legacy trade predates opened_at, so the ledger's
        # created_at >= opened_at filters (Task 4) will exclude all 21.
        assert conn.execute(
            "SELECT COUNT(*) FROM trades WHERE created_at >= ?", (acct['opened_at'],)
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_account_type_margin_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv('STARTING_CASH', '1000')
    monkeypatch.setenv('ACCOUNT_TYPE', 'margin')
    path = str(tmp_path / 'margin.db')
    Database(path, now=NOW).close()
    conn = connect(path)
    try:
        acct = conn.execute("SELECT * FROM account").fetchone()
        assert acct['starting_cash'] == 1000.0
        assert acct['account_type'] == 'margin'
    finally:
        conn.close()


def test_second_construction_keeps_one_account_row(old_db_path, monkeypatch):
    monkeypatch.setenv('STARTING_CASH', '500')
    monkeypatch.delenv('ACCOUNT_TYPE', raising=False)
    Database(old_db_path, now=NOW).close()
    # A later env edit and a later clock must NOT reopen or restate the account:
    # changing starting cash after open would corrupt the all-time return.
    monkeypatch.setenv('STARTING_CASH', '999')
    Database(old_db_path, now=NOW + timedelta(days=1)).close()
    conn = connect(old_db_path)
    try:
        rows = conn.execute("SELECT * FROM account").fetchall()
        assert len(rows) == 1
        assert rows[0]['starting_cash'] == 500.0
        assert rows[0]['cash'] == 500.0
        assert rows[0]['opened_at'] == NOW_ISO
    finally:
        conn.close()


def test_entry_ref_backfilled_for_open_position(old_db_path):
    # An open lot written by the old code has avg_price but no entry_ref. The
    # ref-to-ref barrier test divides by entry_ref, so it must not stay 0.
    conn = sqlite3.connect(old_db_path)
    with conn:
        conn.execute(
            "INSERT INTO positions (symbol, shares, avg_price, updated_at) "
            "VALUES ('NVDA', 3, 120.0, '2026-09-11T19:55:00+00:00')")
    conn.close()

    Database(old_db_path, now=NOW).close()
    conn = connect(old_db_path)
    try:
        refs = {r['symbol']: r['entry_ref']
                for r in conn.execute("SELECT symbol, entry_ref FROM positions")}
        # Flat rows stay 0; the open lot takes its avg_price.
        assert refs == {'AAPL': 0.0, 'MSFT': 0.0, 'NVDA': 120.0}

        # A non-zero entry_ref is never overwritten by a later start.
        with conn:
            conn.execute("UPDATE positions SET entry_ref = 118.0 WHERE symbol = 'NVDA'")
    finally:
        conn.close()
    Database(old_db_path, now=NOW + timedelta(days=1)).close()
    conn = connect(old_db_path)
    try:
        assert conn.execute(
            "SELECT entry_ref FROM positions WHERE symbol = 'NVDA'").fetchone()[0] == 118.0
    finally:
        conn.close()
