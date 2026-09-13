#!/usr/bin/env python3
"""
Signal log - every scored symbol on every 5m bar, labelled after the fact.

The executed-trade record is tiny (a handful of round trips a day), so it can
never resolve whether live precision matches the backtest. The signal log can:
FastTrader.cycle writes one row per scored symbol per bar, and the 16:05 ET
report labels each row with the SAME triple-barrier rule the model was trained
on. That gives ~40 labelled signals a day, enough for a real confidence
interval by the review date.

Connection ownership. record() and mark_executed() run on the fast cycle's
thread, sequentially with BudgetTracker._txn(), so they may share
budget.conn. label_pending() runs on the 16:05 report's worker thread while
the cycle may be mid-transaction on budget.conn (crypto keeps the cycle alive
after the bell); its `with conn:` commit would then commit the cycle's
half-finished trade. Callers pass it a connection BudgetTracker does not own
(IntradayEngine.conn, or a fresh connect(path)).
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.intraday_engine import EARLY_CLOSE_2026, ET, INTERVAL, barriers
from src.labeling import triple_barrier

logger = logging.getLogger(__name__)


def _now_iso(now=None) -> str:
    """UTC ISO seconds; `now` is injected by tests, wall clock otherwise."""
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(timespec='seconds')


def _et_date(ts_iso: str) -> str:
    """ET calendar date of a tz-aware ISO timestamp (the report's "today")."""
    return datetime.fromisoformat(ts_iso).astimezone(ET).strftime('%Y-%m-%d')


def record(conn, signals, now=None) -> int:
    """Insert scan_all() output; returns the number of rows actually inserted.

    INSERT OR IGNORE on (symbol, bar_ts): the fast loop polls every 60 s but a
    5m bar is current for five polls, so without this n would be inflated 5x
    by duplicates of the same signal. `now` is accepted for signature symmetry
    with the other clock-taking functions; the row is stamped by its bar, not
    by the wall clock.
    """
    rows = [(s['bar_ts'], s['symbol'], s['asset_class'], float(s['probability']),
             float(s['bar']), int(bool(s['above_bar'])), float(s['price']),
             _et_date(s['bar_ts']))
            for s in signals]
    if not rows:
        return 0
    with conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO signals "
            "(bar_ts, symbol, asset_class, probability, bar, above_bar, ref_price, trade_date) "
            "VALUES (?,?,?,?,?,?,?,?)", rows)
        inserted = conn.total_changes - before
    if inserted:
        logger.debug(f"[signals] recorded {inserted} of {len(rows)} scored symbols")
    return inserted


def mark_executed(conn, symbol: str, bar_ts: str, trade_id: int) -> None:
    """Link a signal row to the BUY trade it produced."""
    with conn:
        conn.execute("UPDATE signals SET executed_trade_id = ? WHERE symbol = ? AND bar_ts = ?",
                     (trade_id, symbol, bar_ts))


def _forward_bars(conn, symbol: str, bar_ts: str, n_bars: int) -> pd.DataFrame:
    """The signal's own bar plus the next n_bars-1 stored bars, UTC-indexed.

    prices_intraday.ts is stored exactly as yfinance served it, which for this
    universe is an ET offset ('...-04:00' / '...-05:00'), while bar_ts is UTC
    ('...+00:00'). A plain string comparison of the two is wrong (15:55-04:00
    sorts BEFORE 19:55+00:00 although they are the same instant), so both
    sides go through SQLite's datetime(), which normalises any offset to UTC.
    """
    df = pd.read_sql_query(
        "SELECT ts, high, low, close FROM prices_intraday "
        "WHERE symbol = ? AND interval = ? AND datetime(ts) >= datetime(?) "
        "ORDER BY datetime(ts) LIMIT ?",
        conn, params=(symbol, INTERVAL, bar_ts, int(n_bars)))
    if df.empty:
        return df
    df['ts'] = pd.to_datetime(df['ts'], utc=True, format='ISO8601')
    return df.set_index('ts')


def _bar_minutes() -> float:
    """Bar length in minutes from INTERVAL: '5m' -> 5, '15m' -> 15, '1h' -> 60."""
    s = INTERVAL.strip().lower()
    return float(s[:-1]) * (60 if s.endswith('h') else 1)


def _session_close(trade_date: str) -> datetime:
    """ET close of the stock session on an ET date (13:00 on early-close days)."""
    hour = 13 if trade_date in EARLY_CLOSE_2026 else 16
    return datetime.strptime(trade_date, '%Y-%m-%d').replace(hour=hour, tzinfo=ET)


def _session_complete(bars: pd.DataFrame, trade_date: str, now: datetime) -> bool:
    """True once a stock signal's session is over AND its closing bar is stored.

    Both conditions matter. `now` past the close says the session can no
    longer resolve the trade. The closing bar (the one that starts one
    interval before the close, 15:55 ET on a regular day) being present says
    the fetch ran after the close, so a feed outage that stopped bars at 15:30
    is not read as "nothing happened until the bell".
    """
    close = _session_close(trade_date)
    if now.astimezone(ET) < close:
        return False
    et = bars.index.tz_convert(ET)
    in_session = et[et.strftime('%Y-%m-%d') == trade_date]
    return (len(in_session) > 0 and
            in_session[-1].to_pydatetime() >= close - timedelta(minutes=_bar_minutes()))


def label_pending(conn, now=None) -> int:
    """Label every signal whose outcome is decided; returns the count labelled.

    Same rule as training (build_target / tune.py dataset()): the class's tuned
    barriers, and for stocks the walk stops at the ET session boundary because
    the executor flattens before the bell. A stock label therefore never
    depends on the next session's bars, so once the session is over (and its
    closing bar is stored) an unresolved stock signal is 0 the same day - the
    16:05 tick labels nearly every stock row from that day. Crypto runs the
    full horizon; a crypto window that runs past the last stored bar stays
    NULL and is retried on the next call.

    `conn` must not be BudgetTracker.conn: this runs on the report's worker
    thread and its commit would land on whatever transaction the fast cycle
    has open on that connection (see the module docstring).
    """
    now = now or datetime.now(timezone.utc)
    pending = conn.execute(
        "SELECT id, symbol, asset_class, bar_ts, trade_date FROM signals "
        "WHERE label IS NULL ORDER BY symbol, bar_ts").fetchall()
    if not pending:
        return 0
    stamp = _now_iso(now)
    labelled = 0
    for row in pending:
        cls = row['asset_class']
        tp, sl, horizon = barriers(cls)
        bars = _forward_bars(conn, row['symbol'], row['bar_ts'], horizon + 1)
        if bars.empty:
            continue                      # nothing stored from the signal's bar on
        if bars.index[0] != pd.Timestamp(row['bar_ts']):
            # The signal's own bar is missing from the store, so bars[0] is a
            # later bar and its close is not the entry price. Never label
            # against the wrong entry; leave it NULL and say so.
            logger.warning(f"[signals] {row['symbol']} {row['bar_ts']}: signal bar not "
                           f"stored (first stored bar {bars.index[0].isoformat()})")
            continue
        if cls == 'crypto':
            session = None
            if len(bars) < horizon + 1:
                continue                  # window not fully stored yet
        else:
            session = bars.index.tz_convert(ET).strftime('%Y-%m-%d')
            if len(bars) < horizon + 1 and not _session_complete(bars, row['trade_date'], now):
                continue                  # the session may still resolve it
        # Fewer than horizon+1 bars reaches here only for a stock whose session
        # is over: the walk then runs to the last stored bar, which is exactly
        # where the session cut would have stopped it anyway.
        label = triple_barrier(bars['high'], bars['low'], bars['close'], tp, sl,
                               min(horizon, len(bars) - 1), session=session).iloc[0]
        if pd.isna(label):
            continue
        with conn:
            conn.execute("UPDATE signals SET label = ?, labeled_at = ? WHERE id = ?",
                         (int(label), stamp, row['id']))
        labelled += 1
    logger.info(f"[signals] labelled {labelled} of {len(pending)} pending signals")
    return labelled
