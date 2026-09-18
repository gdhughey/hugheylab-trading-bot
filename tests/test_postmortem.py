"""src.postmortem: pairing, forensics and pattern counting for a losing day.

Fixed clock: Wed 2026-09-17. Trades mirror the real ledger that day (four
gap-ups bought at 09:31, all stopped out by 09:41).
"""
from datetime import datetime, timezone

import pytest

from src.budget_tracker import BudgetTracker
from src.collector import ensure_schema
from src.database import Database
from src.intraday_engine import upsert_bars
from src.postmortem import build_postmortem_context, pair_trades

import pandas as pd

OPENED = datetime(2026, 9, 1, tzinfo=timezone.utc)
DAY = '2026-09-17'
T_BUY = datetime(2026, 9, 17, 13, 31, 34, tzinfo=timezone.utc)      # 09:31:34 ET
T_SELL = datetime(2026, 9, 17, 13, 32, 34, tzinfo=timezone.utc)     # 09:32:34 ET
NOW = datetime(2026, 9, 17, 14, 0, tzinfo=timezone.utc)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv('STARTING_CASH', '500')
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '0')
    path = str(tmp_path / 'pm.db')
    db = Database(path, now=OPENED)
    ensure_schema(db.conn)
    bt = BudgetTracker(path)
    bt.ensure_day_state(DAY, 497.48)
    return db, bt


def _bars(sym, rows, interval):
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz='America/New_York') for t, *_ in rows])
    df = pd.DataFrame([r[1:] for r in rows], index=idx,
                      columns=['Open', 'High', 'Low', 'Close']).assign(Volume=1)
    return df


def test_pair_trades_matches_sell_to_buy_and_keeps_open_buys(ledger):
    db, bt = ledger
    a = bt.log_trade('CVNA', 'BUY', 67.53, 1.8416, probability=0.525, now=T_BUY); bt.execute_trade(a, now=T_BUY)
    b = bt.log_trade('AFRM', 'BUY', 74.09, 1.6787, probability=0.488, now=T_BUY); bt.execute_trade(b, now=T_BUY)
    c = bt.log_trade('CVNA', 'SELL', 66.26, 1.8416, exit_reason='sl', now=T_SELL); bt.execute_trade(c, now=T_SELL)
    trades = pair_trades(bt.get_trades_since_open(DAY))
    assert [t['symbol'] for t in trades] == ['CVNA', 'AFRM']
    cv, af = trades
    assert cv['exit_reason'] == 'sl' and cv['held_min'] == pytest.approx(1.0)
    assert cv['entry_p'] == pytest.approx(0.525) and cv['pct'] == pytest.approx(66.26 / 67.53 - 1)
    assert cv['pnl'] < 0 and cv['open'] is False
    assert af['open'] is True and af['exit_at'] is None


def test_context_reports_timing_gap_path_and_patterns(ledger):
    db, bt = ledger
    a = bt.log_trade('CVNA', 'BUY', 67.53, 1.8416, probability=0.525, now=T_BUY); bt.execute_trade(a, now=T_BUY)
    c = bt.log_trade('CVNA', 'SELL', 66.26, 1.8416, exit_reason='sl', now=T_SELL); bt.execute_trade(c, now=T_SELL)
    # prior session's last 5m bar and the 1m path around the trade
    upsert_bars(db.conn, _bars('CVNA', [('2026-09-16 15:55', 65.38, 65.5, 65.3, 65.44)], '5m'), ['CVNA'], '5m')
    upsert_bars(db.conn, _bars('CVNA', [('2026-09-17 09:30', 67.47, 67.59, 66.97, 67.48),
                                        ('2026-09-17 09:31', 67.47, 67.47, 66.18, 66.27),
                                        ('2026-09-17 09:32', 66.27, 66.39, 65.77, 65.90)], '1m'),
                ['CVNA'], '1m')
    with db.conn:
        db.conn.execute("INSERT INTO news (id,symbol,published_at,fetched_at,headline) VALUES "
                        "(1,'CVNA','2026-09-16T22:00:00+00:00','x','Why Carvana dipped')")

    ctx = build_postmortem_context(db.conn, bt, DAY, reason='3 stop-losses in a row', now=NOW)
    assert 'Triggered because: 3 stop-losses in a row' in ctx
    assert 'Start-of-day equity $497.48' in ctx
    assert 'entered 09:31:34 ET (+2 min from the open) at $67.53, model p=0.525' in ctx
    assert 'prior close $65.44, so entry was +3.2% vs yesterday' in ctx
    assert 'exited 09:32:34 ET at $66.26 (sl) after 1 min' in ctx
    assert '1m bars while held: low $65.77, high $67.59, last $65.90' in ctx
    assert 'Headlines in the 24h before entry: 1.' in ctx
    assert ('Patterns Python found: 1 of 1 entries were placed within 15 min of the open; '
            '1 of 1 closed trades were stopped out within 10 min; '
            'average entry was +3.2% above the prior close (1 of 1 gapped up more than 2%).') in ctx


def test_context_with_no_trades(ledger):
    db, bt = ledger
    ctx = build_postmortem_context(db.conn, bt, DAY, now=NOW)
    assert 'No trades on this date.' in ctx and 'Patterns' not in ctx


def test_trade_context_for_one_symbol(ledger):
    from src.postmortem import build_trade_context
    db, bt = ledger
    a = bt.log_trade('CVNA', 'BUY', 67.53, 1.8416, probability=0.525, now=T_BUY); bt.execute_trade(a, now=T_BUY)
    c = bt.log_trade('CVNA', 'SELL', 66.26, 1.8416, exit_reason='sl', now=T_SELL); bt.execute_trade(c, now=T_SELL)
    with db.conn:
        db.conn.execute("INSERT INTO signals (bar_ts,symbol,asset_class,probability,bar,above_bar,ref_price,trade_date) "
                        "VALUES ('2026-09-17T13:30:00+00:00','CVNA','stock',0.525,0.347,1,67.5,'2026-09-17')")
        db.conn.execute("INSERT INTO news (id,symbol,published_at,fetched_at,headline) VALUES "
                        "(9,'CVNA','2026-09-16T22:00:00+00:00','x','Why Carvana dipped')")
    ctx = build_trade_context(db.conn, bt, 'cvna', now=NOW)
    assert ctx.startswith('Most recent CVNA trade (2026-09-17 ET).')
    assert 'Entered 09:31:34 ET (+2 min from the open) at $67.53, model p=0.525.' in ctx
    assert 'Exited 09:32:34 ET at $66.26 (sl) after 1 min, net' in ctx
    assert 'Model probability on the 1 bars within an hour of entry: min 0.525, max 0.525, 1 of them over the 0.347 bar.' in ctx
    assert 'Headlines in the 24h before entry: 1.' in ctx and '- Why Carvana dipped' in ctx
    assert build_trade_context(db.conn, bt, 'ZZZ', now=NOW) == 'No trades in ZZZ since the account opened.'
