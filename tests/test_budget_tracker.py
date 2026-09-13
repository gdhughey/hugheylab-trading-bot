"""Ledger tests for BudgetTracker (Task 4: paper brokerage account).

Every clock-dependent call passes now= explicitly, and the fixture pins the
account's opened_at, so the `created_at >= opened_at` filters behave the same
no matter when the suite runs.
"""
from datetime import datetime, timezone

import pytest

from src.database import Database
from src.budget_tracker import BudgetTracker, _et_date
from src.intraday_engine import ET

OPENED = '2026-09-14T13:30:00+00:00'          # Mon 2026-09-14 09:30 ET


def et(y, m, d, hh, mm):
    """A tz-aware UTC datetime for the given ET wall-clock time."""
    return datetime(y, m, d, hh, mm, tzinfo=ET).astimezone(timezone.utc)


FRI_1000 = et(2026, 9, 18, 10, 0)
FRI_1555 = et(2026, 9, 18, 15, 55)
FRI_1600 = et(2026, 9, 18, 16, 0)
MON_0929 = et(2026, 9, 21, 9, 29)
MON_0930 = et(2026, 9, 21, 9, 30)


def _pin_env(monkeypatch, account_type='cash'):
    # The account seed and costs.fill read these at call time; pin them so
    # the numbers below are exact regardless of the container's environment.
    monkeypatch.setenv('STARTING_CASH', '500')
    monkeypatch.setenv('ACCOUNT_TYPE', account_type)
    monkeypatch.setenv('FAST_MAX_POSITIONS', '4')
    monkeypatch.setenv('MIN_ORDER_USD', '1')
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '5')
    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '60')
    monkeypatch.setenv('SEC_FEE_RATE', '0.0000206')
    monkeypatch.setenv('FINRA_TAF_PER_SHARE', '0.000195')
    monkeypatch.setenv('FINRA_TAF_CAP', '9.79')


def _open_account(tmp_path, monkeypatch, account_type='cash'):
    _pin_env(monkeypatch, account_type)
    path = str(tmp_path / 'ledger.db')
    db = Database(path)                       # creates schema + seeds the account row
    with db.conn:
        # The seed stamps the wall clock; pin it so the test timestamps are
        # always after the account opened.
        db.conn.execute("UPDATE account SET opened_at = ? WHERE id = 1", (OPENED,))
    return BudgetTracker(path)


@pytest.fixture
def bt(tmp_path, monkeypatch):
    return _open_account(tmp_path, monkeypatch)


def _round_trip(bt, symbol, side, ref, qty, now, **kw):
    """log_trade + execute_trade at the same instant; returns the executed row."""
    tid = bt.log_trade(symbol, side, ref, qty, now=now, **kw)
    return bt.execute_trade(tid, now=now)


# --- account reads ---------------------------------------------------------

def test_account_reads_on_a_fresh_account(bt):
    assert bt.db_path.endswith('ledger.db')   # Task 9 opens its own connection from this
    assert bt.opened_at() == OPENED
    assert bt.starting_cash() == 500.0
    assert bt.account_type() == 'cash'
    assert bt.get_cash() == 500.0
    assert bt.get_unsettled(now=FRI_1000) == 0.0
    assert bt.get_buying_power(now=FRI_1000) == 500.0
    assert bt.get_equity({}) == 500.0
    assert bt.get_positions() == []
    assert bt.get_fees_paid() == 0.0
    assert bt.get_realized_pnl() == 0.0
    assert bt.get_gross_pnl() == 0.0


def test_et_date_is_the_et_calendar_day():
    assert _et_date('2026-09-18T23:30:00+00:00') == '2026-09-18'   # 19:30 ET
    assert _et_date('2026-09-19T03:30:00+00:00') == '2026-09-18'   # 23:30 ET, still Friday


# --- trade lifecycle -------------------------------------------------------

def test_log_trade_records_fill_and_pending_hold(bt):
    tid = bt.log_trade('AAPL', 'BUY', 100.0, 1.2493751, probability=0.61, now=FRI_1000)
    row = bt.conn.execute("SELECT * FROM trades WHERE id = ?", (tid,)).fetchone()
    assert row['status'] == 'PENDING'
    assert row['shares'] == 1.249375                      # rounded to 6 dp, stored REAL
    assert row['ref_price'] == 100.0
    assert row['price'] == pytest.approx(100.05)          # 5 bps slippage on BUY
    assert row['fees'] == 0.0
    assert row['amount'] == pytest.approx(1.249375 * 100.05)
    assert row['created_at'] == FRI_1000.isoformat(timespec='seconds')
    assert row['trade_date'] == '2026-09-18'
    assert row['entry_probability'] == 0.61
    assert row['exit_reason'] is None
    assert row['available_at'] is None
    # a PENDING BUY holds buying power but has not moved cash
    assert bt.get_cash() == 500.0
    assert bt.get_buying_power(now=FRI_1000) == pytest.approx(500.0 - row['amount'])
    assert bt.reject_trade(tid, now=FRI_1000) is True
    assert bt.reject_trade(tid, now=FRI_1000) is False   # already decided
    assert bt.get_buying_power(now=FRI_1000) == 500.0
    assert bt.execute_trade(tid, now=FRI_1000) is None   # not PENDING any more


def test_stock_sell_is_unsettled_until_next_trading_day_open(bt):
    buy = _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    assert buy['status'] == 'EXECUTED'
    assert bt.get_cash() == pytest.approx(500.0 - buy['amount'])
    sell = _round_trip(bt, 'AAPL', 'SELL', 101.0, 1.0, FRI_1555, exit_reason='eod')
    assert sell['exit_reason'] == 'eod'
    assert sell['price'] == pytest.approx(101.0 * (1 - 0.0005))
    assert sell['available_at'] == '2026-09-21T13:30:00+00:00'   # Mon 09:30 ET in UTC
    cash = bt.get_cash()
    assert cash == pytest.approx(500.0 - buy['amount'] + sell['amount'])
    # Friday after the fill, after the bell, and Monday pre-open: proceeds are held
    for now in (FRI_1555, FRI_1600, MON_0929):
        assert bt.get_unsettled(now=now) == pytest.approx(sell['amount'])
        assert bt.get_buying_power(now=now) == pytest.approx(cash - sell['amount'])
    # Monday 09:30 ET: settled
    assert bt.get_unsettled(now=MON_0930) == 0.0
    assert bt.get_buying_power(now=MON_0930) == pytest.approx(cash)
    assert bt.get_positions() == []


def test_settlement_skips_thanksgiving(bt):
    _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, et(2026, 11, 25, 10, 0))
    sell = _round_trip(bt, 'AAPL', 'SELL', 100.0, 1.0, et(2026, 11, 25, 15, 55), exit_reason='eod')
    # 2026-11-26 is a holiday; 09:30 ET on the 27th is 14:30 UTC (EST)
    assert sell['available_at'] == et(2026, 11, 27, 9, 30).isoformat(timespec='seconds')


def test_crypto_sell_settles_immediately(bt):
    _round_trip(bt, 'BTC-USD', 'BUY', 50_000.0, 0.002, FRI_1555)
    sell = _round_trip(bt, 'BTC-USD', 'SELL', 50_000.0, 0.002, FRI_1600, exit_reason='tp')
    assert sell['fees'] == 0.0
    assert sell['price'] == pytest.approx(50_000.0 * (1 - 0.006))
    assert sell['available_at'] == sell['created_at']
    assert bt.get_unsettled(now=FRI_1600) == 0.0
    assert bt.get_buying_power(now=FRI_1600) == pytest.approx(bt.get_cash())


def test_margin_account_settles_immediately(tmp_path, monkeypatch):
    bt = _open_account(tmp_path, monkeypatch, account_type='margin')
    assert bt.account_type() == 'margin'
    _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    sell = _round_trip(bt, 'AAPL', 'SELL', 101.0, 1.0, FRI_1555, exit_reason='eod')
    assert sell['available_at'] == sell['created_at']
    assert bt.get_unsettled(now=FRI_1555) == 0.0
    assert bt.get_buying_power(now=FRI_1555) == pytest.approx(bt.get_cash())


def test_trades_before_opened_at_are_ignored(bt):
    # A legacy EXECUTED SELL from before the account opened, with an
    # unsettled available_at far in the future. BudgetTracker's connection is
    # in autocommit mode, so a bare execute() is durable.
    bt.conn.execute(
        "INSERT INTO trades (symbol, side, price, shares, amount, status, created_at, settled_at, "
        "week_key, realized_pnl, fees, gross_pnl, available_at, trade_date) "
        "VALUES ('AAPL', 'SELL', 100, 5, 500, 'EXECUTED', '2026-09-10T15:00:00+00:00', "
        "'2026-09-10T15:00:00+00:00', '2026-W37', 999, 1000, 1, '2099-01-01T00:00:00+00:00', "
        "'2026-09-10')")
    assert bt.get_realized_pnl() == 0.0
    assert bt.get_gross_pnl() == 0.0
    assert bt.get_fees_paid() == 0.0
    assert bt.get_unsettled(now=FRI_1000) == 0.0
    assert bt.get_buying_power(now=FRI_1000) == 500.0
    assert bt.get_trades_since_open() == []
    _round_trip(bt, 'MSFT', 'BUY', 200.0, 0.5, FRI_1000)
    assert [r['symbol'] for r in bt.get_trades_since_open()] == ['MSFT']
    assert [r['symbol'] for r in bt.get_trades_since_open('2026-09-18')] == ['MSFT']
    assert bt.get_trades_since_open('2026-09-17') == []


# --- P&L and positions -----------------------------------------------------

def test_realized_pnl_is_net_and_gross_adds_fees_back(bt):
    buy = _round_trip(bt, 'AAPL', 'BUY', 100.0, 2.0, FRI_1000)
    assert buy['fees'] == 0.0
    assert buy['realized_pnl'] is None and buy['gross_pnl'] is None
    sell = _round_trip(bt, 'AAPL', 'SELL', 110.0, 2.0, FRI_1555, exit_reason='tp')
    gross_proceeds = 2.0 * 110.0 * (1 - 0.0005)
    assert sell['fees'] == pytest.approx(0.0000206 * gross_proceeds + 0.000195 * 2.0)
    assert sell['fees'] > 0
    assert sell['amount'] == pytest.approx(gross_proceeds - sell['fees'])
    assert sell['realized_pnl'] == pytest.approx(sell['amount'] - buy['amount'])
    assert sell['gross_pnl'] == pytest.approx(sell['realized_pnl'] + sell['fees'])
    assert sell['gross_pnl'] == pytest.approx(gross_proceeds - buy['amount'])
    assert sell['realized_pnl'] < sell['gross_pnl']
    assert bt.get_realized_pnl() == pytest.approx(sell['realized_pnl'])
    assert bt.get_gross_pnl() == pytest.approx(sell['gross_pnl'])
    assert bt.get_fees_paid() == pytest.approx(sell['fees'])


def test_entry_ref_is_quantity_weighted_and_survives_partial_sell(bt):
    _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    _round_trip(bt, 'AAPL', 'BUY', 104.0, 3.0, FRI_1000)
    (pos,) = bt.get_positions()
    assert set(pos) == {'symbol', 'shares', 'avg_price', 'entry_ref', 'cost_basis', 'updated_at'}
    assert pos['shares'] == 4.0 and isinstance(pos['shares'], float)
    assert pos['entry_ref'] == pytest.approx((1 * 100.0 + 3 * 104.0) / 4)      # 103.0, ref-weighted
    assert pos['avg_price'] == pytest.approx((100.05 + 3 * 104.052) / 4)      # net cost basis
    assert pos['cost_basis'] == pytest.approx(100.05 + 3 * 104.052)
    _round_trip(bt, 'AAPL', 'SELL', 110.0, 2.0, FRI_1555, exit_reason='tp')
    (pos,) = bt.get_positions()
    assert pos['shares'] == 2.0
    assert pos['entry_ref'] == pytest.approx(103.0)                            # unchanged by a partial SELL
    assert pos['avg_price'] == pytest.approx((100.05 + 3 * 104.052) / 4)
    _round_trip(bt, 'AAPL', 'SELL', 110.0, 2.0, FRI_1555, exit_reason='eod')
    assert bt.get_positions() == []
    raw = bt.conn.execute(
        "SELECT shares, avg_price, entry_ref FROM positions WHERE symbol = 'AAPL'").fetchone()
    assert (raw['shares'], raw['avg_price'], raw['entry_ref']) == (0.0, 0.0, 0.0)


def test_dust_is_written_as_zero_and_hidden(bt):
    _round_trip(bt, 'DOGE-USD', 'BUY', 0.1, 0.1, FRI_1000)
    _round_trip(bt, 'DOGE-USD', 'BUY', 0.1, 0.2, FRI_1000)
    # 0.1 + 0.2 - 0.3 is 5.5e-17 in binary float; the ledger must not keep it
    _round_trip(bt, 'DOGE-USD', 'SELL', 0.1, 0.3, FRI_1555, exit_reason='sl')
    raw = bt.conn.execute(
        "SELECT shares, avg_price, entry_ref FROM positions WHERE symbol = 'DOGE-USD'").fetchone()
    assert raw['shares'] == 0.0 and raw['avg_price'] == 0.0 and raw['entry_ref'] == 0.0
    assert bt.get_positions() == []
