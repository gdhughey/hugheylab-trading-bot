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


def test_sell_of_an_unheld_symbol_is_rejected_and_moves_nothing(bt):
    tid = bt.log_trade('MSFT', 'SELL', 100.0, 1.0, exit_reason='manual', now=FRI_1555)
    assert bt.execute_trade(tid, now=FRI_1555) is None
    row = bt.conn.execute("SELECT * FROM trades WHERE id = ?", (tid,)).fetchone()
    assert row['status'] == 'REJECTED'
    assert row['realized_pnl'] is None and row['gross_pnl'] is None
    assert bt.get_cash() == 500.0
    assert bt.get_buying_power(now=FRI_1555) == 500.0
    assert bt.get_realized_pnl() == 0.0 and bt.get_fees_paid() == 0.0
    assert bt.get_positions() == []
    assert bt.get_trades_since_open() == []
    assert bt.conn.execute("SELECT COUNT(*) AS n FROM positions").fetchone()['n'] == 0


def test_manual_sell_approved_after_auto_exit_is_rejected(bt):
    # The spec's /sell logs a PENDING SELL for the whole position; while it
    # waits for approval the fast cycle exits the same symbol. The approved
    # manual SELL must not mint phantom cash against a flat position.
    buy = _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    manual = bt.log_trade('AAPL', 'SELL', 100.0, 1.0, exit_reason='manual', now=FRI_1000)
    auto = _round_trip(bt, 'AAPL', 'SELL', 100.0, 1.0, FRI_1555, exit_reason='eod')
    cash_after_auto = bt.get_cash()
    assert cash_after_auto == pytest.approx(500.0 - buy['amount'] + auto['amount'])
    assert bt.execute_trade(manual, now=FRI_1555) is None
    assert bt.conn.execute("SELECT status FROM trades WHERE id = ?", (manual,)).fetchone()['status'] == 'REJECTED'
    assert bt.get_cash() == pytest.approx(cash_after_auto)
    assert bt.get_realized_pnl() == pytest.approx(auto['realized_pnl'])
    assert bt.get_positions() == []
    assert [r['id'] for r in bt.get_trades_since_open()] == [buy['id'], auto['id']]


@pytest.mark.parametrize('qty', [0.0, -1.0, 1e-9])
def test_log_trade_rejects_non_positive_qty(bt, qty):
    # A negative-qty BUY would produce a negative amount and CREDIT cash on
    # execute; a zero-qty trade would write a 0-share position row. Neither
    # may reach the ledger.
    with pytest.raises(ValueError, match='qty'):
        bt.log_trade('AAPL', 'BUY', 100.0, qty, now=FRI_1000)
    assert bt.get_cash() == 500.0
    assert bt.conn.execute("SELECT COUNT(*) AS n FROM trades").fetchone()['n'] == 0


@pytest.mark.parametrize('ref', [0.0, -100.0])
def test_log_trade_rejects_non_positive_ref_price(bt, ref):
    with pytest.raises(ValueError, match='ref_price'):
        bt.log_trade('AAPL', 'BUY', ref, 1.0, now=FRI_1000)
    assert bt.conn.execute("SELECT COUNT(*) AS n FROM trades").fetchone()['n'] == 0


def _fill_three_slots(bt):
    """Three $125 BUYs from $500: buying power is then the last ~$125."""
    for sym in ('AAPL', 'MSFT', 'NVDA'):
        qty, _, _ = bt.size_order(sym, 100.0, now=FRI_1000)
        _round_trip(bt, sym, 'BUY', 100.0, qty, FRI_1000)
    bp = bt.get_buying_power(now=FRI_1000)
    assert bp == pytest.approx(125.0, abs=1e-3)
    return bp


def _assert_rejected(bt, tid, now):
    row = bt.conn.execute("SELECT * FROM trades WHERE id = ?", (tid,)).fetchone()
    assert row['status'] == 'REJECTED'
    assert row['settled_at'] == now.isoformat(timespec='seconds')


def test_buy_that_exceeds_buying_power_at_execution_is_rejected(bt):
    # The fast cycle and a Discord /buy can both call size_order before either
    # logs its trade, so both are told the same (last) $125. The fast cycle
    # logs and fills at once; the /buy is logged after and approved later.
    # Its fill must be refused inside the transaction, not overdraw cash.
    _fill_three_slots(bt)
    qty_a, size_a, _ = bt.size_order('AMD', 100.0, now=FRI_1000)
    qty_b, size_b, _ = bt.size_order('TSLA', 100.0, now=FRI_1000)
    assert size_a == pytest.approx(size_b)
    fast = _round_trip(bt, 'AMD', 'BUY', 100.0, qty_a, FRI_1000)
    assert fast['status'] == 'EXECUTED'
    cash_after = bt.get_cash()
    assert cash_after == pytest.approx(0.0, abs=1e-3)
    manual = bt.log_trade('TSLA', 'BUY', 100.0, qty_b, now=FRI_1000)
    assert bt.execute_trade(manual, now=FRI_1555) is None
    _assert_rejected(bt, manual, FRI_1555)
    assert bt.get_cash() == pytest.approx(cash_after)
    assert bt.get_cash() >= -1e-3
    assert [p['symbol'] for p in bt.get_positions()] == ['AAPL', 'AMD', 'MSFT', 'NVDA']
    assert bt.get_buying_power(now=FRI_1555) == pytest.approx(0.0, abs=1e-3)   # hold released
    assert [r['symbol'] for r in bt.get_trades_since_open()] == ['AAPL', 'MSFT', 'NVDA', 'AMD']


def test_pending_buy_hold_is_a_reservation_the_later_buy_must_respect(bt):
    # Both BUYs are PENDING before either fills (a /buy awaiting approval and
    # a fast-cycle entry sized in the same window). A PENDING hold reserves
    # buying power for its own fill and for nothing else: whichever is
    # executed first must not spend the other's reservation. Exactly one
    # fills and cash never goes negative.
    _fill_three_slots(bt)
    qty_a, _, _ = bt.size_order('AMD', 100.0, now=FRI_1000)
    qty_b, _, _ = bt.size_order('TSLA', 100.0, now=FRI_1000)
    first = bt.log_trade('AMD', 'BUY', 100.0, qty_a, now=FRI_1000)
    second = bt.log_trade('TSLA', 'BUY', 100.0, qty_b, now=FRI_1000)
    assert bt.get_buying_power(now=FRI_1000) == 0.0            # both holds taken
    assert bt.execute_trade(first, now=FRI_1000) is None       # second's hold reserves the cash
    _assert_rejected(bt, first, FRI_1000)
    assert bt.get_cash() == pytest.approx(125.0, abs=1e-3)     # nothing moved
    filled = bt.execute_trade(second, now=FRI_1000)
    assert filled['status'] == 'EXECUTED'
    assert bt.get_cash() == pytest.approx(0.0, abs=1e-3)
    assert bt.get_cash() >= -1e-3
    assert [p['symbol'] for p in bt.get_positions()] == ['AAPL', 'MSFT', 'NVDA', 'TSLA']


def test_buy_sized_before_a_sell_settles_is_rejected_in_a_cash_account(bt):
    # Unsettled proceeds are cash but not buying power. A BUY logged against
    # them (a /buy typed with a bigger qty than size_order allows) is refused
    # at fill even though cash would cover it.
    _round_trip(bt, 'AAPL', 'BUY', 100.0, 4.0, FRI_1000)
    _round_trip(bt, 'AAPL', 'SELL', 100.0, 4.0, FRI_1555, exit_reason='eod')
    assert bt.get_cash() > 490.0
    assert bt.get_buying_power(now=FRI_1555) == pytest.approx(500.0 - 4 * 100.05)   # $99.80 that never left
    cash = bt.get_cash()
    tid = bt.log_trade('MSFT', 'BUY', 100.0, 2.0, now=FRI_1555)     # ~$200 > ~$100 buying power
    assert bt.execute_trade(tid, now=FRI_1555) is None
    _assert_rejected(bt, tid, FRI_1555)
    assert bt.get_cash() == pytest.approx(cash)
    assert bt.get_positions() == []
    # Monday the proceeds have settled and the same order fills
    tid = bt.log_trade('MSFT', 'BUY', 100.0, 2.0, now=MON_0930)
    assert bt.execute_trade(tid, now=MON_0930)['status'] == 'EXECUTED'


def test_buy_sized_at_full_buying_power_fills_despite_qty_rounding(bt):
    # size_order rounds qty to 6 dp, which on a high-priced asset can push the
    # debit a few cents over buying power (the "6-dp qty rounding aside" in
    # its docstring). That slack belongs to sizing, not the caller: the guard
    # must not reject the very order size_order produced.
    for sym in ('AAPL', 'MSFT', 'NVDA'):
        qty, _, _ = bt.size_order(sym, 100.0, now=FRI_1000)
        _round_trip(bt, sym, 'BUY', 100.0, qty, FRI_1000)
    bp = bt.get_buying_power(now=FRI_1000)
    qty, size_usd, est_fill = bt.size_order('BTC-USD', 100_000.0, now=FRI_1000)
    assert size_usd == pytest.approx(125.0)
    assert qty == 0.001243                                  # 125 / 100600 = 0.0012425.. rounds UP
    over = qty * est_fill - bp
    assert 0 < over < 0.5e-6 * est_fill                     # a few cents, inside one 6-dp step
    # two 6-dp steps more is the caller's overreach, not rounding: rejected
    too_big = bt.log_trade('BTC-USD', 'BUY', 100_000.0, round(qty + 2e-6, 6), now=FRI_1000)
    assert bt.execute_trade(too_big, now=FRI_1000) is None
    assert bt.get_cash() == pytest.approx(bp)
    # the sized order itself fills
    tid = bt.log_trade('BTC-USD', 'BUY', 100_000.0, qty, now=FRI_1000)
    row = bt.execute_trade(tid, now=FRI_1000)
    assert row['status'] == 'EXECUTED'
    assert row['amount'] == pytest.approx(bp + over)
    assert bt.get_cash() == pytest.approx(-over)            # the documented rounding slack, nothing more
    assert len(bt.get_positions()) == 4


def test_sell_of_more_than_held_is_capped_to_the_position(bt):
    buy = _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    tid = bt.log_trade('AAPL', 'SELL', 100.0, 3.0, exit_reason='manual', now=FRI_1555)
    pending = bt.conn.execute("SELECT * FROM trades WHERE id = ?", (tid,)).fetchone()
    assert pending['shares'] == 3.0
    sell = bt.execute_trade(tid, now=FRI_1555)
    assert sell['status'] == 'EXECUTED'
    assert sell['shares'] == 1.0                                # capped to what was held
    assert sell['price'] == pytest.approx(100.0 * (1 - 0.0005))
    gross = 1.0 * sell['price']
    assert sell['fees'] == pytest.approx(0.0000206 * gross + 0.000195 * 1.0)
    assert sell['amount'] == pytest.approx(gross - sell['fees'])
    assert sell['amount'] < pending['amount'] / 2               # not the 3-share proceeds
    assert sell['realized_pnl'] == pytest.approx(sell['amount'] - buy['amount'])
    assert sell['realized_pnl'] < 0                             # slippage + fees, no phantom gain
    assert sell['gross_pnl'] == pytest.approx(sell['realized_pnl'] + sell['fees'])
    assert bt.get_cash() == pytest.approx(500.0 - buy['amount'] + sell['amount'])
    assert bt.get_realized_pnl() == pytest.approx(sell['realized_pnl'])
    assert bt.get_unsettled(now=FRI_1555) == pytest.approx(sell['amount'])
    assert bt.get_positions() == []


# --- sizing ----------------------------------------------------------------

def test_size_order_is_fractional_and_dollar_based(bt):
    qty, size_usd, est_fill = bt.size_order('AAPL', 100.0, now=FRI_1000)
    assert est_fill == pytest.approx(100.05)
    assert size_usd == pytest.approx(125.0)                 # 500 / FAST_MAX_POSITIONS
    assert qty == round(125.0 / 100.05, 6) == 1.249375
    assert qty * est_fill <= size_usd + 1e-3                # 6-dp qty rounding is the only slack


def test_size_order_below_min_order_returns_zero_qty(bt, monkeypatch):
    monkeypatch.setenv('MIN_ORDER_USD', '200')              # read at call time
    qty, size_usd, est_fill = bt.size_order('AAPL', 100.0, now=FRI_1000)
    assert qty == 0.0
    assert size_usd == pytest.approx(125.0)
    assert est_fill == pytest.approx(100.05)


@pytest.mark.parametrize('symbols', [
    ['AAPL', 'MSFT', 'NVDA', 'AMD'],
    ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD'],
])
def test_filling_every_slot_from_500_never_overdraws(bt, symbols):
    for sym in symbols:
        bp = bt.get_buying_power(now=FRI_1000)
        qty, size_usd, est_fill = bt.size_order(sym, 100.0, now=FRI_1000)
        assert qty > 0
        assert size_usd <= bp
        row = _round_trip(bt, sym, 'BUY', 100.0, qty, FRI_1000)
        assert row['amount'] <= bp + 1e-3                   # slippage is inside amount
        assert bt.get_cash() >= -1e-3
    assert len(bt.get_positions()) == 4
    assert bt.get_cash() == pytest.approx(0.0, abs=1e-3)
    # fully deployed: a fifth order is below MIN_ORDER_USD
    qty, size_usd, _ = bt.size_order('TSLA', 100.0, now=FRI_1000)
    assert qty == 0.0 and size_usd < 1.0


def test_entry_fills_in_full_when_buying_power_binds(bt):
    # Three stock round trips on Friday. Each SELL is profitable, so equity
    # grows to ~$537, but in a cash account the proceeds stay unsettled until
    # Monday - so buying power is only the $125 that never left. That puts
    # buying power BELOW equity / FAST_MAX_POSITIONS (~$134): the next order
    # must be capped at buying power and still fill in full there, not be
    # skipped or overdraw.
    for sym in ('AAPL', 'MSFT', 'NVDA'):
        qty, _, _ = bt.size_order(sym, 100.0, now=FRI_1000)
        _round_trip(bt, sym, 'BUY', 100.0, qty, FRI_1000)
    for sym in ('AAPL', 'MSFT', 'NVDA'):
        (pos,) = [p for p in bt.get_positions() if p['symbol'] == sym]
        sell = _round_trip(bt, sym, 'SELL', 110.0, pos['shares'], FRI_1555, exit_reason='tp')
        assert sell['realized_pnl'] > 0
    assert bt.get_positions() == []
    bp = bt.get_buying_power(now=FRI_1555)
    equity = bt.get_equity({})
    assert bt.get_unsettled(now=FRI_1555) > 400.0          # three SELLs' proceeds, held to Monday
    assert bp == pytest.approx(125.0, abs=1e-3)             # the cash left after three $125 buys
    assert equity > 530.0
    assert bp < equity / 4                                  # buying power binds, not the equity slice
    qty, size_usd, est_fill = bt.size_order('AMD', 100.0, now=FRI_1555)
    assert size_usd == pytest.approx(bp)
    assert qty == round(bp / est_fill, 6)
    row = _round_trip(bt, 'AMD', 'BUY', 100.0, qty, FRI_1555)
    assert row['amount'] == pytest.approx(bp, abs=1e-3)     # filled in full at the cap
    assert bt.get_buying_power(now=FRI_1555) == pytest.approx(0.0, abs=1e-3)
    assert bt.get_cash() > 400.0                            # unsettled proceeds are still cash
    # Monday the proceeds settle and all of it is spendable again
    assert bt.get_buying_power(now=MON_0930) == pytest.approx(bt.get_cash())


def test_compounding_after_a_win_grows_the_next_order(bt):
    qty1, size1, _ = bt.size_order('AAPL', 100.0, now=FRI_1000)
    _round_trip(bt, 'AAPL', 'BUY', 100.0, qty1, FRI_1000)
    sell = _round_trip(bt, 'AAPL', 'SELL', 120.0, qty1, FRI_1555, exit_reason='tp')
    assert sell['realized_pnl'] > 0
    # Monday, once the proceeds settle, equity has grown and so has the slice
    qty2, size2, _ = bt.size_order('MSFT', 100.0, now=MON_0930)
    assert bt.get_equity({}) == pytest.approx(bt.get_cash())
    assert size2 == pytest.approx(bt.get_cash() / 4)
    assert size2 > size1 and qty2 > qty1


# --- get_pnl -----------------------------------------------------------------

def test_get_pnl_keys_and_equity(bt):
    _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    _round_trip(bt, 'MSFT', 'BUY', 200.0, 0.5, FRI_1000)
    quotes = {'AAPL': 110.0}                                # MSFT has no quote -> stale
    pnl = bt.get_pnl(quotes.get)
    assert set(pnl) == {
        'positions', 'stale', 'realized', 'unrealized', 'total', 'cost_basis',
        'market_value', 'cash', 'unsettled', 'buying_power', 'equity', 'starting_cash',
        'all_time_net', 'all_time_pct', 'fees_paid', 'gross_pnl'}
    assert 'return_pct' not in pnl
    assert pnl['stale'] == ['MSFT']
    assert pnl['cash'] == pytest.approx(500.0 - 100.05 - 0.5 * 200.1)
    assert pnl['starting_cash'] == 500.0
    assert pnl['unrealized'] == pytest.approx(110.0 - 100.05)
    assert pnl['market_value'] == pytest.approx(110.0)
    assert pnl['cost_basis'] == pytest.approx(100.05)
    # the stale MSFT is carried at avg_price, so equity moves only with AAPL
    assert pnl['equity'] == pytest.approx(pnl['cash'] + 110.0 + 0.5 * 200.1)
    assert pnl['all_time_net'] == pytest.approx(pnl['equity'] - 500.0)
    assert pnl['all_time_pct'] == pytest.approx(pnl['all_time_net'] / 500.0)
    assert pnl['unsettled'] == 0.0 and pnl['buying_power'] == pytest.approx(pnl['cash'])
    assert pnl['realized'] == 0.0 and pnl['fees_paid'] == 0.0 and pnl['gross_pnl'] == 0.0
    assert pnl['total'] == pytest.approx(pnl['unrealized'])
    assert [p['symbol'] for p in pnl['positions']] == ['AAPL', 'MSFT']
    assert pnl['positions'][1]['price'] is None and pnl['positions'][1]['pnl'] is None


def test_nan_quote_is_treated_as_missing_everywhere(bt):
    # A data gap that has been through pandas arrives as NaN, not None. NaN is
    # not None, so without care it propagates: equity becomes nan, and
    # min(buying_power, nan) keeps buying_power, sizing the whole account into
    # one slot. The ledger must take the avg_price / stale path for NaN too.
    nan = float('nan')
    buy = _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    cash = bt.get_cash()
    carried = cash + 1.0 * 100.05
    assert bt.get_equity({'AAPL': nan}) == pytest.approx(carried)
    assert bt.get_equity({'AAPL': None}) == pytest.approx(carried)
    qty, size_usd, est_fill = bt.size_order('MSFT', 100.0, prices={'AAPL': nan}, now=FRI_1000)
    assert size_usd == pytest.approx(carried / 4)           # the equity slice, not all of buying power
    assert size_usd < bt.get_buying_power(now=FRI_1000)
    assert qty == round(size_usd / est_fill, 6)
    pnl = bt.get_pnl({'AAPL': nan}.get)
    assert pnl['stale'] == ['AAPL']
    assert pnl['positions'][0]['price'] is None and pnl['positions'][0]['pnl'] is None
    assert pnl['unrealized'] == 0.0 and pnl['market_value'] == 0.0
    assert pnl['equity'] == pytest.approx(carried)
    assert pnl['all_time_net'] == pytest.approx(carried - 500.0)
    snap = bt.record_equity('2026-09-18', {'AAPL': nan}, now=FRI_1600)
    assert snap['positions_value'] == pytest.approx(buy['amount'])
    assert snap['equity'] == pytest.approx(carried)


# --- day state / equity history -----------------------------------------------

def test_day_state_baseline_and_flags(bt):
    assert bt.get_day_state('2026-09-18') is None
    row = bt.ensure_day_state('2026-09-18', 500.0)
    assert row['date'] == '2026-09-18'
    assert row['start_equity'] == 500.0
    assert row['loss_tripped_at'] is None
    assert row['loss_announced_at'] is None
    assert row['report_posted_at'] is None
    # a same-date restart must keep the original baseline
    assert bt.ensure_day_state('2026-09-18', 480.0)['start_equity'] == 500.0
    tripped = FRI_1555.isoformat(timespec='seconds')
    posted = et(2026, 9, 18, 16, 5).isoformat(timespec='seconds')
    bt.set_day_flag('2026-09-18', 'loss_tripped_at', tripped)
    bt.set_day_flag('2026-09-18', 'report_posted_at', posted)
    row = bt.get_day_state('2026-09-18')
    assert row['loss_tripped_at'] == tripped
    assert row['loss_announced_at'] is None
    assert row['report_posted_at'] == posted
    with pytest.raises(ValueError):
        bt.set_day_flag('2026-09-18', 'start_equity', tripped)


def test_equity_series_starts_with_starting_cash_and_record_equity_upserts(bt):
    assert bt.equity_series() == [{'date': '2026-09-14', 'equity': 500.0}]
    _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    first = bt.record_equity('2026-09-18', {'AAPL': 90.0}, now=FRI_1600)
    assert first['date'] == '2026-09-18'
    assert first['cash'] == pytest.approx(500.0 - 100.05)
    assert first['positions_value'] == pytest.approx(90.0)
    assert first['equity'] == pytest.approx(first['cash'] + 90.0)
    assert first['fees_to_date'] == 0.0 and first['realized_to_date'] == 0.0
    assert first['recorded_at'] == FRI_1600.isoformat(timespec='seconds')
    # second call for the same date replaces, not duplicates
    later = et(2026, 9, 18, 16, 10)
    second = bt.record_equity('2026-09-18', {'AAPL': 95.0}, now=later)
    assert second['positions_value'] == pytest.approx(95.0)
    assert second['equity'] == pytest.approx(first['cash'] + 95.0)
    assert second['recorded_at'] == later.isoformat(timespec='seconds')
    assert bt.conn.execute("SELECT COUNT(*) AS n FROM equity_history").fetchone()['n'] == 1
    # a missing quote is carried at avg_price
    carried = bt.record_equity('2026-09-21', {}, now=MON_0930)
    assert carried['positions_value'] == pytest.approx(100.05)
    series = bt.equity_series()
    assert [s['date'] for s in series] == ['2026-09-14', '2026-09-18', '2026-09-21']
    assert series[0] == {'date': '2026-09-14', 'equity': 500.0}
    assert series[1]['equity'] == pytest.approx(second['equity'])
    assert series[2]['equity'] == pytest.approx(carried['equity'])


def test_equity_series_day0_is_the_et_date_of_an_evening_open(tmp_path, monkeypatch):
    # An account opened Mon 20:30 ET is already Tue in UTC. Day 0 must be the
    # ET date, like trade_date, day_state and equity_history, or the first
    # record_equity for Tuesday lands on the same date as day 0.
    _pin_env(monkeypatch)
    path = str(tmp_path / 'evening.db')
    Database(path, now=et(2026, 9, 14, 20, 30))
    bt = BudgetTracker(path)
    assert bt.opened_at() == '2026-09-15T00:30:00+00:00'
    assert bt.equity_series() == [{'date': '2026-09-14', 'equity': 500.0}]
    bt.record_equity('2026-09-15', {}, now=et(2026, 9, 15, 16, 5))
    assert [s['date'] for s in bt.equity_series()] == ['2026-09-14', '2026-09-15']
