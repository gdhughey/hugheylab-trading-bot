"""FastTrader tests: class gate, max hold, ref-to-ref barriers, daily loss
limit, signal-log wiring and skip reasons.

Every test builds a file DB under tmp_path and passes `now=` explicitly - to
cycle(), to the ledger calls, AND to Database() so account.opened_at is a
fixed date before every trade stamp (the ledger filters created_at >=
opened_at; an opened_at taken from the wall clock would silently exclude the
trades once the real date passes NOW). The engine is a fake (no model, no
yfinance, no sklearn), the BudgetTracker is real.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src import fast_trader, intraday_engine
from src.budget_tracker import BudgetTracker
from src.database import Database
from src.fast_trader import FastTrader, class_gate, max_hold_min
from src.intraday_engine import ET, asset_class

# The paper account opened two weeks before any trade in these tests, so the
# created_at >= account.opened_at filters in get_unsettled/get_buying_power
# always include the trades below regardless of the machine's clock.
OPENED_AT = datetime(2026, 9, 1, tzinfo=timezone.utc)

# Monday 2026-09-14 10:30 ET: a regular session, well clear of the bell.
NOW = datetime(2026, 9, 14, 14, 30, tzinfo=timezone.utc)
SATURDAY = datetime(2026, 9, 12, 14, 30, tzinfo=timezone.utc)
NEAR_BELL = datetime(2026, 9, 14, 19, 55, tzinfo=timezone.utc)      # 15:55 ET
NEXT_DAY = NOW + timedelta(days=1)                                  # Tue 10:30 ET
BAR_TS = NOW.isoformat()

# data/ is excluded from the test tarball, so there is no tuned.json on the
# container; pin the barriers the spec's numbers assume.
TUNED = {
    'stock': {'take_profit': 0.010, 'stop_loss': 0.006, 'horizon': 24},
    'crypto': {'take_profit': 0.006, 'stop_loss': 0.004, 'horizon': 24},
}

# Cost/account defaults from the contract, pinned so the container's shell
# environment cannot change the arithmetic.
ENV = {
    'STARTING_CASH': '500', 'ACCOUNT_TYPE': 'cash',
    'STOCK_SLIPPAGE_BPS': '5', 'CRYPTO_SPREAD_BPS': '60',
    'SEC_FEE_RATE': '0.0000206', 'FINRA_TAF_PER_SHARE': '0.000195',
    'FINRA_TAF_CAP': '9.79', 'MIN_ORDER_USD': '1',
    'DAILY_LOSS_LIMIT_PCT': '3', 'MIN_EV_TO_TRADE': '0.003',
    'FAST_MAX_POSITIONS': '3', 'FAST_EOD_FLATTEN_MIN': '10',
    'FAST_COOLDOWN_MIN': '15',
}


@pytest.fixture(autouse=True)
def paper_env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('FAST_IGNORE_EV', raising=False)
    monkeypatch.setattr(intraday_engine, 'TUNED', TUNED)
    monkeypatch.setattr(fast_trader, 'INTERVAL', '5m')


class FakeEngine:
    """Stands in for IntradayEngine: quotes and probabilities are dicts the
    test mutates between cycles. A symbol with no quote is not scored."""

    def __init__(self, symbols, quotes, metrics=None, bar=0.5):
        self.symbols = list(symbols)
        self.quotes = dict(quotes)           # symbol -> ref price (bar close)
        self.probs = {}                      # symbol -> probability (default 0.9)
        self.bar = bar
        self.bar_ts = BAR_TS
        self.metrics = metrics if metrics is not None else {
            'stock': {'ev': 0.01, 'precision': 0.5, 'breakeven': 0.4, 'bar': bar},
            'crypto': {'ev': 0.01, 'precision': 0.5, 'breakeven': 0.4, 'bar': bar},
        }

    def fetch(self, *args, **kwargs):
        return 0

    def threshold(self, cls='stock'):
        return self.bar

    def signal(self, symbol):
        if symbol not in self.quotes:
            return None
        return {'symbol': symbol, 'asset_class': asset_class(symbol),
                'probability': self.probs.get(symbol, 0.9), 'bar': self.bar,
                'price': float(self.quotes[symbol]), 'bar_ts': self.bar_ts}

    def scan_all(self):
        out = []
        for sym in self.symbols:
            s = self.signal(sym)
            if s:
                s['above_bar'] = s['probability'] >= s['bar']
                s['margin'] = s['probability'] - s['bar']
                out.append(s)
        out.sort(key=lambda r: r['margin'], reverse=True)
        return out


def make_budget(tmp_path, monkeypatch, starting_cash='500'):
    # STARTING_CASH must be in the environment before Database() seeds the
    # account row; the file path is shared so a second BudgetTracker on the
    # same tmp_path sees the same ledger (restart tests).
    #
    # now=OPENED_AT pins account.opened_at. Without it the seed uses the wall
    # clock, and get_unsettled/get_buying_power (created_at >= opened_at)
    # would drop every NOW-stamped trade as soon as the real date passes
    # 2026-09-14 - the tests would pass today and fail next week.
    monkeypatch.setenv('STARTING_CASH', starting_cash)
    path = str(tmp_path / 'bot.db')
    Database(path, now=OPENED_AT)
    return BudgetTracker(path)


def make_trader(engine, budget):
    return FastTrader(engine, budget, quote_fn=engine.quotes.get)


def open_position(budget, symbol, ref, qty, now=NOW):
    """Open a position straight through the ledger (crypto entries are cost-
    gated, so the cycle cannot open them for us)."""
    tid = budget.log_trade(symbol, 'BUY', ref, qty, probability=0.9, now=now)
    budget.execute_trade(tid, now=now)
    return tid


def trade_row(budget, trade_id):
    return budget.conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()


def signal_count(budget):
    return budget.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]


# --- max hold + class gate --------------------------------------------------

def test_max_hold_min_is_horizon_times_interval():
    assert max_hold_min('stock') == 120
    assert max_hold_min('crypto') == 120


def test_class_gate_crypto_refused_by_round_trip_cost(monkeypatch):
    monkeypatch.setenv('FAST_IGNORE_EV', '1')
    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '60')
    ok, text = class_gate('crypto', {'ev': 0.01})
    assert ok is False
    assert text == "blocked: take-profit 0.60% is below the 1.20% round-trip cost"


def test_class_gate_crypto_allowed_on_cheaper_venue(monkeypatch):
    monkeypatch.setenv('FAST_IGNORE_EV', '1')
    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '10')
    ok, text = class_gate('crypto', {'ev': 0.01})
    assert ok is True
    assert text == "Trading. Edge clears the 0.30% cost floor"


def test_class_gate_stock_allowed_at_defaults():
    assert class_gate('stock', {'ev': 0.01}) == (
        True, "Trading. Edge clears the 0.30% cost floor")


@pytest.mark.parametrize('ev, fragment', [
    (0.0006, 'smaller than the 0.30%'),
    (-0.0003, 'loses money'),
])
def test_class_gate_ev_block_when_flag_unset(ev, fragment):
    ok, text = class_gate('stock', {'ev': ev})
    assert ok is False
    assert text.startswith('Not trading.')
    assert fragment in text


def test_class_gate_ignore_ev_text(monkeypatch):
    monkeypatch.setenv('FAST_IGNORE_EV', '1')
    assert class_gate('stock', {'ev': -0.0003}) == (
        True, "trading on paper despite EV -0.030% below the 0.30% floor (FAST_IGNORE_EV on)")


def test_class_gate_without_metrics_is_ev_blocked():
    ok, text = class_gate('stock', None)
    assert ok is False
    assert text.startswith('Not trading.')


# --- barriers ref-to-ref ---------------------------------------------------

def test_crypto_barriers_measured_ref_to_ref(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['BTC-USD'], {'BTC-USD': 100.0})
    open_position(budget, 'BTC-USD', 100.0, 1.0)
    pos = budget.get_positions()[0]
    assert pos['avg_price'] == pytest.approx(100.60)    # fill carries the 60 bps
    assert pos['entry_ref'] == pytest.approx(100.0)
    trader = make_trader(engine, budget)

    # An unchanged quote is NOT a stop: avg_price sits 0.6% above ref, but the
    # barrier is measured against the ref the position was entered at.
    assert trader.cycle(now=NOW)['exits'] == []
    engine.quotes['BTC-USD'] = 99.61
    assert trader.cycle(now=NOW)['exits'] == []

    engine.quotes['BTC-USD'] = 99.60
    s = trader.cycle(now=NOW)
    assert len(s['exits']) == 1
    x = s['exits'][0]
    assert x['symbol'] == 'BTC-USD'
    assert x['exit_reason'] == 'sl'
    assert x['reason'].startswith('stop loss')
    assert x['ref_price'] == 99.60
    assert x['price'] == pytest.approx(99.60 * 0.994)
    assert x['pct'] == pytest.approx(-0.004)
    assert x['fees'] == 0
    assert x['pnl'] / x['shares'] == pytest.approx(99.60 * 0.994 - 100.60)
    row = trade_row(budget, x['trade_id'])
    assert x['pnl'] == pytest.approx(row['realized_pnl'])
    assert row['exit_reason'] == 'sl'
    assert budget.get_positions() == []


def test_stock_stop_ref_to_ref_then_cooldown(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL'], {'AAPL': 100.0})
    open_position(budget, 'AAPL', 100.0, 2.0)
    assert budget.get_positions()[0]['avg_price'] == pytest.approx(100.05)
    trader = make_trader(engine, budget)

    engine.quotes['AAPL'] = 99.45
    assert trader.cycle(now=NOW)['exits'] == []
    engine.quotes['AAPL'] = 99.40
    s = trader.cycle(now=NOW)
    x = s['exits'][0]
    assert x['exit_reason'] == 'sl'
    assert x['price'] == pytest.approx(99.40 * 0.9995)
    assert x['pct'] == pytest.approx(-0.006)
    assert x['gross'] == pytest.approx(x['pnl'] + x['fees'])
    assert x['fees'] > 0

    # The name it just sold is a candidate again but must cool down first...
    s2 = trader.cycle(now=NOW)
    assert ('AAPL', 'cooling down') in s2['skipped']
    assert s2['entries'] == []
    # ...and re-enters once FAST_COOLDOWN_MIN has passed.
    s3 = trader.cycle(now=NOW + timedelta(minutes=16))
    assert [e['symbol'] for e in s3['entries']] == ['AAPL']


def test_stock_take_profit_ref_to_ref(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL'], {'AAPL': 100.0})
    open_position(budget, 'AAPL', 100.0, 2.0)
    trader = make_trader(engine, budget)

    engine.quotes['AAPL'] = 100.95
    assert trader.cycle(now=NOW)['exits'] == []
    engine.quotes['AAPL'] = 101.00
    s = trader.cycle(now=NOW)
    x = s['exits'][0]
    assert x['exit_reason'] == 'tp'
    assert x['reason'].startswith('take profit')
    assert x['price'] == pytest.approx(101.00 * 0.9995)
    assert x['pct'] == pytest.approx(0.01)
    assert trade_row(budget, x['trade_id'])['exit_reason'] == 'tp'


def test_flat_ref_survives_max_hold_then_times_out(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL'], {'AAPL': 100.0})
    open_position(budget, 'AAPL', 100.0, 1.0, now=NOW)
    # Constructed after the BUY: the entry time is recovered from the ledger,
    # exactly as after a service restart.
    trader = make_trader(engine, budget)
    assert trader.entry_time['AAPL'] == NOW.astimezone(ET)
    assert max_hold_min('stock') == 120

    for minutes in (30, 60, 90, 120):
        assert trader.cycle(now=NOW + timedelta(minutes=minutes))['exits'] == []
    s = trader.cycle(now=NOW + timedelta(minutes=121))
    x = s['exits'][0]
    assert x['exit_reason'] == 'timeout'
    assert x['reason'] == 'held 120 min without hitting a target'
    assert x['pct'] == pytest.approx(0.0)
    assert trade_row(budget, x['trade_id'])['exit_reason'] == 'timeout'
    assert 'AAPL' not in trader.entry_time


# --- daily loss limit -------------------------------------------------------

def test_daily_loss_limit_trips_blocks_entries_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv('FAST_MAX_POSITIONS', '2')
    budget = make_budget(tmp_path, monkeypatch)
    # NVDA has no quote yet, so cycle 1 scores only AAPL and MSFT.
    engine = FakeEngine(['AAPL', 'MSFT', 'NVDA'], {'AAPL': 100.0, 'MSFT': 100.0})
    trader = make_trader(engine, budget)

    s1 = trader.cycle(now=NOW)
    assert sorted(e['symbol'] for e in s1['entries']) == ['AAPL', 'MSFT']
    assert s1['loss_tripped'] is False and s1['loss_announce'] is False
    assert budget.get_day_state('2026-09-14')['start_equity'] == pytest.approx(500.0)

    # AAPL collapses 10%: the stop fires and the realised loss (~$25 on a
    # $250 slot) takes equity below 97% of start_equity.
    engine.quotes['AAPL'] = 90.0
    engine.quotes['NVDA'] = 100.0
    s2 = trader.cycle(now=NOW + timedelta(minutes=5))
    assert [x['exit_reason'] for x in s2['exits']] == ['sl']
    assert s2['equity'] <= 485.0
    assert s2['loss_tripped'] is True and s2['loss_announce'] is True
    assert s2['entries'] == []
    assert ('NVDA', 'daily loss limit') in s2['skipped']
    assert budget.get_day_state('2026-09-14')['loss_tripped_at'] is not None

    # Exits still run while tripped; only the tripping cycle announces.
    engine.quotes['MSFT'] = 101.0
    s3 = trader.cycle(now=NOW + timedelta(minutes=10))
    assert [x['exit_reason'] for x in s3['exits']] == ['tp']
    assert s3['loss_tripped'] is True and s3['loss_announce'] is False
    assert s3['entries'] == []
    assert ('NVDA', 'daily loss limit') in s3['skipped']

    # Restart on the same date: fresh objects on the same file stay blocked,
    # do not re-announce, and keep the original start_equity.
    budget2 = BudgetTracker(str(tmp_path / 'bot.db'))
    trader2 = make_trader(engine, budget2)
    s4 = trader2.cycle(now=NOW + timedelta(minutes=15))
    assert s4['loss_tripped'] is True and s4['loss_announce'] is False
    assert s4['entries'] == []
    assert ('NVDA', 'daily loss limit') in s4['skipped']
    assert budget2.get_day_state('2026-09-14')['start_equity'] == pytest.approx(500.0)

    # Next ET date: a new baseline, entries allowed again.
    s5 = trader2.cycle(now=NEXT_DAY)
    assert s5['loss_tripped'] is False and s5['loss_announce'] is False
    assert len(s5['entries']) == 2
    assert 'daily loss limit' not in [reason for _, reason in s5['skipped']]
    assert budget2.get_day_state('2026-09-15')['start_equity'] == pytest.approx(s4['equity'])


# --- signal log + summary shape --------------------------------------------

def test_signals_recorded_each_cycle_and_marked_on_entry(tmp_path, monkeypatch):
    monkeypatch.setenv('FAST_MAX_POSITIONS', '1')
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL', 'MSFT'], {'AAPL': 100.0, 'MSFT': 50.0})
    engine.probs['MSFT'] = 0.4                      # scored, but below the 0.5 bar
    trader = make_trader(engine, budget)

    s = trader.cycle(now=NOW)
    for key in ('state', 'desc', 'exits', 'entries', 'skipped', 'candidates', 'ts',
                'stocks_open', 'crypto', 'rows', 'minutes_to_close', 'bars', 'bar',
                'gates', 'blocked_classes', 'loss_tripped', 'loss_announce',
                'equity', 'buying_power', 'unsettled'):
        assert key in s, key
    assert [e['symbol'] for e in s['entries']] == ['AAPL']
    e = s['entries'][0]
    assert e['ref_price'] == 100.0
    assert e['price'] == pytest.approx(100.05)
    assert e['shares'] == pytest.approx(round(500 / 100.05, 6))
    assert e['cost'] == pytest.approx(e['shares'] * 100.05)
    assert e['fees'] == 0
    assert e['probability'] == 0.9
    assert trade_row(budget, e['trade_id'])['entry_probability'] == 0.9
    assert s['buying_power'] < 1.0                  # the single slot took everything

    rows = {r['symbol']: r for r in budget.conn.execute(
        "SELECT symbol, above_bar, executed_trade_id FROM signals")}
    assert set(rows) == {'AAPL', 'MSFT'}            # every scored symbol, not just candidates
    assert rows['AAPL']['above_bar'] == 1 and rows['MSFT']['above_bar'] == 0
    assert rows['AAPL']['executed_trade_id'] == e['trade_id']
    assert rows['MSFT']['executed_trade_id'] is None

    # The same 5m bar polled again adds no rows and keeps the executed id.
    trader.cycle(now=NOW + timedelta(minutes=1))
    assert signal_count(budget) == 2
    assert budget.conn.execute(
        "SELECT executed_trade_id FROM signals WHERE symbol = 'AAPL'"
    ).fetchone()[0] == e['trade_id']

    # A new bar adds one fresh row per symbol.
    engine.bar_ts = (NOW + timedelta(minutes=5)).isoformat()
    trader.cycle(now=NOW + timedelta(minutes=5))
    assert signal_count(budget) == 4


# --- skip reasons -----------------------------------------------------------

def test_skip_market_closed_and_class_gated(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL', 'BTC-USD'], {'AAPL': 100.0, 'BTC-USD': 100.0})
    trader = make_trader(engine, budget)

    s = trader.cycle(now=SATURDAY)
    assert s['state'] == 'closed' and s['stocks_open'] is False
    assert s['gates']['stock'] == (True, "Trading. Edge clears the 0.30% cost floor")
    assert s['gates']['crypto'] == (
        False, "blocked: take-profit 0.60% is below the 1.20% round-trip cost")
    assert s['blocked_classes'] == ['crypto']
    assert s['entries'] == []
    assert ('AAPL', 'market closed') in s['skipped']
    assert ('BTC-USD', 'class gated: blocked: take-profit 0.60% is below the '
                       '1.20% round-trip cost') in s['skipped']
    assert signal_count(budget) == 2                # crypto keeps the cycle (and the log) alive


def test_eod_flatten_and_too_close_to_the_bell(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL'], {'AAPL': 100.0})
    open_position(budget, 'AAPL', 100.0, 1.0)
    trader = make_trader(engine, budget)

    s = trader.cycle(now=NEAR_BELL)
    assert s['state'] == 'open'
    assert [x['exit_reason'] for x in s['exits']] == ['eod']
    assert s['exits'][0]['reason'] == 'end of day (5 min to close)'
    assert s['entries'] == []
    assert ('AAPL', 'too close to the bell') in s['skipped']


def test_skip_below_min_order(tmp_path, monkeypatch):
    # $2 / 3 slots = $0.67 per slot, under MIN_ORDER_USD; buying power ($2)
    # is not the binding constraint, the slot size is.
    budget = make_budget(tmp_path, monkeypatch, starting_cash='2')
    engine = FakeEngine(['AAPL'], {'AAPL': 100.0})
    trader = make_trader(engine, budget)

    s = trader.cycle(now=NOW)
    assert s['entries'] == []
    assert s['skipped'] == [('AAPL', 'below min order')]


def test_skip_insufficient_buying_power_while_proceeds_unsettled(tmp_path, monkeypatch):
    monkeypatch.setenv('FAST_MAX_POSITIONS', '1')
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL', 'MSFT'], {'AAPL': 100.0})
    trader = make_trader(engine, budget)

    s1 = trader.cycle(now=NOW)
    assert [e['symbol'] for e in s1['entries']] == ['AAPL']

    # AAPL takes profit; in a cash account the proceeds are unsettled until
    # the next trading day's open, so the freed slot cannot be refilled.
    # (get_unsettled filters created_at >= account.opened_at, which
    # make_budget pinned to 2026-09-01, so the SELL stamped NOW is counted
    # whatever the real date is.)
    engine.quotes['AAPL'] = 101.0
    engine.quotes['MSFT'] = 50.0
    s2 = trader.cycle(now=NOW + timedelta(minutes=5))
    assert [x['exit_reason'] for x in s2['exits']] == ['tp']
    assert s2['unsettled'] > 0
    assert s2['buying_power'] < 1.0
    assert s2['entries'] == []
    assert ('MSFT', 'insufficient buying power') in s2['skipped']


def test_closed_market_without_crypto_returns_early(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL'], {'AAPL': 100.0})
    trader = make_trader(engine, budget)

    s = trader.cycle(now=SATURDAY)
    assert s['entries'] == [] and s['exits'] == [] and s['skipped'] == []
    assert 'not trading' in s['note']
    assert s['gates']['stock'][0] is True
    assert s['equity'] == pytest.approx(500.0)
    assert s['buying_power'] == pytest.approx(500.0)
    assert s['unsettled'] == 0
    assert signal_count(budget) == 0
    assert budget.get_day_state('2026-09-12') is None


# --- signal log must not share the ledger's transaction ----------------------

class _Abort(Exception):
    """Raised inside the simulated Discord transaction so _txn rolls it back."""


def test_signal_log_never_commits_a_ledger_transaction_open_on_another_thread(
        tmp_path, monkeypatch):
    """The Discord approval path (/buy, /sell) calls budget.execute_trade from
    the event-loop thread while the cycle runs in a worker thread. If the
    cycle wrote the signal log through budget.conn, signal_log's `with conn:`
    would commit whatever that other transaction had half-written (cash
    debited, position not yet booked). Simulate exactly that: another thread
    is inside budget._txn() with an uncommitted cash debit at the moment the
    cycle reaches scan_all; the cycle's signal-log write must neither commit
    that debit nor land its rows inside that transaction.
    """
    import threading
    import time

    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['BTC-USD'], {'BTC-USD': 100.0})
    trader = make_trader(engine, budget)

    txn_open = threading.Event()
    seen = {}

    def discord_thread():
        # Mirrors execute_trade: locked BEGIN IMMEDIATE on budget.conn, a cash
        # debit, then (here) a failure that must roll the debit back.
        try:
            with budget._txn():
                budget.conn.execute("UPDATE account SET cash = cash - 100 WHERE id = 1")
                txn_open.set()
                time.sleep(0.5)            # the cycle's signal-log write happens now
                seen['signals_inside_txn'] = budget.conn.execute(
                    "SELECT COUNT(*) FROM signals").fetchone()[0]
                seen['in_transaction'] = budget.conn.in_transaction
                raise _Abort
        except _Abort:
            pass

    real_scan_all = engine.scan_all

    def scan_all_with_concurrent_approval():
        t = threading.Thread(target=discord_thread)
        t.start()
        assert txn_open.wait(5), "simulated approval never opened its transaction"
        seen['thread'] = t
        return real_scan_all()

    engine.scan_all = scan_all_with_concurrent_approval

    # Saturday + crypto-only universe: the cycle runs (crypto keeps it alive),
    # records the scan, and has no entries (crypto is cost-gated), so the only
    # ledger write after scan_all is the signal log's own.
    s = trader.cycle(now=SATURDAY)
    seen['thread'].join(5)
    assert not seen['thread'].is_alive()

    assert ('BTC-USD', 'class gated: blocked: take-profit 0.60% is below the '
                       '1.20% round-trip cost') in s['skipped']
    # The other thread's transaction was still open and untouched by the
    # cycle: no signal rows inside it, and not committed under its feet.
    assert seen['in_transaction'] is True
    assert seen['signals_inside_txn'] == 0
    assert budget.get_cash() == pytest.approx(500.0)        # the debit rolled back
    assert not budget.conn.in_transaction
    # The signal log itself still landed, on its own connection.
    assert signal_count(budget) == 1
