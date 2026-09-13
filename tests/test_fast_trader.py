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
