"""Scorecard arithmetic and the go/no-go verdict (spec sections 4, 7, 8)."""
import math
import statistics
from datetime import datetime, timezone

import pytest

from src import scorecard
from src.budget_tracker import BudgetTracker
from src.database import Database

UTC = timezone.utc

# Barriers pinned so breakeven does not depend on data/tuned.json (absent in
# the test tree) or on FAST_TAKE_PROFIT in the environment.
TUNED = {'stock': {'take_profit': 0.010, 'stop_loss': 0.006, 'horizon': 24},
         'crypto': {'take_profit': 0.006, 'stop_loss': 0.004, 'horizon': 24}}

# With TUNED above and the default stock costs (5 bps slippage per side plus
# the SEC fee): c = 0.0010206, be = (0.006 + 0.0010206) / 0.016.
COST_STOCK = 0.0010206
BE_STOCK = 0.4387875


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr('src.intraday_engine.TUNED', TUNED)
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '5')
    monkeypatch.setenv('SEC_FEE_RATE', '0.0000206')
    monkeypatch.setenv('FINRA_TAF_PER_SHARE', '0.000195')
    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '60')


# --- wilson_ci -------------------------------------------------------------

def test_wilson_ci_empty():
    assert scorecard.wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_known_value():
    lo, hi = scorecard.wilson_ci(50, 100)
    assert lo == pytest.approx(0.404, abs=1e-3)
    assert hi == pytest.approx(0.596, abs=1e-3)


def test_wilson_ci_clamped_at_extremes():
    lo, hi = scorecard.wilson_ci(10, 10)
    assert hi == 1.0 and 0.7 < lo < 1.0
    lo, hi = scorecard.wilson_ci(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.3


# --- mean_ci ---------------------------------------------------------------

def _ledger(n):
    """+1.5% / -0.5% alternating: +0.5%/trade on average, with real spread."""
    return [0.015 if i % 2 == 0 else -0.005 for i in range(n)]


def test_mean_ci_too_few_values():
    assert scorecard.mean_ci([]) == (0.0, 0.0, 0.0)
    assert scorecard.mean_ci([0.01]) == (0.0, 0.0, 0.0)


def test_mean_ci_excludes_zero_and_shrinks_with_n():
    mean, se, hw = scorecard.mean_ci(_ledger(100))
    assert mean == pytest.approx(0.005)
    assert hw == pytest.approx(1.96 * se)
    assert mean - hw > 0                      # 100 trades at +0.5% excludes zero
    _, _, hw400 = scorecard.mean_ci(_ledger(400))
    assert hw400 == pytest.approx(hw / 2, rel=0.01)   # 4x the trades, half the width


# --- cost_breakeven --------------------------------------------------------

def test_cost_breakeven(pinned):
    assert scorecard.cost_breakeven('stock') == pytest.approx(BE_STOCK, abs=1e-6)
    # crypto: round trip 1.2% exceeds the 0.6% take-profit -> breakeven above 100%
    assert scorecard.cost_breakeven('crypto') == pytest.approx(1.6, abs=1e-6)


# --- verdict ---------------------------------------------------------------

def _stats(**over):
    base = dict(gated=False, sig_n=1800, sig_lo=0.46, sig_hi=0.55, sig_lo_day=0.45,
                exec_n=60, exec_mean=0.0025, exec_se=0.001, ev_bt=0.003, cost=COST_STOCK)
    base.update(over)
    return base


def test_verdict_go(pinned):
    label, text = scorecard.verdict('stock', _stats())
    assert label == 'GO'
    assert 'n=1800' in text and 'n=60' in text


def test_verdict_no_go_when_cost_gated(pinned):
    label, text = scorecard.verdict('crypto', _stats(gated=True, cost=0.012))
    assert label == 'NO-GO'
    assert 'cost gate' in text


def test_verdict_no_go_when_ci_below_breakeven(pinned):
    label, text = scorecard.verdict('stock', _stats(sig_lo=0.36, sig_hi=0.42, sig_lo_day=0.35))
    assert label == 'NO-GO'
    assert '42.0%' in text and '43.9%' in text


@pytest.mark.parametrize('over, marker', [
    (dict(sig_lo=0.43), 'signal CI lower 43.0%'),
    (dict(sig_lo_day=0.43), 'day-clustered lower 43.0%'),
    (dict(exec_n=59), 'n=59 < 60'),
    (dict(exec_mean=-0.002), 'executed mean -0.200%'),
    (dict(ev_bt=None), 'no backtest EV'),
    (dict(sig_n=0, sig_lo=0.0, sig_hi=1.0, sig_lo_day=0.0), 'n=0'),
])
def test_verdict_extend(pinned, over, marker):
    label, text = scorecard.verdict('stock', _stats(**over))
    assert label == 'EXTEND'
    assert marker in text


# --- build_scorecard -------------------------------------------------------

class FakeEngine:
    """Only what build_scorecard touches on MLEngine: stored closes and the SPY start."""

    def __init__(self, closes, spy_start=650.0):
        self.closes, self.spy_start, self.asked = closes, spy_start, None

    def stored_close(self, symbol):
        return self.closes.get(symbol)

    def first_close_on_or_after(self, symbol, date_iso):
        self.asked = (symbol, date_iso)
        return self.spy_start


class FakeIntraday:
    """Only what build_scorecard touches on IntradayEngine: the metrics dict."""

    def __init__(self, metrics):
        self.metrics = metrics


OPENED_AT = '2026-09-10T12:00:00+00:00'
# ev 0.004 clears the default MIN_EV_TO_TRADE of 0.003, so class_gate allows
# stocks whether or not FAST_IGNORE_EV is set in the test environment.
STOCK_METRICS = {'precision': 0.55, 'test_signals': 1234, 'ev': 0.004,
                 'breakeven': 0.375, 'take_profit': 0.01, 'stop_loss': 0.006}


def _t(day, hh, mm):
    return datetime(2026, 9, day, hh, mm, tzinfo=UTC)


def _round_trip(budget, sym, ref_in, ref_out, reason, day, hh):
    """BUY 1 unit at hh:00 UTC, SELL it at hh:30 with the given exit reason."""
    b = budget.log_trade(sym, 'BUY', ref_in, 1.0, probability=0.6, now=_t(day, hh, 0))
    budget.execute_trade(b, now=_t(day, hh, 0))
    s = budget.log_trade(sym, 'SELL', ref_out, 1.0, exit_reason=reason, now=_t(day, hh, 30))
    budget.execute_trade(s, now=_t(day, hh, 30))


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """A $500 cash account: four closed stock trades, one open, a signal log.

    Stock costs are zeroed so every dollar below is exact; crypto keeps its
    60 bps per side so the cost gate still fires for it.
    Cash: 500 -100 +99 -100 +101 -100 +99 -50 +50.5 -100 = 399.5.
    Monday's BUYs total 350 <= the 499 settled at Monday's open: a cash
    account cannot spend Monday's own SELL proceeds until Tuesday 09:30 ET,
    and execute_trade REJECTS a BUY that exceeds cash - unsettled.
    """
    monkeypatch.setattr('src.intraday_engine.TUNED', TUNED)
    monkeypatch.setenv('STARTING_CASH', '500')
    monkeypatch.setenv('ACCOUNT_TYPE', 'cash')
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '0')
    monkeypatch.setenv('SEC_FEE_RATE', '0')
    monkeypatch.setenv('FINRA_TAF_PER_SHARE', '0')
    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '60')
    path = str(tmp_path / 'scorecard.db')
    Database(path)
    budget = BudgetTracker(path)
    with budget.conn:
        budget.conn.execute("UPDATE account SET opened_at = ?", (OPENED_AT,))

    # Fri 2026-09-11: one loser (-1%). Equity 499 at the close.
    _round_trip(budget, 'AMD', 100.0, 99.0, 'sl', 11, 14)
    budget.record_equity('2026-09-11', {}, now=_t(11, 21, 0))

    # Mon 2026-09-14: tp +1%, sl -1%, timeout +1%, then TSLA left open.
    _round_trip(budget, 'AAPL', 100.0, 101.0, 'tp', 14, 14)
    _round_trip(budget, 'MSFT', 100.0, 99.0, 'sl', 14, 15)
    _round_trip(budget, 'NVDA', 50.0, 50.5, 'timeout', 14, 16)
    b = budget.log_trade('TSLA', 'BUY', 100.0, 1.0, probability=0.6, now=_t(14, 17, 0))
    budget.execute_trade(b, now=_t(14, 17, 0))
    budget.record_equity('2026-09-14', {'TSLA': 102.0}, now=_t(14, 20, 5))

    # Signal log: 8 labelled above-bar stock signals since open (5 hits) across
    # two days, plus rows that must be EXCLUDED: below bar, unlabelled, before
    # opened_at. One crypto signal.
    sig_rows = [
        # bar_ts, symbol, cls, above_bar, trade_date, label
        ('2026-09-11T14:00:00+00:00', 'AAPL', 'stock', 1, '2026-09-11', 1),
        ('2026-09-11T14:05:00+00:00', 'MSFT', 'stock', 1, '2026-09-11', 1),
        ('2026-09-11T14:10:00+00:00', 'NVDA', 'stock', 1, '2026-09-11', 1),
        ('2026-09-11T14:15:00+00:00', 'AMD', 'stock', 1, '2026-09-11', 0),
        ('2026-09-14T14:00:00+00:00', 'AAPL', 'stock', 1, '2026-09-14', 1),
        ('2026-09-14T14:05:00+00:00', 'MSFT', 'stock', 1, '2026-09-14', 0),
        ('2026-09-14T14:10:00+00:00', 'NVDA', 'stock', 1, '2026-09-14', 1),
        ('2026-09-14T14:15:00+00:00', 'AMD', 'stock', 1, '2026-09-14', 0),
        ('2026-09-14T14:20:00+00:00', 'TSLA', 'stock', 0, '2026-09-14', 1),      # below bar
        ('2026-09-14T14:25:00+00:00', 'META', 'stock', 1, '2026-09-14', None),   # unlabelled
        ('2026-09-09T14:00:00+00:00', 'AAPL', 'stock', 1, '2026-09-09', 1),      # before open
        ('2026-09-14T14:00:00+00:00', 'BTC-USD', 'crypto', 1, '2026-09-14', 1),
    ]
    with budget.conn:
        budget.conn.executemany(
            "INSERT INTO signals (bar_ts, symbol, asset_class, probability, bar, above_bar, "
            "ref_price, trade_date, label, labeled_at) VALUES (?, ?, ?, 0.5, 0.4, ?, 100.0, ?, ?, ?)",
            [(ts, sym, cls, ab, td, lab, None if lab is None else '2026-09-14T20:05:00+00:00')
             for ts, sym, cls, ab, td, lab in sig_rows])

    budget.ensure_day_state('2026-09-14', 499.0)
    budget.set_day_flag('2026-09-14', 'loss_tripped_at', '2026-09-14T15:00:00+00:00')
    return budget


def test_build_scorecard_full(ledger):
    engine = FakeEngine({'TSLA': 102.0, 'SPY': 660.0})
    card = scorecard.build_scorecard(ledger, engine, FakeIntraday({'stock': STOCK_METRICS}),
                                     '2026-09-14', now=_t(14, 21, 0))

    nets = [-1.0, 1.0, -1.0, 0.5]
    hw = 1.96 * statistics.stdev(nets) / math.sqrt(4)
    head = card['headline']
    assert head['n_closed'] == 4
    assert head['all_time_net'] == pytest.approx(1.5)          # -0.5 realised + 2.0 open
    assert head['ci_dollars'] == pytest.approx(hw * 4)

    acct = card['account']
    assert acct['starting_cash'] == 500.0
    assert acct['cash'] == pytest.approx(399.5)
    assert acct['equity'] == pytest.approx(501.5)
    assert acct['unsettled'] == pytest.approx(250.5)            # today's three SELLs, T+1
    assert acct['buying_power'] == pytest.approx(149.0)         # cash - unsettled
    # An ET DATE string, not a timestamp: the embed prints it verbatim as
    # "settles 2026-09-15 09:30 ET" (Task 10 Step 12 looks for that literal).
    assert acct['unsettled_until'] == '2026-09-15'
    assert isinstance(acct['unsettled_until'], str) and len(acct['unsettled_until']) == 10
    assert acct['fees_paid'] == 0.0
    assert acct['gross_pnl'] == pytest.approx(-0.5)

    today = card['today']
    assert today['realized'] == pytest.approx(0.5)
    assert today['unrealized'] == pytest.approx(2.0)
    assert today['n_trades'] == 7
    assert (today['wins'], today['losses']) == (2, 1)
    assert today['loss_tripped'] is True
    assert [c['symbol'] for c in card['closed_today']] == ['AAPL', 'MSFT', 'NVDA']
    assert [c['exit_reason'] for c in card['closed_today']] == ['tp', 'sl', 'timeout']
    first = card['closed_today'][0]
    # Both spellings of the P&L keys are guaranteed (see Contract additions).
    assert first['net'] == pytest.approx(1.0) and first['realized_pnl'] == first['net']
    assert first['gross'] == pytest.approx(1.0) and first['gross_pnl'] == first['gross']
    assert set(first) == {'symbol', 'shares', 'price', 'amount', 'net', 'realized_pnl',
                          'gross', 'gross_pnl', 'fees', 'exit_reason', 'created_at'}

    assert len(card['positions']) == 1
    pos = card['positions'][0]
    assert pos['symbol'] == 'TSLA' and pos['price'] == 102.0
    assert pos['pct_vs_ref'] == pytest.approx(0.02)

    so = card['since_open']
    assert so['n_closed'] == 4 and so['win_rate'] == 0.5
    assert (so['win_lo'], so['win_hi']) == scorecard.wilson_ci(2, 4)
    assert so['mean_net'] == pytest.approx(-0.125)
    assert so['mean_ci'] == pytest.approx(hw)
    assert so['profit_factor'] == pytest.approx(1.5 / 2.0)
    # PERCENTAGE POINTS: 500 -> 499 on day 2 is a 0.2% fall, reported as 0.2.
    assert so['max_drawdown_pct'] == pytest.approx(0.2)
    assert so['days_running'] == 5                              # 10th..14th inclusive

    st = card['classes']['stock']
    assert st['gated'] is False and st['verdict'] == 'EXTEND'
    assert not st['gate_text'].startswith('blocked')
    assert st['exec_n'] == 4 and st['exec_mean_pct'] == pytest.approx(0.0)
    # PERCENTAGE POINTS: the fraction half-width times 100.
    assert st['exec_ci'] == pytest.approx(100 * 1.96 * statistics.stdev([-0.01, 0.01, -0.01, 0.01]) / 2)
    assert st['exec_ci'] == pytest.approx(100 * scorecard.mean_ci([-0.01, 0.01, -0.01, 0.01])[2])
    assert st['exits_by_reason']['tp'] == {'n': 1, 'mean_net': pytest.approx(1.0)}
    assert st['exits_by_reason']['sl'] == {'n': 2, 'mean_net': pytest.approx(-1.0)}
    assert st['exits_by_reason']['timeout'] == {'n': 1, 'mean_net': pytest.approx(0.5)}
    assert st['tp_n'] == 4 and st['tp_first_rate'] == 0.25
    assert (st['tp_lo'], st['tp_hi']) == scorecard.wilson_ci(1, 4)
    assert st['bt_precision'] == 0.55 and st['bt_n'] == 1234
    assert st['ev_bt'] == 0.004 and st['ev_bt_net'] == pytest.approx(0.004)
    assert st['cost'] == 0.0 and st['breakeven'] == pytest.approx(0.375)
    assert st['sig_n'] == 8 and st['sig_hits'] == 5 and st['sig_rate'] == 0.625
    assert (st['sig_lo'], st['sig_hi']) == scorecard.wilson_ci(5, 8)
    assert st['sig_days'] == 2
    assert st['sig_lo_day'] == pytest.approx(
        0.625 - 1.96 * statistics.stdev([0.75, 0.5]) / math.sqrt(2))

    cr = card['classes']['crypto']
    assert cr['gated'] is True and cr['verdict'] == 'NO-GO'
    assert cr['gate_text'].startswith('blocked')
    assert cr['exec_n'] == 0 and cr['sig_n'] == 1 and cr['sig_lo_day'] == 0.0
    # No metrics entry for crypto -> the backtest fields are None, not 0.
    assert cr['bt_precision'] is None and cr['ev_bt'] is None and cr['ev_bt_net'] is None
    assert cr['bt_n'] == 0

    assert engine.asked == ('SPY', '2026-09-10')
    assert card['spy']['start_close'] == 650.0 and card['spy']['last_close'] == 660.0
    assert card['spy']['pct'] == pytest.approx(660 / 650 - 1)
    assert card['spy']['value'] == pytest.approx(500 * 660 / 650)


def test_build_scorecard_without_benchmark_or_metrics(ledger):
    """intraday=None (fast mode off / training failed): every class's backtest
    fields are None and the verdict still comes out, so the 16:05 report can
    post on a day the model did not train."""
    engine = FakeEngine({'TSLA': 102.0}, spy_start=None)
    # Built on Tue 2026-09-15 after 09:30 ET (a /pnl for a past day): every
    # SELL in the ledger, Monday's included, has settled by then.
    card = scorecard.build_scorecard(ledger, engine, None, '2026-09-11', now=_t(15, 14, 0))
    assert card['spy'] is None
    for cls in ('stock', 'crypto'):
        c = card['classes'][cls]
        assert c['bt_precision'] is None and c['bt_n'] == 0
        assert c['ev_bt'] is None and c['ev_bt_net'] is None
    st = card['classes']['stock']
    assert st['verdict'] == 'EXTEND' and 'no backtest EV' in st['verdict_text']
    assert card['classes']['crypto']['verdict'] == 'NO-GO'
    # 2026-09-11 has no day_state row and only the AMD round trip.
    assert card['today']['loss_tripped'] is False
    assert card['today']['n_trades'] == 2
    assert card['today']['realized'] == pytest.approx(-1.0)
    assert [c['symbol'] for c in card['closed_today']] == ['AMD']
    assert card['closed_today'][0]['net'] == pytest.approx(-1.0)
    assert card['closed_today'][0]['realized_pnl'] == pytest.approx(-1.0)
    assert card['since_open']['days_running'] == 2
    # AMD's Friday proceeds settled Monday 09:30 ET and Monday's three SELLs
    # settled Tuesday 09:30 ET, both before `now`: nothing pending.
    assert card['account']['unsettled'] == 0.0
    assert card['account']['unsettled_until'] is None
