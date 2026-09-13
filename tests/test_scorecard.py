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
