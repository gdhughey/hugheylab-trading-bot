"""src.webapp: the state document and the Prometheus exposition.

A stub TradingBot with a real ledger on disk; no network, no aiohttp server.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.budget_tracker import BudgetTracker
from src.collector import ensure_schema
from src.database import Database
from src.webapp import LLM, WebApp

OPENED = datetime(2026, 9, 1, tzinfo=timezone.utc)
T = datetime(2026, 9, 17, 14, 0, tzinfo=timezone.utc)


class Engine:
    def __init__(self):
        self.calls = 0

    def latest_price(self, sym):
        self.calls += 1
        return 101.0

    def stored_close(self, sym):
        return 100.0


class Analyst:
    backend_name = 'local qwen3-8b'
    enabled = True


@pytest.fixture
def bot(tmp_path, monkeypatch):
    monkeypatch.setenv('STARTING_CASH', '500')
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '0')
    monkeypatch.setenv('WEB_PRICE_TTL_S', '30')
    path = str(tmp_path / 'w.db')
    db = Database(path, now=OPENED)
    ensure_schema(db.conn)
    bt = BudgetTracker(path)
    bt.ensure_day_state('2026-09-17', 500.0)
    a = bt.log_trade('AAPL', 'BUY', 100.0, 1.0, probability=0.52, now=T); bt.execute_trade(a, now=T)
    b = bt.log_trade('MSFT', 'BUY', 50.0, 2.0, probability=0.51, now=T); bt.execute_trade(b, now=T)
    c = bt.log_trade('MSFT', 'SELL', 49.0, 2.0, exit_reason='sl', now=T); bt.execute_trade(c, now=T)
    fast = SimpleNamespace(last_summary={'state': 'open', 'desc': 'regular session', 'rows': 93,
                                         'gates': {'stock': (True, 'ok')}, 'skipped': [('NVDA', 'cooling down')],
                                         'candidates': [], 'entries': [], 'exits': [], 'ts': T,
                                         'minutes_to_close': 120.0},
                           max_positions=3, open_delay_min=15)
    intraday = SimpleNamespace(metrics={'stock': {'precision': 0.33, 'breakeven': 0.375, 'ev': -0.001}},
                               threshold=lambda cls: 0.347, symbols=['AAPL', 'MSFT', 'NVDA'])
    return SimpleNamespace(budget_tracker=bt, engine=Engine(), fast=fast, intraday=intraday,
                           claude=Analyst(), warming_up=False)


def test_snapshot_reports_ledger_positions_and_health(bot, monkeypatch):
    monkeypatch.setenv('WEB_PORT', '0')
    w = WebApp(bot, port=8090)
    s = w.snapshot(now=T)
    assert s['account']['starting_cash'] == 500.0
    assert [p['symbol'] for p in s['positions']] == ['AAPL'] and s['positions'][0]['price'] == 101.0
    assert s['today']['n_trades'] == 3 and s['today']['losses'] == 1 and s['today']['realized'] == pytest.approx(-2.0, abs=0.01)   # -$2 less SEC/TAF fees
    assert s['since_open'] == {'n_closed': 1, 'wins': 0, 'win_rate': 0.0, 'exit_reasons': {'sl': 1}}
    assert s['fast']['state'] == 'open' and s['fast']['skipped'] == [['NVDA', 'cooling down']] or s['fast']['skipped'] == [('NVDA', 'cooling down')]
    assert s['models']['stock']['tradeable'] is True and s['models']['stock']['round_trip_cost'] > 0
    assert s['data']['counts']['news'] == 0 and s['data']['universe'] == 3
    assert s['llm']['backend'] == 'local qwen3-8b'
    assert s['trades'][0]['side'] == 'SELL'          # newest first


def test_quote_cache_spares_the_finnhub_quota(bot):
    w = WebApp(bot, port=8090)
    w.snapshot(); w.snapshot(); w.snapshot()
    assert bot.engine.calls == 1                    # one held symbol, one quote, cached after


def test_prometheus_exposition_is_well_formed(bot):
    LLM.record('local', True, 2.5, 'daily_brief')
    w = WebApp(bot, port=8090)
    text = w.prometheus()
    lines = [l for l in text.splitlines() if l and not l.startswith('#')]
    assert 'tradingbot_equity_usd ' in text
    assert 'tradingbot_exits_total{reason="sl"} 1.0' in text
    assert 'tradingbot_model_precision{asset_class="stock"} 0.33' in text
    assert 'tradingbot_llm_calls_total{backend="local",status="ok"}' in text
    assert 'tradingbot_fast_state 1.0' in text
    for l in lines:                                  # every sample: name[{labels}] number
        name, _, val = l.rpartition(' ')
        float(val)
        assert name and ' ' not in name.split('{')[0]
