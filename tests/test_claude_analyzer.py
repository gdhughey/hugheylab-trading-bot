"""src.claude_analyzer: backend routing, number grounding, context rendering.

No network: the local transport is monkeypatched. No wall clock: `now` is
passed everywhere it matters.
"""
import asyncio
import sqlite3
from datetime import datetime, timezone

import pytest

from src import claude_analyzer as ca
from src.claude_analyzer import (ClaudeAnalyzer, build_risk_context, load_calendar_and_news,
                                 render_positions, render_scorecard, unsourced_numbers)
from src.collector import ensure_schema
from src.database import connect

NOW = datetime(2026, 9, 17, 18, 0, tzinfo=timezone.utc)


# --- grounding ----------------------------------------------------------------

@pytest.mark.parametrize('prompt, answer, bad', [
    ("equity $502.08, win rate 51.1%", "Equity is $502.08 with a 51.1% win rate.", []),
    ("equity $502.08", "Equity rose 4.2% to $502.08.", ['4.2']),
    ("45 trades", "45 trades; 2 wins and 3 losses.", []),          # 0-3 ignored (ordinals)
    ("1,250 shares", "1250 shares", []),                            # comma-insensitive
    ("max drawdown 0.90%", "drawdown of 0.9%", []),                 # trailing zeros
    ("", "The Fed cut rates by 25bps on 2026-09-16.", ['16', '2026', '25', '9']),   # an invented date is three unsourced numbers
])
def test_unsourced_numbers(prompt, answer, bad):
    assert unsourced_numbers(prompt, answer) == bad


# --- rendering ----------------------------------------------------------------

def _scorecard():
    return {
        'headline': {'all_time_net': 2.08, 'all_time_pct': 0.00416, 'n_closed': 45, 'ci_dollars': 3.5},
        'account': {'equity': 502.08, 'cash': 400.0, 'buying_power': 390.0, 'unsettled': 10.0,
                    'unsettled_until': None, 'starting_cash': 500.0, 'gross_pnl': 3.0, 'fees_paid': 0.92},
        'today': {'realized': -0.8, 'unrealized': 0.4, 'n_trades': 4, 'wins': 2, 'losses': 1,
                  'loss_tripped': False},
        'closed_today': [{'symbol': 'NVDA', 'shares': 0.5, 'price': 180.0, 'amount': 90.0,
                          'net': -0.8, 'realized_pnl': -0.8, 'gross': -0.7, 'gross_pnl': -0.7,
                          'fees': 0.1, 'exit_reason': 'sl', 'created_at': '2026-09-17T15:00:00+00:00'}],
        'positions': [{'symbol': 'AAPL', 'shares': 0.3, 'avg_price': 230.0, 'entry_ref': 229.9,
                       'price': 231.0, 'market_value': 69.3, 'pnl': 0.3, 'pnl_pct': 0.0043,
                       'cost_basis': 69.0}],
        'since_open': {'n_closed': 45, 'win_rate': 0.511, 'win_lo': 0.37, 'win_hi': 0.65,
                       'mean_net': 0.05, 'mean_ci': 0.08, 'profit_factor': 1.1,
                       'max_drawdown_pct': 0.9, 'days_running': 4},
        'classes': {'stock': {'verdict': 'EXTEND', 'bt_precision': 0.315, 'gated': False, 'exec_n': 45},
                    'crypto': {'verdict': 'NO-GO', 'bt_precision': None, 'gated': True, 'exec_n': 0}},
        'spy': {'start_close': 640.0, 'last_close': 646.4, 'pct': 0.01, 'value': 505.0},
    }


def test_render_scorecard_carries_every_number_the_brief_may_quote():
    text = render_scorecard(_scorecard())
    for needle in ['$502.08', '$500.00', '0.42%', '45 closed trades', '51.1%', '0.90%',
                   'NVDA', 'exit reason sl', 'stock: verdict EXTEND', '31.5%',
                   'crypto: verdict NO-GO', 'cost gate', 'SPY', '1.00%', 'day 4']:
        assert needle in text, needle
    assert 'LOSS LIMIT' not in text


def test_render_positions_handles_stale_and_empty():
    assert render_positions([]) == "Open positions: none."
    t = render_positions([{'symbol': 'X', 'shares': 2, 'avg_price': 5.0, 'price': None,
                           'pnl': None, 'pnl_pct': None}])
    assert 'price unavailable (stale)' in t and 'cost basis $10.00' in t


def test_build_risk_context_states_the_rules():
    pnl = {'positions': [], 'equity': 500.0, 'cash': 500.0, 'cost_basis': 0.0,
           'unrealized': 0.0, 'stale': ['DOT-USD']}
    t = build_risk_context(pnl, max_positions=3, stop_loss_pct=0.007, take_profit_pct=0.012,
                           daily_loss_limit_pct=3)
    assert 'max 3 positions' in t and 'stop-loss 0.7%' in t and 'take-profit 1.2%' in t
    assert '3% daily loss' in t and 'stale): DOT-USD' in t


def test_load_calendar_and_news_reads_collector_tables(tmp_path):
    conn = connect(str(tmp_path / 'n.db'))
    ensure_schema(conn)
    with conn:
        conn.execute("INSERT INTO events VALUES ('earnings','AAPL','2026-09-18','amc','{}','x')")
        conn.execute("INSERT INTO events VALUES ('earnings','AAPL','2026-12-01','amc','{}','x')")  # too far
        conn.execute("INSERT INTO news (id,symbol,published_at,fetched_at,headline) "
                     "VALUES (1,'AAPL','2026-09-17T12:00:00+00:00','x','Apple ships a thing')")
        conn.execute("INSERT INTO news (id,symbol,published_at,fetched_at,headline) "
                     "VALUES (2,'AAPL','2026-09-10T12:00:00+00:00','x','old news')")   # >24h
    t = load_calendar_and_news(conn, ['AAPL', 'BTC-USD'], now=NOW)
    assert 'AAPL on 2026-09-18 (amc)' in t and '2026-12-01' not in t
    assert 'AAPL 1' in t and 'Apple ships a thing' in t and 'old news' not in t
    # crypto is skipped, and no stocks at all is a clean line
    assert load_calendar_and_news(conn, ['BTC-USD'], now=NOW).startswith('Calendar and news: no stock')


def test_load_calendar_and_news_without_collector_tables(tmp_path):
    conn = sqlite3.connect(str(tmp_path / 'bare.db'))
    t = load_calendar_and_news(conn, ['AAPL'], now=NOW)
    assert 'not collected yet' in t


# --- routing ------------------------------------------------------------------

@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(ca, 'LLM_BASE_URL', 'http://gpu:8081/v1')
    monkeypatch.delenv('CLAUDE_API_KEY', raising=False)
    a = ClaudeAnalyzer()
    assert a.enabled and a.backend_name == 'local qwen3-8b' and a.client is None
    return a


def test_daily_brief_goes_local_and_passes_facts(local, monkeypatch):
    seen = {}

    async def fake_local(user, max_tokens=400, temperature=0.3):
        seen['user'] = user
        return "Equity stands at $502.08 after 45 closed trades."

    monkeypatch.setattr(local, '_ask_local', fake_local)
    out = asyncio.run(local.daily_market_analysis("equity $502.08 over 45 closed trades"))
    assert out == "Equity stands at $502.08 after 45 closed trades."
    assert seen['user'].startswith('FACTS:\nequity $502.08')


def test_invented_number_is_flagged_not_hidden(local, monkeypatch):
    async def fake_local(user, max_tokens=400, temperature=0.3):
        return "Equity is $502.08, up 4.2% on the week."
    monkeypatch.setattr(local, '_ask_local', fake_local)
    out = asyncio.run(local.analyze_portfolio_risk([], "equity $502.08"))
    assert out.startswith("Equity is $502.08") and "⚠️ Not in the source data: 4.2" in out


def test_no_context_brief_refuses_to_guess(local):
    out = asyncio.run(local.daily_market_analysis(None))
    assert 'No data was supplied' in out


def test_weekly_prefers_claude_then_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(ca, 'LLM_BASE_URL', 'http://gpu:8081/v1')
    monkeypatch.setenv('CLAUDE_API_KEY', 'sk-test')
    a = ClaudeAnalyzer()
    assert a.client is not None
    calls = []

    async def fake_claude(user, max_tokens=4000):
        calls.append('claude')
        raise RuntimeError('credit balance too low')

    async def fake_local(user, max_tokens=400, temperature=0.3):
        calls.append('local')
        return "ok"

    monkeypatch.setattr(a, '_ask_claude', fake_claude)
    monkeypatch.setattr(a, '_ask_local', fake_local)
    assert asyncio.run(a.weekly_portfolio_review([], [], "flat")) == "ok"
    assert calls == ['claude', 'local']


def test_local_error_surfaces_without_raising(local, monkeypatch):
    async def boom(user, max_tokens=400, temperature=0.3):
        raise ConnectionError('gpu box down')
    monkeypatch.setattr(local, '_ask_local', boom)
    out = asyncio.run(local.analyze_portfolio_risk([], "x"))
    assert out.startswith('AI analyst unavailable') and 'gpu box down' in out


def test_disabled_when_nothing_configured(monkeypatch):
    monkeypatch.setattr(ca, 'LLM_BASE_URL', '')
    monkeypatch.delenv('CLAUDE_API_KEY', raising=False)
    a = ClaudeAnalyzer()
    assert not a.enabled and a.backend_name == 'disabled'
