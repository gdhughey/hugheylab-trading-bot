"""Discord embeds built offline: no gateway login, no network.

TradingBot.__init__ builds its own BudgetTracker / IntradayEngine / FastTrader,
so those names are patched on the module to fakes (the tracker is real, on the
tmp-path DB). commands.Bot and discord.Embed construct without a connection.

The synthetic scorecard below is copied from Task 7's contract additions
(tests/test_scorecard.py::test_build_scorecard_full is the producer-side
twin): closed_today uses net/gross, unsettled_until is a bare ET date,
exec_mean_pct / exec_ci / max_drawdown_pct are percentage points, and a class
without a trained model has bt_precision / ev_bt / ev_bt_net None.
"""
import asyncio
import types
from datetime import datetime, timezone

import discord
import pytest

from src import discord_bot
from src.budget_tracker import BudgetTracker

# Walk-forward metrics from the spec's gate test: stock below the EV floor,
# crypto above zero but under it. (data/ is not shipped to the dev copy, so
# barriers() falls back to FAST_TAKE_PROFIT / FAST_STOP_LOSS defaults and the
# crypto cost gate does NOT fire here - nothing below depends on it firing.)
METRICS = {
    'stock': {'asset_class': 'stock', 'symbols': 80, 'rows': 291156,
              'positive_rate': 0.25, 'interval': '5m', 'horizon_bars': 24,
              'take_profit': 0.01, 'stop_loss': 0.006, 'precision': 0.357,
              'breakeven': 0.375, 'ev': -0.0003, 'test_signals': 947, 'bar': 0.507},
    'crypto': {'asset_class': 'crypto', 'symbols': 13, 'rows': 172332,
               'positive_rate': 0.26, 'interval': '5m', 'horizon_bars': 24,
               'take_profit': 0.006, 'stop_loss': 0.004, 'precision': 0.463,
               'breakeven': 0.40, 'ev': 0.0006, 'test_signals': 378, 'bar': 0.525},
}


class FakeEngine:
    symbols = []
    providers = []
    quote_providers = []
    source_stats = {}

    def latest_price(self, symbol):
        return None            # every held position is valued at avg_price

    def stored_close(self, symbol):
        return None            # no SPY history -> scorecard 'spy' is None

    def first_close_on_or_after(self, symbol, date_iso):
        return None


class FakeIntraday:
    def __init__(self, metrics):
        self.metrics = dict(metrics)
        self.last_metrics = {}
        self.symbols = ['AAPL', 'BTC-USD']

    def scan_all(self):
        return []


class FakeFast:
    max_positions = 3
    eod_flatten_min = 10.0
    cooldown_min = 15.0
    take_profit = 0.008
    stop_loss = 0.005


class FakeChannel:
    """Records what the bot tried to send; can refuse embeds the way Discord does."""

    def __init__(self, fail_embeds=False, fail_all=False):
        self.embeds, self.texts = [], []
        self.fail_embeds, self.fail_all = fail_embeds, fail_all

    async def send(self, content=None, *, embed=None):
        if self.fail_all:
            raise RuntimeError('gateway down')
        if embed is not None:
            if self.fail_embeds:
                raise discord.HTTPException(
                    types.SimpleNamespace(status=400, reason='Bad Request'),
                    'Invalid Form Body')
            self.embeds.append(embed)
        else:
            self.texts.append(content)


def _make_bot(db_path, monkeypatch, metrics=METRICS, ignore_ev='1'):
    monkeypatch.setenv('FAST_MODE', '1')
    monkeypatch.setenv('FAST_IGNORE_EV', ignore_ev)
    monkeypatch.setenv('MIN_EV_TO_TRADE', '0.003')
    monkeypatch.delenv('CLAUDE_API_KEY', raising=False)
    monkeypatch.setattr(discord_bot, 'BudgetTracker', lambda: BudgetTracker(db_path))
    monkeypatch.setattr(discord_bot, 'IntradayEngine', lambda: FakeIntraday(metrics))
    monkeypatch.setattr(discord_bot, 'FastTrader', lambda *a, **k: FakeFast())
    return discord_bot.TradingBot(engine=FakeEngine())


def _stub_report_plumbing(bot, monkeypatch, chan, stub_embed=True):
    """Point the bot at a FakeChannel and neutralise the two pieces the daily
    report tests are not about (signal labelling, and optionally the embed)."""
    async def dest():
        return chan
    monkeypatch.setattr(bot, '_destination', dest)
    monkeypatch.setattr(discord_bot.signal_log, 'label_pending', lambda conn, now=None: 0)
    if stub_embed:
        monkeypatch.setattr(bot, '_scorecard_embed',
                            lambda day_et, now=None: discord.Embed(title='stub'))


NOW = datetime(2026, 9, 19, 20, 10, tzinfo=timezone.utc)   # Saturday 16:10 ET
DAY = '2026-09-19'
# conftest OPENED_AT is 2026-01-01T00:00Z, which is 19:00 ET on New Year's
# Eve: equity_series() day 0 is the ET date of opened_at (contract, Task 4).
OPENED_DAY = '2025-12-31'


def test_startup_embed_live_when_ev_ignored(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch, ignore_ev='1')
    e = bot._startup_embed()
    assert 'LIVE' in e.description
    assert e.color == discord.Color.green()
    assert not any('Not trading' in f.value for f in e.fields)
    names = [f.name for f in e.fields]
    assert 'Buying power' in names and 'Balance' in names
    assert 'Cash left' not in names


def test_startup_embed_gated_without_flag(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch, ignore_ev='0')
    e = bot._startup_embed()
    assert 'Nothing will be traded' in e.description
    assert e.color == discord.Color.orange()


def test_scorecard_embed_on_fresh_account(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch)
    e = bot._scorecard_embed(DAY, now=NOW)
    assert any(f.name.startswith('All-time') for f in e.fields)
    assert any(f.name == 'Verdict' for f in e.fields)
    # Spec section 1: the next open comes from the trading calendar. DAY is a
    # Saturday, so "tomorrow" would be wrong and Monday is right.
    today = next(f for f in e.fields if f.name.startswith('Today'))
    assert 'Stocks resume Mon 21 Sep 09:30 ET' in today.value
    assert len(e) < 6000
    assert len(e.fields) <= 25
    assert all(len(f.value) <= 1024 for f in e.fields)


def test_scorecard_embed_without_intraday_metrics(db_path, monkeypatch):
    """Fast mode on but nothing trained (same shape as FAST_MODE=0): Task 7
    returns bt_precision / ev_bt / ev_bt_net None for every class. The report
    must still render - it is the one message that proves the account is alive."""
    bot = _make_bot(db_path, monkeypatch, metrics={})
    e = bot._scorecard_embed(DAY, now=NOW)
    stock = next(f for f in e.fields if f.name.startswith('📈 Stock'))
    assert 'walk-forward precision n/a' in stock.value
    assert 'Backtest EV net of costs n/a' in stock.value
    verdict = next(f for f in e.fields if f.name == 'Verdict')
    assert 'no backtest EV' in verdict.value
    assert len(e) < 6000 and len(e.fields) <= 25


def _synthetic_scorecard(n_positions):
    """A build_scorecard() result with every key from Task 7's contract
    additions, big enough to overflow a field. Units and key names are the
    producer's, so a mismatch here would fail the producer's test too."""
    stock = {
        # verdict inputs (fractions)
        'gated': False, 'sig_n': 120, 'sig_lo': 0.30, 'sig_hi': 0.48, 'sig_lo_day': 0.28,
        'exec_n': 12, 'exec_mean': 0.001, 'exec_se': 0.002, 'ev_bt': -0.0003, 'cost': 0.00102,
        # verdict output and the rest of the class block
        'verdict': 'EXTEND', 'verdict_text': 'CI straddles breakeven',
        'breakeven': 0.4388,
        'exits_by_reason': {'tp': {'n': 5, 'mean_net': 1.2}, 'sl': {'n': 7, 'mean_net': -0.9}},
        'tp_n': 12, 'tp_first_rate': 0.42, 'tp_lo': 0.18, 'tp_hi': 0.69,
        'bt_precision': 0.357, 'bt_n': 947, 'ev_bt_net': -0.00132,
        'exec_mean_pct': 0.1, 'exec_ci': 0.392,               # PERCENTAGE POINTS
        'gate_text': 'trading on paper despite EV -0.030% below the 0.30% floor '
                     '(FAST_IGNORE_EV on)',
        'sig_hits': 47, 'sig_rate': 0.39, 'sig_days': 6,
    }
    crypto = {
        **stock, 'gated': True, 'verdict': 'NO-GO',
        'verdict_text': 'cost gate blocks crypto: round-trip cost 1.20% is at or above '
                        'the 0.60% take-profit',
        'exec_n': 0, 'exec_mean': 0.0, 'exec_se': 0.0, 'exec_mean_pct': 0.0, 'exec_ci': 0.0,
        'cost': 0.012, 'breakeven': 1.6, 'exits_by_reason': {},
        'tp_n': 0, 'tp_first_rate': 0.0, 'tp_lo': 0.0, 'tp_hi': 1.0,
        # no trained crypto model: exactly what Task 7 emits for a missing class
        'bt_precision': None, 'bt_n': 0, 'ev_bt': None, 'ev_bt_net': None,
        'gate_text': 'blocked: take-profit 0.60% is below the 1.20% round-trip cost',
        'sig_n': 9, 'sig_hits': 4, 'sig_rate': 0.444, 'sig_lo': 0.19, 'sig_hi': 0.73,
        'sig_lo_day': 0.0, 'sig_days': 1,
    }
    return {
        'headline': {'all_time_net': 3.21, 'all_time_pct': 0.00642, 'n_closed': 12,
                     'ci_dollars': 4.5},
        'account': {'equity': 503.21, 'cash': 400.0, 'buying_power': 380.0,
                    'unsettled': 20.0, 'unsettled_until': '2026-09-21',     # ET date string
                    'starting_cash': 500.0, 'gross_pnl': 3.5, 'fees_paid': 0.29},
        'today': {'realized': 1.0, 'unrealized': -0.5, 'n_trades': 4, 'wins': 1,
                  'losses': 1, 'loss_tripped': False},
        'closed_today': [{'symbol': 'AAPL', 'shares': 0.5, 'price': 101.2, 'amount': 50.6,
                          'net': 0.55, 'gross': 0.56, 'fees': 0.01, 'exit_reason': 'tp',
                          'created_at': '2026-09-19T15:10:00+00:00'}],
        'positions': [{'symbol': f'SYM{i:03d}', 'shares': 0.123456, 'avg_price': 100.05,
                       'entry_ref': 100.0, 'price': 101.0, 'market_value': 12.47,
                       'pnl': 0.12, 'pnl_pct': 0.0095, 'pct_vs_ref': 0.01}
                      for i in range(n_positions)],
        'since_open': {'n_closed': 12, 'win_rate': 0.42, 'win_lo': 0.18, 'win_hi': 0.69,
                       'mean_net': 0.27, 'mean_ci': 0.8, 'profit_factor': 1.1,
                       'max_drawdown_pct': 1.2,                             # PERCENTAGE POINTS
                       'days_running': 6},
        'classes': {'stock': stock, 'crypto': crypto},
        'spy': {'start_close': 640.0, 'last_close': 652.8, 'pct': 0.02, 'value': 510.0},
    }


def test_scorecard_embed_truncates_long_lists(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch)
    monkeypatch.setattr(discord_bot, 'build_scorecard',
                        lambda *a, **k: _synthetic_scorecard(80))
    e = bot._scorecard_embed(DAY)
    pos = next(f for f in e.fields if f.name == 'Open positions')
    assert '… and' in pos.value and pos.value.endswith('more')
    assert len(pos.value) <= 1000
    assert '0.123456 SYM000 @ $100.05 → $101.00 (+1.00% vs entry ref)' in pos.value
    closed = next(f for f in e.fields if f.name == 'Closed today')
    assert '🟩 0.5 AAPL @ $101.20 → **+$0.55** (gross +$0.56, fees $0.01) [tp]' in closed.value
    acct = next(f for f in e.fields if f.name == 'Account')
    assert 'settles 2026-09-21 09:30 ET' in acct.value
    since = next(f for f in e.fields if f.name == 'Scorecard since open')
    assert 'max drawdown 1.20%' in since.value                  # pp, not 120.00%
    stock = next(f for f in e.fields if f.name.startswith('📈 Stock'))
    assert 'walk-forward precision 35.7% (n=947)' in stock.value
    assert ('Backtest EV net of costs -0.132%/trade vs realised +0.100% ± 0.392%'
            in stock.value)                                     # pp, not +10.000%
    crypto = next(f for f in e.fields if f.name.startswith('🪙 Crypto'))
    assert 'walk-forward precision n/a' in crypto.value
    assert 'Backtest EV net of costs n/a' in crypto.value
    assert any(f.name == 'Verdict' and 'NO-GO' in f.value for f in e.fields)
    assert len(e) < 6000 and len(e.fields) <= 25


def test_daily_report_text_fallback_on_http_error(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch)
    chan = FakeChannel(fail_embeds=True)
    _stub_report_plumbing(bot, monkeypatch, chan)

    assert asyncio.run(bot._post_daily_report(DAY, now=NOW)) is True
    assert chan.embeds == []
    assert len(chan.texts) == 1 and 'all-time' in chan.texts[0].lower()
    assert bot.budget_tracker.get_day_state(DAY)['report_posted_at'] is not None
    assert any(r['date'] == DAY for r in bot.budget_tracker.equity_series())
    # Guard holds: a second tick on the same date is a no-op.
    assert asyncio.run(bot._post_daily_report(DAY, now=NOW)) is False
    assert len(chan.texts) == 1

    # Spec section 5: posts once per date across a restart inside the window.
    # A brand-new TradingBot (new BudgetTracker on the same file) reads the
    # flag from day_state, not from memory.
    bot2 = _make_bot(db_path, monkeypatch)
    chan2 = FakeChannel()
    _stub_report_plumbing(bot2, monkeypatch, chan2)
    assert asyncio.run(bot2._post_daily_report(DAY, now=NOW)) is False
    assert chan2.embeds == [] and chan2.texts == []


def test_daily_report_retries_when_send_fails(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch)
    chan = FakeChannel(fail_all=True)
    _stub_report_plumbing(bot, monkeypatch, chan)

    with pytest.raises(RuntimeError):
        asyncio.run(bot._post_daily_report(DAY, now=NOW))
    # Flag stays NULL so the next tick retries, but the books were closed first.
    assert bot.budget_tracker.get_day_state(DAY)['report_posted_at'] is None
    assert any(r['date'] == DAY for r in bot.budget_tracker.equity_series())

    # Discord is back: the next tick posts and sets the flag; the equity row
    # is upserted, not duplicated.
    chan.fail_all = False
    assert asyncio.run(bot._post_daily_report(DAY, now=NOW)) is True
    assert [e.title for e in chan.embeds] == ['stub']
    assert bot.budget_tracker.get_day_state(DAY)['report_posted_at'] is not None
    assert sum(1 for r in bot.budget_tracker.equity_series() if r['date'] == DAY) == 1


def test_pnl_before_close_does_not_suppress_report(db_path, monkeypatch):
    """Spec section 5: /pnl renders the same scorecard for the same date but
    never writes day_state or equity_history, so the 16:05 tick still posts."""
    bot = _make_bot(db_path, monkeypatch)
    chan = FakeChannel()
    _stub_report_plumbing(bot, monkeypatch, chan, stub_embed=False)

    e = asyncio.run(bot._embed_pnl(now=NOW))
    assert any(f.name.startswith('All-time') for f in e.fields)
    assert bot.budget_tracker.get_day_state(DAY) is None
    assert [r['date'] for r in bot.budget_tracker.equity_series()] == [OPENED_DAY]

    assert asyncio.run(bot._post_daily_report(DAY, now=NOW)) is True
    assert len(chan.embeds) == 1 and chan.embeds[0].title == f"📊 Paper scorecard — {DAY}"
    assert [r['date'] for r in bot.budget_tracker.equity_series()] == [OPENED_DAY, DAY]


def test_fast_action_embed_reads_ledger_fields(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch)
    summary = {
        'entries': [{'symbol': 'AAPL', 'shares': 0.5, 'price': 100.05, 'ref_price': 100.0,
                     'probability': 0.61, 'cost': 50.025, 'fees': 0.0, 'trade_id': 1}],
        'exits': [{'symbol': 'MSFT', 'shares': 0.25, 'price': 401.8, 'ref_price': 402.0,
                   'reason': 'take profit +1.01%', 'exit_reason': 'tp', 'pnl': 0.93,
                   'gross': 0.94, 'fees': 0.01, 'pct': 0.0101, 'trade_id': 2}],
        'equity': 501.2, 'buying_power': 350.0, 'unsettled': 100.45,
        'minutes_to_close': 90.0,
    }
    e = bot._fast_action_embed(summary)
    names = [f.name for f in e.fields]
    assert 'BOUGHT 0.5 AAPL @ $100.05' in names
    assert 'SOLD 0.25 MSFT @ $401.80' in names
    assert {'Balance', 'Buying power', 'Unsettled'} <= set(names)
    assert 'Cash left' not in names
    sold = next(f for f in e.fields if f.name.startswith('SOLD'))
    assert 'fees $0.01' in sold.value and '[tp]' in sold.value


def test_clip_lines_appends_remainder():
    lines = [f"line {i:03d} " + 'x' * 40 for i in range(60)]
    text = discord_bot._clip_lines(lines, limit=500)
    assert len(text) <= 500
    assert text.endswith('more')
    assert discord_bot._clip_lines(['a', 'b']) == 'a\nb'
