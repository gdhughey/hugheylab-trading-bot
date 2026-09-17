"""src.collector: schema, upserts, timestamp discipline, provider error handling.

No network: yfinance and Finnhub are replaced with fakes. Dates are fixed.
"""
import json
import urllib.error
from datetime import datetime, timezone

import pandas as pd
import pytest

from src import collector
from src.database import connect
from src.intraday_engine import upsert_bars


@pytest.fixture
def conn(tmp_path):
    c = connect(str(tmp_path / 'c.db'))
    collector.ensure_schema(c)
    yield c
    c.close()


# --- 1m bars -----------------------------------------------------------------

def _frame(tz, n=3):
    idx = pd.date_range('2026-09-15 09:30', periods=n, freq='1min', tz=tz)
    return pd.DataFrame({'Open': 10.0, 'High': 11.0, 'Low': 9.0,
                         'Close': [10.5, 10.6, 10.7], 'Volume': 100}, index=idx)


def test_upsert_bars_1m_lands_in_prices_intraday_and_is_idempotent(conn):
    raw = pd.concat({'AAPL': _frame('America/New_York'),
                     'MSFT': _frame('America/New_York')}, axis=1)
    assert upsert_bars(conn, raw, ['AAPL', 'MSFT'], '1m') == 6
    assert upsert_bars(conn, raw, ['AAPL', 'MSFT'], '1m') == 6   # upsert, no dupes
    rows = conn.execute("SELECT symbol, ts, interval FROM prices_intraday ORDER BY symbol, ts").fetchall()
    assert len(rows) == 6
    assert {r['interval'] for r in rows} == {'1m'}
    # ts is stored exactly as yfinance served it (ET offset for a stock frame)
    assert rows[0]['ts'] == '2026-09-15T09:30:00-04:00'


def test_upsert_bars_keeps_crypto_utc_and_separates_intervals(conn):
    # single-symbol frames come back flat (no MultiIndex); crypto is UTC-indexed
    assert upsert_bars(conn, _frame('UTC'), ['BTC-USD'], '1m') == 3
    assert upsert_bars(conn, _frame('UTC'), ['BTC-USD'], '5m') == 3
    rows = conn.execute("SELECT ts, interval FROM prices_intraday ORDER BY interval, ts").fetchall()
    assert len(rows) == 6 and rows[0]['ts'].endswith('+00:00')
    assert [r['interval'] for r in rows] == ['1m'] * 3 + ['5m'] * 3


def test_collect_bars_caps_period_at_7d(conn, monkeypatch):
    seen = {}

    def fake_download(symbols, period, interval, **kw):
        seen.update(period=period, interval=interval)
        return _frame('America/New_York')

    monkeypatch.setattr(collector.yf, 'download', fake_download)
    assert collector.collect_bars(conn, ['AAPL'], period='30d') == 3
    assert seen == {'period': '7d', 'interval': '1m'}


def test_collect_bars_download_failure_returns_zero(conn, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('yahoo down')
    monkeypatch.setattr(collector.yf, 'download', boom)
    assert collector.collect_bars(conn, ['AAPL']) == 0


# --- news --------------------------------------------------------------------

ARTICLE = {'id': 7001, 'datetime': 1758000000, 'headline': 'Apple does a thing',
           'summary': 's', 'source': 'Reuters', 'url': 'https://x/1',
           'category': 'company', 'related': 'AAPL'}


def test_store_news_uses_publication_time_in_utc(conn):
    assert collector.store_news(conn, 'AAPL', [ARTICLE], fetched_at='2026-09-17T00:00:00+00:00') == 1
    r = conn.execute("SELECT * FROM news").fetchone()
    assert r['published_at'] == datetime.fromtimestamp(1758000000, tz=timezone.utc).isoformat(timespec='seconds')
    assert r['published_at'].endswith('+00:00')
    assert r['published_at'] != r['fetched_at']
    assert r['headline'] == 'Apple does a thing'


def test_store_news_ignores_duplicates_and_junk(conn):
    junk = [{'headline': 'no id'}, {'id': 'x', 'datetime': 1}]
    assert collector.store_news(conn, 'AAPL', [ARTICLE] + junk) == 1
    assert collector.store_news(conn, 'AAPL', [ARTICLE]) == 0
    # the same article fetched under another ticker is a separate mapping row
    assert collector.store_news(conn, 'MSFT', [ARTICLE]) == 1
    assert conn.execute("SELECT count(*) FROM news").fetchone()[0] == 2


def test_collect_news_skips_crypto_and_aborts_on_auth_error(conn, monkeypatch):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if 'symbol=MSFT' in url:
            raise urllib.error.HTTPError(url, 429, 'quota', {}, None)
        return [ARTICLE]

    monkeypatch.setattr(collector, '_get_json', fake_get)
    monkeypatch.setattr(collector.time, 'sleep', lambda s: None)
    n = collector.collect_news(conn, ['BTC-USD', 'AAPL', 'MSFT', 'NVDA'], key='k', days=2)
    assert n == 1                                   # AAPL stored, MSFT 429 aborted, NVDA never asked
    assert len(calls) == 2
    assert all('BTC' not in u for u in calls)
    assert 'from=' in calls[0] and 'to=' in calls[0] and 'token=k' in calls[0]


def test_collect_news_without_key_is_a_noop(conn, monkeypatch):
    monkeypatch.delenv('FINNHUB_API_KEY', raising=False)
    assert collector.collect_news(conn, ['AAPL']) == 0


# --- earnings ----------------------------------------------------------------

def test_store_earnings_upserts_latest_view(conn):
    row = {'symbol': 'aapl', 'date': '2026-10-29', 'hour': 'AMC', 'epsEstimate': 1.5}
    assert collector.store_earnings(conn, [row, {'symbol': '', 'date': 'x'}]) == 1
    row2 = dict(row, hour='bmo', epsEstimate=1.6)
    assert collector.store_earnings(conn, [row2]) == 1
    r = conn.execute("SELECT * FROM events").fetchone()
    assert (r['kind'], r['symbol'], r['event_date'], r['hour']) == ('earnings', 'AAPL', '2026-10-29', 'bmo')
    assert json.loads(r['detail'])['epsEstimate'] == 1.6
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 1


def test_collect_earnings_parses_finnhub_envelope(conn, monkeypatch):
    monkeypatch.setattr(collector, '_get_json',
                        lambda url, **kw: {'earningsCalendar': [
                            {'symbol': 'MSFT', 'date': '2026-10-28', 'hour': 'amc'},
                            {'symbol': 'NVDA', 'date': '2026-11-19', 'hour': 'amc'}]})
    assert collector.collect_earnings(conn, key='k') == 2
    monkeypatch.setattr(collector, '_get_json', lambda url, **kw: {'error': 'nope'})
    assert collector.collect_earnings(conn, key='k') == 0


# --- universe / run ----------------------------------------------------------

def test_universe_sp500_reads_file_and_keeps_crypto(tmp_path, monkeypatch):
    p = tmp_path / 'sp500.txt'
    p.write_text('MMM\nAOS\nmmm\n')
    monkeypatch.setenv('COLLECT_UNIVERSE', 'sp500')
    monkeypatch.setenv('SP500_PATH', str(p))
    monkeypatch.setenv('TRADE_CRYPTO', '1')
    monkeypatch.delenv('FAST_SYMBOLS', raising=False)
    u = collector.universe()
    assert u[:2] == ['MMM', 'AOS'] and len(u) == 2 + 13   # de-duped + DEFAULT_CRYPTO
    assert 'BTC-USD' in u and 'AAPL' not in u


def test_run_bars_only_reports_counts(tmp_path, monkeypatch):
    monkeypatch.setenv('COLLECT_UNIVERSE', 'fast')
    monkeypatch.setenv('FAST_SYMBOLS', 'AAPL')
    monkeypatch.setattr(collector.yf, 'download', lambda *a, **k: _frame('America/New_York'))
    out = collector.run('bars', db_path=str(tmp_path / 'r.db'))
    assert out['symbols'] == 1 and out['bars_1m'] == 3 and 'news_new' not in out
