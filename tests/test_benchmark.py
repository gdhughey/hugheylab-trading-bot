"""Benchmark symbols (Task 8).

SPY is fetched alongside the universe so the daily scorecard can print a
buy-and-hold comparison, but it must never be trained on or scanned. These
tests replace the provider chain with an in-memory fake so nothing touches
the network.
"""

import pandas as pd
import pytest

from src import ml_engine
from src.database import Database
from src.ml_engine import BENCHMARK_SYMBOLS, TradingSignalEngine


class FakeProvider:
    """Stands in for the whole history chain.

    Records every symbol list it is asked for and answers each symbol with two
    daily bars, so nothing falls through as 'missing' and the engine never
    tries a second provider.
    """
    name = 'fake'
    provides_history = True
    per_cycle_cap = None

    def __init__(self):
        self.requests = []

    def fetch(self, symbols, period='2y'):
        self.requests.append(list(symbols))
        idx = pd.to_datetime(['2026-09-14', '2026-09-15'])
        frames = {
            s: pd.DataFrame({
                'open': [100.0, 101.0], 'high': [102.0, 103.0],
                'low': [99.0, 100.0], 'close': [101.0, 102.0],
                'volume': [1000.0, 1000.0],
            }, index=idx)
            for s in symbols
        }
        return frames, []

    def requested(self) -> list:
        """Every symbol asked for, in order, across all fetch calls."""
        return [s for req in self.requests for s in req]


@pytest.fixture
def fake():
    return FakeProvider()


@pytest.fixture
def engine(tmp_path, fake, monkeypatch):
    """Engine on a fresh file DB whose only data source is `fake`."""
    path = str(tmp_path / 'bench.db')
    Database(path)  # creates the prices table; connect() alone does not
    # Keep the engine away from data/model.joblib and the real providers.
    monkeypatch.setattr(ml_engine, 'MODEL_PATH', str(tmp_path / 'model.joblib'))
    monkeypatch.setattr(ml_engine, 'build_chain', lambda: [fake])
    monkeypatch.setattr(ml_engine, 'build_quote_chain', lambda: [])
    return TradingSignalEngine(path)


def _insert_closes(conn, symbol: str, closes: dict) -> None:
    """Write daily bars straight into `prices`, bypassing the provider chain."""
    with conn:
        conn.executemany(
            "INSERT INTO prices (symbol, date, open, high, low, close, volume, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'test')",
            [(symbol, d, c, c, c, c, 0.0) for d, c in closes.items()])


def test_benchmark_symbols_is_spy():
    assert BENCHMARK_SYMBOLS == ['SPY']


def test_fetch_requests_spy_once_and_keeps_it_out_of_symbols(engine, fake):
    rows = engine.fetch_and_store_data(['AAPL', 'MSFT'])

    # SPY rides in the same batch as the universe - one extra name, not an
    # extra request - and exactly once.
    assert fake.requested() == ['SPY', 'AAPL', 'MSFT']
    assert fake.requested().count('SPY') == 1
    # 2 bars x 3 symbols landed in prices, SPY included ...
    assert rows == 6
    assert engine.conn.execute(
        "SELECT COUNT(*) AS n FROM prices WHERE symbol = 'SPY'").fetchone()['n'] == 2
    # ... but the training/scanning universe never sees it.
    assert engine.symbols == ['AAPL', 'MSFT']
    assert sorted(engine._stored_symbols()) == ['AAPL', 'MSFT']


def test_fetch_does_not_request_spy_twice_when_universe_already_has_it(engine, fake):
    engine.fetch_and_store_data(['SPY', 'AAPL'])

    assert fake.requested() == ['SPY', 'AAPL']
    assert fake.requested().count('SPY') == 1
    # Stored, but still filtered out of the training list.
    assert sorted(engine._stored_symbols()) == ['AAPL']


def test_stored_symbols_excludes_benchmark_rows_written_directly(engine):
    _insert_closes(engine.conn, 'SPY', {'2026-09-14': 650.0})
    _insert_closes(engine.conn, 'AAPL', {'2026-09-14': 230.0})

    assert engine._stored_symbols() == ['AAPL']


def test_stored_close_and_latest_price_read_the_newest_spy_bar(engine):
    _insert_closes(engine.conn, 'SPY', {
        '2026-09-11': 640.0, '2026-09-14': 650.0, '2026-09-15': 655.0})

    assert engine.stored_close('SPY') == 655.0
    # No quote providers in this fixture, so latest_price falls back to the
    # stored close - the path the scorecard's "last close" uses off-hours.
    assert engine.latest_price('SPY') == 655.0


def test_first_close_on_or_after(engine):
    _insert_closes(engine.conn, 'SPY', {
        '2026-09-11': 640.0, '2026-09-14': 650.0, '2026-09-15': 655.0})

    # Exact date match is inclusive.
    assert engine.first_close_on_or_after('SPY', '2026-09-14') == 650.0
    # Account opened on a Saturday: the first bar after it is Monday's.
    assert engine.first_close_on_or_after('SPY', '2026-09-12') == 650.0
    # A full UTC ISO timestamp (account.opened_at) is truncated to its date,
    # so the bar on that date is still included rather than string-comparing
    # past it.
    assert engine.first_close_on_or_after('SPY', '2026-09-14T13:30:00+00:00') == 650.0
    # Nothing stored on or after the date, or an unknown symbol -> None.
    assert engine.first_close_on_or_after('SPY', '2026-09-16') is None
    assert engine.first_close_on_or_after('QQQ', '2026-09-14') is None
