"""Signal log: dedupe per bar, trade linkage, and after-the-fact labelling."""
from datetime import datetime, timedelta, timezone

import pytest

from src import signal_log
from src.database import connect
from src.intraday_engine import ET, INTERVAL, IntradayEngine

# Fixed barriers so the tests do not depend on data/tuned.json or .env:
# +1.0% take-profit, -0.6% stop, 4-bar horizon.
TP, SL, HORIZON = 0.01, 0.006, 4

# A Monday 15:50 ET bar; its UTC ISO is what IntradayEngine.signal() emits.
BAR_ET = datetime(2026, 9, 14, 15, 50, tzinfo=ET)
BAR_TS = BAR_ET.astimezone(timezone.utc).isoformat()          # '2026-09-14T19:50:00+00:00'
NOW = datetime(2026, 9, 15, 20, 5, tzinfo=timezone.utc)      # Tuesday 16:05 ET
MON_TICK = datetime(2026, 9, 14, 16, 5, tzinfo=ET)           # the same-day 16:05 tick


@pytest.fixture
def conn(db_path):
    # IntradayEngine.__init__ only opens the DB and creates prices_intraday
    # (no fetch, no model load without data/); Database() (db_path fixture)
    # already created the signals table.
    IntradayEngine(db_path)
    c = connect(db_path)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def fixed_barriers(monkeypatch):
    monkeypatch.setattr(signal_log, 'barriers', lambda cls: (TP, SL, HORIZON))


def _sig(symbol, price=100.0, prob=0.6, bar_ts=BAR_TS):
    cls = 'crypto' if symbol.endswith('-USD') else 'stock'
    return {'symbol': symbol, 'asset_class': cls, 'probability': prob, 'bar': 0.5,
            'above_bar': prob >= 0.5, 'price': price, 'bar_ts': bar_ts, 'margin': prob - 0.5}


def _insert_bars(conn, symbol, bars):
    """bars: [(datetime ET-aware, high, low, close)]; stored the way fetch()
    stores yfinance bars, i.e. with the ET offset, NOT in UTC."""
    with conn:
        conn.executemany(
            "INSERT INTO prices_intraday (symbol, ts, open, high, low, close, volume, interval) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [(symbol, ts.isoformat(), close, high, low, close, 1000.0, INTERVAL)
             for ts, high, low, close in bars])


def _path(symbol, moves, start=BAR_ET):
    """Signal bar (close 100) followed by one bar per (high, low, close) in
    `moves`, five minutes apart. `moves` entries may be a datetime to jump the
    clock (next session), followed by that bar's (high, low, close)."""
    bars = [(start, 100.2, 99.8, 100.0)]
    ts = start
    for m in moves:
        if isinstance(m, datetime):
            ts = m - timedelta(minutes=5)
            continue
        ts = ts + timedelta(minutes=5)
        bars.append((ts, *m))
    return bars


def _row(conn, symbol):
    return conn.execute("SELECT * FROM signals WHERE symbol = ?", (symbol,)).fetchone()


# --- record / mark_executed ------------------------------------------------

def test_record_writes_one_row_per_symbol_per_bar(conn):
    sigs = [_sig('AAPL', prob=0.6), _sig('BTC-USD', price=60000.0, prob=0.3)]
    assert signal_log.record(conn, sigs, now=NOW) == 2
    # Same 5m bar scored again on the next 60 s poll: nothing new.
    assert signal_log.record(conn, sigs, now=NOW) == 0
    rows = conn.execute("SELECT * FROM signals ORDER BY symbol").fetchall()
    assert len(rows) == 2
    aapl = _row(conn, 'AAPL')
    assert aapl['bar_ts'] == BAR_TS
    assert aapl['asset_class'] == 'stock'
    assert aapl['above_bar'] == 1
    assert aapl['ref_price'] == 100.0
    assert aapl['trade_date'] == '2026-09-14'
    assert aapl['label'] is None and aapl['labeled_at'] is None
    assert aapl['executed_trade_id'] is None
    assert _row(conn, 'BTC-USD')['above_bar'] == 0
    assert signal_log.record(conn, [], now=NOW) == 0


def test_record_trade_date_is_et_not_utc(conn):
    # 01:00 UTC on the 15th is still 21:00 ET on the 14th.
    late = datetime(2026, 9, 15, 1, 0, tzinfo=timezone.utc).isoformat()
    signal_log.record(conn, [_sig('BTC-USD', bar_ts=late)], now=NOW)
    assert _row(conn, 'BTC-USD')['trade_date'] == '2026-09-14'


def test_mark_executed_sets_trade_id(conn):
    signal_log.record(conn, [_sig('AAPL'), _sig('MSFT')], now=NOW)
    signal_log.mark_executed(conn, 'AAPL', BAR_TS, 42)
    assert _row(conn, 'AAPL')['executed_trade_id'] == 42
    assert _row(conn, 'MSFT')['executed_trade_id'] is None


# --- label_pending -----------------------------------------------------------

def test_label_take_profit_first_is_1(conn):
    signal_log.record(conn, [_sig('AAPL')], now=NOW)
    _insert_bars(conn, 'AAPL', _path('AAPL', [
        (100.5, 99.9, 100.2),      # neither barrier
        (101.0, 99.8, 100.7),      # high touches +1.0% -> TP first
        (100.9, 99.3, 99.4),       # stop would fire here, but TP already won
        (100.0, 99.0, 99.1),
    ]))
    assert signal_log.label_pending(conn, now=NOW) == 1
    row = _row(conn, 'AAPL')
    assert row['label'] == 1
    assert row['labeled_at'] == '2026-09-15T20:05:00+00:00'
    # Already labelled rows are not touched again.
    assert signal_log.label_pending(conn, now=NOW) == 0


def test_label_stop_first_is_0(conn):
    signal_log.record(conn, [_sig('AAPL')], now=NOW)
    _insert_bars(conn, 'AAPL', _path('AAPL', [
        (100.5, 99.9, 100.2),
        (100.6, 99.4, 99.5),       # low touches -0.6% -> stop first
        (102.0, 99.5, 101.5),      # TP only after the stop: too late
        (102.0, 101.0, 101.5),
    ]))
    assert signal_log.label_pending(conn, now=NOW) == 1
    assert _row(conn, 'AAPL')['label'] == 0


def test_label_horizon_expires_flat_is_0(conn):
    signal_log.record(conn, [_sig('AAPL')], now=NOW)
    _insert_bars(conn, 'AAPL', _path('AAPL', [(100.3, 99.7, 100.1)] * 4
                                      + [(102.0, 101.0, 101.5)]))   # TP at bar 5 > horizon
    assert signal_log.label_pending(conn, now=NOW) == 1
    assert _row(conn, 'AAPL')['label'] == 0


def test_label_session_boundary_stops_stock_but_not_crypto(conn):
    next_open = datetime(2026, 9, 15, 9, 30, tzinfo=ET)
    moves = [
        (100.5, 99.9, 100.2),      # 15:55 ET, unresolved at the bell
        next_open,
        (102.0, 100.5, 101.5),     # 09:30 next day: TP, but a stock is flat by then
        (102.0, 100.5, 101.5),
        (102.0, 100.5, 101.5),
    ]
    signal_log.record(conn, [_sig('AAPL'), _sig('BTC-USD')], now=NOW)
    _insert_bars(conn, 'AAPL', _path('AAPL', moves))
    _insert_bars(conn, 'BTC-USD', _path('BTC-USD', moves))
    assert signal_log.label_pending(conn, now=NOW) == 2
    assert _row(conn, 'AAPL')['label'] == 0        # session ended unresolved
    assert _row(conn, 'BTC-USD')['label'] == 1     # crypto runs the full horizon


def test_label_insufficient_bars_stays_null_then_resolves(conn):
    # A 10:00 ET signal with 3 of its 5 bars stored, checked mid-session:
    # the session can still resolve it, so it waits.
    start = datetime(2026, 9, 14, 10, 0, tzinfo=ET)
    bar_ts = start.astimezone(timezone.utc).isoformat()
    signal_log.record(conn, [_sig('AAPL', bar_ts=bar_ts)], now=NOW)
    bars = _path('AAPL', [(100.5, 99.9, 100.2), (101.0, 99.8, 100.7)], start=start)
    _insert_bars(conn, 'AAPL', bars)
    mid_session = datetime(2026, 9, 14, 10, 20, tzinfo=ET)
    assert signal_log.label_pending(conn, now=mid_session) == 0
    row = _row(conn, 'AAPL')
    assert row['label'] is None and row['labeled_at'] is None
    # The next fetch completes the window and the next call labels it.
    last = bars[-1][0]
    _insert_bars(conn, 'AAPL', [(last + timedelta(minutes=5), 100.9, 99.3, 99.4),
                                (last + timedelta(minutes=10), 100.0, 99.0, 99.1)])
    assert signal_log.label_pending(conn, now=mid_session) == 1
    assert _row(conn, 'AAPL')['label'] == 1


def test_label_stock_session_end_is_0_same_day(conn):
    # Two 15:50 ET signals with only the 15:55 closing bar stored (2 of 5).
    # AAPL is unresolved at the bell; MSFT touches TP on the closing bar.
    signal_log.record(conn, [_sig('AAPL'), _sig('MSFT')], now=NOW)
    _insert_bars(conn, 'AAPL', _path('AAPL', [(100.5, 99.9, 100.2)]))
    _insert_bars(conn, 'MSFT', _path('MSFT', [(101.0, 99.8, 100.7)]))
    # Before the bell the window is simply incomplete: nothing is labelled.
    assert signal_log.label_pending(conn, now=datetime(2026, 9, 14, 15, 58, tzinfo=ET)) == 0
    assert _row(conn, 'AAPL')['label'] is None and _row(conn, 'MSFT')['label'] is None
    # The same day's 16:05 tick: the session is over, so the walk stops where
    # the session cut would have stopped it. No next-session bars needed.
    assert signal_log.label_pending(conn, now=MON_TICK) == 2
    aapl = _row(conn, 'AAPL')
    assert aapl['label'] == 0
    assert aapl['labeled_at'] == '2026-09-14T20:05:00+00:00'
    assert _row(conn, 'MSFT')['label'] == 1


def test_label_stock_session_end_needs_the_closing_bar(conn):
    # 15:40 ET signal, bars stored only to 15:45 (the feed died before the
    # close). Session over or not, "nothing happened" cannot be inferred from
    # missing bars: it stays NULL until the closing bar arrives.
    start = datetime(2026, 9, 14, 15, 40, tzinfo=ET)
    bar_ts = start.astimezone(timezone.utc).isoformat()
    signal_log.record(conn, [_sig('AAPL', bar_ts=bar_ts)], now=NOW)
    bars = _path('AAPL', [(100.5, 99.9, 100.2)], start=start)          # 15:45 only
    _insert_bars(conn, 'AAPL', bars)
    assert signal_log.label_pending(conn, now=MON_TICK) == 0
    assert _row(conn, 'AAPL')['label'] is None
    last = bars[-1][0]
    _insert_bars(conn, 'AAPL', [(last + timedelta(minutes=5), 100.4, 99.8, 100.1),   # 15:50
                                (last + timedelta(minutes=10), 100.3, 99.7, 100.0)])  # 15:55
    assert signal_log.label_pending(conn, now=MON_TICK) == 1
    assert _row(conn, 'AAPL')['label'] == 0


def test_label_crypto_insufficient_bars_stays_null_after_the_bell(conn):
    # Crypto has no session: a 15:50 ET signal with 2 of 5 bars stays NULL
    # even on the next day's tick, until the window is actually stored.
    signal_log.record(conn, [_sig('BTC-USD')], now=NOW)
    _insert_bars(conn, 'BTC-USD', _path('BTC-USD', [(100.5, 99.9, 100.2)]))
    assert signal_log.label_pending(conn, now=NOW) == 0
    assert _row(conn, 'BTC-USD')['label'] is None


def test_label_skips_signal_whose_own_bar_is_missing(conn):
    signal_log.record(conn, [_sig('AAPL')], now=NOW)
    bars = _path('AAPL', [(102.0, 100.5, 101.5)] * 5)
    _insert_bars(conn, 'AAPL', bars[1:])          # window present, signal bar absent
    assert signal_log.label_pending(conn, now=NOW) == 0
    assert _row(conn, 'AAPL')['label'] is None


# --- IntradayEngine.signal() carries bar_ts --------------------------------

class _AlwaysUp:
    def predict_proba(self, X):
        return [[0.4, 0.6]] * len(X)


def test_signal_includes_bar_ts(db_path, monkeypatch):
    monkeypatch.setenv('MAX_BAR_AGE_MIN', '1000000000')   # stale-bar guard is not under test
    engine = IntradayEngine(db_path)
    engine.models = {'stock': _AlwaysUp()}
    start = datetime(2026, 9, 14, 9, 30, tzinfo=ET)
    bars = []
    for i in range(80):
        close = 100.0 + (i % 7) * 0.1
        bars.append((start + timedelta(minutes=5 * i), close + 0.2, close - 0.2, close))
    _insert_bars(engine.conn, 'AAPL', bars)
    sig = engine.signal('AAPL')
    assert sig is not None
    assert sig['bar_ts'] == (start + timedelta(minutes=5 * 79)).astimezone(timezone.utc).isoformat()
    assert sig['bar_ts'].endswith('+00:00')
    assert sig['price'] == bars[-1][3]
