#!/usr/bin/env python3
"""
Data collector: 1m bars, company news, earnings calendar.

Runs as its own systemd timer (proxmox/trading-collector.timer, hourly), NOT
inside the bot process. The bot has a 700M memory cap and a history of dying
when a download got large; this process gets its own cap and its own failure
domain, and the bot never notices whether it ran.

Why collect at all: yfinance serves 1m bars for the last 7 days only, and
Finnhub's free tier serves news for the trailing year but nothing older. Both
are "use it or lose it" data. After a few months the 1m store gives exact
barrier-touch labels for the intraday model, and the news store gives the
Phase 2 text features (news-count shocks, calendar flags) a back-test window
that no free API can hand us later. Nothing here is a feature yet - data only.

Timestamp discipline (the leak that makes text features look like they work):
news rows carry Finnhub's PUBLICATION time in UTC, never the fetch time and
never a bare date. A feature that joins on published_at -> next bar open is
honest; one that joins on the calendar day is not.
"""

import os
import sys
import json
import time
import logging
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf

from src.database import connect
from src.intraday_engine import ensure_intraday_schema, fast_universe, is_crypto, upsert_bars
from src.providers import _get_json

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id           INTEGER NOT NULL,      -- Finnhub article id
    symbol       TEXT    NOT NULL,      -- the ticker it was fetched under
    published_at TEXT    NOT NULL,      -- UTC ISO-8601, from Finnhub `datetime`
    fetched_at   TEXT    NOT NULL,      -- UTC ISO-8601, when we stored it
    source       TEXT,
    headline     TEXT,
    summary      TEXT,
    url          TEXT,
    category     TEXT,
    related      TEXT,                  -- Finnhub's comma list of related tickers
    PRIMARY KEY (symbol, id)
);
CREATE INDEX IF NOT EXISTS idx_news_sym_pub ON news (symbol, published_at);

CREATE TABLE IF NOT EXISTS events (
    kind        TEXT NOT NULL,          -- 'earnings' (macro kinds come in Phase 2)
    symbol      TEXT NOT NULL,          -- '' for market-wide events
    event_date  TEXT NOT NULL,          -- YYYY-MM-DD, exchange-local calendar day
    hour        TEXT,                   -- bmo | amc | dmh | '' (Finnhub's field)
    detail      TEXT,                   -- raw provider row as JSON
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (kind, symbol, event_date)
);
CREATE INDEX IF NOT EXISTS idx_events_date ON events (event_date, kind);
"""

FINNHUB = "https://finnhub.io/api/v1"
# Free tier: 60 calls/min. One symbol per call, so pace to stay under it even
# when the timer fires while the bot is also using its quote quota.
FINNHUB_PACE_S = float(os.getenv('COLLECT_FINNHUB_PACE_S', 1.1))


def ensure_schema(conn):
    ensure_intraday_schema(conn)
    with conn:
        conn.executescript(SCHEMA)


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# --- universe --------------------------------------------------------------

def universe():
    """Which symbols to collect for.

    COLLECT_UNIVERSE=fast  (default) the bot's own intraday universe
    COLLECT_UNIVERSE=sp500 every name in data/sp500.txt plus the fast crypto
    The S&P list is ~5x the symbols and ~5x the disk (16 MB/day at 1m); turn it
    on once the store lives on the big NVMe.
    """
    spec = os.getenv('COLLECT_UNIVERSE', 'fast').strip().lower()
    syms = fast_universe()
    if spec == 'sp500':
        path = Path(os.getenv('SP500_PATH', 'data/sp500.txt'))
        if path.exists():
            listed = [ln.strip().upper() for ln in path.read_text().splitlines() if ln.strip()]
            syms = listed + [s for s in syms if is_crypto(s)]
        else:
            logger.warning(f"[collector] {path} missing; falling back to fast universe")
    # de-dupe, keep order
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# --- 1m bars ---------------------------------------------------------------

def collect_bars(conn, symbols, period=None) -> int:
    """Upsert 1m bars for `symbols`. Returns rows written.

    Default window is 2 days: the timer runs hourly, so any single failed run
    is repaired by the next one, and the overlap costs nothing because the
    write is an upsert. `period` is capped at yfinance's 7d limit for 1m.
    """
    period = period or os.getenv('COLLECT_1M_PERIOD', '2d')
    if period.endswith('d') and int(period[:-1]) > 7:
        period = '7d'
    try:
        raw = yf.download(symbols, period=period, interval='1m',
                          auto_adjust=True, progress=False,
                          group_by='ticker', threads=True)
    except Exception as e:
        logger.error(f"[collector] 1m download failed: {e}")
        return 0
    return upsert_bars(conn, raw, symbols, '1m')


# --- Finnhub: news + earnings ---------------------------------------------

def _finnhub_key():
    return os.getenv('FINNHUB_API_KEY', '').strip()


def store_news(conn, symbol, articles, fetched_at=None) -> int:
    """Insert Finnhub company-news rows for one symbol; ignore ones we have."""
    fetched_at = fetched_at or _utc_now_iso()
    rows = []
    for a in articles or []:
        try:
            aid = int(a['id'])
            ts = int(a['datetime'])
        except (KeyError, TypeError, ValueError):
            continue
        published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec='seconds')
        rows.append((aid, symbol, published, fetched_at, a.get('source'),
                     a.get('headline'), a.get('summary'), a.get('url'),
                     a.get('category'), a.get('related')))
    if not rows:
        return 0
    with conn:
        cur = conn.executemany(
            "INSERT OR IGNORE INTO news (id, symbol, published_at, fetched_at, source, "
            "headline, summary, url, category, related) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    return cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else len(rows)


def collect_news(conn, symbols, key=None, days=None) -> int:
    """Fetch company news for each stock symbol. Returns new rows stored.

    Finnhub's company-news endpoint is stocks-only (crypto symbols are
    exchange-qualified and it has no news for them), so crypto is skipped.
    A 401/403/429 aborts the run: the key is bad or the quota is gone, and
    hammering on is pointless. Other errors skip the symbol.
    """
    key = key or _finnhub_key()
    if not key:
        logger.warning("[collector] FINNHUB_API_KEY not set; skipping news")
        return 0
    days = int(days or os.getenv('COLLECT_NEWS_DAYS', 2))
    to = date.today()
    frm = to - timedelta(days=days)
    stored = 0
    for sym in symbols:
        if is_crypto(sym):
            continue
        try:
            arts = _get_json(f"{FINNHUB}/company-news?symbol={sym}"
                             f"&from={frm.isoformat()}&to={to.isoformat()}&token={key}")
            stored += store_news(conn, sym, arts)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 429):
                logger.error(f"[collector] finnhub news HTTP {e.code} on {sym}; aborting run")
                break
            logger.warning(f"[collector] news {sym}: HTTP {e.code}")
        except Exception as e:
            logger.warning(f"[collector] news {sym}: {e}")
        time.sleep(FINNHUB_PACE_S)
    return stored


def store_earnings(conn, rows, fetched_at=None) -> int:
    """Upsert Finnhub earningsCalendar rows into events(kind='earnings')."""
    fetched_at = fetched_at or _utc_now_iso()
    out = []
    for r in rows or []:
        sym, d = (r.get('symbol') or '').upper(), r.get('date')
        if not sym or not d:
            continue
        out.append(('earnings', sym, d, (r.get('hour') or '').lower(),
                    json.dumps(r, separators=(',', ':')), fetched_at))
    if not out:
        return 0
    with conn:
        conn.executemany(
            "INSERT INTO events (kind, symbol, event_date, hour, detail, fetched_at) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(kind, symbol, event_date) DO UPDATE SET "
            "hour=excluded.hour, detail=excluded.detail, fetched_at=excluded.fetched_at", out)
    return len(out)


def collect_earnings(conn, key=None, back_days=7, ahead_days=35) -> int:
    """Earnings dates for every listed name, trailing week to ~5 weeks ahead.

    One call for the whole market. Upserted because estimates and the
    bmo/amc flag change as the date approaches; the stored `detail` is the
    latest view. Rows are kept forever so a back-test can ask "was there an
    earnings print within 2 days" for any past date.
    """
    key = key or _finnhub_key()
    if not key:
        logger.warning("[collector] FINNHUB_API_KEY not set; skipping earnings")
        return 0
    frm = date.today() - timedelta(days=back_days)
    to = date.today() + timedelta(days=ahead_days)
    try:
        d = _get_json(f"{FINNHUB}/calendar/earnings?from={frm.isoformat()}"
                      f"&to={to.isoformat()}&token={key}")
    except Exception as e:
        logger.error(f"[collector] earnings calendar: {e}")
        return 0
    return store_earnings(conn, d.get('earningsCalendar') if isinstance(d, dict) else None)


# --- entry point -----------------------------------------------------------

def run(what='all', db_path=None) -> dict:
    conn = connect(db_path)
    ensure_schema(conn)
    syms = universe()
    out = {'symbols': len(syms)}
    t0 = time.time()
    if what in ('all', 'bars'):
        out['bars_1m'] = collect_bars(conn, syms)
    if what in ('all', 'news'):
        out['news_new'] = collect_news(conn, syms)
    if what in ('all', 'earnings'):
        out['earnings'] = collect_earnings(conn)
    out['seconds'] = round(time.time() - t0, 1)
    conn.close()
    return out


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    what = (argv or sys.argv[1:] or ['all'])[0]
    if what not in ('all', 'bars', 'news', 'earnings'):
        sys.exit(f"usage: python -m src.collector [all|bars|news|earnings]")
    res = run(what)
    logger.info(f"[collector] {what}: {res}")


if __name__ == '__main__':
    main()
