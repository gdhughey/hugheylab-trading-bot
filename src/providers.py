#!/usr/bin/env python3
"""
Market data providers.

The engine asks for daily OHLCV and does not care where it comes from. Each
provider returns {symbol: DataFrame[open,high,low,close,volume]} indexed by
date, and reports which symbols it could not serve so the next provider in the
chain can try them.

Ordering matters: the chain runs cheapest-and-broadest first, and each
subsequent provider only sees the symbols still missing. That keeps rate-limited
sources (Alpha Vantage: 25 requests/day) usable as gap-fillers rather than
burning their quota on symbols that already resolved.

A note on independence: `yahoo` and `yahoo_direct` hit the same upstream data by
two different code paths. That protects against library-level failures (yfinance
silently dropping a symbol mid-batch, which we have observed) but it is NOT a
second opinion on the prices themselves. Genuine cross-source validation needs a
keyed provider.
"""

import os
import json
import time
import logging
import urllib.request
import urllib.error
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
COLS = ['open', 'high', 'low', 'close', 'volume']


def _get_json(url, timeout=25, headers=None):
    req = urllib.request.Request(url, headers={'User-Agent': UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8', 'replace'))


class Provider:
    name = 'base'
    requires_key = False
    # Rough ceiling on symbols to attempt per cycle; None = unlimited.
    per_cycle_cap = None

    def available(self) -> bool:
        return True

    # Some providers serve live quotes but not history (Finnhub's free tier),
    # and vice versa. These are separate capabilities.
    provides_history = True
    provides_quotes = False

    def fetch(self, symbols, period='2y'):
        """Return (frames_by_symbol, missing_symbols)."""
        raise NotImplementedError

    def quote(self, symbol):
        """Latest traded price, or None."""
        return None


class YFinanceProvider(Provider):
    """Primary. Batched - a 500-name universe is ~9 requests."""
    name = 'yahoo'

    def __init__(self, batch_size=None):
        self.batch_size = batch_size or int(os.getenv('FETCH_BATCH_SIZE', 60))

    def fetch(self, symbols, period='2y'):
        import yfinance as yf
        out, symbols = {}, list(symbols)

        for i in range(0, len(symbols), self.batch_size):
            chunk = symbols[i:i + self.batch_size]
            try:
                raw = yf.download(chunk, period=period, interval='1d',
                                  auto_adjust=True, progress=False,
                                  group_by='ticker', threads=True)
            except Exception as e:
                logger.warning(f"[yahoo] batch failed ({chunk[:3]}...): {e}")
                continue
            if raw is None or raw.empty:
                continue
            for sym in chunk:
                try:
                    df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
                    df = df.rename(columns=str.lower).dropna(subset=['close'])
                    if not df.empty:
                        out[sym] = df[COLS]
                except (KeyError, Exception):
                    continue

        missing = [s for s in symbols if s not in out]
        return out, missing


class YahooChartProvider(Provider):
    """Yahoo's chart endpoint, one symbol at a time, no library in between.

    Used to recover symbols yfinance dropped. Same upstream data - redundancy
    against library bugs, not an independent price check.
    """
    name = 'yahoo_direct'
    per_cycle_cap = 60

    _RANGE = {'1y': '1y', '2y': '2y', '5y': '5y', '10y': '10y', 'max': 'max'}

    def fetch(self, symbols, period='2y'):
        rng = self._RANGE.get(period, '2y')
        out = {}
        for sym in symbols:
            try:
                url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
                       f"?range={rng}&interval=1d")
                d = _get_json(url)
                res = (d.get('chart') or {}).get('result') or []
                if not res:
                    continue
                r = res[0]
                ts = r.get('timestamp') or []
                q = ((r.get('indicators') or {}).get('quote') or [{}])[0]
                if not ts or not q:
                    continue
                df = pd.DataFrame({
                    'open': q.get('open'), 'high': q.get('high'),
                    'low': q.get('low'), 'close': q.get('close'),
                    'volume': q.get('volume'),
                }, index=pd.to_datetime(ts, unit='s')).dropna(subset=['close'])
                if not df.empty:
                    out[sym] = df[COLS]
            except Exception as e:
                logger.debug(f"[yahoo_direct] {sym}: {e}")
            time.sleep(0.12)  # be polite; this endpoint is unofficial
        return out, [s for s in symbols if s not in out]


class AlphaVantageProvider(Provider):
    """Genuinely independent source. Free tier is 25 requests/DAY, so this is a
    gap-filler and cross-check only - never the primary."""
    name = 'alphavantage'
    requires_key = True
    per_cycle_cap = int(os.getenv('ALPHAVANTAGE_CAP', 5))

    def __init__(self):
        self.key = os.getenv('ALPHAVANTAGE_API_KEY', '').strip()

    def available(self):
        return bool(self.key)

    def fetch(self, symbols, period='2y'):
        out = {}
        for sym in symbols[:self.per_cycle_cap]:
            try:
                d = _get_json("https://www.alphavantage.co/query"
                              f"?function=TIME_SERIES_DAILY&symbol={sym}"
                              f"&outputsize=full&apikey={self.key}")
                series = d.get('Time Series (Daily)')
                if not series:
                    note = d.get('Note') or d.get('Information') or d.get('Error Message')
                    if note:
                        logger.warning(f"[alphavantage] {sym}: {str(note)[:120]}")
                        break  # quota hit - stop burning calls
                    continue
                rows = {
                    pd.Timestamp(day): {
                        'open': float(v['1. open']), 'high': float(v['2. high']),
                        'low': float(v['3. low']), 'close': float(v['4. close']),
                        'volume': float(v['5. volume']),
                    } for day, v in series.items()
                }
                df = pd.DataFrame.from_dict(rows, orient='index').sort_index()
                if not df.empty:
                    out[sym] = df[COLS]
            except Exception as e:
                logger.debug(f"[alphavantage] {sym}: {e}")
            time.sleep(1.0)
        return out, [s for s in symbols if s not in out]


class TiingoProvider(Provider):
    """Independent source with a usable free tier for daily end-of-day bars."""
    name = 'tiingo'
    requires_key = True
    per_cycle_cap = int(os.getenv('TIINGO_CAP', 100))

    def __init__(self):
        self.key = os.getenv('TIINGO_API_KEY', '').strip()

    def available(self):
        return bool(self.key)

    def fetch(self, symbols, period='2y'):
        years = {'1y': 1, '2y': 2, '5y': 5, '10y': 10}.get(period, 2)
        start = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
        out = {}
        for sym in symbols[:self.per_cycle_cap]:
            try:
                rows = _get_json(
                    f"https://api.tiingo.com/tiingo/daily/{sym}/prices"
                    f"?startDate={start}&format=json",
                    headers={'Authorization': f'Token {self.key}',
                             'Content-Type': 'application/json'})
                if not isinstance(rows, list) or not rows:
                    continue
                df = pd.DataFrame([{
                    'date': pd.Timestamp(r['date']).tz_localize(None),
                    'open': r.get('adjOpen') or r.get('open'),
                    'high': r.get('adjHigh') or r.get('high'),
                    'low': r.get('adjLow') or r.get('low'),
                    'close': r.get('adjClose') or r.get('close'),
                    'volume': r.get('adjVolume') or r.get('volume'),
                } for r in rows]).set_index('date').sort_index().dropna(subset=['close'])
                if not df.empty:
                    out[sym] = df[COLS]
            except Exception as e:
                logger.debug(f"[tiingo] {sym}: {e}")
            time.sleep(0.15)
        return out, [s for s in symbols if s not in out]


class FinnhubProvider(Provider):
    """Free tier serves LIVE QUOTES but returns 403 for historical candles
    (verified 2026-09-04). So this is a quote source, not a history source -
    which is genuinely useful, because everything else here is yesterday's
    close and this is the current price."""
    name = 'finnhub'
    requires_key = True
    provides_history = False
    provides_quotes = True

    def __init__(self):
        self.key = os.getenv('FINNHUB_API_KEY', '').strip()
        self._blocked = False

    def available(self):
        return bool(self.key)

    def fetch(self, symbols, period='2y'):
        return {}, list(symbols)

    def quote(self, symbol):
        if self._blocked:
            return None
        try:
            d = _get_json("https://finnhub.io/api/v1/quote"
                          f"?symbol={symbol}&token={self.key}")
            price = d.get('c')
            # Finnhub returns c=0 for symbols it has no data on.
            return float(price) if price else None
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 429):
                logger.warning(f"[finnhub] quotes unavailable (HTTP {e.code}) "
                               f"- disabling for this run")
                self._blocked = True
        except Exception as e:
            logger.debug(f"[finnhub] quote {symbol}: {e}")
        return None


class YahooQuoteProvider(Provider):
    """Live-ish quote from Yahoo's chart endpoint - keyless fallback."""
    name = 'yahoo_quote'
    provides_history = False
    provides_quotes = True

    def quote(self, symbol):
        try:
            d = _get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/"
                          f"{symbol}?range=1d&interval=1m")
            res = (d.get('chart') or {}).get('result') or []
            if not res:
                return None
            meta = res[0].get('meta') or {}
            price = meta.get('regularMarketPrice')
            return float(price) if price else None
        except Exception as e:
            logger.debug(f"[yahoo_quote] {symbol}: {e}")
        return None


REGISTRY = {p.name: p for p in
            [YFinanceProvider, YahooChartProvider, AlphaVantageProvider,
             TiingoProvider, FinnhubProvider, YahooQuoteProvider]}


def build_chain(spec: str = None):
    """Instantiate the configured providers, dropping any missing their key."""
    spec = spec or os.getenv('DATA_SOURCES', 'yahoo,yahoo_direct')
    chain = []
    for name in [n.strip().lower() for n in spec.split(',') if n.strip()]:
        cls = REGISTRY.get(name)
        if cls is None:
            logger.warning(f"Unknown data source '{name}' - skipping")
            continue
        p = cls()
        if not p.available():
            logger.info(f"Data source '{name}' configured but no API key set - skipping")
            continue
        chain.append(p)
    if not chain:
        logger.warning("No usable data sources configured - falling back to yahoo")
        chain = [YFinanceProvider()]
    logger.info(f"Data sources: {' -> '.join(p.name for p in chain)}")
    return chain


def build_quote_chain(spec: str = None):
    """Providers that can answer 'what is this worth right now'."""
    spec = spec or os.getenv('QUOTE_SOURCES', 'finnhub,yahoo_quote')
    chain = []
    for name in [n.strip().lower() for n in spec.split(',') if n.strip()]:
        cls = REGISTRY.get(name)
        if cls is None:
            continue
        p = cls()
        if p.provides_quotes and p.available():
            chain.append(p)
    if chain:
        logger.info(f"Quote sources: {' -> '.join(p.name for p in chain)}")
    else:
        logger.info("No live quote sources - prices will be last stored close")
    return chain
