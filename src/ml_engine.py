#!/usr/bin/env python3
"""
Trading signal engine - fetches OHLCV, engineers features, trains a pooled
gradient-boosting classifier, and emits directional signals.

The model is pooled across symbols (one model, symbol-agnostic features) so it
trains on far more rows than a per-symbol model would see. Features are all
ratios/z-scores rather than raw prices, which is what makes pooling valid.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.database import connect
from src.providers import build_chain, build_quote_chain

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv('MODEL_PATH', 'data/model.joblib')

# The systemd unit sets ProtectHome=yes, so yfinance cannot write its default
# cache under /root and silently re-fetches timezone metadata on every call.
# Point it at the app's own writable data dir instead.
try:
    _cache = Path('data/yf_cache')
    _cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(_cache))
except Exception as _e:  # non-fatal - yfinance just runs uncached
    logger.debug(f"could not set yfinance cache location: {_e}")
LOOKBACK = os.getenv('LOOKBACK_PERIOD', '2y')

# Which question the classifier is trained to answer.
#   next_day   - does tomorrow close higher than today? (direction, ~coin-flip)
#   forward_5d - does the 5-session forward return clear FORWARD_THRESHOLD?
# The second is the more tradeable framing: it targets a move big enough to be
# worth the spread rather than any move at all.
# Which symbols to scan. 'sp500' pulls the current index constituents;
# 'default' is the original 5; anything else is treated as a comma list.
UNIVERSE = os.getenv('UNIVERSE', 'default')
DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN']
BATCH_SIZE = int(os.getenv('FETCH_BATCH_SIZE', 60))

TARGET_MODE = os.getenv('TARGET_MODE', 'next_day')
FORWARD_DAYS = int(os.getenv('FORWARD_DAYS', 5))
FORWARD_THRESHOLD = float(os.getenv('FORWARD_THRESHOLD', 0.025))

FEATURES = [
    'ret_1d', 'ret_5d', 'ret_10d',
    'sma_5_ratio', 'sma_20_ratio', 'sma_50_ratio',
    'rsi_14', 'volatility_20', 'volume_ratio', 'high_low_range',
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def load_universe(name: str = None) -> list:
    """Resolve the configured universe to a concrete ticker list."""
    name = (name or UNIVERSE).strip()

    if name.lower() == 'default':
        return list(DEFAULT_SYMBOLS)

    if name.lower() == 'sp500':
        cache = Path('data/sp500.txt')
        try:
            # Wikipedia 403s pandas' default urllib UA, so fetch it ourselves.
            import requests
            from io import StringIO
            html = requests.get(
                'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                headers={'User-Agent': 'trading-bot/1.0 (homelab; contact via github)'},
                timeout=30,
            )
            html.raise_for_status()
            tables = pd.read_html(StringIO(html.text))
            syms = [str(t).replace('.', '-').strip()
                    for t in tables[0]['Symbol'].tolist()]
            syms = [t for t in syms if t and t.isascii()]
            if len(syms) > 400:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text("\n".join(syms))
                logger.info(f"Universe sp500: {len(syms)} tickers (refreshed)")
                return syms
            raise ValueError(f"only parsed {len(syms)} tickers")
        except Exception as e:
            logger.warning(f"Could not fetch S&P 500 list ({e})")
            if cache.exists():
                syms = [l.strip() for l in cache.read_text().splitlines() if l.strip()]
                logger.info(f"Universe sp500: {len(syms)} tickers (cached)")
                return syms
            logger.error("No cached S&P 500 list - falling back to defaults")
            return list(DEFAULT_SYMBOLS)

    syms = [t.strip().upper() for t in name.replace(',', ' ').split() if t.strip()]
    logger.info(f"Universe custom: {len(syms)} tickers")
    return syms or list(DEFAULT_SYMBOLS)


def build_target(close: pd.Series, mode: str = None) -> pd.Series:
    """Label each row. Uses only FUTURE bars, so rows near the end drop out."""
    mode = mode or TARGET_MODE
    if mode == 'forward_5d':
        fwd = close.shift(-FORWARD_DAYS) / close - 1
        return (fwd > FORWARD_THRESHOLD).astype(int)
    if mode != 'next_day':
        raise ValueError(f"unknown TARGET_MODE {mode!r}")
    return (close.shift(-1) > close).astype(int)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn an OHLCV frame into the model's feature columns."""
    out = pd.DataFrame(index=df.index)
    close = df['close']

    out['ret_1d'] = close.pct_change(1)
    out['ret_5d'] = close.pct_change(5)
    out['ret_10d'] = close.pct_change(10)
    out['sma_5_ratio'] = close / close.rolling(5).mean() - 1
    out['sma_20_ratio'] = close / close.rolling(20).mean() - 1
    out['sma_50_ratio'] = close / close.rolling(50).mean() - 1
    out['rsi_14'] = _rsi(close) / 100.0
    out['volatility_20'] = close.pct_change().rolling(20).std()
    out['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    out['high_low_range'] = (df['high'] - df['low']) / close
    return out


class TradingSignalEngine:
    def __init__(self, db_path: str = None):
        self.conn = connect(db_path)
        self.model = None
        self.symbols = []
        self.last_metrics = {}
        self.providers = build_chain()
        self.quote_providers = build_quote_chain()
        self.source_stats = {}
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        self._load_model()

    # --- persistence -----------------------------------------------------

    def _load_model(self):
        if Path(MODEL_PATH).exists():
            try:
                self.model = joblib.load(MODEL_PATH)
                logger.info(f"Loaded model from {MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Could not load model ({e}) - retrain required")

    # --- data ------------------------------------------------------------

    def fetch_and_store_data(self, symbols=None) -> int:
        """Fill price history from the provider chain.

        Each provider only sees the symbols still missing after the previous
        one, so a rate-limited source is spent on genuine gaps rather than on
        symbols that already resolved.
        """
        self.symbols = list(symbols) if symbols else (self.symbols or load_universe())
        pending = list(self.symbols)
        total = 0
        self.source_stats = {}

        for provider in self.providers:
            if not pending:
                break
            if not provider.provides_history:
                continue
            cap = provider.per_cycle_cap
            attempt = pending[:cap] if cap else pending
            try:
                frames, missing = provider.fetch(attempt, LOOKBACK)
            except Exception as e:
                logger.error(f"[{provider.name}] fetch failed: {e}")
                continue

            rows_written = 0
            for symbol, df in frames.items():
                rows_written += self._store(symbol, df, provider.name)
            total += rows_written
            self.source_stats[provider.name] = {
                'symbols': len(frames), 'rows': rows_written}
            if frames:
                logger.info(f"[{provider.name}] {len(frames)} symbols, {rows_written} rows")

            resolved = set(frames)
            pending = [s for s in pending if s not in resolved]

        if pending:
            logger.warning(f"{len(pending)} symbol(s) unresolved by every source: "
                           f"{', '.join(pending[:8])}{'...' if len(pending) > 8 else ''}")
            self.source_stats['unresolved'] = {'symbols': len(pending), 'rows': 0}

        logger.info(f"Stored {total} rows across {len(self.symbols) - len(pending)} symbols "
                    f"via {len([p for p in self.providers if p.provides_history])} source(s)")
        return total

    def _store(self, symbol, df, source):
        """Upsert one symbol's OHLCV frame. Returns rows written."""
        try:
            rows = [
                (symbol, idx.strftime('%Y-%m-%d'),
                 float(r['open']), float(r['high']), float(r['low']),
                 float(r['close']), float(r['volume'] or 0), source)
                for idx, r in df.iterrows() if not pd.isna(r['close'])
            ]
            if not rows:
                return 0
            with self.conn:
                self.conn.executemany(
                    "INSERT INTO prices (symbol, date, open, high, low, close, volume, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(symbol, date) DO UPDATE SET "
                    "open=excluded.open, high=excluded.high, low=excluded.low, "
                    "close=excluded.close, volume=excluded.volume, source=excluded.source",
                    rows)
            return len(rows)
        except Exception as e:
            logger.warning(f"store failed for {symbol} from {source}: {e}")
            return 0

    def _history(self, symbol: str) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM prices "
            "WHERE symbol = ? ORDER BY date",
            self.conn, params=(symbol,), parse_dates=['date'],
        )
        return df.set_index('date') if not df.empty else df

    def _stored_symbols(self) -> list:
        rows = self.conn.execute("SELECT DISTINCT symbol FROM prices").fetchall()
        return [r['symbol'] for r in rows]

    # --- training --------------------------------------------------------

    def train_model(self, target_mode: str = None) -> bool:
        """Train the pooled classifier on everything in the prices table."""
        mode = target_mode or TARGET_MODE
        frames = []
        for symbol in (self.symbols or self._stored_symbols()):
            hist = self._history(symbol)
            if len(hist) < 120:
                logger.warning(f"Skipping {symbol}: only {len(hist)} rows (need 120+)")
                continue
            feats = build_features(hist)
            feats['target'] = build_target(hist['close'], mode)
            frames.append(feats.dropna())

        if not frames:
            logger.error("train_model: no usable data - run fetch_and_store_data first")
            return False

        data = pd.concat(frames).replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 200:
            logger.error(f"train_model: only {len(data)} usable rows, need 200+")
            return False

        X, y = data[FEATURES], data['target']
        # shuffle=False keeps the split chronological - no lookahead leakage.
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=42,
        )
        model.fit(X_tr, y_tr)

        acc = accuracy_score(y_te, model.predict(X_te))
        base = max(y_te.mean(), 1 - y_te.mean())
        logger.info(f"Trained [{mode}] on {len(X_tr)} rows - holdout accuracy "
                    f"{acc:.3f} vs baseline {base:.3f} (edge {acc - base:+.3f})")
        self.last_metrics = {'mode': mode, 'accuracy': acc, 'baseline': base,
                             'edge': acc - base, 'rows': len(X_tr),
                             'positive_rate': float(y_tr.mean())}

        self.model = model
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
        return True

    # --- inference -------------------------------------------------------

    def scan(self, symbols=None, min_probability: float = 0.55, limit: int = None):
        """Score the whole universe and return the strongest signals first.

        This is the "which one is best right now" path: every symbol is scored
        by the same pooled model, so the probabilities are directly comparable.
        """
        universe = symbols or self.symbols or load_universe()
        results, skipped = [], 0

        for symbol in universe:
            try:
                sig = self.get_signal(symbol)
            except Exception as e:
                logger.debug(f"scan: {symbol} failed: {e}")
                skipped += 1
                continue
            if sig and sig['probability'] >= min_probability:
                results.append(sig)
            elif sig is None:
                skipped += 1

        results.sort(key=lambda r: r['probability'], reverse=True)
        logger.info(f"Scan: {len(results)} of {len(universe)} symbols cleared "
                    f"{min_probability:.0%} ({skipped} unscoreable)")
        return results[:limit] if limit else results

    def stored_close(self, symbol: str):
        """Most recent stored close from daily bars, else intraday bars.

        Crypto never enters the daily `prices` table - the daily engine only
        fetches the S&P 500 - so without the intraday fallback every crypto
        position reports "price unavailable" in /pnl.
        """
        row = self.conn.execute(
            "SELECT close FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        if row:
            return float(row['close'])
        try:
            row = self.conn.execute(
                "SELECT close FROM prices_intraday WHERE symbol = ? "
                "ORDER BY ts DESC LIMIT 1", (symbol,)).fetchone()
            return float(row['close']) if row else None
        except Exception:
            return None

    def latest_price(self, symbol: str):
        """Live quote if any provider can give one, else the last stored close.

        A large gap between the two usually means our stored history is stale,
        so it is logged rather than silently accepted.
        """
        close = self.stored_close(symbol)
        for provider in self.quote_providers:
            try:
                live = provider.quote(symbol)
            except Exception:
                continue
            if not live:
                continue
            if close and abs(live - close) / close > 0.15:
                logger.warning(f"{symbol}: live {provider.name} quote ${live:,.2f} differs "
                               f"{abs(live - close) / close:.1%} from stored close "
                               f"${close:,.2f} - history may be stale")
            return live
        return close

    def get_signal(self, symbol: str):
        """Return {'symbol', 'signal', 'price', 'probability'} or None."""
        if self.model is None:
            logger.warning("get_signal called before a model was trained")
            return None

        hist = self._history(symbol)
        if len(hist) < 60:
            logger.warning(f"get_signal: not enough history for {symbol}")
            return None

        feats = build_features(hist).replace([np.inf, -np.inf], np.nan).dropna()
        if feats.empty:
            return None

        latest = feats[FEATURES].iloc[[-1]]
        p_up = float(self.model.predict_proba(latest)[0][1])
        signal = 1 if p_up > 0.5 else 0

        # Under forward_5d the negative class means "no 2.5% up-move expected",
        # which is NOT a sell thesis - and because only ~31% of rows are
        # positive, treating it as one would fire confident SELL alerts almost
        # every cycle. Only long signals are actionable in that mode.
        if TARGET_MODE == 'forward_5d' and signal == 0:
            return None

        return {
            'symbol': symbol,
            'signal': signal,
            'price': float(hist['close'].iloc[-1]),
            # Confidence in the direction actually chosen, so the caller's
            # `probability > 0.55` gate works for BUY and SELL alike.
            'probability': p_up if signal == 1 else 1 - p_up,
            'as_of': hist.index[-1].strftime('%Y-%m-%d'),
        }
