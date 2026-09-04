#!/usr/bin/env python3
"""
Intraday signal engine.

The daily engine answers "will this close higher tomorrow" from daily bars, so
polling it faster than once a day tells you nothing new. This engine answers a
different question - "will this move more than THRESHOLD over the next HORIZON
bars" - from 5m (or 1m) bars, which is a question whose answer actually changes
during the session.

Deliberately narrow: a small, liquid universe so a full refresh takes seconds.
"""

import os
import logging
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from src.database import connect

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

INTERVAL = os.getenv('INTRADAY_INTERVAL', '5m')          # 1m | 5m | 15m
PERIOD = os.getenv('INTRADAY_PERIOD', '60d')             # 1m maxes out at 7d
HORIZON = int(os.getenv('INTRADAY_HORIZON_BARS', 6))     # 6 x 5m = 30 min ahead
THRESHOLD = float(os.getenv('INTRADAY_THRESHOLD', 0.0015))   # 0.15% - must beat spread
# Only ~36% of bars are positive, so the model's probabilities cluster near that
# base rate and a fixed 0.55 bar is never cleared. Select relative to the base
# rate instead: a signal counts when p > positive_rate * this ratio.
PROB_RATIO = float(os.getenv('INTRADAY_PROB_RATIO', 1.15))
MODEL_PATH = os.getenv('INTRADAY_MODEL_PATH', 'data/intraday_model.joblib')

DEFAULT_FAST = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META', 'AMD',
                'CRWD', 'GLW', 'NFLX', 'INTC', 'MU', 'PLTR', 'SMCI', 'AVGO',
                'QCOM', 'ORCL', 'ADBE', 'NOW', 'F', 'SOFI', 'BAC', 'T',
                'PFE', 'CSCO', 'WBD', 'RIVN', 'LCID', 'HOOD']

FEATURES = [
    'ret_1', 'ret_3', 'ret_6', 'ret_12',
    'vwap_dist', 'rsi_14', 'vol_12', 'volume_ratio',
    'day_range_pos', 'from_open', 'minutes_norm', 'gap_open',
]


def fast_universe():
    spec = os.getenv('FAST_SYMBOLS', '').strip()
    if spec:
        return [t.strip().upper() for t in spec.replace(',', ' ').split() if t.strip()]
    return list(DEFAULT_FAST)


# --- market hours ---------------------------------------------------------

def market_state(now=None):
    """('open'|'premarket'|'afterhours'|'closed', description)."""
    now = (now or datetime.now(ET)).astimezone(ET)
    if now.weekday() >= 5:
        return 'closed', 'weekend'
    t = now.time()
    if dtime(9, 30) <= t < dtime(16, 0):
        return 'open', 'regular session'
    if dtime(4, 0) <= t < dtime(9, 30):
        return 'premarket', 'pre-market'
    if dtime(16, 0) <= t < dtime(20, 0):
        return 'afterhours', 'after hours'
    return 'closed', 'overnight'


def minutes_to_close(now=None):
    now = (now or datetime.now(ET)).astimezone(ET)
    close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return (close - now).total_seconds() / 60


def _rsi(close, period=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Intraday features. Index must be tz-aware timestamps."""
    out = pd.DataFrame(index=df.index)
    close, vol = df['close'], df['volume']
    session = df.index.tz_convert(ET).date

    out['ret_1'] = close.pct_change(1)
    out['ret_3'] = close.pct_change(3)
    out['ret_6'] = close.pct_change(6)
    out['ret_12'] = close.pct_change(12)

    # session VWAP - where price sits relative to the day's average trade
    typical = (df['high'] + df['low'] + close) / 3
    pv = (typical * vol).groupby(session).cumsum()
    cv = vol.groupby(session).cumsum().replace(0, np.nan)
    out['vwap_dist'] = close / (pv / cv) - 1

    out['rsi_14'] = _rsi(close) / 100.0
    out['vol_12'] = close.pct_change().rolling(12).std()
    out['volume_ratio'] = vol / vol.rolling(20).mean()

    day_hi = df['high'].groupby(session).cummax()
    day_lo = df['low'].groupby(session).cummin()
    out['day_range_pos'] = (close - day_lo) / (day_hi - day_lo).replace(0, np.nan)

    day_open = close.groupby(session).transform('first')
    out['from_open'] = close / day_open - 1

    mins = df.index.tz_convert(ET)
    out['minutes_norm'] = ((mins.hour * 60 + mins.minute) - 570) / 390.0  # 9:30->0, 16:00->1

    prev_close = close.groupby(session).transform('last').shift(1)
    out['gap_open'] = (day_open / prev_close - 1).fillna(0)
    return out


def build_target(close: pd.Series) -> pd.Series:
    """1 when the forward return over HORIZON bars clears THRESHOLD."""
    fwd = close.shift(-HORIZON) / close - 1
    return (fwd > THRESHOLD).astype(int)


class IntradayEngine:
    def __init__(self, db_path=None):
        self.conn = connect(db_path)
        self.model = None
        self.symbols = fast_universe()
        self.last_metrics = {}
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self.positive_rate = 0.36
        if Path(MODEL_PATH).exists():
            try:
                blob = joblib.load(MODEL_PATH)
                if isinstance(blob, dict):
                    self.model = blob['model']
                    self.last_metrics = blob.get('meta', {})
                    self.positive_rate = self.last_metrics.get('positive_rate', 0.36)
                else:
                    self.model = blob
                logger.info(f"Loaded intraday model (base rate "
                            f"{self.positive_rate:.1%}, bar {self.threshold():.3f})")
            except Exception as e:
                logger.warning(f"Could not load intraday model ({e})")

    def threshold(self):
        """Selection bar, scaled to the model's own base rate."""
        return min(0.95, self.positive_rate * PROB_RATIO)

    def _ensure_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS prices_intraday (
                    symbol TEXT NOT NULL, ts TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    interval TEXT NOT NULL,
                    PRIMARY KEY (symbol, ts, interval)
                )""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_sym "
                              "ON prices_intraday (symbol, interval, ts)")

    # --- data ------------------------------------------------------------

    def fetch(self, symbols=None, interval=None, period=None) -> int:
        symbols = symbols or self.symbols
        interval = interval or INTERVAL
        period = period or (('7d' if interval == '1m' else PERIOD))
        try:
            raw = yf.download(symbols, period=period, interval=interval,
                              auto_adjust=True, progress=False,
                              group_by='ticker', threads=True)
        except Exception as e:
            logger.error(f"[intraday] download failed: {e}")
            return 0
        if raw is None or raw.empty:
            return 0

        total = 0
        for sym in symbols:
            try:
                df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.rename(columns=str.lower).dropna(subset=['close'])
                if df.empty:
                    continue
                rows = [(sym, idx.isoformat(), float(r['open']), float(r['high']),
                         float(r['low']), float(r['close']), float(r['volume'] or 0),
                         interval) for idx, r in df.iterrows()]
                with self.conn:
                    self.conn.executemany(
                        "INSERT INTO prices_intraday "
                        "(symbol, ts, open, high, low, close, volume, interval) "
                        "VALUES (?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(symbol, ts, interval) DO UPDATE SET "
                        "open=excluded.open, high=excluded.high, low=excluded.low, "
                        "close=excluded.close, volume=excluded.volume", rows)
                total += len(rows)
            except (KeyError, Exception):
                continue
        return total

    def _history(self, symbol, interval=None) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT ts, open, high, low, close, volume FROM prices_intraday "
            "WHERE symbol = ? AND interval = ? ORDER BY ts",
            self.conn, params=(symbol, interval or INTERVAL))
        if df.empty:
            return df
        df['ts'] = pd.to_datetime(df['ts'], utc=True, format='ISO8601')
        return df.set_index('ts')

    # --- training --------------------------------------------------------

    def train(self) -> bool:
        frames = []
        for sym in self.symbols:
            h = self._history(sym)
            if len(h) < 300:
                continue
            f = build_features(h)
            f['target'] = build_target(h['close'])
            frames.append(f.dropna())
        if not frames:
            logger.error("[intraday] no usable data - fetch first")
            return False

        data = pd.concat(frames).replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 2000:
            logger.error(f"[intraday] only {len(data)} rows, need 2000+")
            return False
        data = data.sort_index()

        X, y = data[FEATURES], data['target']
        cut = int(len(X) * 0.8)              # chronological - no lookahead
        Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

        model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                           max_depth=3, subsample=0.9,
                                           random_state=42)
        model.fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))
        base = max(yte.mean(), 1 - yte.mean())
        self.last_metrics = {
            'accuracy': acc, 'baseline': base, 'edge': acc - base,
            'rows': len(Xtr), 'positive_rate': float(ytr.mean()),
            'interval': INTERVAL, 'horizon_bars': HORIZON, 'threshold': THRESHOLD,
        }
        logger.info(f"[intraday] trained on {len(Xtr):,} rows ({INTERVAL}, "
                    f"{HORIZON} bars ahead, >{THRESHOLD:.2%}) - accuracy {acc:.3f} "
                    f"vs baseline {base:.3f} (edge {acc - base:+.3f})")
        self.model = model
        self.positive_rate = float(ytr.mean())
        joblib.dump({'model': model, 'meta': self.last_metrics}, MODEL_PATH)
        logger.info(f"[intraday] selection bar is p > {self.threshold():.3f} "
                    f"({PROB_RATIO}x the {self.positive_rate:.1%} base rate)")
        return True

    # --- inference -------------------------------------------------------

    def signal(self, symbol):
        if self.model is None:
            return None
        h = self._history(symbol)
        if len(h) < 60:
            return None
        f = build_features(h).replace([np.inf, -np.inf], np.nan).dropna()
        if f.empty:
            return None
        p_up = float(self.model.predict_proba(f[FEATURES].iloc[[-1]])[0][1])
        return {
            'symbol': symbol,
            'probability': p_up,
            'price': float(h['close'].iloc[-1]),
            'bar_time': h.index[-1].astimezone(ET).strftime('%H:%M ET'),
        }

    def scan(self, min_probability=None):
        """Long candidates only - the target is 'moves up more than THRESHOLD',
        so a low probability means 'no move expected', not 'goes down'."""
        bar = self.threshold() if min_probability is None else min_probability
        out = []
        for sym in self.symbols:
            try:
                s = self.signal(sym)
            except Exception as e:
                logger.debug(f"[intraday] {sym}: {e}")
                continue
            if s:
                s['above_bar'] = s['probability'] >= bar
                if s['above_bar']:
                    out.append(s)
        out.sort(key=lambda r: r['probability'], reverse=True)
        return out

    def scan_all(self):
        """Every symbol scored, ranked - for reporting, not trading."""
        bar = self.threshold()
        out = []
        for sym in self.symbols:
            try:
                s = self.signal(sym)
            except Exception:
                continue
            if s:
                s['above_bar'] = s['probability'] >= bar
                out.append(s)
        out.sort(key=lambda r: r['probability'], reverse=True)
        return out
