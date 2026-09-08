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
import json
import logging
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score

from src.database import connect
from src.labeling import triple_barrier, fixed_horizon, purged_split

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

INTERVAL = os.getenv('INTRADAY_INTERVAL', '5m')          # 1m | 5m | 15m
PERIOD = os.getenv('INTRADAY_PERIOD', '60d')             # 1m maxes out at 7d

# Horizon and barriers were re-derived from a measured sweep on 2026-09-08.
# The original +1.5%/-1.0% over 30 minutes was the WORST config tested: a 1.5%
# move inside 30 min happens on only 6.4% of stock bars (1.3% of crypto bars),
# while the 1.0% stop is hit constantly - expected value was NEGATIVE on both
# asset classes. +0.8%/-0.5% over 120 minutes was positive on both, with enough
# signals to be believable.
HORIZON = int(os.getenv('INTRADAY_HORIZON_BARS', 24))    # 24 x 5m = 2 hours
TAKE_PROFIT = float(os.getenv('FAST_TAKE_PROFIT', 0.008))
STOP_LOSS = float(os.getenv('FAST_STOP_LOSS', 0.005))

# Per-asset-class overrides written by tune.py. Stocks and crypto do not want
# the same barriers - a sweep on 2026-09-08 put stocks at +1.2%/-0.7% over 240
# min (EV +1.057%/trade) and crypto at +0.8%/-0.5% over 180 min (+0.355%).
TUNED_PATH = os.getenv('TUNED_PATH', 'data/tuned.json')


def _load_tuned():
    try:
        with open(TUNED_PATH) as fh:
            return json.load(fh)
    except Exception:
        return {}


TUNED = _load_tuned()


def barriers(cls='stock'):
    """(take_profit, stop_loss, horizon_bars) for one asset class."""
    t = TUNED.get(cls) or {}
    return (float(t.get('take_profit', TAKE_PROFIT)),
            float(t.get('stop_loss', STOP_LOSS)),
            int(t.get('horizon', HORIZON)))


def model_params(cls='stock'):
    t = TUNED.get(cls) or {}
    return t.get('params') or dict(max_iter=250, learning_rate=0.06,
                                   max_depth=5, l2_regularization=1.0)


def prob_ratio(cls='stock'):
    t = TUNED.get(cls) or {}
    return float(t.get('ratio', PROB_RATIO))
THRESHOLD = float(os.getenv('INTRADAY_THRESHOLD', 0.0015))   # legacy label only
# Only ~36% of bars are positive, so the model's probabilities cluster near that
# base rate and a fixed 0.55 bar is never cleared. Select relative to the base
# rate instead: a signal counts when p > positive_rate * this ratio.
PROB_RATIO = float(os.getenv('INTRADAY_PROB_RATIO', 1.15))
MODEL_PATH = os.getenv('INTRADAY_MODEL_PATH', 'data/intraday_model.joblib')

DEFAULT_FAST = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META', 'AMD',
                'CRWD', 'GLW', 'NFLX', 'INTC', 'MU', 'PLTR', 'SMCI', 'AVGO',
                'QCOM', 'ORCL', 'ADBE', 'NOW', 'F', 'SOFI', 'BAC', 'T',
                'PFE', 'CSCO', 'WBD', 'RIVN', 'LCID', 'HOOD',
                # Widened 2026-09-08: the tuning sweep showed 80 symbols yields
                # 287k training rows vs 112k for 30, and the wider model scored
                # EV +1.057%/trade against +0.278%. More symbols is the cheapest
                # real gain available - 5m history is capped at 60 days, so
                # breadth is the only way to add data.
                'UBER', 'SHOP', 'SQ', 'COIN', 'MARA', 'RIOT', 'DKNG', 'SNAP',
                'PINS', 'ROKU', 'ZM', 'DOCU', 'TWLO', 'NET', 'DDOG', 'SNOW',
                'ABNB', 'LYFT', 'CVNA', 'AFRM', 'UPST', 'PATH', 'U', 'RBLX',
                'TTD', 'ETSY', 'EBAY', 'PYPL', 'V', 'MA', 'JPM', 'GS', 'MS',
                'WFC', 'C', 'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'JNJ', 'MRK',
                'ABBV', 'LLY', 'UNH', 'WMT', 'TGT', 'COST', 'HD', 'LOW']

# Crypto trades 24/7/365, so it produces ~3.7x the bars per calendar day and is
# tradeable when the stock market is shut. Yahoo serves it under the -USD suffix.
DEFAULT_CRYPTO = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD',
                  'ADA-USD', 'AVAX-USD', 'LINK-USD', 'DOT-USD', 'LTC-USD',
                  'BCH-USD', 'ATOM-USD', 'NEAR-USD']


def is_crypto(symbol: str) -> bool:
    return symbol.upper().endswith(('-USD', '-USDT'))


def asset_class(symbol: str) -> str:
    """Which model a symbol is scored by.

    Stocks and crypto have different volatility regimes and different session
    structure, so one pooled model serves neither well: measured 2026-09-08,
    a combined model scored 47.9% precision where separate models scored 90.0%
    (stocks) and 72.9% (crypto) on the same barriers.
    """
    return 'crypto' if is_crypto(symbol) else 'stock'

FEATURES = [
    'ret_1', 'ret_3', 'ret_6', 'ret_12',
    'vwap_dist', 'rsi_14', 'vol_12', 'volume_ratio',
    'day_range_pos', 'from_open', 'minutes_norm', 'gap_open',
]


def fast_universe():
    spec = os.getenv('FAST_SYMBOLS', '').strip()
    if spec:
        return [t.strip().upper() for t in spec.replace(',', ' ').split() if t.strip()]
    syms = []
    if os.getenv('TRADE_STOCKS', '1') in ('1', 'true', 'yes'):
        syms += DEFAULT_FAST
    if os.getenv('TRADE_CRYPTO', '0') in ('1', 'true', 'yes'):
        syms += DEFAULT_CRYPTO
    return syms or list(DEFAULT_FAST)


# --- market hours ---------------------------------------------------------

def market_state(now=None, symbol=None):
    """('open'|'premarket'|'afterhours'|'closed', description).

    Crypto never closes, so a crypto symbol is always 'open'.
    """
    if symbol and is_crypto(symbol):
        return 'open', '24/7 crypto'
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


def build_target(df: pd.DataFrame, cls: str = 'stock') -> pd.Series:
    """Triple-barrier label: did take-profit come before stop-loss?

    This is the honest target because it is the SAME question the executor
    answers. The old fixed-horizon label ("is price up 0.15% in 30 min") scored
    a much easier event than the trade actually placed, which is why it looked
    accurate while losing money.
    """
    tp, sl, hz = barriers(cls)
    if os.getenv('LABEL_MODE', 'triple') == 'fixed':
        return fixed_horizon(df['close'], THRESHOLD, hz)
    return triple_barrier(df['high'], df['low'], df['close'], tp, sl, hz)


class IntradayEngine:
    def __init__(self, db_path=None):
        self.conn = connect(db_path)
        # One model per asset class, each with its own base rate and bar.
        self.models = {}          # class -> fitted estimator
        self.metrics = {}         # class -> metrics dict
        self.base_rates = {}      # class -> positive rate on that class
        self.symbols = fast_universe()
        self.last_metrics = {}    # the better-performing class, for summaries
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        if Path(MODEL_PATH).exists():
            try:
                blob = joblib.load(MODEL_PATH)
                if isinstance(blob, dict) and 'models' in blob:
                    self.models = blob['models']
                    self.metrics = blob.get('metrics', {})
                    self.base_rates = blob.get('base_rates', {})
                elif isinstance(blob, dict) and 'model' in blob:
                    # Legacy single pooled model - treat it as the stock model.
                    self.models = {'stock': blob['model']}
                    self.metrics = {'stock': blob.get('meta', {})}
                    self.base_rates = {'stock': blob.get('meta', {}).get('positive_rate', 0.3)}
                self._pick_headline()
                for cls, m in self.metrics.items():
                    logger.info(f"Loaded {cls} model: precision {m.get('precision', 0):.1%} "
                                f"vs breakeven {m.get('breakeven', 0):.1%}, "
                                f"bar {self.threshold(cls):.3f}")
            except Exception as e:
                logger.warning(f"Could not load intraday models ({e})")

    @property
    def model(self):
        """Back-compat: any loaded model means the engine is usable."""
        return next(iter(self.models.values()), None)

    def _pick_headline(self):
        """last_metrics summarises whichever class currently looks strongest."""
        if self.metrics:
            self.last_metrics = max(self.metrics.values(),
                                    key=lambda m: m.get('ev', -1))

    def threshold(self, cls='stock'):
        """Selection bar for one asset class, scaled to its own base rate."""
        return min(0.95, self.base_rates.get(cls, 0.30) * prob_ratio(cls))

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
        """Train one model per asset class. Succeeds if at least one trains."""
        by_class = {}
        for sym in self.symbols:
            h = self._history(sym)
            if len(h) < 300:
                continue
            f = build_features(h)
            f['target'] = build_target(h, asset_class(sym))
            f = f.dropna()
            if not f.empty:
                by_class.setdefault(asset_class(sym), []).append(f)

        if not by_class:
            logger.error("[intraday] no usable data - fetch first")
            return False

        trained_any = False
        for cls, frames in by_class.items():
            n_syms = len(frames)
            data = pd.concat(frames).replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) < 2000:
                logger.warning(f"[intraday/{cls}] only {len(data)} rows, need 2000+")
                continue
            data = data.sort_index()
            tp, sl, hz = barriers(cls)
            X, y = data[FEATURES], data['target']
            Xtr, Xte, ytr, yte = purged_split(
                X, y, test_size=0.2, embargo_bars=hz * max(n_syms, 1))
            if len(Xtr) < 1000 or ytr.mean() <= 0.01:
                logger.warning(f"[intraday/{cls}] unusable split "
                               f"({len(Xtr)} rows, {ytr.mean():.1%} positive)")
                continue

            model = HistGradientBoostingClassifier(random_state=42,
                                                   **model_params(cls))
            model.fit(Xtr, ytr)

            acc = accuracy_score(yte, model.predict(Xte))
            base = max(yte.mean(), 1 - yte.mean())
            pos_rate = float(ytr.mean())
            bar = min(0.95, pos_rate * prob_ratio(cls))
            picked = model.predict_proba(Xte)[:, 1] >= bar
            prec = float(precision_score(yte, picked, zero_division=0)) if picked.sum() else 0.0
            breakeven = sl / (tp + sl)
            ev = prec * tp - (1 - prec) * sl

            self.models[cls] = model
            self.base_rates[cls] = pos_rate
            self.metrics[cls] = {
                'asset_class': cls, 'symbols': n_syms,
                'accuracy': acc, 'baseline': base, 'edge': acc - base,
                'rows': len(Xtr), 'positive_rate': pos_rate,
                'interval': INTERVAL, 'horizon_bars': hz,
                'take_profit': tp, 'stop_loss': sl,
                'precision': prec, 'breakeven': breakeven, 'ev': ev,
                'test_signals': int(picked.sum()), 'bar': bar,
            }
            trained_any = True
            verdict = "profitable" if ev > 0 else "LOSES MONEY"
            logger.info(f"[intraday/{cls}] {n_syms} symbols, {len(Xtr):,} rows | "
                        f"precision {prec:.1%} vs breakeven {breakeven:.1%} on "
                        f"{int(picked.sum()):,} signals | EV {ev * 100:+.3f}%/trade "
                        f"({verdict}) | bar p>{bar:.3f}")
            if ev <= 0:
                logger.warning(f"[intraday/{cls}] NEGATIVE EXPECTED VALUE - this "
                               f"asset class is not tradeable on these barriers")

        if not trained_any:
            return False
        self._pick_headline()
        joblib.dump({'models': self.models, 'metrics': self.metrics,
                     'base_rates': self.base_rates}, MODEL_PATH)
        return True

    # --- inference -------------------------------------------------------

    def signal(self, symbol):
        cls = asset_class(symbol)
        model = self.models.get(cls)
        if model is None:
            return None
        h = self._history(symbol)
        if len(h) < 60:
            return None
        f = build_features(h).replace([np.inf, -np.inf], np.nan).dropna()
        if f.empty:
            return None
        p_up = float(model.predict_proba(f[FEATURES].iloc[[-1]])[0][1])
        return {
            'symbol': symbol,
            'asset_class': cls,
            'probability': p_up,
            'bar': self.threshold(cls),
            'price': float(h['close'].iloc[-1]),
            'bar_time': h.index[-1].astimezone(ET).strftime('%H:%M ET'),
        }

    def scan(self, min_probability=None):
        """Long candidates only - the target is 'moves up more than THRESHOLD',
        so a low probability means 'no move expected', not 'goes down'."""
        out = []
        for sym in self.symbols:
            try:
                s = self.signal(sym)
            except Exception as e:
                logger.debug(f"[intraday] {sym}: {e}")
                continue
            if not s:
                continue
            bar = s['bar'] if min_probability is None else min_probability
            s['above_bar'] = s['probability'] >= bar
            # Rank by margin over each class's own bar, not raw probability -
            # the two classes have different base rates, so raw probabilities
            # are not comparable across them.
            s['margin'] = s['probability'] - bar
            if s['above_bar']:
                out.append(s)
        out.sort(key=lambda r: r['margin'], reverse=True)
        return out

    def scan_all(self):
        """Every symbol scored, ranked by margin over its own bar."""
        out = []
        for sym in self.symbols:
            try:
                s = self.signal(sym)
            except Exception:
                continue
            if s:
                s['above_bar'] = s['probability'] >= s['bar']
                s['margin'] = s['probability'] - s['bar']
                out.append(s)
        out.sort(key=lambda r: r['margin'], reverse=True)
        return out
