#!/usr/bin/env python3
"""
Hyperparameter + universe search, scored on held-out expected value.

"Train it more" is not how a trading model improves - extra iterations on the
same data just memorise noise, and the backtest gets prettier while the money
does not. What actually helps is training MANY honest configurations and
keeping the one that wins out of sample.

Every candidate here is scored the same way the live bot is judged:
    EV = precision x take_profit - (1 - precision) x stop_loss
on a purged, embargoed, chronological hold-out. Configs with too few test
signals are rejected outright - a 100% precision on 40 signals is overfitting
wearing a nice suit.
"""
import argparse, itertools, json, os, sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

from src.intraday_engine import build_features, FEATURES, DEFAULT_FAST, DEFAULT_CRYPTO, ET
from src.labeling import triple_barrier, walk_forward

MIN_TEST_SIGNALS = 300      # pooled across windows; below this, precision is not measurable
WF_WINDOWS = 4              # consecutive out-of-sample windows per candidate

EXTRA_STOCKS = ['UBER','SHOP','SQ','COIN','MARA','RIOT','DKNG','SNAP','PINS','ROKU',
                'ZM','DOCU','TWLO','NET','DDOG','SNOW','ABNB','LYFT','CVNA','AFRM',
                'UPST','PATH','U','RBLX','TTD','ETSY','EBAY','PYPL','V','MA',
                'JPM','GS','MS','WFC','C','XOM','CVX','COP','SLB','OXY',
                'JNJ','MRK','ABBV','LLY','UNH','WMT','TGT','COST','HD','LOW']


def load(symbols, period='60d', interval='5m', chunk=60):
    out = {}
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i+chunk]
        raw = yf.download(part, period=period, interval=interval, auto_adjust=True,
                          progress=False, group_by='ticker', threads=True)
        if raw is None or raw.empty:
            continue
        for s in part:
            try:
                d = raw[s].rename(columns=str.lower).dropna(subset=['close'])
                if len(d) > 400:
                    out[s] = d
            except Exception:
                pass
    return out


def dataset(data, tp, sl, hz, cls='stock'):
    frames = []
    for s, d in data.items():
        f = build_features(d)
        # Stocks are flattened at the bell, so their label stops there too.
        session = None if cls == 'crypto' else d.index.tz_convert(ET).date
        f['target'] = triple_barrier(d['high'], d['low'], d['close'], tp, sl, hz,
                                     session=session)
        f = f.dropna()
        if not f.empty:
            frames.append(f)
    if not frames:
        return None, None
    df = pd.concat(frames).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    return df[FEATURES], df['target']


def score(X, y, params, ratios, tp, sl, hz, n_syms):
    """Walk-forward score: retrain on each of WF_WINDOWS consecutive windows
    and pool the out-of-sample signals. A single 80/20 hold-out let one lucky
    (or leaky) fortnight pick the winner out of 180 candidates."""
    hits = {r: 0 for r in ratios}; cnt = {r: 0 for r in ratios}
    per_window = {r: [] for r in ratios}
    base = rows = None
    for Xtr, Xte, ytr, yte in walk_forward(X, y, n_windows=WF_WINDOWS, test_size=0.1,
                                           embargo_bars=hz * max(n_syms, 1)):
        if len(Xtr) < 5000 or ytr.mean() <= 0.02:
            continue
        m = HistGradientBoostingClassifier(random_state=42, **params)
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        base = float(ytr.mean()); rows = len(Xtr)
        for r in ratios:
            picked = proba >= min(0.95, base * r)
            n = int(picked.sum())
            hits[r] += int(yte[picked].sum()); cnt[r] += n
            per_window[r].append(round(float(yte[picked].mean()), 3) if n else None)
    if base is None:
        return None
    best = None
    for r in ratios:
        n = cnt[r]
        if n < MIN_TEST_SIGNALS:
            continue
        prec = hits[r] / n
        ev = prec * tp - (1 - prec) * sl
        if best is None or ev > best['ev']:
            best = dict(ev=float(ev), precision=float(prec), signals=n,
                        ratio=r, bar=float(min(0.95, base * r)), base=base, rows=rows,
                        windows=per_window[r])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', choices=['stock', 'crypto', 'both'], default='both')
    ap.add_argument('--wide', action='store_true', help='add 50 more stock symbols')
    ap.add_argument('--out', default='data/tuned.json')
    args = ap.parse_args()

    classes = {}
    if args.asset in ('stock', 'both'):
        syms = DEFAULT_FAST + (EXTRA_STOCKS if args.wide else [])
        classes['stock'] = syms
    if args.asset in ('crypto', 'both'):
        classes['crypto'] = DEFAULT_CRYPTO

    PARAM_GRID = [
        dict(max_iter=250, learning_rate=0.06, max_depth=5, l2_regularization=1.0),
        dict(max_iter=400, learning_rate=0.04, max_depth=6, l2_regularization=1.0),
        dict(max_iter=250, learning_rate=0.06, max_depth=3, l2_regularization=2.0),
        dict(max_iter=600, learning_rate=0.03, max_depth=6, l2_regularization=3.0),
        dict(max_iter=400, learning_rate=0.05, max_depth=8, l2_regularization=2.0),
        dict(max_iter=300, learning_rate=0.08, max_depth=4, l2_regularization=0.5),
    ]
    BARRIERS = [(0.008, 0.005, 24), (0.010, 0.006, 24), (0.006, 0.004, 24),
                (0.008, 0.005, 36), (0.012, 0.007, 48), (0.010, 0.006, 12)]
    RATIOS = (1.2, 1.4, 1.6, 2.0, 2.5)

    results = {}
    for cls, syms in classes.items():
        t0 = time.time()
        data = load(syms)
        print(f"\n=== {cls.upper()}: {len(data)}/{len(syms)} symbols loaded "
              f"in {time.time()-t0:.0f}s ===")
        print(f"{'TP':>6}{'SL':>7}{'bars':>6}{'iter':>6}{'lr':>6}{'depth':>6}"
              f"{'rows':>9}{'signals':>9}{'precision':>11}{'EV/trade':>11}")
        best = None
        for tp, sl, hz in BARRIERS:
            X, y = dataset(data, tp, sl, hz, cls)
            if X is None:
                continue
            for params in PARAM_GRID:
                r = score(X, y, params, RATIOS, tp, sl, hz, len(data))
                if not r:
                    continue
                cand = dict(asset_class=cls, take_profit=tp, stop_loss=sl,
                            horizon=hz, params=params, **r)
                mark = ''
                if best is None or r['ev'] > best['ev']:
                    best = cand; mark = '  <<< best so far'
                print(f"{tp:>6.1%}{sl:>7.1%}{hz:>6}{params['max_iter']:>6}"
                      f"{params['learning_rate']:>6.2f}{params['max_depth']:>6}"
                      f"{r['rows']:>9,}{r['signals']:>9,}{r['precision']:>11.1%}"
                      f"{r['ev']*100:>+10.3f}%{mark}")
        if best:
            results[cls] = best
            print(f"\n  BEST {cls}: TP +{best['take_profit']:.1%} / SL -{best['stop_loss']:.1%} "
                  f"/ {best['horizon']*5}min | {best['params']} | ratio {best['ratio']}")
            print(f"    precision {best['precision']:.1%} on {best['signals']:,} signals "
                  f"-> EV {best['ev']*100:+.3f}%/trade")
            print(f"    per-window precision: {best.get('windows')}")

    if results:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")
        print("\nApply the winners by setting these in .env:")
        for cls, b in results.items():
            print(f"  # {cls}: EV {b['ev']*100:+.3f}%/trade")
        b = max(results.values(), key=lambda x: x['ev'])
        print(f"  FAST_TAKE_PROFIT={b['take_profit']}")
        print(f"  FAST_STOP_LOSS={b['stop_loss']}")
        print(f"  INTRADAY_HORIZON_BARS={b['horizon']}")
        print(f"  INTRADAY_PROB_RATIO={b['ratio']}")


if __name__ == '__main__':
    main()
