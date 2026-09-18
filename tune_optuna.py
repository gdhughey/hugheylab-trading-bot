#!/usr/bin/env python3
"""
Optuna + CPCV + Deflated-Sharpe tuner for the intraday model.

Replaces tune.py's 180-point grid with a TPE search scored by Combinatorial
Purged CV: each trial trains on C(6,2)=15 purged splits and reports the
distribution of out-of-sample EV, not one lucky path. The best trial is then
held to the Deflated Sharpe bar - its t-statistic must exceed the expected
maximum of N pure-noise trials - and the verdict is written next to the
config. Nothing here is applied to the live bot: read the verdict, then copy
to data/tuned.json by hand if it passes.

Data comes from the ledger DB's prices_intraday table (the collector and the
fast loop already keep it fresh); no Yahoo calls.

  venv/bin/python tune_optuna.py --asset stock --trials 100 --out data/tuned_optuna.json
  OMP_NUM_THREADS=3 nice -n 19 ...   # when sharing a box with the live bot
"""
import argparse, json, os, sys, time, math, statistics, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.database import connect
from src.intraday_engine import build_features, FEATURES, ET, is_crypto
from src.labeling import triple_barrier, cpcv_splits, deflated_threshold

MIN_TEST_SIGNALS = 300       # per split; below this precision is noise
N_GROUPS, K_TEST = 6, 2


def load_from_db(cls, interval='5m', min_rows=400):
    conn = connect()
    syms = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM prices_intraday WHERE interval = ?", (interval,))]
    syms = [s for s in syms if is_crypto(s) == (cls == 'crypto')]
    out = {}
    for s in syms:
        df = pd.read_sql_query(
            "SELECT ts, open, high, low, close, volume FROM prices_intraday "
            "WHERE symbol = ? AND interval = ? ORDER BY ts", conn, params=(s, interval))
        if len(df) < min_rows:
            continue
        df['ts'] = pd.to_datetime(df['ts'], utc=True, format='ISO8601')
        out[s] = df.set_index('ts')
    return out


def dataset(data, tp, sl, hz, cls):
    frames = []
    for s, d in data.items():
        f = build_features(d)
        session = None if cls == 'crypto' else d.index.tz_convert(ET).date
        f['target'] = triple_barrier(d['high'], d['low'], d['close'], tp, sl, hz, session=session)
        f = f.dropna()
        if not f.empty:
            frames.append(f)
    if not frames:
        return None, None
    df = pd.concat(frames).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    return df[FEATURES].to_numpy(), df['target'].to_numpy()


def evaluate(X, y, params, ratio, tp, sl, hz, n_syms):
    """CPCV: list of per-split EV, plus pooled precision/signals."""
    evs, hits, cnt = [], 0, 0
    for tr, te in cpcv_splits(len(X), N_GROUPS, K_TEST, embargo_bars=hz * max(n_syms, 1)):
        ytr, yte = y[tr], y[te]
        if len(tr) < 5000 or ytr.mean() <= 0.02:
            return None
        m = HistGradientBoostingClassifier(random_state=42, **params).fit(X[tr], ytr)
        p = m.predict_proba(X[te])[:, 1]
        picked = p >= min(0.95, ytr.mean() * ratio)
        n = int(picked.sum())
        if n < MIN_TEST_SIGNALS:
            return None
        prec = float(yte[picked].mean())
        evs.append(prec * tp - (1 - prec) * sl)
        hits += int(yte[picked].sum()); cnt += n
    if not evs:
        return None
    return {'ev_mean': float(np.mean(evs)), 'ev_sd': float(np.std(evs, ddof=1)) if len(evs) > 1 else 0.0,
            'ev_splits': [round(e, 5) for e in evs], 'precision': hits / cnt, 'signals': cnt,
            'n_splits': len(evs)}


def main():
    import optuna
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', choices=['stock', 'crypto'], default='stock')
    ap.add_argument('--trials', type=int, default=60)
    ap.add_argument('--timeout-min', type=float, default=None)
    ap.add_argument('--out', default='data/tuned_optuna.json')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.time()
    data = load_from_db(args.asset)
    print(f"{args.asset}: {len(data)} symbols from the DB in {time.time() - t0:.0f}s", flush=True)
    if not data:
        sys.exit("no intraday bars in the DB for that class")
    cache = {}

    def objective(trial):
        tp = trial.suggest_float('take_profit', 0.005, 0.015, step=0.001)
        sl = trial.suggest_float('stop_loss', 0.003, 0.010, step=0.001)
        hz = trial.suggest_categorical('horizon', [12, 24, 36, 48])
        ratio = trial.suggest_float('ratio', 1.1, 2.5, step=0.1)
        params = dict(max_iter=trial.suggest_int('max_iter', 100, 600, step=50),
                      learning_rate=trial.suggest_float('learning_rate', 0.02, 0.12, log=True),
                      max_depth=trial.suggest_int('max_depth', 2, 8),
                      l2_regularization=trial.suggest_float('l2_regularization', 0.0, 5.0, step=0.5),
                      min_samples_leaf=trial.suggest_int('min_samples_leaf', 20, 200, step=20))
        key = (tp, sl, hz)
        if key not in cache:
            cache[key] = dataset(data, tp, sl, hz, args.asset)
        X, y = cache[key]
        if X is None:
            return -1.0
        r = evaluate(X, y, params, ratio, tp, sl, hz, len(data))
        if r is None:
            return -1.0
        trial.set_user_attr('result', r)
        print(f"  trial {trial.number:3d}  TP {tp:.1%} SL {sl:.1%} hz {hz:2d} r {ratio:.1f} "
              f"depth {params['max_depth']} lr {params['learning_rate']:.3f} -> "
              f"EV {r['ev_mean'] * 100:+.3f}% ± {r['ev_sd'] * 100:.3f} on {r['signals']:,} signals "
              f"(prec {r['precision']:.1%})", flush=True)
        return r['ev_mean']

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials,
                   timeout=args.timeout_min * 60 if args.timeout_min else None)

    done = [t for t in study.trials if t.value is not None and t.value > -1.0]
    if not done:
        sys.exit("no trial produced enough test signals")
    best = max(done, key=lambda t: t.value)
    r = best.user_attrs['result']
    # Deflated Sharpe: t-stat of the best trial's split EVs vs the expected max of N noise trials
    t_best = r['ev_mean'] / (r['ev_sd'] / math.sqrt(r['n_splits'])) if r['ev_sd'] > 0 else float('inf')
    trial_ts = [t.user_attrs['result']['ev_mean'] / (t.user_attrs['result']['ev_sd'] / math.sqrt(t.user_attrs['result']['n_splits']))
                for t in done if t.user_attrs['result']['ev_sd'] > 0]
    bar = deflated_threshold(len(done), statistics.pvariance(trial_ts) if len(trial_ts) > 1 else 0.0)
    passes = bool(r['ev_mean'] > 0 and t_best > bar)

    out = {args.asset: {
        'take_profit': best.params['take_profit'], 'stop_loss': best.params['stop_loss'],
        'horizon': best.params['horizon'], 'ratio': best.params['ratio'],
        'params': {k: best.params[k] for k in ('max_iter', 'learning_rate', 'max_depth', 'l2_regularization', 'min_samples_leaf')},
        'ev': r['ev_mean'], 'precision': r['precision'], 'signals': r['signals'],
        'cpcv': {'n_groups': N_GROUPS, 'k_test': K_TEST, 'ev_splits': r['ev_splits'], 'ev_sd': r['ev_sd']},
        'dsr': {'trials_scored': len(done), 't_best': t_best, 'expected_max_noise_t': bar, 'passes': passes},
        'symbols': len(data), 'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'verdict': ('PASS: EV positive and t-stat clears the deflated bar' if passes else
                    'FAIL: keep the current model - this is not evidence of an edge'),
    }}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"\nBEST {args.asset}: EV {r['ev_mean'] * 100:+.3f}%/trade ± {r['ev_sd'] * 100:.3f} across "
          f"{r['n_splits']} CPCV splits, precision {r['precision']:.1%} on {r['signals']:,} signals")
    print(f"  t-stat {t_best:.2f} vs deflated bar {bar:.2f} after {len(done)} scored trials -> {out[args.asset]['verdict']}")
    print(f"  wrote {args.out} (NOT applied; copy into data/tuned.json only if it passes)")


if __name__ == '__main__':
    main()
