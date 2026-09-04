#!/usr/bin/env python3
"""
Manually refresh price data and retrain the model.

    /opt/trading-bot/venv/bin/python /opt/trading-bot/train.py            # fetch + train
    /opt/trading-bot/venv/bin/python /opt/trading-bot/train.py --no-fetch # train on cached data
    /opt/trading-bot/venv/bin/python /opt/trading-bot/train.py --scan     # also show top picks
    /opt/trading-bot/venv/bin/python /opt/trading-bot/train.py --target forward_5d

The running bot retrains on its own at startup; this is for retraining without
restarting it, or for comparing target modes.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-7s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('train')

from src.database import Database
from src.ml_engine import TradingSignalEngine, load_universe


def main():
    ap = argparse.ArgumentParser(description="Refresh data and retrain the model")
    ap.add_argument('--no-fetch', action='store_true', help="skip download, use cached prices")
    ap.add_argument('--scan', action='store_true', help="print the top signals afterwards")
    ap.add_argument('--target', help="override TARGET_MODE (next_day | forward_5d)")
    ap.add_argument('--universe', help="override UNIVERSE (sp500 | default | AAPL,MSFT,...)")
    args = ap.parse_args()

    t0 = time.time()
    Database()
    engine = TradingSignalEngine()

    symbols = load_universe(args.universe) if args.universe else load_universe()
    engine.symbols = symbols
    log.info(f"Universe: {len(symbols)} symbols")

    if not args.no_fetch:
        t = time.time()
        rows = engine.fetch_and_store_data(symbols)
        log.info(f"Fetched {rows:,} rows in {time.time() - t:.0f}s")
        for name, st in engine.source_stats.items():
            log.info(f"  source {name}: {st['symbols']} symbols, {st['rows']:,} rows")
    else:
        log.info("Skipping fetch (--no-fetch)")

    t = time.time()
    ok = engine.train_model(target_mode=args.target)
    if not ok:
        log.error("Training FAILED - not enough usable data")
        return 1

    m = engine.last_metrics
    log.info(f"Trained in {time.time() - t:.0f}s")
    print(f"\n  target      {m['mode']}")
    print(f"  rows        {m['rows']:,}")
    print(f"  accuracy    {m['accuracy']:.4f}")
    print(f"  baseline    {m['baseline']:.4f}   (always guessing the majority class)")
    print(f"  edge        {m['edge']:+.4f}")
    if m['edge'] <= 0.005:
        print("\n  NOTE: an edge at or below ~0.005 is indistinguishable from noise.")
        print("  Treat the output as a shortlist to review, not a prediction.")

    if args.scan:
        top = engine.scan(symbols, float(os.getenv('MIN_PROBABILITY', 0.55)), None)
        buys = [r for r in top if r['signal'] == 1]
        print(f"\n  {len(top)} cleared the bar: {len(buys)} buy / {len(top) - len(buys)} sell")
        print("\n  Top buys:")
        for r in buys[:10]:
            print(f"    {r['symbol']:<6} ${r['price']:>9,.2f}  {r['probability']:.1%}")
        if not buys:
            print("    (none - model is negative on the whole market today)")

    print(f"\n  total {time.time() - t0:.0f}s\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
