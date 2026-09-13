#!/usr/bin/env python3
"""
Scorecard - the numbers behind the daily report and the go/no-go verdict.

Everything here is arithmetic over the ledger (BudgetTracker), the signal log
and the intraday walk-forward metrics; nothing is cached and nothing is
written. The decision rule from the design spec (section 8) lives in
`verdict()` and nowhere else, so the 16:05 report, /pnl and /summary can
never disagree about it.
"""

import logging
import math
import statistics
from datetime import date, datetime, timezone

from src import costs
from src.budget_tracker import _et_date
from src.fast_trader import class_gate
from src.intraday_engine import asset_class, barriers

logger = logging.getLogger(__name__)

# Normal 97.5th percentile. scipy is not a declared dependency (it only rides
# in with scikit-learn), so intervals use z rather than Student's t; at the
# n >= 60 the GO rule requires, t(0.975, 59) = 2.00, a 2% wider band.
Z95 = 1.96

# Reported in this order, always both: the decision rule is defined per class
# and crypto's NO-GO-by-cost is itself a finding worth printing every day.
CLASSES = ('stock', 'crypto')

# SELL exit reasons that count toward the realised TP-first rate. 'manual' is
# a human override, not a barrier outcome, so it is excluded from both the
# numerator and the denominator.
BARRIER_EXITS = ('tp', 'sl', 'timeout', 'eod')


def wilson_ci(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 1) when n == 0.

    Wilson rather than the normal approximation because the executed-trade
    sample is small (tens) and the rates sit well away from 0.5, where the
    normal interval is known to under-cover.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = hits / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    """(mean, standard error, 95% half-width); (0, 0, 0) with fewer than 2 values.

    Sample standard deviation (n - 1), z = 1.96 - see the note on Z95.
    """
    n = len(values)
    if n < 2:
        return (0.0, 0.0, 0.0)
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    return (mean, se, Z95 * se)


def cost_breakeven(cls: str) -> float:
    """TP-first rate at which a class breaks even AFTER round-trip costs.

    A trade wins tp or loses sl and pays c either way, so
    p*tp - (1-p)*sl - c = 0  =>  p = (sl + c) / (tp + sl).
    Stocks at tp 1.0% / sl 0.6% / c 0.1%: 43.9% (naive 37.5%). Crypto at
    defaults comes out above 100%: no crypto trade can be net positive.
    """
    tp, sl, _ = barriers(cls)
    return (sl + costs.round_trip_cost(cls)) / (tp + sl)


def verdict(cls: str, stats: dict) -> tuple[str, str]:
    """Spec section 8 - the only place the go/no-go rule lives.

    NO-GO  the cost gate blocks the class, or the signal-precision Wilson CI
           sits entirely below the cost-adjusted breakeven.
    GO     both signal lower bounds (raw Wilson and day-clustered) clear
           breakeven, at least 60 executed trades, and the executed mean net
           return is within one SE of the backtest's cost-adjusted EV.
    EXTEND anything else - not enough evidence either way.

    stats keys: gated (the COST gate only - an EV-gated class still reads
    EXTEND), sig_n, sig_lo, sig_hi, sig_lo_day, exec_n, exec_mean, exec_se,
    ev_bt (None when the class has no trained model), cost.
    Returns (label, the numbers that decided it).
    """
    be = cost_breakeven(cls)
    if stats['gated']:
        tp, _, _ = barriers(cls)
        return ('NO-GO', f"cost gate blocks {cls}: round-trip cost {stats['cost']:.2%} "
                         f"is at or above the {tp:.2%} take-profit")
    if stats['sig_hi'] < be:
        return ('NO-GO', f"signal precision CI upper bound {stats['sig_hi']:.1%} "
                         f"is below breakeven {be:.1%} (n={stats['sig_n']})")

    ev_bt = stats['ev_bt']
    target = None if ev_bt is None else ev_bt - stats['cost']
    unmet = []
    if not stats['sig_lo'] > be:
        unmet.append(f"signal CI lower {stats['sig_lo']:.1%} ≤ breakeven {be:.1%} "
                     f"(n={stats['sig_n']})")
    if not stats['sig_lo_day'] > be:
        unmet.append(f"day-clustered lower {stats['sig_lo_day']:.1%} ≤ breakeven {be:.1%}")
    if stats['exec_n'] < 60:
        unmet.append(f"executed n={stats['exec_n']} < 60")
    if target is None:
        unmet.append("no backtest EV to compare executed returns against")
    elif not stats['exec_mean'] >= target - stats['exec_se']:
        unmet.append(f"executed mean {stats['exec_mean']:+.3%} < backtest {target:+.3%} "
                     f"− SE {stats['exec_se']:.3%} (n={stats['exec_n']})")
    if unmet:
        return ('EXTEND', '; '.join(unmet))
    return ('GO', f"signal CI lower {stats['sig_lo']:.1%} and day-clustered "
                  f"{stats['sig_lo_day']:.1%} clear breakeven {be:.1%} (n={stats['sig_n']}); "
                  f"executed mean {stats['exec_mean']:+.3%} ≥ backtest {target:+.3%} − SE "
                  f"{stats['exec_se']:.3%} (n={stats['exec_n']})")
