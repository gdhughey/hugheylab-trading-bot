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


# --- ledger helpers --------------------------------------------------------

def _net_return(row) -> float | None:
    """Net return of one closed SELL row as a fraction of the cost it closed.

    execute_trade books realized_pnl = amount - closed * avg_price, so the
    cost basis of the closed lot is amount - realized_pnl; no position lookup
    is needed and a partially closed lot is handled for free.
    """
    pnl = row['realized_pnl']
    if pnl is None:
        return None
    basis = float(row['amount']) - float(pnl)
    return float(pnl) / basis if basis > 0 else None


def _max_drawdown_pct(series: list[dict]) -> float:
    """Largest peak-to-trough fall in equity, in PERCENTAGE POINTS of the
    running peak (0.2 means 0.2%).

    `series` is BudgetTracker.equity_series(): starting_cash as day 0, then
    one point per recorded ET date.
    """
    peak, worst = 0.0, 0.0
    for point in series:
        equity = float(point['equity'])
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak * 100)
    return worst


def _signal_stats(conn, cls: str, opened_at: str) -> dict:
    """Primary endpoint: TP-first rate of labelled above-bar signals since open.

    Two intervals: Wilson on the raw count, and a day-clustered one (mean of
    per-trade_date rates ± 1.96 * sd / sqrt(days)) because signals inside one
    session are not independent - a trending day lifts every symbol's label
    at once. With a single day there is no spread to estimate, so the
    clustered lower bound is 0: one day can never argue for GO.
    """
    rows = conn.execute(
        "SELECT trade_date, label FROM signals "
        "WHERE asset_class = ? AND above_bar = 1 AND label IS NOT NULL AND bar_ts >= ? "
        "ORDER BY trade_date", (cls, opened_at)).fetchall()
    n = len(rows)
    hits = sum(int(r['label']) for r in rows)
    lo, hi = wilson_ci(hits, n)

    by_day = {}
    for r in rows:
        tally = by_day.setdefault(r['trade_date'], [0, 0])
        tally[0] += int(r['label'])
        tally[1] += 1
    day_rates = [h / k for h, k in by_day.values()]
    if len(day_rates) >= 2:
        lo_day = max(0.0, statistics.fmean(day_rates)
                     - Z95 * statistics.stdev(day_rates) / math.sqrt(len(day_rates)))
    else:
        lo_day = 0.0
    return {'sig_n': n, 'sig_hits': hits, 'sig_rate': hits / n if n else 0.0,
            'sig_lo': lo, 'sig_hi': hi, 'sig_lo_day': lo_day, 'sig_days': len(day_rates)}


# --- the scorecard ---------------------------------------------------------

def build_scorecard(budget, engine, intraday, day_et: str, now=None) -> dict:
    """Everything the daily report, /pnl and /summary print, as one dict.

    budget    BudgetTracker (the ledger).
    engine    MLEngine: stored_close() for marks and the SPY benchmark,
              first_close_on_or_after() for the benchmark's start.
    intraday  IntradayEngine (walk-forward metrics per class) or None.
    day_et    the ET date being reported ("today" everywhere).
    now       UTC tz-aware instant used for settlement; defaults to the clock.

    Keys (see the contract): headline, account, today, closed_today,
    positions, since_open, classes, spy.

    Units: fractions are fractions; ONLY exec_mean_pct, exec_ci and
    max_drawdown_pct are percentage points (render with f"{x:.2f}%", never a
    '%' format spec). account.unsettled_until is an ET DATE string
    'YYYY-MM-DD' or None (render verbatim, never parse as a datetime).
    bt_precision / ev_bt / ev_bt_net are None for a class with no metrics
    entry (renderers print 'n/a'). closed_today items carry both `net` /
    `realized_pnl` and `gross` / `gross_pnl`.
    """
    now = now or datetime.now(timezone.utc)
    now_iso = now.isoformat(timespec='seconds')
    opened_at = budget.opened_at()
    starting_cash = budget.starting_cash()
    metrics = (getattr(intraday, 'metrics', None) or {}) if intraday is not None else {}

    # Positions are marked at the last stored close, never a live quote: the
    # 16:05 report must not fail because a quote provider is down, and after
    # the bell the stored close IS the day's price.
    pnl = budget.get_pnl(engine.stored_close)
    unsettled = budget.get_unsettled(now=now)
    buying_power = budget.get_buying_power(now=now)
    until = budget.conn.execute(
        "SELECT MAX(available_at) AS until FROM trades WHERE side = 'SELL' "
        "AND status = 'EXECUTED' AND available_at > ? AND created_at >= ?",
        (now_iso, opened_at)).fetchone()['until']
    # Stocks always settle at 09:30 ET, so the ET DATE is the whole story; the
    # embed prints "settles {unsettled_until} 09:30 ET" verbatim. Deliberately
    # not a timestamp: a date parsed as a naive midnight and converted to ET
    # would land on the previous day.
    unsettled_until = _et_date(until) if until else None

    trades = budget.get_trades_since_open()
    closed = [r for r in trades if r['side'] == 'SELL' and r['realized_pnl'] is not None]
    nets = [float(r['realized_pnl']) for r in closed]
    n_closed = len(nets)
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x < 0]
    win_lo, win_hi = wilson_ci(len(wins), n_closed)
    _, _, net_hw = mean_ci(nets)
    gross_loss = -sum(losses)

    today_rows = [r for r in trades if r['trade_date'] == day_et]
    closed_today = [r for r in today_rows if r['side'] == 'SELL' and r['realized_pnl'] is not None]
    day_state = budget.get_day_state(day_et)

    positions = []
    for p in pnl['positions']:
        price = p.get('price')
        positions.append({
            'symbol': p['symbol'], 'shares': float(p['shares']),
            'avg_price': float(p['avg_price']), 'entry_ref': float(p['entry_ref']),
            'price': price, 'market_value': p.get('market_value'),
            'pnl': p.get('pnl'), 'pnl_pct': p.get('pnl_pct'),
            # Barrier progress is measured ref-to-ref (spec section 2), so the
            # report shows the same number the exit rule is looking at.
            'pct_vs_ref': (price / p['entry_ref'] - 1)
                          if price is not None and p['entry_ref'] else None,
        })

    classes = {}
    for cls in CLASSES:
        m = metrics.get(cls)
        cost = costs.round_trip_cost(cls)
        tp, _, _ = barriers(cls)
        # class_gate accepts None metrics (Task 6 scores a missing entry as
        # ev 0.0, so the text is the EV-gate or FAST_IGNORE_EV line); only the
        # text is used here - the verdict's `gated` is computed below.
        _, gate_text = class_gate(cls, m)
        cls_closed = [r for r in closed if asset_class(r['symbol']) == cls]
        rets = [x for x in (_net_return(r) for r in cls_closed) if x is not None]
        _, exec_se, exec_hw = mean_ci(rets)
        by_reason = {}
        for r in cls_closed:
            by_reason.setdefault(r['exit_reason'] or 'unknown', []).append(float(r['realized_pnl']))
        tp_n = sum(len(by_reason.get(k, [])) for k in BARRIER_EXITS)
        tp_hits = len(by_reason.get('tp', []))
        tp_lo, tp_hi = wilson_ci(tp_hits, tp_n)
        ev_bt = None if m is None else m.get('ev')
        stats = {
            # Section 8 names the COST gate specifically, not the EV gate: a
            # class the EV gate refuses may still be worth extending; one
            # that costs make unwinnable cannot. Same inequality as
            # class_gate's first step.
            'gated': tp <= cost,
            'exec_n': len(rets),
            'exec_mean': statistics.fmean(rets) if rets else 0.0,
            'exec_se': exec_se,
            'ev_bt': ev_bt,
            'cost': cost,
            **_signal_stats(budget.conn, cls, opened_at),
        }
        label, text = verdict(cls, stats)
        classes[cls] = {
            **stats,
            'verdict': label,
            'verdict_text': text,
            'breakeven': cost_breakeven(cls),
            'exits_by_reason': {k: {'n': len(v), 'mean_net': statistics.fmean(v)}
                                for k, v in by_reason.items()},
            'tp_n': tp_n,
            'tp_first_rate': tp_hits / tp_n if tp_n else 0.0,
            'tp_lo': tp_lo, 'tp_hi': tp_hi,
            # None (not 0) when the class has no walk-forward metrics, so the
            # report says "n/a" instead of claiming a 0% backtest precision.
            'bt_precision': None if m is None else m.get('precision'),
            'bt_n': 0 if m is None else int(m.get('test_signals', 0)),
            'ev_bt_net': None if ev_bt is None else ev_bt - cost,
            # PERCENTAGE POINTS for display (0.1 means 0.1%); exec_mean and
            # exec_se above stay fractions for the verdict arithmetic.
            'exec_mean_pct': stats['exec_mean'] * 100,
            'exec_ci': exec_hw * 100,
            'gate_text': gate_text,
        }

    # SPY buy-and-hold on the same starting cash over the same period. Context
    # only, not risk-matched: SPY is exposed 24/7, the bot is flat overnight.
    spy = None
    start_close = engine.first_close_on_or_after('SPY', opened_at[:10])
    last_close = engine.stored_close('SPY')
    if start_close and last_close:
        pct = float(last_close) / float(start_close) - 1
        spy = {'start_close': float(start_close), 'last_close': float(last_close),
               'pct': pct, 'value': starting_cash * (1 + pct)}

    card = {
        'headline': {
            'all_time_net': pnl['all_time_net'],
            'all_time_pct': pnl['all_time_pct'],
            'n_closed': n_closed,
            # 1.96 * SE * n: the interval on the SUM of per-trade nets, i.e.
            # on the realised part of the headline dollar figure.
            'ci_dollars': net_hw * n_closed,
        },
        'account': {
            'equity': pnl['equity'], 'cash': pnl['cash'],
            'buying_power': buying_power, 'unsettled': unsettled,
            'unsettled_until': unsettled_until, 'starting_cash': starting_cash,
            'gross_pnl': pnl['gross_pnl'], 'fees_paid': pnl['fees_paid'],
        },
        'today': {
            'realized': sum(float(r['realized_pnl']) for r in closed_today),
            'unrealized': pnl['unrealized'],
            'n_trades': len(today_rows),
            'wins': sum(1 for r in closed_today if r['realized_pnl'] > 0),
            'losses': sum(1 for r in closed_today if r['realized_pnl'] < 0),
            'loss_tripped': bool(day_state is not None and day_state['loss_tripped_at']),
        },
        # Both spellings of the P&L keys on purpose: `net`/`gross` match the
        # cycle summary's exit dict vocabulary, `realized_pnl`/`gross_pnl`
        # match the ledger columns; a renderer may use either.
        'closed_today': [{
            'symbol': r['symbol'], 'shares': float(r['shares']), 'price': float(r['price']),
            'amount': float(r['amount']),
            'net': float(r['realized_pnl']), 'realized_pnl': float(r['realized_pnl']),
            'gross': float(r['gross_pnl']), 'gross_pnl': float(r['gross_pnl']),
            'fees': float(r['fees']),
            'exit_reason': r['exit_reason'], 'created_at': r['created_at'],
        } for r in closed_today],
        'positions': positions,
        'since_open': {
            'n_closed': n_closed,
            'win_rate': len(wins) / n_closed if n_closed else 0.0,
            'win_lo': win_lo, 'win_hi': win_hi,
            'mean_net': statistics.fmean(nets) if nets else 0.0,
            'mean_ci': net_hw,
            'profit_factor': (sum(wins) / gross_loss) if gross_loss > 0 else None,
            'max_drawdown_pct': _max_drawdown_pct(budget.equity_series()),
            # Calendar days, opening day counted as day 1.
            'days_running': (date.fromisoformat(day_et)
                             - date.fromisoformat(_et_date(opened_at))).days + 1,
        },
        'classes': classes,
        'spy': spy,
    }
    logger.info(f"Scorecard {day_et}: all-time ${card['headline']['all_time_net']:+,.2f} "
                f"on n={n_closed} | "
                + " | ".join(f"{c} {v['verdict']}" for c, v in classes.items()))
    return card
