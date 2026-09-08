#!/usr/bin/env python3
"""
Label generation for intraday models.

WHY THIS EXISTS
---------------
The naive label - "is the price higher N bars from now" - is path-blind, and
that makes it wrong for a system with a stop-loss. Consider a bar where price
first falls 1.2% (our -1.0% stop fires, we take the loss) and only then rallies
to +2.0%. The naive label calls that a WIN and trains the model to seek it. The
actual trade was a LOSS. The model is being taught to chase outcomes the
execution rules cannot capture.

Triple-barrier labeling (Lopez de Prado, "Advances in Financial Machine
Learning") fixes this by walking the path forward bar by bar and labelling by
which barrier is touched FIRST:

    +1  take-profit hit first          -> a trade we would actually have won
     0  stop-loss hit first, or the
        horizon expired flat           -> a trade we would have lost or scratched

Because the barriers are set to the SAME values the FastTrader executes with,
the training target and the live outcome are finally the same question.
"""

import numpy as np
import pandas as pd


def triple_barrier(high, low, close, take_profit, stop_loss, horizon,
                   session=None):
    """Label each bar by which barrier its forward path touches first.

    Uses the bar HIGH/LOW, not just closes: a trade is stopped or filled
    intrabar in reality, and pretending otherwise flatters the backtest.

    `session`, if given, is an array of session ids (e.g. the ET date) aligned
    to `close`. The forward walk then STOPS at the session boundary and an
    unresolved trade counts as 0. This matters for stocks: the executor
    flattens before the bell (FAST_EOD_FLATTEN_MIN), so a take-profit that
    only arrives tomorrow morning is not a trade it can win. Measured on
    2026-09-08: 61% of 48-bar windows crossed the close and 44% of all
    take-profit labels were hit in a LATER session - the unbounded label was
    teaching the model to chase outcomes the execution rules cannot capture,
    the exact failure this module's docstring warns about.

    Returns a Series of {1, 0} aligned to `close`, with NaN where the forward
    window runs off the end of the data.
    """
    n = len(close)
    c = close.to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    sess = None if session is None else np.asarray(session)
    out = np.full(n, np.nan)

    for i in range(n - horizon):
        entry = c[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        up = entry * (1.0 + take_profit)
        dn = entry * (1.0 - stop_loss)
        label = 0.0
        for j in range(i + 1, i + horizon + 1):
            if sess is not None and sess[j] != sess[i]:
                break            # session closed with the trade unresolved
            hit_up = h[j] >= up
            hit_dn = l[j] <= dn
            if hit_up and hit_dn:
                # Both touched inside one bar - we cannot know the order from
                # OHLC, so assume the worst case. Optimism here is how
                # backtests start lying.
                label = 0.0
                break
            if hit_up:
                label = 1.0
                break
            if hit_dn:
                label = 0.0
                break
        out[i] = label
    return pd.Series(out, index=close.index)


def fixed_horizon(close, threshold, horizon):
    """The naive path-blind label, kept for comparison."""
    fwd = close.shift(-horizon) / close - 1
    return (fwd > threshold).astype(float).where(fwd.notna())


def walk_forward(X, y, n_windows=4, test_size=0.1, embargo_bars=0):
    """Yield (Xtr, Xte, ytr, yte) for consecutive test blocks, expanding train.

    One chronological hold-out is ONE sample of the model's performance, and a
    number picked because it was the best of many configs on that one sample
    is not an estimate of anything. Rolling the split forward and retraining
    each step gives several independent test periods; report the precision
    pooled across them and look at the spread. Each step purges the last
    `embargo_bars` rows before its test block, as purged_split does.
    """
    n = len(X)
    block = int(n * test_size)
    if n_windows < 1 or block < 1:
        return
    for k in range(n_windows, 0, -1):
        cut = n - k * block
        end = n - (k - 1) * block
        train_end = max(0, cut - embargo_bars)
        if train_end < 1:
            continue
        yield (X.iloc[:train_end], X.iloc[cut:end],
               y.iloc[:train_end], y.iloc[cut:end])


def purged_split(X, y, test_size=0.2, embargo_bars=0):
    """Chronological split with a gap between train and test.

    A triple-barrier label at bar i depends on bars i+1..i+horizon, so the last
    `horizon` training rows peek into the test window. Dropping (embargoing)
    that overlap is the difference between an honest score and a leaked one.
    """
    n = len(X)
    cut = int(n * (1 - test_size))
    train_end = max(0, cut - embargo_bars)
    return (X.iloc[:train_end], X.iloc[cut:],
            y.iloc[:train_end], y.iloc[cut:])
