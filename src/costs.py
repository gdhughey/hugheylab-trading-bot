#!/usr/bin/env python3
"""
Paper-fill cost model.

WHY THIS EXISTS
---------------
A paper account that fills every order at the quote and charges nothing
overstates a $500 retail account's results by roughly the round-trip cost
per trade, which for the tuned barriers is the whole edge. This module is
the single place where the assumed venue lives: a Robinhood-style starter
account with zero stock commissions, a small per-side slippage on stocks,
the SEC transaction fee and FINRA TAF on stock SELLs, and a per-side
markup on crypto with no other fee. Every assumption is an env setting so
a user on a different venue changes numbers, not code.

`fill` is pure and is called only from `BudgetTracker` (`log_trade`, and
`execute_trade` when it caps an oversized SELL at the held quantity;
`size_order` uses it for the estimated fill); no other caller applies costs
itself. Env is read at call time (not import time) so tests can vary it
with monkeypatch.
"""

import os

from src.intraday_engine import asset_class


def fill(symbol: str, side: str, ref_price: float, qty: float) -> dict:
    """Simulate one fill against the trader.

    Returns {'fill_price', 'fees', 'gross', 'net'} where
      gross = fill_price * qty
      net   = gross + fees on BUY (cash out), gross - fees on SELL (cash in)

    Stocks: fill = ref * (1 +/- STOCK_SLIPPAGE_BPS/1e4). BUY fees are 0.
            SELL fees = SEC_FEE_RATE * gross + min(FINRA_TAF_PER_SHARE * qty,
            FINRA_TAF_CAP), left unrounded so tiny orders still pay.
    Crypto: fill = ref * (1 +/- CRYPTO_SPREAD_BPS/1e4) per side; no other fee.
    """
    if side not in ('BUY', 'SELL'):
        # A wrong side would silently flip the sign of slippage; fail loudly.
        raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")
    cls = asset_class(symbol)
    if cls == 'crypto':
        bps = float(os.getenv('CRYPTO_SPREAD_BPS', 60))
    else:
        bps = float(os.getenv('STOCK_SLIPPAGE_BPS', 5))
    # Slippage/spread always moves the price against the trader.
    sign = 1.0 if side == 'BUY' else -1.0
    fill_price = ref_price * (1 + sign * bps / 1e4)
    gross = fill_price * qty

    fees = 0.0
    if cls == 'stock' and side == 'SELL':
        # SEC Section 31 fee is on sale proceeds; FINRA TAF is per share sold
        # with a per-trade cap. Neither applies to buys or to crypto.
        sec = float(os.getenv('SEC_FEE_RATE', 0.0000206)) * gross
        taf = min(float(os.getenv('FINRA_TAF_PER_SHARE', 0.000195)) * qty,
                  float(os.getenv('FINRA_TAF_CAP', 9.79)))
        fees = sec + taf

    net = gross + fees if side == 'BUY' else gross - fees
    return {'fill_price': fill_price, 'fees': fees, 'gross': gross, 'net': net}


def round_trip_cost(cls: str) -> float:
    """Estimated cost of a BUY+SELL round trip as a fraction of notional.

    Used by the class gate (is take-profit above the cost of trading?) and by
    the scorecard's cost-adjusted breakeven, so it must stay a plain fraction:
      'stock'  -> 2 * STOCK_SLIPPAGE_BPS/1e4 + SEC_FEE_RATE
      'crypto' -> 2 * CRYPTO_SPREAD_BPS/1e4
    The FINRA TAF is per share, not per dollar, so it is not part of this
    estimate; `fill` still charges it on the actual SELL.
    """
    if cls == 'stock':
        return (2 * float(os.getenv('STOCK_SLIPPAGE_BPS', 5)) / 1e4
                + float(os.getenv('SEC_FEE_RATE', 0.0000206)))
    if cls == 'crypto':
        return 2 * float(os.getenv('CRYPTO_SPREAD_BPS', 60)) / 1e4
    raise ValueError(f"unknown asset class {cls!r}")


def qty_str(x: float) -> str:
    """Format a quantity for logs and embeds: 6dp, trailing zeros trimmed.

    qty_str(1.0) == '1', qty_str(0.5) == '0.5', qty_str(10.0) == '10'.
    Quantities are stored rounded to 6dp (BudgetTracker.log_trade), so this
    never loses stored precision; unrounded inputs are rounded by the format.
    """
    return f"{x:.6f}".rstrip('0').rstrip('.')
