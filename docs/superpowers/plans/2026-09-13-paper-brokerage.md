# Paper Brokerage Account Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make paper mode behave like a real $500 cash brokerage account (settlement, fills with costs, fractional shares, compounding balance) and post a daily all-time P&L report with a pre-registered GO/NO-GO verdict, so two months of paper results support a real-money decision.

**Architecture:** `BudgetTracker` becomes an account ledger over new `account`/`day_state`/`equity_history`/`signals` tables; `costs.py` is the single fill/fee model, called only from `log_trade`; `FastTrader` measures barriers reference-to-reference and gates classes via `class_gate`; `scorecard.py` builds the report and verdict for the 16:05 ET daily message, `/pnl` and `/summary`. Every piece of state survives the nightly 03:30 restart because it lives in SQLite.

**Tech Stack:** Python 3.13, sqlite3 (WAL), pandas, scikit-learn, discord.py 2.x, pytest (run in LXC 200 via `dev/ct-test.sh`).

**Spec:** `docs/superpowers/specs/2026-09-12-paper-brokerage-design.md`
**Interface contract (binding names):** `docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md`

---

### Task 1: costs — `src/costs.py` (fills, fees, round-trip cost, quantity formatter)

Pure functions only. No DB, no clock, no I/O. Env is read at **call time** with
`os.getenv` so tests can vary it with `monkeypatch.setenv`. `asset_class` is
imported from `src.intraday_engine` (already exists at
`src/intraday_engine.py:116`; it returns `'crypto'` for `-USD`/`-USDT` suffixed
symbols, `'stock'` otherwise). Nothing else in the repo changes in this task.

**Files:**
- Create: `/home/gdhughey/hugheylab-trading-bot/src/costs.py`
- Create: `/home/gdhughey/hugheylab-trading-bot/tests/test_costs.py`

Reference numbers used by the tests (all with the contract defaults
`STOCK_SLIPPAGE_BPS=5`, `CRYPTO_SPREAD_BPS=60`, `SEC_FEE_RATE=0.0000206`,
`FINRA_TAF_PER_SHARE=0.000195`, `FINRA_TAF_CAP=9.79`):

| case | fill | gross | fees | net |
|---|---|---|---|---|
| stock BUY, ref 100, qty 10 | 100.05 | 1000.5 | 0 | 1000.5 |
| stock SELL, ref 100, qty 10 | 99.95 | 999.5 | 0.0000206×999.5 + 0.000195×10 = 0.0225397 | 999.4774603 |
| stock SELL, ref 1.0, qty 100000 (TAF capped) | 0.9995 | 99950.0 | 2.05897 + 9.79 = 11.84897 | 99938.15103 |
| stock SELL, ref 10, qty 0.1 (tiny, unrounded) | 9.995 | 0.9995 | 0.0000206×0.9995 + 0.000195×0.1 = 0.0000400897 | 0.9994599103 |
| crypto BUY, ref 100, qty 0.5 | 100.6 | 50.3 | 0 | 50.3 |
| crypto SELL, ref 100, qty 0.5 | 99.4 | 49.7 | 0 | 49.7 |

`round_trip_cost('stock')` = 2×5/1e4 + 0.0000206 = **0.0010206**;
`round_trip_cost('crypto')` = 2×60/1e4 = **0.012**.

Every number in the table above and every `pytest.approx` literal in the tests
below was recomputed against the Step 3 / Step 7 code before this section was
published; note in particular that the tiny-order SEC fee is on the **gross
after slippage** (0.9995), not on `ref × qty` (1.0).

- [ ] **Step 1: Write the failing tests for `fill()`**

Create `/home/gdhughey/hugheylab-trading-bot/tests/test_costs.py` with exactly
this content. The autouse fixture pins every cost env var to the documented
default so the result does not depend on whatever `.env` or shell environment
exists on LXC 200; individual tests then override with `monkeypatch.setenv`.

```python
"""Tests for src/costs.py — the paper-fill cost model.

Every function under test is pure and reads its parameters from the
environment at call time, so each test pins the env it needs via monkeypatch
and never touches a database or the clock.
"""
import pytest

from src import costs


# Contract defaults (docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md).
DEFAULT_ENV = {
    'STOCK_SLIPPAGE_BPS': '5',
    'CRYPTO_SPREAD_BPS': '60',
    'SEC_FEE_RATE': '0.0000206',
    'FINRA_TAF_PER_SHARE': '0.000195',
    'FINRA_TAF_CAP': '9.79',
}


@pytest.fixture(autouse=True)
def pin_default_env(monkeypatch):
    # The container's shell or a loaded .env may carry different values;
    # pin the documented defaults so every assertion below is deterministic.
    for k, v in DEFAULT_ENV.items():
        monkeypatch.setenv(k, v)


# ---------------------------------------------------------------- fill(): stocks

def test_stock_buy_fills_above_ref_with_no_fees():
    r = costs.fill('AAPL', 'BUY', 100.0, 10.0)
    assert set(r) == {'fill_price', 'fees', 'gross', 'net'}
    assert r['fill_price'] == pytest.approx(100.05)      # 100 * (1 + 5/1e4)
    assert r['gross'] == pytest.approx(1000.5)
    assert r['fees'] == 0.0
    assert r['net'] == pytest.approx(1000.5)             # BUY: net = gross + fees


def test_stock_sell_fills_below_ref_and_charges_sec_plus_taf():
    r = costs.fill('AAPL', 'SELL', 100.0, 10.0)
    assert r['fill_price'] == pytest.approx(99.95)       # 100 * (1 - 5/1e4)
    assert r['gross'] == pytest.approx(999.5)
    expected_fees = 0.0000206 * 999.5 + 0.000195 * 10    # SEC on proceeds + TAF per share
    assert expected_fees == pytest.approx(0.0225397)     # anchor the literal
    assert r['fees'] == pytest.approx(expected_fees)
    assert r['net'] == pytest.approx(999.5 - expected_fees)   # SELL: net = gross - fees
    assert r['net'] == pytest.approx(999.4774603)


def test_stock_sell_taf_is_capped():
    # 100,000 shares * 0.000195 = 19.50 TAF, which the cap pulls down to 9.79.
    r = costs.fill('AAPL', 'SELL', 1.0, 100_000.0)
    assert r['fill_price'] == pytest.approx(0.9995)
    assert r['gross'] == pytest.approx(99_950.0)
    assert r['fees'] == pytest.approx(0.0000206 * 99_950.0 + 9.79)
    assert r['fees'] == pytest.approx(11.84897)
    assert r['net'] == pytest.approx(99_938.15103)


def test_stock_sell_fees_are_unrounded():
    # Tiny order: fees must not be rounded to cents or zeroed.
    # fill = 10 * (1 - 5/1e4) = 9.995, gross = 9.995 * 0.1 = 0.9995; the SEC
    # fee is on that slipped gross, not on ref * qty.
    r = costs.fill('AAPL', 'SELL', 10.0, 0.1)
    assert r['gross'] == pytest.approx(0.9995)
    assert r['fees'] == pytest.approx(0.0000206 * 0.9995 + 0.000195 * 0.1)
    assert r['fees'] == pytest.approx(0.0000400897)
    assert r['fees'] > 0


def test_stock_fee_env_overrides(monkeypatch):
    monkeypatch.setenv('SEC_FEE_RATE', '0')
    monkeypatch.setenv('FINRA_TAF_PER_SHARE', '0')
    r = costs.fill('AAPL', 'SELL', 100.0, 10.0)
    assert r['fees'] == 0.0
    assert r['net'] == pytest.approx(r['gross'])

    monkeypatch.setenv('FINRA_TAF_PER_SHARE', '1')      # 1 $/share -> 10, capped to 1
    monkeypatch.setenv('FINRA_TAF_CAP', '1')
    r = costs.fill('AAPL', 'SELL', 100.0, 10.0)
    assert r['fees'] == pytest.approx(1.0)


def test_stock_slippage_env_override(monkeypatch):
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '20')
    assert costs.fill('AAPL', 'BUY', 100.0, 1.0)['fill_price'] == pytest.approx(100.2)
    assert costs.fill('AAPL', 'SELL', 100.0, 1.0)['fill_price'] == pytest.approx(99.8)

    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '0')
    assert costs.fill('AAPL', 'BUY', 100.0, 1.0)['fill_price'] == pytest.approx(100.0)


# ---------------------------------------------------------------- fill(): crypto

def test_crypto_buy_fills_above_ref_no_fees():
    r = costs.fill('BTC-USD', 'BUY', 100.0, 0.5)
    assert r['fill_price'] == pytest.approx(100.6)       # 100 * (1 + 60/1e4)
    assert r['gross'] == pytest.approx(50.3)
    assert r['fees'] == 0.0
    assert r['net'] == pytest.approx(50.3)


def test_crypto_sell_fills_below_ref_no_fees():
    r = costs.fill('BTC-USD', 'SELL', 100.0, 0.5)
    assert r['fill_price'] == pytest.approx(99.4)        # 100 * (1 - 60/1e4)
    assert r['gross'] == pytest.approx(49.7)
    assert r['fees'] == 0.0                              # SEC/TAF never apply to crypto
    assert r['net'] == pytest.approx(49.7)


def test_crypto_ignores_stock_slippage_and_stock_fees(monkeypatch):
    # Even with absurd stock parameters, crypto is priced only by CRYPTO_SPREAD_BPS.
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '500')
    monkeypatch.setenv('SEC_FEE_RATE', '0.5')
    monkeypatch.setenv('FINRA_TAF_PER_SHARE', '5')
    r = costs.fill('ETH-USD', 'SELL', 100.0, 1.0)
    assert r['fill_price'] == pytest.approx(99.4)
    assert r['fees'] == 0.0


def test_crypto_spread_env_override(monkeypatch):
    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '10')
    assert costs.fill('BTC-USD', 'BUY', 100.0, 1.0)['fill_price'] == pytest.approx(100.1)
    assert costs.fill('BTC-USD', 'SELL', 100.0, 1.0)['fill_price'] == pytest.approx(99.9)


def test_fill_rejects_unknown_side():
    with pytest.raises(ValueError):
        costs.fill('AAPL', 'buy', 100.0, 1.0)      # case-sensitive by design
    with pytest.raises(ValueError):
        costs.fill('AAPL', 'SHORT', 100.0, 1.0)
```

- [ ] **Step 2: Run the tests and confirm they fail because the module does not exist**

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_costs.py -v
```

Expected: collection error, no tests run. The output ends with lines like

```
ERROR tests/test_costs.py - ImportError: cannot import name 'costs' from 'src' (/opt/trading-bot-dev/src/__init__.py)
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

(`ModuleNotFoundError: No module named 'src.costs'` is the equivalent
acceptable text depending on the pytest version.)

- [ ] **Step 3: Implement `fill()` in `src/costs.py`**

Create `/home/gdhughey/hugheylab-trading-bot/src/costs.py` with exactly this
content (`round_trip_cost` and `qty_str` are added in Step 7; the file is
complete as shown here for this step):

```python
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

`fill` is pure and is called from exactly one place,
`BudgetTracker.log_trade`; no caller applies costs itself. Env is read at
call time (not import time) so tests can vary it with monkeypatch.
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
```

- [ ] **Step 4: Run the `fill()` tests and confirm they pass**

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_costs.py -v
```

Expected: 11 passed, e.g.

```
tests/test_costs.py::test_stock_buy_fills_above_ref_with_no_fees PASSED
tests/test_costs.py::test_stock_sell_fills_below_ref_and_charges_sec_plus_taf PASSED
tests/test_costs.py::test_stock_sell_taf_is_capped PASSED
tests/test_costs.py::test_stock_sell_fees_are_unrounded PASSED
tests/test_costs.py::test_stock_fee_env_overrides PASSED
tests/test_costs.py::test_stock_slippage_env_override PASSED
tests/test_costs.py::test_crypto_buy_fills_above_ref_no_fees PASSED
tests/test_costs.py::test_crypto_sell_fills_below_ref_no_fees PASSED
tests/test_costs.py::test_crypto_ignores_stock_slippage_and_stock_fees PASSED
tests/test_costs.py::test_crypto_spread_env_override PASSED
tests/test_costs.py::test_fill_rejects_unknown_side PASSED
11 passed in ...
```

- [ ] **Step 5: Write the failing tests for `round_trip_cost()` and `qty_str()`**

Append the following to the END of
`/home/gdhughey/hugheylab-trading-bot/tests/test_costs.py` (after
`test_fill_rejects_unknown_side`):

```python


# ---------------------------------------------------------------- round_trip_cost()

def test_round_trip_cost_stock_default():
    # 2 * 5 bps slippage + SEC rate (TAF is per share, not a fraction of notional,
    # so it is deliberately excluded from the fractional estimate).
    assert costs.round_trip_cost('stock') == pytest.approx(2 * 5 / 1e4 + 0.0000206)
    assert costs.round_trip_cost('stock') == pytest.approx(0.0010206)


def test_round_trip_cost_crypto_default():
    assert costs.round_trip_cost('crypto') == pytest.approx(2 * 60 / 1e4)
    assert costs.round_trip_cost('crypto') == pytest.approx(0.012)


def test_round_trip_cost_reads_env_at_call_time(monkeypatch):
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '10')
    monkeypatch.setenv('SEC_FEE_RATE', '0')
    assert costs.round_trip_cost('stock') == pytest.approx(0.002)

    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '10')
    assert costs.round_trip_cost('crypto') == pytest.approx(0.002)


def test_round_trip_cost_rejects_unknown_class():
    with pytest.raises(ValueError):
        costs.round_trip_cost('forex')


# ---------------------------------------------------------------- qty_str()

@pytest.mark.parametrize('x, expected', [
    (1.0, '1'),
    (0.5, '0.5'),
    (10.0, '10'),           # rstrip('0') must not eat the integer zero
    (100.0, '100'),
    (0.0, '0'),
    (0.123457, '0.123457'), # a 6dp-rounded qty (what log_trade stores) round-trips
    (0.123456789, '0.123457'),  # unrounded input is formatted to 6dp; no extra digits
    (2.5e-06, '0.000003'),
    (1234.5, '1234.5'),
])
def test_qty_str(x, expected):
    assert costs.qty_str(x) == expected
```

- [ ] **Step 6: Run the tests and confirm the new ones fail with AttributeError**

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_costs.py -v
```

Expected: 11 passed, 13 failed; every failure reads

```
AttributeError: module 'src.costs' has no attribute 'round_trip_cost'
```

or

```
AttributeError: module 'src.costs' has no attribute 'qty_str'
```

- [ ] **Step 7: Implement `round_trip_cost()` and `qty_str()`**

Append the following to the END of
`/home/gdhughey/hugheylab-trading-bot/src/costs.py` (after the `return` of
`fill`):

```python


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
```

- [ ] **Step 8: Run the full test file and confirm everything passes**

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_costs.py -v
```

Expected: `24 passed` (11 fill tests, 4 round_trip_cost tests, 9 qty_str
parametrized cases), no failures, no errors.

Also confirm the existing smoke test still passes alongside it:

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/ -q
```

Expected: `25 passed` (24 + `tests/test_smoke.py::test_smoke`). If other
tasks have already landed their test files the count is higher; the
requirement is zero failures in `tests/test_costs.py`.

- [ ] **Step 9: Commit**

```bash
cd /home/gdhughey/hugheylab-trading-bot && git add src/costs.py tests/test_costs.py && git commit -m "$(cat <<'EOF'
Add costs module: paper fills, stock SEC/TAF fees, round-trip cost, qty_str

Pure functions reading STOCK_SLIPPAGE_BPS, CRYPTO_SPREAD_BPS, SEC_FEE_RATE,
FINRA_TAF_PER_SHARE and FINRA_TAF_CAP at call time. fill() moves the price
against the trader per side and charges SEC + capped TAF on stock SELLs
only; crypto pays the per-side spread and nothing else. round_trip_cost()
feeds the class gate and the scorecard breakeven; qty_str() is the one
quantity formatter for logs and embeds.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp
EOF
)"
```

**Contract additions**
- None. All names (`fill`, `round_trip_cost`, `qty_str`) and signatures match the contract exactly. Behavioural detail not spelled out in the contract: `fill` raises `ValueError` for a `side` other than exactly `'BUY'`/`'SELL'`, and `round_trip_cost` raises `ValueError` for a `cls` other than `'stock'`/`'crypto'`.

---

### Task 2: Calendar helpers (`is_trading_day`, `next_trading_day_open`)

Pure functions, no DB, no network. `src/intraday_engine.py` imports pandas/sklearn/yfinance at module level, so the tests only run inside the container via `dev/ct-test.sh`; that is expected. The `datetime.now(ET)` default in `market_state` is never exercised — every test passes `now=` explicitly.

Two spec gaps are resolved here, both in favour of "behaviour otherwise unchanged" (contract, Task 2 block):

- Between an early close (13:00 ET) and 16:00 ET on an `EARLY_CLOSE_2026` date, the existing `market_state` falls through every branch and returns `('closed', 'overnight')`. That is pre-existing behaviour; this task pins it in the characterization test rather than changing it. If a later task wants 13:00–20:00 on an early-close day to read `afterhours`, that is a separate behaviour change with its own test.
- The spec's "Stocks resume" line in `_daily_summary_embed` is not touched here: Task 9 deletes `_daily_summary_embed`, and the scorecard already carries the settlement date via `account.unsettled_until` (built from `trades.available_at`, which Task 4 fills from `next_trading_day_open`). Nothing in this task depends on that line.

**Files:**
- Create: `/home/gdhughey/hugheylab-trading-bot/tests/test_calendar.py`
- Modify: `/home/gdhughey/hugheylab-trading-bot/src/intraday_engine.py`
  - line 17 (the `from datetime import ...` line: add `date`)
  - lines 151-154 (`US_HOLIDAYS_2026`: add `'2027-01-01'`)
  - insert two new functions between line 155 (`EARLY_CLOSE_2026 = ...`) and line 158 (`def market_state`)
  - lines 158-179 (`market_state`: weekend + holiday checks become one `is_trading_day` check)

- [ ] **Step 1: Write the characterization tests for `market_state` (they pin the behaviour the Step 5 refactor must keep)**

Create `/home/gdhughey/hugheylab-trading-bot/tests/test_calendar.py` with exactly this content:

```python
"""Trading-calendar helpers in src.intraday_engine.

Every call passes now=/ts= explicitly; nothing here touches the wall clock.
Dates used (all 2026 unless noted):
  Mon 09-14 .. Fri 09-18 is the account's first week; Sat 09-19 / Sun 09-20 weekend
  Thu 11-26 Thanksgiving (holiday), Fri 11-27 early close 13:00 ET
  Thu 12-31, Fri 2027-01-01 New Year's Day (holiday), Mon 2027-01-04
"""
from datetime import date, datetime, timedelta, timezone

import pytest

from src.intraday_engine import (ET, US_HOLIDAYS_2026, is_trading_day,
                                 market_state, next_trading_day_open)


def et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


# --- market_state: behaviour must be unchanged by the refactor -------------

@pytest.mark.parametrize('now, expected', [
    (et(2026, 9, 15, 10, 0), ('open', 'regular session')),          # Tue mid-session
    (et(2026, 9, 15, 8, 0), ('premarket', 'pre-market')),            # Tue 08:00
    (et(2026, 9, 15, 17, 0), ('afterhours', 'after hours')),         # Tue 17:00
    (et(2026, 9, 15, 2, 0), ('closed', 'overnight')),                # Tue 02:00
    (et(2026, 9, 19, 12, 0), ('closed', 'weekend')),                 # Sat
    (et(2026, 9, 20, 12, 0), ('closed', 'weekend')),                 # Sun
    (et(2026, 11, 26, 12, 0), ('closed', 'market holiday')),         # Thanksgiving
    (et(2027, 1, 1, 12, 0), ('closed', 'market holiday')),           # New Year's Day 2027
    (et(2026, 11, 27, 12, 0), ('open', 'early close 13:00 ET')),     # early-close Friday
    # 13:00 on an early-close day: not open, and the after-hours branch only
    # starts at 16:00, so the existing code falls through to 'overnight'.
    # Pinned as-is (pre-existing behaviour); changing it is out of scope.
    (et(2026, 11, 27, 13, 0), ('closed', 'overnight')),
])
def test_market_state_cases(now, expected):
    assert market_state(now=now) == expected


def test_market_state_converts_utc_input():
    # 14:00 UTC on Tue 09-15 is 10:00 EDT -> regular session
    assert market_state(now=datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc)) == ('open', 'regular session')


def test_market_state_crypto_always_open():
    assert market_state(now=et(2026, 9, 19, 12, 0), symbol='BTC-USD') == ('open', '24/7 crypto')
```

- [ ] **Step 2: Run the file and confirm it fails at collection**

The import line already names `is_trading_day` and `next_trading_day_open`, which do not exist yet, so pytest cannot collect the module. That is the intended red state; do not edit the import to make it pass.

Run:
```
/home/gdhughey/hugheylab-trading-bot/dev/ct-test.sh tests/test_calendar.py -v
```
Expected: collection error, nothing runs:
```
ERROR tests/test_calendar.py - ImportError: cannot import name 'is_trading_day' from 'src.intraday_engine' (/opt/trading-bot-dev/src/intraday_engine.py)
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

- [ ] **Step 3: Add the failing calendar tests**

Append to the end of `/home/gdhughey/hugheylab-trading-bot/tests/test_calendar.py`:

```python
# --- is_trading_day -------------------------------------------------------

def test_2027_new_years_day_is_in_holiday_set():
    assert '2027-01-01' in US_HOLIDAYS_2026


@pytest.mark.parametrize('d, expected', [
    (date(2026, 9, 14), True),    # Mon
    (date(2026, 9, 18), True),    # Fri
    (date(2026, 9, 19), False),   # Sat
    (date(2026, 9, 20), False),   # Sun
    (date(2026, 11, 26), False),  # Thanksgiving
    (date(2026, 11, 27), True),   # early-close day is still a trading day
    (date(2026, 12, 25), False),  # Christmas
    (date(2027, 1, 1), False),    # New Year's Day 2027 (the added entry)
    (date(2027, 1, 4), True),     # first trading day of 2027
])
def test_is_trading_day(d, expected):
    assert is_trading_day(d) is expected


# --- next_trading_day_open ------------------------------------------------

def test_friday_sell_settles_monday_0930_et():
    got = next_trading_day_open(et(2026, 9, 18, 15, 55))
    assert got == et(2026, 9, 21, 9, 30)
    assert got.strftime('%a') == 'Mon'


def test_wednesday_before_thanksgiving_settles_friday():
    # Thu 11-26 is a holiday, so T+1 from Wed 11-25 is Fri 11-27.
    assert next_trading_day_open(et(2026, 11, 25, 12, 0)) == et(2026, 11, 27, 9, 30)


def test_new_years_eve_settles_first_trading_day_of_2027():
    # Fri 2027-01-01 is a holiday, then Sat/Sun -> Mon 2027-01-04.
    assert next_trading_day_open(et(2026, 12, 31, 15, 0)) == et(2027, 1, 4, 9, 30)


def test_strictly_after_the_input_date():
    # Monday 09:00 ET (pre-market) must NOT return the same Monday's 09:30.
    assert next_trading_day_open(et(2026, 9, 14, 9, 0)) == et(2026, 9, 15, 9, 30)


def test_utc_input_is_converted_to_et_date():
    # 02:00 UTC on Tue 09-15 is still Mon 09-14 22:00 EDT, so the answer is Tue 09-15,
    # not Wed 09-16 (which is what a naive .date() on the UTC value would give).
    got = next_trading_day_open(datetime(2026, 9, 15, 2, 0, tzinfo=timezone.utc))
    assert got == et(2026, 9, 15, 9, 30)


def test_result_is_tz_aware_et():
    got = next_trading_day_open(datetime(2026, 9, 18, 19, 55, tzinfo=timezone.utc))
    assert got.tzinfo is ET
    assert got.hour == 9 and got.minute == 30 and got.second == 0 and got.microsecond == 0
    # EDT on 09-21: the UTC rendering Task 4 stores must be 13:30Z
    assert got.astimezone(timezone.utc) == datetime(2026, 9, 21, 13, 30, tzinfo=timezone.utc)
    assert got.utcoffset() == timedelta(hours=-4)
```

Run:
```
/home/gdhughey/hugheylab-trading-bot/dev/ct-test.sh tests/test_calendar.py -v
```
Expected: still the collection error
```
ERROR tests/test_calendar.py - ImportError: cannot import name 'is_trading_day' from 'src.intraday_engine' (/opt/trading-bot-dev/src/intraday_engine.py)
```

- [ ] **Step 4: Implement the holiday entry and the two helpers in `src/intraday_engine.py`**

4a. Line 17 — replace

```python
from datetime import datetime, time as dtime, timedelta, timezone
```
with
```python
from datetime import date, datetime, time as dtime, timedelta, timezone
```

4b. Lines 151-155 — replace the holiday block

```python
US_HOLIDAYS_2026 = {
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
    '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
}
EARLY_CLOSE_2026 = {'2026-11-27', '2026-12-24'}   # 13:00 ET
```
with
```python
US_HOLIDAYS_2026 = {
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
    '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
    # 2027-01-01 is here so a SELL on 2026-12-31 settles on Mon 2027-01-04,
    # not on the holiday Friday. The paper run reviews on 2026-11-13, but the
    # service keeps running past year end.
    '2027-01-01',
}
EARLY_CLOSE_2026 = {'2026-11-27', '2026-12-24'}   # 13:00 ET


def is_trading_day(d: date) -> bool:
    """True when the US stock market is open at all on calendar date d.

    Early-close days count as trading days. strftime (not isoformat) so a
    datetime passed by mistake still compares as a bare date.
    """
    return d.weekday() < 5 and d.strftime('%Y-%m-%d') not in US_HOLIDAYS_2026


def next_trading_day_open(ts: datetime) -> datetime:
    """09:30 ET on the first trading day STRICTLY after ts's ET calendar date.

    ts must be tz-aware (any zone); it is converted to ET before the date is
    taken, because a Monday-evening SELL logged as Tuesday 02:00 UTC still
    settles Tuesday, not Wednesday (T+1 for a cash account). Returns a
    tz-aware ET datetime; callers that store it convert to UTC ISO themselves.
    """
    d = ts.astimezone(ET).date() + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return datetime.combine(d, dtime(9, 30), tzinfo=ET)
```

Run:
```
/home/gdhughey/hugheylab-trading-bot/dev/ct-test.sh tests/test_calendar.py -v
```
Expected: `28 passed` (12 market_state: 10 parametrized cases + `test_market_state_converts_utc_input` + `test_market_state_crypto_always_open`; 1 holiday-set; 9 is_trading_day; 6 next_trading_day_open). The market_state characterization cases run against the still-unrefactored `market_state` here and all pass, which is what makes them a valid baseline for Step 5.

- [ ] **Step 5: Refactor `market_state` to use `is_trading_day` (behaviour unchanged, descriptions kept)**

Replace the whole function at (now shifted) lines ~184-205 of `/home/gdhughey/hugheylab-trading-bot/src/intraday_engine.py`:

```python
def market_state(now=None, symbol=None):
    """('open'|'premarket'|'afterhours'|'closed', description).

    Crypto never closes, so a crypto symbol is always 'open'.
    """
    if symbol and is_crypto(symbol):
        return 'open', '24/7 crypto'
    now = (now or datetime.now(ET)).astimezone(ET)
    if now.weekday() >= 5:
        return 'closed', 'weekend'
    day = now.strftime('%Y-%m-%d')
    if day in US_HOLIDAYS_2026:
        return 'closed', 'market holiday'
    t = now.time()
    close = dtime(13, 0) if day in EARLY_CLOSE_2026 else dtime(16, 0)
    if dtime(9, 30) <= t < close:
        return 'open', 'early close 13:00 ET' if day in EARLY_CLOSE_2026 else 'regular session'
    if dtime(4, 0) <= t < dtime(9, 30):
        return 'premarket', 'pre-market'
    if dtime(16, 0) <= t < dtime(20, 0):
        return 'afterhours', 'after hours'
    return 'closed', 'overnight'
```
with
```python
def market_state(now=None, symbol=None):
    """('open'|'premarket'|'afterhours'|'closed', description).

    Crypto never closes, so a crypto symbol is always 'open'.
    """
    if symbol and is_crypto(symbol):
        return 'open', '24/7 crypto'
    now = (now or datetime.now(ET)).astimezone(ET)
    if not is_trading_day(now.date()):
        # Same calendar as settlement (next_trading_day_open) so the two can
        # never disagree about whether today counts; the description is kept
        # because the Discord status line prints it.
        return 'closed', 'weekend' if now.weekday() >= 5 else 'market holiday'
    day = now.strftime('%Y-%m-%d')
    t = now.time()
    close = dtime(13, 0) if day in EARLY_CLOSE_2026 else dtime(16, 0)
    if dtime(9, 30) <= t < close:
        return 'open', 'early close 13:00 ET' if day in EARLY_CLOSE_2026 else 'regular session'
    if dtime(4, 0) <= t < dtime(9, 30):
        return 'premarket', 'pre-market'
    if dtime(16, 0) <= t < dtime(20, 0):
        return 'afterhours', 'after hours'
    return 'closed', 'overnight'
```

Run the calendar file and the existing smoke test together:
```
/home/gdhughey/hugheylab-trading-bot/dev/ct-test.sh tests/test_calendar.py tests/test_smoke.py -v
```
Expected: `29 passed` (28 calendar + `tests/test_smoke.py::test_smoke`). If any `test_market_state_cases[...]` case fails, the refactor changed a return value — the Step 4 run proved every case passes against the original code, so fix `market_state`, never the test.

- [ ] **Step 6: Commit**

```
cd /home/gdhughey/hugheylab-trading-bot && git add src/intraday_engine.py tests/test_calendar.py && git commit -m "$(cat <<'EOF'
Trading calendar: is_trading_day, next_trading_day_open, 2027-01-01 holiday

market_state now derives closed-day status from is_trading_day so the
session check and T+1 settlement share one calendar. Descriptions
('weekend' / 'market holiday') are unchanged; tests/test_calendar.py pins
them plus the settlement cases (Fri -> Mon, Thanksgiving skip, year end,
UTC input converted to the ET date, ET-aware result).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp
EOF
)"
```

**Contract additions**
- None. Names and signatures are exactly the contract's `is_trading_day(d: date) -> bool` and `next_trading_day_open(ts: datetime) -> datetime`.
- Behaviour clarification (no new name): `market_state` between 13:00 and 16:00 ET on an `EARLY_CLOSE_2026` date returns `('closed', 'overnight')` — the pre-existing fall-through, now pinned by `tests/test_calendar.py::test_market_state_cases`.


---

### Task 3: Schema — REAL shares, new tables, additive migrations, account seed

**Files:**
- Modify: `/home/gdhughey/hugheylab-trading-bot/src/database.py` — imports (lines 6-9), `SCHEMA` (lines 15-51), class `Database` (lines 66-98: `__init__` 69-73, `init_schema` 75-78, `_migrate` 80-92, `close` 94-98)
- Create: `/home/gdhughey/hugheylab-trading-bot/tests/conftest.py` (fixtures `db_path`, `old_db_path`)
- Create: `/home/gdhughey/hugheylab-trading-bot/tests/test_migration.py`

Context for the engineer: `Database(path)` is the only thing that creates or upgrades the schema (`main.py:63` and `train.py:45` run it before any other connection). `dev/ct-test.sh` runs pytest with `DB_PATH=:memory:`, so every test passes an explicit file path under `tmp_path`. `src/` does NOT call `load_dotenv`, so tests see only the shell environment plus whatever `monkeypatch` sets. Existing databases keep their `INTEGER`-declared `shares` columns: SQLite column affinity stores `0.5` as REAL losslessly, so the tables are not rebuilt.

Clock rule for every test in this repo (also recorded under Contract additions): the account seed stamps `opened_at` from `now` or, failing that, the wall clock, and every ledger read filters `created_at >= opened_at`. A test that constructs `Database()` must therefore either pass `now=` (a tz-aware UTC datetime earlier than every timestamp the test uses) or pin `account.opened_at` with a direct `UPDATE` immediately after construction. Never let the wall clock decide `opened_at` in a test.

- [ ] **Step 1: Create the shared fixtures**

Create `/home/gdhughey/hugheylab-trading-bot/tests/conftest.py`:

```python
"""
Shared fixtures.

Every test uses a FILE database under tmp_path. dev/ct-test.sh runs pytest with
DB_PATH=:memory:, and each connect(':memory:') would be a separate empty
database, so the same path must be passed explicitly to every component
(Database, BudgetTracker, engines).

Clock rule: Database() seeds account.opened_at from `now` or the wall clock,
and every ledger read filters created_at >= opened_at. A test that constructs
Database() itself MUST pass now= (earlier than every timestamp it uses) or
pin opened_at with an UPDATE right after - never rely on the wall clock, or
the test turns into a time bomb the day the suite runs later than its data.
"""

import sqlite3
from datetime import datetime, timezone

import pytest

from src.database import Database

# Fixed clock for the account seed so account.opened_at never depends on wall
# time. Early enough that any 2026 `now=` a test passes is >= opened_at, which
# is what the "created_at >= opened_at" report filters need.
OPENED_AT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# The schema exactly as shipped BEFORE the paper-brokerage change (git dd28b31):
# INTEGER shares, no account/day_state/equity_history/signals tables, none of
# the new trade/position columns. Kept verbatim so test_migration exercises the
# real upgrade path a production database will take.
OLD_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    source      TEXT,                       -- which provider supplied this bar
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL,          -- BUY | SELL
    price       REAL    NOT NULL,
    shares      INTEGER NOT NULL,
    amount      REAL    NOT NULL,
    status      TEXT    NOT NULL,          -- PENDING | EXECUTED | REJECTED
    created_at  TEXT    NOT NULL,
    settled_at  TEXT,
    week_key    TEXT    NOT NULL,          -- ISO year-week, for weekly budget rollover
    realized_pnl REAL                       -- set on SELL execution; NULL for BUY
);

CREATE INDEX IF NOT EXISTS idx_trades_week   ON trades (week_key, status);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);

CREATE TABLE IF NOT EXISTS positions (
    symbol      TEXT PRIMARY KEY,
    shares      INTEGER NOT NULL DEFAULT 0,
    avg_price   REAL    NOT NULL DEFAULT 0,
    updated_at  TEXT
);
"""

# Twenty-one legacy trades (spec section 5): nine closed round trips (BUY then
# SELL, with the realized_pnl the old code wrote) and three rejected BUYs,
# spread over two ISO weeks. Ids are assigned in list order (1..21), which the
# migration tests rely on: id 1 is the first BUY, id 2 the SELL that closed it.
LEGACY_TRADES = [
    # (symbol, side, price, shares, amount, status, created_at, settled_at, week_key, realized_pnl)
    ('AAPL', 'BUY',  150.0, 2, 300.0, 'EXECUTED', '2026-09-01T14:35:00+00:00', '2026-09-01T14:35:05+00:00', '2026-W36', None),   # 1
    ('AAPL', 'SELL', 155.0, 2, 310.0, 'EXECUTED', '2026-09-02T15:00:00+00:00', '2026-09-02T15:00:05+00:00', '2026-W36', 10.0),   # 2
    ('MSFT', 'BUY',  400.0, 1, 400.0, 'REJECTED', '2026-09-03T14:40:00+00:00', '2026-09-03T14:41:00+00:00', '2026-W36', None),   # 3
    ('NVDA', 'BUY',  120.0, 3, 360.0, 'EXECUTED', '2026-09-03T15:10:00+00:00', '2026-09-03T15:10:05+00:00', '2026-W36', None),   # 4
    ('NVDA', 'SELL', 118.0, 3, 354.0, 'EXECUTED', '2026-09-03T19:50:00+00:00', '2026-09-03T19:50:05+00:00', '2026-W36', -6.0),   # 5
    ('AMD',  'BUY',  160.0, 2, 320.0, 'EXECUTED', '2026-09-04T14:35:00+00:00', '2026-09-04T14:35:05+00:00', '2026-W36', None),   # 6
    ('AMD',  'SELL', 164.0, 2, 328.0, 'EXECUTED', '2026-09-04T17:20:00+00:00', '2026-09-04T17:20:05+00:00', '2026-W36', 8.0),    # 7
    ('GOOG', 'BUY',  200.0, 1, 200.0, 'REJECTED', '2026-09-04T18:00:00+00:00', '2026-09-04T18:05:00+00:00', '2026-W36', None),   # 8
    ('META', 'BUY',  500.0, 1, 500.0, 'EXECUTED', '2026-09-08T14:35:00+00:00', '2026-09-08T14:35:05+00:00', '2026-W37', None),   # 9
    ('META', 'SELL', 495.0, 1, 495.0, 'EXECUTED', '2026-09-08T19:55:00+00:00', '2026-09-08T19:55:05+00:00', '2026-W37', -5.0),   # 10
    ('AMZN', 'BUY',  180.0, 2, 360.0, 'EXECUTED', '2026-09-09T14:35:00+00:00', '2026-09-09T14:35:05+00:00', '2026-W37', None),   # 11
    ('AMZN', 'SELL', 183.0, 2, 366.0, 'EXECUTED', '2026-09-09T16:40:00+00:00', '2026-09-09T16:40:05+00:00', '2026-W37', 6.0),    # 12
    ('TSLA', 'BUY',  250.0, 1, 250.0, 'EXECUTED', '2026-09-09T17:00:00+00:00', '2026-09-09T17:00:05+00:00', '2026-W37', None),   # 13
    ('TSLA', 'SELL', 245.0, 1, 245.0, 'EXECUTED', '2026-09-10T14:45:00+00:00', '2026-09-10T14:45:05+00:00', '2026-W37', -5.0),   # 14
    ('NFLX', 'BUY',  700.0, 1, 700.0, 'REJECTED', '2026-09-10T15:00:00+00:00', '2026-09-10T15:05:00+00:00', '2026-W37', None),   # 15
    ('AAPL', 'BUY',  152.0, 2, 304.0, 'EXECUTED', '2026-09-10T15:30:00+00:00', '2026-09-10T15:30:05+00:00', '2026-W37', None),   # 16
    ('AAPL', 'SELL', 156.0, 2, 312.0, 'EXECUTED', '2026-09-10T19:50:00+00:00', '2026-09-10T19:50:05+00:00', '2026-W37', 8.0),    # 17
    ('MSFT', 'BUY',  405.0, 1, 405.0, 'EXECUTED', '2026-09-11T14:35:00+00:00', '2026-09-11T14:35:05+00:00', '2026-W37', None),   # 18
    ('MSFT', 'SELL', 407.0, 1, 407.0, 'EXECUTED', '2026-09-11T16:10:00+00:00', '2026-09-11T16:10:05+00:00', '2026-W37', 2.0),    # 19
    ('NVDA', 'BUY',  121.0, 3, 363.0, 'EXECUTED', '2026-09-11T16:30:00+00:00', '2026-09-11T16:30:05+00:00', '2026-W37', None),   # 20
    ('NVDA', 'SELL', 119.0, 3, 357.0, 'EXECUTED', '2026-09-11T19:55:00+00:00', '2026-09-11T19:55:05+00:00', '2026-W37', -6.0),   # 21
]

# Derived from LEGACY_TRADES; the migration tests assert these survive the upgrade.
LEGACY_TRADE_COUNT = len(LEGACY_TRADES)                                   # 21
LEGACY_EXECUTED_COUNT = sum(1 for t in LEGACY_TRADES if t[5] == 'EXECUTED')  # 18
LEGACY_REALIZED_TOTAL = sum(t[9] for t in LEGACY_TRADES if t[9] is not None)  # 12.0

# Two flat position rows - the live positions table is flat at account open.
LEGACY_POSITIONS = [
    # (symbol, shares, avg_price, updated_at)
    ('AAPL', 0, 152.0, '2026-09-10T19:50:05+00:00'),
    ('MSFT', 0, 0.0,   '2026-09-11T16:10:05+00:00'),
]


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """A fresh, fully migrated file DB with a $500 cash account opened at OPENED_AT.

    STARTING_CASH / ACCOUNT_TYPE are cleared so the seed uses the code defaults
    regardless of the shell environment on the container. A test that needs a
    margin account or a different balance sets the env and constructs its own
    Database(path, now=...) on a different path.
    """
    monkeypatch.delenv('STARTING_CASH', raising=False)
    monkeypatch.delenv('ACCOUNT_TYPE', raising=False)
    path = str(tmp_path / 'test.db')
    Database(path, now=OPENED_AT).close()
    return path


@pytest.fixture
def old_db_path(tmp_path):
    """A file DB written with the PRE-change schema and 21 legacy trades.

    Database() is deliberately NOT constructed here: the test under migration
    constructs it (with its own now=) so it can observe the upgrade.
    """
    path = str(tmp_path / 'old.db')
    conn = sqlite3.connect(path)
    try:
        conn.executescript(OLD_SCHEMA)
        conn.executemany(
            "INSERT INTO trades (symbol, side, price, shares, amount, status, "
            "created_at, settled_at, week_key, realized_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            LEGACY_TRADES,
        )
        conn.executemany(
            "INSERT INTO positions (symbol, shares, avg_price, updated_at) VALUES (?, ?, ?, ?)",
            LEGACY_POSITIONS,
        )
        conn.commit()
    finally:
        conn.close()
    return path
```

- [ ] **Step 2: Write the failing schema tests (columns, tables, REAL shares, legacy migrations)**

Create `/home/gdhughey/hugheylab-trading-bot/tests/test_migration.py`:

```python
"""
Migration tests: a file DB written with the pre-change schema (tests/conftest.py
OLD_SCHEMA) upgrades in place when Database() opens it, and a fresh DB gets the
same final shape from SCHEMA alone.

Every Database() call here passes now=NOW; nothing depends on wall time.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

from src.database import Database, connect
from tests.conftest import (LEGACY_EXECUTED_COUNT, LEGACY_REALIZED_TOTAL,
                            LEGACY_TRADE_COUNT)

# Fixed clock passed to every Database() here; nothing depends on wall time.
NOW = datetime(2026, 9, 14, 13, 30, 0, tzinfo=timezone.utc)
NOW_ISO = '2026-09-14T13:30:00+00:00'

NEW_TABLES = {'account', 'day_state', 'equity_history', 'signals'}

# column -> declared type, as PRAGMA table_info reports it
NEW_TRADE_COLS = {
    'ref_price': 'REAL',
    'fees': 'REAL',
    'gross_pnl': 'REAL',
    'available_at': 'TEXT',
    'trade_date': 'TEXT',
    'entry_probability': 'REAL',
    'exit_reason': 'TEXT',
}


def _cols(conn, table) -> dict:
    """{column name: declared type} for one table."""
    return {r['name']: r['type'] for r in conn.execute(f"PRAGMA table_info({table})")}


def _names(conn, kind) -> set:
    """Names of every table or index in sqlite_master."""
    return {r['name'] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = ?", (kind,))}


def test_old_db_gains_new_columns_and_tables(old_db_path):
    Database(old_db_path, now=NOW).close()
    conn = connect(old_db_path)
    try:
        trades = _cols(conn, 'trades')
        for col, typ in NEW_TRADE_COLS.items():
            assert trades.get(col) == typ, f"trades.{col} missing or wrong type"
        assert _cols(conn, 'positions').get('entry_ref') == 'REAL'
        assert NEW_TABLES <= _names(conn, 'table')
        assert 'idx_signals_label' in _names(conn, 'index')
        # All 21 legacy rows survive untouched and pick up the fees default.
        assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == LEGACY_TRADE_COUNT
        assert conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status = 'EXECUTED'").fetchone()[0] == LEGACY_EXECUTED_COUNT
        assert conn.execute(
            "SELECT realized_pnl FROM trades WHERE id = 2").fetchone()[0] == 10.0
        assert conn.execute(
            "SELECT SUM(realized_pnl) FROM trades").fetchone()[0] == LEGACY_REALIZED_TOTAL
        assert conn.execute(
            "SELECT COUNT(*) FROM trades WHERE fees = 0").fetchone()[0] == LEGACY_TRADE_COUNT
        assert conn.execute(
            "SELECT COUNT(*) FROM trades WHERE ref_price IS NULL AND available_at IS NULL "
            "AND trade_date IS NULL").fetchone()[0] == LEGACY_TRADE_COUNT
        assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 2
    finally:
        conn.close()


def test_fresh_db_declares_real_shares_and_full_shape(db_path):
    conn = connect(db_path)
    try:
        assert _cols(conn, 'trades')['shares'] == 'REAL'
        assert _cols(conn, 'positions')['shares'] == 'REAL'
        assert set(NEW_TRADE_COLS) <= set(_cols(conn, 'trades'))
        assert 'entry_ref' in _cols(conn, 'positions')
        assert NEW_TABLES <= _names(conn, 'table')
        assert 'idx_signals_label' in _names(conn, 'index')
    finally:
        conn.close()


def test_migrated_integer_shares_column_stores_fraction(old_db_path):
    Database(old_db_path, now=NOW).close()
    conn = connect(old_db_path)
    try:
        # Not rebuilt: still INTEGER-declared. SQLite affinity keeps 0.5 as REAL.
        assert _cols(conn, 'trades')['shares'] == 'INTEGER'
        assert _cols(conn, 'positions')['shares'] == 'INTEGER'
        with conn:
            conn.execute(
                "INSERT INTO trades (symbol, side, price, shares, amount, status, created_at, week_key) "
                "VALUES ('TSLA', 'BUY', 100.0, 0.5, 50.0, 'EXECUTED', ?, '2026-W38')",
                (NOW_ISO,))
            conn.execute(
                "INSERT INTO positions (symbol, shares, avg_price, updated_at) "
                "VALUES ('TSLA', 0.5, 100.0, ?)", (NOW_ISO,))
        t = conn.execute("SELECT shares FROM trades WHERE symbol = 'TSLA'").fetchone()['shares']
        p = conn.execute("SELECT shares FROM positions WHERE symbol = 'TSLA'").fetchone()['shares']
        assert t == 0.5 and isinstance(t, float)
        assert p == 0.5 and isinstance(p, float)
    finally:
        conn.close()


def test_realized_pnl_and_source_migrations_still_run(tmp_path):
    # An even older DB: trades without realized_pnl, prices without source.
    path = str(tmp_path / 'older.db')
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE prices (symbol TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL,
            low REAL, close REAL, volume REAL, PRIMARY KEY (symbol, date));
        CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
            side TEXT NOT NULL, price REAL NOT NULL, shares INTEGER NOT NULL, amount REAL NOT NULL,
            status TEXT NOT NULL, created_at TEXT NOT NULL, settled_at TEXT, week_key TEXT NOT NULL);
        CREATE TABLE positions (symbol TEXT PRIMARY KEY, shares INTEGER NOT NULL DEFAULT 0,
            avg_price REAL NOT NULL DEFAULT 0, updated_at TEXT);
    """)
    conn.close()
    Database(path, now=NOW).close()
    conn = connect(path)
    try:
        assert _cols(conn, 'trades').get('realized_pnl') == 'REAL'
        assert _cols(conn, 'prices').get('source') == 'TEXT'
        assert set(NEW_TRADE_COLS) <= set(_cols(conn, 'trades'))
    finally:
        conn.close()
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_migration.py -v`

Expected: `3 failed, 1 error`. `test_old_db_gains_new_columns_and_tables`, `test_migrated_integer_shares_column_stores_fraction` and `test_realized_pnl_and_source_migrations_still_run` each fail with `TypeError: Database.__init__() got an unexpected keyword argument 'now'`; `test_fresh_db_declares_real_shares_and_full_shape` is reported as ERROR because the same `TypeError` is raised inside the `db_path` fixture before the test body runs.

- [ ] **Step 4: Implement the schema, the column-migration table, and the `now` plumbing**

Edit `/home/gdhughey/hugheylab-trading-bot/src/database.py`. Replace the import block (lines 6-9) with:

```python
import os
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
```

Replace the whole `SCHEMA = """ ... """` block (lines 15-51) with:

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    source      TEXT,                       -- which provider supplied this bar
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL,          -- BUY | SELL
    price       REAL    NOT NULL,          -- simulated fill (ref +/- slippage or spread)
    shares      REAL    NOT NULL,          -- fractional, rounded to 6 dp
    amount      REAL    NOT NULL,          -- net cash movement: gross + fees (BUY) | gross - fees (SELL)
    status      TEXT    NOT NULL,          -- PENDING | EXECUTED | REJECTED
    created_at  TEXT    NOT NULL,
    settled_at  TEXT,                      -- status-decided timestamp (NOT cash settlement)
    week_key    TEXT    NOT NULL,          -- legacy ISO year-week; still stamped, no longer read
    realized_pnl REAL,                     -- SELL rows: net of fees; NULL for BUY
    ref_price   REAL,                      -- the quote the caller passed, before costs
    fees        REAL    NOT NULL DEFAULT 0,
    gross_pnl   REAL,                      -- SELL rows: realized_pnl + fees
    available_at TEXT,                     -- SELL rows: UTC ISO when the proceeds settle
    trade_date  TEXT,                      -- ET calendar date of created_at
    entry_probability REAL,                -- BUY rows: model probability at entry
    exit_reason TEXT                       -- SELL rows: tp | sl | timeout | eod | manual
);

CREATE INDEX IF NOT EXISTS idx_trades_week   ON trades (week_key, status);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);

CREATE TABLE IF NOT EXISTS positions (
    symbol      TEXT PRIMARY KEY,
    shares      REAL    NOT NULL DEFAULT 0,
    avg_price   REAL    NOT NULL DEFAULT 0, -- net cost basis per share (from amount)
    entry_ref   REAL    NOT NULL DEFAULT 0, -- qty-weighted ref_price of the open lots; barriers test against this
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS account (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    opened_at     TEXT NOT NULL,           -- UTC ISO; every report filters created_at >= opened_at
    starting_cash REAL NOT NULL,
    cash          REAL NOT NULL,
    account_type  TEXT NOT NULL            -- cash | margin
);

CREATE TABLE IF NOT EXISTS day_state (
    date              TEXT PRIMARY KEY,    -- ET date
    start_equity      REAL NOT NULL,
    loss_tripped_at   TEXT,
    loss_announced_at TEXT,
    report_posted_at  TEXT
);

CREATE TABLE IF NOT EXISTS equity_history (
    date             TEXT PRIMARY KEY,     -- ET date; written by the 16:05 ET report tick only
    cash             REAL NOT NULL,
    positions_value  REAL NOT NULL,
    equity           REAL NOT NULL,
    fees_to_date     REAL NOT NULL,
    realized_to_date REAL NOT NULL,
    recorded_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_ts            TEXT NOT NULL,       -- UTC ISO of the 5m bar scored
    symbol            TEXT NOT NULL,
    asset_class       TEXT NOT NULL,
    probability       REAL NOT NULL,
    bar               REAL NOT NULL,
    above_bar         INTEGER NOT NULL,
    ref_price         REAL NOT NULL,       -- bar close
    trade_date        TEXT NOT NULL,       -- ET date
    executed_trade_id INTEGER,             -- BUY trade id when this signal was traded
    label             INTEGER,             -- triple-barrier outcome; NULL until labelled
    labeled_at        TEXT,
    UNIQUE (symbol, bar_ts)
);

CREATE INDEX IF NOT EXISTS idx_signals_label ON signals (label, symbol);
"""

# (table, column, declaration) for every column added after its table first
# shipped. Each entry is applied only when PRAGMA table_info lacks the column,
# so the list is safe to run on every start. A NOT NULL addition needs a
# DEFAULT or SQLite refuses the ALTER on a populated table.
COLUMN_MIGRATIONS = [
    ('trades', 'realized_pnl', 'REAL'),
    ('prices', 'source', 'TEXT'),
    ('trades', 'ref_price', 'REAL'),
    ('trades', 'fees', 'REAL NOT NULL DEFAULT 0'),
    ('trades', 'gross_pnl', 'REAL'),
    ('trades', 'available_at', 'TEXT'),
    ('trades', 'trade_date', 'TEXT'),
    ('trades', 'entry_probability', 'REAL'),
    ('trades', 'exit_reason', 'TEXT'),
    ('positions', 'entry_ref', 'REAL NOT NULL DEFAULT 0'),
]
```

Replace the whole `class Database:` (lines 66-98, everything from `class Database:` through the end of `close()`) with:

```python
class Database:
    """Owns schema creation. Other components open their own connections."""

    def __init__(self, db_path: str = None, now: datetime = None):
        self.db_path = db_path or DB_PATH
        self.conn = connect(self.db_path)
        self.init_schema(now=now)
        logger.info(f"Database ready at {self.db_path}")

    def init_schema(self, now: datetime = None):
        with self.conn:
            self.conn.executescript(SCHEMA)
        self._migrate(now=now)

    def _migrate(self, now: datetime = None):
        """Additive migrations for databases created by an earlier version.

        `now` (tz-aware UTC) is only consulted the first time the account row
        is seeded; tests pass it so opened_at is deterministic.
        """
        for table, column, decl in COLUMN_MIGRATIONS:
            cols = {r['name'] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                with self.conn:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                logger.info(f"Migrated: added {table}.{column}")

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
```

- [ ] **Step 5: Run the schema tests and confirm they pass**

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_migration.py -v`

Expected: `4 passed`.

- [ ] **Step 6: Write the failing account-seed tests**

Append to `/home/gdhughey/hugheylab-trading-bot/tests/test_migration.py`:

```python
def test_account_seeded_once_from_env(old_db_path, monkeypatch):
    monkeypatch.setenv('STARTING_CASH', '500')
    monkeypatch.delenv('ACCOUNT_TYPE', raising=False)   # default must be 'cash'
    Database(old_db_path, now=NOW).close()
    conn = connect(old_db_path)
    try:
        rows = conn.execute("SELECT * FROM account").fetchall()
        assert len(rows) == 1
        acct = rows[0]
        assert acct['id'] == 1
        assert acct['opened_at'] == NOW_ISO
        assert acct['starting_cash'] == 500.0
        assert acct['cash'] == 500.0
        assert acct['account_type'] == 'cash'
        # Every legacy trade predates opened_at, so the ledger's
        # created_at >= opened_at filters (Task 4) will exclude all 21.
        assert conn.execute(
            "SELECT COUNT(*) FROM trades WHERE created_at >= ?", (acct['opened_at'],)
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_account_type_margin_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv('STARTING_CASH', '1000')
    monkeypatch.setenv('ACCOUNT_TYPE', 'margin')
    path = str(tmp_path / 'margin.db')
    Database(path, now=NOW).close()
    conn = connect(path)
    try:
        acct = conn.execute("SELECT * FROM account").fetchone()
        assert acct['starting_cash'] == 1000.0
        assert acct['account_type'] == 'margin'
    finally:
        conn.close()


def test_second_construction_keeps_one_account_row(old_db_path, monkeypatch):
    monkeypatch.setenv('STARTING_CASH', '500')
    monkeypatch.delenv('ACCOUNT_TYPE', raising=False)
    Database(old_db_path, now=NOW).close()
    # A later env edit and a later clock must NOT reopen or restate the account:
    # changing starting cash after open would corrupt the all-time return.
    monkeypatch.setenv('STARTING_CASH', '999')
    Database(old_db_path, now=NOW + timedelta(days=1)).close()
    conn = connect(old_db_path)
    try:
        rows = conn.execute("SELECT * FROM account").fetchall()
        assert len(rows) == 1
        assert rows[0]['starting_cash'] == 500.0
        assert rows[0]['cash'] == 500.0
        assert rows[0]['opened_at'] == NOW_ISO
    finally:
        conn.close()
```

- [ ] **Step 7: Run the tests and confirm the account tests fail**

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_migration.py -v`

Expected: `3 failed, 4 passed`. `test_account_seeded_once_from_env` and `test_second_construction_keeps_one_account_row` fail with `assert 0 == 1` (the `account` table exists but is empty); `test_account_type_margin_from_env` fails with `TypeError: 'NoneType' object is not subscriptable` (`fetchone()` returned `None`).

- [ ] **Step 8: Seed the account row in `_migrate`**

In `/home/gdhughey/hugheylab-trading-bot/src/database.py`, replace the whole `_migrate` method with:

```python
    def _migrate(self, now: datetime = None):
        """Additive migrations for databases created by an earlier version.

        `now` (tz-aware UTC) is only consulted the first time the account row
        is seeded; tests pass it so opened_at is deterministic.
        """
        for table, column, decl in COLUMN_MIGRATIONS:
            cols = {r['name'] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                with self.conn:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                logger.info(f"Migrated: added {table}.{column}")

        # Seed the single account row. INSERT OR IGNORE means a later change to
        # STARTING_CASH / ACCOUNT_TYPE never touches an opened account: there is
        # deliberately no setter, because restating starting cash would corrupt
        # the all-time return every report is built on.
        opened_at = (now or datetime.now(timezone.utc)).isoformat(timespec='seconds')
        starting_cash = float(os.getenv('STARTING_CASH', 500))
        account_type = os.getenv('ACCOUNT_TYPE', 'cash').strip().lower()
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO account (id, opened_at, starting_cash, cash, account_type) "
                "VALUES (1, ?, ?, ?, ?)",
                (opened_at, starting_cash, starting_cash, account_type),
            )
        if cur.rowcount:
            logger.info(
                f"Opened paper account: ${starting_cash:,.2f} ({account_type}) at {opened_at}")
```

- [ ] **Step 9: Run the tests and confirm the account tests pass**

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_migration.py -v`

Expected: `7 passed`.

- [ ] **Step 10: Write the failing `entry_ref` backfill test**

Append to `/home/gdhughey/hugheylab-trading-bot/tests/test_migration.py`:

```python
def test_entry_ref_backfilled_for_open_position(old_db_path):
    # An open lot written by the old code has avg_price but no entry_ref. The
    # ref-to-ref barrier test divides by entry_ref, so it must not stay 0.
    conn = sqlite3.connect(old_db_path)
    with conn:
        conn.execute(
            "INSERT INTO positions (symbol, shares, avg_price, updated_at) "
            "VALUES ('NVDA', 3, 120.0, '2026-09-11T19:55:00+00:00')")
    conn.close()

    Database(old_db_path, now=NOW).close()
    conn = connect(old_db_path)
    try:
        refs = {r['symbol']: r['entry_ref']
                for r in conn.execute("SELECT symbol, entry_ref FROM positions")}
        # Flat rows stay 0; the open lot takes its avg_price.
        assert refs == {'AAPL': 0.0, 'MSFT': 0.0, 'NVDA': 120.0}

        # A non-zero entry_ref is never overwritten by a later start.
        with conn:
            conn.execute("UPDATE positions SET entry_ref = 118.0 WHERE symbol = 'NVDA'")
    finally:
        conn.close()
    Database(old_db_path, now=NOW + timedelta(days=1)).close()
    conn = connect(old_db_path)
    try:
        assert conn.execute(
            "SELECT entry_ref FROM positions WHERE symbol = 'NVDA'").fetchone()[0] == 118.0
    finally:
        conn.close()
```

- [ ] **Step 11: Run the tests and confirm the backfill test fails**

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_migration.py -v`

Expected: `1 failed, 7 passed`. `test_entry_ref_backfilled_for_open_position` fails with `AssertionError: assert {'AAPL': 0.0, 'MSFT': 0.0, 'NVDA': 0.0} == {'AAPL': 0.0, 'MSFT': 0.0, 'NVDA': 120.0}`.

- [ ] **Step 12: Add the backfill to `_migrate`**

In `/home/gdhughey/hugheylab-trading-bot/src/database.py`, replace the whole `_migrate` method with its final form:

```python
    def _migrate(self, now: datetime = None):
        """Additive migrations for databases created by an earlier version.

        `now` (tz-aware UTC) is only consulted the first time the account row
        is seeded; tests pass it so opened_at is deterministic.
        """
        for table, column, decl in COLUMN_MIGRATIONS:
            cols = {r['name'] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                with self.conn:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                logger.info(f"Migrated: added {table}.{column}")

        # Lots opened before entry_ref existed have only avg_price. The barrier
        # test measures ref-to-ref against entry_ref (and divides by it), so
        # give those lots their cost basis as the reference instead of 0. Only
        # rows still at 0 are touched, so this is idempotent across restarts.
        with self.conn:
            cur = self.conn.execute(
                "UPDATE positions SET entry_ref = avg_price WHERE shares > 0 AND entry_ref = 0")
        if cur.rowcount:
            logger.info(f"Migrated: backfilled entry_ref on {cur.rowcount} open position(s)")

        # Seed the single account row. INSERT OR IGNORE means a later change to
        # STARTING_CASH / ACCOUNT_TYPE never touches an opened account: there is
        # deliberately no setter, because restating starting cash would corrupt
        # the all-time return every report is built on.
        opened_at = (now or datetime.now(timezone.utc)).isoformat(timespec='seconds')
        starting_cash = float(os.getenv('STARTING_CASH', 500))
        account_type = os.getenv('ACCOUNT_TYPE', 'cash').strip().lower()
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO account (id, opened_at, starting_cash, cash, account_type) "
                "VALUES (1, ?, ?, ?, ?)",
                (opened_at, starting_cash, starting_cash, account_type),
            )
        if cur.rowcount:
            logger.info(
                f"Opened paper account: ${starting_cash:,.2f} ({account_type}) at {opened_at}")
```

- [ ] **Step 13: Run the whole suite and confirm everything passes**

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests -v`

Expected: `9 passed` (8 in `tests/test_migration.py` plus `tests/test_smoke.py::test_smoke`; if Tasks 1-2 have already landed, their tests are included in the count and also pass).

- [ ] **Step 14: Commit**

Run:

```bash
cd /home/gdhughey/hugheylab-trading-bot && git add src/database.py tests/conftest.py tests/test_migration.py && git commit -m "Schema: REAL shares, account/day_state/equity_history/signals tables, additive migrations

SCHEMA declares trades.shares and positions.shares REAL and adds the four
paper-brokerage tables. _migrate() applies PRAGMA-guarded ADD COLUMNs from a
single COLUMN_MIGRATIONS list (the realized_pnl and prices.source migrations
move into it), backfills positions.entry_ref from avg_price for open lots,
and seeds the account row with INSERT OR IGNORE from STARTING_CASH /
ACCOUNT_TYPE. Database() takes now= so tests pin opened_at.

tests/conftest.py adds the db_path and old_db_path fixtures; the latter
writes the pre-change schema verbatim with 21 legacy trades." -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

**Contract additions**
- `Database.__init__(self, db_path: str = None, now: datetime = None)` — `now` is tz-aware UTC; used only for `account.opened_at` on the first seed. `main.py` and `train.py` keep calling `Database()` with no args.
- `Database.init_schema(self, now: datetime = None)` and `Database._migrate(self, now: datetime = None)` — same `now` threaded through.
- `src.database.COLUMN_MIGRATIONS: list[tuple[str, str, str]]` — module-level `(table, column, declaration)` list; the existing `realized_pnl` and `prices.source` migrations live in it.
- `tests/conftest.py` module constants: `OPENED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)`, `OLD_SCHEMA: str`, `LEGACY_TRADES: list[tuple]` (21 rows: 18 EXECUTED forming 9 round trips, 3 REJECTED; ids 1..21 in list order), `LEGACY_TRADE_COUNT = 21`, `LEGACY_EXECUTED_COUNT = 18`, `LEGACY_REALIZED_TOTAL = 12.0`, `LEGACY_POSITIONS: list[tuple]` (2 flat rows).
- Fixture `db_path(tmp_path, monkeypatch)` also clears `STARTING_CASH` and `ACCOUNT_TYPE` from the environment before constructing `Database(path, now=OPENED_AT)`, so the fixture DB is always a $500 cash account opened at `2026-01-01T00:00:00+00:00`.
- Test convention (applies to every task): any test that constructs `Database()` itself must pass `now=` (a tz-aware UTC datetime earlier than every timestamp the test writes) or pin `account.opened_at` with a direct `UPDATE account SET opened_at = ? WHERE id = 1` immediately after construction (as Task 4's `_open_account` does). Relying on the wall-clock seed is forbidden: every ledger read filters `created_at >= opened_at`, so a wall-clock `opened_at` makes the test's fixed timestamps fall before the account opened as soon as the suite runs later than its data.


---

### Task 4: Account ledger — rewrite `BudgetTracker` as the paper brokerage account

**Depends on:** Task 1 (`src/costs.py`), Task 2 (`next_trading_day_open`, `ET` in `src/intraday_engine.py`), Task 3 (`account`, `day_state`, `equity_history` tables and the new `trades`/`positions` columns).

**Files:**
- Modify: `/home/gdhughey/hugheylab-trading-bot/src/budget_tracker.py` (full rewrite; current file is lines 1-284 — the weekly-budget module docstring, `_week_key`/`_now`, and a `BudgetTracker` with `_sum/_deployed/_committed/get_weekly_spent/get_turnover/get_remaining_budget/can_trade/log_trade/execute_trade/reject_trade/_apply_position/get_positions/get_realized_pnl/get_pnl/set_weekly_budget/get_statistics`)
- Create: `/home/gdhughey/hugheylab-trading-bot/tests/test_budget_tracker.py`
- Modify (documentation only, Step 16): `/home/gdhughey/hugheylab-trading-bot/docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md` line 22 (the Task 7 dependency row); `/home/gdhughey/hugheylab-trading-bot/docs/superpowers/specs/2026-09-12-paper-brokerage-design.md` lines 93-94 (the locked-method list) and lines 343-344 (the sizing test item)

Design notes the engineer needs (why, not just what):
- The tracker's connection is switched to autocommit (`conn.isolation_level = None`) and every mutating method runs inside `_txn()`: one `threading.Lock` + `BEGIN IMMEDIATE`. The write lock is taken up front so the fast-cycle worker thread and the Discord event-loop thread can never interleave a read-then-write on cash.
- **`budget.conn` is owned by `BudgetTracker`.** In autocommit mode a transaction belongs to the connection, not to a thread, so a `with budget.conn:` block (whose `__exit__` calls `commit()`) or a bare `commit()` issued from ANOTHER thread would commit whatever `_txn()` has half-written on the fast-cycle thread, and the later rollback in `_txn` would have nothing to undo. Rule: other modules may READ through `budget.conn` from any thread, and may WRITE through it only on the same thread that runs `FastTrader.cycle` (Task 6's `signal_log.record` / `mark_executed` are sequential with `_txn` there). Anything that writes from a different thread — the 16:05 ET tick in Task 9 (`signal_log.label_pending`) runs in its own `to_thread` worker while the fast cycle may be mid-transaction — must open its own connection: `connect(budget.db_path)` (or reuse `intraday.conn`). `BudgetTracker.db_path` exists for exactly that.
- Time is never read from the wall clock in the hot path when the caller supplies `now=`; `_now(now)` formats any tz-aware datetime as UTC ISO seconds, and all timestamp columns share that format, so `available_at > ?` is a correct lexical comparison.
- The test file pins `account.opened_at` to `2026-09-14T13:30:00+00:00` with a direct SQL UPDATE right after `Database()` seeds the row, because the seed uses the wall clock and every ledger read filters `created_at >= opened_at`. It also pins every env var it depends on with `monkeypatch.setenv` BEFORE constructing `Database()` (the seed reads `STARTING_CASH`/`ACCOUNT_TYPE` then). It therefore does not use `conftest.py`'s `db_path` fixture.
- Legacy callers (`src/fast_trader.py` `get_remaining_budget`/`can_trade`, `src/discord_bot.py` `weekly_budget`/`get_statistics`) stop working at call time after this task; that is expected and is fixed by Tasks 6 and 9. Nothing imports the deleted names at module import time, so `tests/test_smoke.py` keeps passing.

- [ ] **Step 1: Write the failing ledger tests (account reads, trade lifecycle, settlement, P&L, positions)**

Create `/home/gdhughey/hugheylab-trading-bot/tests/test_budget_tracker.py`:

```python
"""Ledger tests for BudgetTracker (Task 4: paper brokerage account).

Every clock-dependent call passes now= explicitly, and the fixture pins the
account's opened_at, so the `created_at >= opened_at` filters behave the same
no matter when the suite runs.
"""
from datetime import datetime, timezone

import pytest

from src.database import Database
from src.budget_tracker import BudgetTracker, _et_date
from src.intraday_engine import ET

OPENED = '2026-09-14T13:30:00+00:00'          # Mon 2026-09-14 09:30 ET


def et(y, m, d, hh, mm):
    """A tz-aware UTC datetime for the given ET wall-clock time."""
    return datetime(y, m, d, hh, mm, tzinfo=ET).astimezone(timezone.utc)


FRI_1000 = et(2026, 9, 18, 10, 0)
FRI_1555 = et(2026, 9, 18, 15, 55)
FRI_1600 = et(2026, 9, 18, 16, 0)
MON_0929 = et(2026, 9, 21, 9, 29)
MON_0930 = et(2026, 9, 21, 9, 30)


def _pin_env(monkeypatch, account_type='cash'):
    # The account seed and costs.fill read these at call time; pin them so
    # the numbers below are exact regardless of the container's environment.
    monkeypatch.setenv('STARTING_CASH', '500')
    monkeypatch.setenv('ACCOUNT_TYPE', account_type)
    monkeypatch.setenv('FAST_MAX_POSITIONS', '4')
    monkeypatch.setenv('MIN_ORDER_USD', '1')
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '5')
    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '60')
    monkeypatch.setenv('SEC_FEE_RATE', '0.0000206')
    monkeypatch.setenv('FINRA_TAF_PER_SHARE', '0.000195')
    monkeypatch.setenv('FINRA_TAF_CAP', '9.79')


def _open_account(tmp_path, monkeypatch, account_type='cash'):
    _pin_env(monkeypatch, account_type)
    path = str(tmp_path / 'ledger.db')
    db = Database(path)                       # creates schema + seeds the account row
    with db.conn:
        # The seed stamps the wall clock; pin it so the test timestamps are
        # always after the account opened.
        db.conn.execute("UPDATE account SET opened_at = ? WHERE id = 1", (OPENED,))
    return BudgetTracker(path)


@pytest.fixture
def bt(tmp_path, monkeypatch):
    return _open_account(tmp_path, monkeypatch)


def _round_trip(bt, symbol, side, ref, qty, now, **kw):
    """log_trade + execute_trade at the same instant; returns the executed row."""
    tid = bt.log_trade(symbol, side, ref, qty, now=now, **kw)
    return bt.execute_trade(tid, now=now)


# --- account reads ---------------------------------------------------------

def test_account_reads_on_a_fresh_account(bt):
    assert bt.db_path.endswith('ledger.db')   # Task 9 opens its own connection from this
    assert bt.opened_at() == OPENED
    assert bt.starting_cash() == 500.0
    assert bt.account_type() == 'cash'
    assert bt.get_cash() == 500.0
    assert bt.get_unsettled(now=FRI_1000) == 0.0
    assert bt.get_buying_power(now=FRI_1000) == 500.0
    assert bt.get_equity({}) == 500.0
    assert bt.get_positions() == []
    assert bt.get_fees_paid() == 0.0
    assert bt.get_realized_pnl() == 0.0
    assert bt.get_gross_pnl() == 0.0


def test_et_date_is_the_et_calendar_day():
    assert _et_date('2026-09-18T23:30:00+00:00') == '2026-09-18'   # 19:30 ET
    assert _et_date('2026-09-19T03:30:00+00:00') == '2026-09-18'   # 23:30 ET, still Friday


# --- trade lifecycle -------------------------------------------------------

def test_log_trade_records_fill_and_pending_hold(bt):
    tid = bt.log_trade('AAPL', 'BUY', 100.0, 1.2493751, probability=0.61, now=FRI_1000)
    row = bt.conn.execute("SELECT * FROM trades WHERE id = ?", (tid,)).fetchone()
    assert row['status'] == 'PENDING'
    assert row['shares'] == 1.249375                      # rounded to 6 dp, stored REAL
    assert row['ref_price'] == 100.0
    assert row['price'] == pytest.approx(100.05)          # 5 bps slippage on BUY
    assert row['fees'] == 0.0
    assert row['amount'] == pytest.approx(1.249375 * 100.05)
    assert row['created_at'] == FRI_1000.isoformat(timespec='seconds')
    assert row['trade_date'] == '2026-09-18'
    assert row['entry_probability'] == 0.61
    assert row['exit_reason'] is None
    assert row['available_at'] is None
    # a PENDING BUY holds buying power but has not moved cash
    assert bt.get_cash() == 500.0
    assert bt.get_buying_power(now=FRI_1000) == pytest.approx(500.0 - row['amount'])
    assert bt.reject_trade(tid, now=FRI_1000) is True
    assert bt.reject_trade(tid, now=FRI_1000) is False   # already decided
    assert bt.get_buying_power(now=FRI_1000) == 500.0
    assert bt.execute_trade(tid, now=FRI_1000) is None   # not PENDING any more


def test_stock_sell_is_unsettled_until_next_trading_day_open(bt):
    buy = _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    assert buy['status'] == 'EXECUTED'
    assert bt.get_cash() == pytest.approx(500.0 - buy['amount'])
    sell = _round_trip(bt, 'AAPL', 'SELL', 101.0, 1.0, FRI_1555, exit_reason='eod')
    assert sell['exit_reason'] == 'eod'
    assert sell['price'] == pytest.approx(101.0 * (1 - 0.0005))
    assert sell['available_at'] == '2026-09-21T13:30:00+00:00'   # Mon 09:30 ET in UTC
    cash = bt.get_cash()
    assert cash == pytest.approx(500.0 - buy['amount'] + sell['amount'])
    # Friday after the fill, after the bell, and Monday pre-open: proceeds are held
    for now in (FRI_1555, FRI_1600, MON_0929):
        assert bt.get_unsettled(now=now) == pytest.approx(sell['amount'])
        assert bt.get_buying_power(now=now) == pytest.approx(cash - sell['amount'])
    # Monday 09:30 ET: settled
    assert bt.get_unsettled(now=MON_0930) == 0.0
    assert bt.get_buying_power(now=MON_0930) == pytest.approx(cash)
    assert bt.get_positions() == []


def test_settlement_skips_thanksgiving(bt):
    _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, et(2026, 11, 25, 10, 0))
    sell = _round_trip(bt, 'AAPL', 'SELL', 100.0, 1.0, et(2026, 11, 25, 15, 55), exit_reason='eod')
    # 2026-11-26 is a holiday; 09:30 ET on the 27th is 14:30 UTC (EST)
    assert sell['available_at'] == et(2026, 11, 27, 9, 30).isoformat(timespec='seconds')


def test_crypto_sell_settles_immediately(bt):
    _round_trip(bt, 'BTC-USD', 'BUY', 50_000.0, 0.002, FRI_1555)
    sell = _round_trip(bt, 'BTC-USD', 'SELL', 50_000.0, 0.002, FRI_1600, exit_reason='tp')
    assert sell['fees'] == 0.0
    assert sell['price'] == pytest.approx(50_000.0 * (1 - 0.006))
    assert sell['available_at'] == sell['created_at']
    assert bt.get_unsettled(now=FRI_1600) == 0.0
    assert bt.get_buying_power(now=FRI_1600) == pytest.approx(bt.get_cash())


def test_margin_account_settles_immediately(tmp_path, monkeypatch):
    bt = _open_account(tmp_path, monkeypatch, account_type='margin')
    assert bt.account_type() == 'margin'
    _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    sell = _round_trip(bt, 'AAPL', 'SELL', 101.0, 1.0, FRI_1555, exit_reason='eod')
    assert sell['available_at'] == sell['created_at']
    assert bt.get_unsettled(now=FRI_1555) == 0.0
    assert bt.get_buying_power(now=FRI_1555) == pytest.approx(bt.get_cash())


def test_trades_before_opened_at_are_ignored(bt):
    # A legacy EXECUTED SELL from before the account opened, with an
    # unsettled available_at far in the future. BudgetTracker's connection is
    # in autocommit mode, so a bare execute() is durable.
    bt.conn.execute(
        "INSERT INTO trades (symbol, side, price, shares, amount, status, created_at, settled_at, "
        "week_key, realized_pnl, fees, gross_pnl, available_at, trade_date) "
        "VALUES ('AAPL', 'SELL', 100, 5, 500, 'EXECUTED', '2026-09-10T15:00:00+00:00', "
        "'2026-09-10T15:00:00+00:00', '2026-W37', 999, 1000, 1, '2099-01-01T00:00:00+00:00', "
        "'2026-09-10')")
    assert bt.get_realized_pnl() == 0.0
    assert bt.get_gross_pnl() == 0.0
    assert bt.get_fees_paid() == 0.0
    assert bt.get_unsettled(now=FRI_1000) == 0.0
    assert bt.get_buying_power(now=FRI_1000) == 500.0
    assert bt.get_trades_since_open() == []
    _round_trip(bt, 'MSFT', 'BUY', 200.0, 0.5, FRI_1000)
    assert [r['symbol'] for r in bt.get_trades_since_open()] == ['MSFT']
    assert [r['symbol'] for r in bt.get_trades_since_open('2026-09-18')] == ['MSFT']
    assert bt.get_trades_since_open('2026-09-17') == []


# --- P&L and positions -----------------------------------------------------

def test_realized_pnl_is_net_and_gross_adds_fees_back(bt):
    buy = _round_trip(bt, 'AAPL', 'BUY', 100.0, 2.0, FRI_1000)
    assert buy['fees'] == 0.0
    assert buy['realized_pnl'] is None and buy['gross_pnl'] is None
    sell = _round_trip(bt, 'AAPL', 'SELL', 110.0, 2.0, FRI_1555, exit_reason='tp')
    gross_proceeds = 2.0 * 110.0 * (1 - 0.0005)
    assert sell['fees'] == pytest.approx(0.0000206 * gross_proceeds + 0.000195 * 2.0)
    assert sell['fees'] > 0
    assert sell['amount'] == pytest.approx(gross_proceeds - sell['fees'])
    assert sell['realized_pnl'] == pytest.approx(sell['amount'] - buy['amount'])
    assert sell['gross_pnl'] == pytest.approx(sell['realized_pnl'] + sell['fees'])
    assert sell['gross_pnl'] == pytest.approx(gross_proceeds - buy['amount'])
    assert sell['realized_pnl'] < sell['gross_pnl']
    assert bt.get_realized_pnl() == pytest.approx(sell['realized_pnl'])
    assert bt.get_gross_pnl() == pytest.approx(sell['gross_pnl'])
    assert bt.get_fees_paid() == pytest.approx(sell['fees'])


def test_entry_ref_is_quantity_weighted_and_survives_partial_sell(bt):
    _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    _round_trip(bt, 'AAPL', 'BUY', 104.0, 3.0, FRI_1000)
    (pos,) = bt.get_positions()
    assert set(pos) == {'symbol', 'shares', 'avg_price', 'entry_ref', 'cost_basis', 'updated_at'}
    assert pos['shares'] == 4.0 and isinstance(pos['shares'], float)
    assert pos['entry_ref'] == pytest.approx((1 * 100.0 + 3 * 104.0) / 4)      # 103.0, ref-weighted
    assert pos['avg_price'] == pytest.approx((100.05 + 3 * 104.052) / 4)      # net cost basis
    assert pos['cost_basis'] == pytest.approx(100.05 + 3 * 104.052)
    _round_trip(bt, 'AAPL', 'SELL', 110.0, 2.0, FRI_1555, exit_reason='tp')
    (pos,) = bt.get_positions()
    assert pos['shares'] == 2.0
    assert pos['entry_ref'] == pytest.approx(103.0)                            # unchanged by a partial SELL
    assert pos['avg_price'] == pytest.approx((100.05 + 3 * 104.052) / 4)
    _round_trip(bt, 'AAPL', 'SELL', 110.0, 2.0, FRI_1555, exit_reason='eod')
    assert bt.get_positions() == []
    raw = bt.conn.execute(
        "SELECT shares, avg_price, entry_ref FROM positions WHERE symbol = 'AAPL'").fetchone()
    assert (raw['shares'], raw['avg_price'], raw['entry_ref']) == (0.0, 0.0, 0.0)


def test_dust_is_written_as_zero_and_hidden(bt):
    _round_trip(bt, 'DOGE-USD', 'BUY', 0.1, 0.1, FRI_1000)
    _round_trip(bt, 'DOGE-USD', 'BUY', 0.1, 0.2, FRI_1000)
    # 0.1 + 0.2 - 0.3 is 5.5e-17 in binary float; the ledger must not keep it
    _round_trip(bt, 'DOGE-USD', 'SELL', 0.1, 0.3, FRI_1555, exit_reason='sl')
    raw = bt.conn.execute(
        "SELECT shares, avg_price, entry_ref FROM positions WHERE symbol = 'DOGE-USD'").fetchone()
    assert raw['shares'] == 0.0 and raw['avg_price'] == 0.0 and raw['entry_ref'] == 0.0
    assert bt.get_positions() == []
```

- [ ] **Step 2: Run the tests to confirm they fail on the missing new API**

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_budget_tracker.py -v`

Expected: collection error, nothing runs:
```
ERROR tests/test_budget_tracker.py - ImportError: cannot import name '_et_date' from 'src.budget_tracker' (/opt/trading-bot-dev/src/budget_tracker.py)
...
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

- [ ] **Step 3: Rewrite `src/budget_tracker.py` as the account ledger (everything except sizing, `get_pnl`, day state and equity history, which come in later steps)**

Replace the ENTIRE contents of `/home/gdhughey/hugheylab-trading-bot/src/budget_tracker.py` with:

```python
#!/usr/bin/env python3
"""
Budget tracker - the paper brokerage ledger: one cash account, the positions
it holds, and the trade lifecycle (PENDING -> EXECUTED | REJECTED).

Account semantics:
  * One `account` row (seeded by Database._migrate) holds `cash`. A BUY debits
    `amount` (fill x qty + fees); a SELL credits `amount` (fill x qty - fees).
    Cash never resets, so wins compound and losses shrink the next order.
  * Equity = cash + market value of open positions. A symbol without a quote
    is carried at its avg_price (and reported as stale by get_pnl).
  * Buying power = cash - unsettled SELL proceeds - `amount` of PENDING BUYs,
    floored at 0. In a cash account a stock SELL settles at the next trading
    day's 09:30 ET (T+1, weekends and US_HOLIDAYS_2026 skipped); crypto
    settles at once; a margin account never waits. Settlement is stored on
    the SELL row (`trades.available_at`), never recomputed.
  * Every sum, count and report filters `created_at >= account.opened_at`, so
    trades from before the account opened stay in the DB but never count.
  * Costs are applied in exactly one place: log_trade calls costs.fill and
    stores ref_price (the caller's quote), price (the fill), fees and amount.
  * Every mutating method takes one threading.Lock and runs inside a single
    BEGIN IMMEDIATE transaction, so a buying-power check and the cash debit
    are atomic across the fast-cycle worker thread and the event-loop thread.
  * `self.conn` is OWNED by this class. In autocommit mode a transaction
    belongs to the connection, not to a thread, so a `with budget.conn:` or
    `commit()` issued from another thread would commit whatever _txn() has
    half-written. Other modules may read through it from any thread, and may
    write through it only on the thread that runs FastTrader.cycle (which is
    sequential with _txn). Anything else opens its own connection with
    `connect(budget.db_path)`.
"""

import os
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from src import costs
from src.database import DB_PATH, connect
from src.intraday_engine import ET, is_crypto, next_trading_day_open

logger = logging.getLogger(__name__)

# The only day_state columns set_day_flag may stamp (the column name is
# interpolated into SQL, so it must be whitelisted).
DAY_FLAGS = ('loss_tripped_at', 'loss_announced_at', 'report_posted_at')


def _week_key(dt: datetime = None) -> str:
    """ISO year-week, e.g. '2026-W36'. Legacy NOT NULL column: still stamped, never read."""
    dt = dt or datetime.now()
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _now(now: datetime | None = None) -> str:
    """UTC ISO seconds. A tz-aware `now` overrides the wall clock so tests are deterministic.

    Every timestamp column uses this one format, which is what makes
    `available_at > ?` a correct lexical comparison in SQL.
    """
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).isoformat(timespec='seconds')


def _et_date(ts_iso: str) -> str:
    """ET calendar date of a UTC ISO string - the trading day a timestamp belongs to."""
    return datetime.fromisoformat(ts_iso).astimezone(ET).strftime('%Y-%m-%d')


class BudgetTracker:
    def __init__(self, db_path: str = None):
        # Kept so other threads can open their OWN connection to the same file
        # (see the module docstring: they must never commit on self.conn).
        self.db_path = db_path or DB_PATH
        self.conn = connect(self.db_path)
        # Autocommit mode: transactions are opened explicitly with BEGIN
        # IMMEDIATE in _txn() so the write lock is taken up front rather than
        # on the first UPDATE, where a deferred transaction can hit SQLITE_BUSY
        # halfway through a trade.
        self.conn.isolation_level = None
        self._lock = threading.Lock()
        logger.info(f"BudgetTracker ready ({self.account_type()} account opened {self.opened_at()}, "
                    f"cash ${self.get_cash():,.2f} of ${self.starting_cash():,.2f} starting)")

    # --- internals -------------------------------------------------------

    @contextmanager
    def _txn(self):
        """One locked BEGIN IMMEDIATE transaction: commit on success, rollback on error.

        The lock serialises this class's own writers. It cannot protect
        against another module committing self.conn from a different thread
        while this block is open - that is why such modules open their own
        connection (connect(self.db_path)) instead.
        """
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.conn.rollback()
                raise
            else:
                self.conn.commit()

    def _account(self) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM account WHERE id = 1").fetchone()
        if row is None:
            raise RuntimeError("account row missing - Database() must run before BudgetTracker")
        return row

    def _sum_since_open(self, column: str) -> float:
        """SUM(column) over EXECUTED trades since the account opened. `column` is a literal."""
        row = self.conn.execute(
            f"SELECT COALESCE(SUM({column}), 0) AS v FROM trades "
            "WHERE status = 'EXECUTED' AND created_at >= ?", (self.opened_at(),)
        ).fetchone()
        return float(row['v'])

    def _positions_value(self, prices: dict) -> float:
        """Market value of open positions; a symbol without a quote is carried at avg_price."""
        total = 0.0
        for pos in self.get_positions():
            price = prices.get(pos['symbol'])
            total += pos['shares'] * (pos['avg_price'] if price is None else float(price))
        return total

    # --- account ---------------------------------------------------------

    def opened_at(self) -> str:
        return self._account()['opened_at']

    def starting_cash(self) -> float:
        return float(self._account()['starting_cash'])

    def account_type(self) -> str:
        return self._account()['account_type']

    def get_cash(self) -> float:
        return float(self._account()['cash'])

    def get_unsettled(self, now: datetime | None = None) -> float:
        """SELL proceeds that cannot be spent yet (cash-account T+1 rule)."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS v FROM trades "
            "WHERE side = 'SELL' AND status = 'EXECUTED' AND available_at > ? AND created_at >= ?",
            (_now(now), self.opened_at()),
        ).fetchone()
        return float(row['v'])

    def get_buying_power(self, now: datetime | None = None) -> float:
        """Cash minus unsettled proceeds minus PENDING BUY holds; never negative."""
        pending = self.conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS v FROM trades "
            "WHERE side = 'BUY' AND status = 'PENDING'"
        ).fetchone()
        return max(0.0, self.get_cash() - self.get_unsettled(now) - float(pending['v']))

    def get_equity(self, prices: dict) -> float:
        """Cash + market value of open positions (missing quotes valued at avg_price)."""
        return self.get_cash() + self._positions_value(prices)

    def get_fees_paid(self) -> float:
        return self._sum_since_open('fees')

    def get_realized_pnl(self) -> float:
        """Booked P&L, net of fees, from closed paper positions since the account opened."""
        return self._sum_since_open('realized_pnl')

    def get_gross_pnl(self) -> float:
        """Booked P&L before fees (= realized + fees)."""
        return self._sum_since_open('gross_pnl')

    # --- trade lifecycle -------------------------------------------------

    def log_trade(self, symbol: str, side: str, ref_price: float, qty: float, *,
                  probability: float | None = None, exit_reason: str | None = None,
                  now: datetime | None = None) -> int:
        """Record a PENDING trade at its simulated fill. Returns its id.

        This is the only place costs are applied: the row stores the caller's
        quote (ref_price), the fill (price), fees and the net cash movement
        (amount). Callers never compute costs themselves.
        """
        qty = round(float(qty), 6)
        f = costs.fill(symbol, side, float(ref_price), qty)
        created_at = _now(now)
        with self._txn():
            cur = self.conn.execute(
                "INSERT INTO trades (symbol, side, ref_price, price, shares, fees, amount, status, "
                "created_at, week_key, trade_date, entry_probability, exit_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?)",
                (symbol, side, float(ref_price), f['fill_price'], qty, f['fees'], f['net'],
                 created_at, _week_key(now), _et_date(created_at), probability, exit_reason),
            )
            trade_id = cur.lastrowid
        logger.info(f"Logged PENDING trade #{trade_id}: {side} {costs.qty_str(qty)} {symbol} "
                    f"@ ${f['fill_price']:,.4f} (ref ${float(ref_price):,.4f}, fees ${f['fees']:.4f})")
        return trade_id

    def execute_trade(self, trade_id: int, now: datetime | None = None) -> sqlite3.Row | None:
        """PENDING -> EXECUTED: move cash, book settlement and P&L, apply the position.

        Returns the executed row so callers report the ledger's fill, amount,
        fees and P&L instead of recomputing them; None when the trade is not
        PENDING (already decided, or unknown id).
        """
        ts = _now(now)
        with self._txn():
            row = self.conn.execute(
                "SELECT * FROM trades WHERE id = ? AND status = 'PENDING'", (trade_id,)
            ).fetchone()
            if row is None:
                logger.warning(f"execute_trade: trade #{trade_id} not found or not pending")
                return None
            available_at = None
            if row['side'] == 'BUY':
                self.conn.execute("UPDATE account SET cash = cash - ? WHERE id = 1", (row['amount'],))
            else:
                self.conn.execute("UPDATE account SET cash = cash + ? WHERE id = 1", (row['amount'],))
                available_at = self._available_at(row)
            self._apply_position(row, ts)
            self.conn.execute(
                "UPDATE trades SET status = 'EXECUTED', settled_at = ?, available_at = ? WHERE id = ?",
                (ts, available_at, trade_id),
            )
            row = self.conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        logger.info(f"Trade #{trade_id} EXECUTED: {row['side']} {costs.qty_str(float(row['shares']))} "
                    f"{row['symbol']} @ ${row['price']:,.4f}, cash ${self.get_cash():,.2f}")
        return row

    def _available_at(self, row: sqlite3.Row) -> str:
        """When a SELL's proceeds become spendable. Stored on the row, never recomputed."""
        if is_crypto(row['symbol']) or self.account_type() == 'margin':
            return row['created_at']
        opens = next_trading_day_open(datetime.fromisoformat(row['created_at']))
        return opens.astimezone(timezone.utc).isoformat(timespec='seconds')

    def reject_trade(self, trade_id: int, now: datetime | None = None) -> bool:
        """PENDING -> REJECTED (declined or timed out); releases its buying-power hold."""
        with self._txn():
            cur = self.conn.execute(
                "UPDATE trades SET status = 'REJECTED', settled_at = ? "
                "WHERE id = ? AND status = 'PENDING'",
                (_now(now), trade_id),
            )
            n = cur.rowcount
        if n:
            logger.info(f"Trade #{trade_id} REJECTED")
        return bool(n)

    def _apply_position(self, row: sqlite3.Row, ts: str) -> None:
        """Weighted-average position update. Caller holds the transaction.

        avg_price is the net cost basis (built from `amount`, so fees are in
        it). entry_ref is the quantity-weighted ref_price of the open lots:
        the fast trader measures barriers against it (reference-to-reference,
        the same move the labels use), so costs show up in cash and P&L but
        never in the trigger. A SELL books realized_pnl (net) and gross_pnl.
        """
        pos = self.conn.execute(
            "SELECT shares, avg_price, entry_ref FROM positions WHERE symbol = ?", (row['symbol'],)
        ).fetchone()
        held = float(pos['shares']) if pos else 0.0
        avg = float(pos['avg_price']) if pos else 0.0
        entry_ref = float(pos['entry_ref']) if pos else 0.0
        qty = float(row['shares'])

        if row['side'] == 'BUY':
            new_shares = held + qty
            new_avg = ((held * avg) + row['amount']) / new_shares if new_shares else 0.0
            new_ref = ((held * entry_ref) + qty * row['ref_price']) / new_shares if new_shares else 0.0
        else:
            closed = min(qty, held)
            realized = row['amount'] - closed * avg              # net of fees
            self.conn.execute(
                "UPDATE trades SET realized_pnl = ?, gross_pnl = ? WHERE id = ?",
                (realized, realized + row['fees'], row['id']),
            )
            new_shares = held - qty
            new_avg, new_ref = avg, entry_ref                    # a partial SELL leaves both alone
            if new_shares < -1e-6:
                logger.warning(f"SELL exceeds held shares for {row['symbol']} - clamping to 0")
                new_shares = 0.0

        # Fractional round trips leave binary-float dust (0.1 + 0.2 - 0.3); a
        # dust position would otherwise count as "held" and block re-entry.
        if abs(new_shares) < 1e-6:
            new_shares, new_avg, new_ref = 0.0, 0.0, 0.0

        self.conn.execute(
            "INSERT INTO positions (symbol, shares, avg_price, entry_ref, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET shares=excluded.shares, avg_price=excluded.avg_price, "
            "entry_ref=excluded.entry_ref, updated_at=excluded.updated_at",
            (row['symbol'], new_shares, new_avg, new_ref, ts),
        )

    # --- positions / reporting -------------------------------------------

    def get_positions(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT symbol, shares, avg_price, entry_ref, updated_at FROM positions "
            "WHERE shares > 1e-9 ORDER BY symbol"
        ).fetchall()
        return [
            {
                'symbol': r['symbol'],
                'shares': float(r['shares']),
                # NOT rounded: FastTrader computes stop-loss and take-profit
                # against entry_ref and P&L against avg_price, so rounding to
                # cents moved the executed barriers away from the trained
                # ones. On sub-$1 crypto the error reached 5% of entry - a
                # "take profit" could fire at a real loss. Round for display
                # only, never for arithmetic.
                'avg_price': float(r['avg_price']),
                'entry_ref': float(r['entry_ref']),
                'cost_basis': float(r['shares']) * float(r['avg_price']),
                'updated_at': r['updated_at'],
            }
            for r in rows
        ]

    def get_trades_since_open(self, day: str | None = None) -> list[sqlite3.Row]:
        """EXECUTED trades since the account opened, optionally for one ET trade_date."""
        sql = "SELECT * FROM trades WHERE status = 'EXECUTED' AND created_at >= ?"
        params = [self.opened_at()]
        if day:
            sql += " AND trade_date = ?"
            params.append(day)
        return self.conn.execute(sql + " ORDER BY created_at, id", params).fetchall()
```

- [ ] **Step 4: Run the ledger tests and confirm they pass**

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_budget_tracker.py -v`

Expected: 11 passed —
```
tests/test_budget_tracker.py::test_account_reads_on_a_fresh_account PASSED
tests/test_budget_tracker.py::test_et_date_is_the_et_calendar_day PASSED
tests/test_budget_tracker.py::test_log_trade_records_fill_and_pending_hold PASSED
tests/test_budget_tracker.py::test_stock_sell_is_unsettled_until_next_trading_day_open PASSED
tests/test_budget_tracker.py::test_settlement_skips_thanksgiving PASSED
tests/test_budget_tracker.py::test_crypto_sell_settles_immediately PASSED
tests/test_budget_tracker.py::test_margin_account_settles_immediately PASSED
tests/test_budget_tracker.py::test_trades_before_opened_at_are_ignored PASSED
tests/test_budget_tracker.py::test_realized_pnl_is_net_and_gross_adds_fees_back PASSED
tests/test_budget_tracker.py::test_entry_ref_is_quantity_weighted_and_survives_partial_sell PASSED
tests/test_budget_tracker.py::test_dust_is_written_as_zero_and_hidden PASSED
11 passed
```

- [ ] **Step 5: Commit the ledger core**

```bash
cd /home/gdhughey/hugheylab-trading-bot && git add src/budget_tracker.py tests/test_budget_tracker.py && git commit -m "BudgetTracker: paper account ledger with cash, settlement, entry_ref and net/gross P&L

Replaces the weekly budget with one account row. log_trade applies costs.fill
once; execute_trade moves cash, stores available_at (T+1 for stocks in a cash
account, immediate for crypto/margin) and books realized (net) and gross P&L.
Every mutating method runs under one lock in a BEGIN IMMEDIATE transaction on
a connection the tracker owns; other threads open connect(budget.db_path).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 6: Write the failing sizing, compounding and `get_pnl` tests**

Append to `/home/gdhughey/hugheylab-trading-bot/tests/test_budget_tracker.py`:

```python
# --- sizing ----------------------------------------------------------------

def test_size_order_is_fractional_and_dollar_based(bt):
    qty, size_usd, est_fill = bt.size_order('AAPL', 100.0, now=FRI_1000)
    assert est_fill == pytest.approx(100.05)
    assert size_usd == pytest.approx(125.0)                 # 500 / FAST_MAX_POSITIONS
    assert qty == round(125.0 / 100.05, 6) == 1.249375
    assert qty * est_fill <= size_usd + 1e-3                # 6-dp qty rounding is the only slack


def test_size_order_below_min_order_returns_zero_qty(bt, monkeypatch):
    monkeypatch.setenv('MIN_ORDER_USD', '200')              # read at call time
    qty, size_usd, est_fill = bt.size_order('AAPL', 100.0, now=FRI_1000)
    assert qty == 0.0
    assert size_usd == pytest.approx(125.0)
    assert est_fill == pytest.approx(100.05)


@pytest.mark.parametrize('symbols', [
    ['AAPL', 'MSFT', 'NVDA', 'AMD'],
    ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD'],
])
def test_filling_every_slot_from_500_never_overdraws(bt, symbols):
    for sym in symbols:
        bp = bt.get_buying_power(now=FRI_1000)
        qty, size_usd, est_fill = bt.size_order(sym, 100.0, now=FRI_1000)
        assert qty > 0
        assert size_usd <= bp
        row = _round_trip(bt, sym, 'BUY', 100.0, qty, FRI_1000)
        assert row['amount'] <= bp + 1e-3                   # slippage is inside amount
        assert bt.get_cash() >= -1e-3
    assert len(bt.get_positions()) == 4
    assert bt.get_cash() == pytest.approx(0.0, abs=1e-3)
    # fully deployed: a fifth order is below MIN_ORDER_USD
    qty, size_usd, _ = bt.size_order('TSLA', 100.0, now=FRI_1000)
    assert qty == 0.0 and size_usd < 1.0


def test_entry_fills_in_full_when_buying_power_binds(bt):
    # Three stock round trips on Friday. Each SELL is profitable, so equity
    # grows to ~$537, but in a cash account the proceeds stay unsettled until
    # Monday - so buying power is only the $125 that never left. That puts
    # buying power BELOW equity / FAST_MAX_POSITIONS (~$134): the next order
    # must be capped at buying power and still fill in full there, not be
    # skipped or overdraw.
    for sym in ('AAPL', 'MSFT', 'NVDA'):
        qty, _, _ = bt.size_order(sym, 100.0, now=FRI_1000)
        _round_trip(bt, sym, 'BUY', 100.0, qty, FRI_1000)
    for sym in ('AAPL', 'MSFT', 'NVDA'):
        (pos,) = [p for p in bt.get_positions() if p['symbol'] == sym]
        sell = _round_trip(bt, sym, 'SELL', 110.0, pos['shares'], FRI_1555, exit_reason='tp')
        assert sell['realized_pnl'] > 0
    assert bt.get_positions() == []
    bp = bt.get_buying_power(now=FRI_1555)
    equity = bt.get_equity({})
    assert bt.get_unsettled(now=FRI_1555) > 400.0          # three SELLs' proceeds, held to Monday
    assert bp == pytest.approx(125.0, abs=1e-3)             # the cash left after three $125 buys
    assert equity > 530.0
    assert bp < equity / 4                                  # buying power binds, not the equity slice
    qty, size_usd, est_fill = bt.size_order('AMD', 100.0, now=FRI_1555)
    assert size_usd == pytest.approx(bp)
    assert qty == round(bp / est_fill, 6)
    row = _round_trip(bt, 'AMD', 'BUY', 100.0, qty, FRI_1555)
    assert row['amount'] == pytest.approx(bp, abs=1e-3)     # filled in full at the cap
    assert bt.get_buying_power(now=FRI_1555) == pytest.approx(0.0, abs=1e-3)
    assert bt.get_cash() > 400.0                            # unsettled proceeds are still cash
    # Monday the proceeds settle and all of it is spendable again
    assert bt.get_buying_power(now=MON_0930) == pytest.approx(bt.get_cash())


def test_compounding_after_a_win_grows_the_next_order(bt):
    qty1, size1, _ = bt.size_order('AAPL', 100.0, now=FRI_1000)
    _round_trip(bt, 'AAPL', 'BUY', 100.0, qty1, FRI_1000)
    sell = _round_trip(bt, 'AAPL', 'SELL', 120.0, qty1, FRI_1555, exit_reason='tp')
    assert sell['realized_pnl'] > 0
    # Monday, once the proceeds settle, equity has grown and so has the slice
    qty2, size2, _ = bt.size_order('MSFT', 100.0, now=MON_0930)
    assert bt.get_equity({}) == pytest.approx(bt.get_cash())
    assert size2 == pytest.approx(bt.get_cash() / 4)
    assert size2 > size1 and qty2 > qty1


# --- get_pnl -----------------------------------------------------------------

def test_get_pnl_keys_and_equity(bt):
    _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    _round_trip(bt, 'MSFT', 'BUY', 200.0, 0.5, FRI_1000)
    quotes = {'AAPL': 110.0}                                # MSFT has no quote -> stale
    pnl = bt.get_pnl(quotes.get)
    assert set(pnl) == {
        'positions', 'stale', 'realized', 'unrealized', 'total', 'cost_basis',
        'market_value', 'cash', 'unsettled', 'buying_power', 'equity', 'starting_cash',
        'all_time_net', 'all_time_pct', 'fees_paid', 'gross_pnl'}
    assert 'return_pct' not in pnl
    assert pnl['stale'] == ['MSFT']
    assert pnl['cash'] == pytest.approx(500.0 - 100.05 - 0.5 * 200.1)
    assert pnl['starting_cash'] == 500.0
    assert pnl['unrealized'] == pytest.approx(110.0 - 100.05)
    assert pnl['market_value'] == pytest.approx(110.0)
    assert pnl['cost_basis'] == pytest.approx(100.05)
    # the stale MSFT is carried at avg_price, so equity moves only with AAPL
    assert pnl['equity'] == pytest.approx(pnl['cash'] + 110.0 + 0.5 * 200.1)
    assert pnl['all_time_net'] == pytest.approx(pnl['equity'] - 500.0)
    assert pnl['all_time_pct'] == pytest.approx(pnl['all_time_net'] / 500.0)
    assert pnl['unsettled'] == 0.0 and pnl['buying_power'] == pytest.approx(pnl['cash'])
    assert pnl['realized'] == 0.0 and pnl['fees_paid'] == 0.0 and pnl['gross_pnl'] == 0.0
    assert pnl['total'] == pytest.approx(pnl['unrealized'])
    assert [p['symbol'] for p in pnl['positions']] == ['AAPL', 'MSFT']
    assert pnl['positions'][1]['price'] is None and pnl['positions'][1]['pnl'] is None
```

- [ ] **Step 7: Run to confirm the new tests fail on the missing methods**

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_budget_tracker.py -v`

Expected: 11 passed, 7 failed; each failure ends with one of
```
AttributeError: 'BudgetTracker' object has no attribute 'size_order'
AttributeError: 'BudgetTracker' object has no attribute 'get_pnl'
```

- [ ] **Step 8: Add `size_order` and `get_pnl`**

In `/home/gdhughey/hugheylab-trading-bot/src/budget_tracker.py`, insert the following two methods into `class BudgetTracker` immediately AFTER `get_gross_pnl` and BEFORE the `# --- trade lifecycle` comment:

```python
    # --- sizing ----------------------------------------------------------

    def size_order(self, symbol: str, ref_price: float, prices: dict | None = None,
                   now: datetime | None = None) -> tuple[float, float, float]:
        """(qty, size_usd, est_fill) for a BUY; qty is 0.0 when the order is too small.

        Every entry is the same fraction of equity (equity / FAST_MAX_POSITIONS)
        capped by buying power, so wins compound and losses shrink the next
        order. Orders are dollar-based (qty = size_usd / estimated fill), so the
        cash debit can never exceed buying power (6-dp qty rounding aside).
        `prices` is the dict of quotes the same cycle already fetched for held
        symbols - no second quote is taken for sizing.
        """
        buying_power = self.get_buying_power(now)
        equity = self.get_equity(prices or {})
        slots = int(os.getenv('FAST_MAX_POSITIONS', 4))
        size_usd = min(buying_power, equity / slots)
        est_fill = costs.fill(symbol, 'BUY', float(ref_price), 1)['fill_price']
        if size_usd < float(os.getenv('MIN_ORDER_USD', 1)):
            return 0.0, size_usd, est_fill
        return round(size_usd / est_fill, 6), size_usd, est_fill
```

Then insert this method immediately AFTER `get_positions` and BEFORE `get_trades_since_open`:

```python
    def get_pnl(self, price_fn) -> dict:
        """Mark open positions to market and roll up the whole account.

        `price_fn(symbol)` returns a current price, or None when unavailable.
        Those positions are listed as stale with pnl None (a data outage never
        masquerades as break-even); for equity they are carried at avg_price
        because the account needs one number.
        """
        positions, unrealized, cost_total, market_total, stale, prices = [], 0.0, 0.0, 0.0, [], {}

        for pos in self.get_positions():
            price = None
            try:
                price = price_fn(pos['symbol'])
            except Exception as e:
                logger.warning(f"price lookup failed for {pos['symbol']}: {e}")

            if price is None:
                stale.append(pos['symbol'])
                positions.append({**pos, 'price': None, 'pnl': None, 'pnl_pct': None})
                continue

            prices[pos['symbol']] = price
            market = price * pos['shares']
            pnl = market - pos['cost_basis']
            unrealized += pnl
            cost_total += pos['cost_basis']
            market_total += market
            positions.append({
                **pos,
                'price': round(price, 2),
                'market_value': round(market, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': (pnl / pos['cost_basis']) if pos['cost_basis'] else 0.0,
            })

        realized = self.get_realized_pnl()
        starting = self.starting_cash()
        equity = self.get_equity(prices)
        return {
            'positions': positions,
            'stale': stale,
            'realized': realized,
            'unrealized': unrealized,
            'total': realized + unrealized,
            'cost_basis': cost_total,
            'market_value': market_total,
            'cash': self.get_cash(),
            'unsettled': self.get_unsettled(),
            'buying_power': self.get_buying_power(),
            'equity': equity,
            'starting_cash': starting,
            'all_time_net': equity - starting,
            # a fraction, like pnl_pct: format with :+.2%
            'all_time_pct': ((equity - starting) / starting) if starting else 0.0,
            'fees_paid': self.get_fees_paid(),
            'gross_pnl': self.get_gross_pnl(),
        }
```

- [ ] **Step 9: Run the tests and confirm all pass**

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_budget_tracker.py -v`

Expected: `18 passed` (the 11 from Step 4 plus `test_size_order_is_fractional_and_dollar_based`, `test_size_order_below_min_order_returns_zero_qty`, `test_filling_every_slot_from_500_never_overdraws[symbols0]`, `test_filling_every_slot_from_500_never_overdraws[symbols1]`, `test_entry_fills_in_full_when_buying_power_binds`, `test_compounding_after_a_win_grows_the_next_order`, `test_get_pnl_keys_and_equity`).

- [ ] **Step 10: Commit sizing and P&L roll-up**

```bash
cd /home/gdhughey/hugheylab-trading-bot && git add src/budget_tracker.py tests/test_budget_tracker.py && git commit -m "BudgetTracker: equity-fraction size_order and account-wide get_pnl

size_order = min(buying power, equity / FAST_MAX_POSITIONS) in dollars, so a
full set of slots from \$500 never overdraws, an order capped by unsettled
proceeds still fills in full at buying power, and a win compounds into the
next order. get_pnl gains cash/unsettled/buying_power/equity/all_time_*;
return_pct is gone.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 11: Write the failing day-state and equity-history tests**

Append to `/home/gdhughey/hugheylab-trading-bot/tests/test_budget_tracker.py`:

```python
# --- day state / equity history -----------------------------------------------

def test_day_state_baseline_and_flags(bt):
    assert bt.get_day_state('2026-09-18') is None
    row = bt.ensure_day_state('2026-09-18', 500.0)
    assert row['date'] == '2026-09-18'
    assert row['start_equity'] == 500.0
    assert row['loss_tripped_at'] is None
    assert row['loss_announced_at'] is None
    assert row['report_posted_at'] is None
    # a same-date restart must keep the original baseline
    assert bt.ensure_day_state('2026-09-18', 480.0)['start_equity'] == 500.0
    tripped = FRI_1555.isoformat(timespec='seconds')
    posted = et(2026, 9, 18, 16, 5).isoformat(timespec='seconds')
    bt.set_day_flag('2026-09-18', 'loss_tripped_at', tripped)
    bt.set_day_flag('2026-09-18', 'report_posted_at', posted)
    row = bt.get_day_state('2026-09-18')
    assert row['loss_tripped_at'] == tripped
    assert row['loss_announced_at'] is None
    assert row['report_posted_at'] == posted
    with pytest.raises(ValueError):
        bt.set_day_flag('2026-09-18', 'start_equity', tripped)


def test_equity_series_starts_with_starting_cash_and_record_equity_upserts(bt):
    assert bt.equity_series() == [{'date': '2026-09-14', 'equity': 500.0}]
    _round_trip(bt, 'AAPL', 'BUY', 100.0, 1.0, FRI_1000)
    first = bt.record_equity('2026-09-18', {'AAPL': 90.0}, now=FRI_1600)
    assert first['date'] == '2026-09-18'
    assert first['cash'] == pytest.approx(500.0 - 100.05)
    assert first['positions_value'] == pytest.approx(90.0)
    assert first['equity'] == pytest.approx(first['cash'] + 90.0)
    assert first['fees_to_date'] == 0.0 and first['realized_to_date'] == 0.0
    assert first['recorded_at'] == FRI_1600.isoformat(timespec='seconds')
    # second call for the same date replaces, not duplicates
    later = et(2026, 9, 18, 16, 10)
    second = bt.record_equity('2026-09-18', {'AAPL': 95.0}, now=later)
    assert second['positions_value'] == pytest.approx(95.0)
    assert second['equity'] == pytest.approx(first['cash'] + 95.0)
    assert second['recorded_at'] == later.isoformat(timespec='seconds')
    assert bt.conn.execute("SELECT COUNT(*) AS n FROM equity_history").fetchone()['n'] == 1
    # a missing quote is carried at avg_price
    carried = bt.record_equity('2026-09-21', {}, now=MON_0930)
    assert carried['positions_value'] == pytest.approx(100.05)
    series = bt.equity_series()
    assert [s['date'] for s in series] == ['2026-09-14', '2026-09-18', '2026-09-21']
    assert series[0] == {'date': '2026-09-14', 'equity': 500.0}
    assert series[1]['equity'] == pytest.approx(second['equity'])
    assert series[2]['equity'] == pytest.approx(carried['equity'])
```

- [ ] **Step 12: Run to confirm the new tests fail on the missing methods**

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_budget_tracker.py -v`

Expected: 18 passed, 2 failed, with
```
AttributeError: 'BudgetTracker' object has no attribute 'get_day_state'
AttributeError: 'BudgetTracker' object has no attribute 'equity_series'
```

- [ ] **Step 13: Add the day-state and equity-history methods**

In `/home/gdhughey/hugheylab-trading-bot/src/budget_tracker.py`, append the following block to the END of `class BudgetTracker` (after `get_trades_since_open`):

```python
    # --- day state / equity history --------------------------------------

    def get_day_state(self, date_et: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM day_state WHERE date = ?", (date_et,)).fetchone()

    def ensure_day_state(self, date_et: str, equity: float) -> sqlite3.Row:
        """Create the day's row with its start-of-day equity; a same-date restart keeps the original."""
        with self._txn():
            self.conn.execute(
                "INSERT OR IGNORE INTO day_state (date, start_equity) VALUES (?, ?)",
                (date_et, float(equity)),
            )
        return self.get_day_state(date_et)

    def set_day_flag(self, date_et: str, column: str, ts: str | None = None) -> None:
        """Stamp one of the day's latches (loss tripped / announced, report posted)."""
        if column not in DAY_FLAGS:
            raise ValueError(f"set_day_flag: {column!r} is not one of {DAY_FLAGS}")
        with self._txn():
            cur = self.conn.execute(
                f"UPDATE day_state SET {column} = ? WHERE date = ?", (ts or _now(), date_et)
            )
            n = cur.rowcount
        if not n:
            logger.warning(f"set_day_flag: no day_state row for {date_et} ({column} not set)")

    def record_equity(self, date_et: str, prices: dict, now: datetime | None = None) -> dict:
        """Upsert the day's equity snapshot (written by the 16:05 ET report tick, not by /pnl)."""
        with self._txn():
            # Read inside the transaction so the snapshot cannot straddle a
            # trade executing on the other thread.
            cash = self.get_cash()
            positions_value = self._positions_value(prices)
            self.conn.execute(
                "INSERT INTO equity_history (date, cash, positions_value, equity, fees_to_date, "
                "realized_to_date, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(date) DO UPDATE SET cash=excluded.cash, "
                "positions_value=excluded.positions_value, equity=excluded.equity, "
                "fees_to_date=excluded.fees_to_date, realized_to_date=excluded.realized_to_date, "
                "recorded_at=excluded.recorded_at",
                (date_et, cash, positions_value, cash + positions_value,
                 self.get_fees_paid(), self.get_realized_pnl(), _now(now)),
            )
            row = self.conn.execute(
                "SELECT * FROM equity_history WHERE date = ?", (date_et,)).fetchone()
        return dict(row)

    def equity_series(self) -> list[dict]:
        """Daily equity, oldest first, with day 0 = starting cash on the open date.

        Day 0 is synthetic (only `date` and `equity`); the rest are full
        equity_history rows. Drawdown and the chart need the starting point
        so a losing first day is not a 0% drawdown.
        """
        rows = self.conn.execute("SELECT * FROM equity_history ORDER BY date").fetchall()
        return ([{'date': self.opened_at()[:10], 'equity': self.starting_cash()}]
                + [dict(r) for r in rows])
```

- [ ] **Step 14: Run the ledger tests, then the whole suite**

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_budget_tracker.py -v`

Expected: `20 passed`.

Run: `cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/ -q`

Expected: every test in `tests/test_costs.py`, `tests/test_calendar.py`, `tests/test_migration.py`, `tests/test_budget_tracker.py` and `tests/test_smoke.py` passes; last line `N passed` with no failures or errors.

- [ ] **Step 15: Commit day state and equity history**

```bash
cd /home/gdhughey/hugheylab-trading-bot && git add src/budget_tracker.py tests/test_budget_tracker.py && git commit -m "BudgetTracker: day_state latches and equity_history snapshots

ensure_day_state keeps the first start_equity across a same-date restart;
set_day_flag stamps the whitelisted loss/report latches; record_equity upserts
one row per ET date; equity_series prefixes starting cash as day 0.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 16: Correct the spec and contract text this task revealed as wrong (documentation only, no code)**

Three edits, each an exact string replacement. None changes any name the code uses.

16a. In `/home/gdhughey/hugheylab-trading-bot/docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md`, line 22 currently reads

```
| 7 | scorecard | `src/scorecard.py`, `tests/test_scorecard.py` | 4 |
```

Replace it with

```
| 7 | scorecard | `src/scorecard.py`, `tests/test_scorecard.py` | 4, 6 (8 for the live report) |
```

(Task 7 imports `class_gate` from `src/fast_trader.py`, which Task 6 creates, and the live report calls `first_close_on_or_after` from `src/ml_engine.py`, which Task 8 adds.)

16b. In `/home/gdhughey/hugheylab-trading-bot/docs/superpowers/specs/2026-09-12-paper-brokerage-design.md`, lines 93-94 currently read

```
Every mutating `BudgetTracker` method (`log_trade`, `execute_trade`,
`reject_trade`, `record_day_state`, `record_equity`) takes one
```

Replace them with

```
Every mutating `BudgetTracker` method (`log_trade`, `execute_trade`,
`reject_trade`, `ensure_day_state`, `set_day_flag`, `record_equity`) takes one
```

(`record_day_state` never existed; the contract and the code call it `ensure_day_state`, and `set_day_flag` is also a locked writer.)

16c. In the same spec file, lines 343-344 currently read

```
  third entry when buying power binds fills in full; compounding after a
  win.
```

Replace them with

```
  an entry when buying power binds (unsettled proceeds leave it below
  equity / FAST_MAX_POSITIONS) fills in full at buying power; compounding
  after a win.
```

(This is what `test_entry_fills_in_full_when_buying_power_binds` covers; from a fresh account the equity slice and buying power are equal on every slot, so "third entry" could never bind.)

Verify the edits landed and nothing else changed:

```bash
cd /home/gdhughey/hugheylab-trading-bot && git diff --stat docs/ && grep -n "4, 6 (8 for the live report)" docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md && grep -n "ensure_day_state" docs/superpowers/specs/2026-09-12-paper-brokerage-design.md && grep -c "record_day_state" docs/superpowers/specs/2026-09-12-paper-brokerage-design.md
```

Expected: the `--stat` shows exactly the two doc files changed; the first two greps print one matching line each; the last prints `0`.

```bash
cd /home/gdhughey/hugheylab-trading-bot && git add docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md docs/superpowers/specs/2026-09-12-paper-brokerage-design.md && git commit -m "docs: Task 7 depends on 6 (and 8 live); record_day_state -> ensure_day_state; sizing test wording

The contract's dependency row for the scorecard task omitted class_gate
(Task 6) and first_close_on_or_after (Task 8). The spec named a locked
method record_day_state that the contract and code call ensure_day_state.
The 'third entry when buying power binds' test item could never bind from a
fresh account; it now describes the unsettled-proceeds case the test covers.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

**Contract additions**
- `BudgetTracker.db_path: str` — the resolved database path (`db_path or DB_PATH`), so another thread can open its own connection with `connect(budget.db_path)`. Task 9's `_close_the_books` must use this (or `intraday.conn`) for `signal_log.label_pending`, never `budget.conn`.
- `_now(now: datetime | None = None) -> str` — the existing module-level helper gains an optional tz-aware override; unchanged output when called with no argument.
- `DAY_FLAGS = ('loss_tripped_at', 'loss_announced_at', 'report_posted_at')` — module constant; `set_day_flag` raises `ValueError` for any other column.
- `BudgetTracker._txn()` — `@contextmanager`; acquires `self._lock`, runs `BEGIN IMMEDIATE`, commits on success / rolls back on exception. `BudgetTracker.conn.isolation_level` is set to `None` (autocommit) in `__init__`. **Ownership rule:** `budget.conn` may be read from any thread, but written through (bare `execute`, or `with budget.conn:`, whose `__exit__` commits) only on the thread that runs `FastTrader.cycle`, sequentially with `_txn` — a commit from any other thread would commit a half-finished `_txn`. Other threads open `connect(budget.db_path)`.
- `BudgetTracker._account() -> sqlite3.Row` — the `account` row; raises `RuntimeError` when missing.
- `BudgetTracker._sum_since_open(column: str) -> float` — `SUM(column)` over EXECUTED trades with `created_at >= opened_at` (column is a literal name).
- `BudgetTracker._positions_value(prices: dict) -> float` — Σ shares × `prices.get(symbol)` with `avg_price` fallback; shared by `get_equity` and `record_equity`.
- `BudgetTracker._available_at(row: sqlite3.Row) -> str` — settlement timestamp for a SELL row (crypto or margin: `created_at`; else `next_trading_day_open(created_at)` as UTC ISO seconds).
- `BudgetTracker._apply_position(row: sqlite3.Row, ts: str) -> None` — existing private name, now takes the transaction timestamp instead of calling `_now()`.

---

### Task 5: Signal log

Every scored symbol on every 5m bar goes into the `signals` table (one row per `(symbol, bar_ts)`, so the 60 s poll cannot inflate n), and the 16:05 ET tick labels each row after the fact with the same `triple_barrier` rule the model was trained on. This is the primary endpoint of the decision rule (spec §7/§8): ~40 labelled signals a day instead of ≤ 4 executed trades.

One thing the contract's prose glosses over, verified on the live DB (LXC 200, `prices_intraday`): stored `ts` strings carry the ET offset exactly as yfinance served them (`'2026-09-11T15:55:00-04:00'`, `-05:00` in winter), while `bar_ts` from `signal()` is UTC (`'...+00:00'`). A plain SQL string comparison `ts >= bar_ts` is therefore WRONG (`15:55-04:00` sorts before `19:55+00:00` although they are the same instant). The implementation below normalises both sides through SQLite's `datetime()`, which converts any `[+-]HH:MM` suffix to UTC; the tests store bars with the ET offset on purpose so this stays covered.

**When a stock signal gets its label (resolves the spec gap).** A stock's training label never depends on the next session's bars: `triple_barrier` stops the walk at the ET session boundary and scores an unresolved trade 0. `label_pending` applies the same rule *the same day*: once `now` is past the session close (16:00 ET, 13:00 on `EARLY_CLOSE_2026` days) AND the session's closing bar (15:55 ET on a regular day) is stored, a stock signal with fewer than `horizon + 1` bars is labelled by walking to the last stored bar — which is exactly where the session cut would have stopped it. So the 16:05 tick labels nearly every stock row from that day (Task 10 Step 12 relies on this). The closing-bar condition is deliberate: a feed outage that stopped bars at 15:30 must not be read as "nothing happened until the bell"; such rows stay NULL until the bars arrive. Crypto has no session and runs the full horizon: a crypto window that runs past the last stored bar stays NULL and is retried on the next call.

**Connection ownership (read this before wiring Tasks 6 and 9).** `BudgetTracker.conn` (Task 4) is autocommit and driven with explicit `BEGIN IMMEDIATE` under `_txn()`'s lock. `record()` and `mark_executed()` are called by `FastTrader.cycle` on the cycle's own worker thread, sequentially with `_txn()`, so they may run on `budget.conn` (Task 6 does this; its `with conn:` on an autocommit connection is harmless there). `label_pending()` is different: Task 9's `_close_the_books` calls it from the 16:05 report's `to_thread` worker while the fast cycle (crypto keeps it alive after the bell with `TRADE_CRYPTO=1`) may be inside `_txn()` on the same connection, and `with conn:` would then commit the cycle's half-finished `log_trade`/`execute_trade`. `label_pending` must therefore be given a connection BudgetTracker does not own: `self.intraday.conn` (IntradayEngine's own connection, default isolation) or a fresh `connect(db_path)` closed after the call — never `self.budget_tracker.conn`. Task 9's `FakeIntraday` already exposes `conn` for this.

Depends on Task 3 (the `signals` table in `SCHEMA` and the `db_path` fixture in `tests/conftest.py`). Exactly this code was run on LXC 200 (Python 3.13.5, pandas 3.0.5, SQLite 3.46.1) against a stand-in `db_path` fixture carrying the contract's `signals` DDL: 13 passed; with `src/signal_log.py` absent the collection error of Step 2 is reproduced, and with the pre-change `signal()` the `KeyError: 'bar_ts'` of Step 4 is reproduced.

**Files:**
- Create: `/home/gdhughey/hugheylab-trading-bot/src/signal_log.py`
- Create: `/home/gdhughey/hugheylab-trading-bot/tests/test_signal_log.py`
- Modify: `/home/gdhughey/hugheylab-trading-bot/src/intraday_engine.py` — `IntradayEngine.signal()` return dict, currently lines 529-538 (the `return {...}` block inside `signal()`, lines 507-538)

- [ ] **Step 1: Write the failing tests**

Create `/home/gdhughey/hugheylab-trading-bot/tests/test_signal_log.py` with exactly this content:

```python
"""Signal log: dedupe per bar, trade linkage, and after-the-fact labelling."""
from datetime import datetime, timedelta, timezone

import pytest

from src import signal_log
from src.database import connect
from src.intraday_engine import ET, INTERVAL, IntradayEngine

# Fixed barriers so the tests do not depend on data/tuned.json or .env:
# +1.0% take-profit, -0.6% stop, 4-bar horizon.
TP, SL, HORIZON = 0.01, 0.006, 4

# A Monday 15:50 ET bar; its UTC ISO is what IntradayEngine.signal() emits.
BAR_ET = datetime(2026, 9, 14, 15, 50, tzinfo=ET)
BAR_TS = BAR_ET.astimezone(timezone.utc).isoformat()          # '2026-09-14T19:50:00+00:00'
NOW = datetime(2026, 9, 15, 20, 5, tzinfo=timezone.utc)      # Tuesday 16:05 ET
MON_TICK = datetime(2026, 9, 14, 16, 5, tzinfo=ET)           # the same-day 16:05 tick


@pytest.fixture
def conn(db_path):
    # IntradayEngine.__init__ only opens the DB and creates prices_intraday
    # (no fetch, no model load without data/); Database() (db_path fixture)
    # already created the signals table.
    IntradayEngine(db_path)
    c = connect(db_path)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def fixed_barriers(monkeypatch):
    monkeypatch.setattr(signal_log, 'barriers', lambda cls: (TP, SL, HORIZON))


def _sig(symbol, price=100.0, prob=0.6, bar_ts=BAR_TS):
    cls = 'crypto' if symbol.endswith('-USD') else 'stock'
    return {'symbol': symbol, 'asset_class': cls, 'probability': prob, 'bar': 0.5,
            'above_bar': prob >= 0.5, 'price': price, 'bar_ts': bar_ts, 'margin': prob - 0.5}


def _insert_bars(conn, symbol, bars):
    """bars: [(datetime ET-aware, high, low, close)]; stored the way fetch()
    stores yfinance bars, i.e. with the ET offset, NOT in UTC."""
    with conn:
        conn.executemany(
            "INSERT INTO prices_intraday (symbol, ts, open, high, low, close, volume, interval) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [(symbol, ts.isoformat(), close, high, low, close, 1000.0, INTERVAL)
             for ts, high, low, close in bars])


def _path(symbol, moves, start=BAR_ET):
    """Signal bar (close 100) followed by one bar per (high, low, close) in
    `moves`, five minutes apart. `moves` entries may be a datetime to jump the
    clock (next session), followed by that bar's (high, low, close)."""
    bars = [(start, 100.2, 99.8, 100.0)]
    ts = start
    for m in moves:
        if isinstance(m, datetime):
            ts = m - timedelta(minutes=5)
            continue
        ts = ts + timedelta(minutes=5)
        bars.append((ts, *m))
    return bars


def _row(conn, symbol):
    return conn.execute("SELECT * FROM signals WHERE symbol = ?", (symbol,)).fetchone()


# --- record / mark_executed ------------------------------------------------

def test_record_writes_one_row_per_symbol_per_bar(conn):
    sigs = [_sig('AAPL', prob=0.6), _sig('BTC-USD', price=60000.0, prob=0.3)]
    assert signal_log.record(conn, sigs, now=NOW) == 2
    # Same 5m bar scored again on the next 60 s poll: nothing new.
    assert signal_log.record(conn, sigs, now=NOW) == 0
    rows = conn.execute("SELECT * FROM signals ORDER BY symbol").fetchall()
    assert len(rows) == 2
    aapl = _row(conn, 'AAPL')
    assert aapl['bar_ts'] == BAR_TS
    assert aapl['asset_class'] == 'stock'
    assert aapl['above_bar'] == 1
    assert aapl['ref_price'] == 100.0
    assert aapl['trade_date'] == '2026-09-14'
    assert aapl['label'] is None and aapl['labeled_at'] is None
    assert aapl['executed_trade_id'] is None
    assert _row(conn, 'BTC-USD')['above_bar'] == 0
    assert signal_log.record(conn, [], now=NOW) == 0


def test_record_trade_date_is_et_not_utc(conn):
    # 01:00 UTC on the 15th is still 21:00 ET on the 14th.
    late = datetime(2026, 9, 15, 1, 0, tzinfo=timezone.utc).isoformat()
    signal_log.record(conn, [_sig('BTC-USD', bar_ts=late)], now=NOW)
    assert _row(conn, 'BTC-USD')['trade_date'] == '2026-09-14'


def test_mark_executed_sets_trade_id(conn):
    signal_log.record(conn, [_sig('AAPL'), _sig('MSFT')], now=NOW)
    signal_log.mark_executed(conn, 'AAPL', BAR_TS, 42)
    assert _row(conn, 'AAPL')['executed_trade_id'] == 42
    assert _row(conn, 'MSFT')['executed_trade_id'] is None


# --- label_pending -----------------------------------------------------------

def test_label_take_profit_first_is_1(conn):
    signal_log.record(conn, [_sig('AAPL')], now=NOW)
    _insert_bars(conn, 'AAPL', _path('AAPL', [
        (100.5, 99.9, 100.2),      # neither barrier
        (101.0, 99.8, 100.7),      # high touches +1.0% -> TP first
        (100.9, 99.3, 99.4),       # stop would fire here, but TP already won
        (100.0, 99.0, 99.1),
    ]))
    assert signal_log.label_pending(conn, now=NOW) == 1
    row = _row(conn, 'AAPL')
    assert row['label'] == 1
    assert row['labeled_at'] == '2026-09-15T20:05:00+00:00'
    # Already labelled rows are not touched again.
    assert signal_log.label_pending(conn, now=NOW) == 0


def test_label_stop_first_is_0(conn):
    signal_log.record(conn, [_sig('AAPL')], now=NOW)
    _insert_bars(conn, 'AAPL', _path('AAPL', [
        (100.5, 99.9, 100.2),
        (100.6, 99.4, 99.5),       # low touches -0.6% -> stop first
        (102.0, 99.5, 101.5),      # TP only after the stop: too late
        (102.0, 101.0, 101.5),
    ]))
    assert signal_log.label_pending(conn, now=NOW) == 1
    assert _row(conn, 'AAPL')['label'] == 0


def test_label_horizon_expires_flat_is_0(conn):
    signal_log.record(conn, [_sig('AAPL')], now=NOW)
    _insert_bars(conn, 'AAPL', _path('AAPL', [(100.3, 99.7, 100.1)] * 4
                                      + [(102.0, 101.0, 101.5)]))   # TP at bar 5 > horizon
    assert signal_log.label_pending(conn, now=NOW) == 1
    assert _row(conn, 'AAPL')['label'] == 0


def test_label_session_boundary_stops_stock_but_not_crypto(conn):
    next_open = datetime(2026, 9, 15, 9, 30, tzinfo=ET)
    moves = [
        (100.5, 99.9, 100.2),      # 15:55 ET, unresolved at the bell
        next_open,
        (102.0, 100.5, 101.5),     # 09:30 next day: TP, but a stock is flat by then
        (102.0, 100.5, 101.5),
        (102.0, 100.5, 101.5),
    ]
    signal_log.record(conn, [_sig('AAPL'), _sig('BTC-USD')], now=NOW)
    _insert_bars(conn, 'AAPL', _path('AAPL', moves))
    _insert_bars(conn, 'BTC-USD', _path('BTC-USD', moves))
    assert signal_log.label_pending(conn, now=NOW) == 2
    assert _row(conn, 'AAPL')['label'] == 0        # session ended unresolved
    assert _row(conn, 'BTC-USD')['label'] == 1     # crypto runs the full horizon


def test_label_insufficient_bars_stays_null_then_resolves(conn):
    # A 10:00 ET signal with 3 of its 5 bars stored, checked mid-session:
    # the session can still resolve it, so it waits.
    start = datetime(2026, 9, 14, 10, 0, tzinfo=ET)
    bar_ts = start.astimezone(timezone.utc).isoformat()
    signal_log.record(conn, [_sig('AAPL', bar_ts=bar_ts)], now=NOW)
    bars = _path('AAPL', [(100.5, 99.9, 100.2), (101.0, 99.8, 100.7)], start=start)
    _insert_bars(conn, 'AAPL', bars)
    mid_session = datetime(2026, 9, 14, 10, 20, tzinfo=ET)
    assert signal_log.label_pending(conn, now=mid_session) == 0
    row = _row(conn, 'AAPL')
    assert row['label'] is None and row['labeled_at'] is None
    # The next fetch completes the window and the next call labels it.
    last = bars[-1][0]
    _insert_bars(conn, 'AAPL', [(last + timedelta(minutes=5), 100.9, 99.3, 99.4),
                                (last + timedelta(minutes=10), 100.0, 99.0, 99.1)])
    assert signal_log.label_pending(conn, now=mid_session) == 1
    assert _row(conn, 'AAPL')['label'] == 1


def test_label_stock_session_end_is_0_same_day(conn):
    # Two 15:50 ET signals with only the 15:55 closing bar stored (2 of 5).
    # AAPL is unresolved at the bell; MSFT touches TP on the closing bar.
    signal_log.record(conn, [_sig('AAPL'), _sig('MSFT')], now=NOW)
    _insert_bars(conn, 'AAPL', _path('AAPL', [(100.5, 99.9, 100.2)]))
    _insert_bars(conn, 'MSFT', _path('MSFT', [(101.0, 99.8, 100.7)]))
    # Before the bell the window is simply incomplete: nothing is labelled.
    assert signal_log.label_pending(conn, now=datetime(2026, 9, 14, 15, 58, tzinfo=ET)) == 0
    assert _row(conn, 'AAPL')['label'] is None and _row(conn, 'MSFT')['label'] is None
    # The same day's 16:05 tick: the session is over, so the walk stops where
    # the session cut would have stopped it. No next-session bars needed.
    assert signal_log.label_pending(conn, now=MON_TICK) == 2
    aapl = _row(conn, 'AAPL')
    assert aapl['label'] == 0
    assert aapl['labeled_at'] == '2026-09-14T20:05:00+00:00'
    assert _row(conn, 'MSFT')['label'] == 1


def test_label_stock_session_end_needs_the_closing_bar(conn):
    # 15:40 ET signal, bars stored only to 15:45 (the feed died before the
    # close). Session over or not, "nothing happened" cannot be inferred from
    # missing bars: it stays NULL until the closing bar arrives.
    start = datetime(2026, 9, 14, 15, 40, tzinfo=ET)
    bar_ts = start.astimezone(timezone.utc).isoformat()
    signal_log.record(conn, [_sig('AAPL', bar_ts=bar_ts)], now=NOW)
    bars = _path('AAPL', [(100.5, 99.9, 100.2)], start=start)          # 15:45 only
    _insert_bars(conn, 'AAPL', bars)
    assert signal_log.label_pending(conn, now=MON_TICK) == 0
    assert _row(conn, 'AAPL')['label'] is None
    last = bars[-1][0]
    _insert_bars(conn, 'AAPL', [(last + timedelta(minutes=5), 100.4, 99.8, 100.1),   # 15:50
                                (last + timedelta(minutes=10), 100.3, 99.7, 100.0)])  # 15:55
    assert signal_log.label_pending(conn, now=MON_TICK) == 1
    assert _row(conn, 'AAPL')['label'] == 0


def test_label_crypto_insufficient_bars_stays_null_after_the_bell(conn):
    # Crypto has no session: a 15:50 ET signal with 2 of 5 bars stays NULL
    # even on the next day's tick, until the window is actually stored.
    signal_log.record(conn, [_sig('BTC-USD')], now=NOW)
    _insert_bars(conn, 'BTC-USD', _path('BTC-USD', [(100.5, 99.9, 100.2)]))
    assert signal_log.label_pending(conn, now=NOW) == 0
    assert _row(conn, 'BTC-USD')['label'] is None


def test_label_skips_signal_whose_own_bar_is_missing(conn):
    signal_log.record(conn, [_sig('AAPL')], now=NOW)
    bars = _path('AAPL', [(102.0, 100.5, 101.5)] * 5)
    _insert_bars(conn, 'AAPL', bars[1:])          # window present, signal bar absent
    assert signal_log.label_pending(conn, now=NOW) == 0
    assert _row(conn, 'AAPL')['label'] is None


# --- IntradayEngine.signal() carries bar_ts --------------------------------

class _AlwaysUp:
    def predict_proba(self, X):
        return [[0.4, 0.6]] * len(X)


def test_signal_includes_bar_ts(db_path, monkeypatch):
    monkeypatch.setenv('MAX_BAR_AGE_MIN', '1000000000')   # stale-bar guard is not under test
    engine = IntradayEngine(db_path)
    engine.models = {'stock': _AlwaysUp()}
    start = datetime(2026, 9, 14, 9, 30, tzinfo=ET)
    bars = []
    for i in range(80):
        close = 100.0 + (i % 7) * 0.1
        bars.append((start + timedelta(minutes=5 * i), close + 0.2, close - 0.2, close))
    _insert_bars(engine.conn, 'AAPL', bars)
    sig = engine.signal('AAPL')
    assert sig is not None
    assert sig['bar_ts'] == (start + timedelta(minutes=5 * 79)).astimezone(timezone.utc).isoformat()
    assert sig['bar_ts'].endswith('+00:00')
    assert sig['price'] == bars[-1][3]
```

- [ ] **Step 2: Run the tests and watch them fail at import**

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_signal_log.py -v
```

Expected: collection error, no tests run:

```
ImportError while importing test module '/opt/trading-bot-dev/tests/test_signal_log.py'.
E   ImportError: cannot import name 'signal_log' from 'src' (/opt/trading-bot-dev/src/__init__.py)
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.13s
```

- [ ] **Step 3: Create `src/signal_log.py`**

Create `/home/gdhughey/hugheylab-trading-bot/src/signal_log.py` with exactly this content:

```python
#!/usr/bin/env python3
"""
Signal log - every scored symbol on every 5m bar, labelled after the fact.

The executed-trade record is tiny (a handful of round trips a day), so it can
never resolve whether live precision matches the backtest. The signal log can:
FastTrader.cycle writes one row per scored symbol per bar, and the 16:05 ET
report labels each row with the SAME triple-barrier rule the model was trained
on. That gives ~40 labelled signals a day, enough for a real confidence
interval by the review date.

Connection ownership. record() and mark_executed() run on the fast cycle's
thread, sequentially with BudgetTracker._txn(), so they may share
budget.conn. label_pending() runs on the 16:05 report's worker thread while
the cycle may be mid-transaction on budget.conn (crypto keeps the cycle alive
after the bell); its `with conn:` commit would then commit the cycle's
half-finished trade. Callers pass it a connection BudgetTracker does not own
(IntradayEngine.conn, or a fresh connect(path)).
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.intraday_engine import EARLY_CLOSE_2026, ET, INTERVAL, barriers
from src.labeling import triple_barrier

logger = logging.getLogger(__name__)


def _now_iso(now=None) -> str:
    """UTC ISO seconds; `now` is injected by tests, wall clock otherwise."""
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(timespec='seconds')


def _et_date(ts_iso: str) -> str:
    """ET calendar date of a tz-aware ISO timestamp (the report's "today")."""
    return datetime.fromisoformat(ts_iso).astimezone(ET).strftime('%Y-%m-%d')


def record(conn, signals, now=None) -> int:
    """Insert scan_all() output; returns the number of rows actually inserted.

    INSERT OR IGNORE on (symbol, bar_ts): the fast loop polls every 60 s but a
    5m bar is current for five polls, so without this n would be inflated 5x
    by duplicates of the same signal. `now` is accepted for signature symmetry
    with the other clock-taking functions; the row is stamped by its bar, not
    by the wall clock.
    """
    rows = [(s['bar_ts'], s['symbol'], s['asset_class'], float(s['probability']),
             float(s['bar']), int(bool(s['above_bar'])), float(s['price']),
             _et_date(s['bar_ts']))
            for s in signals]
    if not rows:
        return 0
    with conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO signals "
            "(bar_ts, symbol, asset_class, probability, bar, above_bar, ref_price, trade_date) "
            "VALUES (?,?,?,?,?,?,?,?)", rows)
        inserted = conn.total_changes - before
    if inserted:
        logger.debug(f"[signals] recorded {inserted} of {len(rows)} scored symbols")
    return inserted


def mark_executed(conn, symbol: str, bar_ts: str, trade_id: int) -> None:
    """Link a signal row to the BUY trade it produced."""
    with conn:
        conn.execute("UPDATE signals SET executed_trade_id = ? WHERE symbol = ? AND bar_ts = ?",
                     (trade_id, symbol, bar_ts))


def _forward_bars(conn, symbol: str, bar_ts: str, n_bars: int) -> pd.DataFrame:
    """The signal's own bar plus the next n_bars-1 stored bars, UTC-indexed.

    prices_intraday.ts is stored exactly as yfinance served it, which for this
    universe is an ET offset ('...-04:00' / '...-05:00'), while bar_ts is UTC
    ('...+00:00'). A plain string comparison of the two is wrong (15:55-04:00
    sorts BEFORE 19:55+00:00 although they are the same instant), so both
    sides go through SQLite's datetime(), which normalises any offset to UTC.
    """
    df = pd.read_sql_query(
        "SELECT ts, high, low, close FROM prices_intraday "
        "WHERE symbol = ? AND interval = ? AND datetime(ts) >= datetime(?) "
        "ORDER BY datetime(ts) LIMIT ?",
        conn, params=(symbol, INTERVAL, bar_ts, int(n_bars)))
    if df.empty:
        return df
    df['ts'] = pd.to_datetime(df['ts'], utc=True, format='ISO8601')
    return df.set_index('ts')


def _bar_minutes() -> float:
    """Bar length in minutes from INTERVAL: '5m' -> 5, '15m' -> 15, '1h' -> 60."""
    s = INTERVAL.strip().lower()
    return float(s[:-1]) * (60 if s.endswith('h') else 1)


def _session_close(trade_date: str) -> datetime:
    """ET close of the stock session on an ET date (13:00 on early-close days)."""
    hour = 13 if trade_date in EARLY_CLOSE_2026 else 16
    return datetime.strptime(trade_date, '%Y-%m-%d').replace(hour=hour, tzinfo=ET)


def _session_complete(bars: pd.DataFrame, trade_date: str, now: datetime) -> bool:
    """True once a stock signal's session is over AND its closing bar is stored.

    Both conditions matter. `now` past the close says the session can no
    longer resolve the trade. The closing bar (the one that starts one
    interval before the close, 15:55 ET on a regular day) being present says
    the fetch ran after the close, so a feed outage that stopped bars at 15:30
    is not read as "nothing happened until the bell".
    """
    close = _session_close(trade_date)
    if now.astimezone(ET) < close:
        return False
    et = bars.index.tz_convert(ET)
    in_session = et[et.strftime('%Y-%m-%d') == trade_date]
    return (len(in_session) > 0 and
            in_session[-1].to_pydatetime() >= close - timedelta(minutes=_bar_minutes()))


def label_pending(conn, now=None) -> int:
    """Label every signal whose outcome is decided; returns the count labelled.

    Same rule as training (build_target / tune.py dataset()): the class's tuned
    barriers, and for stocks the walk stops at the ET session boundary because
    the executor flattens before the bell. A stock label therefore never
    depends on the next session's bars, so once the session is over (and its
    closing bar is stored) an unresolved stock signal is 0 the same day - the
    16:05 tick labels nearly every stock row from that day. Crypto runs the
    full horizon; a crypto window that runs past the last stored bar stays
    NULL and is retried on the next call.

    `conn` must not be BudgetTracker.conn: this runs on the report's worker
    thread and its commit would land on whatever transaction the fast cycle
    has open on that connection (see the module docstring).
    """
    now = now or datetime.now(timezone.utc)
    pending = conn.execute(
        "SELECT id, symbol, asset_class, bar_ts, trade_date FROM signals "
        "WHERE label IS NULL ORDER BY symbol, bar_ts").fetchall()
    if not pending:
        return 0
    stamp = _now_iso(now)
    labelled = 0
    for row in pending:
        cls = row['asset_class']
        tp, sl, horizon = barriers(cls)
        bars = _forward_bars(conn, row['symbol'], row['bar_ts'], horizon + 1)
        if bars.empty:
            continue                      # nothing stored from the signal's bar on
        if bars.index[0] != pd.Timestamp(row['bar_ts']):
            # The signal's own bar is missing from the store, so bars[0] is a
            # later bar and its close is not the entry price. Never label
            # against the wrong entry; leave it NULL and say so.
            logger.warning(f"[signals] {row['symbol']} {row['bar_ts']}: signal bar not "
                           f"stored (first stored bar {bars.index[0].isoformat()})")
            continue
        if cls == 'crypto':
            session = None
            if len(bars) < horizon + 1:
                continue                  # window not fully stored yet
        else:
            session = bars.index.tz_convert(ET).strftime('%Y-%m-%d')
            if len(bars) < horizon + 1 and not _session_complete(bars, row['trade_date'], now):
                continue                  # the session may still resolve it
        # Fewer than horizon+1 bars reaches here only for a stock whose session
        # is over: the walk then runs to the last stored bar, which is exactly
        # where the session cut would have stopped it anyway.
        label = triple_barrier(bars['high'], bars['low'], bars['close'], tp, sl,
                               min(horizon, len(bars) - 1), session=session).iloc[0]
        if pd.isna(label):
            continue
        with conn:
            conn.execute("UPDATE signals SET label = ?, labeled_at = ? WHERE id = ?",
                         (int(label), stamp, row['id']))
        labelled += 1
    logger.info(f"[signals] labelled {labelled} of {len(pending)} pending signals")
    return labelled
```

Why `LIMIT horizon + 1`: `triple_barrier` labels every bar in the frame it is given (an O(n × horizon) Python loop), so handing it a symbol's full history for each of ~1800 signals would take minutes. With exactly `horizon + 1` rows it computes one label (index 0) and NaNs for the rest.

Why `min(horizon, len(bars) - 1)`: `triple_barrier` only computes `out[i]` for `i < n - horizon`, so with fewer than `horizon + 1` rows it returns NaN for index 0 even when a session boundary sits inside the rows. For a stock whose session is over the walk cannot legitimately go past the last same-session bar anyway, so shrinking the horizon to the rows available gives the identical answer the session cut would give once next-session bars exist — just a day earlier. A closing-bar signal (15:55 ET) gets horizon 0 and label 0: the executor could never have traded it.

- [ ] **Step 4: Run the tests — the log passes, the engine test still fails**

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_signal_log.py -v
```

Expected: 12 passed, 1 failed:

```
E       KeyError: 'bar_ts'
FAILED tests/test_signal_log.py::test_signal_includes_bar_ts - KeyError: 'bar_ts'
1 failed, 12 passed in 1.46s
```

- [ ] **Step 5: Add `bar_ts` to `IntradayEngine.signal()`**

In `/home/gdhughey/hugheylab-trading-bot/src/intraday_engine.py`, inside `signal()` (the method starting `def signal(self, symbol):` at line 507), replace the return block that currently reads

```python
        p_up = float(model.predict_proba(f[FEATURES].iloc[[-1]])[0][1])
        return {
            'symbol': symbol,
            'asset_class': cls,
            'probability': p_up,
            'bar': self.threshold(cls),
            'price': float(h['close'].iloc[-1]),
            'bar_time': h.index[-1].astimezone(ET).strftime('%H:%M ET'),
            'bar_age_min': age_min,
        }
```

with

```python
        p_up = float(model.predict_proba(f[FEATURES].iloc[[-1]])[0][1])
        return {
            'symbol': symbol,
            'asset_class': cls,
            'probability': p_up,
            'bar': self.threshold(cls),
            'price': float(h['close'].iloc[-1]),
            # UTC ISO of the bar scored - the signal log's dedupe key, so the
            # 60 s poll writes one row per 5m bar instead of five.
            'bar_ts': h.index[-1].isoformat(),
            'bar_time': h.index[-1].astimezone(ET).strftime('%H:%M ET'),
            'bar_age_min': age_min,
        }
```

`h.index` is already UTC (`_history()` parses with `utc=True`), so `isoformat()` yields `'...+00:00'`, which is what `record()` stores and `mark_executed()` matches on. Nothing else in `signal()` changes.

- [ ] **Step 6: Run the tests — all green**

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_signal_log.py -v
```

Expected:

```
tests/test_signal_log.py::test_record_writes_one_row_per_symbol_per_bar PASSED
tests/test_signal_log.py::test_record_trade_date_is_et_not_utc PASSED
tests/test_signal_log.py::test_mark_executed_sets_trade_id PASSED
tests/test_signal_log.py::test_label_take_profit_first_is_1 PASSED
tests/test_signal_log.py::test_label_stop_first_is_0 PASSED
tests/test_signal_log.py::test_label_horizon_expires_flat_is_0 PASSED
tests/test_signal_log.py::test_label_session_boundary_stops_stock_but_not_crypto PASSED
tests/test_signal_log.py::test_label_insufficient_bars_stays_null_then_resolves PASSED
tests/test_signal_log.py::test_label_stock_session_end_is_0_same_day PASSED
tests/test_signal_log.py::test_label_stock_session_end_needs_the_closing_bar PASSED
tests/test_signal_log.py::test_label_crypto_insufficient_bars_stays_null_after_the_bell PASSED
tests/test_signal_log.py::test_label_skips_signal_whose_own_bar_is_missing PASSED
tests/test_signal_log.py::test_signal_includes_bar_ts PASSED
13 passed
```

Then run the whole suite to confirm nothing earlier regressed: `dev/ct-test.sh tests -v` — expected: every test passes (the count depends on Tasks 1-4).

- [ ] **Step 7: Commit**

```
cd /home/gdhughey/hugheylab-trading-bot && git add src/signal_log.py src/intraday_engine.py tests/test_signal_log.py && git commit -m "Signal log: record every scored bar, label with triple_barrier at 16:05

One row per (symbol, bar_ts) via INSERT OR IGNORE so the 60 s poll cannot
inflate n; label_pending applies the class's tuned barriers with the ET
session cut for stocks, and labels a stock the same day once its session is
over and the closing bar is stored (a stock label never depends on the next
session's bars). Bars are matched through SQLite datetime() because
prices_intraday.ts carries the ET offset while bar_ts is UTC.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

**Contract additions**
- `src/signal_log.py: _now_iso(now: datetime | None = None) -> str` — UTC ISO seconds, `now` injected by tests.
- `src/signal_log.py: _et_date(ts_iso: str) -> str` — ET calendar date of a tz-aware ISO string; a local copy of Task 4's `budget_tracker._et_date` so Task 5 depends only on Task 3.
- `src/signal_log.py: _forward_bars(conn, symbol: str, bar_ts: str, n_bars: int) -> pd.DataFrame` — the signal's bar plus the next `n_bars-1` stored bars, UTC `DatetimeIndex`, columns `high, low, close`; matches `ts` through SQLite `datetime()`.
- `src/signal_log.py: _bar_minutes() -> float` — bar length in minutes parsed from `INTERVAL` (`'5m' -> 5.0`, `'15m' -> 15.0`, `'1h' -> 60.0`); a local sibling of Task 6's `fast_trader._interval_minutes` (Task 6 depends on Task 5, so Task 5 cannot import it).
- `src/signal_log.py: _session_close(trade_date: str) -> datetime` — the ET close of the stock session on an ET date: 16:00, or 13:00 when the date is in `intraday_engine.EARLY_CLOSE_2026`.
- `src/signal_log.py: _session_complete(bars: pd.DataFrame, trade_date: str, now: datetime) -> bool` — True only when `now` (tz-aware) is at or past `_session_close(trade_date)` AND the latest stored bar dated `trade_date` starts at or after `close - _bar_minutes()` minutes (the closing bar, 15:55 ET on a regular day).
- `label_pending` semantics beyond the contract text: a stock signal with fewer than `horizon + 1` stored bars is still labelled when `_session_complete` holds, by calling `triple_barrier` with horizon `min(horizon, len(bars) - 1)`; a signal whose own bar is not stored (first stored bar ≠ `bar_ts`) is skipped with a WARNING and left NULL; crypto is never labelled short of `horizon + 1` bars. `label_pending` reads `signals.trade_date` (in addition to `id, symbol, asset_class, bar_ts`) and must be called on a connection BudgetTracker does not own.

---

### Task 6: Fast trader — class gate, ref-to-ref barriers, account sizing, daily loss limit, signal log

Depends on Tasks 1 (`src/costs.py`), 2 (`market_state(now)` / `is_trading_day`), 3 (`Database(path, now=)` — the clock kwarg the test fixture pins `account.opened_at` with), 4 (`BudgetTracker` ledger: `size_order`, `log_trade(..., now=)`, `execute_trade` returning the row, `get_positions()` with `entry_ref`, `ensure_day_state`, `get_day_state`, `set_day_flag`, `get_equity`, `get_buying_power`, `get_unsettled`) and 5 (`src/signal_log.py`). Run this task after those five are merged.

**Files:**
- Modify: `/home/gdhughey/hugheylab-trading-bot/src/fast_trader.py` — imports (lines 16-21), config properties (lines 69-81: delete the `max_hold_min` property at 80-81), `_recover_entry_times` docstring (lines 43-50), and a full rewrite of `cycle()` (lines 93-253). The file is replaced whole in Step 8.
- Create: `/home/gdhughey/hugheylab-trading-bot/tests/test_fast_trader.py`

What changes and why:
- Two module-level helpers, `class_gate(cls, metrics)` and `max_hold_min(cls)`, replace the inline EV block and the `FAST_MAX_HOLD_MIN` property. Crypto's 0.6% take-profit cannot beat a 1.2% round trip, so the cost gate refuses it regardless of `FAST_IGNORE_EV`.
- Exits test barriers **reference-to-reference** (`pos['entry_ref']`), never against `avg_price` (which carries the entry spread and would fire the crypto stop on an unchanged quote).
- Entries use `budget.size_order` (dollar-based fraction of equity) instead of `cash / free_slots` with an `int()` cast.
- One quote per held symbol per cycle feeds exits, the `day_state` baseline, the loss-limit check and sizing.
- Daily loss limit: `-DAILY_LOSS_LIMIT_PCT%` of `day_state.start_equity` latches `loss_tripped_at`; entries skip with `'daily loss limit'`; exits still run; persists across restarts on the same ET date.
- Every cycle writes `scan_all()` output to the `signals` table and marks the traded signal with its BUY trade id. Both `signal_log.record` and `signal_log.mark_executed` run on `self.budget.conn` from the cycle's worker thread — the same thread that runs `BudgetTracker._txn`, and the connection is autocommit so `signal_log`'s `with conn:` blocks are safe there. This is the expectation Task 5's note documents; only `label_pending` (called from the Discord event-loop thread in Task 9) must use a connection other than `budget.conn`.
- `cycle(now=None)` takes an injectable UTC clock so tests never touch the wall clock. The test fixture also pins `account.opened_at` (via `Database(path, now=)`) because `get_unsettled` / `get_buying_power` filter `created_at >= opened_at`: an unpinned `opened_at` would be the wall clock, which passes the fixed `NOW` stamp on every trade only while the run happens before 2026-09-14 10:30 ET.

- [ ] **Step 1: Write the failing tests for `class_gate` and `max_hold_min`**

Create `/home/gdhughey/hugheylab-trading-bot/tests/test_fast_trader.py` with exactly this content (the cycle tests are appended in Step 6):

```python
"""FastTrader tests: class gate, max hold, ref-to-ref barriers, daily loss
limit, signal-log wiring and skip reasons.

Every test builds a file DB under tmp_path and passes `now=` explicitly - to
cycle(), to the ledger calls, AND to Database() so account.opened_at is a
fixed date before every trade stamp (the ledger filters created_at >=
opened_at; an opened_at taken from the wall clock would silently exclude the
trades once the real date passes NOW). The engine is a fake (no model, no
yfinance, no sklearn), the BudgetTracker is real.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src import fast_trader, intraday_engine
from src.budget_tracker import BudgetTracker
from src.database import Database
from src.fast_trader import FastTrader, class_gate, max_hold_min
from src.intraday_engine import ET, asset_class

# The paper account opened two weeks before any trade in these tests, so the
# created_at >= account.opened_at filters in get_unsettled/get_buying_power
# always include the trades below regardless of the machine's clock.
OPENED_AT = datetime(2026, 9, 1, tzinfo=timezone.utc)

# Monday 2026-09-14 10:30 ET: a regular session, well clear of the bell.
NOW = datetime(2026, 9, 14, 14, 30, tzinfo=timezone.utc)
SATURDAY = datetime(2026, 9, 12, 14, 30, tzinfo=timezone.utc)
NEAR_BELL = datetime(2026, 9, 14, 19, 55, tzinfo=timezone.utc)      # 15:55 ET
NEXT_DAY = NOW + timedelta(days=1)                                  # Tue 10:30 ET
BAR_TS = NOW.isoformat()

# data/ is excluded from the test tarball, so there is no tuned.json on the
# container; pin the barriers the spec's numbers assume.
TUNED = {
    'stock': {'take_profit': 0.010, 'stop_loss': 0.006, 'horizon': 24},
    'crypto': {'take_profit': 0.006, 'stop_loss': 0.004, 'horizon': 24},
}

# Cost/account defaults from the contract, pinned so the container's shell
# environment cannot change the arithmetic.
ENV = {
    'STARTING_CASH': '500', 'ACCOUNT_TYPE': 'cash',
    'STOCK_SLIPPAGE_BPS': '5', 'CRYPTO_SPREAD_BPS': '60',
    'SEC_FEE_RATE': '0.0000206', 'FINRA_TAF_PER_SHARE': '0.000195',
    'FINRA_TAF_CAP': '9.79', 'MIN_ORDER_USD': '1',
    'DAILY_LOSS_LIMIT_PCT': '3', 'MIN_EV_TO_TRADE': '0.003',
    'FAST_MAX_POSITIONS': '3', 'FAST_EOD_FLATTEN_MIN': '10',
    'FAST_COOLDOWN_MIN': '15',
}


@pytest.fixture(autouse=True)
def paper_env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('FAST_IGNORE_EV', raising=False)
    monkeypatch.setattr(intraday_engine, 'TUNED', TUNED)
    monkeypatch.setattr(fast_trader, 'INTERVAL', '5m')


class FakeEngine:
    """Stands in for IntradayEngine: quotes and probabilities are dicts the
    test mutates between cycles. A symbol with no quote is not scored."""

    def __init__(self, symbols, quotes, metrics=None, bar=0.5):
        self.symbols = list(symbols)
        self.quotes = dict(quotes)           # symbol -> ref price (bar close)
        self.probs = {}                      # symbol -> probability (default 0.9)
        self.bar = bar
        self.bar_ts = BAR_TS
        self.metrics = metrics if metrics is not None else {
            'stock': {'ev': 0.01, 'precision': 0.5, 'breakeven': 0.4, 'bar': bar},
            'crypto': {'ev': 0.01, 'precision': 0.5, 'breakeven': 0.4, 'bar': bar},
        }

    def fetch(self, *args, **kwargs):
        return 0

    def threshold(self, cls='stock'):
        return self.bar

    def signal(self, symbol):
        if symbol not in self.quotes:
            return None
        return {'symbol': symbol, 'asset_class': asset_class(symbol),
                'probability': self.probs.get(symbol, 0.9), 'bar': self.bar,
                'price': float(self.quotes[symbol]), 'bar_ts': self.bar_ts}

    def scan_all(self):
        out = []
        for sym in self.symbols:
            s = self.signal(sym)
            if s:
                s['above_bar'] = s['probability'] >= s['bar']
                s['margin'] = s['probability'] - s['bar']
                out.append(s)
        out.sort(key=lambda r: r['margin'], reverse=True)
        return out


def make_budget(tmp_path, monkeypatch, starting_cash='500'):
    # STARTING_CASH must be in the environment before Database() seeds the
    # account row; the file path is shared so a second BudgetTracker on the
    # same tmp_path sees the same ledger (restart tests).
    #
    # now=OPENED_AT pins account.opened_at. Without it the seed uses the wall
    # clock, and get_unsettled/get_buying_power (created_at >= opened_at)
    # would drop every NOW-stamped trade as soon as the real date passes
    # 2026-09-14 - the tests would pass today and fail next week.
    monkeypatch.setenv('STARTING_CASH', starting_cash)
    path = str(tmp_path / 'bot.db')
    Database(path, now=OPENED_AT)
    return BudgetTracker(path)


def make_trader(engine, budget):
    return FastTrader(engine, budget, quote_fn=engine.quotes.get)


def open_position(budget, symbol, ref, qty, now=NOW):
    """Open a position straight through the ledger (crypto entries are cost-
    gated, so the cycle cannot open them for us)."""
    tid = budget.log_trade(symbol, 'BUY', ref, qty, probability=0.9, now=now)
    budget.execute_trade(tid, now=now)
    return tid


def trade_row(budget, trade_id):
    return budget.conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()


def signal_count(budget):
    return budget.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]


# --- max hold + class gate --------------------------------------------------

def test_max_hold_min_is_horizon_times_interval():
    assert max_hold_min('stock') == 120
    assert max_hold_min('crypto') == 120


def test_class_gate_crypto_refused_by_round_trip_cost(monkeypatch):
    monkeypatch.setenv('FAST_IGNORE_EV', '1')
    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '60')
    ok, text = class_gate('crypto', {'ev': 0.01})
    assert ok is False
    assert text == "blocked: take-profit 0.60% is below the 1.20% round-trip cost"


def test_class_gate_crypto_allowed_on_cheaper_venue(monkeypatch):
    monkeypatch.setenv('FAST_IGNORE_EV', '1')
    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '10')
    ok, text = class_gate('crypto', {'ev': 0.01})
    assert ok is True
    assert text == "Trading. Edge clears the 0.30% cost floor"


def test_class_gate_stock_allowed_at_defaults():
    assert class_gate('stock', {'ev': 0.01}) == (
        True, "Trading. Edge clears the 0.30% cost floor")


@pytest.mark.parametrize('ev, fragment', [
    (0.0006, 'smaller than the 0.30%'),
    (-0.0003, 'loses money'),
])
def test_class_gate_ev_block_when_flag_unset(ev, fragment):
    ok, text = class_gate('stock', {'ev': ev})
    assert ok is False
    assert text.startswith('Not trading.')
    assert fragment in text


def test_class_gate_ignore_ev_text(monkeypatch):
    monkeypatch.setenv('FAST_IGNORE_EV', '1')
    assert class_gate('stock', {'ev': -0.0003}) == (
        True, "trading on paper despite EV -0.030% below the 0.30% floor (FAST_IGNORE_EV on)")


def test_class_gate_without_metrics_is_ev_blocked():
    ok, text = class_gate('stock', None)
    assert ok is False
    assert text.startswith('Not trading.')
```

- [ ] **Step 2: Run the tests, confirm they fail at import**

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_fast_trader.py -v
```

Expected: collection error, output contains
`ImportError: cannot import name 'class_gate' from 'src.fast_trader'`.

- [ ] **Step 3: Add `BARRIER_EPS`, `_interval_minutes`, `max_hold_min`, `class_gate`; delete the `max_hold_min` property**

In `/home/gdhughey/hugheylab-trading-bot/src/fast_trader.py`, replace lines 16-31 (the imports through `_cfg`) with:

```python
import os
import logging
from datetime import datetime, timedelta, timezone

from src import costs, signal_log
from src.intraday_engine import (market_state, minutes_to_close, ET, INTERVAL,
                                 is_crypto, asset_class, barriers)

logger = logging.getLogger(__name__)

# Barrier comparisons carry a tolerance because decimal quotes are not exact in
# binary: (99.40 - 100) / 100 evaluates to -0.005999999999999943, which a bare
# `<= -0.006` would NOT treat as a stop hit even though the quote sits exactly
# on the barrier. 1e-9 is far below any tick and far above float error.
BARRIER_EPS = 1e-9


def _cfg(name, default, cast=float):
    try:
        return cast(os.getenv(name, default))
    except (TypeError, ValueError):
        return cast(default)


def _interval_minutes(interval=None) -> float:
    """Bar length in minutes for a yfinance interval string: '5m' -> 5, '1h' -> 60."""
    s = str(interval or INTERVAL).strip().lower()
    try:
        return float(s[:-1]) * 60 if s.endswith('h') else float(s.rstrip('m'))
    except ValueError:
        return 5.0


def max_hold_min(cls: str) -> float:
    """Max hold per class = the label horizon in minutes (24 bars x 5m = 120).

    Derived, not configured: the model was trained to call a move within this
    window, so holding longer is a bet it never made.
    """
    return barriers(cls)[2] * _interval_minutes()


def class_gate(cls: str, metrics: dict | None) -> tuple[bool, str]:
    """(tradeable, text) for one asset class. `metrics` is that class's entry
    from engine.metrics (None when there is no model; treated as EV 0).

    Evaluated in this order:
      1. cost gate - never bypassed: with a take-profit at or below the
         round-trip cost no trade can be net positive;
      2. EV gate - the backtest's verdict, blocks unless FAST_IGNORE_EV;
      3. FAST_IGNORE_EV text - trading on paper despite a sub-floor EV;
      4. ok text.
    `ev` in metrics is a fraction (0.0006 = 0.06%), printed as a percentage.
    """
    tp, _sl, _horizon = barriers(cls)
    cost = costs.round_trip_cost(cls)
    if tp <= cost:
        return False, (f"blocked: take-profit {tp:.2%} is below the "
                       f"{cost:.2%} round-trip cost")
    floor = _cfg('MIN_EV_TO_TRADE', 0.003)
    ev = float((metrics or {}).get('ev', 0.0))
    ignore_ev = os.getenv('FAST_IGNORE_EV', '0') in ('1', 'true', 'yes')
    if ev < floor and not ignore_ev:
        if ev > 0:
            return False, (f"Not trading. Edge of {ev * 100:+.3f}% per trade is "
                           f"real but smaller than the {floor:.2%} it costs to get "
                           f"in and out, so it would lose money after fees.")
        return False, (f"Not trading. Model loses money ({ev * 100:+.3f}% per "
                       f"trade) on these settings.")
    if ev < floor:
        return True, (f"trading on paper despite EV {ev * 100:+.3f}% below the "
                      f"{floor:.2%} floor (FAST_IGNORE_EV on)")
    return True, f"Trading. Edge clears the {floor:.2%} cost floor"
```

Then delete the two lines of the `max_hold_min` property (currently lines 80-81):

```python
    @property
    def max_hold_min(self): return _cfg('FAST_MAX_HOLD_MIN', 120)
```

Leave the rest of the file (including the old `cycle()`) untouched for now; Step 8 replaces it.

- [ ] **Step 4: Run the tests, confirm the gate tests pass**

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_fast_trader.py -v
```

Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
cd /home/gdhughey/hugheylab-trading-bot && git add src/fast_trader.py tests/test_fast_trader.py && git commit -m "fast_trader: class_gate and max_hold_min derived from the label horizon

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 6: Write the failing cycle tests**

Append the following to the end of `/home/gdhughey/hugheylab-trading-bot/tests/test_fast_trader.py`:

```python
# --- barriers ref-to-ref ---------------------------------------------------

def test_crypto_barriers_measured_ref_to_ref(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['BTC-USD'], {'BTC-USD': 100.0})
    open_position(budget, 'BTC-USD', 100.0, 1.0)
    pos = budget.get_positions()[0]
    assert pos['avg_price'] == pytest.approx(100.60)    # fill carries the 60 bps
    assert pos['entry_ref'] == pytest.approx(100.0)
    trader = make_trader(engine, budget)

    # An unchanged quote is NOT a stop: avg_price sits 0.6% above ref, but the
    # barrier is measured against the ref the position was entered at.
    assert trader.cycle(now=NOW)['exits'] == []
    engine.quotes['BTC-USD'] = 99.61
    assert trader.cycle(now=NOW)['exits'] == []

    engine.quotes['BTC-USD'] = 99.60
    s = trader.cycle(now=NOW)
    assert len(s['exits']) == 1
    x = s['exits'][0]
    assert x['symbol'] == 'BTC-USD'
    assert x['exit_reason'] == 'sl'
    assert x['reason'].startswith('stop loss')
    assert x['ref_price'] == 99.60
    assert x['price'] == pytest.approx(99.60 * 0.994)
    assert x['pct'] == pytest.approx(-0.004)
    assert x['fees'] == 0
    assert x['pnl'] / x['shares'] == pytest.approx(99.60 * 0.994 - 100.60)
    row = trade_row(budget, x['trade_id'])
    assert x['pnl'] == pytest.approx(row['realized_pnl'])
    assert row['exit_reason'] == 'sl'
    assert budget.get_positions() == []


def test_stock_stop_ref_to_ref_then_cooldown(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL'], {'AAPL': 100.0})
    open_position(budget, 'AAPL', 100.0, 2.0)
    assert budget.get_positions()[0]['avg_price'] == pytest.approx(100.05)
    trader = make_trader(engine, budget)

    engine.quotes['AAPL'] = 99.45
    assert trader.cycle(now=NOW)['exits'] == []
    engine.quotes['AAPL'] = 99.40
    s = trader.cycle(now=NOW)
    x = s['exits'][0]
    assert x['exit_reason'] == 'sl'
    assert x['price'] == pytest.approx(99.40 * 0.9995)
    assert x['pct'] == pytest.approx(-0.006)
    assert x['gross'] == pytest.approx(x['pnl'] + x['fees'])
    assert x['fees'] > 0

    # The name it just sold is a candidate again but must cool down first...
    s2 = trader.cycle(now=NOW)
    assert ('AAPL', 'cooling down') in s2['skipped']
    assert s2['entries'] == []
    # ...and re-enters once FAST_COOLDOWN_MIN has passed.
    s3 = trader.cycle(now=NOW + timedelta(minutes=16))
    assert [e['symbol'] for e in s3['entries']] == ['AAPL']


def test_stock_take_profit_ref_to_ref(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL'], {'AAPL': 100.0})
    open_position(budget, 'AAPL', 100.0, 2.0)
    trader = make_trader(engine, budget)

    engine.quotes['AAPL'] = 100.95
    assert trader.cycle(now=NOW)['exits'] == []
    engine.quotes['AAPL'] = 101.00
    s = trader.cycle(now=NOW)
    x = s['exits'][0]
    assert x['exit_reason'] == 'tp'
    assert x['reason'].startswith('take profit')
    assert x['price'] == pytest.approx(101.00 * 0.9995)
    assert x['pct'] == pytest.approx(0.01)
    assert trade_row(budget, x['trade_id'])['exit_reason'] == 'tp'


def test_flat_ref_survives_max_hold_then_times_out(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL'], {'AAPL': 100.0})
    open_position(budget, 'AAPL', 100.0, 1.0, now=NOW)
    # Constructed after the BUY: the entry time is recovered from the ledger,
    # exactly as after a service restart.
    trader = make_trader(engine, budget)
    assert trader.entry_time['AAPL'] == NOW.astimezone(ET)
    assert max_hold_min('stock') == 120

    for minutes in (30, 60, 90, 120):
        assert trader.cycle(now=NOW + timedelta(minutes=minutes))['exits'] == []
    s = trader.cycle(now=NOW + timedelta(minutes=121))
    x = s['exits'][0]
    assert x['exit_reason'] == 'timeout'
    assert x['reason'] == 'held 120 min without hitting a target'
    assert x['pct'] == pytest.approx(0.0)
    assert trade_row(budget, x['trade_id'])['exit_reason'] == 'timeout'
    assert 'AAPL' not in trader.entry_time


# --- daily loss limit -------------------------------------------------------

def test_daily_loss_limit_trips_blocks_entries_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv('FAST_MAX_POSITIONS', '2')
    budget = make_budget(tmp_path, monkeypatch)
    # NVDA has no quote yet, so cycle 1 scores only AAPL and MSFT.
    engine = FakeEngine(['AAPL', 'MSFT', 'NVDA'], {'AAPL': 100.0, 'MSFT': 100.0})
    trader = make_trader(engine, budget)

    s1 = trader.cycle(now=NOW)
    assert sorted(e['symbol'] for e in s1['entries']) == ['AAPL', 'MSFT']
    assert s1['loss_tripped'] is False and s1['loss_announce'] is False
    assert budget.get_day_state('2026-09-14')['start_equity'] == pytest.approx(500.0)

    # AAPL collapses 10%: the stop fires and the realised loss (~$25 on a
    # $250 slot) takes equity below 97% of start_equity.
    engine.quotes['AAPL'] = 90.0
    engine.quotes['NVDA'] = 100.0
    s2 = trader.cycle(now=NOW + timedelta(minutes=5))
    assert [x['exit_reason'] for x in s2['exits']] == ['sl']
    assert s2['equity'] <= 485.0
    assert s2['loss_tripped'] is True and s2['loss_announce'] is True
    assert s2['entries'] == []
    assert ('NVDA', 'daily loss limit') in s2['skipped']
    assert budget.get_day_state('2026-09-14')['loss_tripped_at'] is not None

    # Exits still run while tripped; only the tripping cycle announces.
    engine.quotes['MSFT'] = 101.0
    s3 = trader.cycle(now=NOW + timedelta(minutes=10))
    assert [x['exit_reason'] for x in s3['exits']] == ['tp']
    assert s3['loss_tripped'] is True and s3['loss_announce'] is False
    assert s3['entries'] == []
    assert ('NVDA', 'daily loss limit') in s3['skipped']

    # Restart on the same date: fresh objects on the same file stay blocked,
    # do not re-announce, and keep the original start_equity.
    budget2 = BudgetTracker(str(tmp_path / 'bot.db'))
    trader2 = make_trader(engine, budget2)
    s4 = trader2.cycle(now=NOW + timedelta(minutes=15))
    assert s4['loss_tripped'] is True and s4['loss_announce'] is False
    assert s4['entries'] == []
    assert ('NVDA', 'daily loss limit') in s4['skipped']
    assert budget2.get_day_state('2026-09-14')['start_equity'] == pytest.approx(500.0)

    # Next ET date: a new baseline, entries allowed again.
    s5 = trader2.cycle(now=NEXT_DAY)
    assert s5['loss_tripped'] is False and s5['loss_announce'] is False
    assert len(s5['entries']) == 2
    assert 'daily loss limit' not in [reason for _, reason in s5['skipped']]
    assert budget2.get_day_state('2026-09-15')['start_equity'] == pytest.approx(s4['equity'])


# --- signal log + summary shape --------------------------------------------

def test_signals_recorded_each_cycle_and_marked_on_entry(tmp_path, monkeypatch):
    monkeypatch.setenv('FAST_MAX_POSITIONS', '1')
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL', 'MSFT'], {'AAPL': 100.0, 'MSFT': 50.0})
    engine.probs['MSFT'] = 0.4                      # scored, but below the 0.5 bar
    trader = make_trader(engine, budget)

    s = trader.cycle(now=NOW)
    for key in ('state', 'desc', 'exits', 'entries', 'skipped', 'candidates', 'ts',
                'stocks_open', 'crypto', 'rows', 'minutes_to_close', 'bars', 'bar',
                'gates', 'blocked_classes', 'loss_tripped', 'loss_announce',
                'equity', 'buying_power', 'unsettled'):
        assert key in s, key
    assert [e['symbol'] for e in s['entries']] == ['AAPL']
    e = s['entries'][0]
    assert e['ref_price'] == 100.0
    assert e['price'] == pytest.approx(100.05)
    assert e['shares'] == pytest.approx(round(500 / 100.05, 6))
    assert e['cost'] == pytest.approx(e['shares'] * 100.05)
    assert e['fees'] == 0
    assert e['probability'] == 0.9
    assert trade_row(budget, e['trade_id'])['entry_probability'] == 0.9
    assert s['buying_power'] < 1.0                  # the single slot took everything

    rows = {r['symbol']: r for r in budget.conn.execute(
        "SELECT symbol, above_bar, executed_trade_id FROM signals")}
    assert set(rows) == {'AAPL', 'MSFT'}            # every scored symbol, not just candidates
    assert rows['AAPL']['above_bar'] == 1 and rows['MSFT']['above_bar'] == 0
    assert rows['AAPL']['executed_trade_id'] == e['trade_id']
    assert rows['MSFT']['executed_trade_id'] is None

    # The same 5m bar polled again adds no rows and keeps the executed id.
    trader.cycle(now=NOW + timedelta(minutes=1))
    assert signal_count(budget) == 2
    assert budget.conn.execute(
        "SELECT executed_trade_id FROM signals WHERE symbol = 'AAPL'"
    ).fetchone()[0] == e['trade_id']

    # A new bar adds one fresh row per symbol.
    engine.bar_ts = (NOW + timedelta(minutes=5)).isoformat()
    trader.cycle(now=NOW + timedelta(minutes=5))
    assert signal_count(budget) == 4


# --- skip reasons -----------------------------------------------------------

def test_skip_market_closed_and_class_gated(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL', 'BTC-USD'], {'AAPL': 100.0, 'BTC-USD': 100.0})
    trader = make_trader(engine, budget)

    s = trader.cycle(now=SATURDAY)
    assert s['state'] == 'closed' and s['stocks_open'] is False
    assert s['gates']['stock'] == (True, "Trading. Edge clears the 0.30% cost floor")
    assert s['gates']['crypto'] == (
        False, "blocked: take-profit 0.60% is below the 1.20% round-trip cost")
    assert s['blocked_classes'] == ['crypto']
    assert s['entries'] == []
    assert ('AAPL', 'market closed') in s['skipped']
    assert ('BTC-USD', 'class gated: blocked: take-profit 0.60% is below the '
                       '1.20% round-trip cost') in s['skipped']
    assert signal_count(budget) == 2                # crypto keeps the cycle (and the log) alive


def test_eod_flatten_and_too_close_to_the_bell(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL'], {'AAPL': 100.0})
    open_position(budget, 'AAPL', 100.0, 1.0)
    trader = make_trader(engine, budget)

    s = trader.cycle(now=NEAR_BELL)
    assert s['state'] == 'open'
    assert [x['exit_reason'] for x in s['exits']] == ['eod']
    assert s['exits'][0]['reason'] == 'end of day (5 min to close)'
    assert s['entries'] == []
    assert ('AAPL', 'too close to the bell') in s['skipped']


def test_skip_below_min_order(tmp_path, monkeypatch):
    # $2 / 3 slots = $0.67 per slot, under MIN_ORDER_USD; buying power ($2)
    # is not the binding constraint, the slot size is.
    budget = make_budget(tmp_path, monkeypatch, starting_cash='2')
    engine = FakeEngine(['AAPL'], {'AAPL': 100.0})
    trader = make_trader(engine, budget)

    s = trader.cycle(now=NOW)
    assert s['entries'] == []
    assert s['skipped'] == [('AAPL', 'below min order')]


def test_skip_insufficient_buying_power_while_proceeds_unsettled(tmp_path, monkeypatch):
    monkeypatch.setenv('FAST_MAX_POSITIONS', '1')
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL', 'MSFT'], {'AAPL': 100.0})
    trader = make_trader(engine, budget)

    s1 = trader.cycle(now=NOW)
    assert [e['symbol'] for e in s1['entries']] == ['AAPL']

    # AAPL takes profit; in a cash account the proceeds are unsettled until
    # the next trading day's open, so the freed slot cannot be refilled.
    # (get_unsettled filters created_at >= account.opened_at, which
    # make_budget pinned to 2026-09-01, so the SELL stamped NOW is counted
    # whatever the real date is.)
    engine.quotes['AAPL'] = 101.0
    engine.quotes['MSFT'] = 50.0
    s2 = trader.cycle(now=NOW + timedelta(minutes=5))
    assert [x['exit_reason'] for x in s2['exits']] == ['tp']
    assert s2['unsettled'] > 0
    assert s2['buying_power'] < 1.0
    assert s2['entries'] == []
    assert ('MSFT', 'insufficient buying power') in s2['skipped']


def test_closed_market_without_crypto_returns_early(tmp_path, monkeypatch):
    budget = make_budget(tmp_path, monkeypatch)
    engine = FakeEngine(['AAPL'], {'AAPL': 100.0})
    trader = make_trader(engine, budget)

    s = trader.cycle(now=SATURDAY)
    assert s['entries'] == [] and s['exits'] == [] and s['skipped'] == []
    assert 'not trading' in s['note']
    assert s['gates']['stock'][0] is True
    assert s['equity'] == pytest.approx(500.0)
    assert s['buying_power'] == pytest.approx(500.0)
    assert s['unsettled'] == 0
    assert signal_count(budget) == 0
    assert budget.get_day_state('2026-09-12') is None
```

- [ ] **Step 7: Run the tests, confirm the cycle tests fail**

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_fast_trader.py -v
```

Expected: `11 failed, 8 passed`; every failure reads
`TypeError: FastTrader.cycle() got an unexpected keyword argument 'now'`.

- [ ] **Step 8: Rewrite `FastTrader` (cycle, summary, entries/exits)**

Replace the ENTIRE contents of `/home/gdhughey/hugheylab-trading-bot/src/fast_trader.py` with:

```python
#!/usr/bin/env python3
"""
Intraday auto-trading loop.

Runs every FAST_POLL_SECONDS during regular market hours. Unlike the daily loop
this one manages exits, because an intraday entry without an exit rule is just a
buy-and-hold with extra steps.

Order of operations each cycle matters: refresh prices, then EXIT before ENTER.
Exiting first frees capital and position slots in the same cycle, and means a
stop-loss is never delayed by an unrelated entry.

Still paper. Every "trade" is a row in SQLite; no broker is connected. The
ledger (BudgetTracker) models a cash account: fills carry slippage/spread,
stock sells settle T+1, and every entry is a fraction of equity.
"""

import os
import logging
from datetime import datetime, timedelta, timezone

from src import costs, signal_log
from src.intraday_engine import (market_state, minutes_to_close, ET, INTERVAL,
                                 is_crypto, asset_class, barriers)

logger = logging.getLogger(__name__)

# Barrier comparisons carry a tolerance because decimal quotes are not exact in
# binary: (99.40 - 100) / 100 evaluates to -0.005999999999999943, which a bare
# `<= -0.006` would NOT treat as a stop hit even though the quote sits exactly
# on the barrier. 1e-9 is far below any tick and far above float error.
BARRIER_EPS = 1e-9


def _cfg(name, default, cast=float):
    try:
        return cast(os.getenv(name, default))
    except (TypeError, ValueError):
        return cast(default)


def _interval_minutes(interval=None) -> float:
    """Bar length in minutes for a yfinance interval string: '5m' -> 5, '1h' -> 60."""
    s = str(interval or INTERVAL).strip().lower()
    try:
        return float(s[:-1]) * 60 if s.endswith('h') else float(s.rstrip('m'))
    except ValueError:
        return 5.0


def max_hold_min(cls: str) -> float:
    """Max hold per class = the label horizon in minutes (24 bars x 5m = 120).

    Derived, not configured: the model was trained to call a move within this
    window, so holding longer is a bet it never made.
    """
    return barriers(cls)[2] * _interval_minutes()


def class_gate(cls: str, metrics: dict | None) -> tuple[bool, str]:
    """(tradeable, text) for one asset class. `metrics` is that class's entry
    from engine.metrics (None when there is no model; treated as EV 0).

    Evaluated in this order:
      1. cost gate - never bypassed: with a take-profit at or below the
         round-trip cost no trade can be net positive;
      2. EV gate - the backtest's verdict, blocks unless FAST_IGNORE_EV;
      3. FAST_IGNORE_EV text - trading on paper despite a sub-floor EV;
      4. ok text.
    `ev` in metrics is a fraction (0.0006 = 0.06%), printed as a percentage.
    """
    tp, _sl, _horizon = barriers(cls)
    cost = costs.round_trip_cost(cls)
    if tp <= cost:
        return False, (f"blocked: take-profit {tp:.2%} is below the "
                       f"{cost:.2%} round-trip cost")
    floor = _cfg('MIN_EV_TO_TRADE', 0.003)
    ev = float((metrics or {}).get('ev', 0.0))
    ignore_ev = os.getenv('FAST_IGNORE_EV', '0') in ('1', 'true', 'yes')
    if ev < floor and not ignore_ev:
        if ev > 0:
            return False, (f"Not trading. Edge of {ev * 100:+.3f}% per trade is "
                           f"real but smaller than the {floor:.2%} it costs to get "
                           f"in and out, so it would lose money after fees.")
        return False, (f"Not trading. Model loses money ({ev * 100:+.3f}% per "
                       f"trade) on these settings.")
    if ev < floor:
        return True, (f"trading on paper despite EV {ev * 100:+.3f}% below the "
                      f"{floor:.2%} floor (FAST_IGNORE_EV on)")
    return True, f"Trading. Edge clears the {floor:.2%} cost floor"


class FastTrader:
    def __init__(self, engine, budget_tracker, quote_fn=None):
        self.engine = engine
        self.budget = budget_tracker
        self.quote_fn = quote_fn          # live price, falls back to bar close
        self.cooldown = {}                # symbol -> datetime it may be re-entered
        self.entry_time = {}              # symbol -> when we opened it
        self.last_summary = {}
        self._recover_entry_times()

    def _recover_entry_times(self):
        """Rebuild entry times from the ledger after a restart.

        These live only in memory, so any position opened before a restart had
        no entry time and the max-hold exit could never fire for it - the
        position would sit indefinitely (crypto has no end-of-day backstop).
        The service restarted four times in two days, so this was live.

        Deliberately NOT filtered by account.opened_at: a held position keeps
        its clock whichever account it was opened under.
        """
        try:
            rows = self.budget.conn.execute(
                "SELECT symbol, MAX(created_at) AS opened FROM trades "
                "WHERE status = 'EXECUTED' AND side = 'BUY' GROUP BY symbol").fetchall()
            held = {p['symbol'] for p in self.budget.get_positions()}
            for r in rows:
                if r['symbol'] not in held or not r['opened']:
                    continue
                ts = datetime.fromisoformat(r['opened'])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                self.entry_time[r['symbol']] = ts.astimezone(ET)
            if self.entry_time:
                logger.info(f"Recovered entry times for {len(self.entry_time)} "
                            f"open position(s) from the ledger")
        except Exception as e:
            logger.warning(f"Could not recover entry times: {e}")

    # --- config ----------------------------------------------------------
    @property
    def max_positions(self): return int(_cfg('FAST_MAX_POSITIONS', 3, int))
    @property
    def stop_loss(self): return _cfg('FAST_STOP_LOSS', 0.005)
    @property
    def take_profit(self): return _cfg('FAST_TAKE_PROFIT', 0.008)
    @property
    def eod_flatten_min(self): return _cfg('FAST_EOD_FLATTEN_MIN', 10)
    @property
    def cooldown_min(self): return _cfg('FAST_COOLDOWN_MIN', 15)

    def _price(self, symbol, fallback):
        if self.quote_fn:
            try:
                p = self.quote_fn(symbol)
                if p:
                    return float(p)
            except Exception:
                pass
        return fallback

    def _account_fields(self, prices, now):
        """Equity / buying power / unsettled for the cycle summary.

        Always derived from the ledger, never cached, so a restart cannot
        disagree with the DB.
        """
        return {'equity': self.budget.get_equity(prices),
                'buying_power': self.budget.get_buying_power(now=now),
                'unsettled': self.budget.get_unsettled(now=now)}

    # --- the cycle -------------------------------------------------------

    def cycle(self, now=None):
        """One pass. Returns a dict describing what happened (for reporting).

        Synchronous by design - the caller runs it in a worker thread, because
        everything in here (yfinance, sklearn, sqlite) blocks. `now` (UTC,
        tz-aware) is injectable so tests never depend on the wall clock.

        Order: market state -> gates -> fetch -> one quote per held symbol ->
        day_state baseline -> exits -> loss-limit check -> scan_all + signal
        log -> entries. Exits run before the loss check so a stop that trips
        the limit is booked in the same cycle; the check runs before entries
        so a tripped limit blocks them immediately.
        """
        now = now or datetime.now(timezone.utc)
        now_et = now.astimezone(ET)
        today_et = now_et.strftime('%Y-%m-%d')
        state, desc = market_state(now)
        stocks_open = state == 'open'
        crypto_syms = [s for s in self.engine.symbols if is_crypto(s)]
        metrics = self.engine.metrics or {}
        # One verdict per class in the universe. A gated class still gets its
        # exits run; the gate only blocks entries.
        gates = {cls: class_gate(cls, metrics.get(cls))
                 for cls in sorted({asset_class(s) for s in self.engine.symbols})}
        summary = {'state': state, 'desc': desc, 'exits': [], 'entries': [],
                   'skipped': [], 'candidates': [], 'ts': now_et,
                   'stocks_open': stocks_open, 'crypto': len(crypto_syms),
                   'gates': gates,
                   'blocked_classes': [c for c, (ok, _) in gates.items() if not ok],
                   'loss_tripped': False, 'loss_announce': False}

        # Crypto never closes, so an outside-hours cycle is still a working
        # cycle whenever the universe holds any coins.
        if not stocks_open and not crypto_syms:
            summary['note'] = f"Market {desc}, no crypto in universe - not trading."
            summary.update(self._account_fields({}, now))
            self.last_summary = summary
            return summary

        rows = self.engine.fetch()
        summary['rows'] = rows
        to_close = minutes_to_close(now)
        summary['minutes_to_close'] = to_close

        # One quote per held symbol. The same dict prices exits, the day_state
        # baseline, the loss-limit check and entry sizing - no second quote.
        held = {p['symbol']: p for p in self.budget.get_positions()}
        prices = {}
        for sym, pos in held.items():
            sig = self.engine.signal(sym)
            prices[sym] = self._price(sym, sig['price'] if sig else pos['avg_price'])

        # INSERT OR IGNORE: a same-date restart keeps the original baseline,
        # so the loss limit cannot be reset by bouncing the service.
        day = self.budget.ensure_day_state(today_et, self.budget.get_equity(prices))

        # ---- EXITS first: frees cash and slots within this same cycle ----
        for sym, pos in held.items():
            # Exits ignore the class gate and the loss limit: an open position
            # must always be closeable, otherwise a newly-gated class would
            # strand it.
            if not (stocks_open or is_crypto(sym)):
                continue
            ref = prices[sym]
            # Barriers are measured reference-to-reference, the same move the
            # labels use. avg_price carries the entry spread/slippage, so
            # measuring against it would fire the 0.4% crypto stop on an
            # unchanged quote (avg sits 0.6% above ref at entry).
            entry_ref = pos['entry_ref']
            change = (ref - entry_ref) / entry_ref if entry_ref else 0.0

            # Exits must use the SAME barriers the model was trained on, and
            # those differ by asset class.
            cls = asset_class(sym)
            tp, sl, _ = barriers(cls)
            reason = exit_reason = None

            # Crypto has no close to flatten into - holding it overnight is
            # normal, so the EOD rule applies to stocks only.
            if not is_crypto(sym) and to_close <= self.eod_flatten_min:
                reason, exit_reason = f"end of day ({to_close:.0f} min to close)", 'eod'
            elif change <= -sl + BARRIER_EPS:
                reason, exit_reason = f"stop loss {change:+.2%}", 'sl'
            elif change >= tp - BARRIER_EPS:
                reason, exit_reason = f"take profit {change:+.2%}", 'tp'
            else:
                opened = self.entry_time.get(sym)
                hold = max_hold_min(cls)
                if opened and (now_et - opened).total_seconds() / 60 > hold:
                    reason, exit_reason = (f"held {hold:.0f} min without hitting a target",
                                           'timeout')
            if not reason:
                continue

            tid = self.budget.log_trade(sym, 'SELL', ref, pos['shares'],
                                        exit_reason=exit_reason, now=now)
            row = self.budget.execute_trade(tid, now=now)
            if row is None:
                logger.error(f"[fast] EXIT {sym}: trade #{tid} did not execute")
                continue
            summary['exits'].append({
                'symbol': sym, 'shares': pos['shares'], 'price': row['price'],
                'ref_price': ref, 'reason': reason, 'exit_reason': exit_reason,
                'pnl': row['realized_pnl'], 'gross': row['gross_pnl'],
                'fees': row['fees'], 'pct': change, 'trade_id': tid})
            self.cooldown[sym] = now_et + timedelta(minutes=self.cooldown_min)
            self.entry_time.pop(sym, None)
            logger.info(f"[fast] EXIT {costs.qty_str(pos['shares'])} {sym} @ "
                        f"${row['price']:,.2f} (ref ${ref:,.2f}, {reason}) "
                        f"P&L ${row['realized_pnl']:+,.2f} after ${row['fees']:,.2f} fees")

        # ---- DAILY LOSS LIMIT: measured after exits so a stop that just fired counts ----
        acct = self._account_fields(prices, now)
        summary.update(acct)
        loss_tripped = day['loss_tripped_at'] is not None
        limit_pct = _cfg('DAILY_LOSS_LIMIT_PCT', 3)
        floor_equity = day['start_equity'] * (1 - limit_pct / 100)
        if not loss_tripped and acct['equity'] <= floor_equity:
            self.budget.set_day_flag(today_et, 'loss_tripped_at',
                                     now.isoformat(timespec='seconds'))
            loss_tripped = True
            summary['loss_announce'] = True     # only the cycle that trips announces
            logger.warning(f"[fast] Daily loss limit tripped: equity ${acct['equity']:,.2f} "
                           f"<= ${floor_equity:,.2f} ({limit_pct:g}% below the "
                           f"${day['start_equity']:,.2f} start) - no entries until "
                           f"the next ET date")
        summary['loss_tripped'] = loss_tripped

        # ---- SIGNAL LOG + ENTRIES -----------------------------------------
        # Every scored symbol is logged every cycle (INSERT OR IGNORE on the
        # bar, so the 60 s poll cannot inflate n); candidates are the above-bar
        # rows, already ranked by margin over each class's own bar.
        # The log writes go through budget.conn on THIS thread - the same
        # worker thread that runs the ledger's transactions, so there is no
        # cross-thread use of the connection (label_pending, which runs on the
        # event-loop thread, opens its own connection).
        signals = self.engine.scan_all()
        try:
            signal_log.record(self.budget.conn, signals, now=now)
        except Exception as e:
            # The log is measurement, not trading - never let it stop a cycle.
            logger.warning(f"[fast] signal log write failed: {e}")
        candidates = [s for s in signals if s.get('above_bar')]
        summary['candidates'] = candidates[:5]
        summary['bars'] = {c: self.engine.threshold(c)
                           for c in (metrics or {'stock': {}})}
        summary['bar'] = min(summary['bars'].values(), default=0.0)

        held = {p['symbol']: p for p in self.budget.get_positions()}
        slots = self.max_positions - len(held)
        if slots <= 0:
            summary['note'] = f"Holding {len(held)}/{self.max_positions} - no free slots."
            self.last_summary = summary
            return summary

        near_bell = stocks_open and to_close <= self.eod_flatten_min
        min_order = _cfg('MIN_ORDER_USD', 1)
        for sig in candidates:
            if slots <= 0:
                break
            sym = sig['symbol']
            cls = asset_class(sym)
            if sym in held:
                continue
            if loss_tripped:
                summary['skipped'].append((sym, 'daily loss limit'))
                continue
            ok, text = gates[cls]
            if not ok:
                summary['skipped'].append((sym, f'class gated: {text}'))
                continue
            if not (stocks_open or is_crypto(sym)):
                summary['skipped'].append((sym, 'market closed'))
                continue
            if near_bell and not is_crypto(sym):
                summary['skipped'].append((sym, 'too close to the bell'))
                continue
            if self.cooldown.get(sym, now_et) > now_et:
                summary['skipped'].append((sym, 'cooling down'))
                continue

            ref = self._price(sym, sig['price'])
            qty, size_usd, _est_fill = self.budget.size_order(sym, ref, prices=prices, now=now)
            if qty <= 0:
                # size_order returns 0 when the slot is worth less than
                # MIN_ORDER_USD; say which constraint bound.
                bp = self.budget.get_buying_power(now=now)
                summary['skipped'].append(
                    (sym, 'insufficient buying power' if bp < min_order else 'below min order'))
                continue

            tid = self.budget.log_trade(sym, 'BUY', ref, qty,
                                        probability=sig['probability'], now=now)
            row = self.budget.execute_trade(tid, now=now)
            if row is None:
                logger.error(f"[fast] ENTER {sym}: trade #{tid} did not execute")
                continue
            self.entry_time[sym] = now_et
            summary['entries'].append({
                'symbol': sym, 'shares': qty, 'price': row['price'], 'ref_price': ref,
                'probability': sig['probability'], 'cost': row['amount'],
                'fees': row['fees'], 'trade_id': tid})
            try:
                signal_log.mark_executed(self.budget.conn, sym, sig['bar_ts'], tid)
            except Exception as e:
                logger.warning(f"[fast] could not mark signal {sym}@{sig.get('bar_ts')} "
                               f"executed: {e}")
            slots -= 1
            logger.info(f"[fast] ENTER {costs.qty_str(qty)} {sym} @ ${row['price']:,.2f} "
                        f"(ref ${ref:,.2f}, p={sig['probability']:.3f}) "
                        f"cost ${row['amount']:,.2f}")

        if summary['entries']:
            # Entries moved cash; report the post-trade account, not the pre-trade one.
            summary.update(self._account_fields(prices, now))
        if not summary['entries'] and not summary['exits']:
            summary['note'] = (f"{len(candidates)} candidate(s) over the "
                               f"{summary['bar']:.3f} bar; nothing actionable.")
        self.last_summary = summary
        return summary
```

- [ ] **Step 9: Run the tests, confirm all pass**

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_fast_trader.py -v
```

Expected: `19 passed`.

Then run the whole suite to confirm nothing else imports the deleted property:

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests -q
```

Expected: every test passes (the `discord_bot` reference to `self.fast.max_hold_min` at `src/discord_bot.py:297` is inside a method body, so it does not break import; Task 9 replaces it).

- [ ] **Step 10: Commit**

```bash
cd /home/gdhughey/hugheylab-trading-bot && git add src/fast_trader.py tests/test_fast_trader.py && git commit -m "fast_trader: ref-to-ref barriers, equity sizing, daily loss limit, signal log

cycle(now=None) now runs: market state -> class gates -> fetch -> one quote
per held symbol -> day_state baseline -> exits (vs entry_ref) -> loss limit
-> scan_all + signal_log.record -> entries via size_order -> mark_executed.
FAST_MAX_HOLD_MIN is gone; max hold is the label horizon in minutes.
Tests pin account.opened_at via Database(path, now=) so the ledger's
created_at >= opened_at filters never depend on the wall clock.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

**Contract additions**
- `src/fast_trader.py`: `BARRIER_EPS = 1e-9` (module constant) — stop/TP comparisons are `change <= -sl + BARRIER_EPS` and `change >= tp - BARRIER_EPS`, because `(99.40 - 100) / 100` is `-0.005999999999999943` in binary and a bare comparison would miss a quote sitting exactly on the barrier.
- `src/fast_trader.py`: `_interval_minutes(interval: str | None = None) -> float` — parses `INTERVAL` (`'5m' -> 5.0`, `'1h' -> 60.0`, unparseable -> 5.0); used only by `max_hold_min`.
- `src/fast_trader.py`: `FastTrader.cycle(self, now: datetime | None = None) -> dict` — `now` is UTC tz-aware; defaults to `datetime.now(timezone.utc)`; threaded to `market_state`, `minutes_to_close`, `log_trade`, `execute_trade`, `size_order`, `get_buying_power`, `get_unsettled`, `signal_log.record`, and `set_day_flag` (as `now.isoformat(timespec='seconds')`).
- `src/fast_trader.py`: `FastTrader._account_fields(self, prices: dict, now) -> dict` with keys `equity, buying_power, unsettled` — merged into the cycle summary (also on the closed-market early return, with `prices={}`).
- `src/fast_trader.py`: `class_gate(cls, None)` semantics (contract is silent): `metrics=None` — no model for the class — is treated as `ev = 0.0`, so it passes the cost gate exactly as a real metrics dict would, then returns `(False, "Not trading. Model loses money (+0.000% per trade) on these settings.")` unless `FAST_IGNORE_EV` is set, in which case `(True, "trading on paper despite EV +0.000% below the 0.30% floor (FAST_IGNORE_EV on)")`. Task 7 must not assume `(True, ...)` for a missing-metrics class unless the flag is on.
- `src/fast_trader.py`: the "Not trading" texts are plain (no Discord markdown or emoji): `"Not trading. Edge of {ev*100:+.3f}% per trade is real but smaller than the {floor:.2%} it costs to get in and out, so it would lose money after fees."` and `"Not trading. Model loses money ({ev*100:+.3f}% per trade) on these settings."`. EV is printed as `ev * 100` because `metrics[cls]['ev']` is a fraction.
- Spec §3 step 3 literal correction: the FAST_IGNORE_EV text is `"trading on paper despite EV {ev*100:+.3f}% below the {floor:.2%} floor (FAST_IGNORE_EV on)"`, NOT `{ev:+.3f}%` as written in the spec (which would print the fraction 0.0006 as `+0.001%`). Task 6 prints `ev * 100`; Tasks 9 and 10 assert on that output. Amend the spec string.
- Contract dependency table correction: Task 7 (scorecard) depends on 4 **and 6** (it imports `class_gate` from `src/fast_trader.py`), and on 8 for the live report (`first_close_on_or_after` in `src/ml_engine.py` at runtime). The row should read "4, 6 (8 for the live report)".
- Contract test-conventions rule (add to the "Tests" paragraph at the top of the contract): any test that constructs `Database()` and then reads anything filtered by `account.opened_at` (`get_unsettled`, `get_buying_power`, `get_fees_paid`, `get_realized_pnl`, `get_gross_pnl`, `get_trades_since_open`, `build_scorecard`) MUST either pass `Database(path, now=<fixed UTC datetime earlier than every trade stamp>)` or pin `opened_at` with `UPDATE account SET opened_at = ?`. An unpinned `opened_at` is the wall clock and silently excludes fixed-date trades once the real date passes them. `tests/test_fast_trader.py::make_budget` uses `Database(path, now=OPENED_AT)` with `OPENED_AT = 2026-09-01T00:00Z`.
- `FastTrader.stop_loss` / `take_profit` legacy properties are kept (still read by `src/discord_bot.py:294-295`); only `max_hold_min` is deleted.
- Threading note for `src/signal_log.py` callers (aligns with Task 5's amended note): `record` and `mark_executed` are called on `budget.conn` from the cycle worker thread — the same thread as `BudgetTracker._txn`; the connection is autocommit so `with conn:` is safe. Only `label_pending` (Task 9, Discord event-loop thread) must use its own connection rather than `budget.conn`.

---

### Task 7: Scorecard — intervals, cost-adjusted breakeven, the go/no-go verdict, `build_scorecard`

**Depends on:** Task 3 (schema: `account`, `day_state`, `equity_history`, `signals`), Task 4 (`BudgetTracker` ledger methods, `_et_date`), Task 5 (transitively: `src/fast_trader.py` imports `src.signal_log`), Task 6 (`fast_trader.class_gate` for `gate_text`). Task 8's `first_close_on_or_after` is only reached through the `engine` object, which the tests fake, so Task 8 is needed for the live report but not to run this task. The contract's dependency table says "4" only; Step 5 corrects it to "4, 5, 6 (8 for the live report)".

**Files:**
- Create: `/home/gdhughey/hugheylab-trading-bot/src/scorecard.py`
- Create: `/home/gdhughey/hugheylab-trading-bot/tests/test_scorecard.py`
- Modify: `/home/gdhughey/hugheylab-trading-bot/docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md` — line 22 (the Task 7 row of the dependency table)
- Modify: `/home/gdhughey/hugheylab-trading-bot/docs/superpowers/specs/2026-09-12-paper-brokerage-design.md` — line 271 (`SE, 95% t-interval).` in section 4)

No other existing file is modified. `scipy` is NOT in `requirements.txt` (it only rides in transitively with scikit-learn), so `mean_ci` uses the normal z = 1.96 rather than a Student-t quantile; at the n ≥ 60 the GO rule requires, t(0.975, 59) = 2.00, a 2% wider band, and the GO comparison itself uses the SE, not the half-width. Step 5 amends the spec's "95% t-interval" wording to match.

**Units and types convention (binding for every consumer, Task 9 included):**
- Fractions are fractions (0.01 = 1%): `sig_*`, `tp_*`, `win_*`, `exec_mean`, `exec_se`, `ev_bt`, `ev_bt_net`, `cost`, `breakeven`, `bt_precision`, `spy.pct`, `pct_vs_ref`, `pnl_pct`, `all_time_pct`. Render with `:.1%` / `:+.3%`.
- Exactly three keys are PERCENTAGE POINTS (0.2 means 0.2%): `classes[cls]['exec_mean_pct']`, `classes[cls]['exec_ci']`, `since_open['max_drawdown_pct']`. Render with `f"{x:+.3f}%"` / `f"{x:.2f}%"`, NEVER with a `%` format spec (that multiplies by 100 again).
- Dollar keys are dollars.
- `account['unsettled_until']` is an ET calendar DATE string `'YYYY-MM-DD'` or `None`, never a timestamp. Render verbatim: `f"settles {unsettled_until} 09:30 ET"`. Do not parse it as a datetime or convert its timezone (a naive midnight taken as UTC lands on the previous ET day).
- `classes[cls]['bt_precision']`, `['ev_bt']`, `['ev_bt_net']` are `None` (and `bt_n` is `0`) when the class has no metrics entry — fast mode off, `intraday is None`, or the class did not train. Renderers must print `n/a` for a `None`; formatting them as numbers raises `TypeError`.
- `closed_today[]` items carry BOTH spellings of the P&L keys: `net` and `realized_pnl` (same value), `gross` and `gross_pnl` (same value). Either may be read.

- [ ] **Step 1: Write the failing tests for the statistics half (`wilson_ci`, `mean_ci`, `cost_breakeven`, `verdict`)**

Create `/home/gdhughey/hugheylab-trading-bot/tests/test_scorecard.py` with exactly this content:

```python
"""Scorecard arithmetic and the go/no-go verdict (spec sections 4, 7, 8)."""
import math
import statistics
from datetime import datetime, timezone

import pytest

from src import scorecard
from src.budget_tracker import BudgetTracker
from src.database import Database

UTC = timezone.utc

# Barriers pinned so breakeven does not depend on data/tuned.json (absent in
# the test tree) or on FAST_TAKE_PROFIT in the environment.
TUNED = {'stock': {'take_profit': 0.010, 'stop_loss': 0.006, 'horizon': 24},
         'crypto': {'take_profit': 0.006, 'stop_loss': 0.004, 'horizon': 24}}

# With TUNED above and the default stock costs (5 bps slippage per side plus
# the SEC fee): c = 0.0010206, be = (0.006 + 0.0010206) / 0.016.
COST_STOCK = 0.0010206
BE_STOCK = 0.4387875


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr('src.intraday_engine.TUNED', TUNED)
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '5')
    monkeypatch.setenv('SEC_FEE_RATE', '0.0000206')
    monkeypatch.setenv('FINRA_TAF_PER_SHARE', '0.000195')
    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '60')


# --- wilson_ci -------------------------------------------------------------

def test_wilson_ci_empty():
    assert scorecard.wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_known_value():
    lo, hi = scorecard.wilson_ci(50, 100)
    assert lo == pytest.approx(0.404, abs=1e-3)
    assert hi == pytest.approx(0.596, abs=1e-3)


def test_wilson_ci_clamped_at_extremes():
    lo, hi = scorecard.wilson_ci(10, 10)
    assert hi == 1.0 and 0.7 < lo < 1.0
    lo, hi = scorecard.wilson_ci(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.3


# --- mean_ci ---------------------------------------------------------------

def _ledger(n):
    """+1.5% / -0.5% alternating: +0.5%/trade on average, with real spread."""
    return [0.015 if i % 2 == 0 else -0.005 for i in range(n)]


def test_mean_ci_too_few_values():
    assert scorecard.mean_ci([]) == (0.0, 0.0, 0.0)
    assert scorecard.mean_ci([0.01]) == (0.0, 0.0, 0.0)


def test_mean_ci_excludes_zero_and_shrinks_with_n():
    mean, se, hw = scorecard.mean_ci(_ledger(100))
    assert mean == pytest.approx(0.005)
    assert hw == pytest.approx(1.96 * se)
    assert mean - hw > 0                      # 100 trades at +0.5% excludes zero
    _, _, hw400 = scorecard.mean_ci(_ledger(400))
    assert hw400 == pytest.approx(hw / 2, rel=0.01)   # 4x the trades, half the width


# --- cost_breakeven --------------------------------------------------------

def test_cost_breakeven(pinned):
    assert scorecard.cost_breakeven('stock') == pytest.approx(BE_STOCK, abs=1e-6)
    # crypto: round trip 1.2% exceeds the 0.6% take-profit -> breakeven above 100%
    assert scorecard.cost_breakeven('crypto') == pytest.approx(1.6, abs=1e-6)


# --- verdict ---------------------------------------------------------------

def _stats(**over):
    base = dict(gated=False, sig_n=1800, sig_lo=0.46, sig_hi=0.55, sig_lo_day=0.45,
                exec_n=60, exec_mean=0.0025, exec_se=0.001, ev_bt=0.003, cost=COST_STOCK)
    base.update(over)
    return base


def test_verdict_go(pinned):
    label, text = scorecard.verdict('stock', _stats())
    assert label == 'GO'
    assert 'n=1800' in text and 'n=60' in text


def test_verdict_no_go_when_cost_gated(pinned):
    label, text = scorecard.verdict('crypto', _stats(gated=True, cost=0.012))
    assert label == 'NO-GO'
    assert 'cost gate' in text


def test_verdict_no_go_when_ci_below_breakeven(pinned):
    label, text = scorecard.verdict('stock', _stats(sig_lo=0.36, sig_hi=0.42, sig_lo_day=0.35))
    assert label == 'NO-GO'
    assert '42.0%' in text and '43.9%' in text


@pytest.mark.parametrize('over, marker', [
    (dict(sig_lo=0.43), 'signal CI lower 43.0%'),
    (dict(sig_lo_day=0.43), 'day-clustered lower 43.0%'),
    (dict(exec_n=59), 'n=59 < 60'),
    (dict(exec_mean=-0.002), 'executed mean -0.200%'),
    (dict(ev_bt=None), 'no backtest EV'),
    (dict(sig_n=0, sig_lo=0.0, sig_hi=1.0, sig_lo_day=0.0), 'n=0'),
])
def test_verdict_extend(pinned, over, marker):
    label, text = scorecard.verdict('stock', _stats(**over))
    assert label == 'EXTEND'
    assert marker in text
```

- [ ] **Step 2: Run the tests and watch them fail at import**

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_scorecard.py -v
```

Expected: collection error, output contains
```
ModuleNotFoundError: No module named 'src.scorecard'
```
and `1 error`.

- [ ] **Step 3: Create `src/scorecard.py` with the statistics half**

Create `/home/gdhughey/hugheylab-trading-bot/src/scorecard.py` with exactly this content:

```python
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
```

- [ ] **Step 4: Run the statistics tests — all pass**

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_scorecard.py -v
```

Expected: `15 passed` (3 wilson, 2 mean_ci, 1 cost_breakeven, 3 verdict + 6 parametrized EXTEND cases), no failures.

- [ ] **Step 5: Amend the two doc lines (z-interval, dependency row) and commit the statistics half**

In `/home/gdhughey/hugheylab-trading-bot/docs/superpowers/specs/2026-09-12-paper-brokerage-design.md`, line 271 currently reads:

```
SE, 95% t-interval). `discord_bot._scorecard_embed(day)` renders it and is
```

Replace that line with:

```
SE, 95% normal interval with z = 1.96 — scipy is not a declared dependency, and
at the n ≥ 60 the GO rule requires t(0.975, 59) = 2.00 differs by 2%).
`discord_bot._scorecard_embed(day)` renders it and is
```

In `/home/gdhughey/hugheylab-trading-bot/docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md`, line 22 currently reads:

```
| 7 | scorecard | `src/scorecard.py`, `tests/test_scorecard.py` | 4 |
```

Replace it with:

```
| 7 | scorecard | `src/scorecard.py`, `tests/test_scorecard.py` | 4, 5, 6 (8 for the live report) |
```

Verify both edits took:

```
cd /home/gdhughey/hugheylab-trading-bot && grep -n "95% t-interval" docs/superpowers/specs/2026-09-12-paper-brokerage-design.md; echo "t-interval grep exit=$?"; grep -n "^| 7 | scorecard" docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md
```

Expected: the first grep prints nothing and `t-interval grep exit=1`; the second prints the line ending in `| 4, 5, 6 (8 for the live report) |`.

```
cd /home/gdhughey/hugheylab-trading-bot && git add src/scorecard.py tests/test_scorecard.py docs/superpowers/specs/2026-09-12-paper-brokerage-design.md docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md && git commit -m "$(cat <<'EOF'
Scorecard: Wilson and mean intervals, cost breakeven, go/no-go verdict

wilson_ci / mean_ci / cost_breakeven and verdict() per spec section 8.
mean_ci uses z=1.96: scipy is not a declared dependency; the spec's
"t-interval" wording is amended to match. The contract's dependency row
for Task 7 now lists 4, 5, 6 (scorecard imports fast_trader.class_gate,
which imports signal_log) and 8 for the live SPY benchmark.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp
EOF
)"
```

- [ ] **Step 6: Append the failing `build_scorecard` tests**

Append the following to the END of `/home/gdhughey/hugheylab-trading-bot/tests/test_scorecard.py`:

```python


# --- build_scorecard -------------------------------------------------------

class FakeEngine:
    """Only what build_scorecard touches on MLEngine: stored closes and the SPY start."""

    def __init__(self, closes, spy_start=650.0):
        self.closes, self.spy_start, self.asked = closes, spy_start, None

    def stored_close(self, symbol):
        return self.closes.get(symbol)

    def first_close_on_or_after(self, symbol, date_iso):
        self.asked = (symbol, date_iso)
        return self.spy_start


class FakeIntraday:
    """Only what build_scorecard touches on IntradayEngine: the metrics dict."""

    def __init__(self, metrics):
        self.metrics = metrics


OPENED_AT = '2026-09-10T12:00:00+00:00'
# ev 0.004 clears the default MIN_EV_TO_TRADE of 0.003, so class_gate allows
# stocks whether or not FAST_IGNORE_EV is set in the test environment.
STOCK_METRICS = {'precision': 0.55, 'test_signals': 1234, 'ev': 0.004,
                 'breakeven': 0.375, 'take_profit': 0.01, 'stop_loss': 0.006}


def _t(day, hh, mm):
    return datetime(2026, 9, day, hh, mm, tzinfo=UTC)


def _round_trip(budget, sym, ref_in, ref_out, reason, day, hh):
    """BUY 1 unit at hh:00 UTC, SELL it at hh:30 with the given exit reason."""
    b = budget.log_trade(sym, 'BUY', ref_in, 1.0, probability=0.6, now=_t(day, hh, 0))
    budget.execute_trade(b, now=_t(day, hh, 0))
    s = budget.log_trade(sym, 'SELL', ref_out, 1.0, exit_reason=reason, now=_t(day, hh, 30))
    budget.execute_trade(s, now=_t(day, hh, 30))


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """A $500 cash account: four closed stock trades, one open, a signal log.

    Stock costs are zeroed so every dollar below is exact; crypto keeps its
    60 bps per side so the cost gate still fires for it.
    Cash: 500 -100 +99 -200 +202 -200 +198 -50 +50.5 -100 = 399.5.
    """
    monkeypatch.setattr('src.intraday_engine.TUNED', TUNED)
    monkeypatch.setenv('STARTING_CASH', '500')
    monkeypatch.setenv('ACCOUNT_TYPE', 'cash')
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '0')
    monkeypatch.setenv('SEC_FEE_RATE', '0')
    monkeypatch.setenv('FINRA_TAF_PER_SHARE', '0')
    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '60')
    path = str(tmp_path / 'scorecard.db')
    Database(path)
    budget = BudgetTracker(path)
    with budget.conn:
        budget.conn.execute("UPDATE account SET opened_at = ?", (OPENED_AT,))

    # Fri 2026-09-11: one loser (-1%). Equity 499 at the close.
    _round_trip(budget, 'AMD', 100.0, 99.0, 'sl', 11, 14)
    budget.record_equity('2026-09-11', {}, now=_t(11, 21, 0))

    # Mon 2026-09-14: tp +1%, sl -1%, timeout +1%, then TSLA left open.
    _round_trip(budget, 'AAPL', 200.0, 202.0, 'tp', 14, 14)
    _round_trip(budget, 'MSFT', 200.0, 198.0, 'sl', 14, 15)
    _round_trip(budget, 'NVDA', 50.0, 50.5, 'timeout', 14, 16)
    b = budget.log_trade('TSLA', 'BUY', 100.0, 1.0, probability=0.6, now=_t(14, 17, 0))
    budget.execute_trade(b, now=_t(14, 17, 0))
    budget.record_equity('2026-09-14', {'TSLA': 102.0}, now=_t(14, 20, 5))

    # Signal log: 8 labelled above-bar stock signals since open (5 hits) across
    # two days, plus rows that must be EXCLUDED: below bar, unlabelled, before
    # opened_at. One crypto signal.
    sig_rows = [
        # bar_ts, symbol, cls, above_bar, trade_date, label
        ('2026-09-11T14:00:00+00:00', 'AAPL', 'stock', 1, '2026-09-11', 1),
        ('2026-09-11T14:05:00+00:00', 'MSFT', 'stock', 1, '2026-09-11', 1),
        ('2026-09-11T14:10:00+00:00', 'NVDA', 'stock', 1, '2026-09-11', 1),
        ('2026-09-11T14:15:00+00:00', 'AMD', 'stock', 1, '2026-09-11', 0),
        ('2026-09-14T14:00:00+00:00', 'AAPL', 'stock', 1, '2026-09-14', 1),
        ('2026-09-14T14:05:00+00:00', 'MSFT', 'stock', 1, '2026-09-14', 0),
        ('2026-09-14T14:10:00+00:00', 'NVDA', 'stock', 1, '2026-09-14', 1),
        ('2026-09-14T14:15:00+00:00', 'AMD', 'stock', 1, '2026-09-14', 0),
        ('2026-09-14T14:20:00+00:00', 'TSLA', 'stock', 0, '2026-09-14', 1),      # below bar
        ('2026-09-14T14:25:00+00:00', 'META', 'stock', 1, '2026-09-14', None),   # unlabelled
        ('2026-09-09T14:00:00+00:00', 'AAPL', 'stock', 1, '2026-09-09', 1),      # before open
        ('2026-09-14T14:00:00+00:00', 'BTC-USD', 'crypto', 1, '2026-09-14', 1),
    ]
    with budget.conn:
        budget.conn.executemany(
            "INSERT INTO signals (bar_ts, symbol, asset_class, probability, bar, above_bar, "
            "ref_price, trade_date, label, labeled_at) VALUES (?, ?, ?, 0.5, 0.4, ?, 100.0, ?, ?, ?)",
            [(ts, sym, cls, ab, td, lab, None if lab is None else '2026-09-14T20:05:00+00:00')
             for ts, sym, cls, ab, td, lab in sig_rows])

    budget.ensure_day_state('2026-09-14', 499.0)
    budget.set_day_flag('2026-09-14', 'loss_tripped_at', '2026-09-14T15:00:00+00:00')
    return budget


def test_build_scorecard_full(ledger):
    engine = FakeEngine({'TSLA': 102.0, 'SPY': 660.0})
    card = scorecard.build_scorecard(ledger, engine, FakeIntraday({'stock': STOCK_METRICS}),
                                     '2026-09-14', now=_t(14, 21, 0))

    nets = [-1.0, 2.0, -2.0, 0.5]
    hw = 1.96 * statistics.stdev(nets) / math.sqrt(4)
    head = card['headline']
    assert head['n_closed'] == 4
    assert head['all_time_net'] == pytest.approx(1.5)          # -0.5 realised + 2.0 open
    assert head['ci_dollars'] == pytest.approx(hw * 4)

    acct = card['account']
    assert acct['starting_cash'] == 500.0
    assert acct['cash'] == pytest.approx(399.5)
    assert acct['equity'] == pytest.approx(501.5)
    assert acct['unsettled'] == pytest.approx(450.5)            # today's three SELLs, T+1
    assert acct['buying_power'] == 0.0
    # An ET DATE string, not a timestamp: the embed prints it verbatim as
    # "settles 2026-09-15 09:30 ET" (Task 10 Step 12 looks for that literal).
    assert acct['unsettled_until'] == '2026-09-15'
    assert isinstance(acct['unsettled_until'], str) and len(acct['unsettled_until']) == 10
    assert acct['fees_paid'] == 0.0
    assert acct['gross_pnl'] == pytest.approx(-0.5)

    today = card['today']
    assert today['realized'] == pytest.approx(0.5)
    assert today['unrealized'] == pytest.approx(2.0)
    assert today['n_trades'] == 7
    assert (today['wins'], today['losses']) == (2, 1)
    assert today['loss_tripped'] is True
    assert [c['symbol'] for c in card['closed_today']] == ['AAPL', 'MSFT', 'NVDA']
    assert [c['exit_reason'] for c in card['closed_today']] == ['tp', 'sl', 'timeout']
    first = card['closed_today'][0]
    # Both spellings of the P&L keys are guaranteed (see Contract additions).
    assert first['net'] == pytest.approx(2.0) and first['realized_pnl'] == first['net']
    assert first['gross'] == pytest.approx(2.0) and first['gross_pnl'] == first['gross']
    assert set(first) == {'symbol', 'shares', 'price', 'amount', 'net', 'realized_pnl',
                          'gross', 'gross_pnl', 'fees', 'exit_reason', 'created_at'}

    assert len(card['positions']) == 1
    pos = card['positions'][0]
    assert pos['symbol'] == 'TSLA' and pos['price'] == 102.0
    assert pos['pct_vs_ref'] == pytest.approx(0.02)

    so = card['since_open']
    assert so['n_closed'] == 4 and so['win_rate'] == 0.5
    assert (so['win_lo'], so['win_hi']) == scorecard.wilson_ci(2, 4)
    assert so['mean_net'] == pytest.approx(-0.125)
    assert so['mean_ci'] == pytest.approx(hw)
    assert so['profit_factor'] == pytest.approx(2.5 / 3.0)
    # PERCENTAGE POINTS: 500 -> 499 on day 2 is a 0.2% fall, reported as 0.2.
    assert so['max_drawdown_pct'] == pytest.approx(0.2)
    assert so['days_running'] == 5                              # 10th..14th inclusive

    st = card['classes']['stock']
    assert st['gated'] is False and st['verdict'] == 'EXTEND'
    assert not st['gate_text'].startswith('blocked')
    assert st['exec_n'] == 4 and st['exec_mean_pct'] == pytest.approx(0.0)
    # PERCENTAGE POINTS: the fraction half-width times 100.
    assert st['exec_ci'] == pytest.approx(100 * 1.96 * statistics.stdev([-0.01, 0.01, -0.01, 0.01]) / 2)
    assert st['exec_ci'] == pytest.approx(100 * scorecard.mean_ci([-0.01, 0.01, -0.01, 0.01])[2])
    assert st['exits_by_reason']['tp'] == {'n': 1, 'mean_net': pytest.approx(2.0)}
    assert st['exits_by_reason']['sl'] == {'n': 2, 'mean_net': pytest.approx(-1.5)}
    assert st['exits_by_reason']['timeout'] == {'n': 1, 'mean_net': pytest.approx(0.5)}
    assert st['tp_n'] == 4 and st['tp_first_rate'] == 0.25
    assert (st['tp_lo'], st['tp_hi']) == scorecard.wilson_ci(1, 4)
    assert st['bt_precision'] == 0.55 and st['bt_n'] == 1234
    assert st['ev_bt'] == 0.004 and st['ev_bt_net'] == pytest.approx(0.004)
    assert st['cost'] == 0.0 and st['breakeven'] == pytest.approx(0.375)
    assert st['sig_n'] == 8 and st['sig_hits'] == 5 and st['sig_rate'] == 0.625
    assert (st['sig_lo'], st['sig_hi']) == scorecard.wilson_ci(5, 8)
    assert st['sig_days'] == 2
    assert st['sig_lo_day'] == pytest.approx(
        0.625 - 1.96 * statistics.stdev([0.75, 0.5]) / math.sqrt(2))

    cr = card['classes']['crypto']
    assert cr['gated'] is True and cr['verdict'] == 'NO-GO'
    assert cr['gate_text'].startswith('blocked')
    assert cr['exec_n'] == 0 and cr['sig_n'] == 1 and cr['sig_lo_day'] == 0.0
    # No metrics entry for crypto -> the backtest fields are None, not 0.
    assert cr['bt_precision'] is None and cr['ev_bt'] is None and cr['ev_bt_net'] is None
    assert cr['bt_n'] == 0

    assert engine.asked == ('SPY', '2026-09-10')
    assert card['spy']['start_close'] == 650.0 and card['spy']['last_close'] == 660.0
    assert card['spy']['pct'] == pytest.approx(660 / 650 - 1)
    assert card['spy']['value'] == pytest.approx(500 * 660 / 650)


def test_build_scorecard_without_benchmark_or_metrics(ledger):
    """intraday=None (fast mode off / training failed): every class's backtest
    fields are None and the verdict still comes out, so the 16:05 report can
    post on a day the model did not train."""
    engine = FakeEngine({'TSLA': 102.0}, spy_start=None)
    card = scorecard.build_scorecard(ledger, engine, None, '2026-09-11', now=_t(14, 21, 0))
    assert card['spy'] is None
    for cls in ('stock', 'crypto'):
        c = card['classes'][cls]
        assert c['bt_precision'] is None and c['bt_n'] == 0
        assert c['ev_bt'] is None and c['ev_bt_net'] is None
    st = card['classes']['stock']
    assert st['verdict'] == 'EXTEND' and 'no backtest EV' in st['verdict_text']
    assert card['classes']['crypto']['verdict'] == 'NO-GO'
    # 2026-09-11 has no day_state row and only the AMD round trip.
    assert card['today']['loss_tripped'] is False
    assert card['today']['n_trades'] == 2
    assert card['today']['realized'] == pytest.approx(-1.0)
    assert [c['symbol'] for c in card['closed_today']] == ['AMD']
    assert card['closed_today'][0]['net'] == pytest.approx(-1.0)
    assert card['closed_today'][0]['realized_pnl'] == pytest.approx(-1.0)
    assert card['since_open']['days_running'] == 2
    # AMD's Friday proceeds settled Monday 09:30 ET, before `now`: nothing pending.
    assert card['account']['unsettled'] == 0.0
    assert card['account']['unsettled_until'] is None
```

- [ ] **Step 7: Run the tests — the two new ones fail**

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_scorecard.py -v
```

Expected: `15 passed, 2 failed`; both failures show
```
AttributeError: module 'src.scorecard' has no attribute 'build_scorecard'
```

- [ ] **Step 8: Add the helpers and `build_scorecard` to `src/scorecard.py`**

Append the following to the END of `/home/gdhughey/hugheylab-trading-bot/src/scorecard.py` (after `verdict`):

```python


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
```

- [ ] **Step 9: Run the full scorecard test file — all pass**

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_scorecard.py -v
```

Expected: `17 passed`. Then run the whole suite to confirm nothing else moved:

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests -q
```

Expected: every test passes (no `failed`, no `error`).

- [ ] **Step 10: Commit `build_scorecard`**

```
cd /home/gdhughey/hugheylab-trading-bot && git add src/scorecard.py tests/test_scorecard.py && git commit -m "$(cat <<'EOF'
Scorecard: build_scorecard over the ledger, signal log and SPY benchmark

Headline/account/today/since-open blocks from BudgetTracker, per-class
verdict inputs from the signals table and intraday metrics, max drawdown
from equity_series, SPY buy-and-hold context (None when either close is
missing). Display units are fixed in the docstring: exec_mean_pct,
exec_ci and max_drawdown_pct are percentage points, unsettled_until is
an ET date string, backtest fields are None for an untrained class, and
closed_today rows carry both net/realized_pnl and gross/gross_pnl.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp
EOF
)"
```

**Contract additions**

- `src/scorecard.py` module constants: `Z95 = 1.96`, `CLASSES = ('stock', 'crypto')` (always both reported, in this order), `BARRIER_EXITS = ('tp', 'sl', 'timeout', 'eod')`.
- `mean_ci` is a 95% NORMAL interval (z = 1.96), not a Student-t interval; the spec's section-4 wording is amended in Step 5.
- Task 7 depends on Tasks 4, 5 (transitively: `src/fast_trader.py` imports `src.signal_log`), 6 (`class_gate`), and on Task 8 only for the live SPY benchmark (`engine.first_close_on_or_after`, faked in tests); the contract's dependency row is amended in Step 5.
- `_net_return(row) -> float | None` — net return of a closed SELL row = `realized_pnl / (amount - realized_pnl)`.
- `_max_drawdown_pct(series: list[dict]) -> float` — PERCENTAGE POINTS of the running peak over `equity_series()` (0.2 means 0.2%).
- `_signal_stats(conn, cls: str, opened_at: str) -> dict` — keys `sig_n, sig_hits, sig_rate, sig_lo, sig_hi, sig_lo_day, sig_days`; filters `asset_class = cls AND above_bar = 1 AND label IS NOT NULL AND bar_ts >= opened_at`; `sig_lo_day = 0.0` when fewer than 2 trade dates.
- Extra keys in `classes[cls]` beyond the contract list: `sig_hits`, `sig_days`, `tp_n` (denominator of the TP-first rate: tp+sl+timeout+eod, 'manual' excluded), `breakeven` (= `cost_breakeven(cls)`), plus the raw verdict inputs `gated, exec_n, exec_mean, exec_se, ev_bt, cost`.
- `classes[cls]['gated']` (and `verdict()`'s `stats['gated']`) is the COST gate only: `barriers(cls)[0] <= costs.round_trip_cost(cls)` (spec §8), NOT `not class_gate(...)[0]`. An EV-gated class therefore reads EXTEND, not NO-GO. `gate_text` is `class_gate(cls, metrics.get(cls))[1]`; `class_gate` is called with `None` when the class has no metrics entry (Task 6 scores that as ev 0.0 — the text is then the EV-gate or FAST_IGNORE_EV line, after the cost gate), and nothing in Task 7 asserts on that text.
- **None-handling (binding for renderers):** `classes[cls]['bt_precision']`, `['ev_bt']`, `['ev_bt_net']` are `None` and `bt_n` is `0` whenever `metrics` has no entry for the class (`intraday is None`, fast mode off, or the class did not train); `verdict()` then reports EXTEND with "no backtest EV to compare executed returns against" (unless NO-GO applies). A renderer must print `n/a` for a `None` — formatting it with `:.1%` / `:+.3%` raises `TypeError`. `since_open.profit_factor` is likewise `None` when there is no losing trade; `positions[].price`, `market_value`, `pnl`, `pnl_pct`, `pct_vs_ref` are `None` when the mark is missing.
- **Units (binding for renderers):** `classes[cls]['exec_mean']`, `exec_se`, `ev_bt`, `ev_bt_net`, `cost`, `breakeven`, `bt_precision`, `sig_*`, `tp_*`, `win_*`, `spy.pct`, `pct_vs_ref`, `all_time_pct` are FRACTIONS (render `:.1%` / `:+.3%`). Exactly three keys are PERCENTAGE POINTS: `classes[cls]['exec_mean_pct']` (= `exec_mean * 100`), `classes[cls]['exec_ci']` (= 95% half-width × 100), `since_open['max_drawdown_pct']`; render them as `f"{x:+.3f}%"` / `f"{x:.2f}%"`, never with a `%` format spec.
- **`account.unsettled_until` (binding for renderers):** ET calendar DATE string `'YYYY-MM-DD'` of the latest still-unsettled `available_at`, or `None`. Render verbatim as `f"settles {unsettled_until} 09:30 ET"` (the literal Task 10 Step 12 checks); never pass it through a datetime parser or timezone conversion.
- `closed_today[]` item keys (exactly): `symbol, shares, price, amount, net, realized_pnl, gross, gross_pnl, fees, exit_reason, created_at`, where `net == realized_pnl` and `gross == gross_pnl` (both spellings always present); all rows for the day, in ledger order (embed truncates to 10).
- `positions[]` item keys: `symbol, shares, avg_price, entry_ref, price, market_value, pnl, pnl_pct, pct_vs_ref`.
- `since_open.days_running` counts the opening ET date as day 1; `today.n_trades` counts every EXECUTED row (BUY and SELL) with `trade_date == day_et`.
- `spy = {'start_close', 'last_close', 'pct' (fraction), 'value' (starting_cash × (1+pct))}` or `None`.
- Position marks and the benchmark use `engine.stored_close` (not `latest_price`), so the report never depends on a live quote provider.


---

### Task 8: Benchmark symbols (SPY) in the daily engine

Goal: `fetch_and_store_data` pulls SPY into the `prices` table alongside the universe (one extra name in the same provider batch) so the daily scorecard (Task 7) can print a buy-and-hold comparison, while training and scanning never see it. Adds `first_close_on_or_after(symbol, date_iso)` for the "first SPY close on or after the account opened" lookup. No other task touches `src/ml_engine.py`, so the line numbers below hold regardless of task order.

Dependency note (corrects the contract's task table): Task 8 itself depends on nothing, but Task 7 (scorecard) is NOT "depends on 4" only. `src/scorecard.py` imports `class_gate` from `src/fast_trader.py` (Task 6) and, at runtime, `build_scorecard` calls `engine.first_close_on_or_after` from this task for the live `spy` block. The contract row for Task 7 must therefore read `4, 6 (8 for the live report)`. Task 7's unit tests stub the engine, so Task 7 can be *written and tested* before this task lands, but the live 16:05 report will raise `AttributeError` until this task is merged. Step 11 below fixes the contract table so the next reader is not misled; if Task 7 has not been executed yet, execute this task first.

**Files:**
- Modify: `/home/gdhughey/hugheylab-trading-bot/src/ml_engine.py`
  - lines 46-50 (universe constants block: `UNIVERSE`, `DEFAULT_SYMBOLS`, `BATCH_SIZE`) — add `BENCHMARK_SYMBOLS`
  - lines 167-211 (`fetch_and_store_data`) — prepend benchmarks to `pending`, fix the summary log count
  - lines 245-247 (`_stored_symbols`) — exclude benchmarks
  - lines 324-343 (`stored_close`) — unchanged; insert `first_close_on_or_after` immediately after it (before `latest_price`, line 345)
- Modify: `/home/gdhughey/hugheylab-trading-bot/docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md`
  - line 22 (the `| 7 | scorecard | ... | 4 |` row of the dependency table) — correct the "Depends on" cell
- Create: `/home/gdhughey/hugheylab-trading-bot/tests/test_benchmark.py`

Context for the engineer: `TradingSignalEngine.__init__(db_path)` calls `connect(db_path)` (opens SQLite, does NOT create tables), `build_chain()` and `build_quote_chain()` (instantiate provider objects, no network), and `_load_model()` (loads `MODEL_PATH` if the file exists). `fetch_and_store_data` iterates `self.providers`, skipping any with `provides_history == False`, slicing to `per_cycle_cap` when set, and calls `provider.fetch(symbols, LOOKBACK)` which returns `(frames_by_symbol: dict[str, DataFrame], missing: list)`. `_store` expects a DataFrame with a DatetimeIndex and lowercase columns `open, high, low, close, volume`. `connect()` sets `row_factory = sqlite3.Row`.

- [ ] **Step 1: Write the failing tests for the fetch list and `_stored_symbols`**

Create `/home/gdhughey/hugheylab-trading-bot/tests/test_benchmark.py` with exactly this content:

```python
"""Benchmark symbols (Task 8).

SPY is fetched alongside the universe so the daily scorecard can print a
buy-and-hold comparison, but it must never be trained on or scanned. These
tests replace the provider chain with an in-memory fake so nothing touches
the network.
"""

import pandas as pd
import pytest

from src import ml_engine
from src.database import Database
from src.ml_engine import BENCHMARK_SYMBOLS, TradingSignalEngine


class FakeProvider:
    """Stands in for the whole history chain.

    Records every symbol list it is asked for and answers each symbol with two
    daily bars, so nothing falls through as 'missing' and the engine never
    tries a second provider.
    """
    name = 'fake'
    provides_history = True
    per_cycle_cap = None

    def __init__(self):
        self.requests = []

    def fetch(self, symbols, period='2y'):
        self.requests.append(list(symbols))
        idx = pd.to_datetime(['2026-09-14', '2026-09-15'])
        frames = {
            s: pd.DataFrame({
                'open': [100.0, 101.0], 'high': [102.0, 103.0],
                'low': [99.0, 100.0], 'close': [101.0, 102.0],
                'volume': [1000.0, 1000.0],
            }, index=idx)
            for s in symbols
        }
        return frames, []

    def requested(self) -> list:
        """Every symbol asked for, in order, across all fetch calls."""
        return [s for req in self.requests for s in req]


@pytest.fixture
def fake():
    return FakeProvider()


@pytest.fixture
def engine(tmp_path, fake, monkeypatch):
    """Engine on a fresh file DB whose only data source is `fake`."""
    path = str(tmp_path / 'bench.db')
    Database(path)  # creates the prices table; connect() alone does not
    # Keep the engine away from data/model.joblib and the real providers.
    monkeypatch.setattr(ml_engine, 'MODEL_PATH', str(tmp_path / 'model.joblib'))
    monkeypatch.setattr(ml_engine, 'build_chain', lambda: [fake])
    monkeypatch.setattr(ml_engine, 'build_quote_chain', lambda: [])
    return TradingSignalEngine(path)


def _insert_closes(conn, symbol: str, closes: dict) -> None:
    """Write daily bars straight into `prices`, bypassing the provider chain."""
    with conn:
        conn.executemany(
            "INSERT INTO prices (symbol, date, open, high, low, close, volume, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'test')",
            [(symbol, d, c, c, c, c, 0.0) for d, c in closes.items()])


def test_benchmark_symbols_is_spy():
    assert BENCHMARK_SYMBOLS == ['SPY']


def test_fetch_requests_spy_once_and_keeps_it_out_of_symbols(engine, fake):
    rows = engine.fetch_and_store_data(['AAPL', 'MSFT'])

    # SPY rides in the same batch as the universe - one extra name, not an
    # extra request - and exactly once.
    assert fake.requested() == ['SPY', 'AAPL', 'MSFT']
    assert fake.requested().count('SPY') == 1
    # 2 bars x 3 symbols landed in prices, SPY included ...
    assert rows == 6
    assert engine.conn.execute(
        "SELECT COUNT(*) AS n FROM prices WHERE symbol = 'SPY'").fetchone()['n'] == 2
    # ... but the training/scanning universe never sees it.
    assert engine.symbols == ['AAPL', 'MSFT']
    assert sorted(engine._stored_symbols()) == ['AAPL', 'MSFT']


def test_fetch_does_not_request_spy_twice_when_universe_already_has_it(engine, fake):
    engine.fetch_and_store_data(['SPY', 'AAPL'])

    assert fake.requested() == ['SPY', 'AAPL']
    assert fake.requested().count('SPY') == 1
    # Stored, but still filtered out of the training list.
    assert sorted(engine._stored_symbols()) == ['AAPL']


def test_stored_symbols_excludes_benchmark_rows_written_directly(engine):
    _insert_closes(engine.conn, 'SPY', {'2026-09-14': 650.0})
    _insert_closes(engine.conn, 'AAPL', {'2026-09-14': 230.0})

    assert engine._stored_symbols() == ['AAPL']
```

- [ ] **Step 2: Run the tests and confirm they fail on the missing constant**

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_benchmark.py -v
```

Expected: collection error, no tests run. Output contains:

```
ImportError while importing test module '/opt/trading-bot-dev/tests/test_benchmark.py'.
...
ImportError: cannot import name 'BENCHMARK_SYMBOLS' from 'src.ml_engine'
```

- [ ] **Step 3: Add `BENCHMARK_SYMBOLS`, prepend benchmarks in `fetch_and_store_data`, filter `_stored_symbols`**

In `/home/gdhughey/hugheylab-trading-bot/src/ml_engine.py`, replace the constants block at lines 46-50:

```python
# Which symbols to scan. 'sp500' pulls the current index constituents;
# 'default' is the original 5; anything else is treated as a comma list.
UNIVERSE = os.getenv('UNIVERSE', 'default')
DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN']
BATCH_SIZE = int(os.getenv('FETCH_BATCH_SIZE', 60))
```

with:

```python
# Which symbols to scan. 'sp500' pulls the current index constituents;
# 'default' is the original 5; anything else is treated as a comma list.
UNIVERSE = os.getenv('UNIVERSE', 'default')
DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN']
BATCH_SIZE = int(os.getenv('FETCH_BATCH_SIZE', 60))

# Fetched alongside the universe so the daily scorecard can quote SPY
# buy-and-hold for context, but never trained on or scanned:
# fetch_and_store_data keeps them out of self.symbols and _stored_symbols()
# filters them out of the training list.
BENCHMARK_SYMBOLS = ['SPY']
```

Replace the whole of `fetch_and_store_data` (lines 167-211) with:

```python
    def fetch_and_store_data(self, symbols=None) -> int:
        """Fill price history from the provider chain.

        Each provider only sees the symbols still missing after the previous
        one, so a rate-limited source is spent on genuine gaps rather than on
        symbols that already resolved.
        """
        self.symbols = list(symbols) if symbols else (self.symbols or load_universe())
        # Benchmarks ride along in the same provider batches (one extra name
        # per request) but never enter self.symbols, so training and scanning
        # do not see them. A caller that lists SPY itself is not asked twice.
        pending = [s for s in BENCHMARK_SYMBOLS if s not in self.symbols] + list(self.symbols)
        requested = len(pending)
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

        # `requested` rather than len(self.symbols): the benchmark names were
        # fetched too, and the count would otherwise go negative-looking short.
        logger.info(f"Stored {total} rows across {requested - len(pending)} symbols "
                    f"via {len([p for p in self.providers if p.provides_history])} source(s)")
        return total
```

Replace `_stored_symbols` (lines 245-247, now shifted by the lines added above; find it by name) with:

```python
    def _stored_symbols(self) -> list:
        rows = self.conn.execute("SELECT DISTINCT symbol FROM prices").fetchall()
        # Benchmarks live in `prices` for the daily report only; feeding the
        # index itself to the pooled model would just teach it what the
        # market did.
        return [r['symbol'] for r in rows if r['symbol'] not in BENCHMARK_SYMBOLS]
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_benchmark.py -v
```

Expected:

```
tests/test_benchmark.py::test_benchmark_symbols_is_spy PASSED
tests/test_benchmark.py::test_fetch_requests_spy_once_and_keeps_it_out_of_symbols PASSED
tests/test_benchmark.py::test_fetch_does_not_request_spy_twice_when_universe_already_has_it PASSED
tests/test_benchmark.py::test_stored_symbols_excludes_benchmark_rows_written_directly PASSED
4 passed
```

- [ ] **Step 5: Commit**

```bash
cd /home/gdhughey/hugheylab-trading-bot && git add src/ml_engine.py tests/test_benchmark.py && git commit -m "ml_engine: fetch SPY as a benchmark without adding it to the universe

BENCHMARK_SYMBOLS = ['SPY'] rides along in fetch_and_store_data's provider
batches so the daily scorecard can quote buy-and-hold, but self.symbols and
_stored_symbols() never include it, so training and scanning are unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 6: Write the failing tests for `stored_close('SPY')` and `first_close_on_or_after`**

Append these two functions to the end of `/home/gdhughey/hugheylab-trading-bot/tests/test_benchmark.py`:

```python
def test_stored_close_and_latest_price_read_the_newest_spy_bar(engine):
    _insert_closes(engine.conn, 'SPY', {
        '2026-09-11': 640.0, '2026-09-14': 650.0, '2026-09-15': 655.0})

    assert engine.stored_close('SPY') == 655.0
    # No quote providers in this fixture, so latest_price falls back to the
    # stored close - the path the scorecard's "last close" uses off-hours.
    assert engine.latest_price('SPY') == 655.0


def test_first_close_on_or_after(engine):
    _insert_closes(engine.conn, 'SPY', {
        '2026-09-11': 640.0, '2026-09-14': 650.0, '2026-09-15': 655.0})

    # Exact date match is inclusive.
    assert engine.first_close_on_or_after('SPY', '2026-09-14') == 650.0
    # Account opened on a Saturday: the first bar after it is Monday's.
    assert engine.first_close_on_or_after('SPY', '2026-09-12') == 650.0
    # A full UTC ISO timestamp (account.opened_at) is truncated to its date,
    # so the bar on that date is still included rather than string-comparing
    # past it.
    assert engine.first_close_on_or_after('SPY', '2026-09-14T13:30:00+00:00') == 650.0
    # Nothing stored on or after the date, or an unknown symbol -> None.
    assert engine.first_close_on_or_after('SPY', '2026-09-16') is None
    assert engine.first_close_on_or_after('QQQ', '2026-09-14') is None
```

- [ ] **Step 7: Run the tests and confirm the new one fails on the missing method**

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_benchmark.py -v
```

Expected: 5 passed, 1 failed. The failure is:

```
tests/test_benchmark.py::test_first_close_on_or_after FAILED
...
AttributeError: 'TradingSignalEngine' object has no attribute 'first_close_on_or_after'
```

- [ ] **Step 8: Implement `first_close_on_or_after`**

In `/home/gdhughey/hugheylab-trading-bot/src/ml_engine.py`, directly after the end of `stored_close` (the method whose body ends with `except Exception:` / `return None`) and before `def latest_price(self, symbol: str):`, insert:

```python
    def first_close_on_or_after(self, symbol: str, date_iso: str):
        """Earliest stored daily close dated `date_iso` or later, else None.

        The scorecard's SPY buy-and-hold line starts the day the paper account
        opened; if that fell on a weekend or holiday, the first bar after it
        is the price a buy-and-hold investor would actually have paid.
        `date_iso` may be a bare date or a full ISO timestamp (account.opened_at)
        - only the date part is compared, because 'YYYY-MM-DDTHH:MM' sorts
        after 'YYYY-MM-DD' and would silently skip that day's bar.
        """
        row = self.conn.execute(
            "SELECT close FROM prices WHERE symbol = ? AND date >= ? "
            "ORDER BY date ASC LIMIT 1",
            (symbol, date_iso[:10]),
        ).fetchone()
        return float(row['close']) if row else None
```

The surrounding code after the edit reads:

```python
    def stored_close(self, symbol: str):
        ...
        try:
            row = self.conn.execute(
                "SELECT close FROM prices_intraday WHERE symbol = ? "
                "ORDER BY ts DESC LIMIT 1", (symbol,)).fetchone()
            return float(row['close']) if row else None
        except Exception:
            return None

    def first_close_on_or_after(self, symbol: str, date_iso: str):
        ...  (as above)

    def latest_price(self, symbol: str):
        """Live quote if any provider can give one, else the last stored close.
```

`stored_close` and `latest_price` are NOT changed: `stored_close` already queries `prices` by symbol with no universe filter, so `'SPY'` works once its rows exist.

- [ ] **Step 9: Run the tests and confirm they all pass**

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_benchmark.py -v
```

Expected:

```
tests/test_benchmark.py::test_benchmark_symbols_is_spy PASSED
tests/test_benchmark.py::test_fetch_requests_spy_once_and_keeps_it_out_of_symbols PASSED
tests/test_benchmark.py::test_fetch_does_not_request_spy_twice_when_universe_already_has_it PASSED
tests/test_benchmark.py::test_stored_symbols_excludes_benchmark_rows_written_directly PASSED
tests/test_benchmark.py::test_stored_close_and_latest_price_read_the_newest_spy_bar PASSED
tests/test_benchmark.py::test_first_close_on_or_after PASSED
6 passed
```

Also run the existing smoke test to be sure the module still imports cleanly alongside everything else:

```bash
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_smoke.py tests/test_benchmark.py -q
```

Expected: `7 passed`.

- [ ] **Step 10: Commit**

```bash
cd /home/gdhughey/hugheylab-trading-bot && git add src/ml_engine.py tests/test_benchmark.py && git commit -m "ml_engine: first_close_on_or_after for the SPY buy-and-hold line

The scorecard needs the first SPY close on or after account.opened_at. The
lookup truncates a full ISO timestamp to its date so the opening day's bar is
not string-compared away.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 11: Correct the contract's dependency row for Task 7**

The contract's task table says Task 7 depends only on Task 4, but `src/scorecard.py` imports `class_gate` from `src/fast_trader.py` (Task 6) and `build_scorecard` calls `engine.first_close_on_or_after` (this task) at report time. Fix the table so whoever sequences the remaining tasks does not run the live report before this task is merged.

In `/home/gdhughey/hugheylab-trading-bot/docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md`, replace line 22:

```
| 7 | scorecard | `src/scorecard.py`, `tests/test_scorecard.py` | 4 |
```

with:

```
| 7 | scorecard | `src/scorecard.py`, `tests/test_scorecard.py` | 4, 6 (8 for the live report) |
```

No other line of the contract changes. Verify the edit took:

```bash
cd /home/gdhughey/hugheylab-trading-bot && grep -n '^| 7 |' docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md
```

Expected output:

```
22:| 7 | scorecard | `src/scorecard.py`, `tests/test_scorecard.py` | 4, 6 (8 for the live report) |
```

- [ ] **Step 12: Commit the contract correction**

```bash
cd /home/gdhughey/hugheylab-trading-bot && git add docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md && git commit -m "contract: scorecard (Task 7) depends on fast_trader and the SPY benchmark

scorecard.py imports class_gate from src/fast_trader.py (Task 6) and the
live report calls engine.first_close_on_or_after (Task 8); the dependency
table listed only Task 4.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

**Contract additions**
- None. All names (`BENCHMARK_SYMBOLS`, `fetch_and_store_data`, `_stored_symbols`, `stored_close`, `first_close_on_or_after(symbol, date_iso) -> float | None`) are as in the contract. One behavioural note, not a new name: `first_close_on_or_after` compares only `date_iso[:10]`, so callers may pass either `opened_at[:10]` or the full `opened_at` string.
- Contract table correction (not a new name): the Task 7 "Depends on" cell becomes `4, 6 (8 for the live report)` — `src/scorecard.py` imports `class_gate` from `src/fast_trader.py` (Task 6) and `build_scorecard` calls `engine.first_close_on_or_after` (Task 8) at report time. Applied in Step 11.


---

### Task 9: Discord — scorecard report, class gates, account labels, ledger-driven embeds

Depends on Tasks 4 (`BudgetTracker` account API), 6 (`class_gate`, `max_hold_min`, cycle summary keys), 7 (`build_scorecard`, `cost_breakeven`) and 5 (`signal_log.label_pending`). Nothing here touches the network or logs into Discord in tests: `commands.Bot(...)` and `discord.Embed` construct offline (verified on LXC 200, discord.py 2.7.1).

Shapes this task CONSUMES from Task 7's contract additions (the embed is written against these, and the synthetic fixture in the tests copies them verbatim):

- `closed_today[]` items carry `net` (= realized_pnl) and `gross` (= gross_pnl), not `realized_pnl`/`gross_pnl`.
- `account.unsettled_until` is an ET date string `'YYYY-MM-DD'` (or `None`); it is rendered verbatim as `settles 2026-09-15 09:30 ET` — never parsed as a timestamp.
- `classes[cls]['exec_mean_pct']`, `['exec_ci']` and `since_open['max_drawdown_pct']` are PERCENTAGE POINTS (0.2 means 0.2%), so they are formatted with `:.3f}%` / `:.2f}%`, never `:%`. Every other rate (`ev_bt_net`, `tp_*`, `sig_*`, `win_*`, `all_time_pct`, `spy.pct`, `pct_vs_ref`) is a fraction and uses `:%`.
- `classes[cls]['bt_precision']`, `['ev_bt']`, `['ev_bt_net']` are `None` (and `bt_n` is 0) whenever the class has no trained model — `intraday is None` (FAST_MODE=0) or `metrics` lacks the class. The embed prints `n/a` for those and must never raise: the 16:05 report is the one message that proves the account is alive.
- `positions[]` items carry `pct_vs_ref` (ref-to-ref move, the same number the exit rule watches); the embed prints it rather than recomputing from `entry_ref`.

**Files:**
- Create: `/home/gdhughey/hugheylab-trading-bot/tests/test_discord_embeds.py`
- Modify: `/home/gdhughey/hugheylab-trading-bot/src/discord_bot.py` — current line ranges (1216-line file at commit `dd28b31`):
  - imports, lines 7–20
  - `TradingBot.__init__`, lines 25–53 (drop `_last_daily_summary`, line 49)
  - `_heartbeat_embed`, lines 56–149
  - `_register_slash`, lines 168–362 (bind list 189–195; `/fast` 257–316; `/summary` 318–323; `/budget` 350–360; count log 362)
  - `on_ready`, lines 364–450
  - `_send_startup_notice`, lines 452–519
  - `daily_summary`, lines 551–572
  - `_daily_summary_embed`, lines 574–662 (DELETE)
  - `fast_cycle`, lines 664–703
  - `_fast_action_embed`, lines 705–732
  - `_fast_idle_embed`, lines 734–763
  - `send_trade_alert`, lines 836–949
  - `execute_approved_trade`, lines 986–1003
  - `_embed_status`, lines 1038–1044
  - `_embed_budget`, lines 1069–1091 (REPLACE with `_embed_account`)
  - `_embed_stats`, lines 1093–1100 (DELETE)
  - `_embed_pnl`, lines 1102–1140
  - `_embed_scan`, lines 1142–1186

Untouched: `_symbols`, `_warming_embed`, `_destination`, `monitor_trading`, `wait_for_approval`, `reject_trade`, `auto_reject_trade`, `_embed_daily_brief`, `_embed_risk_check`, `_embed_pause`, `_embed_resume`, `run`, the `__main__` block.

- [ ] **Step 1: Write the failing tests**

Create `/home/gdhughey/hugheylab-trading-bot/tests/test_discord_embeds.py`:

```python
"""Discord embeds built offline: no gateway login, no network.

TradingBot.__init__ builds its own BudgetTracker / IntradayEngine / FastTrader,
so those names are patched on the module to fakes (the tracker is real, on the
tmp-path DB). commands.Bot and discord.Embed construct without a connection.

The synthetic scorecard below is copied from Task 7's contract additions
(tests/test_scorecard.py::test_build_scorecard_full is the producer-side
twin): closed_today uses net/gross, unsettled_until is a bare ET date,
exec_mean_pct / exec_ci / max_drawdown_pct are percentage points, and a class
without a trained model has bt_precision / ev_bt / ev_bt_net None.
"""
import asyncio
import types
from datetime import datetime, timezone

import discord
import pytest

from src import discord_bot
from src.budget_tracker import BudgetTracker

# Walk-forward metrics from the spec's gate test: stock below the EV floor,
# crypto above zero but under it. (data/ is not shipped to the dev copy, so
# barriers() falls back to FAST_TAKE_PROFIT / FAST_STOP_LOSS defaults and the
# crypto cost gate does NOT fire here - nothing below depends on it firing.)
METRICS = {
    'stock': {'asset_class': 'stock', 'symbols': 80, 'rows': 291156,
              'positive_rate': 0.25, 'interval': '5m', 'horizon_bars': 24,
              'take_profit': 0.01, 'stop_loss': 0.006, 'precision': 0.357,
              'breakeven': 0.375, 'ev': -0.0003, 'test_signals': 947, 'bar': 0.507},
    'crypto': {'asset_class': 'crypto', 'symbols': 13, 'rows': 172332,
               'positive_rate': 0.26, 'interval': '5m', 'horizon_bars': 24,
               'take_profit': 0.006, 'stop_loss': 0.004, 'precision': 0.463,
               'breakeven': 0.40, 'ev': 0.0006, 'test_signals': 378, 'bar': 0.525},
}


class FakeEngine:
    symbols = []
    providers = []
    quote_providers = []
    source_stats = {}

    def latest_price(self, symbol):
        return None            # every held position is valued at avg_price

    def stored_close(self, symbol):
        return None            # no SPY history -> scorecard 'spy' is None

    def first_close_on_or_after(self, symbol, date_iso):
        return None


class FakeIntraday:
    def __init__(self, metrics):
        self.metrics = dict(metrics)
        self.last_metrics = {}
        self.symbols = ['AAPL', 'BTC-USD']

    def scan_all(self):
        return []


class FakeFast:
    max_positions = 3
    eod_flatten_min = 10.0
    cooldown_min = 15.0
    take_profit = 0.008
    stop_loss = 0.005


class FakeChannel:
    """Records what the bot tried to send; can refuse embeds the way Discord does."""

    def __init__(self, fail_embeds=False, fail_all=False):
        self.embeds, self.texts = [], []
        self.fail_embeds, self.fail_all = fail_embeds, fail_all

    async def send(self, content=None, *, embed=None):
        if self.fail_all:
            raise RuntimeError('gateway down')
        if embed is not None:
            if self.fail_embeds:
                raise discord.HTTPException(
                    types.SimpleNamespace(status=400, reason='Bad Request'),
                    'Invalid Form Body')
            self.embeds.append(embed)
        else:
            self.texts.append(content)


def _make_bot(db_path, monkeypatch, metrics=METRICS, ignore_ev='1'):
    monkeypatch.setenv('FAST_MODE', '1')
    monkeypatch.setenv('FAST_IGNORE_EV', ignore_ev)
    monkeypatch.setenv('MIN_EV_TO_TRADE', '0.003')
    monkeypatch.delenv('CLAUDE_API_KEY', raising=False)
    monkeypatch.setattr(discord_bot, 'BudgetTracker', lambda: BudgetTracker(db_path))
    monkeypatch.setattr(discord_bot, 'IntradayEngine', lambda: FakeIntraday(metrics))
    monkeypatch.setattr(discord_bot, 'FastTrader', lambda *a, **k: FakeFast())
    return discord_bot.TradingBot(engine=FakeEngine())


def _stub_report_plumbing(bot, monkeypatch, chan, stub_embed=True):
    """Point the bot at a FakeChannel and neutralise the two pieces the daily
    report tests are not about (signal labelling, and optionally the embed)."""
    async def dest():
        return chan
    monkeypatch.setattr(bot, '_destination', dest)
    monkeypatch.setattr(discord_bot.signal_log, 'label_pending', lambda conn, now=None: 0)
    if stub_embed:
        monkeypatch.setattr(bot, '_scorecard_embed',
                            lambda day_et, now=None: discord.Embed(title='stub'))


NOW = datetime(2026, 9, 19, 20, 10, tzinfo=timezone.utc)   # Saturday 16:10 ET
DAY = '2026-09-19'
OPENED_DAY = '2026-01-01'                                   # conftest OPENED_AT


def test_startup_embed_live_when_ev_ignored(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch, ignore_ev='1')
    e = bot._startup_embed()
    assert 'LIVE' in e.description
    assert e.color == discord.Color.green()
    assert not any('Not trading' in f.value for f in e.fields)
    names = [f.name for f in e.fields]
    assert 'Buying power' in names and 'Balance' in names
    assert 'Cash left' not in names


def test_startup_embed_gated_without_flag(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch, ignore_ev='0')
    e = bot._startup_embed()
    assert 'Nothing will be traded' in e.description
    assert e.color == discord.Color.orange()


def test_scorecard_embed_on_fresh_account(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch)
    e = bot._scorecard_embed(DAY, now=NOW)
    assert any(f.name.startswith('All-time') for f in e.fields)
    assert any(f.name == 'Verdict' for f in e.fields)
    # Spec section 1: the next open comes from the trading calendar. DAY is a
    # Saturday, so "tomorrow" would be wrong and Monday is right.
    today = next(f for f in e.fields if f.name.startswith('Today'))
    assert 'Stocks resume Mon 21 Sep 09:30 ET' in today.value
    assert len(e) < 6000
    assert len(e.fields) <= 25
    assert all(len(f.value) <= 1024 for f in e.fields)


def test_scorecard_embed_without_intraday_metrics(db_path, monkeypatch):
    """Fast mode on but nothing trained (same shape as FAST_MODE=0): Task 7
    returns bt_precision / ev_bt / ev_bt_net None for every class. The report
    must still render - it is the one message that proves the account is alive."""
    bot = _make_bot(db_path, monkeypatch, metrics={})
    e = bot._scorecard_embed(DAY, now=NOW)
    stock = next(f for f in e.fields if f.name.startswith('📈 Stock'))
    assert 'walk-forward precision n/a' in stock.value
    assert 'Backtest EV net of costs n/a' in stock.value
    verdict = next(f for f in e.fields if f.name == 'Verdict')
    assert 'no backtest EV' in verdict.value
    assert len(e) < 6000 and len(e.fields) <= 25


def _synthetic_scorecard(n_positions):
    """A build_scorecard() result with every key from Task 7's contract
    additions, big enough to overflow a field. Units and key names are the
    producer's, so a mismatch here would fail the producer's test too."""
    stock = {
        # verdict inputs (fractions)
        'gated': False, 'sig_n': 120, 'sig_lo': 0.30, 'sig_hi': 0.48, 'sig_lo_day': 0.28,
        'exec_n': 12, 'exec_mean': 0.001, 'exec_se': 0.002, 'ev_bt': -0.0003, 'cost': 0.00102,
        # verdict output and the rest of the class block
        'verdict': 'EXTEND', 'verdict_text': 'CI straddles breakeven',
        'breakeven': 0.4388,
        'exits_by_reason': {'tp': {'n': 5, 'mean_net': 1.2}, 'sl': {'n': 7, 'mean_net': -0.9}},
        'tp_n': 12, 'tp_first_rate': 0.42, 'tp_lo': 0.18, 'tp_hi': 0.69,
        'bt_precision': 0.357, 'bt_n': 947, 'ev_bt_net': -0.00132,
        'exec_mean_pct': 0.1, 'exec_ci': 0.392,               # PERCENTAGE POINTS
        'gate_text': 'trading on paper despite EV -0.030% below the 0.30% floor '
                     '(FAST_IGNORE_EV on)',
        'sig_hits': 47, 'sig_rate': 0.39, 'sig_days': 6,
    }
    crypto = {
        **stock, 'gated': True, 'verdict': 'NO-GO',
        'verdict_text': 'cost gate blocks crypto: round-trip cost 1.20% is at or above '
                        'the 0.60% take-profit',
        'exec_n': 0, 'exec_mean': 0.0, 'exec_se': 0.0, 'exec_mean_pct': 0.0, 'exec_ci': 0.0,
        'cost': 0.012, 'breakeven': 1.6, 'exits_by_reason': {},
        'tp_n': 0, 'tp_first_rate': 0.0, 'tp_lo': 0.0, 'tp_hi': 1.0,
        # no trained crypto model: exactly what Task 7 emits for a missing class
        'bt_precision': None, 'bt_n': 0, 'ev_bt': None, 'ev_bt_net': None,
        'gate_text': 'blocked: take-profit 0.60% is below the 1.20% round-trip cost',
        'sig_n': 9, 'sig_hits': 4, 'sig_rate': 0.444, 'sig_lo': 0.19, 'sig_hi': 0.73,
        'sig_lo_day': 0.0, 'sig_days': 1,
    }
    return {
        'headline': {'all_time_net': 3.21, 'all_time_pct': 0.00642, 'n_closed': 12,
                     'ci_dollars': 4.5},
        'account': {'equity': 503.21, 'cash': 400.0, 'buying_power': 380.0,
                    'unsettled': 20.0, 'unsettled_until': '2026-09-21',     # ET date string
                    'starting_cash': 500.0, 'gross_pnl': 3.5, 'fees_paid': 0.29},
        'today': {'realized': 1.0, 'unrealized': -0.5, 'n_trades': 4, 'wins': 1,
                  'losses': 1, 'loss_tripped': False},
        'closed_today': [{'symbol': 'AAPL', 'shares': 0.5, 'price': 101.2, 'amount': 50.6,
                          'net': 0.55, 'gross': 0.56, 'fees': 0.01, 'exit_reason': 'tp',
                          'created_at': '2026-09-19T15:10:00+00:00'}],
        'positions': [{'symbol': f'SYM{i:03d}', 'shares': 0.123456, 'avg_price': 100.05,
                       'entry_ref': 100.0, 'price': 101.0, 'market_value': 12.47,
                       'pnl': 0.12, 'pnl_pct': 0.0095, 'pct_vs_ref': 0.01}
                      for i in range(n_positions)],
        'since_open': {'n_closed': 12, 'win_rate': 0.42, 'win_lo': 0.18, 'win_hi': 0.69,
                       'mean_net': 0.27, 'mean_ci': 0.8, 'profit_factor': 1.1,
                       'max_drawdown_pct': 1.2,                             # PERCENTAGE POINTS
                       'days_running': 6},
        'classes': {'stock': stock, 'crypto': crypto},
        'spy': {'start_close': 640.0, 'last_close': 652.8, 'pct': 0.02, 'value': 510.0},
    }


def test_scorecard_embed_truncates_long_lists(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch)
    monkeypatch.setattr(discord_bot, 'build_scorecard',
                        lambda *a, **k: _synthetic_scorecard(80))
    e = bot._scorecard_embed(DAY)
    pos = next(f for f in e.fields if f.name == 'Open positions')
    assert '… and' in pos.value and pos.value.endswith('more')
    assert len(pos.value) <= 1000
    assert '0.123456 SYM000 @ $100.05 → $101.00 (+1.00% vs entry ref)' in pos.value
    closed = next(f for f in e.fields if f.name == 'Closed today')
    assert '🟩 0.5 AAPL @ $101.20 → **+$0.55** (gross +$0.56, fees $0.01) [tp]' in closed.value
    acct = next(f for f in e.fields if f.name == 'Account')
    assert 'settles 2026-09-21 09:30 ET' in acct.value
    since = next(f for f in e.fields if f.name == 'Scorecard since open')
    assert 'max drawdown 1.20%' in since.value                  # pp, not 120.00%
    stock = next(f for f in e.fields if f.name.startswith('📈 Stock'))
    assert 'walk-forward precision 35.7% (n=947)' in stock.value
    assert ('Backtest EV net of costs -0.132%/trade vs realised +0.100% ± 0.392%'
            in stock.value)                                     # pp, not +10.000%
    crypto = next(f for f in e.fields if f.name.startswith('🪙 Crypto'))
    assert 'walk-forward precision n/a' in crypto.value
    assert 'Backtest EV net of costs n/a' in crypto.value
    assert any(f.name == 'Verdict' and 'NO-GO' in f.value for f in e.fields)
    assert len(e) < 6000 and len(e.fields) <= 25


def test_daily_report_text_fallback_on_http_error(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch)
    chan = FakeChannel(fail_embeds=True)
    _stub_report_plumbing(bot, monkeypatch, chan)

    assert asyncio.run(bot._post_daily_report(DAY, now=NOW)) is True
    assert chan.embeds == []
    assert len(chan.texts) == 1 and 'all-time' in chan.texts[0].lower()
    assert bot.budget_tracker.get_day_state(DAY)['report_posted_at'] is not None
    assert any(r['date'] == DAY for r in bot.budget_tracker.equity_series())
    # Guard holds: a second tick on the same date is a no-op.
    assert asyncio.run(bot._post_daily_report(DAY, now=NOW)) is False
    assert len(chan.texts) == 1

    # Spec section 5: posts once per date across a restart inside the window.
    # A brand-new TradingBot (new BudgetTracker on the same file) reads the
    # flag from day_state, not from memory.
    bot2 = _make_bot(db_path, monkeypatch)
    chan2 = FakeChannel()
    _stub_report_plumbing(bot2, monkeypatch, chan2)
    assert asyncio.run(bot2._post_daily_report(DAY, now=NOW)) is False
    assert chan2.embeds == [] and chan2.texts == []


def test_daily_report_retries_when_send_fails(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch)
    chan = FakeChannel(fail_all=True)
    _stub_report_plumbing(bot, monkeypatch, chan)

    with pytest.raises(RuntimeError):
        asyncio.run(bot._post_daily_report(DAY, now=NOW))
    # Flag stays NULL so the next tick retries, but the books were closed first.
    assert bot.budget_tracker.get_day_state(DAY)['report_posted_at'] is None
    assert any(r['date'] == DAY for r in bot.budget_tracker.equity_series())

    # Discord is back: the next tick posts and sets the flag; the equity row
    # is upserted, not duplicated.
    chan.fail_all = False
    assert asyncio.run(bot._post_daily_report(DAY, now=NOW)) is True
    assert [e.title for e in chan.embeds] == ['stub']
    assert bot.budget_tracker.get_day_state(DAY)['report_posted_at'] is not None
    assert sum(1 for r in bot.budget_tracker.equity_series() if r['date'] == DAY) == 1


def test_pnl_before_close_does_not_suppress_report(db_path, monkeypatch):
    """Spec section 5: /pnl renders the same scorecard for the same date but
    never writes day_state or equity_history, so the 16:05 tick still posts."""
    bot = _make_bot(db_path, monkeypatch)
    chan = FakeChannel()
    _stub_report_plumbing(bot, monkeypatch, chan, stub_embed=False)

    e = asyncio.run(bot._embed_pnl(now=NOW))
    assert any(f.name.startswith('All-time') for f in e.fields)
    assert bot.budget_tracker.get_day_state(DAY) is None
    assert [r['date'] for r in bot.budget_tracker.equity_series()] == [OPENED_DAY]

    assert asyncio.run(bot._post_daily_report(DAY, now=NOW)) is True
    assert len(chan.embeds) == 1 and chan.embeds[0].title == f"📊 Paper scorecard — {DAY}"
    assert [r['date'] for r in bot.budget_tracker.equity_series()] == [OPENED_DAY, DAY]


def test_fast_action_embed_reads_ledger_fields(db_path, monkeypatch):
    bot = _make_bot(db_path, monkeypatch)
    summary = {
        'entries': [{'symbol': 'AAPL', 'shares': 0.5, 'price': 100.05, 'ref_price': 100.0,
                     'probability': 0.61, 'cost': 50.025, 'fees': 0.0, 'trade_id': 1}],
        'exits': [{'symbol': 'MSFT', 'shares': 0.25, 'price': 401.8, 'ref_price': 402.0,
                   'reason': 'take profit +1.01%', 'exit_reason': 'tp', 'pnl': 0.93,
                   'gross': 0.94, 'fees': 0.01, 'pct': 0.0101, 'trade_id': 2}],
        'equity': 501.2, 'buying_power': 350.0, 'unsettled': 100.45,
        'minutes_to_close': 90.0,
    }
    e = bot._fast_action_embed(summary)
    names = [f.name for f in e.fields]
    assert 'BOUGHT 0.5 AAPL @ $100.05' in names
    assert 'SOLD 0.25 MSFT @ $401.80' in names
    assert {'Balance', 'Buying power', 'Unsettled'} <= set(names)
    assert 'Cash left' not in names
    sold = next(f for f in e.fields if f.name.startswith('SOLD'))
    assert 'fees $0.01' in sold.value and '[tp]' in sold.value


def test_clip_lines_appends_remainder():
    lines = [f"line {i:03d} " + 'x' * 40 for i in range(60)]
    text = discord_bot._clip_lines(lines, limit=500)
    assert len(text) <= 500
    assert text.endswith('more')
    assert discord_bot._clip_lines(['a', 'b']) == 'a\nb'
```

- [ ] **Step 2: Run the tests and watch them fail**

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_discord_embeds.py -v
```

Expected: `10 failed`. Every bot test fails with an `AttributeError` naming something that does not exist yet: `'TradingBot' object has no attribute '_startup_embed'` (2 startup tests), `'_scorecard_embed'` (3 scorecard tests and the 2 daily-report tests, raised by `monkeypatch.setattr`), `module 'src.discord_bot' has no attribute 'signal_log'` (`test_pnl_before_close_does_not_suppress_report`), `'BudgetTracker' object has no attribute 'get_remaining_budget'` (`test_fast_action_embed_reads_ledger_fields` — Task 4 deleted it); `test_clip_lines_appends_remainder` fails with `AttributeError: module 'src.discord_bot' has no attribute '_clip_lines'`.

- [ ] **Step 3: Imports, module helpers and `__init__`**

Replace lines 7–22 of `src/discord_bot.py` (from `import discord` through `logger = ...`) with:

```python
import discord
from discord.ext import commands, tasks
from discord import app_commands
import os
import asyncio
import math
from datetime import datetime, timezone, time as dtime
import logging
from src import costs, signal_log
from src.claude_analyzer import ClaudeAnalyzer
from src.budget_tracker import BudgetTracker
from src.costs import qty_str
from src.database import connect
from src.ml_engine import load_universe
from src.intraday_engine import (IntradayEngine, market_state,
                                 minutes_to_close, ET, is_crypto,
                                 asset_class, barriers, next_trading_day_open)
from src.fast_trader import FastTrader, class_gate, max_hold_min
from src.scorecard import build_scorecard, cost_breakeven

logger = logging.getLogger(__name__)


def _signed_usd(x: float) -> str:
    """'+$1.23' / '-$1.23' - the sign goes before the dollar sign, which an
    f-string format spec cannot do on its own."""
    return f"{'-' if x < 0 else '+'}${abs(x):,.2f}"


def _icon(cls: str) -> str:
    return '🪙' if cls == 'crypto' else '📈'


def _clip_lines(lines, limit: int = 1000) -> str:
    """Join lines for one embed field, dropping trailing lines until the text
    (including its '… and N more' tail) fits. Discord caps a field value at
    1024 chars; 1000 leaves headroom for markdown the caller adds."""
    lines = list(lines)
    kept = len(lines)
    while kept > 0:
        text = "\n".join(lines[:kept])
        if kept < len(lines):
            text += f"\n… and {len(lines) - kept} more"
        if len(text) <= limit:
            return text
        kept -= 1
    return f"… and {len(lines)} more"
```

Then in `TradingBot.__init__` (lines 25–53) delete the single line `self._last_daily_summary = None` (line 49). The rest of `__init__` is unchanged.

- [ ] **Step 4: Startup notice via `class_gate`; shared account/gate helpers**

Replace `_send_startup_notice` (lines 452–519) with these six methods:

```python
    def _class_gates(self):
        """[(cls, tradeable, text)] for every trained class, from the ONE gate
        FastTrader uses - so the log, the startup notice and the trades can
        never disagree about what is being traded."""
        if not self.fast_mode or not self.intraday:
            return []
        metrics = self.intraday.metrics or {}
        return [(cls, *class_gate(cls, metrics[cls])) for cls in sorted(metrics)]

    def _gate_lines(self) -> str:
        return "\n".join(f"{'✅' if ok else '⛔'} {_icon(cls)} {cls}: {text}"
                         for cls, ok, text in self._class_gates())

    def _held_prices(self) -> dict:
        """Quotes for held symbols only, keyed by symbol. A failed lookup is
        left out so get_equity() values that position at avg_price."""
        prices = {}
        for p in self.budget_tracker.get_positions():
            try:
                q = self.engine.latest_price(p['symbol'])
            except Exception as exc:
                logger.warning(f"quote failed for {p['symbol']}: {exc}")
                q = None
            if q:
                prices[p['symbol']] = float(q)
        return prices

    def _add_account_fields(self, e, s=None):
        """The Balance / Buying power / Unsettled trio every money embed shows.
        Prefers the cycle summary's figures (quoted this cycle, no second
        lookup) and falls back to the ledger."""
        s = s or {}
        bt = self.budget_tracker
        equity = s.get('equity')
        bp = s.get('buying_power')
        unsettled = s.get('unsettled')
        if equity is None:
            equity = bt.get_equity(self._held_prices())
        if bp is None:
            bp = bt.get_buying_power()
        if unsettled is None:
            unsettled = bt.get_unsettled()
        e.add_field(name="Balance",
                    value=f"${equity:,.2f} (started with ${bt.starting_cash():,.2f})",
                    inline=True)
        e.add_field(name="Buying power", value=f"${bp:,.2f}", inline=True)
        e.add_field(name="Unsettled", value=f"${unsettled:,.2f}", inline=True)

    def _startup_embed(self) -> discord.Embed:
        """The restart notice. Headline and colour come from whether ANY class
        clears class_gate; 'Nothing will be traded' only when every class is
        gated (cost floor or EV)."""
        gates = self._class_gates()
        will_trade = [cls for cls, ok, _ in gates if ok]
        metrics = (self.intraday.metrics or {}) if gates else {}

        e = discord.Embed(
            title="🔄 Trading bot restarted",
            description=("**Trading is LIVE** - you will hear from me when I buy or sell."
                         if will_trade else
                         "**Nothing will be traded right now.** Every asset class is "
                         "gated (cost floor or expected value), so the bot is watching "
                         "only. This is the risk gate working, not a crash."),
            color=discord.Color.green() if will_trade else discord.Color.orange(),
            timestamp=datetime.now().astimezone())

        for cls, ok, text in gates:
            m = metrics[cls]
            e.add_field(
                name=f"{_icon(cls)} {cls.title()}",
                value=(f"{'✅' if ok else '⛔'} {text}\n"
                       f"Gets it right {m.get('precision', 0):.1%} of the time; needs "
                       f"{m.get('breakeven', 0):.1%} just to break even."),
                inline=False)

        if not gates:
            e.add_field(name="Models",
                        value="No intraday model is loaded - fast mode is off.",
                        inline=False)

        try:
            held = self.budget_tracker.get_positions()
            e.add_field(
                name="Open paper positions",
                value=("none" if not held else
                       ", ".join(f"{qty_str(p['shares'])} {p['symbol']}" for p in held)),
                inline=False)
            self._add_account_fields(e)
        except Exception as exc:
            logger.warning(f"startup notice: account fields skipped: {exc}")

        e.set_footer(text="Quiet mode: no routine updates. You only hear from me "
                          "when I trade, or when I restart. PAPER TRADING.")
        return e

    async def _send_startup_notice(self):
        """Post exactly one message on startup saying whether the bot will
        trade and, if not, why.

        Routine chatter is off (HEARTBEAT=0, FAST_SUMMARY_MINUTES=0), so the
        channel is silent unless a trade happens. That makes a dead process and
        a deliberately idle one look identical. This notice is the one message
        that distinguishes them: it fires on every restart, and it names the
        gate that is blocking each asset class.
        """
        if os.getenv('STARTUP_NOTICE', '1') in ('0', 'false', 'no'):
            return
        channel = await self._destination()
        if not channel:
            return
        # _startup_embed quotes held symbols - keep that off the event loop.
        await channel.send(embed=await asyncio.to_thread(self._startup_embed))
```

- [ ] **Step 5: Run the startup tests, commit**

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_discord_embeds.py -v -k "startup or clip_lines"
```

Expected: `3 passed` (`test_startup_embed_live_when_ev_ignored`, `test_startup_embed_gated_without_flag`, `test_clip_lines_appends_remainder`).

```
cd /home/gdhughey/hugheylab-trading-bot && git add src/discord_bot.py tests/test_discord_embeds.py && git commit -m "discord: startup notice and helpers use class_gate and account labels" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 6: `_scorecard_embed`; `/pnl` and `/summary` use it**

Replace `_embed_pnl` (lines 1102–1140) with:

```python
    async def _embed_pnl(self, now=None):
        """/pnl and /summary: today's scorecard. Never writes day_state or
        equity_history - only the 16:05 tick does that. `now` (UTC, tz-aware)
        is for tests; the slash commands call this with no arguments."""
        now = now or datetime.now(timezone.utc)
        day_et = now.astimezone(ET).strftime('%Y-%m-%d')
        return await asyncio.to_thread(self._scorecard_embed, day_et, now)
```

Insert `_scorecard_embed` directly BELOW the `daily_summary` loop (after line 572, where `_daily_summary_embed` currently starts; that function is deleted in Step 8):

```python
    def _scorecard_embed(self, day_et: str, now=None) -> discord.Embed:
        """Render build_scorecard() for one ET date. Shared by the 16:05
        report, /pnl and /summary so there is one layout to get right.

        Units follow Task 7's contract: exec_mean_pct, exec_ci and
        max_drawdown_pct are PERCENTAGE POINTS (formatted with :f and a literal
        %); every other rate is a fraction (formatted with :%). A class without
        a trained model has bt_precision / ev_bt_net None and prints n/a - the
        report must render even when training failed or fast mode is off.

        Every list field goes through _clip_lines: Discord refuses the whole
        message when any field passes 1024 chars or the total passes 6000.
        """
        sc = build_scorecard(self.budget_tracker, self.engine, self.intraday,
                             day_et, now=now)
        h, acct, today = sc['headline'], sc['account'], sc['today']
        opened = self.budget_tracker.opened_at()[:10]
        net = h['all_time_net']
        colour = (discord.Color.green() if net > 0 else
                  discord.Color.red() if net < 0 else discord.Color.greyple())
        e = discord.Embed(title=f"📊 Paper scorecard — {day_et}", color=colour,
                          timestamp=datetime.now().astimezone())

        # The $ headline is never shown without its n and CI (spec section 8).
        e.add_field(
            name=f"All-time: {_signed_usd(net)} ({h['all_time_pct']:+.2%})",
            value=(f"since {opened} · n={h['n_closed']} closed trade(s), "
                   f"95% CI ±${h['ci_dollars']:,.2f}"),
            inline=False)

        classes = sc['classes']
        e.add_field(
            name="Verdict",
            value=_clip_lines([f"{_icon(cls)} **{cls}: {c['verdict']}** — {c['verdict_text']}"
                               for cls, c in sorted(classes.items())]) or "no classes scored",
            inline=False)

        # unsettled_until is already the ET date ('YYYY-MM-DD') on which the
        # proceeds settle at 09:30 ET; print it as-is, no timezone maths.
        settles = ""
        if acct['unsettled'] > 0 and acct.get('unsettled_until'):
            settles = f" (settles {acct['unsettled_until']} 09:30 ET)"
        e.add_field(
            name="Account",
            value=(f"Gross P&L {_signed_usd(acct['gross_pnl'])} · "
                   f"fees paid ${acct['fees_paid']:,.2f}\n"
                   f"Balance **${acct['equity']:,.2f}** "
                   f"(started with ${acct['starting_cash']:,.2f})\n"
                   f"Cash ${acct['cash']:,.2f} · buying power ${acct['buying_power']:,.2f}\n"
                   f"Unsettled ${acct['unsettled']:,.2f}{settles}"),
            inline=False)

        block = ("🛑 daily-loss block TRIPPED" if today['loss_tripped']
                 else "daily-loss block not tripped")
        # Spec section 1: the next open comes from the trading calendar, so a
        # Friday, weekend or holiday-eve report names the right day.
        resume = next_trading_day_open(datetime.fromisoformat(day_et).replace(tzinfo=ET))
        e.add_field(
            name=f"Today ({day_et})",
            value=(f"Realised {_signed_usd(today['realized'])} · "
                   f"unrealised {_signed_usd(today['unrealized'])}\n"
                   f"{today['n_trades']} trade(s) · won {today['wins']} · "
                   f"lost {today['losses']}\n{block}\n"
                   f"⏰ Stocks resume {resume.strftime('%a %d %b')} 09:30 ET; "
                   f"crypto keeps trading."),
            inline=False)

        if sc['closed_today']:
            e.add_field(
                name="Closed today",
                value=_clip_lines([
                    f"{'🟩' if r['net'] > 0 else '🟥'} {qty_str(r['shares'])} "
                    f"{r['symbol']} @ ${r['price']:,.2f} → **{_signed_usd(r['net'])}** "
                    f"(gross {_signed_usd(r['gross'])}, fees ${r['fees']:,.2f}) "
                    f"[{r['exit_reason']}]"
                    for r in sc['closed_today'][:10]]),
                inline=False)

        if sc['positions']:
            lines = []
            for p in sc['positions']:
                head = (f"{_icon(asset_class(p['symbol']))} {qty_str(p['shares'])} "
                        f"{p['symbol']} @ ${p['avg_price']:,.2f}")
                if p['price'] is None:
                    lines.append(f"{head} → price unavailable")
                    continue
                # pct_vs_ref is the ref-to-ref move the exit rule watches, not
                # the move against avg_price (which has the fill costs in it).
                pct = ("" if p['pct_vs_ref'] is None
                       else f" ({p['pct_vs_ref']:+.2%} vs entry ref)")
                lines.append(f"{head} → ${p['price']:,.2f}{pct}")
            e.add_field(name="Open positions", value=_clip_lines(lines), inline=False)

        so = sc['since_open']
        pf = so['profit_factor']
        pf_txt = f"{pf:.2f}" if pf is not None and math.isfinite(pf) else "n/a"
        e.add_field(
            name="Scorecard since open",
            value=(f"{so['n_closed']} trade(s) closed · win rate {so['win_rate']:.1%} "
                   f"(95% CI {so['win_lo']:.1%}–{so['win_hi']:.1%})\n"
                   f"Mean net per trade {_signed_usd(so['mean_net'])} ± ${so['mean_ci']:,.2f}\n"
                   f"Profit factor {pf_txt} · max drawdown {so['max_drawdown_pct']:.2f}% · "
                   f"{so['days_running']} day(s) running"),
            inline=False)

        for cls, c in sorted(classes.items()):
            exits = ", ".join(f"{r}: {v['n']} ({_signed_usd(v['mean_net'])})"
                              for r, v in sorted(c['exits_by_reason'].items())) or "none"
            if c['bt_precision'] is None:
                bt_txt = "walk-forward precision n/a (no trained model)"
            else:
                bt_txt = (f"walk-forward precision {c['bt_precision']:.1%} "
                          f"(n={c['bt_n']:,})")
            ev_txt = "n/a" if c['ev_bt_net'] is None else f"{c['ev_bt_net']:+.3%}/trade"
            e.add_field(
                name=f"{_icon(cls)} {cls.title()} — {c['exec_n']} closed",
                value=(f"Exits: {exits}\n"
                       f"TP-first {c['tp_first_rate']:.1%} (95% CI {c['tp_lo']:.1%}–"
                       f"{c['tp_hi']:.1%}) vs {bt_txt}\n"
                       f"Backtest EV net of costs {ev_txt} vs realised "
                       f"{c['exec_mean_pct']:+.3f}% ± {c['exec_ci']:.3f}%\n"
                       f"Gate: {c['gate_text']}\n"
                       f"Signals: n={c['sig_n']} labelled, TP-first {c['sig_rate']:.1%} "
                       f"(95% CI {c['sig_lo']:.1%}–{c['sig_hi']:.1%}, day-clustered low "
                       f"{c['sig_lo_day']:.1%}) vs cost-adjusted breakeven "
                       f"{cost_breakeven(cls):.1%}"),
                inline=False)

        spy = sc.get('spy')
        if spy:
            e.add_field(
                name="SPY buy-and-hold (context only)",
                value=(f"${acct['starting_cash']:,.2f} in SPY on {opened} → "
                       f"**${spy['value']:,.2f}** ({spy['pct']:+.2%}; "
                       f"${spy['start_close']:,.2f} → ${spy['last_close']:,.2f})\n"
                       f"Not risk-matched: SPY is exposed 24/7, the bot is flat overnight."),
                inline=False)

        e.set_footer(text="Backtest scores timeouts/EOD as −sl, assumes exact-barrier "
                          "fills, and samples intrabar highs/lows; live exits are checked "
                          "on one quote every FAST_POLL_SECONDS. PAPER TRADING.")
        return e
```

- [ ] **Step 7: Run the scorecard tests, commit**

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_discord_embeds.py -v -k scorecard
```

Expected: `3 passed` (`test_scorecard_embed_on_fresh_account`, `test_scorecard_embed_without_intraday_metrics`, `test_scorecard_embed_truncates_long_lists`).

```
cd /home/gdhughey/hugheylab-trading-bot && git add src/discord_bot.py && git commit -m "discord: _scorecard_embed renders build_scorecard for /pnl, /summary and the daily report" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 8: Daily report loop, `on_ready`, delete `_daily_summary_embed`**

Replace `daily_summary` (lines 551–572) with the loop plus two helpers, and DELETE `_daily_summary_embed` entirely (lines 574–662; `_scorecard_embed` from Step 6 now sits where it was):

```python
    @tasks.loop(minutes=5)
    async def daily_summary(self):
        """Post the day's scorecard once, any time after 16:05 ET - weekends
        and holidays included, so a quiet Saturday still proves the account is
        alive. The once-per-date guard is day_state.report_posted_at, not
        memory, so a restart inside the window cannot double-post."""
        now = datetime.now(ET)
        if now.time() < dtime(16, 5):
            return
        day_et = now.strftime('%Y-%m-%d')
        try:
            await self._post_daily_report(day_et)
        except Exception:
            logger.exception(f"Daily report for {day_et} failed - retrying next tick")

    def _close_the_books(self, day_et: str, now=None):
        """Steps 1-2 of the 16:05 tick: label pending signals, then freeze the
        day's equity row. Runs BEFORE any Discord call so the record of the
        day never depends on delivery."""
        bt = self.budget_tracker
        # label_pending commits per row (`with conn:`). On the ledger's own
        # connection that commit would also commit whatever the fast-cycle
        # thread has half-done inside BudgetTracker._txn() - crypto keeps that
        # loop alive after the bell - and a later rollback there would then
        # have nothing to undo. So the labelling pass gets its own connection
        # to the same file (PRAGMA database_list names it) and closes it after.
        path = bt.conn.execute("PRAGMA database_list").fetchone()['file']
        conn = connect(path)
        try:
            n = signal_log.label_pending(conn, now=now)
        finally:
            conn.close()
        row = bt.record_equity(day_et, self._held_prices(), now=now)
        logger.info(f"{day_et}: labelled {n} signal(s); equity ${row['equity']:,.2f} recorded")

    async def _post_daily_report(self, day_et: str, now=None) -> bool:
        """One attempt at the daily report. Returns True when it was posted
        (embed or text fallback), False when already posted or undeliverable.
        Any other exception propagates so the loop logs it and retries."""
        bt = self.budget_tracker
        state = bt.get_day_state(day_et)
        if state is None:
            # No cycle ran today (weekend, holiday, fast mode off) - the report
            # still needs a baseline row to hang its flag on.
            state = await asyncio.to_thread(
                lambda: bt.ensure_day_state(day_et, bt.get_equity(self._held_prices())))
        if state['report_posted_at']:
            return False

        await asyncio.to_thread(self._close_the_books, day_et, now)

        channel = await self._destination()
        if channel is None:
            logger.error(f"Daily report for {day_et}: no Discord destination - "
                         f"retrying next tick")
            return False
        try:
            embed = await asyncio.to_thread(self._scorecard_embed, day_et, now)
            await channel.send(embed=embed)
        except discord.HTTPException as exc:
            # Discord refused the embed itself (too long, bad field). A one-liner
            # still carries the headline; the flag is set so we do not spam.
            logger.error(f"Daily report embed for {day_et} rejected by Discord: {exc}")
            pnl = await asyncio.to_thread(bt.get_pnl, self.engine.latest_price)
            await channel.send(
                f"📊 Paper scorecard {day_et}: all-time {_signed_usd(pnl['all_time_net'])} "
                f"({pnl['all_time_pct']:+.2%}) — full report failed: {exc}")
        bt.set_day_flag(day_et, 'report_posted_at')
        logger.info(f"Posted daily report for {day_et}")
        return True
```

Replace `on_ready` (lines 364–450) with:

```python
    async def on_ready(self):
        """Bot startup event"""
        logger.info(f"✅ Bot logged in as {self.bot.user}")
        logger.info("📝 PAPER TRADING MODE - no broker is connected; approvals "
                    "only write to the local ledger")
        # Guild sync is instant; the global sync is what makes the commands
        # usable in DMs, and can take up to an hour to propagate.
        try:
            for guild in self.bot.guilds:
                self.bot.tree.copy_global_to(guild=guild)
                await self.bot.tree.sync(guild=guild)
                logger.info(f"⚡ Slash commands synced to '{guild.name}' (instant)")
            synced = await self.bot.tree.sync()
            logger.info(f"⚡ {len(synced)} slash commands synced globally "
                        f"(DM availability may take up to 1h to propagate)")
        except Exception as e:
            logger.error(f"Slash command sync failed: {e}")

        logger.info("📊 Starting trading monitor...")

        # The daily scorecard must post even when training fails or fast mode
        # is off: it is the one message that says whether the account is alive.
        if not self.daily_summary.is_running():
            self.daily_summary.start()

        # Startup fetch (~110s) and training (~130s) are synchronous. Run them
        # OFF the event loop - inline they block every interaction for minutes
        # and Discord answers slash commands with "application did not respond".
        # A PENDING trade's approval watcher lives in memory, so anything left
        # PENDING by a previous run can never be approved - it would just hold
        # buying power forever. Clear them at startup.
        try:
            stale = self.budget_tracker.conn.execute(
                "SELECT id, symbol, side FROM trades WHERE status = 'PENDING'").fetchall()
            for row in stale:
                self.budget_tracker.reject_trade(row['id'])
            if stale:
                logger.info(f"🧹 Cleared {len(stale)} stale PENDING trade(s) from a "
                            f"previous run: " +
                            ", ".join(f"#{r['id']} {r['side']} {r['symbol']}" for r in stale))
        except Exception as e:
            logger.error(f"Stale-pending cleanup failed: {e}")

        self.warming_up = True
        try:
            logger.info("📥 Fetching market data...")
            symbols = await asyncio.to_thread(load_universe)
            logger.info(f"📊 Universe: {len(symbols)} symbols")
            await asyncio.to_thread(self.engine.fetch_and_store_data, symbols)

            logger.info("🧠 Training ML model...")
            trained = await asyncio.to_thread(self.engine.train_model)
            if trained:
                logger.info("✅ Model trained successfully")
                if not self.monitor_trading.is_running():
                    self.monitor_trading.start()
            else:
                logger.error("❌ Model training failed")
            if self.fast_mode:
                logger.info("⚡ FAST MODE - preparing intraday model...")
                await asyncio.to_thread(self.intraday.full_fetch)
                if await asyncio.to_thread(self.intraday.train):
                    # Logged once per class at startup, from the same gate the
                    # trader applies, so the log never contradicts the trades.
                    for cls, ok, text in self._class_gates():
                        m = self.intraday.metrics[cls]
                        logger.info(f"⚡ {cls} model ready: precision "
                                    f"{m['precision']:.1%} vs breakeven "
                                    f"{m['breakeven']:.1%}, EV {m['ev'] * 100:+.3f}%/trade "
                                    f"- {'TRADEABLE' if ok else 'GATED'}: {text}; "
                                    f"bar p>{m['bar']:.3f}")
                    if not self.fast_cycle.is_running():
                        self.fast_cycle.start()
                    state, desc = market_state()
                    logger.info(f"⚡ Fast loop started every "
                                f"{os.getenv('FAST_POLL_SECONDS', 60)}s - market is {desc}")
                else:
                    logger.error("⚡ Intraday training failed - fast mode disabled")
                    self.fast_mode = False
        finally:
            self.warming_up = False
            logger.info("🟢 Ready - slash commands are live")
            try:
                await self._send_startup_notice()
            except Exception as e:
                logger.error(f"startup notice failed: {e}")
```

- [ ] **Step 9: Run the daily-report tests, commit**

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_discord_embeds.py -v -k "daily_report or pnl_before"
```

Expected: `3 passed` (`test_daily_report_text_fallback_on_http_error`, `test_daily_report_retries_when_send_fails`, `test_pnl_before_close_does_not_suppress_report`).

```
cd /home/gdhughey/hugheylab-trading-bot && git add src/discord_bot.py && git commit -m "discord: 16:05 daily report guarded by day_state, labels signals and records equity first" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 10: Fast-cycle reporting: loss-limit announcement, ledger-driven BOUGHT/SOLD, gates**

Replace `fast_cycle` (lines 664–703), `_fast_action_embed` (705–732) and `_fast_idle_embed` (734–763) with:

```python
    @tasks.loop(seconds=float(os.getenv('FAST_POLL_SECONDS', 60)))
    async def fast_cycle(self):
        """Intraday entries and exits. Reports only when something happened, or
        every FAST_SUMMARY_MINUTES, so a 60s loop doesn't spam the channel."""
        if self.warming_up or not self.fast:
            return
        # Run the trading cycle FIRST. Resolving the Discord channel before this
        # meant a DM/API failure silently stopped every stop-loss and every
        # end-of-day flatten - reporting must never gate risk management.
        try:
            summary = await asyncio.to_thread(self.fast.cycle)
        except Exception:
            logger.exception("fast cycle failed")
            return
        channel = await self._destination()
        if not channel:
            if summary['entries'] or summary['exits']:
                logger.warning("Traded but could not reach Discord to report: "
                               f"{len(summary['entries'])} in, {len(summary['exits'])} out")
            return

        # Loss-limit announcement, once per ET date. The flag lives in
        # day_state, so a restart cannot repeat it and a failed send cannot
        # lose it - the next cycle while tripped simply tries again.
        if summary.get('loss_announce') or summary.get('loss_tripped'):
            today_et = summary['ts'].strftime('%Y-%m-%d')
            try:
                ds = self.budget_tracker.get_day_state(today_et)
                if ds is not None and ds['loss_announced_at'] is None:
                    await channel.send(embed=self._loss_limit_embed(summary))
                    self.budget_tracker.set_day_flag(today_et, 'loss_announced_at')
                    logger.info(f"Announced daily loss limit for {today_et}")
            except Exception as e:
                logger.error(f"loss-limit announcement failed: {e}")

        acted = summary['entries'] or summary['exits']
        gap = float(os.getenv('FAST_SUMMARY_MINUTES', 30))
        # 0 (or less) disables the idle summary entirely: report only when the
        # bot actually did something. Without this guard a gap of 0 makes every
        # single 60s cycle "due" and floods the channel.
        due = gap > 0 and (self._last_fast_summary is None or
                           (datetime.now() - self._last_fast_summary).total_seconds() / 60 >= gap)

        if acted:
            try:
                await channel.send(embed=self._fast_action_embed(summary))
            except Exception as e:
                logger.error(f"fast action report failed: {e}")
        elif due and summary['state'] == 'open':
            try:
                await channel.send(embed=self._fast_idle_embed(summary))
                self._last_fast_summary = datetime.now()
            except Exception as e:
                logger.error(f"fast summary failed: {e}")

    def _loss_limit_embed(self, s):
        limit = float(os.getenv('DAILY_LOSS_LIMIT_PCT', 3))
        e = discord.Embed(
            title="🛑 Daily loss limit hit — no new entries today",
            description=(f"Balance is down {limit:g}% or more from this morning's "
                         f"start, so the bot stops opening positions until the next "
                         f"ET date. Exits still run for anything open."),
            color=discord.Color.red(),
            timestamp=datetime.now().astimezone())
        self._add_account_fields(e, s)
        e.set_footer(text="AUTO INTRADAY · PAPER TRADING — no broker, no real money")
        return e

    def _fast_action_embed(self, s):
        """Every number here is the ledger's: fill, amount, fees and P&L come
        from the executed row, never recomputed from the quote."""
        e = discord.Embed(
            title="⚡ Intraday activity",
            color=discord.Color.gold(),
            timestamp=datetime.now().astimezone())
        for x in s['exits']:
            verdict = "🟩 profit" if x['pnl'] > 0 else "🟥 loss" if x['pnl'] < 0 else "flat"
            e.add_field(
                name=f"SOLD {qty_str(x['shares'])} {x['symbol']} @ ${x['price']:,.2f}",
                value=(f"Why: {x['reason']} [{x['exit_reason']}]\n"
                       f"Result: **{_signed_usd(x['pnl'])}** net ({x['pct']:+.2%} ref-to-ref; "
                       f"gross {_signed_usd(x['gross'])}, fees ${x['fees']:,.2f}) — {verdict}"),
                inline=False)
        for x in s['entries']:
            cls = asset_class(x['symbol'])
            e.add_field(
                name=f"BOUGHT {qty_str(x['shares'])} {x['symbol']} @ ${x['price']:,.2f}",
                value=(f"Cost ${x['cost']:,.2f} (ref ${x['ref_price']:,.2f}, "
                       f"fees ${x['fees']:,.2f}) · model confidence {x['probability']:.1%}\n"
                       f"Will sell on **+{barriers(cls)[0]:.1%}** or "
                       f"**−{barriers(cls)[1]:.1%}** from the reference price"
                       + ("" if is_crypto(x['symbol']) else ", or before the close.")),
                inline=False)
        self._add_account_fields(e, s)
        e.add_field(name="Minutes to close",
                    value=f"{s.get('minutes_to_close', 0):.0f}", inline=True)
        gates = self._gate_lines()
        if gates:
            e.add_field(name="Class gates", value=gates, inline=False)
        e.set_footer(text="AUTO INTRADAY · PAPER TRADING — no broker, no real money")
        return e

    def _fast_idle_embed(self, s):
        positions = self.budget_tracker.get_positions()
        e = discord.Embed(
            title="⚡ Intraday check — no trades",
            color=discord.Color.greyple(),
            timestamp=datetime.now().astimezone())
        cands = s.get('candidates') or []
        e.add_field(
            name="What I looked at",
            value=(f"{len(self.intraday.symbols)} liquid names on "
                   f"{self.intraday.__class__.__module__.split('.')[-1]} "
                   f"{os.getenv('INTRADAY_INTERVAL', '5m')} bars.\n"
                   f"Selection bar: **p > {s.get('bar', 0):.3f}**. "
                   f"{len(cands)} cleared it."),
            inline=False)
        if cands:
            e.add_field(name="Closest candidates",
                        value="\n".join(f"**{c['symbol']}** {c['probability']:.1%} "
                                         f"· ${c['price']:,.2f}" for c in cands[:3]),
                        inline=False)
        e.add_field(
            name="Open positions",
            value=("\n".join(f"**{p['symbol']}** x{qty_str(p['shares'])} @ "
                              f"${p['avg_price']:,.2f}" for p in positions)
                   if positions else "none"),
            inline=False)
        gates = self._gate_lines()
        if gates:
            e.add_field(name="Class gates", value=gates, inline=False)
        if s.get('loss_tripped'):
            e.add_field(name="🛑 Daily loss limit",
                        value="Tripped for today — no new entries until the next ET "
                              "date; exits still run.",
                        inline=False)
        if s.get('note'):
            e.add_field(name="Note", value=s['note'], inline=False)
        e.set_footer(text=f"{s.get('minutes_to_close', 0):.0f} min to close · "
                          f"buying power ${s.get('buying_power', 0):,.2f} · PAPER TRADING")
        return e
```

- [ ] **Step 11: Run the fast-embed test, commit**

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_discord_embeds.py -v -k fast_action
```

Expected: `1 passed` (`test_fast_action_embed_reads_ledger_fields`).

```
cd /home/gdhughey/hugheylab-trading-bot && git add src/discord_bot.py && git commit -m "discord: fast-cycle embeds read the ledger row; loss-limit announced once per day_state" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 12: Manual alerts via `size_order`, approval flow, heartbeat, `/status`, `/account`, `/scan`, slash registrations**

Replace `send_trade_alert` (lines 836–949) with:

```python
    async def send_trade_alert(self, channel, signal):
        """Send trade alert and wait for approval.

        Sizing is the account's (size_order): a fraction of equity capped by
        buying power, fractional quantities allowed. A SELL closes the whole
        held position - there is no partial manual exit and no short side.
        """
        symbol = signal['symbol']
        side = 'BUY' if signal['signal'] == 1 else 'SELL'
        signal_type = "🟢 BUY" if side == 'BUY' else "🔴 SELL"
        price = signal['price']          # reference quote; the ledger applies slippage/fees
        prob = signal['probability']
        bt = self.budget_tracker

        if side == 'BUY':
            qty, size_usd, est_fill = bt.size_order(symbol, price)
            if qty == 0:
                # Under MIN_ORDER_USD. Say so rather than silently skipping every
                # signal until proceeds settle.
                embed = discord.Embed(
                    title=f"⚠️ SKIPPED: {symbol}",
                    description="Insufficient buying power",
                    color=discord.Color.orange())
                self._add_account_fields(embed)
                await channel.send(embed=embed)
                return
        else:
            pos = next((p for p in bt.get_positions() if p['symbol'] == symbol), None)
            if pos is None:
                logger.info(f"{symbol}: SELL signal but nothing held - skipped")
                return
            qty = pos['shares']

        est = costs.fill(symbol, side, price, qty)
        trade_id = bt.log_trade(symbol, side, price, qty,
                                probability=prob if side == 'BUY' else None,
                                exit_reason='manual' if side == 'SELL' else None)

        # AUTO_TRADE: execute straight away and report, rather than waiting on a
        # reaction. Still paper - this writes to the ledger, not to a broker.
        if os.getenv('AUTO_TRADE', '0') in ('1', 'true', 'yes'):
            row = bt.execute_trade(trade_id)
            if row is None:
                logger.error(f"AUTO {side} {symbol}: trade #{trade_id} was not pending")
                return
            done = discord.Embed(
                title=f"{'🟢 BOUGHT' if side == 'BUY' else '🔴 SOLD'} "
                      f"{qty_str(row['shares'])} {symbol} @ ${row['price']:,.2f}",
                description=(f"Automatic — no approval needed. "
                             f"{'Cost' if side == 'BUY' else 'Proceeds'} ${row['amount']:,.2f} "
                             f"(ref ${row['ref_price']:,.2f}, fees ${row['fees']:,.2f})."),
                color=discord.Color.green() if side == 'BUY' else discord.Color.red(),
                timestamp=datetime.now().astimezone(),
            )
            done.add_field(name="Confidence", value=f"{prob:.1%}", inline=True)
            if side == 'SELL':
                done.add_field(name="Realised", value=_signed_usd(row['realized_pnl']),
                               inline=True)
            self._add_account_fields(done)
            pos = next((p for p in bt.get_positions() if p['symbol'] == symbol), None)
            if pos:
                done.add_field(name="You now hold",
                               value=f"{qty_str(pos['shares'])} @ avg ${pos['avg_price']:,.2f}",
                               inline=True)
            done.set_footer(text="AUTO MODE · PAPER TRADING — no broker, no real money")
            try:
                await channel.send(embed=done)
            except Exception as e:
                logger.error(f"Auto-trade report failed for {symbol}: {e}")
            logger.info(f"AUTO {side} {qty_str(row['shares'])} {symbol} @ "
                        f"${row['price']:.2f} (trade #{trade_id})")
            return

        # Create alert embed - the fill is an estimate until execute_trade runs.
        embed = discord.Embed(
            title=f"{signal_type} {symbol}",
            description="Waiting for approval...",
            color=discord.Color.green() if side == 'BUY' else discord.Color.red()
        )
        embed.add_field(name="Ref price", value=f"${price:,.2f}", inline=True)
        embed.add_field(name="Est. fill", value=f"${est['fill_price']:,.2f}", inline=True)
        embed.add_field(name="Qty", value=qty_str(qty), inline=True)
        embed.add_field(name="Est. amount",
                        value=f"${est['net']:,.2f} (fees ${est['fees']:,.2f})", inline=True)
        embed.add_field(name="Confidence", value=f"{prob:.2%}", inline=True)
        # Spec section 4: Balance beside Buying power on every money embed,
        # the approval one included. (Buying power already reflects this
        # trade's PENDING hold.)
        self._add_account_fields(embed)
        embed.add_field(name="Approval Timeout", value="5 minutes", inline=True)
        embed.set_footer(text=f"Trade ID: {trade_id} | ✅ approve / ❌ reject "
                              f"| PAPER TRADING - no broker connected")

        # Send message. If delivery fails the trade must not stay PENDING -
        # it would hold buying power forever for an alert nobody ever saw.
        try:
            msg = await channel.send(embed=embed)
            await msg.add_reaction('✅')
            await msg.add_reaction('❌')
        except Exception as e:
            bt.reject_trade(trade_id)
            logger.error(f"Alert delivery failed for {symbol} - "
                         f"rolled back pending trade #{trade_id}: {e}")
            return

        # Store for tracking
        self.pending_approvals[trade_id] = {
            'message': msg,
            'symbol': symbol,
            'signal': signal['signal'],
            'price': price,
            'shares': qty,
            'amount': est['net'],
            'timestamp': datetime.now()
        }

        # Wait for approval
        await self.wait_for_approval(trade_id, channel)
```

Replace `execute_approved_trade` (lines 986–1003) with:

```python
    async def execute_approved_trade(self, trade_id, channel):
        """Execute approved trade"""
        trade_data = self.pending_approvals[trade_id]
        row = self.budget_tracker.execute_trade(trade_id)

        if row is None:
            # Already decided (or cleared by a restart) - report, never re-book.
            embed = discord.Embed(
                title=f"⚠️ Trade #{trade_id} could not be executed",
                description="It was no longer pending.",
                color=discord.Color.orange())
        else:
            embed = discord.Embed(
                title=f"📝 PAPER TRADE RECORDED: {trade_data['symbol']}",
                description="Logged to the paper ledger - **no broker order was placed**",
                color=discord.Color.brand_green()
            )
            embed.add_field(name="Fill", value=f"${row['price']:.2f}")
            embed.add_field(name="Qty", value=qty_str(row['shares']))
            embed.add_field(name="Amount",
                            value=f"${row['amount']:.2f} (fees ${row['fees']:.2f})")
            if row['side'] == 'SELL':
                embed.add_field(name="Realised", value=_signed_usd(row['realized_pnl']))
            self._add_account_fields(embed)

        await channel.send(embed=embed)
        del self.pending_approvals[trade_id]
        logger.info(f"📝 Trade {trade_id} recorded to paper ledger")
```

Replace `_heartbeat_embed` (lines 56–149) with:

```python
    def _heartbeat_embed(self, symbols, ranked, actionable, alerting, held,
                         min_prob, pnl):
        """Plain-English hourly report: what you own, what it's worth, what to do."""
        buys = [r for r in ranked if r['signal'] == 1]
        sells = [r for r in ranked if r['signal'] == 0]
        positions = pnl['positions']
        total_pl = pnl['unrealized']
        realized = pnl['realized']
        spent = pnl['cost_basis']
        bp = pnl['buying_power']
        univ = os.getenv('UNIVERSE', 'default')
        univ_label = 'S&P 500' if univ.lower() == 'sp500' else univ

        if total_pl > 0:
            mood, color = "📈 UP", discord.Color.green()
        elif total_pl < 0:
            mood, color = "📉 DOWN", discord.Color.red()
        else:
            mood, color = "➖ FLAT", discord.Color.greyple()

        embed = discord.Embed(
            title=f"⏱️ Hourly Check — {datetime.now().strftime('%-I:%M %p')}",
            color=color if positions else discord.Color.greyple(),
            timestamp=datetime.now().astimezone(),
        )

        # --- money ---------------------------------------------------------
        money = (f"Balance **${pnl['equity']:,.2f}** (started with "
                 f"${pnl['starting_cash']:,.2f})\n"
                 f"Buying power **${bp:,.2f}** · Unsettled ${pnl['unsettled']:,.2f}\n")
        if positions:
            pct = f" ({total_pl / spent:+.1%})" if spent else ""
            money += (f"You spent **${spent:,.2f}** on {len(positions)} position(s), "
                      f"worth **${pnl['market_value']:,.2f}** right now\n\n"
                      f"**{mood} ${abs(total_pl):,.2f}{pct}**")
            if realized:
                money += f"\nAlready banked from past sales: **${realized:,.2f}**"
        else:
            money += "You haven't bought anything yet."
        embed.add_field(name="💰 YOUR MONEY", value=money, inline=False)

        # --- holdings ------------------------------------------------------
        if positions:
            lines = []
            for p in positions[:8]:
                if p['pnl'] is None:
                    lines.append(f"**{p['symbol']}** {qty_str(p['shares'])} — price unavailable")
                    continue
                arrow = "🟩 UP" if p['pnl'] > 0 else "🟥 DOWN" if p['pnl'] < 0 else "⬜ flat"
                lines.append(
                    f"**{p['symbol']}** {qty_str(p['shares'])} — paid ${p['avg_price']:,.2f}, "
                    f"now ${p['price']:,.2f} → {arrow} **${abs(p['pnl']):,.2f}** "
                    f"({p['pnl_pct']:+.1%})")
            embed.add_field(name="📊 WHAT YOU OWN", value="\n".join(lines), inline=False)

        # --- what to do ----------------------------------------------------
        held_sells = [r for r in actionable if r['signal'] == 0]
        if alerting:
            todo = (f"**I sent you {len(alerting)} alert(s) — look just below this message.**\n"
                    f"React ✅ to take the trade, ❌ to skip it. "
                    f"If you ignore it for 5 minutes it cancels itself.")
            if held_sells:
                todo += (f"\n\n⚠️ One of them is a **SELL of something you own** — "
                         f"the model thinks it's about to drop.")
        elif not ranked:
            todo = ("**Nothing to do.** Nothing looked strong enough this hour. "
                    "That's normal — most hours are quiet.")
        elif buys and not alerting:
            todo = (f"**Nothing to do.** {len(buys)} stock(s) looked good but none fit "
                    f"your ${bp:,.2f} of buying power.")
        else:
            todo = ("**Nothing to do.** The model thinks the whole market is heading "
                    "down right now. You can't sell what you don't own, so there's "
                    "nothing to act on.")
            if positions:
                todo = ("**Nothing to do.** The model is negative on the market, but "
                        "nothing you own crossed the sell threshold. Holding is fine.")
        todo += "\n\nWant to look yourself? Type **/scan** for a live ranking, or **/pnl** for detail."
        embed.add_field(name="👉 WHAT TO DO", value=todo, inline=False)

        # --- what it checked ------------------------------------------------
        embed.add_field(
            name="🔍 WHAT I CHECKED",
            value=(f"All **{len(symbols)} {univ_label}** stocks (big US companies only), "
                   f"using yesterday's closing prices from Yahoo Finance.\n"
                   f"**{len(ranked)}** looked interesting: **{len(buys)} buy**, "
                   f"**{len(sells)} sell**. I ignored **{len(sells) - len(held_sells)}** "
                   f"sells on stocks you don't own."),
            inline=False)

        embed.set_footer(text="⚠️ PRACTICE MONEY — no broker is connected, no real trades happen. "
                              "Next check in 1 hour.")
        return embed
```

Replace `_embed_status` (lines 1038–1044) with:

```python
    async def _embed_status(self):
        data = await asyncio.to_thread(self.budget_tracker.get_pnl, self.engine.latest_price)
        embed = discord.Embed(title="💼 Portfolio Status", color=discord.Color.blue())
        embed.add_field(name="Balance", value=f"${data['equity']:,.2f}")
        embed.add_field(name="Buying power", value=f"${data['buying_power']:,.2f}")
        embed.add_field(name="Unsettled", value=f"${data['unsettled']:,.2f}")
        embed.add_field(name="Open positions", value=str(len(data['positions'])))
        embed.add_field(name="Pending Trades", value=str(len(self.pending_approvals)))
        return embed
```

Replace `_embed_budget` (lines 1069–1091) with `_embed_account`, and DELETE `_embed_stats` (lines 1093–1100):

```python
    async def _embed_account(self):
        """/account: the paper brokerage account, read-only. There is no setter
        - changing starting cash after open would corrupt the all-time return."""
        bt = self.budget_tracker
        data = await asyncio.to_thread(bt.get_pnl, self.engine.latest_price)
        embed = discord.Embed(
            title="🏦 Paper account",
            description=f"{bt.account_type()} account — starting cash is fixed at open, "
                        f"there is no setter",
            color=discord.Color.blue())
        embed.add_field(name="Balance", value=f"${data['equity']:,.2f}")
        embed.add_field(name="Cash", value=f"${data['cash']:,.2f}")
        embed.add_field(name="Unsettled", value=f"${data['unsettled']:,.2f}")
        embed.add_field(name="Buying power", value=f"${data['buying_power']:,.2f}")
        embed.add_field(name="Starting cash", value=f"${data['starting_cash']:,.2f}")
        embed.add_field(name="Opened", value=bt.opened_at())
        embed.add_field(name="All-time",
                        value=f"{_signed_usd(data['all_time_net'])} ({data['all_time_pct']:+.2%})")
        if data['stale']:
            embed.add_field(name="⚠️ Stale quotes (valued at cost)",
                            value=", ".join(data['stale']), inline=False)
        embed.set_footer(text="PAPER TRADING - no broker connected")
        return embed
```

In `_embed_scan` (lines 1142–1186) replace the block from `remaining = self.budget_tracker.get_remaining_budget()` (line 1165) through `return embed` (line 1186) with:

```python
        bp = self.budget_tracker.get_buying_power()
        min_order = float(os.getenv('MIN_ORDER_USD', 1))
        for i, sig in enumerate(ranked, 1):
            side = "🟢 BUY" if sig['signal'] == 1 else "🔴 SELL"
            embed.add_field(
                name=f"{i}. {sig['symbol']} - {side}",
                value=(f"${sig['price']:,.2f} | confidence {sig['probability']:.1%} "
                       f"| as of {sig['as_of']}"),
                inline=False,
            )
        sides = {s['signal'] for s in ranked}
        if len(sides) == 1 and len(ranked) > 3:
            embed.add_field(
                name="⚠️ One-sided",
                value=("Every result is the same direction - the pooled model is "
                       "making a single market-wide call, not picking names. "
                       "Treat this as one bet, not a diversified list."),
                inline=False,
            )
        # Fractional shares mean price never gates a buy; only buying power does.
        if bp < min_order:
            embed.add_field(
                name="⚠️ Insufficient buying power",
                value=(f"${bp:,.2f} is under the ${min_order:,.2f} minimum order - "
                       f"nothing can be bought until proceeds settle."),
                inline=False)
        embed.set_footer(text=f"Buying power ${bp:,.2f} | "
                              f"PAPER TRADING - model edge is ~1pp, treat as a shortlist")
        return embed
```

Finally replace `_register_slash` (lines 168–362) in full:

```python
    def _register_slash(self):
        """Expose every command as a / slash command.

        All of these defer first: Discord kills an interaction that isn't
        acknowledged within 3 seconds, and scan/Claude calls take longer.
        """
        tree = self.bot.tree

        def bind(name, description, builder):
            @tree.command(name=name, description=description)
            async def _cmd(interaction: discord.Interaction):
                await interaction.response.defer(thinking=True)
                try:
                    embed = self._warming_embed() or await builder()
                except Exception as e:
                    logger.exception(f"/{name} failed")
                    embed = discord.Embed(title=f"/{name} failed", description=str(e),
                                          color=discord.Color.red())
                await interaction.followup.send(embed=embed)
            return _cmd

        bind('status', 'Portfolio status and buying power', self._embed_status)
        bind('account', 'Paper account: balance, cash, unsettled, buying power',
             self._embed_account)
        bind('pnl', "Today's scorecard: all-time P&L and the go/no-go verdict",
             self._embed_pnl)
        bind('summary', "Today's scorecard (same as /pnl)", self._embed_pnl)
        bind('daily_brief', 'Claude daily market analysis', self._embed_daily_brief)
        bind('risk_check', 'Claude risk assessment of open positions', self._embed_risk_check)
        bind('pause', 'Pause the monitoring loop', self._embed_pause)
        bind('resume', 'Resume the monitoring loop', self._embed_resume)

        @tree.command(name='scan', description='Rank the whole universe by model confidence')
        @app_commands.describe(top='How many to show (1-20, default 10)')
        async def _scan(interaction: discord.Interaction, top: int = 10):
            await interaction.response.defer(thinking=True)
            try:
                embed = self._warming_embed() or await self._embed_scan(top)
            except Exception as e:
                logger.exception("/scan failed")
                embed = discord.Embed(title="/scan failed", description=str(e),
                                      color=discord.Color.red())
            await interaction.followup.send(embed=embed)

        @tree.command(name='retrain',
                      description='Refresh prices from all sources and retrain (~4 min)')
        @app_commands.describe(fetch='Also re-download prices (default yes)')
        async def _retrain(interaction: discord.Interaction, fetch: bool = True):
            await interaction.response.defer(thinking=True)
            if self.warming_up:
                await interaction.followup.send(embed=self._warming_embed())
                return
            await interaction.followup.send(
                "🔄 Refreshing data and retraining — this takes about 4 minutes. "
                "I'll post the result here when it's done.")
            try:
                self.warming_up = True
                if fetch:
                    await asyncio.to_thread(
                        self.engine.fetch_and_store_data, await self._symbols())
                if self.intraday:
                    await asyncio.to_thread(self.intraday.full_fetch)
                ok = await asyncio.to_thread(self.engine.train_model)
                m = self.engine.last_metrics
                if ok and m:
                    embed = discord.Embed(
                        title="✅ Retrained",
                        color=discord.Color.green(),
                        description=(f"Target **{m['mode']}** on **{m['rows']:,}** rows\n"
                                     f"Accuracy **{m['accuracy']:.3f}** vs baseline "
                                     f"**{m['baseline']:.3f}** → edge **{m['edge']:+.3f}**"))
                    if m['edge'] <= 0.005:
                        embed.add_field(
                            name="⚠️ Reality check",
                            value=("An edge at or below 0.005 is indistinguishable from "
                                   "noise. Treat signals as a shortlist, not a prediction."),
                            inline=False)
                    src = ", ".join(f"{k}: {v['symbols']}"
                                    for k, v in self.engine.source_stats.items()) or "cached"
                    embed.add_field(name="Data sources", value=src, inline=False)
                else:
                    embed = discord.Embed(title="❌ Training failed",
                                          description="Not enough usable data.",
                                          color=discord.Color.red())
            except Exception as e:
                logger.exception("/retrain failed")
                embed = discord.Embed(title="❌ /retrain failed", description=str(e),
                                      color=discord.Color.red())
            finally:
                self.warming_up = False
            await interaction.followup.send(embed=embed)

        @tree.command(name='fast', description='Intraday auto-trading status')
        async def _fast(interaction: discord.Interaction):
            await interaction.response.defer(thinking=True)
            if not self.fast_mode or not self.intraday:
                await interaction.followup.send(embed=discord.Embed(
                    title="Fast mode is off",
                    description="Set `FAST_MODE=1` in .env and restart to enable "
                                "intraday auto-trading.",
                    color=discord.Color.greyple()))
                return
            state, desc = market_state()
            e = discord.Embed(
                title=f"⚡ Intraday mode — market is {desc}",
                color=discord.Color.gold() if state == 'open' else discord.Color.greyple())
            metrics = self.intraday.metrics or {}
            for cls, ok, text in self._class_gates():
                mm = metrics[cls]
                e.add_field(
                    name=f"{_icon(cls)} {cls.title()} model ({mm.get('symbols', 0)} symbols)",
                    value=(f"Target **+{mm.get('take_profit', 0):.2%}** before "
                           f"**−{mm.get('stop_loss', 0):.2%}** within "
                           f"{mm.get('horizon_bars', 0) * 5} min\n"
                           f"Precision **{mm.get('precision', 0):.1%}** vs break-even "
                           f"**{mm.get('breakeven', 0):.1%}** "
                           f"({mm.get('test_signals', 0):,} held-out signals)\n"
                           f"**EV {mm.get('ev', 0) * 100:+.3f}% per trade** — "
                           f"{'✅' if ok else '⛔'} {text}\n"
                           f"Bar p>{mm.get('bar', 0):.3f} · {mm.get('rows', 0):,} rows"),
                    inline=False)
            if not metrics:
                e.add_field(name="Model", value="not trained yet", inline=False)
            # Max hold is derived from each class's label horizon, not a setting.
            rules = "\n".join(
                f"{_icon(c)} {c}: take profit **+{barriers(c)[0]:.1%}** · "
                f"stop **−{barriers(c)[1]:.1%}** · max hold **{max_hold_min(c):.0f} min**"
                for c in (sorted(metrics) or ['stock']))
            e.add_field(
                name="Rules",
                value=(f"Poll every **{os.getenv('FAST_POLL_SECONDS', 60)}s** · "
                       f"max **{self.fast.max_positions}** positions · flatten stocks "
                       f"**{self.fast.eod_flatten_min:.0f} min** before the close · "
                       f"re-entry cooldown **{self.fast.cooldown_min:.0f} min**\n{rules}"),
                inline=False)
            ranked = await asyncio.to_thread(self.intraday.scan_all)
            e.add_field(
                name="Right now (each scored against its own class bar)",
                value=("\n".join(
                    f"{'✅' if r['above_bar'] else '▫️'} "
                    f"{_icon(r['asset_class'])} "
                    f"**{r['symbol']}** {r['probability']:.1%} "
                    f"(bar {r['bar']:.3f}, {r['margin']:+.3f}) · ${r['price']:,.2f}"
                    for r in ranked[:8]) or "no scores yet"),
                inline=False)
            pos = self.budget_tracker.get_positions()
            e.add_field(name="Open positions",
                        value=("\n".join(f"**{p['symbol']}** x{qty_str(p['shares'])} @ "
                                          f"${p['avg_price']:,.2f}" for p in pos)
                               if pos else "none"), inline=False)
            e.set_footer(text="PAPER TRADING — no broker connected")
            await interaction.followup.send(embed=e)

        @tree.command(name='sources', description='Show where market data is coming from')
        async def _sources(interaction: discord.Interaction):
            await interaction.response.defer(thinking=True)
            e = discord.Embed(title="📡 Market data sources", color=discord.Color.blurple())
            hist = [p.name for p in self.engine.providers if p.provides_history]
            quotes = [p.name for p in self.engine.quote_providers]
            e.add_field(name="Price history (in order)",
                        value=" → ".join(hist) or "none", inline=False)
            e.add_field(name="Live quotes (in order)",
                        value=" → ".join(quotes) or "none — using last stored close",
                        inline=False)
            if self.engine.source_stats:
                e.add_field(
                    name="Last refresh",
                    value="\n".join(f"**{k}**: {v['symbols']} symbols, {v['rows']:,} rows"
                                     for k, v in self.engine.source_stats.items()),
                    inline=False)
            e.add_field(
                name="How the chain works",
                value=("Each source only sees the symbols the previous one missed, so a "
                       "rate-limited source is spent on real gaps. Live quotes are also "
                       "cross-checked against stored closes — a big gap means stale history."),
                inline=False)
            await interaction.followup.send(embed=e)

        logger.info("Registered 12 slash commands")
```

- [ ] **Step 13: Full run, leftover-name sweep, commit**

```
cd /home/gdhughey/hugheylab-trading-bot && grep -nE "get_remaining_budget|weekly_budget|get_weekly_spent|can_trade|get_statistics|set_weekly_budget|return_pct|_last_daily_summary|_daily_summary_embed|_et_day_label|_embed_stats|_embed_budget|max_hold_min\b[^(]|Cash left|Remaining Budget|Budget Remaining|timedelta|realized_pnl'\]|gross_pnl'\]" src/discord_bot.py | grep -vE "row\['(realized_pnl|gross_pnl)'\]"; echo "exit=$?"
```

Expected: no output lines, `exit=1` (nothing matched). The final `grep -v` lets the ledger-row reads in `send_trade_alert` / `execute_approved_trade` (`row['realized_pnl']`) through; anything else that prints — in particular an `r['realized_pnl']` / `r['gross_pnl']` inside `_scorecard_embed`, which would be the closed_today key mismatch — still references a deleted API or the wrong key set and must be fixed before running the suite.

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh tests/test_discord_embeds.py -v
```

Expected: `10 passed`.

```
cd /home/gdhughey/hugheylab-trading-bot && dev/ct-test.sh -q
```

Expected: every test file passes (`N passed`, zero failed) — this confirms the import surface of `src.discord_bot` still resolves against Tasks 4–7.

```
cd /home/gdhughey/hugheylab-trading-bot && git add src/discord_bot.py && git commit -m "discord: size_order-driven alerts, /account replaces /budget, /stats removed, buying-power labels everywhere" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

**Contract additions**
- `TradingBot._startup_embed(self) -> discord.Embed` — pure builder split out of `_send_startup_notice` so the notice is testable offline.
- `TradingBot._class_gates(self) -> list[tuple[str, bool, str]]` — `(cls, tradeable, text)` per trained class via `class_gate(cls, self.intraday.metrics[cls])`.
- `TradingBot._gate_lines(self) -> str` — `_class_gates` rendered one line per class (empty string when fast mode is off).
- `TradingBot._held_prices(self) -> dict` — `{symbol: latest_price}` for held symbols; failed/None quotes omitted.
- `TradingBot._add_account_fields(self, embed, summary=None) -> None` — adds the Balance / Buying power / Unsettled fields from the cycle summary when present, else from the ledger.
- `TradingBot._scorecard_embed(self, day_et: str, now=None) -> discord.Embed` — as in the contract, plus an optional `now` passed through to `build_scorecard`. Consumes Task 7's contract additions as written: `closed_today[]` keys `net`/`gross`; `account.unsettled_until` as a bare ET date rendered verbatim (`settles YYYY-MM-DD 09:30 ET`); `exec_mean_pct`/`exec_ci`/`max_drawdown_pct` as percentage points (`:.3f}%` / `:.2f}%`); `bt_precision`/`ev_bt_net` `None` → `n/a`; `positions[].pct_vs_ref` printed directly. Adds the spec section 1 "Stocks resume <Day> 09:30 ET" line from `next_trading_day_open(<day_et at 00:00 ET>)`.
- `async TradingBot._embed_pnl(self, now=None) -> discord.Embed` — `/pnl` and `/summary`; `now` (UTC, tz-aware) picks the ET date and is threaded into `build_scorecard`; slash commands call it with no arguments. Never writes `day_state` or `equity_history`.
- `async TradingBot._post_daily_report(self, day_et: str, now=None) -> bool` — one attempt at the daily report (guard → label → equity → send → flag; HTTPException fallback); `daily_summary` calls it.
- `TradingBot._close_the_books(self, day_et: str, now=None) -> None` — `signal_log.label_pending` on a fresh `connect(<ledger file from PRAGMA database_list>)` connection (closed afterwards; never `BudgetTracker.conn`, whose autocommit + `BEGIN IMMEDIATE` discipline `label_pending`'s `with conn:` commits would break), then `record_equity`; run in a worker thread.
- `TradingBot._loss_limit_embed(self, summary) -> discord.Embed`.
- `async TradingBot._embed_account(self) -> discord.Embed` — named in the contract; signature fixed here (no parameters). Fields: Balance, Cash, Unsettled, Buying power, Starting cash, Opened (= `account.opened_at`, the field Task 10 Step 11 checks), All-time, plus a stale-quotes field when any position has no quote.
- Module-level in `src/discord_bot.py`: `_clip_lines(lines, limit: int = 1000) -> str`, `_signed_usd(x: float) -> str`, `_icon(cls: str) -> str`. (No `_et_day_label`: `unsettled_until` is a date string and needs no timezone conversion.)


---

### Task 10: Config, deploy, live cut-over and rollback

Depends on Tasks 1–9 (all committed, full suite green). Everything below runs on the pve host (this shell); the container is reached with `sudo pct exec 200 -- ...`, exactly like `dev/ct-test.sh`.

Facts about the target that shape the steps (verified 2026-09-13):

- LXC 200 has **no `sqlite3` CLI and no `git`**, and `/opt/trading-bot` is **not a git checkout** (it was populated by tar). Every ledger query goes through the venv's Python (`sqlite3` 3.46.1); the DB is in WAL mode (`trading_bot.db` + `-wal` + `-shm`), so a backup must use the sqlite backup API, not `cp`.
- The live `.env` currently contains `WEEKLY_BUDGET=500`, `FAST_MAX_HOLD_MIN=240`, `BUDGET_MODE=deployed`, `FAST_MAX_POSITIONS=4`, `MIN_EV_TO_TRADE=0.003`, `STARTUP_NOTICE=1`, `TRADE_CRYPTO=1`, `FAST_MODE=1`, and none of the new keys. It has no inline `# comments` on value lines (systemd's `EnvironmentFile` would keep them as part of the value).
- The live ledger has 20 EXECUTED + 1 REJECTED legacy trades and zero open positions; `data/` is 141 MB DB + 13 MB WAL; 1.7 GB free.
- Live `data/tuned.json`: stock tp 1.0% / sl 0.6% / horizon 24, EV −0.029%; crypto tp 0.6% / sl 0.4% / horizon 24, EV +0.063%. With `MIN_EV_TO_TRADE=0.003`, `FAST_IGNORE_EV=1` and `CRYPTO_SPREAD_BPS=60`: stock is tradeable via the "FAST_IGNORE_EV on" branch of `class_gate`, crypto is blocked by the cost gate (0.60% ≤ 1.20%).
- Service unit: `Restart=always`, nightly `trading-bot-restart.timer` at 03:30 CT (runs `_migrate()` again — `INSERT OR IGNORE` keeps one `account` row).
- Branch is `main`; the last commit before this plan's work is `dd28b31` (spec commit). No tags exist yet.
- **The daily report fires at startup when the restart happens after 16:05 ET.** Task 9's `daily_summary` loop is started unconditionally in `on_ready` (before the fetch/train warm-up) and has no weekday check; `discord.ext.tasks` runs the first iteration immediately. A Sunday deploy at or after 16:05 ET (15:05 CT) therefore posts a `📊 Paper scorecard — 2026-09-13` embed during warm-up, before the restart notice. `IntradayEngine.__init__` loads `metrics` from the saved model blob, so the per-class blocks render; SPY is absent (Task 8's benchmark rows do not exist until the first fetch after deploy, and no SPY close on/after 2026-09-13 exists until Monday), and `n=0`. Step 11 expects this embed; a deploy earlier than 16:05 ET posts it at the first 5-minute tick after 16:05 ET instead.
- **Late-day stock signals are not labelled the same day.** Task 5's `label_pending` returns before the session cut when fewer than `horizon + 1` bars are stored after the signal (`if len(bars) < horizon + 1: continue`), and `labeling.triple_barrier` only labels indices `< n - horizon` in any case. A stock signal from within 24 bars (~2 h) of the 16:00 ET close stays NULL until Tuesday's session bars are stored and is labelled 0 at Tuesday's 16:05 tick. Step 12's expectations are written for that.

**Files:**
- Create: `/home/gdhughey/hugheylab-trading-bot/dev/deploy.sh`
- Create: `/home/gdhughey/hugheylab-trading-bot/dev/ct-sql.sh`
- Modify: `/home/gdhughey/hugheylab-trading-bot/.env.example` (full rewrite; current lines 9–11 "Trading" block, line 52 `FAST_MAX_HOLD_MIN`, lines 73–80 "Budget accounting" block)
- Modify: `/home/gdhughey/hugheylab-trading-bot/configure.sh` (full rewrite, current lines 1–51)
- Modify: `/home/gdhughey/hugheylab-trading-bot/README.md` (line 19; lines 128–135; lines 158–160; lines 187–193; lines 257–271; lines 272–289; line 344; lines 360–362)
- Live (not in repo): `/opt/trading-bot/.env` on LXC 200; backup copies `/opt/trading-bot/.env.bak-paper-account` and `/opt/trading-bot/data/backups/trading_bot.pre-paper-account.db`

- [ ] **Step 1: Tag the rollback anchor**

The container has no git, so "what was running before" must be pinned on the host. `dd28b31` is the spec commit — the last commit before Task 1's first commit and the code that is live today (plus docs).

```bash
cd /home/gdhughey/hugheylab-trading-bot
git tag -a pre-paper-account dd28b31 -m "Last commit before the paper brokerage account (rollback anchor)"
git tag --list
```

Expected output: `pre-paper-account`.

- [ ] **Step 2: Rewrite `.env.example`**

Replace the whole file with the following. Changes vs today: the "Trading" block (`WEEKLY_BUDGET`) becomes "Paper account"; `FAST_MAX_HOLD_MIN=120` is replaced by a comment explaining it is derived; the "Budget accounting" block (`BUDGET_MODE`) becomes "Fills, costs and risk". New keys deliberately have no trailing comments.

```bash
# --- Discord (required) ---
DISCORD_TOKEN=your_bot_token_here
# CHANNEL_ID is OPTIONAL. Leave it blank to have every alert DM'd to USER_ID
# (the bot must still share a server with you for DMs to be deliverable).
# Set it only if you want alerts in a guild channel instead.
CHANNEL_ID=
USER_ID=000000000000000000

# --- Paper account ---
# One simulated cash brokerage account. STARTING_CASH is copied into the
# `account` table the FIRST time the bot starts and never read again: there is
# no setter, because changing it mid-run would corrupt the all-time return.
STARTING_CASH=500
# cash   = a stock sale's proceeds settle at the next trading day's 09:30 ET
#          (T+1) and cannot be spent before then; crypto settles immediately
# margin = no settlement wait (nothing else changes)
ACCOUNT_TYPE=cash
LOOKBACK_PERIOD=2y

# --- Claude analysis (optional; bot degrades gracefully without it) ---
CLAUDE_API_KEY=
CLAUDE_MODEL=claude-opus-5

# --- Paths (relative to /opt/trading-bot) ---
DB_PATH=data/trading_bot.db
MODEL_PATH=data/model.joblib

# --- Market data sources ---
# History chain, tried in order; each only sees symbols the previous one missed.
#   yahoo         batched yfinance (primary, keyless)
#   yahoo_direct  Yahoo chart API, per-symbol gap filler (keyless)
#   alphavantage  independent, but free tier is 25 requests/DAY (key required)
#   tiingo        independent daily bars (key required)
DATA_SOURCES=yahoo,yahoo_direct
# Live quote chain, for P&L and detecting stale history.
#   finnhub     free tier serves quotes (NOT historical candles)
#   yahoo_quote keyless fallback
QUOTE_SOURCES=finnhub,yahoo_quote
FINNHUB_API_KEY=
ALPHAVANTAGE_API_KEY=
TIINGO_API_KEY=

# --- Execution ---
# 1 = execute immediately and report what it did; 0 = wait for ✅/❌ approval
AUTO_TRADE=0
# Minutes between scans. A full 503-symbol refresh takes ~76s, so below 2 is pointless.
CHECK_INTERVAL_MINUTES=60

# --- Intraday fast mode ---
# Trades 5m-bar signals during regular market hours, with real exit rules.
# When enabled the hourly daily loop becomes report-only (one portfolio, one
# set of risk rules). Still paper: no broker is connected.
FAST_MODE=0
FAST_POLL_SECONDS=60        # a full 30-symbol intraday refresh takes ~6s
FAST_MAX_POSITIONS=3
FAST_STOP_LOSS=0.010        # sell at -1.0%
FAST_TAKE_PROFIT=0.015      # sell at +1.5%
FAST_EOD_FLATTEN_MIN=10     # close everything N minutes before the bell
# Max hold is not a setting any more: it is the label horizon x the bar
# interval (INTRADAY_HORIZON_BARS or data/tuned.json; 24 x 5m = 120 min), so
# the executor gives up exactly where the label stops looking.
FAST_COOLDOWN_MIN=15        # don't re-enter a name immediately after exiting
FAST_SUMMARY_MINUTES=30     # idle status cadence (actions always report)
# FAST_SYMBOLS=AAPL,MSFT,... # override the default 30 liquid names

INTRADAY_INTERVAL=5m        # 1m (7d history) | 5m (60d) | 15m (60d)
INTRADAY_HORIZON_BARS=6     # 6 x 5m = predict 30 minutes ahead
INTRADAY_THRESHOLD=0.0015   # count a 0.15% move as a win (must beat the spread)
INTRADAY_PROB_RATIO=1.15    # buy when p > 1.15x the model's base rate

# --- Asset classes ---
TRADE_STOCKS=1
TRADE_CRYPTO=0          # crypto trades 24/7: ~3.7x the bars/day, and tradeable
                        # when the stock market is shut. No EOD flatten applies.

# --- Labeling (see src/labeling.py) ---
# triple = label by which barrier is hit FIRST (matches the executor's TP/SL).
# fixed  = legacy path-blind "is price up in N bars" label. Do not use: it
#          scores an easier question than the trade you actually place.
LABEL_MODE=triple

# --- Fills, costs and risk (paper model of a Robinhood-style account) ---
# Keys below sit on their own lines with no trailing comment: systemd's
# EnvironmentFile keeps "   # text" as part of the value.
# Stocks fill at ref * (1 +/- STOCK_SLIPPAGE_BPS/1e4). No commission.
STOCK_SLIPPAGE_BPS=5
# Crypto fills at ref * (1 +/- CRYPTO_SPREAD_BPS/1e4) PER SIDE (Coinbase
# Advanced base taker tier; round trip = 2x). At the default 60 the crypto
# take-profit is below the round-trip cost, so the cost gate blocks crypto
# entries; lower this for a cheaper venue and crypto re-enables itself.
CRYPTO_SPREAD_BPS=60
# Stock sell-side regulatory fees (2026 schedules). SEC fee is a rate on
# proceeds; FINRA TAF is per share, capped per trade.
SEC_FEE_RATE=0.0000206
FINRA_TAF_PER_SHARE=0.000195
FINRA_TAF_CAP=9.79
# Smallest order the sizer will place, in dollars.
MIN_ORDER_USD=1
# Once equity is down this much from the day's starting equity, no new
# positions open for the rest of the ET day. Exits still run.
DAILY_LOSS_LIMIT_PCT=3
# 1 = open positions even when a class's backtested EV is below
# MIN_EV_TO_TRADE. Paper only: the point of the paper run is to test the
# backtest. The cost gate above is never bypassed.
FAST_IGNORE_EV=0
```

Verify:

```bash
cd /home/gdhughey/hugheylab-trading-bot
grep -nE '^(WEEKLY_BUDGET|BUDGET_MODE|FAST_MAX_HOLD_MIN)=' .env.example; echo "removed-keys grep exit=$?"
grep -cE '^(STARTING_CASH|ACCOUNT_TYPE|STOCK_SLIPPAGE_BPS|CRYPTO_SPREAD_BPS|SEC_FEE_RATE|FINRA_TAF_PER_SHARE|FINRA_TAF_CAP|MIN_ORDER_USD|DAILY_LOSS_LIMIT_PCT|FAST_IGNORE_EV)=' .env.example
```

Expected: first grep prints nothing and `removed-keys grep exit=1`; second prints `10`.

```bash
git add .env.example
git commit -m "env: replace weekly budget with paper account, fills/costs and risk keys

WEEKLY_BUDGET, BUDGET_MODE and FAST_MAX_HOLD_MIN are gone from the code
(Tasks 4 and 6). New keys sit on their own lines with no trailing comment
because systemd's EnvironmentFile keeps a trailing comment in the value.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 3: Rewrite `configure.sh` to prompt for starting cash**

Replace the whole file. Changes: prompt "Starting cash [500]" writes `STARTING_CASH` and `ACCOUNT_TYPE=cash` instead of `WEEKLY_BUDGET`; the value is validated as a number (a typo would be baked into the account forever); the output path is `ENV_FILE`, overridable so the script can be exercised against a temp file.

```bash
#!/bin/bash
# Interactive configurator for the trading bot .env - run from the Proxmox
# console so the token is never echoed or captured in a transcript.
# ENV_FILE is overridable so the script can be exercised against a temp file.
set -euo pipefail
ENV_FILE=${ENV_FILE:-/opt/trading-bot/.env}

echo "=== Trading bot configuration ==="
read -rsp "Discord bot token (hidden, paste + Enter): " TOKEN; echo
echo
echo "Delivery: leave Channel ID BLANK to get alerts as direct messages (recommended)."
read -rp  "Channel ID [blank = DM you]: " CHANNEL
read -rp  "Your user ID  (right-click yourself  -> Copy User ID): " USERID
read -rp  "Starting cash [500]: " CASH; CASH=${CASH:-500}
read -rsp "Anthropic API key for Claude analysis (optional, Enter to skip): " CKEY; echo

[[ -z "$TOKEN"   ]] && { echo "ERROR: token is required"; exit 1; }
if [[ -n "$CHANNEL" ]]; then
  [[ "$CHANNEL" =~ ^[0-9]{17,20}$ ]] || { echo "ERROR: channel ID must be 17-20 digits (or blank for DM)"; exit 1; }
fi
[[ "$USERID"  =~ ^[0-9]{17,20}$ ]] || { echo "ERROR: user ID must be 17-20 digits"; exit 1; }
# The account row is seeded from this value on the first start and never
# reset (there is no setter), so a typo here corrupts the all-time return.
[[ "$CASH" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "ERROR: starting cash must be a number"; exit 1; }

umask 077
cat > "$ENV_FILE" <<ENVEOF
DISCORD_TOKEN=$TOKEN
CHANNEL_ID=$CHANNEL
USER_ID=$USERID

STARTING_CASH=$CASH
ACCOUNT_TYPE=cash
LOOKBACK_PERIOD=2y

CLAUDE_API_KEY=$CKEY
CLAUDE_MODEL=claude-opus-5

DB_PATH=data/trading_bot.db
MODEL_PATH=data/model.joblib
ENVEOF
chmod 600 "$ENV_FILE"

echo
echo "Wrote $ENV_FILE (mode 600). Values recorded:"
sed -E 's/^(DISCORD_TOKEN=).*/\1<hidden>/; s/^(CLAUDE_API_KEY=).+/\1<hidden>/' "$ENV_FILE"
echo
if [[ -z "$CHANNEL" ]]; then
  echo "Delivery mode: DIRECT MESSAGE to user $USERID"
else
  echo "Delivery mode: channel $CHANNEL"
fi
echo
echo "Now start it:  systemctl enable --now trading-bot"
echo "Watch it:      journalctl -u trading-bot -f"
```

Test it on the host against a scratch file (answers piped in the prompt order: token, channel, user id, cash, Claude key):

```bash
cd /home/gdhughey/hugheylab-trading-bot
T=$(mktemp -d)
printf 'tok123\n\n459076947087982592\n\n\n' | ENV_FILE=$T/a.env bash configure.sh | grep -E '^(STARTING_CASH|ACCOUNT_TYPE|WEEKLY_BUDGET)='
printf 'tok123\n\n459076947087982592\n750\n\n' | ENV_FILE=$T/b.env bash configure.sh >/dev/null; grep -E '^STARTING_CASH=' $T/b.env; stat -c %a $T/b.env
printf 'tok123\n\n459076947087982592\nabc\n\n' | ENV_FILE=$T/c.env bash configure.sh | tail -1; echo "exit=${PIPESTATUS[1]}"
rm -rf "$T"
```

Expected:

```
STARTING_CASH=500
ACCOUNT_TYPE=cash
STARTING_CASH=750
600
ERROR: starting cash must be a number
exit=1
```

```bash
git add configure.sh
git commit -m "configure.sh: prompt for starting cash instead of weekly budget

Writes STARTING_CASH and ACCOUNT_TYPE=cash. The value is validated because
the account is seeded from it once and has no setter. ENV_FILE is
overridable so the script can be tested against a temp file.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 4: Update `README.md`**

Eight edits, each a full replacement of the quoted block. Line numbers are as of commit `dd28b31`; find each by its first line if they have shifted.

**4a. Line 19** (the "What it actually does" table's last row). Replace

```
| **Ledger** | SQLite — trades, positions, realized P&L |
```

with

```
| **Ledger** | SQLite — one simulated cash account, trades with modelled fills and fees, positions, T+1 settlement, daily equity history, signal log |
```

**4b. Lines 128–135** (the exit-rules table). Replace

```
| Rule | Default |
|---|---|
| Take profit | +1.5% |
| Stop loss | −1.0% |
| End-of-day flatten | 10 min before the bell |
| Max hold | 120 min |
| Re-entry cooldown | 15 min |
| Max concurrent positions | 3 |
```

with

```
| Rule | Default |
|---|---|
| Take profit | +1.0% stocks / +0.6% crypto (tuned, `data/tuned.json`) |
| Stop loss | −0.6% stocks / −0.4% crypto (tuned) |
| End-of-day flatten | 10 min before the bell |
| Max hold | label horizon × bar interval (24 × 5m = 120 min); not a setting |
| Re-entry cooldown | 15 min |
| Max concurrent positions | 3 (`FAST_MAX_POSITIONS`) |
```

**4c. Lines 158–160.** Replace

```
**A class whose measured EV is negative is not traded at all** — `/fast` shows
it as blocked. Exits still run on a blocked class, so an open position is never
stranded.
```

with

```
**Two gates decide whether a class may open positions** (`class_gate` in
`src/fast_trader.py`, evaluated in this order): a *cost gate* — the take-profit
must exceed the class's round-trip cost, never bypassed — and an *EV gate* —
the backtested EV must clear `MIN_EV_TO_TRADE`, bypassed by `FAST_IGNORE_EV=1`
for the paper run (see "Paper account" below). `/fast` and the startup notice
name a blocked class and why. Exits still run on a blocked class, so an open
position is never stranded.
```

**4d. Lines 187–193** (the whole "### Budget accounting" subsection). Replace

```
### Budget accounting

`BUDGET_MODE=deployed` (default) caps **capital at risk** — the cost basis of
open positions — and selling frees it again. The original `cumulative` mode
capped total buy volume for the week and never refunded it, which halts a
day-trading loop after a handful of round trips: recycling the same $100 ten
times "spends" $1,000 against a $500 cap despite never risking more than $100.
```

with

```
### Paper account

Paper mode models one **$500 retail cash brokerage account** (Robinhood-style:
zero stock commissions, fractional shares, a per-side crypto markup) closely
enough that two months of results are a fair basis for a go/no-go decision on
real money. Every assumption is a setting (table at the end of this section).

**Account.** One row in the `account` table, seeded from `STARTING_CASH` on the
first start and never reset — there is no setter, because changing starting
cash mid-run would corrupt the all-time return. Cash falls on BUY by
fill × qty + fees and rises on SELL by fill × qty − fees. Equity = cash + market
value of open positions. Every report counts only trades with
`created_at >= account.opened_at`; older rows stay in the DB but are ignored.

**Buying power** = cash − unsettled proceeds − pending BUYs. With
`ACCOUNT_TYPE=cash` (default) a stock sale's proceeds are unsettled until the
next trading day's 09:30 ET (T+1, weekends and `US_HOLIDAYS_2026` skipped);
crypto settles immediately. `ACCOUNT_TYPE=margin` removes the wait and changes
nothing else.

**Fills and costs** (`src/costs.py`, applied in exactly one place,
`BudgetTracker.log_trade`):

| | Fill | Fees |
|---|---|---|
| Stock | ref × (1 ± `STOCK_SLIPPAGE_BPS`/1e4), 5 bps default | BUY none; SELL `SEC_FEE_RATE` × proceeds + min(`FINRA_TAF_PER_SHARE` × qty, `FINRA_TAF_CAP`) |
| Crypto | ref × (1 ± `CRYPTO_SPREAD_BPS`/1e4) per side, 60 bps default | none |

Round-trip cost: stocks ≈ 0.1%, crypto 1.2%. The crypto take-profit (0.6%) is
below its round-trip cost, so the cost gate blocks crypto entries at the
defaults; lower `CRYPTO_SPREAD_BPS` for a cheaper venue and it re-enables.

**Sizing.** Every entry is `min(buying power, equity / FAST_MAX_POSITIONS)`
dollars as a fractional quantity, skipped below `MIN_ORDER_USD`. Wins compound;
losses shrink the next order.

**Barriers are measured reference-to-reference** — the quote at entry against
the quote now, the same move the labels use — so costs show up in cash and P&L,
never in the stop trigger. Max hold is the label horizon × bar interval
(24 × 5m = 120 min); `FAST_MAX_HOLD_MIN` no longer exists.

**Daily loss limit.** Once equity is down `DAILY_LOSS_LIMIT_PCT` (3%) from the
day's starting equity, no new positions open for the rest of the ET day. Exits
still run. Announced once on Discord.

**Daily report.** At 16:05 ET every day (weekends included) the bot labels the
day's logged signals, writes the day's `equity_history` row, and posts one
scorecard: all-time P&L with n and a 95% CI, a GO / NO-GO / EXTEND verdict per
class, gross P&L and fees, balance, buying power, unsettled cash, today's
trades, open positions, win rate with a Wilson CI, profit factor, max drawdown,
per-class realised vs backtested precision, and SPY buy-and-hold on the same
starting cash as context. `/pnl` and `/summary` show the same scorecard on
demand.

**Decision rule** (`src/scorecard.py: verdict`, pre-registered and frozen at
`account.opened_at`; review 2026-11-13). Per class, with cost-adjusted
breakeven `be_c = (sl + c) / (tp + sl)` and `[lo, hi]` the Wilson 95% CI of the
live signal TP-first rate (n ≈ 40 signals/day — the executed-trade sample is
too small to decide anything):

- **NO-GO** if `hi < be_c`, or the cost gate blocks the class.
- **GO** if `lo > be_c`, the day-clustered lower bound also clears `be_c`, at
  least 60 trades executed, and the realised mean net return is within one SE
  of the backtested EV net of costs.
- **EXTEND** otherwise — treated as NO-GO for real money until it turns GO.

`FAST_IGNORE_EV=1` (set for the paper run) lets a class trade on paper even
when its backtested EV is below `MIN_EV_TO_TRADE`: the point of the run is to
test whether live behaviour matches the backtest, not to overturn it. The cost
gate is never bypassed.

| Variable | Default | Notes |
|---|---|---|
| `STARTING_CASH` | `500` | seeded once on first start; no setter |
| `ACCOUNT_TYPE` | `cash` | `cash` = T+1 stock settlement; `margin` = none |
| `STOCK_SLIPPAGE_BPS` | `5` | per side |
| `CRYPTO_SPREAD_BPS` | `60` | per side; round trip 120 bps |
| `SEC_FEE_RATE` | `0.0000206` | stock sells, on proceeds |
| `FINRA_TAF_PER_SHARE` | `0.000195` | stock sells, capped by `FINRA_TAF_CAP` (`9.79`) |
| `MIN_ORDER_USD` | `1` | smallest order |
| `DAILY_LOSS_LIMIT_PCT` | `3` | account-wide, per ET day |
| `FAST_IGNORE_EV` | `0` | `1` on the paper run |
```

**4e. Lines 257–271** (the slash-command table body). Replace

```
| Command | What it does |
|---|---|
| `/status` | Budget and open positions |
| `/scan [top]` | Rank the whole universe right now |
| `/pnl` | Mark the paper portfolio to market |
| `/budget [amount]` | Check or set the weekly budget |
| `/stats` | Executed / rejected counts, approval rate, realized P&L |
| `/daily_brief` | Claude market commentary (optional) |
| `/risk_check` | Claude risk read on open positions (optional) |
| `/retrain` | Refresh all sources and retrain (~4 min) |
| `/fast` | Intraday model stats, exit rules, and live scores |
| `/summary` | Today's results and tomorrow's game plan |
| `/sources` | Where market data is coming from, and last refresh counts |
| `/pause` · `/resume` | Stop or restart the loop |
```

with

```
| Command | What it does |
|---|---|
| `/status` | Balance, buying power and open positions |
| `/scan [top]` | Rank the whole universe right now |
| `/pnl` | The scorecard now: all-time P&L, verdicts, positions marked to market |
| `/account` | Balance, cash, unsettled, buying power, starting cash, opened |
| `/daily_brief` | Claude market commentary (optional) |
| `/risk_check` | Claude risk read on open positions (optional) |
| `/retrain` | Refresh all sources and retrain (~4 min) |
| `/fast` | Intraday model stats, exit rules, gates, and live scores |
| `/summary` | The same scorecard the 16:05 ET report posts |
| `/sources` | Where market data is coming from, and last refresh counts |
| `/pause` · `/resume` | Stop or restart the loop |
```

**4f. Lines 272–289** ("### Hourly heartbeat" through the end of "### Daily wrap"). Replace

```
### Hourly heartbeat

Every hour it posts a plain-English report, so silence is never ambiguous:

- **💰 Your money** — budget, cash left, what your holdings are worth, up/down in
  dollars and percent
- **📊 What you own** — per position: what you paid, what it's worth now, gain/loss
- **👉 What to do** — in plain words: react to an alert, or nothing, and why
- **🔍 What I checked** — how many symbols, from where, how many were buys vs sells

Set `HEARTBEAT=0` to turn it off.

### Daily wrap

After the close it posts one summary: what was booked today, every position
closed, anything held overnight (crypto only — stocks are flattened), and a game
plan for tomorrow with each model's EV, available budget, and when trading
resumes. `/summary` runs it on demand.
```

with

```
### Hourly heartbeat

Every hour it posts a plain-English report, so silence is never ambiguous:

- **💰 Your money** — balance vs starting cash, buying power, unsettled cash,
  what your holdings are worth, up/down in dollars and percent
- **📊 What you own** — per position: what you paid, what it's worth now, gain/loss
- **👉 What to do** — in plain words: react to an alert, or nothing, and why
- **🔍 What I checked** — how many symbols, from where, how many were buys vs sells

Set `HEARTBEAT=0` to turn it off.

### Daily report

At 16:05 ET every day (weekends included, so a quiet Saturday still gets an
equity row) it labels the day's logged signals, records the day's equity, and
posts the scorecard described under "Paper account". A failed post is retried
every 5 minutes; `day_state.report_posted_at` guarantees at most one post per
date, even across the nightly restart. `/summary` shows it on demand.
```

**4g. Line 344** (configuration table). Replace

```
| `WEEKLY_BUDGET` | `5000` | small budgets skip expensive stocks; a 1-share floor applies |
```

with

```
| `STARTING_CASH` | `500` | seeded into the paper account on first start; no setter |
| `ACCOUNT_TYPE` | `cash` | `cash` = T+1 stock settlement; `margin` = none |
```

**4h. Lines 360–362** (last bullet of "Notes for anyone extending this"). Replace

```
- Pending trades hold budget. They are rolled back if the alert fails to send,
  and cleared at startup, since their approval watcher lives in memory.
```

with

```
- Pending trades hold buying power. They are rolled back if the alert fails to
  send, and cleared at startup, since their approval watcher lives in memory.
```

Verify and commit:

```bash
cd /home/gdhughey/hugheylab-trading-bot
grep -nE 'WEEKLY_BUDGET|BUDGET_MODE|FAST_MAX_HOLD_MIN=|/budget|/stats' README.md; echo "stale grep exit=$?"
grep -c '^### Paper account' README.md
git add README.md
git commit -m "README: paper account section, gates, daily report, /account

Replaces the budget-accounting section, the weekly budget rows and the
/budget and /stats commands with the paper brokerage model from the spec.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

Expected: no output, `stale grep exit=1`; count prints `1`. (All four pre-edit hits — lines 189, 262, 263, 344 — sit inside the replaced blocks, and the one surviving mention of `FAST_MAX_HOLD_MIN` in 4d has no `=` after it, so the `FAST_MAX_HOLD_MIN=` pattern does not match it.)

- [ ] **Step 5: Write `dev/deploy.sh`**

Create `/home/gdhughey/hugheylab-trading-bot/dev/deploy.sh`:

```bash
#!/usr/bin/env bash
# Deploy the committed working tree to the LIVE bot in LXC 200 and restart it.
# Usage: dev/deploy.sh [--dry-run]
#   --dry-run  print the list of files that would be shipped; touches nothing.
#
# Sibling of dev/ct-test.sh, but aimed at /opt/trading-bot (the production
# copy, not /opt/trading-bot-dev). The container has no git, so this is the
# only way code reaches it. Untarring over the live tree never deletes files,
# and the container's own data/, logs/, venv/ and .env are excluded here, so a
# deploy can never clobber the ledger, the models or the secrets.
set -euo pipefail
cd "$(dirname "$0")/.."

CT=200
APP=/opt/trading-bot
EXCLUDES=(--exclude=.git --exclude=venv --exclude=data --exclude=logs --exclude=.env
          --exclude=__pycache__ --exclude=tests --exclude=dev --exclude=docs)

if [[ "${1:-}" == "--dry-run" ]]; then
  tar cf - "${EXCLUDES[@]}" . | tar tf - | sort
  exit 0
fi

# Only ship a commit: a rollback is "git checkout <tag> and deploy again",
# which is meaningless if what was running never existed in git.
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: uncommitted changes in tracked files - commit first" >&2
  exit 1
fi
rev=$(git rev-parse --short HEAD)

echo "==> shipping $rev to CT $CT:$APP"
tar czf - "${EXCLUDES[@]}" . | sudo pct exec "$CT" -- tar xzf - -C "$APP"

echo "==> restarting trading-bot"
sudo pct exec "$CT" -- systemctl restart trading-bot
sudo pct exec "$CT" -- systemctl is-active trading-bot
echo "==> follow it with: sudo pct exec $CT -- journalctl -u trading-bot -f"
```

Test the exclusions and the guard without touching the container:

```bash
cd /home/gdhughey/hugheylab-trading-bot
chmod +x dev/deploy.sh && bash -n dev/deploy.sh && echo SYNTAX-OK
dev/deploy.sh --dry-run | grep -E '^\./(tests|dev|docs|data|logs|venv|\.git|\.env)(/|$)'; echo "excluded grep exit=$?"
dev/deploy.sh --dry-run | grep -E '^\./(main\.py|configure\.sh|\.env\.example|src/costs\.py|src/scorecard\.py|src/signal_log\.py)$'
echo x >> README.md; dev/deploy.sh; echo "guard exit=$?"; git checkout README.md
```

Expected:

```
SYNTAX-OK
excluded grep exit=1
./.env.example
./configure.sh
./main.py
./src/costs.py
./src/scorecard.py
./src/signal_log.py
ERROR: uncommitted changes in tracked files - commit first
guard exit=1
Updated 1 path from the index
```

```bash
git add dev/deploy.sh
git commit -m "dev/deploy.sh: tar the committed tree into LXC 200 and restart

Excludes .git, venv, data, logs, .env, __pycache__, tests, dev and docs so
a deploy can never overwrite the ledger, models or secrets. Refuses a dirty
tree so every deployed state is a commit that can be re-deployed by tag.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 6: Write `dev/ct-sql.sh` (read-only ledger queries; the container has no sqlite3 CLI)**

Create `/home/gdhughey/hugheylab-trading-bot/dev/ct-sql.sh`:

```bash
#!/usr/bin/env bash
# Run one read-only SQL statement against the LIVE ledger in LXC 200.
# The container has no sqlite3 CLI, so this goes through the venv's python.
# Usage: dev/ct-sql.sh "select * from account"
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: dev/ct-sql.sh \"<sql>\"" >&2; exit 2; }
sudo pct exec 200 -- /opt/trading-bot/venv/bin/python - "$1" <<'PY'
import sqlite3, sys
# mode=ro: this tool must never take a write lock under the running bot.
c = sqlite3.connect('file:/opt/trading-bot/data/trading_bot.db?mode=ro', uri=True)
cur = c.execute(sys.argv[1])
if cur.description:
    print(' | '.join(d[0] for d in cur.description))
for row in cur:
    print(' | '.join(str(v) for v in row))
PY
```

Test against the live ledger (read-only, safe while the bot runs):

```bash
cd /home/gdhughey/hugheylab-trading-bot
chmod +x dev/ct-sql.sh
dev/ct-sql.sh "select status, count(*) as n from trades group by status"
```

Expected (pre-deploy numbers):

```
status | n
EXECUTED | 20
REJECTED | 1
```

```bash
git add dev/ct-sql.sh
git commit -m "dev/ct-sql.sh: read-only SQL against the live ledger via the venv

LXC 200 has no sqlite3 CLI; the verify and rollback steps need to inspect
the account, day_state, signals and equity_history tables.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01B6YgCaXDNQwA42mCUVXQkp"
```

- [ ] **Step 7: Run the whole suite one last time against the committed tree**

```bash
cd /home/gdhughey/hugheylab-trading-bot
git status --porcelain --untracked-files=no; echo "dirty exit=$?"
dev/ct-test.sh tests/ -v 2>&1 | tail -15
```

Expected: the status line prints nothing; pytest's last line is `======= N passed in X.XXs =======` with no `failed` or `error`, covering `tests/test_costs.py`, `test_calendar.py`, `test_migration.py`, `test_budget_tracker.py`, `test_signal_log.py`, `test_fast_trader.py`, `test_scorecard.py`, `test_benchmark.py`, `test_discord_embeds.py`. Do not continue with a red suite.

- [ ] **Step 8: Stop the bot and back up the live ledger**

Stop first so the backup is of a quiescent DB and the `.env` edit in Step 9 cannot race the nightly restart. It is Sunday; the market is closed, so downtime costs nothing. `sqlite3.Connection.backup` copies a consistent snapshot including pages still in the `-wal` file, which a plain `cp` of the `.db` would miss.

```bash
sudo pct exec 200 -- systemctl stop trading-bot
sudo pct exec 200 -- systemctl is-active trading-bot; echo "(inactive is expected; exit 3 is fine)"
sudo pct exec 200 -- bash -s <<'EOF'
set -euo pipefail
cd /opt/trading-bot/data
mkdir -p backups
/opt/trading-bot/venv/bin/python - <<'PY'
import sqlite3
src = sqlite3.connect('trading_bot.db')
dst = sqlite3.connect('backups/trading_bot.pre-paper-account.db')
src.backup(dst)
print('integrity', dst.execute('PRAGMA integrity_check').fetchone()[0])
print('trades', dst.execute('SELECT status, COUNT(*) FROM trades GROUP BY status').fetchall())
print('tables', [r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1")])
dst.close(); src.close()
PY
ls -la backups/
EOF
```

Expected:

```
inactive
(inactive is expected; exit 3 is fine)
integrity ok
trades [('EXECUTED', 20), ('REJECTED', 1)]
tables ['positions', 'prices', 'prices_intraday', 'sqlite_sequence', 'trades']
-rw-r--r-- 1 root root 1xxxxxxxx ... trading_bot.pre-paper-account.db
```

(the file is ~140–155 MB; `data/` is excluded from every deploy, so the backup survives redeploys.)

- [ ] **Step 9: Migrate the live `.env` on LXC 200**

Deletes `WEEKLY_BUDGET`, `BUDGET_MODE`, `FAST_MAX_HOLD_MIN` with `sed` and appends the seven keys the contract names. Idempotent: a second run is a no-op. The original is kept as `.env.bak-paper-account` for Step 13. Values match the contract's live settings (`FAST_IGNORE_EV=1`; the other five are the defaults, written explicitly so the running configuration is visible in the file).

```bash
sudo pct exec 200 -- bash -s <<'EOF'
set -euo pipefail
ENV=/opt/trading-bot/.env
if grep -q '^STARTING_CASH=' "$ENV"; then echo "already migrated - nothing to do"; exit 0; fi
cp -p "$ENV" "$ENV.bak-paper-account"
# Deleted keys: the code no longer reads them; leaving them would suggest the
# weekly budget still exists.
sed -i -E '/^(WEEKLY_BUDGET|BUDGET_MODE|FAST_MAX_HOLD_MIN)=/d' "$ENV"
# No trailing comments on value lines: systemd's EnvironmentFile keeps them.
cat >> "$ENV" <<'ENVEOF'

# --- Paper brokerage account (added 2026-09-13; every key documented in .env.example) ---
STARTING_CASH=500
ACCOUNT_TYPE=cash
FAST_IGNORE_EV=1
STOCK_SLIPPAGE_BPS=5
CRYPTO_SPREAD_BPS=60
DAILY_LOSS_LIMIT_PCT=3
MIN_ORDER_USD=1
ENVEOF
chmod 600 "$ENV"
echo "--- removed keys still present (expect nothing):"
grep -nE '^(WEEKLY_BUDGET|BUDGET_MODE|FAST_MAX_HOLD_MIN)=' "$ENV" || true
echo "--- new keys present (expect 7):"
grep -cE '^(STARTING_CASH|ACCOUNT_TYPE|FAST_IGNORE_EV|STOCK_SLIPPAGE_BPS|CRYPTO_SPREAD_BPS|DAILY_LOSS_LIMIT_PCT|MIN_ORDER_USD)=' "$ENV"
ls -l "$ENV" "$ENV.bak-paper-account"
EOF
```

Expected:

```
--- removed keys still present (expect nothing):
--- new keys present (expect 7):
7
-rw------- 1 ... /opt/trading-bot/.env
-rw------- 1 ... /opt/trading-bot/.env.bak-paper-account
```

Sanity-read the result with secrets hidden:

```bash
sudo pct exec 200 -- sed -E 's/^([A-Z_]*(TOKEN|KEY))=.*/\1=<hidden>/' /opt/trading-bot/.env | grep -nE '^(FAST_MAX_POSITIONS|MIN_EV_TO_TRADE|TRADE_CRYPTO|FAST_MODE|STARTUP_NOTICE|STARTING_CASH|ACCOUNT_TYPE|FAST_IGNORE_EV)='
```

Expected: eight lines, including `FAST_MAX_POSITIONS=4`, `MIN_EV_TO_TRADE=0.003`, `TRADE_CRYPTO=1`, `FAST_MODE=1`, `STARTUP_NOTICE=1`, `STARTING_CASH=500`, `ACCOUNT_TYPE=cash`, `FAST_IGNORE_EV=1`.

- [ ] **Step 10: Deploy**

Note the wall-clock time of the restart: if it is at or after 15:05 CT (16:05 ET), Step 11 expects the Sunday scorecard to post during warm-up; if earlier, it posts at the first 5-minute tick after 16:05 ET.

```bash
cd /home/gdhughey/hugheylab-trading-bot
git log --oneline -1
date
dev/deploy.sh
```

Expected:

```
==> shipping <short-sha> to CT 200:/opt/trading-bot
==> restarting trading-bot
active
==> follow it with: sudo pct exec 200 -- journalctl -u trading-bot -f
```

Confirm the new modules landed and nothing protected was touched:

```bash
sudo pct exec 200 -- bash -c 'ls /opt/trading-bot/src/costs.py /opt/trading-bot/src/scorecard.py /opt/trading-bot/src/signal_log.py; ls /opt/trading-bot/tests 2>&1 | head -1; ls -la /opt/trading-bot/.env /opt/trading-bot/data/backups/'
```

Expected: the three `src/` files listed; `ls: cannot access '/opt/trading-bot/tests': No such file or directory`; `.env` still mode 600 with the Step 9 timestamp; the backup still present.

- [ ] **Step 11: Verify the restart — journal, Sunday scorecard, startup notice, account row**

Startup fetch + training takes ~4–5 minutes. Follow the journal until the ready line appears (Ctrl-C to stop following):

```bash
sudo pct exec 200 -- journalctl -u trading-bot -f -n 0
```

Then pull the relevant lines in one shot:

```bash
sudo pct exec 200 -- journalctl -u trading-bot --since "-15min" --no-pager | grep -E "Initializing database|Bot logged in|Cleared .* stale|labelled|Scorecard 2026-09-13|Posted daily report|model ready|Fast loop started|Ready - slash|startup notice|Traceback|ERROR|Error"
```

Expected lines, in this order, and NO `Traceback`/`ERROR` lines:

- `📊 Initializing database...` (this is where `_migrate()` seeds the account row)
- `✅ Bot logged in as <bot name>`
- **Only when the restart was at or after 16:05 ET** — three lines from the daily report, which fires immediately because the `daily_summary` loop starts in `on_ready` before warm-up (they interleave with the fetch's own log lines): `2026-09-13: labelled 0 signal(s); equity $500.00 recorded`, `Scorecard 2026-09-13: all-time $+0.00 on n=0 | stock EXTEND | crypto NO-GO`, `Posted daily report for 2026-09-13`. When the restart was earlier, these same three lines appear at the first 5-minute tick after 16:05 ET instead.
- one `model ready` line per class; the crypto line's verdict text contains `blocked: take-profit 0.60% is below the 1.20% round-trip cost`, the stock line's contains `FAST_IGNORE_EV on` (stock EV −0.029% is under the 0.30% floor but the flag lets it trade on paper)
- `⚡ Fast loop started every 60s - market is weekend` (`market_state()` returns the description `weekend` on a Saturday or Sunday)
- `🟢 Ready - slash commands are live`

In Discord (DM), two embeds:

1. `📊 Paper scorecard — 2026-09-13` (grey; arrives first, during warm-up, when the restart was at or after 16:05 ET — otherwise at 16:05 ET). Field `All-time: +$0.00 (+0.00%)` with value `since 2026-09-13 · n=0 closed trade(s), 95% CI ±$0.00`; `Verdict` showing `stock: EXTEND` (its text includes `n=0`) and `crypto: NO-GO — cost gate blocks crypto`; `Account` with Balance `$500.00`, cash `$500.00`, buying power `$500.00`, `Unsettled $0.00` and no `settles` suffix; `Today (2026-09-13)` with `0 trade(s)` and `daily-loss block not tripped`; no `Closed today` and no `Open positions` field; `Scorecard since open` with `0 trade(s) closed`, `1 day(s) running`; one block per class whose `Gate:` line matches the model-ready journal text; **no `SPY buy-and-hold` field** (no SPY close exists on or after 2026-09-13 until Monday's fetch); the footer caveat.
2. `🔄 Trading bot restarted` whose description says **Trading is LIVE** (green); the Stock field contains `FAST_IGNORE_EV on`; the Crypto field contains `blocked: take-profit 0.60% is below the 1.20% round-trip cost`; a Balance/Buying power pair reading `$500.00`. If it says "Nothing will be traded", `FAST_IGNORE_EV=1` did not reach the process — recheck Step 9 and `systemctl show trading-bot -p EnvironmentFiles`.

Confirm the migration on the live ledger:

```bash
cd /home/gdhughey/hugheylab-trading-bot
dev/ct-sql.sh "select * from account"
dev/ct-sql.sh "select name from sqlite_master where type='table' and name in ('account','day_state','equity_history','signals') order by 1"
dev/ct-sql.sh "select count(*) as new_cols from pragma_table_info('trades') where name in ('ref_price','fees','gross_pnl','available_at','trade_date','entry_probability','exit_reason')"
dev/ct-sql.sh "select count(*) as n from pragma_table_info('positions') where name='entry_ref'"
dev/ct-sql.sh "select count(*) as legacy from trades where created_at < (select opened_at from account)"
```

Expected:

```
id | opened_at | starting_cash | cash | account_type
1 | 2026-09-13T2?:??:??+00:00 | 500.0 | 500.0 | cash
name
account
day_state
equity_history
signals
new_cols
7
n
1
legacy
21
```

Exactly one `account` row; `opened_at` is a UTC ISO timestamp from the restart in Step 10; all 21 legacy trades predate it, so they are excluded from every report.

Once the Sunday scorecard has posted (immediately, or after 16:05 ET), confirm its durable side effects:

```bash
dev/ct-sql.sh "select date, start_equity, loss_tripped_at, loss_announced_at, report_posted_at from day_state"
dev/ct-sql.sh "select * from equity_history"
```

Expected:

```
date | start_equity | loss_tripped_at | loss_announced_at | report_posted_at
2026-09-13 | 500.0 | None | None | 2026-09-13T2?:??:??+00:00
date | cash | positions_value | equity | fees_to_date | realized_to_date | recorded_at
2026-09-13 | 500.0 | 0.0 | 500.0 | 0.0 | 0.0 | 2026-09-13T2?:??:??+00:00
```

Run `/account` in Discord: title `🏦 Paper account`, description `cash account, opened 2026-09-13` (the opened date lives in the description, not in a field), fields Balance `$500.00`, Cash `$500.00`, Unsettled `$0.00`, Buying power `$500.00`, Starting cash `$500.00`, All-time `+$0.00 (+0.00%)`; no "Stale quotes" field.

- [ ] **Step 12: Live check on Monday 2026-09-14 (the spec's closing check)**

The nightly 03:30 CT restart runs first; at 08:30 CT confirm it kept one account row with the ORIGINAL `opened_at` (`INSERT OR IGNORE`), and that the 03:30 restart did not re-post Sunday's report (the ET date was already 2026-09-14 and the time was before 16:05):

```bash
cd /home/gdhughey/hugheylab-trading-bot
dev/ct-sql.sh "select count(*) as rows, min(opened_at) as opened_at from account"
dev/ct-sql.sh "select count(*) as posted from day_state where report_posted_at is not null"
```

Expected: `1 | <Sunday's timestamp from Step 11>` and `posted` = `1`.

**09:30–10:00 ET (08:30–09:00 CT) — first cycles.** Follow the fast loop:

```bash
sudo pct exec 200 -- journalctl -u trading-bot -f -n 0 | grep -E '\[fast\]|loss limit|Traceback|ERROR'
```

Within the first few cycles expect `[fast] ENTER <qty> <SYM> @ $<fill> ...` lines for stocks only (crypto is cost-gated; no crypto ENTER may appear), each quantity fractional (e.g. `0.412345`), and each entry ≈ $125 (equity $500 / `FAST_MAX_POSITIONS`=4). Then check the ledger:

```bash
dev/ct-sql.sh "select id, symbol, side, status, shares, ref_price, price, fees, amount, trade_date, entry_probability, exit_reason, available_at from trades where created_at >= (select opened_at from account) order by id"
dev/ct-sql.sh "select symbol, shares, avg_price, entry_ref from positions where shares > 0"
dev/ct-sql.sh "select * from day_state order by date"
dev/ct-sql.sh "select count(*) as n, count(distinct bar_ts) as bars, sum(above_bar) as above from signals"
dev/ct-sql.sh "select cash from account"
```

Expected:

- BUY rows: `status=EXECUTED`, `price = ref_price × 1.0005` (5 bps), `fees = 0`, `amount = price × shares`, `trade_date = 2026-09-14`, `entry_probability` set, `exit_reason` and `available_at` NULL.
- `positions`: one row per open BUY with `entry_ref = ref_price` of that BUY and `avg_price = price`.
- `day_state`: two rows — Sunday's `2026-09-13 | 500.0 | None | None | <Sunday's report timestamp>` and today's `2026-09-14 | 500.0 (or current equity at the first cycle) | None | None | None`.
- `signals`: `n` grows by roughly one row per scored symbol per 5-minute bar (`bars` increases by 1 every 5 min, not every 60 s poll); `above` ≥ 0.
- `account.cash` = 500 − Σ BUY `amount`.

**After the first exit** (TP/SL/timeout — at most 120 min after entry, or EOD flatten at 15:50 ET):

```bash
dev/ct-sql.sh "select id, symbol, exit_reason, ref_price, price, fees, amount, realized_pnl, gross_pnl, available_at from trades where side='SELL' and created_at >= (select opened_at from account) order by id"
```

Expected: `exit_reason` in `tp|sl|timeout|eod`; `price = ref_price × 0.9995`; `fees > 0` (≈ `0.0000206 × amount + 0.000195 × shares`); `gross_pnl = realized_pnl + fees`; `available_at = 2026-09-15T13:30:00+00:00` (Tuesday 09:30 ET as UTC). Then in Discord `/account` shows `Unsettled` = that SELL's `amount` and `Buying power` = cash − unsettled.

**16:05 ET (15:05 CT) — the daily report.** In Discord expect one scorecard embed: an **All-time: ±$X.XX (±Y.YY%)** headline with `n=<closed trades>` and a `95% CI ±$…`; a **Verdict** field with `stock: EXTEND` or `NO-GO` (GO is impossible on day 1: `n_exec < 60`) and `crypto: NO-GO` (cost gate); Balance / Cash / Buying power / Unsettled, the Unsettled line carrying a `settles … 09:30 ET` suffix that names **Tuesday 2026-09-15** — it is rendered from `account.unsettled_until = '2026-09-15'` (Task 7's ET date string), so it reads `settles 2026-09-15 09:30 ET` (or `settles Tue 15 Sep 09:30 ET` if Task 9 renders the date through a day label); it must NOT name Monday the 14th, which is the symptom of the timezone bug the Task 9 fix removes; Closed today; Open positions (empty — flattened); the SPY context line (Monday's SPY close now exists); the footer caveat. Then:

```bash
dev/ct-sql.sh "select * from equity_history order by date"
dev/ct-sql.sh "select date, start_equity, loss_tripped_at, report_posted_at from day_state where date='2026-09-14'"
dev/ct-sql.sh "select asset_class, count(*) as n, sum(label is not null) as labelled, sum(label=1) as tp_first, min(case when label is null then bar_ts end) as first_unlabelled from signals where trade_date='2026-09-14' group by asset_class"
sudo pct exec 200 -- journalctl -u trading-bot --since "15:00" --until "15:15" --no-pager | grep -iE "report|scorecard|label|Traceback|ERROR" | head -20
```

Expected: two `equity_history` rows — Sunday's `2026-09-13` row from Step 11 and one for `2026-09-14` with `equity = cash + positions_value`; the `2026-09-14` `report_posted_at` is a UTC timestamp at or just after `20:05:00+00:00`; for `stock`, `labelled > 0` and `labelled < n`: signals from before ~14:00 ET have a full 24-bar window inside the session and are labelled; signals from ~14:00 ET onward have fewer than 25 stored bars after them, so `label_pending` leaves them NULL (`first_unlabelled` is a `bar_ts` around `2026-09-14T18:00:00+00:00` = 14:00 ET) and they resolve on Tuesday: once Tuesday's bars are stored the session cut labels them 0 at Tuesday's 16:05 tick; for `crypto`, rows whose 120-minute window runs past the last stored bar stay NULL until the next fetch; no `Traceback`/`ERROR` in the journal around 15:05 CT. If `report_posted_at` is still NULL at 15:15 CT, read the ERROR line the loop logged (it retries every 5 minutes) and fix before Tuesday.

Write down what was observed (first ENTER time and symbol, first exit and its `exit_reason`, that no crypto ENTER appeared, the `signals` count and `labelled` count at 16:05, the `report_posted_at` timestamp and the verdict line): these facts go into the tag message of Step 14.

- [ ] **Step 13: Rollback (only if Step 11 or Step 12 fails and cannot be fixed in place)**

Order matters: stop → restore config → (optionally) restore DB → ship the old tree → restart. Restoring the DB is only necessary if the new ledger itself is broken; the pre-change code ignores the extra tables and columns, so keeping the DB loses nothing and preserves the trades made since deploy.

```bash
sudo pct exec 200 -- systemctl stop trading-bot
sudo pct exec 200 -- bash -s <<'EOF'
set -euo pipefail
cd /opt/trading-bot
cp -p .env.bak-paper-account .env
chmod 600 .env
grep -cE '^(WEEKLY_BUDGET|BUDGET_MODE|FAST_MAX_HOLD_MIN)=' .env   # expect 3
EOF
```

Only if the ledger must be restored (destroys everything written since Step 8):

```bash
sudo pct exec 200 -- bash -s <<'EOF'
set -euo pipefail
cd /opt/trading-bot/data
rm -f trading_bot.db trading_bot.db-wal trading_bot.db-shm
cp -p backups/trading_bot.pre-paper-account.db trading_bot.db
/opt/trading-bot/venv/bin/python -c "import sqlite3; c=sqlite3.connect('trading_bot.db'); print(c.execute('PRAGMA integrity_check').fetchone()[0], c.execute('select count(*) from trades').fetchone()[0])"
EOF
```

Expected: `ok 21`.

Ship the pre-change tree. `dev/deploy.sh` does not exist at the anchor commit, so the tar line is inlined here (same exclusions):

```bash
cd /home/gdhughey/hugheylab-trading-bot
git checkout pre-paper-account
tar czf - --exclude=.git --exclude=venv --exclude=data --exclude=logs --exclude=.env \
    --exclude=__pycache__ --exclude=tests --exclude=dev --exclude=docs . \
  | sudo pct exec 200 -- tar xzf - -C /opt/trading-bot
sudo pct exec 200 -- systemctl restart trading-bot
git checkout main
sudo pct exec 200 -- journalctl -u trading-bot -f -n 0
```

Expected in the journal within ~5 minutes: `🟢 Ready - slash commands are live` and the old-style startup notice (`Nothing will be traded right now` — both classes are under the EV floor and `FAST_IGNORE_EV` does not exist in that code). The new `src/costs.py`, `src/scorecard.py`, `src/signal_log.py` files remain on disk but nothing imports them. Fix forward on `main`, then repeat Steps 7–12.

- [ ] **Step 14: Tag and push the release**

After Step 12 has passed (report posted with a verdict line). The tag message records what was actually observed: replace the two angle-bracket lines below with the facts written down at the end of Step 12 before running the command — do not tag with the brackets still in place.

```bash
cd /home/gdhughey/hugheylab-trading-bot
git status --porcelain --untracked-files=no; echo "dirty exit=$?"
git tag -a paper-account-v1 -m "Paper brokerage account v1: costs, T+1 settlement, ref-to-ref barriers, daily loss limit, signal log, 16:05 ET scorecard with pre-registered GO/NO-GO rule.

Deployed to LXC 200 on 2026-09-13; account opened at the first start after deploy.
Live check 2026-09-14: <first ENTER time/symbol; first exit and its exit_reason; whether any crypto ENTER appeared (expected: none, cost-gated)>
<signals rows and labelled count at 16:05 ET; report_posted_at timestamp; the verdict line as posted>"
git push origin main paper-account-v1 pre-paper-account
git tag --list
```

Expected: the status line prints nothing; push reports the branch and both tags; the list shows `paper-account-v1` and `pre-paper-account`.

**Contract additions**

- No new code names. Two behavioural notes this task relies on, for the record:
  - `signal_log.label_pending` (Task 5) leaves a stock signal NULL while fewer than `horizon + 1` bars are stored after it, even when the ET session has already ended; the session cut only applies once the next session's bars exist. Same-day labelling therefore covers stock signals up to ~14:00 ET; later ones are labelled 0 at the following trading day's 16:05 tick.
  - `TradingBot.daily_summary` (Task 9) fires on its first iteration at startup; a restart at or after 16:05 ET posts that ET date's scorecard during warm-up, before the restart notice, using the metrics loaded from the saved model blob and with `spy = None` until a SPY close on or after `opened_at[:10]` is stored.


---

## Contract additions declared by task writers

- Task 2: Behaviour clarification (no new name): market_state(now) between 13:00 and 16:00 ET on an EARLY_CLOSE_2026 date returns ('closed', 'overnight') — pre-existing fall-through, pinned by tests/test_calendar.py.
- Task 3: Database.__init__(self, db_path: str = None, now: datetime = None) — now is tz-aware UTC; used only for account.opened_at on the first seed. main.py and train.py keep calling Database() with no args.
- Task 3: Database.init_schema(self, now: datetime = None) and Database._migrate(self, now: datetime = None) — same now threaded through.
- Task 3: src.database.COLUMN_MIGRATIONS: list[tuple[str, str, str]] — module-level (table, column, declaration) list; the existing realized_pnl and prices.source migrations live in it.
- Task 3: tests/conftest.py module constants: OPENED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc), OLD_SCHEMA: str, LEGACY_TRADES: list[tuple] (21 rows: 18 EXECUTED forming 9 round trips, 3 REJECTED; ids 1..21 in list order), LEGACY_TRADE_COUNT = 21, LEGACY_EXECUTED_COUNT = 18, LEGACY_REALIZED_TOTAL = 12.0, LEGACY_POSITIONS: list[tuple] (2 flat rows).
- Task 3: Fixture db_path(tmp_path, monkeypatch) also clears STARTING_CASH and ACCOUNT_TYPE from the environment before constructing Database(path, now=OPENED_AT), so the fixture DB is always a $500 cash account opened at 2026-01-01T00:00:00+00:00.
- Task 3: Test convention (applies to every task): any test that constructs Database() itself must pass now= (a tz-aware UTC datetime earlier than every timestamp the test writes) or pin account.opened_at with a direct UPDATE account SET opened_at = ? WHERE id = 1 immediately after construction (as Task 4's _open_account does). Relying on the wall-clock seed is forbidden because every ledger read filters created_at >= opened_at.
- Task 4: BudgetTracker.db_path: str — resolved database path (db_path or DB_PATH); other threads open their own connection with connect(budget.db_path). Task 9 _close_the_books must use this (or intraday.conn) for signal_log.label_pending, never budget.conn.
- Task 4: _now(now: datetime | None = None) -> str — module-level helper gains an optional tz-aware override; unchanged output with no argument.
- Task 4: DAY_FLAGS = ('loss_tripped_at', 'loss_announced_at', 'report_posted_at') — module constant; set_day_flag raises ValueError for any other column.
- Task 4: BudgetTracker._txn() — @contextmanager; acquires self._lock, BEGIN IMMEDIATE, commit on success / rollback on exception. conn.isolation_level = None (autocommit). Ownership rule: budget.conn may be read from any thread but written through (bare execute or `with budget.conn:`) only on the FastTrader.cycle thread, sequentially with _txn; other threads use connect(budget.db_path).
- Task 4: BudgetTracker._account() -> sqlite3.Row — the account row; RuntimeError when missing.
- Task 4: BudgetTracker._sum_since_open(column: str) -> float — SUM(column) over EXECUTED trades with created_at >= opened_at.
- Task 4: BudgetTracker._positions_value(prices: dict) -> float — Σ shares × prices.get(symbol) with avg_price fallback; shared by get_equity and record_equity.
- Task 4: BudgetTracker._available_at(row: sqlite3.Row) -> str — settlement timestamp for a SELL row (crypto/margin: created_at; else next_trading_day_open(created_at) as UTC ISO seconds).
- Task 4: BudgetTracker._apply_position(row: sqlite3.Row, ts: str) -> None — existing private name, now takes the transaction timestamp.
- Task 5: src/signal_log.py: _now_iso(now: datetime | None = None) -> str — UTC ISO seconds, `now` injected by tests.
- Task 5: src/signal_log.py: _et_date(ts_iso: str) -> str — ET calendar date of a tz-aware ISO string; local copy of Task 4's budget_tracker._et_date so Task 5 depends only on Task 3.
- Task 5: src/signal_log.py: _forward_bars(conn, symbol: str, bar_ts: str, n_bars: int) -> pd.DataFrame — the signal's bar plus the next n_bars-1 stored bars, UTC DatetimeIndex, columns high, low, close; matches ts through SQLite datetime().
- Task 5: src/signal_log.py: _bar_minutes() -> float — bar length in minutes parsed from INTERVAL ('5m' -> 5.0, '15m' -> 15.0, '1h' -> 60.0); local sibling of Task 6's fast_trader._interval_minutes (Task 6 depends on Task 5, so Task 5 cannot import it).
- Task 5: src/signal_log.py: _session_close(trade_date: str) -> datetime — ET close of the stock session on an ET date: 16:00, or 13:00 when the date is in intraday_engine.EARLY_CLOSE_2026.
- Task 5: src/signal_log.py: _session_complete(bars: pd.DataFrame, trade_date: str, now: datetime) -> bool — True only when now (tz-aware) >= _session_close(trade_date) AND the latest stored bar dated trade_date starts at or after close - _bar_minutes() minutes (the closing bar, 15:55 ET on a regular day).
- Task 5: label_pending semantics beyond the contract text: a stock signal with fewer than horizon+1 stored bars is still labelled when _session_complete holds, via triple_barrier with horizon min(horizon, len(bars)-1) (same-day labelling of late stock signals); a signal whose own bar is not stored is skipped with a WARNING and left NULL; crypto is never labelled short of horizon+1 bars; label_pending reads signals.trade_date and must be called on a connection BudgetTracker does not own (IntradayEngine.conn or a fresh connect(path)), never budget_tracker.conn.
- Task 6: src/fast_trader.py: BARRIER_EPS = 1e-9 (module constant); stop/TP comparisons are `change <= -sl + BARRIER_EPS` and `change >= tp - BARRIER_EPS`
- Task 6: src/fast_trader.py: _interval_minutes(interval: str | None = None) -> float — parses INTERVAL ('5m' -> 5.0, '1h' -> 60.0, unparseable -> 5.0); used only by max_hold_min
- Task 6: src/fast_trader.py: FastTrader.cycle(self, now: datetime | None = None) -> dict — now is UTC tz-aware, defaults to datetime.now(timezone.utc); threaded to market_state, minutes_to_close, log_trade, execute_trade, size_order, get_buying_power, get_unsettled, signal_log.record, set_day_flag (as now.isoformat(timespec='seconds'))
- Task 6: src/fast_trader.py: FastTrader._account_fields(self, prices: dict, now) -> dict with keys equity, buying_power, unsettled — merged into the cycle summary (also on the closed-market early return, with prices={})
- Task 6: src/fast_trader.py: class_gate(cls, None) semantics — metrics=None is treated as ev=0.0: passes the cost gate, then returns (False, 'Not trading. Model loses money (+0.000% per trade) on these settings.') unless FAST_IGNORE_EV is set, in which case (True, 'trading on paper despite EV +0.000% below the 0.30% floor (FAST_IGNORE_EV on)'). Task 7 must not assume (True, ...) for a missing-metrics class unless the flag is on
- Task 6: src/fast_trader.py: 'Not trading' texts are plain (no markdown/emoji): 'Not trading. Edge of {ev*100:+.3f}% per trade is real but smaller than the {floor:.2%} it costs to get in and out, so it would lose money after fees.' and 'Not trading. Model loses money ({ev*100:+.3f}% per trade) on these settings.' — EV printed as ev*100 because metrics[cls]['ev'] is a fraction
- Task 6: Spec §3 step 3 literal correction: the FAST_IGNORE_EV text is 'trading on paper despite EV {ev*100:+.3f}% below the {floor:.2%} floor (FAST_IGNORE_EV on)', not '{ev:+.3f}%'; Tasks 9 and 10 rely on the ev*100 output
- Task 6: Contract dependency table correction: Task 7 depends on 4 and 6 (imports class_gate from src/fast_trader.py) and on 8 for the live report (first_close_on_or_after in src/ml_engine.py); row should read '4, 6 (8 for the live report)'
- Task 6: Contract test-conventions rule: any test that constructs Database() and then reads anything filtered by account.opened_at (get_unsettled, get_buying_power, get_fees_paid, get_realized_pnl, get_gross_pnl, get_trades_since_open, build_scorecard) MUST pass Database(path, now=<fixed UTC datetime earlier than every trade stamp>) or pin opened_at with UPDATE account SET opened_at = ?; tests/test_fast_trader.py::make_budget uses Database(path, now=OPENED_AT) with OPENED_AT = 2026-09-01T00:00Z
- Task 6: FastTrader.stop_loss / take_profit legacy properties are kept (still read by src/discord_bot.py:294-295); only max_hold_min is deleted
- Task 6: Threading note for src/signal_log.py callers: record and mark_executed run on budget.conn from the cycle worker thread (same thread as BudgetTracker._txn; autocommit connection so `with conn:` is safe); only label_pending (Task 9, event-loop thread) must use its own connection rather than budget.conn
- Task 7: src/scorecard.py module constants: Z95 = 1.96, CLASSES = ('stock', 'crypto'), BARRIER_EXITS = ('tp', 'sl', 'timeout', 'eod')
- Task 7: mean_ci is a 95% normal interval (z = 1.96), not Student-t; spec section 4 line 271 amended in Step 5
- Task 7: Task 7 dependency row amended (Step 5) to '4, 5, 6 (8 for the live report)'
- Task 7: _net_return(row) -> float | None  # realized_pnl / (amount - realized_pnl)
- Task 7: _max_drawdown_pct(series: list[dict]) -> float  # PERCENTAGE POINTS of running peak over equity_series()
- Task 7: _signal_stats(conn, cls: str, opened_at: str) -> dict  # sig_n, sig_hits, sig_rate, sig_lo, sig_hi, sig_lo_day, sig_days; sig_lo_day = 0.0 with < 2 trade dates
- Task 7: classes[cls] extra keys: sig_hits, sig_days, tp_n, breakeven, plus raw verdict inputs gated, exec_n, exec_mean, exec_se, ev_bt, cost
- Task 7: classes[cls]['gated'] / verdict stats['gated'] = COST gate only (barriers(cls)[0] <= round_trip_cost(cls)); EV-gated class reads EXTEND not NO-GO; gate_text = class_gate(cls, metrics.get(cls))[1], called with None when the class has no metrics (Task 6 scores None as ev 0.0)
- Task 7: None-handling: bt_precision, ev_bt, ev_bt_net are None and bt_n is 0 when metrics has no entry for the class; since_open.profit_factor None when no losing trade; positions[] price/market_value/pnl/pnl_pct/pct_vs_ref None when the mark is missing; renderers must print 'n/a' for None
- Task 7: Units: exec_mean_pct, exec_ci, since_open.max_drawdown_pct are PERCENTAGE POINTS (render f'{x:+.3f}%' / f'{x:.2f}%'); every other rate key is a fraction (render with a % format spec)
- Task 7: account.unsettled_until: ET date string 'YYYY-MM-DD' or None; render verbatim as f'settles {unsettled_until} 09:30 ET', never parse as datetime
- Task 7: closed_today[] item keys: symbol, shares, price, amount, net, realized_pnl, gross, gross_pnl, fees, exit_reason, created_at  (net == realized_pnl, gross == gross_pnl; both spellings always present)
- Task 7: positions[] item keys: symbol, shares, avg_price, entry_ref, price, market_value, pnl, pnl_pct, pct_vs_ref
- Task 7: since_open.days_running counts the opening ET date as day 1; today.n_trades counts every EXECUTED BUY and SELL with trade_date == day_et
- Task 7: spy = {'start_close', 'last_close', 'pct' (fraction), 'value' (starting_cash * (1+pct))} or None
- Task 7: Position marks and the SPY benchmark use engine.stored_close, never latest_price
- Task 8: Contract dependency-table correction: Task 7 (scorecard) 'Depends on' cell changes from '4' to '4, 6 (8 for the live report)' because src/scorecard.py imports class_gate from src/fast_trader.py (Task 6) and build_scorecard calls engine.first_close_on_or_after (Task 8) at runtime. No new names or signatures.
- Task 9: TradingBot._startup_embed(self) -> discord.Embed
- Task 9: TradingBot._class_gates(self) -> list[tuple[str, bool, str]]
- Task 9: TradingBot._gate_lines(self) -> str
- Task 9: TradingBot._held_prices(self) -> dict
- Task 9: TradingBot._add_account_fields(self, embed, summary=None) -> None
- Task 9: TradingBot._scorecard_embed(self, day_et: str, now=None) -> discord.Embed  # consumes Task 7 shapes: closed_today net/gross; unsettled_until 'YYYY-MM-DD' rendered verbatim; exec_mean_pct/exec_ci/max_drawdown_pct are percentage points; bt_precision/ev_bt_net None -> 'n/a'; positions[].pct_vs_ref; adds 'Stocks resume <Day> 09:30 ET' via next_trading_day_open
- Task 9: async TradingBot._embed_pnl(self, now=None) -> discord.Embed  # /pnl and /summary; now picks the ET date; never writes day_state/equity_history
- Task 9: async TradingBot._post_daily_report(self, day_et: str, now=None) -> bool
- Task 9: TradingBot._close_the_books(self, day_et: str, now=None) -> None  # label_pending on a fresh connect(<ledger file via PRAGMA database_list>) connection, never BudgetTracker.conn; then record_equity
- Task 9: TradingBot._loss_limit_embed(self, summary) -> discord.Embed
- Task 9: async TradingBot._embed_account(self) -> discord.Embed  # fields Balance, Cash, Unsettled, Buying power, Starting cash, Opened (=account.opened_at), All-time, optional stale-quotes
- Task 9: src/discord_bot.py module-level: _clip_lines(lines, limit: int = 1000) -> str, _signed_usd(x: float) -> str, _icon(cls: str) -> str  (no _et_day_label)
- Task 10: Behavioural note (Task 5, no new name): signal_log.label_pending leaves a stock signal NULL while fewer than horizon+1 bars are stored after it, even if the ET session has ended; same-day labelling covers stock signals up to ~14:00 ET, later ones are labelled 0 at the next trading day's 16:05 tick.
- Task 10: Behavioural note (Task 9, no new name): TradingBot.daily_summary runs its first iteration immediately at startup; a restart at or after 16:05 ET posts that ET date's scorecard during warm-up (before the restart notice), with metrics from the saved model blob and spy=None until a SPY close on/after opened_at[:10] is stored.
