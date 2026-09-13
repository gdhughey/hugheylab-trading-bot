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

    def get_trades_since_open(self, day: str | None = None) -> list[sqlite3.Row]:
        """EXECUTED trades since the account opened, optionally for one ET trade_date."""
        sql = "SELECT * FROM trades WHERE status = 'EXECUTED' AND created_at >= ?"
        params = [self.opened_at()]
        if day:
            sql += " AND trade_date = ?"
            params.append(day)
        return self.conn.execute(sql + " ORDER BY created_at, id", params).fetchall()
