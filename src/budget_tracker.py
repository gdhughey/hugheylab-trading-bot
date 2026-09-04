#!/usr/bin/env python3
"""
Budget tracker - enforces the weekly spend cap and records trade lifecycle.

Budget semantics:
  * Only BUY trades consume budget; SELL returns capital and is not counted.
  * "Committed" = EXECUTED + PENDING. can_trade()/get_remaining_budget() work
    against committed spend so that trades awaiting Discord approval cannot
    collectively overshoot the weekly cap.
  * get_weekly_spent() reports EXECUTED spend only - that is what actually left
    the account, and it is what the Discord embeds label "Spent".
"""

import os
import logging
from datetime import datetime, timezone

from src.database import connect

logger = logging.getLogger(__name__)


def _week_key(dt: datetime = None) -> str:
    """ISO year-week, e.g. '2026-W36'. Budget rolls over on Monday."""
    dt = dt or datetime.now()
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class BudgetTracker:
    def __init__(self, db_path: str = None):
        self.conn = connect(db_path)
        self.weekly_budget = float(os.getenv('WEEKLY_BUDGET', 5000))
        logger.info(f"BudgetTracker ready (weekly budget ${self.weekly_budget:,.2f})")

    # --- internals -------------------------------------------------------

    def _sum(self, statuses) -> float:
        placeholders = ','.join('?' for _ in statuses)
        row = self.conn.execute(
            f"SELECT COALESCE(SUM(amount), 0) AS total FROM trades "
            f"WHERE week_key = ? AND side = 'BUY' AND status IN ({placeholders})",
            (_week_key(), *statuses),
        ).fetchone()
        return float(row['total'])

    def _committed(self) -> float:
        return self._sum(('EXECUTED', 'PENDING'))

    # --- budget ----------------------------------------------------------

    def get_weekly_spent(self) -> float:
        """Money actually spent this week (approved trades only)."""
        return self._sum(('EXECUTED',))

    def get_remaining_budget(self) -> float:
        """Budget left after executed AND pending-approval trades."""
        return max(0.0, self.weekly_budget - self._committed())

    def can_trade(self, amount: float) -> bool:
        if amount <= 0:
            return False
        return (self._committed() + amount) <= self.weekly_budget

    # --- trade lifecycle -------------------------------------------------

    def log_trade(self, symbol: str, side: str, price: float, shares: int) -> int:
        """Record a PENDING trade awaiting Discord approval. Returns its id."""
        amount = float(price) * int(shares)
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO trades (symbol, side, price, shares, amount, status, created_at, week_key) "
                "VALUES (?, ?, ?, ?, ?, 'PENDING', ?, ?)",
                (symbol, side, float(price), int(shares), amount, _now(), _week_key()),
            )
        trade_id = cur.lastrowid
        logger.info(f"Logged PENDING trade #{trade_id}: {side} {shares} {symbol} @ ${price:.2f}")
        return trade_id

    def execute_trade(self, trade_id: int):
        """Mark an approved trade executed and update the position."""
        row = self.conn.execute(
            "SELECT * FROM trades WHERE id = ? AND status = 'PENDING'", (trade_id,)
        ).fetchone()
        if row is None:
            logger.warning(f"execute_trade: trade #{trade_id} not found or not pending")
            return False

        with self.conn:
            self.conn.execute(
                "UPDATE trades SET status = 'EXECUTED', settled_at = ? WHERE id = ?",
                (_now(), trade_id),
            )
            self._apply_position(row)
        logger.info(f"Trade #{trade_id} EXECUTED")
        return True

    def reject_trade(self, trade_id: int):
        """Mark a trade rejected (declined or timed out); frees its budget hold."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE trades SET status = 'REJECTED', settled_at = ? "
                "WHERE id = ? AND status = 'PENDING'",
                (_now(), trade_id),
            )
        if cur.rowcount:
            logger.info(f"Trade #{trade_id} REJECTED")
        return bool(cur.rowcount)

    def _apply_position(self, row):
        """Weighted-average position update. Caller holds the transaction.

        On a SELL this also books realized P&L against the position's average
        cost, which is what makes the paper-trading record scoreable.
        """
        pos = self.conn.execute(
            "SELECT shares, avg_price FROM positions WHERE symbol = ?", (row['symbol'],)
        ).fetchone()
        held = pos['shares'] if pos else 0
        avg = pos['avg_price'] if pos else 0.0

        if row['side'] == 'BUY':
            new_shares = held + row['shares']
            new_avg = ((held * avg) + row['amount']) / new_shares if new_shares else 0.0
        else:
            closed = min(row['shares'], held)
            realized = (row['price'] - avg) * closed if held else 0.0
            self.conn.execute(
                "UPDATE trades SET realized_pnl = ? WHERE id = ?", (realized, row['id'])
            )
            new_shares = held - row['shares']
            new_avg = avg if new_shares > 0 else 0.0
            if new_shares < 0:
                logger.warning(f"SELL exceeds held shares for {row['symbol']} - clamping to 0")
                new_shares, new_avg = 0, 0.0

        self.conn.execute(
            "INSERT INTO positions (symbol, shares, avg_price, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET shares=excluded.shares, "
            "avg_price=excluded.avg_price, updated_at=excluded.updated_at",
            (row['symbol'], new_shares, new_avg, _now()),
        )

    # --- reporting -------------------------------------------------------

    def get_positions(self) -> list:
        rows = self.conn.execute(
            "SELECT symbol, shares, avg_price, updated_at FROM positions WHERE shares > 0 "
            "ORDER BY symbol"
        ).fetchall()
        return [
            {
                'symbol': r['symbol'],
                'shares': r['shares'],
                'avg_price': round(r['avg_price'], 2),
                'cost_basis': round(r['shares'] * r['avg_price'], 2),
                'updated_at': r['updated_at'],
            }
            for r in rows
        ]

    def get_realized_pnl(self) -> float:
        """Total booked profit/loss from closed (sold) paper positions."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS total FROM trades "
            "WHERE status = 'EXECUTED' AND realized_pnl IS NOT NULL"
        ).fetchone()
        return float(row['total'])

    def get_pnl(self, price_fn) -> dict:
        """Mark open positions to market.

        `price_fn(symbol)` returns a current price, or None when unavailable -
        those positions are reported as stale rather than silently valued at
        cost, so a data outage can never masquerade as break-even.
        """
        positions, unrealized, cost_total, market_total, stale = [], 0.0, 0.0, 0.0, []

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
        return {
            'positions': positions,
            'stale': stale,
            'realized': realized,
            'unrealized': unrealized,
            'total': realized + unrealized,
            'cost_basis': cost_total,
            'market_value': market_total,
            'return_pct': (unrealized / cost_total) if cost_total else 0.0,
        }

    def set_weekly_budget(self, amount: float):
        self.weekly_budget = float(amount)
        logger.info(f"Weekly budget changed to ${self.weekly_budget:,.2f}")
        return self.weekly_budget

    def get_statistics(self) -> dict:
        agg = self.conn.execute(
            "SELECT status, COUNT(*) AS n, COALESCE(SUM(amount), 0) AS total "
            "FROM trades GROUP BY status"
        ).fetchall()
        counts = {r['status']: r['n'] for r in agg}
        executed = counts.get('EXECUTED', 0)
        rejected = counts.get('REJECTED', 0)
        decided = executed + rejected

        return {
            'Trades Executed': executed,
            'Trades Rejected': rejected,
            'Awaiting Approval': counts.get('PENDING', 0),
            'Approval Rate': f"{(executed / decided):.0%}" if decided else "n/a",
            'Open Positions': len(self.get_positions()),
            'Week': _week_key(),
            'Weekly Budget': f"${self.weekly_budget:,.2f}",
            'Spent This Week': f"${self.get_weekly_spent():,.2f}",
            'Remaining': f"${self.get_remaining_budget():,.2f}",
            'Realized P&L': f"${self.get_realized_pnl():,.2f}",
        }
