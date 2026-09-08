#!/usr/bin/env python3
"""
Intraday auto-trading loop.

Runs every FAST_POLL_SECONDS during regular market hours. Unlike the daily loop
this one manages exits, because an intraday entry without an exit rule is just a
buy-and-hold with extra steps.

Order of operations each cycle matters: refresh prices, then EXIT before ENTER.
Exiting first frees capital and position slots in the same cycle, and means a
stop-loss is never delayed by an unrelated entry.

Still paper. Every "trade" is a row in SQLite; no broker is connected.
"""

import os
import logging
from datetime import datetime, timedelta, timezone

from src.intraday_engine import (market_state, minutes_to_close, ET,
                                 is_crypto, asset_class, barriers)

logger = logging.getLogger(__name__)


def _cfg(name, default, cast=float):
    try:
        return cast(os.getenv(name, default))
    except (TypeError, ValueError):
        return cast(default)


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

        These lived only in memory, so any position opened before a restart had
        no entry time and FAST_MAX_HOLD_MIN could never fire for it - the
        position would sit indefinitely (crypto has no end-of-day backstop).
        The service restarted four times in two days, so this was live.
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
    @property
    def max_hold_min(self): return _cfg('FAST_MAX_HOLD_MIN', 120)

    def _price(self, symbol, fallback):
        if self.quote_fn:
            try:
                p = self.quote_fn(symbol)
                if p:
                    return float(p)
            except Exception:
                pass
        return fallback

    # --- the cycle -------------------------------------------------------

    def cycle(self):
        """One pass. Returns a dict describing what happened (for reporting).

        Synchronous by design - the caller runs it in a worker thread, because
        everything in here (yfinance, sklearn, sqlite) blocks.
        """
        state, desc = market_state()
        stocks_open = state == 'open'
        crypto_syms = [s for s in self.engine.symbols if is_crypto(s)]
        summary = {'state': state, 'desc': desc, 'exits': [], 'entries': [],
                   'skipped': [], 'candidates': [], 'ts': datetime.now(ET),
                   'stocks_open': stocks_open, 'crypto': len(crypto_syms)}

        # Crypto never closes, so an outside-hours cycle is still a working
        # cycle whenever the universe holds any coins.
        if not stocks_open and not crypto_syms:
            summary['note'] = f"Market {desc}, no crypto in universe - not trading."
            self.last_summary = summary
            return summary

        # Refuse to trade an asset class whose backtested expected value is
        # negative. Now that stocks and crypto have separate models this is
        # per-class, so a losing crypto model no longer drags stocks down with
        # it (or vice versa). Set FAST_IGNORE_EV=1 to override deliberately.
        respect_ev = os.getenv('FAST_IGNORE_EV', '0') not in ('1', 'true', 'yes')
        # An EV barely above zero is not tradeable: a round trip costs roughly
        # 5-40 bps on stocks and 22-100+ bps on retail crypto venues, none of
        # which the backtest models. Require a real margin over costs, not just
        # a positive sign.
        min_ev = float(os.getenv('MIN_EV_TO_TRADE', 0.003))
        blocked = set()
        if respect_ev:
            for cls, m in (self.engine.metrics or {}).items():
                if m.get('ev', 0) < min_ev:
                    blocked.add(cls)
        summary['blocked_classes'] = sorted(blocked)

        def tradeable(sym):
            if respect_ev and asset_class(sym) in blocked:
                return False
            return stocks_open or is_crypto(sym)

        rows = self.engine.fetch()
        summary['rows'] = rows

        held = {p['symbol']: p for p in self.budget.get_positions()}
        to_close = minutes_to_close()
        summary['minutes_to_close'] = to_close

        # ---- EXITS first: frees cash and slots within this same cycle ----
        for sym, pos in held.items():
            # Exits ignore the EV block: an already-open position must always be
            # closeable, otherwise a newly-negative model would strand it.
            if not (stocks_open or is_crypto(sym)):
                continue
            sig = self.engine.signal(sym)
            price = self._price(sym, sig['price'] if sig else pos['avg_price'])
            change = (price - pos['avg_price']) / pos['avg_price'] if pos['avg_price'] else 0
            reason = None

            # Exits must use the SAME barriers the model was trained on, and
            # those differ by asset class.
            cls = asset_class(sym)
            tp, sl, _ = barriers(cls)

            # Crypto has no close to flatten into - holding it overnight is
            # normal, so the EOD rule applies to stocks only.
            if not is_crypto(sym) and to_close <= self.eod_flatten_min:
                reason = f"end of day ({to_close:.0f} min to close)"
            elif change <= -sl:
                reason = f"stop loss {change:+.2%}"
            elif change >= tp:
                reason = f"take profit {change:+.2%}"
            else:
                opened = self.entry_time.get(sym)
                if opened and (datetime.now(ET) - opened).total_seconds() / 60 > self.max_hold_min:
                    reason = f"held {self.max_hold_min:.0f} min without hitting a target"

            if reason:
                tid = self.budget.log_trade(sym, 'SELL', price, pos['shares'])
                self.budget.execute_trade(tid)
                pnl = (price - pos['avg_price']) * pos['shares']
                summary['exits'].append({
                    'symbol': sym, 'shares': pos['shares'], 'price': price,
                    'reason': reason, 'pnl': pnl, 'pct': change, 'trade_id': tid})
                self.cooldown[sym] = datetime.now(ET) + timedelta(minutes=self.cooldown_min)
                self.entry_time.pop(sym, None)
                logger.info(f"[fast] EXIT {pos['shares']} {sym} @ ${price:,.2f} "
                            f"({reason}) P&L ${pnl:+,.2f}")

        # ---- ENTRIES ------------------------------------------------------
        near_bell = stocks_open and to_close <= self.eod_flatten_min
        if near_bell and not crypto_syms:
            summary['note'] = "Too close to the bell to open anything new."
            self.last_summary = summary
            return summary

        held = {p['symbol']: p for p in self.budget.get_positions()}
        slots = self.max_positions - len(held)
        candidates = self.engine.scan()
        summary['candidates'] = candidates[:5]
        summary['bars'] = {c: self.engine.threshold(c)
                           for c in (self.engine.metrics or {'stock': {}})}
        summary['bar'] = min(summary['bars'].values(), default=0.0)

        if slots <= 0:
            summary['note'] = f"Holding {len(held)}/{self.max_positions} - no free slots."
            self.last_summary = summary
            return summary

        now = datetime.now(ET)
        for sig in candidates:
            if slots <= 0:
                break
            sym = sig['symbol']
            if sym in held:
                continue
            if respect_ev and asset_class(sym) in blocked:
                summary['skipped'].append(
                    (sym, f"{asset_class(sym)} EV below the {min_ev:.2%} cost floor"))
                continue
            if not tradeable(sym):
                summary['skipped'].append((sym, 'market closed'))
                continue
            if near_bell and not is_crypto(sym):
                summary['skipped'].append((sym, 'too close to the bell'))
                continue
            if self.cooldown.get(sym, now) > now:
                summary['skipped'].append((sym, 'cooling down'))
                continue

            price = self._price(sym, sig['price'])
            cash = self.budget.get_remaining_budget()
            shares = int((cash / max(slots, 1)) / price)
            if shares == 0 and price <= cash:
                shares = 1
            if shares == 0:
                summary['skipped'].append((sym, f"${price:,.2f} > ${cash:,.2f} cash"))
                continue
            if not self.budget.can_trade(price * shares):
                summary['skipped'].append((sym, 'over weekly budget'))
                continue

            tid = self.budget.log_trade(sym, 'BUY', price, shares)
            self.budget.execute_trade(tid)
            self.entry_time[sym] = now
            summary['entries'].append({
                'symbol': sym, 'shares': shares, 'price': price,
                'probability': sig['probability'], 'cost': price * shares,
                'trade_id': tid})
            slots -= 1
            logger.info(f"[fast] ENTER {shares} {sym} @ ${price:,.2f} "
                        f"(p={sig['probability']:.3f})")

        if not summary['entries'] and not summary['exits']:
            summary['note'] = (f"{len(candidates)} candidate(s) over the "
                               f"{summary['bar']:.3f} bar; nothing actionable.")
        self.last_summary = summary
        return summary
