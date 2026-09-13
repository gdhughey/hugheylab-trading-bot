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
