#!/usr/bin/env python3
"""
Loss review: turn one day's trades into a facts block the analyst can reason
about, and pre-compute the patterns a human would look for.

Python does the forensics - pairing entries with exits, minutes held, the
gap against the prior close, the 1m price path around each entry, headline
counts - and the model writes the narrative. Nothing here changes what the
bot does; the daily loss limit is the control, this is the thinking.

Triggered by FastTrader (summary['review_due']) once per ET date, and on
demand with /postmortem [YYYY-MM-DD].
"""

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
OPEN_WINDOW_MIN = 15      # "entered in the first N minutes" pattern
QUICK_STOP_MIN = 10       # "stopped out within N minutes" pattern


def _et(ts: str) -> datetime:
    d = datetime.fromisoformat(ts)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(ET)


def _usd(x):
    return "n/a" if x is None else (f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}")


def pair_trades(rows) -> list[dict]:
    """Match each SELL to the most recent unmatched BUY of the same symbol.

    Rows are EXECUTED ledger rows in created_at order (BudgetTracker.
    get_trades_since_open). A BUY with no SELL yet is returned as still open.
    """
    open_buys: dict[str, list] = {}
    out = []
    for r in rows:
        if r['side'] == 'BUY':
            open_buys.setdefault(r['symbol'], []).append(r)
            continue
        buys = open_buys.get(r['symbol']) or []
        buy = buys.pop(0) if buys else None
        entry, exit_ = (_et(buy['created_at']) if buy else None), _et(r['created_at'])
        out.append({
            'symbol': r['symbol'],
            'entry_at': entry, 'exit_at': exit_,
            'held_min': (exit_ - entry).total_seconds() / 60 if entry else None,
            'entry_px': float(buy['price']) if buy else None,
            'exit_px': float(r['price']),
            'entry_p': float(buy['entry_probability']) if buy and buy['entry_probability'] is not None else None,
            'exit_reason': r['exit_reason'],
            'pnl': float(r['realized_pnl'] or 0),
            'pct': ((float(r['price']) / float(buy['price']) - 1) if buy else None),
            'open': False,
        })
    for sym, buys in open_buys.items():
        for b in buys:
            out.append({'symbol': sym, 'entry_at': _et(b['created_at']), 'exit_at': None,
                        'held_min': None, 'entry_px': float(b['price']), 'exit_px': None,
                        'entry_p': float(b['entry_probability']) if b['entry_probability'] is not None else None,
                        'exit_reason': None, 'pnl': 0.0, 'pct': None, 'open': True})
    out.sort(key=lambda t: t['entry_at'] or t['exit_at'])
    return out


def prior_close(conn, symbol: str, day_et: str):
    """Last stored 5m close before `day_et` (the previous session's close)."""
    row = conn.execute(
        "SELECT close FROM prices_intraday WHERE symbol = ? AND interval = '5m' "
        "AND ts < ? ORDER BY ts DESC LIMIT 1", (symbol, f"{day_et}T00:00")).fetchone()
    return float(row[0]) if row else None


def price_path(conn, symbol: str, start: datetime, end: datetime):
    """(low, high, last) of the 1m bars whose start lies in [start, end].

    Timestamps are stored as yfinance served them (ET offset for stocks, UTC
    for crypto), so the window is applied to parsed datetimes, not to the
    ISO strings - a lexical compare across offsets is wrong.
    """
    day = start.astimezone(timezone.utc).date()
    rows = conn.execute(
        "SELECT ts, low, high, close FROM prices_intraday WHERE symbol = ? AND interval = '1m' "
        "AND ts >= ? AND ts < ? ORDER BY ts",
        (symbol, (day - timedelta(days=1)).isoformat(), (day + timedelta(days=2)).isoformat())).fetchall()
    keep = []
    for r in rows:
        ts = datetime.fromisoformat(r[0])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if start <= ts <= end:
            keep.append(r)
    if not keep:
        return None
    return (min(float(r[1]) for r in keep), max(float(r[2]) for r in keep), float(keep[-1][3]))


def headline_count(conn, symbol: str, before: datetime, hours=24) -> int:
    try:
        row = conn.execute(
            "SELECT count(*) FROM news WHERE symbol = ? AND published_at >= ? AND published_at <= ?",
            (symbol, (before - timedelta(hours=hours)).astimezone(timezone.utc).isoformat(timespec='seconds'),
             before.astimezone(timezone.utc).isoformat(timespec='seconds'))).fetchone()
        return int(row[0])
    except Exception:
        return 0


def build_postmortem_context(conn, budget, day_et: str, reason: str = None, now=None) -> str:
    """Facts block for one ET trade date. `conn` is any connection to the
    ledger DB (for bars/news); `budget` is the BudgetTracker."""
    now = now or datetime.now(timezone.utc)
    rows = budget.get_trades_since_open(day_et)
    trades = pair_trades(rows)
    day = budget.get_day_state(day_et)
    start_eq = float(day['start_equity']) if day else None

    closed = [t for t in trades if not t['open']]
    realised = sum(t['pnl'] for t in closed)
    lines = [f"Loss review for {day_et} (ET)." + (f" Triggered because: {reason}." if reason else "")]
    if start_eq:
        lines.append(f"Start-of-day equity {_usd(start_eq)}; realised so far {_usd(realised)} "
                     f"({realised / start_eq * 100:+.2f}% of the start).")
    if not trades:
        lines.append("No trades on this date.")
        return "\n".join(lines)

    lines.append(f"{len(closed)} closed trades: "
                 f"{sum(1 for t in closed if t['pnl'] > 0)} wins, "
                 f"{sum(1 for t in closed if t['pnl'] < 0)} losses; exit reasons: "
                 + ", ".join(f"{k} x{v}" for k, v in _counts(t['exit_reason'] for t in closed).items()) + ".")

    lines.append("Trades, in order:")
    early, quick, gaps = 0, 0, []
    for t in trades:
        sym = t['symbol']
        e_at = t['entry_at']
        parts = [f"- {sym}:"]
        if e_at:
            since_open = (e_at - e_at.replace(hour=9, minute=30, second=0, microsecond=0)).total_seconds() / 60
            parts.append(f"entered {e_at.strftime('%H:%M:%S')} ET ({since_open:+.0f} min from the open) "
                         f"at {_usd(t['entry_px'])}"
                         + (f", model p={t['entry_p']:.3f}" if t['entry_p'] is not None else ""))
            if 0 <= since_open < OPEN_WINDOW_MIN:
                early += 1
            pc = prior_close(conn, sym, day_et)
            if pc and t['entry_px']:
                gap = t['entry_px'] / pc - 1
                gaps.append(gap)
                parts.append(f"; prior close {_usd(pc)}, so entry was {gap * 100:+.1f}% vs yesterday")
        if t['open']:
            parts.append("; still open.")
        else:
            parts.append(f"; exited {t['exit_at'].strftime('%H:%M:%S')} ET at {_usd(t['exit_px'])} "
                         f"({t['exit_reason'] or 'n/a'}) after {t['held_min']:.0f} min, "
                         f"net {_usd(t['pnl'])}"
                         + (f" ({t['pct'] * 100:+.2f}%)" if t['pct'] is not None else "") + ".")
            if t['held_min'] is not None and t['held_min'] <= QUICK_STOP_MIN and t['exit_reason'] == 'sl':
                quick += 1
            if e_at:
                path = price_path(conn, sym, e_at.replace(second=0, microsecond=0) - timedelta(minutes=1),
                                  t['exit_at'])
                if path:
                    lo, hi, last = path
                    parts.append(f" 1m bars while held: low {_usd(lo)}, high {_usd(hi)}, last {_usd(last)}.")
        if e_at:
            n = headline_count(conn, sym, e_at)
            parts.append(f" Headlines in the 24h before entry: {n}.")
        lines.append(" ".join(parts))

    pats = []
    if early:
        pats.append(f"{early} of {len(trades)} entries were placed within {OPEN_WINDOW_MIN} min of the open")
    if quick:
        pats.append(f"{quick} of {len(closed)} closed trades were stopped out within {QUICK_STOP_MIN} min")
    if gaps:
        pats.append(f"average entry was {sum(gaps) / len(gaps) * 100:+.1f}% above the prior close "
                    f"({sum(1 for g in gaps if g > 0.02)} of {len(gaps)} gapped up more than 2%)")
    if pats:
        lines.append("Patterns Python found: " + "; ".join(pats) + ".")
    return "\n".join(lines)


def _counts(items):
    out = {}
    for x in items:
        out[x or 'n/a'] = out.get(x or 'n/a', 0) + 1
    return out


def build_trade_context(conn, budget, symbol: str, now=None) -> str:
    """Facts for `/why SYMBOL`: the most recent round trip (or open position)
    in that name, with the same forensics as the day review - plus the
    signal log's view of what the model was saying around entry."""
    now = now or datetime.now(timezone.utc)
    symbol = symbol.upper()
    rows = [r for r in budget.get_trades_since_open() if r['symbol'] == symbol]
    trades = pair_trades(rows)
    if not trades:
        return f"No trades in {symbol} since the account opened."
    t = trades[-1]
    e_at = t['entry_at']
    day_et = (e_at or t['exit_at']).strftime('%Y-%m-%d')
    lines = [f"Most recent {symbol} trade ({day_et} ET)."]
    if e_at:
        since_open = (e_at - e_at.replace(hour=9, minute=30, second=0, microsecond=0)).total_seconds() / 60
        lines.append(f"Entered {e_at.strftime('%H:%M:%S')} ET ({since_open:+.0f} min from the open) at {_usd(t['entry_px'])}"
                     + (f", model p={t['entry_p']:.3f}" if t['entry_p'] is not None else "") + ".")
        pc = prior_close(conn, symbol, day_et)
        if pc and t['entry_px']:
            lines.append(f"Prior session close {_usd(pc)}: entry was {(t['entry_px'] / pc - 1) * 100:+.1f}% vs yesterday.")
    if t['open']:
        lines.append("Still open.")
    else:
        lines.append(f"Exited {t['exit_at'].strftime('%H:%M:%S')} ET at {_usd(t['exit_px'])} ({t['exit_reason'] or 'n/a'})"
                     + (f" after {t['held_min']:.0f} min" if t['held_min'] is not None else "")
                     + f", net {_usd(t['pnl'])}" + (f" ({t['pct'] * 100:+.2f}%)" if t['pct'] is not None else "") + ".")
        if e_at:
            path = price_path(conn, symbol, e_at.replace(second=0, microsecond=0) - timedelta(minutes=1), t['exit_at'])
            if path:
                lo, hi, last = path
                lines.append(f"1m bars while held: low {_usd(lo)}, high {_usd(hi)}, last {_usd(last)}.")
    if e_at:
        # what the model thought in the hour around entry (the signal log records every scored bar)
        try:
            sig = conn.execute(
                "SELECT bar_ts, probability, bar, above_bar, label FROM signals WHERE symbol = ? "
                "AND bar_ts BETWEEN ? AND ? ORDER BY bar_ts",
                (symbol, (e_at - timedelta(minutes=60)).astimezone(timezone.utc).isoformat(),
                 (e_at + timedelta(minutes=60)).astimezone(timezone.utc).isoformat())).fetchall()
            if sig:
                above = sum(1 for r in sig if r[2] is not None and r[1] >= r[2])
                lines.append(f"Model probability on the {len(sig)} bars within an hour of entry: "
                             f"min {min(r[1] for r in sig):.3f}, max {max(r[1] for r in sig):.3f}, "
                             f"{above} of them over the {sig[0][2]:.3f} bar.")
        except Exception:
            pass
        lines.append(f"Headlines in the 24h before entry: {headline_count(conn, symbol, e_at)}.")
        try:
            heads = conn.execute(
                "SELECT published_at, headline FROM news WHERE symbol = ? AND published_at <= ? "
                "ORDER BY published_at DESC LIMIT 4",
                (symbol, e_at.astimezone(timezone.utc).isoformat(timespec='seconds'))).fetchall()
            for p, h in heads:
                lines.append(f"- {h} ({p[:16]}Z)")
        except Exception:
            pass
    return "\n".join(lines)
