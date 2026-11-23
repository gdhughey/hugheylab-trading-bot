#!/usr/bin/env python3
"""
Read-only metrics while the bot is PARKED (NO-GO review 2026-11-14).

The live console (src/webapp.py) runs inside the bot process, so stopping
trading-bot.service also took /metrics and /api/state away, and every tile on
the Noroshi "Trading bot" panel went blank. This stands in on the same port:

  GET /metrics     the same tradingbot_* names, from the ledger and the
                   collector's tables, plus tradingbot_parked 1
  GET /api/state   a minimal document Noroshi's trading_live() understands
                   (no live signals: nothing is being scored)

Nothing here writes: the collector tables are opened mode=ro, and the ledger
is read only through BudgetTracker's get_* methods (no trades, no _txn).
Run it ONLY while trading-bot.service is stopped: both bind WEB_PORT (8090).
The unit file is proxmox/trading-parked-exporter.service (Conflicts= the bot).
"""

import os
import json
import sqlite3
import collections
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
DB = os.getenv('PARKED_DB', os.path.join(os.path.dirname(__file__), '..', 'data', 'trading_bot.db'))
PARKED_NOTE = 'PARKED since 2026-11-14 (NO-GO review): trading loop off, hourly collector still running'


def _conn():
    c = sqlite3.connect(f'file:{os.path.abspath(DB)}?mode=ro', uri=True, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def _age(ts, now):
    if not ts:
        return None
    d = datetime.fromisoformat(str(ts))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return round((now - d).total_seconds())


def snapshot():
    """Ledger numbers via BudgetTracker (positions valued at cost: no quotes are
    fetched while parked) + the collector's freshness. Read-only throughout."""
    from src.budget_tracker import BudgetTracker
    now = datetime.now(timezone.utc)
    day_et = now.astimezone(ET).strftime('%Y-%m-%d')
    bt = BudgetTracker()
    pnl = bt.get_pnl(lambda s: None)
    trades = [dict(r) for r in bt.get_trades_since_open()]
    sells = [t for t in trades if t['side'] == 'SELL']
    today = [t for t in trades if t.get('trade_date') == day_et]
    c = _conn()
    try:
        bars = {iv: c.execute("SELECT max(ts) FROM prices_intraday WHERE interval = ?", (iv,)).fetchone()[0]
                for iv in ('5m', '1m')}
        counts = {}
        for tbl in ('news', 'events', 'signals', 'prices_intraday'):
            try:
                counts[tbl] = c.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
            except sqlite3.Error:
                counts[tbl] = None
        try:
            news_24h = c.execute("SELECT count(*) FROM news WHERE published_at >= ?",
                                 ((now.replace(microsecond=0) - timedelta(hours=24)).isoformat(),)).fetchone()[0]
        except sqlite3.Error:
            news_24h = None
    finally:
        c.close()
    return {
        'now': now.isoformat(timespec='seconds'), 'parked': True, 'note': PARKED_NOTE,
        'account': pnl, 'positions': pnl.get('positions') or [],
        'today': {'n_trades': len(today),
                  'realized': sum(float(t.get('realized_pnl') or 0) for t in today if t['side'] == 'SELL')},
        'since_open': {'n_closed': len(sells),
                       'win_rate': (sum(1 for t in sells if (t.get('realized_pnl') or 0) > 0) / len(sells)) if sells else None,
                       'exit_reasons': dict(collections.Counter(t.get('exit_reason') or 'n/a' for t in sells))},
        'data': {'newest_5m': bars['5m'], 'newest_5m_age_s': _age(bars['5m'], now),
                 'newest_1m': bars['1m'], 'newest_1m_age_s': _age(bars['1m'], now),
                 'counts': counts, 'news_24h': news_24h},
    }


def prometheus(s):
    out = []

    def g(name, value, help_, labels=None):
        if value is None:
            return
        lab = '' if not labels else '{' + ','.join(f'{k}="{v}"' for k, v in labels.items()) + '}'
        out.append(f"# HELP {name} {help_}\n# TYPE {name} gauge\n{name}{lab} {float(value)}")

    a, so, d = s['account'], s['since_open'], s['data']
    g('tradingbot_parked', 1, 'Bot parked: trading loop off, numbers are the final ledger (1/0)')
    g('tradingbot_equity_usd', a['equity'], 'Paper account equity')
    g('tradingbot_cash_usd', a['cash'], 'Settled cash')
    g('tradingbot_buying_power_usd', a['buying_power'], 'Cash available for entries')
    g('tradingbot_unsettled_usd', a['unsettled'], 'Sale proceeds not yet settled')
    g('tradingbot_all_time_net_usd', a['all_time_net'], 'Equity minus starting cash')
    g('tradingbot_all_time_return', a['all_time_pct'], 'All-time return as a fraction')
    g('tradingbot_unrealized_usd', a['unrealized'], 'Open-position P&L (at cost while parked)')
    g('tradingbot_fees_paid_usd', a['fees_paid'], 'Cumulative modelled fees')
    g('tradingbot_positions_open', len(s['positions']), 'Open positions')
    g('tradingbot_today_realized_usd', s['today']['realized'], "Today's realised P&L (ET)")
    g('tradingbot_today_trades', s['today']['n_trades'], "Today's executed trade rows")
    g('tradingbot_closed_trades_total', so['n_closed'], 'Closed trades since account open')
    g('tradingbot_win_rate', so['win_rate'], 'Win rate since account open')
    for reason, n in so['exit_reasons'].items():
        g('tradingbot_exits_total', n, 'Closed trades by exit reason', {'reason': reason})
    # no 5m age: only the (stopped) fast loop writes 5m bars, so it would just read as a red "stale" tile
    g('tradingbot_bar_age_seconds', d['newest_1m_age_s'], 'Age of newest stored bar', {'interval': '1m'})
    for tbl, n in (d['counts'] or {}).items():
        g('tradingbot_table_rows', n, 'Row count', {'table': tbl})
    g('tradingbot_news_rows_24h', d['news_24h'], 'Headlines published in the last 24h')
    return "\n".join(out) + "\n"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        b = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        try:
            if self.path == '/metrics':
                return self._send(200, prometheus(snapshot()), 'text/plain; version=0.0.4; charset=utf-8')
            if self.path in ('/api/state', '/'):
                s = snapshot()
                doc = {'now': s['now'], 'parked': True, 'signals': [],
                       'fast': {'desc': PARKED_NOTE, 'state': 'parked'},
                       'data': s['data'], 'account': {k: s['account'][k] for k in
                                                      ('equity', 'cash', 'all_time_net', 'all_time_pct')}}
                return self._send(200, json.dumps(doc), 'application/json')
            self._send(404, 'not found\n', 'text/plain')
        except Exception as e:   # a bad read must not kill the server
            self._send(500, f'error: {type(e).__name__}\n', 'text/plain')

    def log_message(self, *a):
        pass


def main():
    host, port = os.getenv('WEB_HOST', '0.0.0.0'), int(os.getenv('WEB_PORT', 8090))
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == '__main__':
    main()
