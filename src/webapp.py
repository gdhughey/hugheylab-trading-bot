#!/usr/bin/env python3
"""
Live console: a small aiohttp app inside the bot process.

  GET /              the single-page console (web/)
  GET /api/state     everything the page shows, as one JSON document
  GET /api/events    server-sent events; a `state` event every WEB_PUSH_S
                     seconds and immediately after each fast cycle
  GET /metrics       Prometheus exposition for the Grafana dashboard

Read-only by construction: no route writes to the ledger or the env. It
binds WEB_HOST:WEB_PORT (LAN, no auth) - do not put it behind the public
tunnel without adding one.

Everything that touches SQLite runs in a worker thread; the event loop only
serialises. A slow query here must never stall Discord (see the
heartbeat-blocked history in discord_bot.py).
"""

import os
import json
import time
import asyncio
import logging
import collections
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.database import connect
from src.intraday_engine import ET, INTERVAL, barriers
from src import costs

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / 'web'
STARTED_AT = time.time()


class RingLog(logging.Handler):
    """Keeps the last N INFO+ records from the bot's own modules for the
    activity feed. Attached to the 'src' logger, so library noise
    (yfinance, discord) never reaches the page."""

    def __init__(self, size=150):
        super().__init__(level=logging.INFO)
        self.records = collections.deque(maxlen=size)

    def emit(self, record):
        try:
            self.records.appendleft({
                'ts': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec='seconds'),
                'level': record.levelname,
                'src': record.name.replace('src.', ''),
                'msg': record.getMessage()[:300],
            })
        except Exception:
            pass


class LLMStats:
    """Counters the analyzer bumps; exported as metrics and shown on the page."""

    def __init__(self):
        self.calls = collections.Counter()      # (backend, status) -> n
        self.last_latency_s = None
        self.last_call_at = None
        self.last_kind = None

    def record(self, backend, ok, seconds, kind):
        self.calls[(backend, 'ok' if ok else 'error')] += 1
        self.last_latency_s, self.last_call_at, self.last_kind = seconds, time.time(), kind


LLM = LLMStats()


class WebApp:
    def __init__(self, bot, host=None, port=None):
        self.bot = bot                          # TradingBot: engine, intraday, fast, budget_tracker, claude
        self.host = host or os.getenv('WEB_HOST', '0.0.0.0')
        self.port = int(port or os.getenv('WEB_PORT', 8090))
        self.push_s = float(os.getenv('WEB_PUSH_S', 5))
        self.ring = RingLog()
        logging.getLogger('src').addHandler(self.ring)
        self._subscribers: set[asyncio.Queue] = set()
        self._runner = None
        self._conn = None
        self.throttled_fetches = 0
        self._px_cache = {}                     # symbol -> (unix_ts, price)

    # --- lifecycle -----------------------------------------------------------

    async def start(self):
        from aiohttp import web
        app = web.Application()
        app.router.add_get('/', self._index)
        app.router.add_get('/api/state', self._state)
        app.router.add_get('/api/events', self._events)
        app.router.add_get('/metrics', self._metrics)
        app.router.add_static('/static/', WEB_DIR, show_index=False)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()
        asyncio.create_task(self._ticker(), name='web-ticker')
        logger.info(f"Live console on http://{self.host}:{self.port}/  (metrics at /metrics)")

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()

    def notify(self):
        """Called after a fast cycle: push fresh state to every open SSE client."""
        for q in list(self._subscribers):
            if q.qsize() < 2:
                q.put_nowait('cycle')

    async def _ticker(self):
        while True:
            await asyncio.sleep(self.push_s)
            for q in list(self._subscribers):
                if q.empty():
                    q.put_nowait('tick')

    # --- handlers ----------------------------------------------------------------

    async def _index(self, request):
        from aiohttp import web
        return web.FileResponse(WEB_DIR / 'index.html')

    async def _state(self, request):
        from aiohttp import web
        state = await asyncio.to_thread(self.snapshot)
        return web.json_response(state, dumps=lambda o: json.dumps(o, default=str))

    async def _events(self, request):
        from aiohttp import web
        resp = web.StreamResponse(headers={'Content-Type': 'text/event-stream',
                                           'Cache-Control': 'no-cache',
                                           'X-Accel-Buffering': 'no'})
        await resp.prepare(request)
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        try:
            while True:
                state = await asyncio.to_thread(self.snapshot)
                payload = json.dumps(state, default=str)
                await resp.write(f"event: state\ndata: {payload}\n\n".encode())
                await q.get()
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            self._subscribers.discard(q)
        return resp

    async def _metrics(self, request):
        from aiohttp import web
        text = await asyncio.to_thread(self.prometheus)
        return web.Response(text=text, content_type='text/plain', charset='utf-8')

    # --- data --------------------------------------------------------------------

    def _price(self, symbol):
        """Quote for the console, cached WEB_PRICE_TTL_S (30 s). engine.latest_price
        asks Finnhub per symbol (60/min free tier) and the bot's own exits need
        that quota more than a page refreshing every 5 s does."""
        ttl = float(os.getenv('WEB_PRICE_TTL_S', 30))
        hit = self._px_cache.get(symbol)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        px = self.bot.engine.latest_price(symbol)
        self._px_cache[symbol] = (time.time(), px)
        return px

    def _db(self):
        if self._conn is None:
            self._conn = connect(self.bot.budget_tracker.db_path)
        return self._conn

    def snapshot(self, now=None) -> dict:
        """One JSON document for the page. Worker thread only. `now` is for tests."""
        bot = self.bot
        bt = bot.budget_tracker
        now = now or datetime.now(timezone.utc)
        day_et = now.astimezone(ET).strftime('%Y-%m-%d')
        conn = self._db()

        pnl = bt.get_pnl(self._price)
        day = bt.get_day_state(day_et)
        trades = [dict(r) for r in bt.get_trades_since_open()]
        today = [t for t in trades if t.get('trade_date') == day_et]
        sells = [t for t in trades if t['side'] == 'SELL']
        wins = [t for t in sells if (t.get('realized_pnl') or 0) > 0]
        summary = (bot.fast.last_summary if bot.fast else {}) or {}

        # newest stored bar per interval -> data-health tiles
        bars = {}
        for iv in ('5m', '1m'):
            r = conn.execute("SELECT max(ts) FROM prices_intraday WHERE interval = ?", (iv,)).fetchone()
            bars[iv] = r[0]
        counts = {}
        for tbl in ('news', 'events', 'signals', 'prices_intraday'):
            try:
                counts[tbl] = conn.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
            except Exception:
                counts[tbl] = None
        news_24h = None
        try:
            news_24h = conn.execute(
                "SELECT count(*) FROM news WHERE published_at >= ?",
                ((now.replace(microsecond=0) - timedelta(hours=24)).isoformat(),)).fetchone()[0]
        except Exception:
            pass

        # latest scored signals: the top of the last scan, executed or not
        latest = conn.execute(
            "SELECT symbol, asset_class, probability, bar, above_bar, ref_price, bar_ts, executed_trade_id "
            "FROM signals WHERE bar_ts = (SELECT max(bar_ts) FROM signals) "
            "ORDER BY probability - bar DESC LIMIT 12").fetchall()

        models = {}
        if bot.intraday:
            for cls, m in (bot.intraday.metrics or {}).items():
                tp, sl, hz = barriers(cls)
                models[cls] = {
                    'precision': m.get('precision'), 'breakeven': m.get('breakeven'),
                    'ev': m.get('ev'), 'bar': bot.intraday.threshold(cls),
                    'take_profit': tp, 'stop_loss': sl, 'horizon_bars': hz,
                    'round_trip_cost': costs.round_trip_cost(cls),
                    'gated': (summary.get('gates') or {}).get(cls, (None, ''))[1] if summary.get('gates') else '',
                    'tradeable': (summary.get('gates') or {}).get(cls, (None, ''))[0] if summary.get('gates') else None,
                }

        def _age(ts):
            if not ts:
                return None
            d = datetime.fromisoformat(ts)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return round((now - d).total_seconds())

        return {
            'now': now.isoformat(timespec='seconds'),
            'day_et': day_et,
            'uptime_s': round(time.time() - STARTED_AT),
            'account': {
                'equity': pnl['equity'], 'cash': pnl['cash'], 'buying_power': pnl['buying_power'],
                'unsettled': pnl['unsettled'], 'starting_cash': pnl['starting_cash'],
                'all_time_net': pnl['all_time_net'], 'all_time_pct': pnl['all_time_pct'],
                'realized': pnl['realized'], 'unrealized': pnl['unrealized'],
                'fees_paid': pnl['fees_paid'], 'opened_at': bt.opened_at(),
            },
            'today': {
                'start_equity': day['start_equity'] if day else None,
                'loss_tripped': bool(day and day['loss_tripped_at']),
                'review_posted': bool(day and day['review_posted_at']),
                'n_trades': len(today),
                'realized': sum(float(t.get('realized_pnl') or 0) for t in today if t['side'] == 'SELL'),
                'wins': sum(1 for t in today if t['side'] == 'SELL' and (t.get('realized_pnl') or 0) > 0),
                'losses': sum(1 for t in today if t['side'] == 'SELL' and (t.get('realized_pnl') or 0) < 0),
            },
            'since_open': {
                'n_closed': len(sells), 'wins': len(wins),
                'win_rate': len(wins) / len(sells) if sells else None,
                'exit_reasons': dict(collections.Counter(t.get('exit_reason') or 'n/a' for t in sells)),
            },
            'positions': pnl['positions'],
            'stale': pnl.get('stale', []),
            'trades': trades[-40:][::-1],
            'equity_series': bt.equity_series(),
            'signals': [dict(r) for r in latest],
            'fast': {
                'enabled': bool(bot.fast), 'state': summary.get('state'), 'desc': summary.get('desc'),
                'note': summary.get('note'), 'ts': summary.get('ts'),
                'minutes_to_close': summary.get('minutes_to_close'),
                'rows': summary.get('rows'), 'bars_skipped': summary.get('bars_skipped', False),
                'candidates': summary.get('candidates', [])[:5],
                'skipped': summary.get('skipped', [])[:12],
                'entries': summary.get('entries', []), 'exits': summary.get('exits', []),
                'blocked_classes': summary.get('blocked_classes', []),
                'review_due': summary.get('review_due', False),
                'warming_up': bool(getattr(bot, 'warming_up', False)),
                'poll_s': float(os.getenv('FAST_POLL_SECONDS', 60)),
                'max_positions': bot.fast.max_positions if bot.fast else None,
                'open_delay_min': bot.fast.open_delay_min if bot.fast else None,
            },
            'models': models,
            'data': {
                'newest_5m': bars['5m'], 'newest_5m_age_s': _age(bars['5m']),
                'newest_1m': bars['1m'], 'newest_1m_age_s': _age(bars['1m']),
                'interval': INTERVAL, 'counts': counts, 'news_24h': news_24h,
                'universe': len(bot.intraday.symbols) if bot.intraday else 0,
                'throttled_fetches': self.throttled_fetches,
            },
            'llm': {
                'backend': bot.claude.backend_name, 'enabled': bot.claude.enabled,
                'calls': {f"{b}/{s}": n for (b, s), n in LLM.calls.items()},
                'last_latency_s': LLM.last_latency_s, 'last_call_at': LLM.last_call_at,
                'last_kind': LLM.last_kind,
            },
            'log': list(self.ring.records)[:80],
        }

    def prometheus(self) -> str:
        s = self.snapshot()
        a, t, so, d, f = s['account'], s['today'], s['since_open'], s['data'], s['fast']
        out = []

        def g(name, value, help_, labels=None):
            if value is None:
                return
            lab = '' if not labels else '{' + ','.join(f'{k}="{v}"' for k, v in labels.items()) + '}'
            out.append(f"# HELP {name} {help_}\n# TYPE {name} gauge\n{name}{lab} {float(value)}")

        g('tradingbot_equity_usd', a['equity'], 'Paper account equity')
        g('tradingbot_cash_usd', a['cash'], 'Settled cash')
        g('tradingbot_buying_power_usd', a['buying_power'], 'Cash available for entries')
        g('tradingbot_unsettled_usd', a['unsettled'], 'Sale proceeds not yet settled')
        g('tradingbot_all_time_net_usd', a['all_time_net'], 'Equity minus starting cash')
        g('tradingbot_all_time_return', a['all_time_pct'], 'All-time return as a fraction')
        g('tradingbot_unrealized_usd', a['unrealized'], 'Open-position P&L at last quote')
        g('tradingbot_fees_paid_usd', a['fees_paid'], 'Cumulative modelled fees')
        g('tradingbot_positions_open', len(s['positions']), 'Open positions')
        g('tradingbot_today_realized_usd', t['realized'], "Today's realised P&L (ET)")
        g('tradingbot_today_trades', t['n_trades'], "Today's executed trade rows")
        g('tradingbot_today_loss_tripped', int(t['loss_tripped']), 'Daily loss limit latched (1/0)')
        g('tradingbot_closed_trades_total', so['n_closed'], 'Closed trades since account open')
        g('tradingbot_win_rate', so['win_rate'], 'Win rate since account open')
        for reason, n in so['exit_reasons'].items():
            g('tradingbot_exits_total', n, 'Closed trades by exit reason', {'reason': reason})
        for cls, m in s['models'].items():
            g('tradingbot_model_precision', m['precision'], 'Walk-forward precision', {'asset_class': cls})
            g('tradingbot_model_breakeven', m['breakeven'], 'Breakeven precision', {'asset_class': cls})
            g('tradingbot_model_ev', m['ev'], 'Backtest EV per trade (fraction)', {'asset_class': cls})
            g('tradingbot_model_tradeable', None if m['tradeable'] is None else int(m['tradeable']),
              'Class passes the cost/EV gate (1/0)', {'asset_class': cls})
        g('tradingbot_bar_age_seconds', d['newest_5m_age_s'], 'Age of newest stored bar', {'interval': '5m'})
        g('tradingbot_bar_age_seconds', d['newest_1m_age_s'], 'Age of newest stored bar', {'interval': '1m'})
        for tbl, n in (d['counts'] or {}).items():
            g('tradingbot_table_rows', n, 'Row count', {'table': tbl})
        g('tradingbot_news_rows_24h', d['news_24h'], 'Headlines published in the last 24h')
        g('tradingbot_throttled_fetches_total', d['throttled_fetches'], 'Bar refreshes that returned 0 rows')
        g('tradingbot_fast_state', {'open': 1, 'premarket': 2, 'afterhours': 3, 'closed': 0}.get(f['state'], -1),
          'Market state seen by the fast loop (1 open, 2 pre, 3 after, 0 closed)')
        g('tradingbot_fast_last_cycle_timestamp', datetime.fromisoformat(str(f['ts'])).timestamp() if f['ts'] else None,
          'Unix time of the last fast cycle')
        g('tradingbot_warming_up', int(f['warming_up']), 'Startup fetch/train in progress (1/0)')
        for k, n in s['llm']['calls'].items():
            b, st = k.split('/')
            g('tradingbot_llm_calls_total', n, 'LLM analyst calls', {'backend': b, 'status': st})
        g('tradingbot_llm_last_latency_seconds', s['llm']['last_latency_s'], 'Latency of the last LLM call')
        g('tradingbot_process_uptime_seconds', s['uptime_s'], 'Bot process uptime')
        return "\n".join(out) + "\n"
