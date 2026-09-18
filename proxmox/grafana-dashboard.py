#!/usr/bin/env python3
"""
Build the "Trading Bot" Grafana dashboard (uid tradingbot) from the bot's
/metrics plus the LXC's node_exporter, and push it with the HTTP API.

Usage (on the Grafana host, or anywhere that can reach it):
  GRAFANA_URL=http://localhost:3001 GRAFANA_USER=admin GRAFANA_PASSWORD=... \
      python3 proxmox/grafana-dashboard.py            # push
  python3 proxmox/grafana-dashboard.py --print         # dump JSON only

Prometheus must scrape the bot: see the `tradingbot` job in the README
(192.168.1.247:8090/metrics). Datasource uid defaults to the Prometheus
default datasource.
"""
import os
import sys
import json
import urllib.request

DS = {'type': 'prometheus', 'uid': os.getenv('GRAFANA_DS_UID', 'PBFA97CFB590B2093')}
AMBER, GREEN, RED, GREY = '#ffb000', '#3ddc84', '#ff4d5e', '#8a8f98'

_id = [0]


def nid():
    _id[0] += 1
    return _id[0]


def target(expr, legend='', instant=False):
    t = {'datasource': DS, 'expr': expr, 'legendFormat': legend, 'refId': 'A'}
    if instant:
        t['instant'] = True
    return t


def _refids(panel):
    """Grafana needs distinct refIds per panel; assign A, B, C... by position."""
    for i, t in enumerate(panel.get('targets', [])):
        t['refId'] = chr(65 + i)
    return panel


def stat(title, expr, x, y, w=3, h=4, unit='short', decimals=None, thresholds=None, mappings=None,
         color_mode='value', graph=True, desc=''):
    return {
        'id': nid(), 'type': 'stat', 'title': title, 'description': desc,
        'gridPos': {'x': x, 'y': y, 'w': w, 'h': h}, 'datasource': DS,
        'targets': [target(expr)],
        'options': {'reduceOptions': {'calcs': ['lastNotNull'], 'fields': '', 'values': False},
                    'colorMode': color_mode, 'graphMode': 'area' if graph else 'none',
                    'textMode': 'value', 'justifyMode': 'center'},
        'fieldConfig': {'defaults': {'unit': unit, 'decimals': decimals,
                                     'thresholds': {'mode': 'absolute', 'steps': thresholds or [{'color': AMBER, 'value': None}]},
                                     'mappings': mappings or []}, 'overrides': []},
    }


def ts(title, targets, x, y, w=12, h=8, unit='short', decimals=None, fill=10, thresholds_line=None,
       overrides=None, desc='', stack=False, min_=None, max_=None):
    fc = {'defaults': {'unit': unit, 'decimals': decimals, 'min': min_, 'max': max_,
                       'color': {'mode': 'palette-classic'},
                       'custom': {'lineWidth': 2, 'fillOpacity': fill, 'gradientMode': 'opacity',
                                  'showPoints': 'never', 'spanNulls': True,
                                  'stacking': {'mode': 'normal' if stack else 'none'}}},
          'overrides': overrides or []}
    if thresholds_line:
        fc['defaults']['thresholds'] = {'mode': 'absolute', 'steps': thresholds_line}
        fc['defaults']['custom']['thresholdsStyle'] = {'mode': 'line+area' if len(thresholds_line) > 1 else 'line'}
    return {'id': nid(), 'type': 'timeseries', 'title': title, 'description': desc,
            'gridPos': {'x': x, 'y': y, 'w': w, 'h': h}, 'datasource': DS,
            'targets': targets, 'fieldConfig': fc,
            'options': {'legend': {'displayMode': 'list', 'placement': 'bottom'}, 'tooltip': {'mode': 'multi'}}}


def color_override(name, color):
    return {'matcher': {'id': 'byName', 'options': name}, 'properties': [{'id': 'color', 'value': {'mode': 'fixed', 'fixedColor': color}}]}


def row(title, y):
    return {'id': nid(), 'type': 'row', 'title': title, 'collapsed': False, 'gridPos': {'x': 0, 'y': y, 'w': 24, 'h': 1}, 'panels': []}


def build():
    p = []
    # ---- row 0: the money -------------------------------------------------------
    p.append(row('Paper account', 0))
    p.append(stat('Equity', 'tradingbot_equity_usd', 0, 1, w=4, unit='currencyUSD', decimals=2,
                  desc='Cash + positions at last quote. Started at $500.'))
    p.append(stat('All-time net', 'tradingbot_all_time_net_usd', 4, 1, w=4, unit='currencyUSD', decimals=2,
                  thresholds=[{'color': RED, 'value': None}, {'color': GREY, 'value': -0.005}, {'color': GREEN, 'value': 0.005}]))
    p.append(stat('Today realised', 'tradingbot_today_realized_usd', 8, 1, w=4, unit='currencyUSD', decimals=2,
                  thresholds=[{'color': RED, 'value': None}, {'color': GREY, 'value': -0.005}, {'color': GREEN, 'value': 0.005}]))
    p.append(stat('Win rate', 'tradingbot_win_rate', 12, 1, w=3, unit='percentunit', decimals=0, graph=False,
                  thresholds=[{'color': RED, 'value': None}, {'color': AMBER, 'value': 0.375}, {'color': GREEN, 'value': 0.45}],
                  desc='Since account open. Stock breakeven is 37.5%.'))
    p.append(stat('Open positions', 'tradingbot_positions_open', 15, 1, w=3, decimals=0, graph=False))
    p.append(stat('Closed trades', 'tradingbot_closed_trades_total', 18, 1, w=3, decimals=0, graph=False))
    p.append(stat('Loss limit', 'tradingbot_today_loss_tripped', 21, 1, w=3, graph=False, color_mode='background',
                  thresholds=[{'color': GREEN, 'value': None}, {'color': RED, 'value': 1}],
                  mappings=[{'type': 'value', 'options': {'0': {'text': 'clear'}, '1': {'text': 'TRIPPED'}}}],
                  desc='1 = no new entries for the rest of the ET day'))

    p.append(ts('Equity', [target('tradingbot_equity_usd', 'equity'),
                           target('tradingbot_cash_usd', 'cash')], 0, 5, w=16, h=9, unit='currencyUSD', decimals=2,
                thresholds_line=[{'color': GREY, 'value': 500}],
                overrides=[color_override('equity', AMBER), color_override('cash', GREY)],
                desc='Dashed line = $500 starting cash.'))
    p.append(ts('Unrealised P&L', [target('tradingbot_unrealized_usd', 'unrealised')], 16, 5, w=8, h=9,
                unit='currencyUSD', decimals=2, thresholds_line=[{'color': GREY, 'value': 0}],
                overrides=[color_override('unrealised', AMBER)]))

    # ---- row 1: the model -------------------------------------------------------
    p.append(row('Model', 14))
    p.append(ts('Precision vs breakeven (walk-forward)',
                [target('tradingbot_model_precision', 'precision {{asset_class}}'),
                 target('tradingbot_model_breakeven', 'breakeven {{asset_class}}')], 0, 15, w=12, h=8,
                unit='percentunit', decimals=1, fill=0, min_=0, max_=0.7,
                desc='Precision must clear breakeven = SL/(TP+SL) after costs before the class is allowed to trade.'))
    p.append(ts('Backtest EV per trade', [target('tradingbot_model_ev', '{{asset_class}}')], 12, 15, w=6, h=8,
                unit='percentunit', decimals=3, thresholds_line=[{'color': GREY, 'value': 0}]))
    p.append({'id': nid(), 'type': 'piechart', 'title': 'Exits by reason', 'datasource': DS,
              'gridPos': {'x': 18, 'y': 15, 'w': 6, 'h': 8},
              'targets': [target('tradingbot_exits_total', '{{reason}}', instant=True)],
              'options': {'reduceOptions': {'calcs': ['lastNotNull'], 'fields': '', 'values': False},
                          'pieType': 'donut', 'legend': {'displayMode': 'list', 'placement': 'right', 'values': ['value']},
                          'displayLabels': ['percent']},
              'fieldConfig': {'defaults': {'unit': 'short'},
                              'overrides': [color_override('sl', RED), color_override('tp', GREEN),
                                            color_override('eod', AMBER), color_override('timeout', GREY)]},
              'description': 'sl = stop-loss, tp = take-profit, eod = flattened before the bell, timeout = held the full horizon'})

    # ---- row 2: data + AI -------------------------------------------------------
    p.append(row('Data feed · AI analyst', 23))
    p.append(ts('Newest stored bar age', [target('tradingbot_bar_age_seconds', '{{interval}}')], 0, 24, w=8, h=7,
                unit='s', fill=0, thresholds_line=[{'color': GREEN, 'value': None}, {'color': RED, 'value': 900}],
                desc='During the session a 5m bar older than 15 min means Yahoo is throttling or the loop is stuck.'))
    p.append(ts('Throttled bar refreshes (0-row fetches)', [target('increase(tradingbot_throttled_fetches_total[1h])', 'per hour')],
                8, 24, w=8, h=7, decimals=0, overrides=[color_override('per hour', RED)]))
    p.append(ts('Collector tables', [target('tradingbot_table_rows{table="news"}', 'news'),
                                     target('tradingbot_table_rows{table="signals"}', 'signals'),
                                     target('tradingbot_table_rows{table="events"}', 'earnings events')],
                16, 24, w=8, h=7, decimals=0, fill=0))
    p.append(ts('LLM analyst calls', [target('increase(tradingbot_llm_calls_total[1h])', '{{backend}} {{status}}')],
                0, 31, w=8, h=6, decimals=0, stack=True,
                overrides=[color_override('local ok', AMBER), color_override('claude ok', GREEN),
                           color_override('local error', RED), color_override('claude error', RED)]))
    p.append(ts('LLM last-call latency', [target('tradingbot_llm_last_latency_seconds', 'latency')], 8, 31, w=8, h=6,
                unit='s', decimals=1, fill=0, overrides=[color_override('latency', AMBER)]))
    p.append(stat('Headlines stored (24h)', 'tradingbot_news_rows_24h', 16, 31, w=4, h=6, decimals=0))
    p.append(stat('Market state', 'tradingbot_fast_state', 20, 31, w=4, h=6, graph=False, color_mode='background',
                  thresholds=[{'color': GREY, 'value': None}, {'color': GREEN, 'value': 1}, {'color': AMBER, 'value': 2}],
                  mappings=[{'type': 'value', 'options': {'0': {'text': 'closed'}, '1': {'text': 'OPEN'},
                                                           '2': {'text': 'pre-market'}, '3': {'text': 'after hours'},
                                                           '-1': {'text': 'no cycle yet'}}}]))

    # ---- row 3: the process (carried over from the 2026-09-10 dashboard) ----------
    p.append(row('Process · LXC 200', 37))
    ok_bad = [{'color': RED, 'value': None}, {'color': GREEN, 'value': 1}]
    p.append(stat('Container reachable', 'up{job="node-tradingbot"}', 0, 38, w=4, h=4, graph=False, color_mode='background',
                  thresholds=ok_bad, mappings=[{'type': 'value', 'options': {'0': {'text': 'DOWN'}, '1': {'text': 'up'}}}]))
    p.append(stat('trading-bot.service', 'node_systemd_unit_state{name="trading-bot.service",state="active"}', 4, 38, w=4, h=4,
                  graph=False, color_mode='background', thresholds=ok_bad,
                  mappings=[{'type': 'value', 'options': {'0': {'text': 'STOPPED'}, '1': {'text': 'active'}}}]))
    p.append(stat('Restarts (24h)', 'increase(node_systemd_service_restart_total{name="trading-bot.service"}[24h])', 8, 38, w=4, h=4,
                  decimals=0, graph=False, thresholds=[{'color': GREEN, 'value': None}, {'color': AMBER, 'value': 1}, {'color': RED, 'value': 3}]))
    p.append(stat('Up since restart', 'tradingbot_process_uptime_seconds', 12, 38, w=4, h=4, unit='s', graph=False))
    p.append(stat('Warming up', 'tradingbot_warming_up', 16, 38, w=4, h=4, graph=False, color_mode='background',
                  thresholds=[{'color': GREEN, 'value': None}, {'color': AMBER, 'value': 1}],
                  mappings=[{'type': 'value', 'options': {'0': {'text': 'ready'}, '1': {'text': 'fetching/training'}}}]))
    p.append(stat('Last fast cycle', 'time() - tradingbot_fast_last_cycle_timestamp', 20, 38, w=4, h=4, unit='s', graph=False,
                  thresholds=[{'color': GREEN, 'value': None}, {'color': AMBER, 'value': 180}, {'color': RED, 'value': 600}],
                  desc='Seconds since the fast loop last completed a cycle. It runs every 60 s while the bot is up.'))
    p.append(ts('Memory used % of LXC 200', [target('(1 - (node_memory_MemAvailable_bytes{job="node-tradingbot"} / node_memory_MemTotal_bytes{job="node-tradingbot"})) * 100', 'used %')],
                0, 42, w=8, h=7, unit='percent', decimals=0, min_=0, max_=100,
                thresholds_line=[{'color': GREEN, 'value': None}, {'color': RED, 'value': 88}],
                desc='The service cap is 3 GB inside a 4 GB container. OOM-killed twice on 2026-09-18 at the old 1.2 GB cap.'))
    p.append(ts('CPU %', [target('100 - (avg(rate(node_cpu_seconds_total{job="node-tradingbot",mode="idle"}[5m])) * 100)', 'cpu %')],
                8, 42, w=8, h=7, unit='percent', decimals=0, min_=0, max_=100))
    p.append(ts('Restart counter (flat = healthy)', [target('node_systemd_service_restart_total{name="trading-bot.service"}', 'restarts')],
                16, 42, w=8, h=7, decimals=0, fill=0, overrides=[color_override('restarts', RED)]))

    p = [_refids(x) for x in p]
    return {
        'uid': 'tradingbot', 'title': 'Trading Bot', 'tags': ['trading', 'paper'],
        'timezone': 'America/New_York', 'refresh': '30s', 'schemaVersion': 39, 'editable': True,
        'time': {'from': 'now-7d', 'to': 'now'},
        'links': [{'title': 'Live console (TAPE)', 'type': 'link', 'url': 'http://192.168.1.247:8090/', 'targetBlank': True, 'icon': 'external link'}],
        'panels': p,
    }


def main():
    dash = build()
    if '--print' in sys.argv:
        print(json.dumps(dash, indent=1))
        return
    url = os.getenv('GRAFANA_URL', 'http://localhost:3001').rstrip('/')
    user, pw = os.getenv('GRAFANA_USER', 'admin'), os.environ['GRAFANA_PASSWORD']
    body = json.dumps({'dashboard': dash, 'overwrite': True, 'message': 'proxmox/grafana-dashboard.py'}).encode()
    req = urllib.request.Request(f"{url}/api/dashboards/db", data=body, method='POST',
                                 headers={'Content-Type': 'application/json'})
    import base64
    req.add_header('Authorization', 'Basic ' + base64.b64encode(f"{user}:{pw}".encode()).decode())
    with urllib.request.urlopen(req, timeout=20) as r:
        print(r.read().decode())


if __name__ == '__main__':
    main()
