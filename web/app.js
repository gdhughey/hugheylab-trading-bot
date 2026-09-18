/* HUGHEYLAB / TAPE — live console. Vanilla JS, one SSE stream, uPlot for the equity line.
   Every number rendered here came out of /api/state; nothing is computed client-side
   except formatting and the bar/margin widths. */
(() => {
  const $ = (id) => document.getElementById(id);
  const ET = 'America/New_York';
  const fmtUsd = (v, sign = false) => v == null ? '—' :
    (sign && v > 0 ? '+' : v < 0 ? '−' : '') + '$' + Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtPct = (v, d = 2, sign = false) => v == null ? '—' : (sign && v > 0 ? '+' : '') + (v * 100).toFixed(d) + '%';
  const qty = (v) => v == null ? '—' : (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(4));
  const timeET = (iso, withSec = true) => {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleTimeString('en-US', { timeZone: ET, hour12: false, hour: '2-digit', minute: '2-digit', ...(withSec ? { second: '2-digit' } : {}) });
  };
  const ago = (s) => s == null ? '—' : s < 90 ? `${Math.round(s)}s` : s < 5400 ? `${Math.round(s / 60)}m` : s < 172800 ? `${(s / 3600).toFixed(1)}h` : `${Math.round(s / 86400)}d`;
  const sgn = (el, v) => { el.classList.remove('up', 'down', 'flat'); el.classList.add(v > 0 ? 'up' : v < 0 ? 'down' : 'flat'); };
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // ---- clock ----
  setInterval(() => { $('clock-et').textContent = new Date().toLocaleTimeString('en-US', { timeZone: ET, hour12: false }); }, 250);

  // ---- equity chart ----
  let plot, lastEquity = null, lastCycleTs = null;
  const chartEl = $('equity-chart');
  const amber = '#ffb000';
  function drawChart(series, starting) {
    const xs = series.map(p => Math.floor(new Date(p.date + 'T21:00:00Z').getTime() / 1000));
    const ys = series.map(p => p.equity);
    const data = [xs, ys, xs.map(() => starting)];
    const opts = {
      width: chartEl.clientWidth, height: chartEl.clientHeight,
      cursor: { drag: { x: false, y: false }, points: { size: 7 } },
      scales: { x: { time: true }, y: { range: (u, min, max) => { const pad = Math.max((max - min) * 0.25, 2); return [min - pad, max + pad]; } } },
      axes: [
        { stroke: '#6b6a5e', grid: { stroke: '#1c2230', width: 1 }, ticks: { stroke: '#1c2230' }, font: '11px "IBM Plex Mono"',
          values: (u, ts) => ts.map(t => new Date(t * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })) },
        { stroke: '#6b6a5e', grid: { stroke: '#1c2230', width: 1 }, ticks: { stroke: '#1c2230' }, font: '11px "IBM Plex Mono"', size: 56,
          values: (u, vs) => vs.map(v => '$' + v.toFixed(0)) },
      ],
      series: [
        {},
        { stroke: amber, width: 2, fill: (u) => { const g = u.ctx.createLinearGradient(0, 0, 0, u.bbox.height); g.addColorStop(0, 'rgba(255,176,0,.28)'); g.addColorStop(1, 'rgba(255,176,0,0)'); return g; }, points: { show: true, size: 6, fill: '#0e1116', stroke: amber } },
        { stroke: '#3a4358', width: 1, dash: [4, 4], points: { show: false } },
      ],
    };
    if (!plot) plot = new uPlot(opts, data, chartEl);
    else { plot.setSize({ width: chartEl.clientWidth, height: chartEl.clientHeight }); plot.setData(data); }
  }
  addEventListener('resize', () => plot && plot.setSize({ width: chartEl.clientWidth, height: chartEl.clientHeight }));

  // ---- render ----
  function render(s) {
    const a = s.account, t = s.today, f = s.fast, d = s.data;

    // hero
    const eqEl = $('equity');
    eqEl.textContent = fmtUsd(a.equity);
    if (lastEquity != null && a.equity !== lastEquity) {
      const h = eqEl.parentElement; h.classList.remove('tick-up', 'tick-down'); void h.offsetWidth;
      h.classList.add(a.equity > lastEquity ? 'tick-up' : 'tick-down');
    }
    lastEquity = a.equity;
    $('alltime').textContent = `${fmtUsd(a.all_time_net, true)} (${fmtPct(a.all_time_pct, 2, true)})`; sgn($('alltime'), a.all_time_net);
    $('starting').textContent = fmtUsd(a.starting_cash);
    $('today-real').textContent = fmtUsd(t.realized, true); sgn($('today-real'), t.realized);
    $('unreal').textContent = fmtUsd(a.unrealized, true); sgn($('unreal'), a.unrealized);
    $('cash').textContent = fmtUsd(a.cash); $('bp').textContent = fmtUsd(a.buying_power); $('unsettled').textContent = fmtUsd(a.unsettled);
    const days = s.equity_series.length;
    $('day-count').textContent = `day ${days} of the paper run · ${s.since_open.n_closed} closed · win rate ${s.since_open.win_rate == null ? '—' : fmtPct(s.since_open.win_rate, 0)} · today ${t.n_trades} fills, ${t.wins}W ${t.losses}L`;
    $('loss-flag').classList.toggle('hidden', !t.loss_tripped);
    $('review-flag').classList.toggle('hidden', !t.review_posted);
    if (s.equity_series && s.equity_series.length) {
      // one point per date (day 0 is synthetic and shares the open date's row); uPlot needs ascending x
      const byDate = new Map(); for (const p of s.equity_series) byDate.set(p.date, p);
      const series = [...byDate.values()].sort((a, b) => a.date < b.date ? -1 : 1);
      // live point for today so the line ends at the current equity
      if (series[series.length - 1].date !== s.day_et) series.push({ date: s.day_et, equity: a.equity });
      else series[series.length - 1] = { date: s.day_et, equity: a.equity };
      try { if (window.uPlot) drawChart(series, a.starting_cash); } catch (e) { console.error('chart', e); }
    }

    // header state
    const st = f.state || (f.warming_up ? 'warming' : 'closed');
    const pill = $('market-state'); pill.textContent = f.state ? (f.desc || st).toUpperCase() : (f.warming_up ? 'WARMING UP' : '—'); pill.className = 'pill ' + (f.state || '');
    lastCycleTs = f.ts ? new Date(f.ts) : null;

    // doing now
    const pulse = $('pulse'); pulse.className = 'pulse ' + (f.warming_up ? 'warming' : st === 'open' ? 'open' : 'closed');
    let head = f.warming_up ? 'Warming up — fetching & training' :
      !f.enabled ? 'Fast mode off — hourly daily loop only' :
      st === 'open' ? (f.entries.length ? `Entered ${f.entries.map(e => e.symbol).join(', ')}` :
        f.exits.length ? `Exited ${f.exits.map(e => e.symbol).join(', ')}` : 'Scanning · managing exits') :
      `Market ${f.desc || st}${s.positions.length ? ' · holding ' + s.positions.length : ''}`;
    $('now-headline').textContent = head;
    $('now-note').textContent = f.note || (f.candidates.length ? `${f.candidates.length} candidate(s) over the bar` : (f.blocked_classes.length ? `blocked: ${f.blocked_classes.join(', ')}` : ''));
    $('now-ts').textContent = f.ts ? 'cycle ' + timeET(f.ts) + ' ET' : '';
    $('now-rows').textContent = f.bars_skipped ? 'held' : (f.rows ?? '—');
    $('now-close').textContent = f.minutes_to_close == null ? '—' : (f.minutes_to_close > 0 ? `${Math.round(f.minutes_to_close)}m` : 'closed');
    $('now-slots').textContent = f.max_positions == null ? '—' : `${s.positions.length}/${f.max_positions}`;
    $('now-poll').textContent = `${f.poll_s}s`;
    const sk = $('skipped'); sk.innerHTML = '';
    if (!f.skipped.length) sk.innerHTML = '<li class="dim">nothing skipped</li>';
    for (const [sym, why] of f.skipped) sk.insertAdjacentHTML('beforeend', `<li><b>${esc(sym)}</b><span>${esc(why)}</span></li>`);

    // positions
    $('pos-count').textContent = s.positions.length;
    const pb = $('positions').tBodies[0]; pb.innerHTML = '';
    if (!s.positions.length) pb.innerHTML = '<tr><td colspan="7" class="empty">flat</td></tr>';
    for (const p of s.positions) {
      const m = s.models[p.symbol.endsWith('-USD') ? 'crypto' : 'stock'] || {};
      const ref = p.entry_ref || p.avg_price;
      const sl = m.stop_loss != null ? ref * (1 - m.stop_loss) : null, tp = m.take_profit != null ? ref * (1 + m.take_profit) : null;
      const pnlCls = p.pnl > 0 ? 'up' : p.pnl < 0 ? 'down' : 'flat';
      pb.insertAdjacentHTML('beforeend', `<tr>
        <td class="sym">${esc(p.symbol)}</td><td class="r">${qty(p.shares)}</td><td class="r">${fmtUsd(p.avg_price)}</td>
        <td class="r">${p.price == null ? '<span class="dim">stale</span>' : fmtUsd(p.price)}</td>
        <td class="r signed ${pnlCls}">${fmtUsd(p.pnl, true)}</td><td class="r signed ${pnlCls}">${fmtPct(p.pnl_pct, 2, true)}</td>
        <td class="barriers"><span class="sl">${sl ? fmtUsd(sl) : '—'}</span> ◂ ref ${fmtUsd(ref)} ▸ <span class="tp">${tp ? fmtUsd(tp) : '—'}</span></td></tr>`);
    }

    // models
    const ml = $('models'); ml.innerHTML = '';
    const names = Object.keys(s.models);
    if (!names.length) ml.innerHTML = '<div class="dim">no trained intraday model</div>';
    for (const cls of names) {
      const m = s.models[cls];
      const prec = m.precision ?? 0, be = m.breakeven ?? 0;
      const lo = 0.15, hi = 0.65, pct = (v) => Math.max(0, Math.min(100, (v - lo) / (hi - lo) * 100));
      const ok = m.tradeable === true, no = m.tradeable === false;
      ml.insertAdjacentHTML('beforeend', `<div class="model">
        <span class="name">${esc(cls)}</span>
        <span class="verdict ${ok ? 'ok' : no ? 'no' : 'na'}">${ok ? 'TRADING' : no ? 'GATED' : 'N/A'}</span>
        <div class="gauge"><div class="fill" style="width:${pct(prec)}%"></div><div class="mark" style="left:${pct(be)}%"></div></div>
        <div class="facts">
          <span>precision <b>${fmtPct(prec, 1)}</b></span><span>breakeven <b>${fmtPct(be, 1)}</b></span>
          <span>EV <b>${m.ev == null ? '—' : fmtPct(m.ev, 3, true)}</b>/trade</span>
          <span>bar <b>${m.bar == null ? '—' : m.bar.toFixed(3)}</b></span>
          <span>TP <b>${fmtPct(m.take_profit, 2, true)}</b> SL <b>−${fmtPct(m.stop_loss, 2)}</b> · ${m.horizon_bars} bars</span>
          <span>round trip <b>${fmtPct(m.round_trip_cost, 2)}</b></span>
        </div>
        <div class="gate">${esc(m.gated || '')}</div></div>`);
    }

    // signals + tape
    const sb = $('signals').tBodies[0]; sb.innerHTML = '';
    if (!s.signals.length) sb.innerHTML = '<tr><td colspan="6" class="empty">no scan yet</td></tr>';
    else $('scan-ts').textContent = 'bar ' + timeET(s.signals[0].bar_ts, false) + ' ET';
    const tape = [];
    for (const g of s.signals) {
      const margin = g.probability - g.bar, w = Math.min(100, Math.abs(margin) / 0.25 * 100);
      sb.insertAdjacentHTML('beforeend', `<tr>
        <td class="sym">${esc(g.symbol)}${g.executed_trade_id ? ' <span class="dim">●</span>' : ''}</td><td class="dim">${esc(g.asset_class)}</td>
        <td class="r">${g.probability.toFixed(3)}</td><td class="r dim">${g.bar.toFixed(3)}</td>
        <td><div class="bar"><i class="${margin < 0 ? 'neg' : ''}" style="width:${w}%"></i></div></td>
        <td class="r">${fmtUsd(g.ref_price)}</td></tr>`);
      tape.push(`<span class="tape-item ${g.above_bar ? 'hit' : ''} ${g.executed_trade_id ? 'exec' : ''}"><b>${esc(g.symbol)}</b><span class="p">${g.probability.toFixed(3)}</span> <span class="dim">/ ${g.bar.toFixed(3)}</span> ${fmtUsd(g.ref_price)}</span>`);
    }
    if (tape.length) $('tape').innerHTML = tape.join('') + tape.join('');   // doubled for the seamless loop

    // trades
    const tb = $('trades').tBodies[0]; tb.innerHTML = '';
    const er = s.since_open.exit_reasons || {};
    $('trade-stats').textContent = `${s.since_open.n_closed} closed · ${Object.entries(er).map(([k, v]) => `${k} ×${v}`).join(' · ') || '—'}`;
    if (!s.trades.length) tb.innerHTML = '<tr><td colspan="7" class="empty">no trades yet</td></tr>';
    for (const tr of s.trades) {
      const net = tr.side === 'SELL' ? tr.realized_pnl : null;
      const why = tr.side === 'SELL' ? (tr.exit_reason || '') : (tr.entry_probability != null ? `p=${tr.entry_probability.toFixed(3)}` : '');
      tb.insertAdjacentHTML('beforeend', `<tr>
        <td class="dim">${esc(tr.trade_date?.slice(5) ?? '')} ${timeET(tr.created_at)}</td>
        <td class="side-${tr.side}">${tr.side}</td><td class="sym">${esc(tr.symbol)}</td>
        <td class="r">${qty(tr.shares)}</td><td class="r">${fmtUsd(tr.price)}</td>
        <td class="r signed ${net > 0 ? 'up' : net < 0 ? 'down' : 'flat'}">${net == null ? '<span class="dim">—</span>' : fmtUsd(net, true)}</td>
        <td class="why ${esc(tr.exit_reason || '')}">${esc(why)}</td></tr>`);
    }

    // data / ai
    const ageCls = (sec, warn, bad) => sec == null ? '' : sec > bad ? 'bad' : sec > warn ? 'warn' : 'good';
    const openish = st === 'open';
    $('age5').textContent = ago(d.newest_5m_age_s); $('age5').className = 'tile-num ' + (openish ? ageCls(d.newest_5m_age_s, 900, 2700) : '');
    $('age1').textContent = ago(d.newest_1m_age_s); $('age1').className = 'tile-num ' + (openish ? ageCls(d.newest_1m_age_s, 5400, 10800) : '');
    $('universe').textContent = d.universe;
    $('throttled').textContent = d.throttled_fetches; $('throttled').className = 'tile-num ' + (d.throttled_fetches > 5 ? 'warn' : '');
    $('news24').textContent = d.news_24h == null ? '—' : d.news_24h.toLocaleString();
    $('sigcount').textContent = d.counts?.signals == null ? '—' : d.counts.signals.toLocaleString();
    const calls = Object.entries(s.llm.calls || {}).map(([k, v]) => `${k} ×${v}`).join(' · ');
    $('llm').textContent = `${s.llm.backend}${s.llm.enabled ? '' : ' (disabled)'}${calls ? ' · ' + calls : ''}`;
    $('llm-last').textContent = s.llm.last_call_at ? `${s.llm.last_kind} · ${s.llm.last_latency_s.toFixed(1)}s · ${ago(Date.now() / 1000 - s.llm.last_call_at)} ago` : 'no calls yet';

    // log
    const lg = $('log'); lg.innerHTML = '';
    for (const r of s.log) lg.insertAdjacentHTML('beforeend', `<li class="${r.level}"><span class="t">${timeET(r.ts)}</span><span class="s">${esc(r.src)}</span><span class="m">${esc(r.msg)}</span></li>`);

    $('uptime').textContent = ago(s.uptime_s);
  }

  // cycle-age ticker in the header
  setInterval(() => {
    if (!lastCycleTs) return;
    const sec = (Date.now() - lastCycleTs.getTime()) / 1000;
    $('cycle-age').textContent = `cycle ${ago(sec)} ago`;
  }, 1000);

  // ---- SSE with fallback polling ----
  let es, pollTimer;
  function connect() {
    document.body.dataset.state = 'connecting'; $('link-text').textContent = 'connecting';
    try { es = new EventSource('/api/events'); } catch { return poll(); }
    es.addEventListener('state', (e) => { document.body.dataset.state = 'live'; $('link-text').textContent = 'live'; render(JSON.parse(e.data)); });
    es.onerror = () => { document.body.dataset.state = 'lost'; $('link-text').textContent = 'reconnecting'; es.close(); setTimeout(connect, 4000); };
  }
  async function poll() {
    try { const r = await fetch('/api/state'); render(await r.json()); document.body.dataset.state = 'live'; }
    catch { document.body.dataset.state = 'lost'; }
    pollTimer = setTimeout(poll, 5000);
  }
  connect();
})();
