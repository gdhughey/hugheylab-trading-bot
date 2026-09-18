#!/usr/bin/env python3
"""
LLM analyst: daily brief, risk assessment, weekly review.

Two backends behind one interface:

  local   an OpenAI-compatible server (llama.cpp in LXC 202, Qwen3-8B on the
          W6600) at LLM_BASE_URL. Free, on the LAN, ~40 tok/s. Used for every
          routine call.
  claude  the Anthropic API, used only for the weekly review when a key is
          set - the one job where multi-step reasoning is worth paying for.
          Falls back to local if the call fails (e.g. no credit, 2026-09-17).

The rule that makes an 8B model safe here: THE MODEL NEVER COMPUTES. Python
builds a context of facts (scorecard, positions, calendar, headlines) and the
model writes prose about exactly that. Every reply is checked for numbers that
do not appear in its prompt; unsourced figures are flagged in the output and
logged, because "confidently made up" is the failure mode that matters. The
original daily brief asked Claude about "major economic events happening
today" with no data at all - it could only invent an answer.
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _stats(backend, ok, seconds, kind):
    """Feed the live console's LLM counters; never let telemetry break a call."""
    try:
        from src.webapp import LLM
        LLM.record(backend, ok, seconds, kind)
    except Exception:
        pass

# Local (OpenAI-compatible) endpoint. Set LLM_BASE_URL to enable.
LLM_BASE_URL = os.getenv('LLM_BASE_URL', '').rstrip('/')
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen3-8b')
LLM_TIMEOUT_S = float(os.getenv('LLM_TIMEOUT_S', 90))
# Anthropic. Only the weekly review goes here, and only when a key is set.
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
MODEL = CLAUDE_MODEL   # back-compat name

SYSTEM = (
    "You are the desk analyst for a small PAPER-trading bot (no real money). "
    "You are given a block of FACTS computed by the bot's own code. Write for a "
    "retail trader in plain English. Rules: use ONLY the facts given; never "
    "introduce a number, ticker, company, date or event that is not in the facts; "
    "if the facts don't cover something, say it isn't in the data rather than "
    "guessing; do not recommend buying or selling anything - the bot's model "
    "decides that, you explain. No headings, no bullet lists, no markdown."
)

# --- number grounding --------------------------------------------------------

_NUM = re.compile(r'(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?%?')


def _numbers(text: str) -> list[float]:
    """Numeric values in `text` as magnitudes ("a loss of $8.29" is a fair
    reading of "-$8.29"). Duplicates kept; callers set() what they need."""
    out = []
    for tok in _NUM.findall(text or ''):
        t = tok.replace('$', '').replace(',', '').rstrip('%').lstrip('+')
        try:
            out.append(abs(float(t)))
        except ValueError:
            continue
    return out


def _decimals(tok: str) -> int:
    return len(tok.split('.')[1]) if '.' in tok else 0


def unsourced_numbers(prompt: str, answer: str) -> list[str]:
    """Numbers in the answer that appear nowhere in the prompt.

    A number counts as sourced if some prompt figure ROUNDS to it at the
    precision the answer used ("0.471" -> "0.47" is a paraphrase, not an
    invention). Small integers 0-3 are ignored (list ordinals, "two trades"
    spelled as a digit). Everything else must be traceable to the facts.
    """
    have = _numbers(prompt)
    bad = set()
    for tok in _NUM.findall(answer or ''):
        t = tok.replace('$', '').replace(',', '').rstrip('%').lstrip('+')
        try:
            v = abs(float(t))
        except ValueError:
            continue
        if v == int(v) and v <= 3:
            continue
        d = _decimals(t)
        if any(round(h, d) == v for h in have):
            continue
        bad.add(f"{v:g}")
    return sorted(bad)


# --- context builders (Python computes, the model narrates) -------------------

def _fmt_usd(x):
    if x is None:
        return "n/a"
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _pct(x, digits=1):
    return f"{x * 100:.{digits}f}%" if x is not None else "n/a"


def render_scorecard(sc: dict) -> str:
    """Compact facts block from scorecard.build_scorecard()'s dict."""
    h, a, t, so = sc['headline'], sc['account'], sc['today'], sc['since_open']
    lines = [
        f"Account: equity {_fmt_usd(a['equity'])} (started {_fmt_usd(a['starting_cash'])}), "
        f"cash {_fmt_usd(a['cash'])}, buying power {_fmt_usd(a['buying_power'])}, "
        f"unsettled {_fmt_usd(a['unsettled'])}, fees paid so far {_fmt_usd(a['fees_paid'])}.",
        f"All-time net P&L: {_fmt_usd(h['all_time_net'])} ({_pct(h['all_time_pct'], 2)}) over "
        f"{h['n_closed']} closed trades, 95% CI ±{_fmt_usd(h['ci_dollars'])}, "
        f"day {so['days_running']} of the paper run.",
        f"Since open: win rate {_pct(so['win_rate'])} (CI {_pct(so['win_lo'])}–{_pct(so['win_hi'])}), "
        f"mean net per trade {_fmt_usd(so['mean_net'])}, "
        f"max drawdown {so['max_drawdown_pct']:.2f}%"
        + (f", profit factor {so['profit_factor']:.2f}" if so.get('profit_factor') else "") + ".",
        f"Today: {t['n_trades']} trades, {t['wins']} wins, {t['losses']} losses, "
        f"realised {_fmt_usd(t['realized'])}, unrealised {_fmt_usd(t['unrealized'])}"
        + (", DAILY LOSS LIMIT TRIPPED - no new entries today" if t['loss_tripped'] else "") + ".",
    ]
    for c in sc.get('closed_today') or []:
        lines.append(f"Closed today: {c['symbol']} {c['shares']:g} sh at {_fmt_usd(c['price'])}, "
                     f"net {_fmt_usd(c['net'])}, exit reason {c['exit_reason'] or 'n/a'}.")
    for cls, v in (sc.get('classes') or {}).items():
        bt = v.get('bt_precision')
        lines.append(f"Model {cls}: verdict {v.get('verdict')}"
                     + (f", backtest precision {_pct(bt)}" if bt is not None else "")
                     + (", blocked by the cost gate" if v.get('gated') else "")
                     + f", {v.get('exec_n', 0)} executed trades.")
    spy = sc.get('spy')
    if spy:
        lines.append(f"SPY buy-and-hold over the same period: {_pct(spy['pct'], 2)} "
                     f"(would be {_fmt_usd(spy['value'])}).")
    return "\n".join(lines)


def render_positions(positions: list[dict]) -> str:
    if not positions:
        return "Open positions: none."
    lines = ["Open positions:"]
    for p in positions:
        px = p.get('price')
        lines.append(
            f"- {p['symbol']}: {p['shares']:g} sh, avg cost {_fmt_usd(p['avg_price'])}, "
            + (f"now {_fmt_usd(px)}, P&L {_fmt_usd(p.get('pnl'))} ({_pct(p.get('pnl_pct'), 2)}), "
               if px else "price unavailable (stale), ")
            + f"cost basis {_fmt_usd(p.get('cost_basis') or p['shares'] * p['avg_price'])}.")
    return "\n".join(lines)


def load_calendar_and_news(conn, symbols, now=None, days_ahead=3, news_hours=24, max_headlines=8):
    """Earnings in the next `days_ahead` days and recent headline counts for
    `symbols`, from the collector's tables. Empty strings if the tables are
    missing (collector never ran) - the brief still renders."""
    now = now or datetime.now(timezone.utc)
    syms = [s for s in symbols if s and not s.upper().endswith(('-USD', '-USDT'))]
    out = []
    if not syms:
        return "Calendar and news: no stock symbols to look up."
    q = ",".join("?" * len(syms))
    try:
        rows = conn.execute(
            f"SELECT symbol, event_date, hour FROM events WHERE kind='earnings' "
            f"AND symbol IN ({q}) AND event_date >= ? AND event_date <= ? ORDER BY event_date",
            (*syms, now.date().isoformat(), (now.date() + timedelta(days=days_ahead)).isoformat())
        ).fetchall()
        if rows:
            out.append("Earnings within the next " + str(days_ahead) + " days: "
                       + ", ".join(f"{r[0]} on {r[1]}" + (f" ({r[2]})" if r[2] else "") for r in rows) + ".")
        else:
            out.append(f"Earnings within the next {days_ahead} days: none for these symbols.")
    except Exception:
        out.append("Earnings calendar: not collected yet.")
    try:
        since = (now - timedelta(hours=news_hours)).isoformat(timespec='seconds')
        counts = conn.execute(
            f"SELECT symbol, count(*) FROM news WHERE symbol IN ({q}) AND published_at >= ? "
            f"GROUP BY symbol ORDER BY count(*) DESC", (*syms, since)).fetchall()
        if counts:
            out.append(f"Headline count last {news_hours}h: "
                       + ", ".join(f"{s} {n}" for s, n in counts) + ".")
            heads = conn.execute(
                f"SELECT symbol, published_at, headline FROM news WHERE symbol IN ({q}) "
                f"AND published_at >= ? ORDER BY published_at DESC LIMIT ?",
                (*syms, since, max_headlines)).fetchall()
            out.append("Latest headlines (newest first):")
            out += [f"- [{s}] {h} ({p[:16]}Z)" for s, p, h in heads]
        else:
            out.append(f"Headline count last {news_hours}h: none stored for these symbols.")
    except Exception:
        out.append("News: not collected yet.")
    return "\n".join(out)


def build_brief_context(budget, engine, intraday, day_et, conn=None, now=None) -> str:
    """Everything the daily brief may talk about, as one facts block.
    Runs in a thread (SQLite + arithmetic); never call from the event loop."""
    from src.scorecard import build_scorecard
    now = now or datetime.now(timezone.utc)
    sc = build_scorecard(budget, engine, intraday, day_et, now=now)
    parts = [f"Date: {day_et} (ET). Time now: {now.astimezone(timezone.utc).strftime('%H:%M')} UTC.",
             render_scorecard(sc), render_positions(sc.get('positions') or [])]
    held = [p['symbol'] for p in sc.get('positions') or []]
    watch = held or [c['symbol'] for c in sc.get('closed_today') or []]
    if conn is not None:
        parts.append(load_calendar_and_news(conn, watch, now=now) if watch
                     else "Calendar and news: nothing held or closed today to look up.")
    return "\n\n".join(parts)


def build_risk_context(pnl: dict, max_positions, stop_loss_pct=None, take_profit_pct=None,
                       daily_loss_limit_pct=None) -> str:
    parts = [render_positions(pnl.get('positions') or []),
             f"Equity {_fmt_usd(pnl.get('equity'))}, cash {_fmt_usd(pnl.get('cash'))}, "
             f"cost basis deployed {_fmt_usd(pnl.get('cost_basis'))}, "
             f"unrealised {_fmt_usd(pnl.get('unrealized'))}."]
    rules = [f"max {max_positions} positions at once"]
    if stop_loss_pct is not None:
        rules.append(f"stop-loss {_pct(stop_loss_pct)} below entry")
    if take_profit_pct is not None:
        rules.append(f"take-profit {_pct(take_profit_pct)} above entry")
    if daily_loss_limit_pct is not None:
        rules.append(f"no new entries after a {daily_loss_limit_pct:g}% daily loss")
    parts.append("Bot's own risk rules: " + ", ".join(rules) + ". Stocks are flattened before the close.")
    if pnl.get('stale'):
        parts.append(f"Positions with no current price (stale): {', '.join(pnl['stale'])}.")
    return "\n".join(parts)


# --- the analyst ---------------------------------------------------------------

class ClaudeAnalyzer:
    """Name kept for the call sites; it is the LLM analyst, whichever backend."""

    def __init__(self):
        self.base_url = LLM_BASE_URL
        self.local_model = LLM_MODEL
        self.api_key = os.getenv('CLAUDE_API_KEY', '').strip()
        self.client = None
        if self.api_key:
            try:
                from anthropic import AsyncAnthropic
                self.client = AsyncAnthropic(api_key=self.api_key)
            except Exception as e:   # SDK missing or broken: local still works
                logger.warning(f"anthropic client unavailable ({e})")
        self.enabled = bool(self.base_url or self.client)
        if self.base_url:
            logger.info(f"LLM analyst: local {self.local_model} at {self.base_url}"
                        + (f"; Claude {CLAUDE_MODEL} for the weekly review" if self.client else ""))
        elif self.client:
            logger.info(f"LLM analyst: Claude {CLAUDE_MODEL} only (no LLM_BASE_URL)")
        else:
            logger.warning("⚠️ Neither LLM_BASE_URL nor CLAUDE_API_KEY set - analyst disabled")

    @property
    def backend_name(self) -> str:
        return (f"local {self.local_model}" if self.base_url
                else f"Claude {CLAUDE_MODEL}" if self.client else "disabled")

    def is_enabled(self):
        return self.enabled

    # -- transports --

    async def _ask_local(self, user: str, max_tokens=400, temperature=0.3) -> str:
        import aiohttp
        body = {
            "model": self.local_model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": temperature,
            # Qwen3 "thinking" doubles latency and adds nothing to a summary.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(f"{self.base_url}/chat/completions", json=body) as r:
                r.raise_for_status()
                data = await r.json()
        text = data['choices'][0]['message']['content'] or ''
        # Belt and braces: strip a <think> block if the template ignored the flag.
        return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()

    async def _ask_claude(self, user: str, max_tokens=4000) -> str:
        response = await self.client.messages.create(
            model=CLAUDE_MODEL, max_tokens=max_tokens, system=SYSTEM,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": user}])
        if getattr(response, "stop_reason", None) == "refusal":
            return "Claude declined to answer that request."
        return "\n".join(b.text for b in response.content if b.type == "text").strip()

    async def _ask(self, user: str, prefer='local', max_tokens=400, kind='ask') -> str:
        """Route a grounded prompt; flag any number the model invented."""
        if not self.enabled:
            return "AI analyst not configured (set LLM_BASE_URL or CLAUDE_API_KEY)."
        order = ['local', 'claude'] if prefer == 'local' else ['claude', 'local']
        last_err = None
        for backend in order:
            if backend == 'local' and not self.base_url:
                continue
            if backend == 'claude' and not self.client:
                continue
            t0 = time.monotonic()
            try:
                answer = (await self._ask_local(user, max_tokens) if backend == 'local'
                          else await self._ask_claude(user, max(max_tokens, 4000)))
                _stats(backend, True, time.monotonic() - t0, kind)
                bad = unsourced_numbers(user, answer)
                if bad:
                    logger.warning(f"[{backend}] answer contains figures not in the facts: {bad}")
                    answer += f"\n\n⚠️ Not in the source data: {', '.join(bad)}"
                return answer
            except Exception as e:
                last_err = e
                _stats(backend, False, time.monotonic() - t0, kind)
                logger.error(f"❌ {backend} LLM error: {e}")
        return f"AI analyst unavailable: {last_err}"

    # -- public API (names unchanged for the Discord call sites) --

    async def daily_market_analysis(self, context: str = None):
        """Narrate the facts block from build_brief_context(). With no
        context there is nothing honest to say, so say that."""
        if not context:
            return ("No data was supplied for the brief. The old behaviour - asking "
                    "the model what is happening in the market today - was removed "
                    "because it could only invent an answer.")
        prompt = (f"FACTS:\n{context}\n\n"
                  "Write a daily brief of 3-5 sentences: how the paper account stands, "
                  "what happened today, anything on the calendar or in the headlines for "
                  "names we hold, and what the bot's own verdicts say about whether it "
                  "should keep trading. Copy figures exactly as written in the facts.")
        out = await self._ask(prompt, prefer='local', max_tokens=450, kind='daily_brief')
        logger.info("✅ Daily brief generated")
        return out

    async def analyze_portfolio_risk(self, positions, context: str = None):
        """Risk read on open positions. `context` should come from
        build_risk_context(); bare `positions` is accepted for back-compat."""
        facts = context or render_positions(positions or [])
        prompt = (f"FACTS:\n{facts}\n\n"
                  "Write a 3-4 sentence risk assessment: concentration (is one name "
                  "or one theme most of the book), position size relative to equity, "
                  "and what the bot's own rules will do about a move against us. "
                  "Copy figures exactly as written in the facts.")
        out = await self._ask(prompt, prefer='local', max_tokens=400, kind='risk_check')
        logger.info("✅ Risk assessment generated")
        return out

    async def loss_review(self, context: str):
        """Think about a losing day. `context` comes from
        postmortem.build_postmortem_context(): the trades, the price paths,
        the gaps, and the patterns Python already counted."""
        # An 8B model handed "no trades" will still write a story about the
        # trades (observed 2026-09-17). Nothing to review means no call.
        if not context or 'No trades on this date' in context:
            return "No trades on this date - nothing to review."
        prompt = (f"FACTS:\n{context}\n\n"
                  "How to read the facts: 'entry was +X% vs yesterday' means the stock "
                  "had already GAPPED UP by X% before the bot bought - the bot did not "
                  "earn that move, it paid up for it. 'model p' is the model's "
                  "probability at entry; compare it with the others. 'stopped out "
                  "within N min' means the stop-loss fired that quickly. The same "
                  "pattern across every trade on a day is a rule problem, not luck.\n\n"
                  "Write a post-mortem of 4-6 sentences for the trader. First say "
                  "plainly what the losing trades had in common, using the patterns "
                  "and figures given. Then say whether this looks like bad luck or a "
                  "repeatable mistake, and why. Finish with ONE specific question to "
                  "investigate - about the bot's rules, timing or data - not a trade "
                  "to place. Copy figures exactly as written in the facts.")
        out = await self._ask(prompt, prefer='local', max_tokens=500, kind='loss_review')
        logger.info("✅ Loss review generated")
        return out

    async def weekly_portfolio_review(self, trades, positions, performance):
        """The one call worth Claude: multi-step review. Falls back to local."""
        facts = (f"Trades this week:\n{json.dumps(trades, indent=1, default=str)}\n\n"
                 f"{render_positions(positions or [])}\n\nPerformance:\n{performance}")
        prompt = (f"FACTS:\n{facts}\n\n"
                  "Write a weekly review of 4-6 sentences: what worked, what did not, "
                  "and one concrete thing to examine next week - phrased as a question "
                  "to investigate, not a trade to place.")
        out = await self._ask(prompt, prefer='claude', max_tokens=800, kind='weekly_review')
        logger.info("✅ Weekly review generated")
        return out

    async def signal_explanation(self, symbol, signal, price, confidence):
        """Kept for callers; there is nothing an LLM can add to a model
        probability, so this is a formatted line, not a call."""
        signal_type = "BUY" if signal == 1 else "SELL"
        return f"Signal: {symbol} - {signal_type} at ${price:.2f} (model probability {confidence:.2%})"

    async def news_impact_analysis(self, symbol, news_headline):
        prompt = (f"FACTS:\nStock: {symbol}\nHeadline: {news_headline}\n\n"
                  "In one or two sentences, say what the headline is about and whether "
                  "it reads as positive, negative or neutral for the company. Do not "
                  "predict the price.")
        return await self._ask(prompt, prefer='local', max_tokens=150, kind='news_impact')
