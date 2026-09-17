#!/usr/bin/env python3
"""
Discord Bot Integration
Handles alerts, approvals, and Claude analysis
"""

import discord
from discord.ext import commands, tasks
from discord import app_commands
import os
import asyncio
import math
from datetime import datetime, timezone, time as dtime
import logging
from src import costs, signal_log
from src.claude_analyzer import ClaudeAnalyzer, build_brief_context, build_risk_context
from src.budget_tracker import BudgetTracker, _et_date
from src.costs import qty_str
from src.database import connect
from src.ml_engine import load_universe
from src.intraday_engine import (IntradayEngine, market_state,
                                 minutes_to_close, ET, is_crypto,
                                 asset_class, barriers, next_trading_day_open)
from src.fast_trader import FastTrader, class_gate, max_hold_min
from src.scorecard import build_scorecard, cost_breakeven

logger = logging.getLogger(__name__)


def _signed_usd(x: float) -> str:
    """'+$1.23' / '-$1.23' - the sign goes before the dollar sign, which an
    f-string format spec cannot do on its own."""
    return f"{'-' if x < 0 else '+'}${abs(x):,.2f}"


def _icon(cls: str) -> str:
    return '🪙' if cls == 'crypto' else '📈'


def _clip_lines(lines, limit: int = 1000) -> str:
    """Join lines for one embed field, dropping trailing lines until the text
    (including its '… and N more' tail) fits. Discord caps a field value at
    1024 chars; 1000 leaves headroom for markdown the caller adds."""
    lines = list(lines)
    kept = len(lines)
    while kept > 0:
        text = "\n".join(lines[:kept])
        if kept < len(lines):
            text += f"\n… and {len(lines) - kept} more"
        if len(text) <= limit:
            return text
        kept -= 1
    return f"… and {len(lines)} more"

class TradingBot(commands.Cog):
    def __init__(self, bot_instance=None, engine=None, db=None):
        if bot_instance is None:
            # Slash commands arrive as interactions over the gateway, so the
            # privileged MESSAGE CONTENT intent is NOT required - the toggle in
            # the Developer Portal can stay off.
            intents = discord.Intents.default()
            bot_instance = commands.Bot(command_prefix='!', intents=intents)
        self.bot = bot_instance
        self.engine = engine
        self.db = db
        self.budget_tracker = BudgetTracker()
        self.claude = ClaudeAnalyzer()
        self.pending_approvals = {}
        # True while the startup fetch/train is still running, so commands can
        # say "warming up" instead of failing on a model that doesn't exist yet.
        self.warming_up = False

        # Intraday fast-trading stack (only started when FAST_MODE=1)
        self.fast_mode = os.getenv('FAST_MODE', '0') in ('1', 'true', 'yes')
        self.intraday = IntradayEngine() if self.fast_mode else None
        self.fast = (FastTrader(self.intraday, self.budget_tracker,
                                quote_fn=lambda s: self.engine.latest_price(s))
                     if self.fast_mode else None)
        self._last_fast_summary = None

        # Add cogs
        self.bot.add_listener(self.on_ready)
        self._register_slash()
    

    def _heartbeat_embed(self, symbols, ranked, actionable, alerting, held,
                         min_prob, pnl):
        """Plain-English hourly report: what you own, what it's worth, what to do."""
        buys = [r for r in ranked if r['signal'] == 1]
        sells = [r for r in ranked if r['signal'] == 0]
        positions = pnl['positions']
        total_pl = pnl['unrealized']
        realized = pnl['realized']
        spent = pnl['cost_basis']
        bp = pnl['buying_power']
        univ = os.getenv('UNIVERSE', 'default')
        univ_label = 'S&P 500' if univ.lower() == 'sp500' else univ

        if total_pl > 0:
            mood, color = "📈 UP", discord.Color.green()
        elif total_pl < 0:
            mood, color = "📉 DOWN", discord.Color.red()
        else:
            mood, color = "➖ FLAT", discord.Color.greyple()

        embed = discord.Embed(
            title=f"⏱️ Hourly Check — {datetime.now().strftime('%-I:%M %p')}",
            color=color if positions else discord.Color.greyple(),
            timestamp=datetime.now().astimezone(),
        )

        # --- money ---------------------------------------------------------
        money = (f"Balance **${pnl['equity']:,.2f}** (started with "
                 f"${pnl['starting_cash']:,.2f})\n"
                 f"Buying power **${bp:,.2f}** · Unsettled ${pnl['unsettled']:,.2f}\n")
        if positions:
            pct = f" ({total_pl / spent:+.1%})" if spent else ""
            money += (f"You spent **${spent:,.2f}** on {len(positions)} position(s), "
                      f"worth **${pnl['market_value']:,.2f}** right now\n\n"
                      f"**{mood} ${abs(total_pl):,.2f}{pct}**")
            if realized:
                money += f"\nAlready banked from past sales: **${realized:,.2f}**"
        else:
            money += "You haven't bought anything yet."
        embed.add_field(name="💰 YOUR MONEY", value=money, inline=False)

        # --- holdings ------------------------------------------------------
        if positions:
            lines = []
            for p in positions[:8]:
                if p['pnl'] is None:
                    lines.append(f"**{p['symbol']}** {qty_str(p['shares'])} — price unavailable")
                    continue
                arrow = "🟩 UP" if p['pnl'] > 0 else "🟥 DOWN" if p['pnl'] < 0 else "⬜ flat"
                lines.append(
                    f"**{p['symbol']}** {qty_str(p['shares'])} — paid ${p['avg_price']:,.2f}, "
                    f"now ${p['price']:,.2f} → {arrow} **${abs(p['pnl']):,.2f}** "
                    f"({p['pnl_pct']:+.1%})")
            embed.add_field(name="📊 WHAT YOU OWN", value="\n".join(lines), inline=False)

        # --- what to do ----------------------------------------------------
        held_sells = [r for r in actionable if r['signal'] == 0]
        if alerting:
            todo = (f"**I sent you {len(alerting)} alert(s) — look just below this message.**\n"
                    f"React ✅ to take the trade, ❌ to skip it. "
                    f"If you ignore it for 5 minutes it cancels itself.")
            if held_sells:
                todo += (f"\n\n⚠️ One of them is a **SELL of something you own** — "
                         f"the model thinks it's about to drop.")
        elif not ranked:
            todo = ("**Nothing to do.** Nothing looked strong enough this hour. "
                    "That's normal — most hours are quiet.")
        elif buys and not alerting:
            todo = (f"**Nothing to do.** {len(buys)} stock(s) looked good but none fit "
                    f"your ${bp:,.2f} of buying power.")
        else:
            todo = ("**Nothing to do.** The model thinks the whole market is heading "
                    "down right now. You can't sell what you don't own, so there's "
                    "nothing to act on.")
            if positions:
                todo = ("**Nothing to do.** The model is negative on the market, but "
                        "nothing you own crossed the sell threshold. Holding is fine.")
        todo += "\n\nWant to look yourself? Type **/scan** for a live ranking, or **/pnl** for detail."
        embed.add_field(name="👉 WHAT TO DO", value=todo, inline=False)

        # --- what it checked ------------------------------------------------
        embed.add_field(
            name="🔍 WHAT I CHECKED",
            value=(f"All **{len(symbols)} {univ_label}** stocks (big US companies only), "
                   f"using yesterday's closing prices from Yahoo Finance.\n"
                   f"**{len(ranked)}** looked interesting: **{len(buys)} buy**, "
                   f"**{len(sells)} sell**. I ignored **{len(sells) - len(held_sells)}** "
                   f"sells on stocks you don't own."),
            inline=False)

        embed.set_footer(text="⚠️ PRACTICE MONEY — no broker is connected, no real trades happen. "
                              "Next check in 1 hour.")
        return embed

    async def _symbols(self):
        """Universe list without ever touching the network on the event loop."""
        if self.engine.symbols:
            return self.engine.symbols
        return await asyncio.to_thread(load_universe)

    def _warming_embed(self):
        """Non-None while startup fetch/training is still in flight."""
        if not self.warming_up:
            return None
        return discord.Embed(
            title="⏳ Still warming up",
            description=("Fetching prices and training the model - this takes "
                         "about 4 minutes after a restart. Try again shortly."),
            color=discord.Color.greyple(),
        )

    def _register_slash(self):
        """Expose every command as a / slash command.

        All of these defer first: Discord kills an interaction that isn't
        acknowledged within 3 seconds, and scan/Claude calls take longer.
        """
        tree = self.bot.tree

        def bind(name, description, builder):
            @tree.command(name=name, description=description)
            async def _cmd(interaction: discord.Interaction):
                await interaction.response.defer(thinking=True)
                try:
                    embed = self._warming_embed() or await builder()
                except Exception as e:
                    logger.exception(f"/{name} failed")
                    embed = discord.Embed(title=f"/{name} failed", description=str(e),
                                          color=discord.Color.red())
                await interaction.followup.send(embed=embed)
            return _cmd

        bind('status', 'Portfolio status and buying power', self._embed_status)
        bind('account', 'Paper account: balance, cash, unsettled, buying power',
             self._embed_account)
        bind('pnl', "Today's scorecard: all-time P&L and the go/no-go verdict",
             self._embed_pnl)
        bind('summary', "Today's scorecard (same as /pnl)", self._embed_pnl)
        bind('daily_brief', 'Claude daily market analysis', self._embed_daily_brief)
        bind('risk_check', 'Claude risk assessment of open positions', self._embed_risk_check)
        bind('pause', 'Pause the monitoring loop', self._embed_pause)
        bind('resume', 'Resume the monitoring loop', self._embed_resume)

        @tree.command(name='scan', description='Rank the whole universe by model confidence')
        @app_commands.describe(top='How many to show (1-20, default 10)')
        async def _scan(interaction: discord.Interaction, top: int = 10):
            await interaction.response.defer(thinking=True)
            try:
                embed = self._warming_embed() or await self._embed_scan(top)
            except Exception as e:
                logger.exception("/scan failed")
                embed = discord.Embed(title="/scan failed", description=str(e),
                                      color=discord.Color.red())
            await interaction.followup.send(embed=embed)

        @tree.command(name='retrain',
                      description='Refresh prices from all sources and retrain (~4 min)')
        @app_commands.describe(fetch='Also re-download prices (default yes)')
        async def _retrain(interaction: discord.Interaction, fetch: bool = True):
            await interaction.response.defer(thinking=True)
            if self.warming_up:
                await interaction.followup.send(embed=self._warming_embed())
                return
            await interaction.followup.send(
                "🔄 Refreshing data and retraining — this takes about 4 minutes. "
                "I'll post the result here when it's done.")
            try:
                self.warming_up = True
                if fetch:
                    await asyncio.to_thread(
                        self.engine.fetch_and_store_data, await self._symbols())
                if self.intraday:
                    await asyncio.to_thread(self.intraday.full_fetch)
                ok = await asyncio.to_thread(self.engine.train_model)
                m = self.engine.last_metrics
                if ok and m:
                    embed = discord.Embed(
                        title="✅ Retrained",
                        color=discord.Color.green(),
                        description=(f"Target **{m['mode']}** on **{m['rows']:,}** rows\n"
                                     f"Accuracy **{m['accuracy']:.3f}** vs baseline "
                                     f"**{m['baseline']:.3f}** → edge **{m['edge']:+.3f}**"))
                    if m['edge'] <= 0.005:
                        embed.add_field(
                            name="⚠️ Reality check",
                            value=("An edge at or below 0.005 is indistinguishable from "
                                   "noise. Treat signals as a shortlist, not a prediction."),
                            inline=False)
                    src = ", ".join(f"{k}: {v['symbols']}"
                                    for k, v in self.engine.source_stats.items()) or "cached"
                    embed.add_field(name="Data sources", value=src, inline=False)
                else:
                    embed = discord.Embed(title="❌ Training failed",
                                          description="Not enough usable data.",
                                          color=discord.Color.red())
            except Exception as e:
                logger.exception("/retrain failed")
                embed = discord.Embed(title="❌ /retrain failed", description=str(e),
                                      color=discord.Color.red())
            finally:
                self.warming_up = False
            await interaction.followup.send(embed=embed)

        @tree.command(name='fast', description='Intraday auto-trading status')
        async def _fast(interaction: discord.Interaction):
            await interaction.response.defer(thinking=True)
            if not self.fast_mode or not self.intraday:
                await interaction.followup.send(embed=discord.Embed(
                    title="Fast mode is off",
                    description="Set `FAST_MODE=1` in .env and restart to enable "
                                "intraday auto-trading.",
                    color=discord.Color.greyple()))
                return
            state, desc = market_state()
            e = discord.Embed(
                title=f"⚡ Intraday mode — market is {desc}",
                color=discord.Color.gold() if state == 'open' else discord.Color.greyple())
            metrics = self.intraday.metrics or {}
            for cls, ok, text in self._class_gates():
                mm = metrics[cls]
                e.add_field(
                    name=f"{_icon(cls)} {cls.title()} model ({mm.get('symbols', 0)} symbols)",
                    value=(f"Target **+{mm.get('take_profit', 0):.2%}** before "
                           f"**−{mm.get('stop_loss', 0):.2%}** within "
                           f"{mm.get('horizon_bars', 0) * 5} min\n"
                           f"Precision **{mm.get('precision', 0):.1%}** vs break-even "
                           f"**{mm.get('breakeven', 0):.1%}** "
                           f"({mm.get('test_signals', 0):,} held-out signals)\n"
                           f"**EV {mm.get('ev', 0) * 100:+.3f}% per trade** — "
                           f"{'✅' if ok else '⛔'} {text}\n"
                           f"Bar p>{mm.get('bar', 0):.3f} · {mm.get('rows', 0):,} rows"),
                    inline=False)
            if not metrics:
                e.add_field(name="Model", value="not trained yet", inline=False)
            # Max hold is derived from each class's label horizon, not a setting.
            rules = "\n".join(
                f"{_icon(c)} {c}: take profit **+{barriers(c)[0]:.1%}** · "
                f"stop **−{barriers(c)[1]:.1%}** · max hold **{max_hold_min(c):.0f} min**"
                for c in (sorted(metrics) or ['stock']))
            e.add_field(
                name="Rules",
                value=(f"Poll every **{os.getenv('FAST_POLL_SECONDS', 60)}s** · "
                       f"max **{self.fast.max_positions}** positions · flatten stocks "
                       f"**{self.fast.eod_flatten_min:.0f} min** before the close · "
                       f"re-entry cooldown **{self.fast.cooldown_min:.0f} min**\n{rules}"),
                inline=False)
            ranked = await asyncio.to_thread(self.intraday.scan_all)
            e.add_field(
                name="Right now (each scored against its own class bar)",
                value=("\n".join(
                    f"{'✅' if r['above_bar'] else '▫️'} "
                    f"{_icon(r['asset_class'])} "
                    f"**{r['symbol']}** {r['probability']:.1%} "
                    f"(bar {r['bar']:.3f}, {r['margin']:+.3f}) · ${r['price']:,.2f}"
                    for r in ranked[:8]) or "no scores yet"),
                inline=False)
            pos = self.budget_tracker.get_positions()
            e.add_field(name="Open positions",
                        value=("\n".join(f"**{p['symbol']}** x{qty_str(p['shares'])} @ "
                                          f"${p['avg_price']:,.2f}" for p in pos)
                               if pos else "none"), inline=False)
            e.set_footer(text="PAPER TRADING — no broker connected")
            await interaction.followup.send(embed=e)

        @tree.command(name='sources', description='Show where market data is coming from')
        async def _sources(interaction: discord.Interaction):
            await interaction.response.defer(thinking=True)
            e = discord.Embed(title="📡 Market data sources", color=discord.Color.blurple())
            hist = [p.name for p in self.engine.providers if p.provides_history]
            quotes = [p.name for p in self.engine.quote_providers]
            e.add_field(name="Price history (in order)",
                        value=" → ".join(hist) or "none", inline=False)
            e.add_field(name="Live quotes (in order)",
                        value=" → ".join(quotes) or "none — using last stored close",
                        inline=False)
            if self.engine.source_stats:
                e.add_field(
                    name="Last refresh",
                    value="\n".join(f"**{k}**: {v['symbols']} symbols, {v['rows']:,} rows"
                                     for k, v in self.engine.source_stats.items()),
                    inline=False)
            e.add_field(
                name="How the chain works",
                value=("Each source only sees the symbols the previous one missed, so a "
                       "rate-limited source is spent on real gaps. Live quotes are also "
                       "cross-checked against stored closes — a big gap means stale history."),
                inline=False)
            await interaction.followup.send(embed=e)

        logger.info("Registered 12 slash commands")

    async def on_ready(self):
        """Bot startup event"""
        logger.info(f"✅ Bot logged in as {self.bot.user}")
        logger.info("📝 PAPER TRADING MODE - no broker is connected; approvals "
                    "only write to the local ledger")
        # Guild sync is instant; the global sync is what makes the commands
        # usable in DMs, and can take up to an hour to propagate.
        try:
            for guild in self.bot.guilds:
                self.bot.tree.copy_global_to(guild=guild)
                await self.bot.tree.sync(guild=guild)
                logger.info(f"⚡ Slash commands synced to '{guild.name}' (instant)")
            synced = await self.bot.tree.sync()
            logger.info(f"⚡ {len(synced)} slash commands synced globally "
                        f"(DM availability may take up to 1h to propagate)")
        except Exception as e:
            logger.error(f"Slash command sync failed: {e}")

        logger.info("📊 Starting trading monitor...")

        # The daily scorecard must post even when training fails or fast mode
        # is off: it is the one message that says whether the account is alive.
        if not self.daily_summary.is_running():
            self.daily_summary.start()

        # Startup fetch (~110s) and training (~130s) are synchronous. Run them
        # OFF the event loop - inline they block every interaction for minutes
        # and Discord answers slash commands with "application did not respond".
        # A PENDING trade's approval watcher lives in memory, so anything left
        # PENDING by a previous run can never be approved - it would just hold
        # buying power forever. Clear them at startup.
        try:
            stale = self.budget_tracker.conn.execute(
                "SELECT id, symbol, side FROM trades WHERE status = 'PENDING'").fetchall()
            for row in stale:
                self.budget_tracker.reject_trade(row['id'])
            if stale:
                logger.info(f"🧹 Cleared {len(stale)} stale PENDING trade(s) from a "
                            f"previous run: " +
                            ", ".join(f"#{r['id']} {r['side']} {r['symbol']}" for r in stale))
        except Exception as e:
            logger.error(f"Stale-pending cleanup failed: {e}")

        self.warming_up = True
        try:
            logger.info("📥 Fetching market data...")
            symbols = await asyncio.to_thread(load_universe)
            logger.info(f"📊 Universe: {len(symbols)} symbols")
            await asyncio.to_thread(self.engine.fetch_and_store_data, symbols)

            logger.info("🧠 Training ML model...")
            trained = await asyncio.to_thread(self.engine.train_model)
            if trained:
                logger.info("✅ Model trained successfully")
                if not self.monitor_trading.is_running():
                    self.monitor_trading.start()
            else:
                logger.error("❌ Model training failed")
            if self.fast_mode:
                logger.info("⚡ FAST MODE - preparing intraday model...")
                await asyncio.to_thread(self.intraday.full_fetch)
                if await asyncio.to_thread(self.intraday.train):
                    # Logged once per class at startup, from the same gate the
                    # trader applies, so the log never contradicts the trades.
                    for cls, ok, text in self._class_gates():
                        m = self.intraday.metrics[cls]
                        logger.info(f"⚡ {cls} model ready: precision "
                                    f"{m['precision']:.1%} vs breakeven "
                                    f"{m['breakeven']:.1%}, EV {m['ev'] * 100:+.3f}%/trade "
                                    f"- {'TRADEABLE' if ok else 'GATED'}: {text}; "
                                    f"bar p>{m['bar']:.3f}")
                    if not self.fast_cycle.is_running():
                        self.fast_cycle.start()
                    state, desc = market_state()
                    logger.info(f"⚡ Fast loop started every "
                                f"{os.getenv('FAST_POLL_SECONDS', 60)}s - market is {desc}")
                else:
                    logger.error("⚡ Intraday training failed - fast mode disabled")
                    self.fast_mode = False
        finally:
            self.warming_up = False
            logger.info("🟢 Ready - slash commands are live")
            try:
                await self._send_startup_notice()
            except Exception as e:
                logger.error(f"startup notice failed: {e}")

    def _class_gates(self):
        """[(cls, tradeable, text)] for every trained class, from the ONE gate
        FastTrader uses - so the log, the startup notice and the trades can
        never disagree about what is being traded."""
        if not self.fast_mode or not self.intraday:
            return []
        metrics = self.intraday.metrics or {}
        return [(cls, *class_gate(cls, metrics[cls])) for cls in sorted(metrics)]

    def _gate_lines(self) -> str:
        return "\n".join(f"{'✅' if ok else '⛔'} {_icon(cls)} {cls}: {text}"
                         for cls, ok, text in self._class_gates())

    def _held_prices(self) -> dict:
        """Quotes for held symbols only, keyed by symbol. A failed lookup is
        left out so get_equity() values that position at avg_price."""
        prices = {}
        for p in self.budget_tracker.get_positions():
            try:
                q = self.engine.latest_price(p['symbol'])
            except Exception as exc:
                logger.warning(f"quote failed for {p['symbol']}: {exc}")
                q = None
            if q:
                prices[p['symbol']] = float(q)
        return prices

    def _add_account_fields(self, e, s=None):
        """The Balance / Buying power / Unsettled trio every money embed shows.
        Prefers the cycle summary's figures (quoted this cycle, no second
        lookup) and falls back to the ledger."""
        s = s or {}
        bt = self.budget_tracker
        equity = s.get('equity')
        bp = s.get('buying_power')
        unsettled = s.get('unsettled')
        if equity is None:
            equity = bt.get_equity(self._held_prices())
        if bp is None:
            bp = bt.get_buying_power()
        if unsettled is None:
            unsettled = bt.get_unsettled()
        e.add_field(name="Balance",
                    value=f"${equity:,.2f} (started with ${bt.starting_cash():,.2f})",
                    inline=True)
        e.add_field(name="Buying power", value=f"${bp:,.2f}", inline=True)
        e.add_field(name="Unsettled", value=f"${unsettled:,.2f}", inline=True)

    def _startup_embed(self) -> discord.Embed:
        """The restart notice. Headline and colour come from whether ANY class
        clears class_gate; 'Nothing will be traded' only when every class is
        gated (cost floor or EV)."""
        gates = self._class_gates()
        will_trade = [cls for cls, ok, _ in gates if ok]
        metrics = (self.intraday.metrics or {}) if gates else {}

        e = discord.Embed(
            title="🔄 Trading bot restarted",
            description=("**Trading is LIVE** - you will hear from me when I buy or sell."
                         if will_trade else
                         "**Nothing will be traded right now.** Every asset class is "
                         "gated (cost floor or expected value), so the bot is watching "
                         "only. This is the risk gate working, not a crash."),
            color=discord.Color.green() if will_trade else discord.Color.orange(),
            timestamp=datetime.now().astimezone())

        for cls, ok, text in gates:
            m = metrics[cls]
            e.add_field(
                name=f"{_icon(cls)} {cls.title()}",
                value=(f"{'✅' if ok else '⛔'} {text}\n"
                       f"Gets it right {m.get('precision', 0):.1%} of the time; needs "
                       f"{m.get('breakeven', 0):.1%} just to break even."),
                inline=False)

        if not gates:
            e.add_field(name="Models",
                        value="No intraday model is loaded - fast mode is off.",
                        inline=False)

        try:
            held = self.budget_tracker.get_positions()
            e.add_field(
                name="Open paper positions",
                value=("none" if not held else
                       ", ".join(f"{qty_str(p['shares'])} {p['symbol']}" for p in held)),
                inline=False)
            self._add_account_fields(e)
        except Exception as exc:
            logger.warning(f"startup notice: account fields skipped: {exc}")

        e.set_footer(text="Quiet mode: no routine updates. You only hear from me "
                          "when I trade, or when I restart. PAPER TRADING.")
        return e

    async def _send_startup_notice(self):
        """Post exactly one message on startup saying whether the bot will
        trade and, if not, why.

        Routine chatter is off (HEARTBEAT=0, FAST_SUMMARY_MINUTES=0), so the
        channel is silent unless a trade happens. That makes a dead process and
        a deliberately idle one look identical. This notice is the one message
        that distinguishes them: it fires on every restart, and it names the
        gate that is blocking each asset class.
        """
        if os.getenv('STARTUP_NOTICE', '1') in ('0', 'false', 'no'):
            return
        channel = await self._destination()
        if not channel:
            return
        # _startup_embed quotes held symbols - keep that off the event loop.
        await channel.send(embed=await asyncio.to_thread(self._startup_embed))

    async def _destination(self):
        """Where alerts go: your DM by default, a guild channel if CHANNEL_ID is set.

        DM mode needs no CHANNEL_ID - the bot opens (or reuses) a DM with
        USER_ID. Discord still requires that the bot share a server with you
        for the DM to be deliverable, so the invite step is unchanged.
        """
        channel_id = (os.getenv('CHANNEL_ID') or '').strip()

        if channel_id and channel_id.strip('0'):
            channel = self.bot.get_channel(int(channel_id))
            if channel is None:
                try:
                    channel = await self.bot.fetch_channel(int(channel_id))
                except Exception as e:
                    logger.error(f"❌ CHANNEL_ID {channel_id} not reachable: {e}")
                    return None
            return channel

        try:
            user = self.bot.get_user(int(os.getenv('USER_ID'))) or \
                   await self.bot.fetch_user(int(os.getenv('USER_ID')))
            return user.dm_channel or await user.create_dm()
        except discord.Forbidden:
            logger.error("❌ Cannot DM you - check that you share a server with "
                         "the bot and that DMs from server members are allowed")
        except Exception as e:
            logger.error(f"❌ Could not open DM with USER_ID: {e}")
        return None

    @tasks.loop(minutes=5)
    async def daily_summary(self):
        """Post the day's scorecard once, any time after 16:05 ET - weekends
        and holidays included, so a quiet Saturday still proves the account is
        alive. The once-per-date guard is day_state.report_posted_at, not
        memory, so a restart inside the window cannot double-post.

        on_ready starts this loop before the warm-up (the report must post
        even when training fails), and tasks.loop runs the first iteration at
        once. Skip while warming up: a restart after 16:05 ET on an unposted
        date would otherwise render every class as 'no trained model' and
        latch report_posted_at on it. warming_up is cleared in on_ready's
        finally, so a failed training still lets the report through."""
        if self.warming_up:
            return
        now = datetime.now(ET)
        if now.time() < dtime(16, 5):
            return
        day_et = now.strftime('%Y-%m-%d')
        try:
            await self._post_daily_report(day_et)
        except Exception:
            logger.exception(f"Daily report for {day_et} failed - retrying next tick")

    def _close_the_books(self, day_et: str, now=None):
        """Steps 1-2 of the 16:05 tick: label pending signals, then freeze the
        day's equity row. Runs BEFORE any Discord call so the record of the
        day never depends on delivery."""
        bt = self.budget_tracker
        # label_pending commits per row (`with conn:`). On the ledger's own
        # connection that commit would also commit whatever the fast-cycle
        # thread has half-done inside BudgetTracker._txn() - crypto keeps that
        # loop alive after the bell - and a later rollback there would then
        # have nothing to undo. So the labelling pass gets its own connection
        # to the same file (bt.db_path is exposed for exactly this) and closes
        # it after.
        conn = connect(bt.db_path)
        try:
            n = signal_log.label_pending(conn, now=now)
        finally:
            conn.close()
        row = bt.record_equity(day_et, self._held_prices(), now=now)
        logger.info(f"{day_et}: labelled {n} signal(s); equity ${row['equity']:,.2f} recorded")

    async def _post_daily_report(self, day_et: str, now=None) -> bool:
        """One attempt at the daily report. Returns True when it was posted
        (embed or text fallback), False when already posted or undeliverable.
        Any other exception propagates so the loop logs it and retries."""
        bt = self.budget_tracker
        state = bt.get_day_state(day_et)
        if state is None:
            # No cycle ran today (weekend, holiday, fast mode off) - the report
            # still needs a baseline row to hang its flag on.
            state = await asyncio.to_thread(
                lambda: bt.ensure_day_state(day_et, bt.get_equity(self._held_prices())))
        if state['report_posted_at']:
            return False

        await asyncio.to_thread(self._close_the_books, day_et, now)

        channel = await self._destination()
        if channel is None:
            logger.error(f"Daily report for {day_et}: no Discord destination - "
                         f"retrying next tick")
            return False
        try:
            embed = await asyncio.to_thread(self._scorecard_embed, day_et, now)
            await channel.send(embed=embed)
        except discord.HTTPException as exc:
            # Discord refused the embed itself (too long, bad field). A one-liner
            # still carries the headline; the flag is set so we do not spam.
            logger.error(f"Daily report embed for {day_et} rejected by Discord: {exc}")
            pnl = await asyncio.to_thread(bt.get_pnl, self.engine.latest_price)
            await channel.send(
                f"📊 Paper scorecard {day_et}: all-time {_signed_usd(pnl['all_time_net'])} "
                f"({pnl['all_time_pct']:+.2%}) — full report failed: {exc}")
        bt.set_day_flag(day_et, 'report_posted_at')
        logger.info(f"Posted daily report for {day_et}")
        return True

    def _scorecard_embed(self, day_et: str, now=None) -> discord.Embed:
        """Render build_scorecard() for one ET date. Shared by the 16:05
        report, /pnl and /summary so there is one layout to get right.

        Units follow Task 7's contract: exec_mean_pct, exec_ci and
        max_drawdown_pct are PERCENTAGE POINTS (formatted with :f and a literal
        %); every other rate is a fraction (formatted with :%). A class without
        a trained model has bt_precision / ev_bt_net None and prints n/a - the
        report must render even when training failed or fast mode is off.

        Every list field goes through _clip_lines: Discord refuses the whole
        message when any field passes 1024 chars or the total passes 6000.
        """
        sc = build_scorecard(self.budget_tracker, self.engine, self.intraday,
                             day_et, now=now)
        h, acct, today = sc['headline'], sc['account'], sc['today']
        opened = _et_date(self.budget_tracker.opened_at())   # ET date; an evening open is already tomorrow in UTC
        net = h['all_time_net']
        colour = (discord.Color.green() if net > 0 else
                  discord.Color.red() if net < 0 else discord.Color.greyple())
        e = discord.Embed(title=f"📊 Paper scorecard — {day_et}", color=colour,
                          timestamp=datetime.now().astimezone())

        # The $ headline is never shown without its n and CI (spec section 8).
        e.add_field(
            name=f"All-time: {_signed_usd(net)} ({h['all_time_pct']:+.2%})",
            value=(f"since {opened} · n={h['n_closed']} closed trade(s), "
                   f"95% CI ±${h['ci_dollars']:,.2f}"),
            inline=False)

        classes = sc['classes']
        e.add_field(
            name="Verdict",
            value=_clip_lines([f"{_icon(cls)} **{cls}: {c['verdict']}** — {c['verdict_text']}"
                               for cls, c in sorted(classes.items())]) or "no classes scored",
            inline=False)

        # unsettled_until is already the ET date ('YYYY-MM-DD') on which the
        # proceeds settle at 09:30 ET; print it as-is, no timezone maths.
        settles = ""
        if acct['unsettled'] > 0 and acct.get('unsettled_until'):
            settles = f" (settles {acct['unsettled_until']} 09:30 ET)"
        e.add_field(
            name="Account",
            value=(f"Gross P&L {_signed_usd(acct['gross_pnl'])} · "
                   f"fees paid ${acct['fees_paid']:,.2f}\n"
                   f"Balance **${acct['equity']:,.2f}** "
                   f"(started with ${acct['starting_cash']:,.2f})\n"
                   f"Cash ${acct['cash']:,.2f} · buying power ${acct['buying_power']:,.2f}\n"
                   f"Unsettled ${acct['unsettled']:,.2f}{settles}"),
            inline=False)

        block = ("🛑 daily-loss block TRIPPED" if today['loss_tripped']
                 else "daily-loss block not tripped")
        # Spec section 1: the next open comes from the trading calendar, so a
        # Friday, weekend or holiday-eve report names the right day.
        resume = next_trading_day_open(datetime.fromisoformat(day_et).replace(tzinfo=ET))
        e.add_field(
            name=f"Today ({day_et})",
            value=(f"Realised {_signed_usd(today['realized'])} · "
                   f"unrealised {_signed_usd(today['unrealized'])}\n"
                   f"{today['n_trades']} trade(s) · won {today['wins']} · "
                   f"lost {today['losses']}\n{block}\n"
                   f"⏰ Stocks resume {resume.strftime('%a %d %b')} 09:30 ET; "
                   f"crypto keeps trading."),
            inline=False)

        if sc['closed_today']:
            e.add_field(
                name="Closed today",
                value=_clip_lines([
                    f"{'🟩' if r['net'] > 0 else '🟥'} {qty_str(r['shares'])} "
                    f"{r['symbol']} @ ${r['price']:,.2f} → **{_signed_usd(r['net'])}** "
                    f"(gross {_signed_usd(r['gross'])}, fees ${r['fees']:,.2f}) "
                    f"[{r['exit_reason']}]"
                    for r in sc['closed_today'][:10]]),
                inline=False)

        if sc['positions']:
            lines = []
            for p in sc['positions']:
                head = (f"{_icon(asset_class(p['symbol']))} {qty_str(p['shares'])} "
                        f"{p['symbol']} @ ${p['avg_price']:,.2f}")
                if p['price'] is None:
                    lines.append(f"{head} → price unavailable")
                    continue
                # pct_vs_ref is the ref-to-ref move the exit rule watches, not
                # the move against avg_price (which has the fill costs in it).
                pct = ("" if p['pct_vs_ref'] is None
                       else f" ({p['pct_vs_ref']:+.2%} vs entry ref)")
                lines.append(f"{head} → ${p['price']:,.2f}{pct}")
            e.add_field(name="Open positions", value=_clip_lines(lines), inline=False)

        so = sc['since_open']
        pf = so['profit_factor']
        pf_txt = f"{pf:.2f}" if pf is not None and math.isfinite(pf) else "n/a"
        e.add_field(
            name="Scorecard since open",
            value=(f"{so['n_closed']} trade(s) closed · win rate {so['win_rate']:.1%} "
                   f"(95% CI {so['win_lo']:.1%}–{so['win_hi']:.1%})\n"
                   f"Mean net per trade {_signed_usd(so['mean_net'])} ± ${so['mean_ci']:,.2f}\n"
                   f"Profit factor {pf_txt} · max drawdown {so['max_drawdown_pct']:.2f}% · "
                   f"{so['days_running']} day(s) running"),
            inline=False)

        for cls, c in sorted(classes.items()):
            exits = ", ".join(f"{r}: {v['n']} ({_signed_usd(v['mean_net'])})"
                              for r, v in sorted(c['exits_by_reason'].items())) or "none"
            if c['bt_precision'] is None:
                bt_txt = "walk-forward precision n/a (no trained model)"
            else:
                bt_txt = (f"walk-forward precision {c['bt_precision']:.1%} "
                          f"(n={c['bt_n']:,})")
            ev_txt = "n/a" if c['ev_bt_net'] is None else f"{c['ev_bt_net']:+.3%}/trade"
            e.add_field(
                name=f"{_icon(cls)} {cls.title()} — {c['exec_n']} closed",
                value=(f"Exits: {exits}\n"
                       f"TP-first {c['tp_first_rate']:.1%} (95% CI {c['tp_lo']:.1%}–"
                       f"{c['tp_hi']:.1%}) vs {bt_txt}\n"
                       f"Backtest EV net of costs {ev_txt} vs realised "
                       f"{c['exec_mean_pct']:+.3f}% ± {c['exec_ci']:.3f}%\n"
                       f"Gate: {c['gate_text']}\n"
                       f"Signals: n={c['sig_n']} labelled, TP-first {c['sig_rate']:.1%} "
                       f"(95% CI {c['sig_lo']:.1%}–{c['sig_hi']:.1%}, day-clustered low "
                       f"{c['sig_lo_day']:.1%}) vs cost-adjusted breakeven "
                       f"{cost_breakeven(cls):.1%}"),
                inline=False)

        spy = sc.get('spy')
        if spy:
            e.add_field(
                name="SPY buy-and-hold (context only)",
                value=(f"${acct['starting_cash']:,.2f} in SPY on {opened} → "
                       f"**${spy['value']:,.2f}** ({spy['pct']:+.2%}; "
                       f"${spy['start_close']:,.2f} → ${spy['last_close']:,.2f})\n"
                       f"Not risk-matched: SPY is exposed 24/7, the bot is flat overnight."),
                inline=False)

        e.set_footer(text="Backtest scores timeouts/EOD as −sl, assumes exact-barrier "
                          "fills, and samples intrabar highs/lows; live exits are checked "
                          "on one quote every FAST_POLL_SECONDS. PAPER TRADING.")
        return e

    @tasks.loop(seconds=float(os.getenv('FAST_POLL_SECONDS', 60)))
    async def fast_cycle(self):
        """Intraday entries and exits. Reports only when something happened, or
        every FAST_SUMMARY_MINUTES, so a 60s loop doesn't spam the channel."""
        if self.warming_up or not self.fast:
            return
        # Run the trading cycle FIRST. Resolving the Discord channel before this
        # meant a DM/API failure silently stopped every stop-loss and every
        # end-of-day flatten - reporting must never gate risk management.
        try:
            summary = await asyncio.to_thread(self.fast.cycle)
        except Exception:
            logger.exception("fast cycle failed")
            return
        channel = await self._destination()
        if not channel:
            if summary['entries'] or summary['exits']:
                logger.warning("Traded but could not reach Discord to report: "
                               f"{len(summary['entries'])} in, {len(summary['exits'])} out")
            return

        # Loss-limit announcement, once per ET date. The flag lives in
        # day_state, so a restart cannot repeat it and a failed send cannot
        # lose it - the next cycle while tripped simply tries again.
        if summary.get('loss_announce') or summary.get('loss_tripped'):
            today_et = summary['ts'].strftime('%Y-%m-%d')
            try:
                ds = self.budget_tracker.get_day_state(today_et)
                if ds is not None and ds['loss_announced_at'] is None:
                    await channel.send(embed=self._loss_limit_embed(summary))
                    self.budget_tracker.set_day_flag(today_et, 'loss_announced_at')
                    logger.info(f"Announced daily loss limit for {today_et}")
            except Exception as e:
                logger.error(f"loss-limit announcement failed: {e}")

        acted = summary['entries'] or summary['exits']
        gap = float(os.getenv('FAST_SUMMARY_MINUTES', 30))
        # 0 (or less) disables the idle summary entirely: report only when the
        # bot actually did something. Without this guard a gap of 0 makes every
        # single 60s cycle "due" and floods the channel.
        due = gap > 0 and (self._last_fast_summary is None or
                           (datetime.now() - self._last_fast_summary).total_seconds() / 60 >= gap)

        if acted:
            try:
                await channel.send(embed=self._fast_action_embed(summary))
            except Exception as e:
                logger.error(f"fast action report failed: {e}")
        elif due and summary['state'] == 'open':
            try:
                await channel.send(embed=self._fast_idle_embed(summary))
                self._last_fast_summary = datetime.now()
            except Exception as e:
                logger.error(f"fast summary failed: {e}")

    def _loss_limit_embed(self, s):
        limit = float(os.getenv('DAILY_LOSS_LIMIT_PCT', 3))
        e = discord.Embed(
            title="🛑 Daily loss limit hit — no new entries today",
            description=(f"Balance is down {limit:g}% or more from this morning's "
                         f"start, so the bot stops opening positions until the next "
                         f"ET date. Exits still run for anything open."),
            color=discord.Color.red(),
            timestamp=datetime.now().astimezone())
        self._add_account_fields(e, s)
        e.set_footer(text="AUTO INTRADAY · PAPER TRADING — no broker, no real money")
        return e

    def _fast_action_embed(self, s):
        """Every number here is the ledger's: fill, amount, fees and P&L come
        from the executed row, never recomputed from the quote."""
        e = discord.Embed(
            title="⚡ Intraday activity",
            color=discord.Color.gold(),
            timestamp=datetime.now().astimezone())
        for x in s['exits']:
            verdict = "🟩 profit" if x['pnl'] > 0 else "🟥 loss" if x['pnl'] < 0 else "flat"
            e.add_field(
                name=f"SOLD {qty_str(x['shares'])} {x['symbol']} @ ${x['price']:,.2f}",
                value=(f"Why: {x['reason']} [{x['exit_reason']}]\n"
                       f"Result: **{_signed_usd(x['pnl'])}** net ({x['pct']:+.2%} ref-to-ref; "
                       f"gross {_signed_usd(x['gross'])}, fees ${x['fees']:,.2f}) — {verdict}"),
                inline=False)
        for x in s['entries']:
            cls = asset_class(x['symbol'])
            e.add_field(
                name=f"BOUGHT {qty_str(x['shares'])} {x['symbol']} @ ${x['price']:,.2f}",
                value=(f"Cost ${x['cost']:,.2f} (ref ${x['ref_price']:,.2f}, "
                       f"fees ${x['fees']:,.2f}) · model confidence {x['probability']:.1%}\n"
                       f"Will sell on **+{barriers(cls)[0]:.1%}** or "
                       f"**−{barriers(cls)[1]:.1%}** from the reference price"
                       + ("" if is_crypto(x['symbol']) else ", or before the close.")),
                inline=False)
        self._add_account_fields(e, s)
        e.add_field(name="Minutes to close",
                    value=f"{s.get('minutes_to_close', 0):.0f}", inline=True)
        gates = self._gate_lines()
        if gates:
            e.add_field(name="Class gates", value=gates, inline=False)
        e.set_footer(text="AUTO INTRADAY · PAPER TRADING — no broker, no real money")
        return e

    def _fast_idle_embed(self, s):
        positions = self.budget_tracker.get_positions()
        e = discord.Embed(
            title="⚡ Intraday check — no trades",
            color=discord.Color.greyple(),
            timestamp=datetime.now().astimezone())
        cands = s.get('candidates') or []
        e.add_field(
            name="What I looked at",
            value=(f"{len(self.intraday.symbols)} liquid names on "
                   f"{self.intraday.__class__.__module__.split('.')[-1]} "
                   f"{os.getenv('INTRADAY_INTERVAL', '5m')} bars.\n"
                   f"Selection bar: **p > {s.get('bar', 0):.3f}**. "
                   f"{len(cands)} cleared it."),
            inline=False)
        if cands:
            e.add_field(name="Closest candidates",
                        value="\n".join(f"**{c['symbol']}** {c['probability']:.1%} "
                                         f"· ${c['price']:,.2f}" for c in cands[:3]),
                        inline=False)
        e.add_field(
            name="Open positions",
            value=("\n".join(f"**{p['symbol']}** x{qty_str(p['shares'])} @ "
                              f"${p['avg_price']:,.2f}" for p in positions)
                   if positions else "none"),
            inline=False)
        gates = self._gate_lines()
        if gates:
            e.add_field(name="Class gates", value=gates, inline=False)
        if s.get('loss_tripped'):
            e.add_field(name="🛑 Daily loss limit",
                        value="Tripped for today — no new entries until the next ET "
                              "date; exits still run.",
                        inline=False)
        if s.get('note'):
            e.add_field(name="Note", value=s['note'], inline=False)
        e.set_footer(text=f"{s.get('minutes_to_close', 0):.0f} min to close · "
                          f"buying power ${s.get('buying_power', 0):,.2f} · PAPER TRADING")
        return e

    @tasks.loop(minutes=float(os.getenv('CHECK_INTERVAL_MINUTES', 60)))
    async def monitor_trading(self):
        """Continuous monitoring loop"""
        channel = await self._destination()
        if not channel:
            return
        
        logger.info(f"⏰ Monitoring at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        symbols = await self._symbols()
        max_alerts = int(os.getenv('MAX_ALERTS_PER_CYCLE', 3))
        min_prob = float(os.getenv('MIN_PROBABILITY', 0.55))

        # Refresh prices first - otherwise every cycle after startup would score
        # the same frozen bars and emit identical signals forever. yfinance is
        # blocking, so keep it off the event loop.
        try:
            await asyncio.to_thread(self.engine.fetch_and_store_data, symbols)
        except Exception as e:
            logger.error(f"Price refresh failed, scoring stale data: {e}")

        # Rank the entire universe, then alert only the strongest few - with a
        # 500-name universe, alerting everything over the threshold would be
        # hundreds of DMs an hour.
        try:
            # Rank the WHOLE universe - no limit here. Truncating first meant
            # the top-N were all high-confidence SELLs, the filter below dropped
            # every one, and genuine BUY candidates ranked below the cut were
            # never even considered. Scanning all 503 costs ~5s.
            ranked = await asyncio.to_thread(
                self.engine.scan, symbols, min_prob, None
            )
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return

        # A SELL is only actionable on something we actually hold - there is no
        # short side to this paper portfolio. Without this filter a market-wide
        # down day produces hundreds of unactionable SELL alerts, because the
        # pooled model gives every symbol the same regime read.
        held = {p['symbol'] for p in self.budget_tracker.get_positions()}
        actionable = [s for s in ranked if s['signal'] == 1 or s['symbol'] in held]
        dropped = len(ranked) - len(actionable)
        if dropped:
            logger.info(f"Filtered {dropped} SELL signal(s) on unheld symbols")

        alerting = [] if self.fast_mode else actionable[:max_alerts]
        if self.fast_mode and actionable:
            logger.info(f"Daily loop is report-only in FAST_MODE "
                        f"({len(actionable)} daily signals not traded)")

        if os.getenv('HEARTBEAT', '1') not in ('0', 'false', 'no'):
            try:
                pnl = await asyncio.to_thread(
                    self.budget_tracker.get_pnl, self.engine.latest_price)
                await channel.send(embed=self._heartbeat_embed(
                    symbols, ranked, actionable, alerting, held, min_prob, pnl))
            except Exception as e:
                logger.error(f"Heartbeat send failed: {e}")

        if not actionable:
            logger.info(f"No actionable signal cleared {min_prob:.0%} this cycle "
                        f"({len(ranked)} ranked, all unheld SELLs)")
            return

        for signal in alerting:
            try:
                await self.send_trade_alert(channel, signal)
            except Exception as e:
                logger.error(f"Error processing {signal['symbol']}: {e}")

    async def send_trade_alert(self, channel, signal):
        """Send trade alert and wait for approval.

        Sizing is the account's (size_order): a fraction of equity capped by
        buying power, fractional quantities allowed. A SELL closes the whole
        held position - there is no partial manual exit and no short side.
        """
        symbol = signal['symbol']
        side = 'BUY' if signal['signal'] == 1 else 'SELL'
        signal_type = "🟢 BUY" if side == 'BUY' else "🔴 SELL"
        price = signal['price']          # reference quote; the ledger applies slippage/fees
        prob = signal['probability']
        bt = self.budget_tracker

        if side == 'BUY':
            qty, size_usd, est_fill = bt.size_order(symbol, price)
            if qty == 0:
                # Under MIN_ORDER_USD. Say so rather than silently skipping every
                # signal until proceeds settle.
                embed = discord.Embed(
                    title=f"⚠️ SKIPPED: {symbol}",
                    description="Insufficient buying power",
                    color=discord.Color.orange())
                # No cycle summary here, so this quotes held symbols (HTTP) -
                # keep it off the event loop like every other latest_price call.
                await asyncio.to_thread(self._add_account_fields, embed)
                await channel.send(embed=embed)
                return
        else:
            pos = next((p for p in bt.get_positions() if p['symbol'] == symbol), None)
            if pos is None:
                logger.info(f"{symbol}: SELL signal but nothing held - skipped")
                return
            qty = pos['shares']

        est = costs.fill(symbol, side, price, qty)
        trade_id = bt.log_trade(symbol, side, price, qty,
                                probability=prob if side == 'BUY' else None,
                                exit_reason='manual' if side == 'SELL' else None)

        # AUTO_TRADE: execute straight away and report, rather than waiting on a
        # reaction. Still paper - this writes to the ledger, not to a broker.
        if os.getenv('AUTO_TRADE', '0') in ('1', 'true', 'yes'):
            row = bt.execute_trade(trade_id)
            if row is None:
                logger.error(f"AUTO {side} {symbol}: trade #{trade_id} was not pending")
                return
            done = discord.Embed(
                title=f"{'🟢 BOUGHT' if side == 'BUY' else '🔴 SOLD'} "
                      f"{qty_str(row['shares'])} {symbol} @ ${row['price']:,.2f}",
                description=(f"Automatic — no approval needed. "
                             f"{'Cost' if side == 'BUY' else 'Proceeds'} ${row['amount']:,.2f} "
                             f"(ref ${row['ref_price']:,.2f}, fees ${row['fees']:,.2f})."),
                color=discord.Color.green() if side == 'BUY' else discord.Color.red(),
                timestamp=datetime.now().astimezone(),
            )
            done.add_field(name="Confidence", value=f"{prob:.1%}", inline=True)
            if side == 'SELL':
                done.add_field(name="Realised", value=_signed_usd(row['realized_pnl']),
                               inline=True)
            await asyncio.to_thread(self._add_account_fields, done)   # quotes held symbols
            pos = next((p for p in bt.get_positions() if p['symbol'] == symbol), None)
            if pos:
                done.add_field(name="You now hold",
                               value=f"{qty_str(pos['shares'])} @ avg ${pos['avg_price']:,.2f}",
                               inline=True)
            done.set_footer(text="AUTO MODE · PAPER TRADING — no broker, no real money")
            try:
                await channel.send(embed=done)
            except Exception as e:
                logger.error(f"Auto-trade report failed for {symbol}: {e}")
            logger.info(f"AUTO {side} {qty_str(row['shares'])} {symbol} @ "
                        f"${row['price']:.2f} (trade #{trade_id})")
            return

        # Create alert embed - the fill is an estimate until execute_trade runs.
        embed = discord.Embed(
            title=f"{signal_type} {symbol}",
            description="Waiting for approval...",
            color=discord.Color.green() if side == 'BUY' else discord.Color.red()
        )
        embed.add_field(name="Ref price", value=f"${price:,.2f}", inline=True)
        embed.add_field(name="Est. fill", value=f"${est['fill_price']:,.2f}", inline=True)
        embed.add_field(name="Qty", value=qty_str(qty), inline=True)
        embed.add_field(name="Est. amount",
                        value=f"${est['net']:,.2f} (fees ${est['fees']:,.2f})", inline=True)
        embed.add_field(name="Confidence", value=f"{prob:.2%}", inline=True)
        # Spec section 4: Balance beside Buying power on every money embed,
        # the approval one included. (Buying power already reflects this
        # trade's PENDING hold.) Quotes held symbols - off the event loop.
        await asyncio.to_thread(self._add_account_fields, embed)
        embed.add_field(name="Approval Timeout", value="5 minutes", inline=True)
        embed.set_footer(text=f"Trade ID: {trade_id} | ✅ approve / ❌ reject "
                              f"| PAPER TRADING - no broker connected")

        # Send message. If delivery fails the trade must not stay PENDING -
        # it would hold buying power forever for an alert nobody ever saw.
        try:
            msg = await channel.send(embed=embed)
            await msg.add_reaction('✅')
            await msg.add_reaction('❌')
        except Exception as e:
            bt.reject_trade(trade_id)
            logger.error(f"Alert delivery failed for {symbol} - "
                         f"rolled back pending trade #{trade_id}: {e}")
            return

        # Store for tracking
        self.pending_approvals[trade_id] = {
            'message': msg,
            'symbol': symbol,
            'signal': signal['signal'],
            'price': price,
            'shares': qty,
            'amount': est['net'],
            'timestamp': datetime.now()
        }

        # Wait for approval
        await self.wait_for_approval(trade_id, channel)
    
    async def wait_for_approval(self, trade_id, channel):
        """Wait for user approval via reactions"""
        start_time = datetime.now()
        msg = self.pending_approvals[trade_id]['message']
        timeout = 300  # 5 minutes
        
        while True:
            try:
                msg = await msg.channel.fetch_message(msg.id)
                
                for reaction in msg.reactions:
                    if reaction.emoji == '✅':
                        async for user in reaction.users():
                            if user.id == int(os.getenv('USER_ID')):
                                await self.execute_approved_trade(trade_id, channel)
                                return
                    
                    elif reaction.emoji == '❌':
                        async for user in reaction.users():
                            if user.id == int(os.getenv('USER_ID')):
                                await self.reject_trade(trade_id, channel)
                                return
                
                # Check timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    await self.auto_reject_trade(trade_id, channel)
                    return
                
                await asyncio.sleep(5)
            
            except Exception as e:
                logger.error(f"Error waiting for approval: {e}")
                break
    
    async def execute_approved_trade(self, trade_id, channel):
        """Execute approved trade"""
        trade_data = self.pending_approvals[trade_id]
        row = self.budget_tracker.execute_trade(trade_id)

        if row is None:
            # Already decided (or cleared by a restart) - report, never re-book.
            embed = discord.Embed(
                title=f"⚠️ Trade #{trade_id} could not be executed",
                description="It was no longer pending.",
                color=discord.Color.orange())
        else:
            embed = discord.Embed(
                title=f"📝 PAPER TRADE RECORDED: {trade_data['symbol']}",
                description="Logged to the paper ledger - **no broker order was placed**",
                color=discord.Color.brand_green()
            )
            embed.add_field(name="Fill", value=f"${row['price']:.2f}")
            embed.add_field(name="Qty", value=qty_str(row['shares']))
            embed.add_field(name="Amount",
                            value=f"${row['amount']:.2f} (fees ${row['fees']:.2f})")
            if row['side'] == 'SELL':
                embed.add_field(name="Realised", value=_signed_usd(row['realized_pnl']))
            await asyncio.to_thread(self._add_account_fields, embed)   # quotes held symbols

        await channel.send(embed=embed)
        del self.pending_approvals[trade_id]
        logger.info(f"📝 Trade {trade_id} recorded to paper ledger")
    
    async def reject_trade(self, trade_id, channel):
        """Reject trade"""
        trade_data = self.pending_approvals[trade_id]
        self.budget_tracker.reject_trade(trade_id)
        
        embed = discord.Embed(
            title=f"❌ REJECTED: {trade_data['symbol']}",
            color=discord.Color.brand_red()
        )
        embed.add_field(name="Would have cost", value=f"${trade_data['amount']:.2f}")
        
        await channel.send(embed=embed)
        del self.pending_approvals[trade_id]
        logger.info(f"❌ Trade {trade_id} rejected")
    
    async def auto_reject_trade(self, trade_id, channel):
        """Auto-reject after timeout"""
        trade_data = self.pending_approvals[trade_id]
        self.budget_tracker.reject_trade(trade_id)
        
        embed = discord.Embed(
            title=f"⏱️ EXPIRED: {trade_data['symbol']}",
            description="Approval timeout - trade rejected",
            color=discord.Color.greyple()
        )
        
        await channel.send(embed=embed)
        del self.pending_approvals[trade_id]
    
    # ═══════════════════════════════════════════════════════════════
    # DISCORD COMMANDS
    # ═══════════════════════════════════════════════════════════════
    
    async def _embed_status(self):
        data = await asyncio.to_thread(self.budget_tracker.get_pnl, self.engine.latest_price)
        embed = discord.Embed(title="💼 Portfolio Status", color=discord.Color.blue())
        embed.add_field(name="Balance", value=f"${data['equity']:,.2f}")
        embed.add_field(name="Buying power", value=f"${data['buying_power']:,.2f}")
        embed.add_field(name="Unsettled", value=f"${data['unsettled']:,.2f}")
        embed.add_field(name="Open positions", value=str(len(data['positions'])))
        embed.add_field(name="Pending Trades", value=str(len(self.pending_approvals)))
        return embed
    
    async def _embed_daily_brief(self, now=None):
        """/daily_brief: the model narrates a facts block Python built
        (scorecard, positions, earnings dates, stored headlines). It is
        never asked what is happening in the market - it has no way to know."""
        now = now or datetime.now(timezone.utc)
        day_et = now.astimezone(ET).strftime('%Y-%m-%d')
        context = await asyncio.to_thread(
            build_brief_context, self.budget_tracker, self.engine, self.intraday,
            day_et, self.db.conn, now)
        brief = await self.claude.daily_market_analysis(context)

        embed = discord.Embed(
            title=f"📊 Daily brief — {day_et}",
            description=brief,
            color=discord.Color.blue()
        )
        embed.set_footer(text=f"Narrated by {self.claude.backend_name} from the bot's own numbers")
        return embed

    async def _embed_risk_check(self):
        pnl = await asyncio.to_thread(self.budget_tracker.get_pnl, self.engine.latest_price)
        tp, sl, _ = barriers('stock')
        context = build_risk_context(
            pnl, max_positions=(self.fast.max_positions if self.fast
                                else int(os.getenv('FAST_MAX_POSITIONS', 3))),
            stop_loss_pct=sl, take_profit_pct=tp,
            daily_loss_limit_pct=float(os.getenv('DAILY_LOSS_LIMIT_PCT', 3)))
        risk_analysis = await self.claude.analyze_portfolio_risk(pnl['positions'], context)

        embed = discord.Embed(
            title="⚠️ Risk Assessment",
            description=risk_analysis,
            color=discord.Color.orange()
        )
        embed.set_footer(text=f"Narrated by {self.claude.backend_name} from the bot's own numbers")
        return embed
    
    async def _embed_account(self):
        """/account: the paper brokerage account, read-only. There is no setter
        - changing starting cash after open would corrupt the all-time return."""
        bt = self.budget_tracker
        data = await asyncio.to_thread(bt.get_pnl, self.engine.latest_price)
        embed = discord.Embed(
            title="🏦 Paper account",
            description=f"{bt.account_type()} account — starting cash is fixed at open, "
                        f"there is no setter",
            color=discord.Color.blue())
        embed.add_field(name="Balance", value=f"${data['equity']:,.2f}")
        embed.add_field(name="Cash", value=f"${data['cash']:,.2f}")
        embed.add_field(name="Unsettled", value=f"${data['unsettled']:,.2f}")
        embed.add_field(name="Buying power", value=f"${data['buying_power']:,.2f}")
        embed.add_field(name="Starting cash", value=f"${data['starting_cash']:,.2f}")
        embed.add_field(name="Opened", value=bt.opened_at())
        embed.add_field(name="All-time",
                        value=f"{_signed_usd(data['all_time_net'])} ({data['all_time_pct']:+.2%})")
        if data['stale']:
            embed.add_field(name="⚠️ Stale quotes (valued at cost)",
                            value=", ".join(data['stale']), inline=False)
        embed.set_footer(text="PAPER TRADING - no broker connected")
        return embed
    
    async def _embed_pnl(self, now=None):
        """/pnl and /summary: today's scorecard. Never writes day_state or
        equity_history - only the 16:05 tick does that. `now` (UTC, tz-aware)
        is for tests; the slash commands call this with no arguments."""
        now = now or datetime.now(timezone.utc)
        day_et = now.astimezone(ET).strftime('%Y-%m-%d')
        return await asyncio.to_thread(self._scorecard_embed, day_et, now)

    async def _embed_scan(self, top: int = 10):
        top = max(1, min(top, 20))
        min_prob = float(os.getenv('MIN_PROBABILITY', 0.55))
        symbols = await self._symbols()
        try:
            ranked = await asyncio.to_thread(
                self.engine.scan, symbols, min_prob, top
            )
        except Exception as e:
            return discord.Embed(title="Scan failed", description=str(e),
                                 color=discord.Color.red())

        if not ranked:
            return discord.Embed(
                title="No signals",
                description=f"Nothing cleared the {min_prob:.0%} confidence bar right now.",
                color=discord.Color.greyple())

        embed = discord.Embed(
            title=f"🔎 Top {len(ranked)} by model confidence",
            description="Ranked across the full universe - **suggestions, not advice**",
            color=discord.Color.blurple(),
        )
        bp = self.budget_tracker.get_buying_power()
        min_order = float(os.getenv('MIN_ORDER_USD', 1))
        for i, sig in enumerate(ranked, 1):
            side = "🟢 BUY" if sig['signal'] == 1 else "🔴 SELL"
            embed.add_field(
                name=f"{i}. {sig['symbol']} - {side}",
                value=(f"${sig['price']:,.2f} | confidence {sig['probability']:.1%} "
                       f"| as of {sig['as_of']}"),
                inline=False,
            )
        sides = {s['signal'] for s in ranked}
        if len(sides) == 1 and len(ranked) > 3:
            embed.add_field(
                name="⚠️ One-sided",
                value=("Every result is the same direction - the pooled model is "
                       "making a single market-wide call, not picking names. "
                       "Treat this as one bet, not a diversified list."),
                inline=False,
            )
        # Fractional shares mean price never gates a buy; only buying power does.
        if bp < min_order:
            embed.add_field(
                name="⚠️ Insufficient buying power",
                value=(f"${bp:,.2f} is under the ${min_order:,.2f} minimum order - "
                       f"nothing can be bought until proceeds settle."),
                inline=False)
        embed.set_footer(text=f"Buying power ${bp:,.2f} | "
                              f"PAPER TRADING - model edge is ~1pp, treat as a shortlist")
        return embed

    async def _embed_pause(self):
        """Pause trading"""
        self.monitor_trading.stop()
        embed = discord.Embed(
            title="⏸️ Trading Paused",
            color=discord.Color.orange()
        )
        return embed
    
    async def _embed_resume(self):
        """Resume trading"""
        self.monitor_trading.start()
        embed = discord.Embed(
            title="▶️ Trading Resumed",
            color=discord.Color.green()
        )
        return embed
    
    def run(self, token):
        """Start the bot"""
        self.bot.run(token)

# Standalone bot runner
if __name__ == '__main__':
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix='!', intents=intents)
    trading_bot = TradingBot(bot_instance=bot)
    trading_bot.run(os.getenv('DISCORD_TOKEN'))
