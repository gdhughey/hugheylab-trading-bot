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
from datetime import datetime
import logging
from src.claude_analyzer import ClaudeAnalyzer
from src.budget_tracker import BudgetTracker
from src.ml_engine import load_universe

logger = logging.getLogger(__name__)

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
        
        # Add cogs
        self.bot.add_listener(self.on_ready)
        self._register_slash()
    

    def _heartbeat_embed(self, symbols, ranked, actionable, alerting, held, min_prob):
        """Per-cycle report that spells out WHAT it looked at, WHERE the data and
        the money live, and HOW it decided - silence alone is ambiguous."""
        buys = [r for r in ranked if r['signal'] == 1]
        sells = [r for r in ranked if r['signal'] == 0]
        positions = self.budget_tracker.get_positions()
        quiet = not alerting
        univ = os.getenv('UNIVERSE', 'default')
        univ_label = 'S&P 500' if univ.lower() == 'sp500' else univ

        embed = discord.Embed(
            title="⏱️ Hourly check" + (" — nothing to act on" if quiet else
                                        f" — {len(alerting)} alert(s) below"),
            color=discord.Color.greyple() if quiet else discord.Color.green(),
            timestamp=datetime.now().astimezone(),
        )

        embed.add_field(
            name="🔍 WHAT I scanned",
            value=(f"All **{len(symbols)} {univ_label}** stocks — US large-cap only "
                   f"(NYSE/NASDAQ). Not the whole market: no small-caps, ETFs, "
                   f"crypto, options or non-US exchanges."),
            inline=False)

        embed.add_field(
            name="📡 WHERE the data came from",
            value=("Yahoo Finance **daily closing bars** (2y history), refreshed "
                   "at the top of this cycle. Not live intraday quotes — prices "
                   "are last close."),
            inline=False)

        embed.add_field(
            name="🧮 HOW I decided",
            value=(f"Scored every symbol with the ML model → **{len(ranked)}** cleared "
                   f"{min_prob:.0%} confidence (**{len(buys)} buy**, **{len(sells)} sell**).\n"
                   f"Dropped **{len(sells) - len([r for r in actionable if r['signal'] == 0])}** "
                   f"sells on stocks you don't own → **{len(actionable)}** actionable → "
                   f"alerting the top **{len(alerting)}**."),
            inline=False)

        if quiet:
            if sells and not buys:
                why = (f"Model called the market down: {len(sells)} sells, 0 buys. "
                       f"You can't sell what you don't own"
                       + (f" — you hold {len(positions)} position(s)." if positions
                          else " — and you hold nothing yet.")) 
            elif not ranked:
                why = f"Nothing reached {min_prob:.0%} confidence."
            else:
                why = ("Candidates existed but none fit the budget "
                       f"(${self.budget_tracker.get_remaining_budget():,.2f} left).")
            embed.add_field(name="🤔 WHY no alert", value=why, inline=False)

        top = (buys or sells)[:3]
        if top:
            embed.add_field(
                name=("🏆 Strongest buys" if buys else "🏆 Strongest sells (can't act — you own none)"),
                value="\n".join(
                    f"{'🟢' if t['signal'] == 1 else '🔴'} **{t['symbol']}** "
                    f"${t['price']:,.2f} · {t['probability']:.0%} confident"
                    for t in top),
                inline=False)

        pos_txt = ("none — nothing bought yet" if not positions else
                   ", ".join(f"{p['symbol']} x{p['shares']}" for p in positions[:6]))
        embed.add_field(
            name="💼 WHERE the money is",
            value=(f"Holdings: **{pos_txt}**\n"
                   f"Budget: **${self.budget_tracker.get_remaining_budget():,.2f}** left of "
                   f"${self.budget_tracker.weekly_budget:,.2f} this week\n"
                   f"⚠️ **Paper only** — trades are written to a local ledger. "
                   f"No broker is connected and no real money moves."),
            inline=False)

        embed.set_footer(text="Next check in 1 hour · /scan to rank now · /pnl for P&L")
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

        bind('status', 'Portfolio status and budget', self._embed_status)
        bind('stats', 'Trading statistics', self._embed_stats)
        bind('pnl', 'Mark the paper portfolio to market', self._embed_pnl)
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

        @tree.command(name='budget', description='Check or set the weekly budget')
        @app_commands.describe(amount='New weekly budget in dollars (omit to just check)')
        async def _budget(interaction: discord.Interaction, amount: float = None):
            await interaction.response.defer(thinking=True)
            try:
                embed = await self._embed_budget(amount)
            except Exception as e:
                logger.exception("/budget failed")
                embed = discord.Embed(title="/budget failed", description=str(e),
                                      color=discord.Color.red())
            await interaction.followup.send(embed=embed)

        logger.info("Registered 9 slash commands")

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

        # Startup fetch (~110s) and training (~130s) are synchronous. Run them
        # OFF the event loop - inline they block every interaction for minutes
        # and Discord answers slash commands with "application did not respond".
        # A PENDING trade's approval watcher lives in memory, so anything left
        # PENDING by a previous run can never be approved - it would just hold
        # budget forever. Clear them at startup.
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
        finally:
            self.warming_up = False
            logger.info("🟢 Ready - slash commands are live")
    
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

    @tasks.loop(hours=1)
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

        alerting = actionable[:max_alerts]

        if os.getenv('HEARTBEAT', '1') not in ('0', 'false', 'no'):
            try:
                await channel.send(embed=self._heartbeat_embed(
                    symbols, ranked, actionable, alerting, held, min_prob))
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
        """Send trade alert and wait for approval"""
        symbol = signal['symbol']
        signal_type = "🟢 BUY" if signal['signal'] == 1 else "🔴 SELL"
        price = signal['price']
        prob = signal['probability']
        
        # Calculate trade size
        budget_remaining = self.budget_tracker.get_remaining_budget()
        shares = min(
            int(budget_remaining * 0.2 / price),
            int(100000 * 0.1 / price)
        )

        # The 20% slice can't buy a single share of a $300 stock on a small
        # weekly budget, which would silently skip every signal forever. Fall
        # back to one share whenever the FULL remaining budget still covers it -
        # risk stays bounded by the budget, it just isn't self-blocking.
        if shares == 0 and price <= budget_remaining:
            shares = 1
            logger.info(f"{symbol}: 20%% slice under one share - sizing to 1 @ ${price:.2f}")

        if shares == 0:
            logger.info(f"{symbol}: skipped, ${price:.2f}/share exceeds "
                        f"${budget_remaining:.2f} remaining")
            return

        trade_amount = price * shares
        
        # Check budget
        if not self.budget_tracker.can_trade(trade_amount):
            embed = discord.Embed(
                title=f"⚠️ SKIPPED: {symbol}",
                description="Budget limit reached this week",
                color=discord.Color.orange()
            )
            embed.add_field(name="Weekly Budget", value=f"${self.budget_tracker.weekly_budget:,.2f}")
            embed.add_field(name="Spent", value=f"${self.budget_tracker.get_weekly_spent():.2f}")
            embed.add_field(name="Remaining", value=f"${budget_remaining:.2f}")
            await channel.send(embed=embed)
            return
        
        # Log pending trade
        trade_id = self.budget_tracker.log_trade(symbol, 'BUY' if signal['signal'] == 1 else 'SELL', price, shares)
        
        # Create alert embed
        embed = discord.Embed(
            title=f"{signal_type} {symbol}",
            description="Waiting for approval...",
            color=discord.Color.green() if signal['signal'] == 1 else discord.Color.red()
        )
        embed.add_field(name="Price", value=f"${price:.2f}", inline=True)
        embed.add_field(name="Shares", value=f"{shares}", inline=True)
        embed.add_field(name="Amount", value=f"${trade_amount:.2f}", inline=True)
        embed.add_field(name="Confidence", value=f"{prob:.2%}", inline=True)
        embed.add_field(name="Budget Remaining", value=f"${budget_remaining:.2f}", inline=True)
        embed.add_field(name="Approval Timeout", value="5 minutes", inline=True)
        embed.set_footer(text=f"Trade ID: {trade_id} | ✅ approve / ❌ reject "
                              f"| PAPER TRADING - no broker connected")
        
        # Send message. If delivery fails the trade must not stay PENDING -
        # it would hold budget forever for an alert nobody ever saw.
        try:
            msg = await channel.send(embed=embed)
            await msg.add_reaction('✅')
            await msg.add_reaction('❌')
        except Exception as e:
            self.budget_tracker.reject_trade(trade_id)
            logger.error(f"Alert delivery failed for {symbol} - "
                         f"rolled back pending trade #{trade_id}: {e}")
            return
        
        # Store for tracking
        self.pending_approvals[trade_id] = {
            'message': msg,
            'symbol': symbol,
            'signal': signal['signal'],
            'price': price,
            'shares': shares,
            'amount': trade_amount,
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
        self.budget_tracker.execute_trade(trade_id)
        
        embed = discord.Embed(
            title=f"📝 PAPER TRADE RECORDED: {trade_data['symbol']}",
            description="Logged to the paper ledger - **no broker order was placed**",
            color=discord.Color.brand_green()
        )
        embed.add_field(name="Price", value=f"${trade_data['price']:.2f}")
        embed.add_field(name="Shares", value=str(trade_data['shares']))
        embed.add_field(name="Amount", value=f"${trade_data['amount']:.2f}")
        embed.add_field(name="Remaining Budget", value=f"${self.budget_tracker.get_remaining_budget():.2f}")
        
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
        embed = discord.Embed(title="💼 Portfolio Status", color=discord.Color.blue())
        embed.add_field(name="Weekly Budget", value=f"${self.budget_tracker.weekly_budget:,.2f}")
        embed.add_field(name="Spent", value=f"${self.budget_tracker.get_weekly_spent():.2f}")
        embed.add_field(name="Remaining", value=f"${self.budget_tracker.get_remaining_budget():.2f}")
        embed.add_field(name="Pending Trades", value=str(len(self.pending_approvals)))
        return embed
    
    async def _embed_daily_brief(self):
        brief = await self.claude.daily_market_analysis()

        embed = discord.Embed(
            title="📊 Daily Market Brief",
            description=brief,
            color=discord.Color.blue()
        )
        embed.set_footer(text="Powered by Claude AI")
        return embed
    
    async def _embed_risk_check(self):
        risk_analysis = await self.claude.analyze_portfolio_risk(
            self.budget_tracker.get_positions())

        embed = discord.Embed(
            title="⚠️ Risk Assessment",
            description=risk_analysis,
            color=discord.Color.orange()
        )
        embed.set_footer(text="Powered by Claude AI")
        return embed
    
    async def _embed_budget(self, new_budget=None):
        if new_budget:
            # Must go through the tracker: it reads WEEKLY_BUDGET once at init,
            # so poking os.environ here would silently change nothing.
            if new_budget <= 0:
                return discord.Embed(title="Budget must be greater than zero.",
                                     color=discord.Color.red())
            self.budget_tracker.set_weekly_budget(new_budget)
            os.environ['WEEKLY_BUDGET'] = str(new_budget)
            embed = discord.Embed(
                title="💰 Budget Updated",
                description=f"Weekly budget set to ${new_budget:,.2f}",
                color=discord.Color.green()
            )
            embed.add_field(name="Spent", value=f"${self.budget_tracker.get_weekly_spent():,.2f}")
            embed.add_field(name="Remaining", value=f"${self.budget_tracker.get_remaining_budget():,.2f}")
        else:
            embed = discord.Embed(title="💰 Current Budget", color=discord.Color.blue())
            embed.add_field(name="Weekly Budget", value=f"${self.budget_tracker.weekly_budget:,.2f}")
            embed.add_field(name="Spent", value=f"${self.budget_tracker.get_weekly_spent():.2f}")
            embed.add_field(name="Remaining", value=f"${self.budget_tracker.get_remaining_budget():.2f}")
        
        return embed
    
    async def _embed_stats(self):
        stats_data = self.budget_tracker.get_statistics()
        
        embed = discord.Embed(title="📊 Trading Statistics", color=discord.Color.purple())
        for key, value in stats_data.items():
            embed.add_field(name=key, value=str(value), inline=True)
        
        return embed
    
    async def _embed_pnl(self):
        data = await asyncio.to_thread(
            self.budget_tracker.get_pnl, self.engine.latest_price)

        total = data['total']
        color = (discord.Color.green() if total > 0
                 else discord.Color.red() if total < 0 else discord.Color.greyple())
        embed = discord.Embed(
            title="📈 Paper P&L",
            description="Simulated - no broker order was ever placed",
            color=color,
        )
        embed.add_field(name="Realized", value=f"${data['realized']:,.2f}", inline=True)
        embed.add_field(name="Unrealized", value=f"${data['unrealized']:,.2f}", inline=True)
        embed.add_field(name="Total", value=f"${total:,.2f}", inline=True)

        if data['cost_basis']:
            embed.add_field(name="Cost Basis", value=f"${data['cost_basis']:,.2f}", inline=True)
            embed.add_field(name="Market Value", value=f"${data['market_value']:,.2f}", inline=True)
            embed.add_field(name="Return", value=f"{data['return_pct']:.2%}", inline=True)

        for pos in data['positions'][:10]:
            if pos['pnl'] is None:
                embed.add_field(name=pos['symbol'], value="price unavailable", inline=False)
            else:
                embed.add_field(
                    name=f"{pos['symbol']} x{pos['shares']}",
                    value=(f"avg ${pos['avg_price']:.2f} -> ${pos['price']:.2f}  "
                           f"**${pos['pnl']:+,.2f}** ({pos['pnl_pct']:+.2%})"),
                    inline=False,
                )

        if not data['positions']:
            embed.add_field(name="Open positions", value="none", inline=False)
        if data['stale']:
            embed.add_field(name="⚠️ Stale", value=", ".join(data['stale']), inline=False)

        embed.set_footer(text="PAPER TRADING - prices are the latest stored close")
        return embed

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
        remaining = self.budget_tracker.get_remaining_budget()
        for i, sig in enumerate(ranked, 1):
            side = "🟢 BUY" if sig['signal'] == 1 else "🔴 SELL"
            afford = "" if sig['price'] <= remaining else "  ⚠️ over budget"
            embed.add_field(
                name=f"{i}. {sig['symbol']} - {side}",
                value=(f"${sig['price']:,.2f} | confidence {sig['probability']:.1%} "
                       f"| as of {sig['as_of']}{afford}"),
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
        embed.set_footer(text=f"Remaining budget ${remaining:,.2f} | "
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
