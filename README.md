# Hugheylab Trading Bot

Scans the S&P 500 hourly with a gradient-boosting model, DMs you the strongest
candidates on Discord, and records a trade only after you react ✅.

> **This is paper trading.** No broker integration exists anywhere in this
> repository. Approving a trade writes a row to a local SQLite ledger and
> updates a simulated position. No real money can move.

## What it actually does

| | |
|---|---|
| **Universe** | 503 S&P 500 tickers (constituents scraped from Wikipedia, cached) |
| **Data** | Yahoo Finance daily bars, 2y history, refreshed every cycle |
| **Model** | Pooled `GradientBoostingClassifier` over 10 ratio/z-score features |
| **Cadence** | Hourly: refresh prices → score all 503 → alert the top N |
| **Approval** | Discord DM with ✅ / ❌ reactions, 5-minute timeout |
| **Ledger** | SQLite — trades, positions, realized P&L |

## Honest performance note

On 181,641 rows the model scores **0.524 holdout accuracy against a 0.510
baseline — an edge of about 1.4 percentage points.** That is inside what
spread and slippage would consume in real trading.

Two things follow, and they are the whole design:

1. **The approval gate is the strategy, not the model.** The bot's job is to
   narrow 503 names to a handful you actually look at.
2. **A pooled model makes one market-wide call, not stock picks.** Every symbol
   shares the same regime features on a given day, so it is common for 300+ of
   503 to clear the confidence bar in the same direction. `/scan` says so
   explicitly when results are one-sided. Treat a one-sided list as one bet.

Sell signals on stocks you do not own are filtered out — there is no short side.

## Slash commands

| Command | What it does |
|---|---|
| `/status` | Budget and open positions |
| `/scan [top]` | Rank the whole universe right now |
| `/pnl` | Mark the paper portfolio to market |
| `/budget [amount]` | Check or set the weekly budget |
| `/stats` | Executed / rejected counts, approval rate, realized P&L |
| `/daily_brief` | Claude market commentary (optional) |
| `/risk_check` | Claude risk read on open positions (optional) |
| `/pause` · `/resume` | Stop or restart the hourly loop |

Every hour it also posts a heartbeat saying what it scanned, where the data came
from, how it decided, and why it did or didn't alert — so silence is never
ambiguous.

## Install

**On a Proxmox host** (creates the LXC for you):

```bash
bash proxmox/create-lxc.sh 200 192.168.1.247/24 192.168.1.1
```

**On any Debian/Ubuntu box or existing LXC:**

```bash
git clone https://github.com/gdhughey/hugheylab-trading-bot.git
cd hugheylab-trading-bot
sudo bash install.sh
/opt/trading-bot/configure.sh
sudo systemctl enable --now trading-bot
```

First start takes ~4 minutes — it downloads ~250k price rows and trains before
the first scan.

## Discord setup

1. Create an app at <https://discord.com/developers/applications>.
2. **Bot → Reset Token** — this is `DISCORD_TOKEN`.
3. Invite it with scopes `bot applications.commands` and permissions `85056`
   (View Channels, Send Messages, Embed Links, Read History, **Add Reactions**).
4. **You must share a server with the bot.** Discord refuses to deliver a DM
   otherwise (`50278: no mutual guilds`) — a private server with just you works.
   The bot never posts there.

Message Content Intent is **not** required; commands are slash commands.

## Configuration

All via `.env` (see `.env.example`):

| Variable | Default | Notes |
|---|---|---|
| `DISCORD_TOKEN` | — | required |
| `USER_ID` | — | required; who may approve trades |
| `CHANNEL_ID` | *blank* | blank = DM you |
| `WEEKLY_BUDGET` | `5000` | small budgets skip expensive stocks; a 1-share floor applies |
| `UNIVERSE` | `default` | `sp500`, `default` (5 tickers), or a comma list |
| `MIN_PROBABILITY` | `0.55` | confidence bar |
| `MAX_ALERTS_PER_CYCLE` | `3` | caps DMs per hour |
| `TARGET_MODE` | `next_day` | or `forward_5d` (5-day return > 2.5%) |
| `HEARTBEAT` | `1` | hourly status report |
| `CLAUDE_API_KEY` | *blank* | optional; Claude commentary degrades gracefully |
| `CLAUDE_MODEL` | `claude-opus-5` | |

## Notes for anyone extending this

- **Never call blocking code inside `async def`.** `yfinance`, training, and the
  Wikipedia fetch all block; run them through `asyncio.to_thread`. When the loop
  stalls, Discord answers every slash command with "the application did not
  respond". Check with `journalctl -u trading-bot | grep -i "heartbeat blocked"`.
- Slash handlers must `defer()` immediately — Discord kills an unacknowledged
  interaction after 3 seconds.
- Pending trades hold budget. They are rolled back if the alert fails to send,
  and cleared at startup, since their approval watcher lives in memory.

## License

MIT
