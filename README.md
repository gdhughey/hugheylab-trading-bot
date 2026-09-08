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
| **Data** | Multi-source chain — see below |
| **Model** | Pooled `GradientBoostingClassifier` over 10 ratio/z-score features |
| **Cadence** | Hourly: refresh prices → score all 503 → alert the top N |
| **Approval** | Discord ✅ / ❌ reactions, or `AUTO_TRADE=1` to execute and report |
| **Ledger** | SQLite — trades, positions, realized P&L |

## Data sources

Two independent chains, each tried in order:

**History** (`DATA_SOURCES`) — each provider only sees symbols the previous one
missed, so rate-limited sources are spent on real gaps:

| Source | Key | Notes |
|---|---|---|
| `yahoo` | no | Batched `yfinance`; ~9 requests for 503 symbols |
| `yahoo_direct` | no | Yahoo chart API per symbol; recovers names `yfinance` drops mid-batch |
| `alphavantage` | yes | Genuinely independent, but **25 requests/day** free — gap-fill only |
| `tiingo` | yes | Independent daily bars |

**Live quotes** (`QUOTE_SOURCES`) — used for mark-to-market and to detect stale
history:

| Source | Key | Notes |
|---|---|---|
| `finnhub` | yes | Free tier serves quotes; **historical candles return 403** |
| `yahoo_quote` | no | Keyless fallback |

Measured cross-check (2026-09-04): Finnhub live quotes agreed with stored Yahoo
closes to within **0.06%** across AAPL/MSFT/NVDA/CRWD/GLW. A divergence over 15%
is logged as probable stale history.

Note that `yahoo` and `yahoo_direct` are the same upstream data by two code
paths — redundancy against library bugs, not a second opinion on price. Real
cross-validation needs a keyed source.

Stooq was evaluated and rejected: it serves a JavaScript proof-of-work challenge
and cannot be used without a browser.

## Speed limits

Measured on 2 cores / 1 GB:

| Universe | Refresh | Scan | Minimum cycle |
|---|---|---|---|
| 20 symbols | 2.9s | 0.2s | **~3s** |
| 503 symbols | 70.3s | 5.4s | **~76s** |

Live quotes run ~0.13s/symbol, so one pass over 503 names is ~1.1 min.

The *daily* model cannot trade faster than daily, regardless of cycle speed —
it predicts next-day direction, so polling faster re-emits the same answer until
the next close. That is what the intraday engine below exists for.

## Intraday mode (`FAST_MODE=1`)

A second engine trained on **5-minute bars** over 60 days, predicting whether a
name moves more than **0.15% within 30 minutes** — a question whose answer
actually changes during the session.

**It has roughly double the edge of the daily model:**

| Model | Accuracy | Baseline | Edge |
|---|---|---|---|
| Daily (next-day direction) | 0.523 | 0.510 | **+0.013** |
| Intraday (5m, 30-min horizon) | 0.699 | 0.671 | **+0.029** |

Trained on 111,666 rows across 30 liquid names; a full refresh takes ~6s, so a
60-second loop is comfortable.

### Labeling: the fix that mattered most

The original label asked *"is price higher in 30 minutes?"* But the executor
exits at **+TP or −SL, whichever comes first**. Those are different questions.
If price first drops through the stop and only then rallies, the old label
called it a **win** while the real trade was a **loss** — the model was being
trained to chase outcomes it could not capture.

`src/labeling.py` implements **triple-barrier labeling** (López de Prado): walk
the forward path bar by bar using highs and lows, and label by which barrier is
touched first. Training target and live outcome are now the same question.

Training also uses a **purged split** — a label at bar *i* depends on bars
*i+1…i+horizon*, so without an embargo gap the last training rows peek into the
test window and the score is inflated.

### What the honest label revealed

Measured sweep, 2026-09-08, 60 days of 5m bars:

| TP / SL | Horizon | Base rate | Precision | Break-even | EV/trade |
|---|---|---|---|---|---|
| +1.5% / −1.0% | 30 min | 6.4% | 35.5% | 40.0% | **−0.113%** |
| +0.8% / −0.5% | 120 min | 30.5% | 90.0% | 38.5% | **+0.670%** |

**The original +1.5%/−1.0% over 30 minutes was the worst configuration tested —
negative expected value on both stocks and crypto.** A 1.5% move inside 30
minutes happens on only 6.4% of stock bars (1.3% of crypto bars), while the 1.0%
stop is hit constantly. Defaults are now **+0.8% / −0.5% over 120 minutes**.

The metric that decides profitability is **precision against break-even**, not
accuracy: `EV = precision × TP − (1 − precision) × SL`, break-even at
`SL / (TP + SL)`. `/fast` reports it and warns when EV is negative.

**Threshold calibration matters.** Only ~36% of intraday bars are positive, so
the model's probabilities cluster near that base rate and a fixed 0.55 bar is
*never* cleared. The selection bar is therefore relative:
`p > base_rate x INTRADAY_PROB_RATIO` (default 1.15 → about 0.416).

**Exit rules** — an intraday entry without an exit is just buy-and-hold with
extra steps. Each cycle exits *before* entering, so a stop-loss is never delayed
and freed capital is reusable in the same pass:

| Rule | Default |
|---|---|
| Take profit | +1.5% |
| Stop loss | −1.0% |
| End-of-day flatten | 10 min before the bell |
| Max hold | 120 min |
| Re-entry cooldown | 15 min |
| Max concurrent positions | 3 |

Stock trading happens only during the regular session (09:30–16:00 ET,
weekdays). **Crypto (`TRADE_CRYPTO=1`) trades 24/7** — it produces ~3.7× the bars
per calendar day, is tradeable when the stock market is shut, and is exempt from
the end-of-day flatten since there is no close to flatten into.

**Known limitation:** stocks and crypto currently share one pooled model. On the
combined universe precision is 47.9% (EV +0.122%/trade over 20,184 held-out
signals); trained separately the sweep showed 90.0% for stocks and 72.9% for
crypto. Separate per-asset-class models are the obvious next improvement.

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
| `/retrain` | Refresh all sources and retrain (~4 min) |
| `/fast` | Intraday model stats, exit rules, and live scores |
| `/sources` | Where market data is coming from, and last refresh counts |
| `/pause` · `/resume` | Stop or restart the loop |

### Hourly heartbeat

Every hour it posts a plain-English report, so silence is never ambiguous:

- **💰 Your money** — budget, cash left, what your holdings are worth, up/down in
  dollars and percent
- **📊 What you own** — per position: what you paid, what it's worth now, gain/loss
- **👉 What to do** — in plain words: react to an alert, or nothing, and why
- **🔍 What I checked** — how many symbols, from where, how many were buys vs sells

Set `HEARTBEAT=0` to turn it off.

## Retraining manually

```bash
/opt/trading-bot/venv/bin/python /opt/trading-bot/train.py            # fetch + train
/opt/trading-bot/venv/bin/python /opt/trading-bot/train.py --no-fetch # cached data only
/opt/trading-bot/venv/bin/python /opt/trading-bot/train.py --scan     # also print top picks
/opt/trading-bot/venv/bin/python /opt/trading-bot/train.py --target forward_5d
```

Or `/retrain` from Discord.

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
