# Hugheylab Trading Bot

> **Want to read the whole bot in one place?** [`docs/CODE.md`](docs/CODE.md) has every source file on a single page, with a table of contents.

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
| **Ledger** | SQLite — one simulated cash account, trades with modelled fills and fees, positions, T+1 settlement, daily equity history, signal log |

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
| Take profit | +1.0% stocks / +0.6% crypto (tuned, `data/tuned.json`) |
| Stop loss | −0.6% stocks / −0.4% crypto (tuned) |
| End-of-day flatten | 10 min before the bell |
| Max hold | label horizon × bar interval (24 × 5m = 120 min); not a setting |
| Re-entry cooldown | 15 min |
| Max concurrent positions | 3 (`FAST_MAX_POSITIONS`) |

Stock trading happens only during the regular session (09:30–16:00 ET,
weekdays). **Crypto (`TRADE_CRYPTO=1`) trades 24/7** — it produces ~3.7× the bars
per calendar day, is tradeable when the stock market is shut, and is exempt from
the end-of-day flatten since there is no close to flatten into.

### Separate models per asset class

Stocks and crypto get their own model, their own selection bar, and their own
barriers — they have different volatility regimes, and one pooled model served
neither. Measured on the same data:

| | Precision | Break-even | EV/trade |
|---|---|---|---|
| Pooled (both classes, one model) | 47.9% | 38.5% | +0.122% |
| **Stock model** (79 symbols, 289k rows) | **92.6%** | 36.8% | **+1.059%** |
| **Crypto model** (13 coins, 172k rows) | **66.0%** | 38.5% | **+0.358%** |

The pooled number was an average hiding a strong stock model and a weak crypto
one. Symbols are ranked by *margin over their own class bar*, since raw
probabilities are not comparable across models with different base rates.

**Two gates decide whether a class may open positions** (`class_gate` in
`src/fast_trader.py`, evaluated in this order): a *cost gate* — the take-profit
must exceed the class's round-trip cost, never bypassed — and an *EV gate* —
the backtested EV must clear `MIN_EV_TO_TRADE`, bypassed by `FAST_IGNORE_EV=1`
for the paper run (see "Paper account" below). `/fast` and the startup notice
name a blocked class and why. Exits still run on a blocked class, so an open
position is never stranded.

### Tuning (`tune.py`)

"Train it more" is not how this improves; extra iterations on the same data
memorise noise. `tune.py` instead trains many honest configurations and keeps
the one that wins out of sample:

```bash
./venv/bin/python tune.py --wide      # 80 stocks + 13 coins, 36 configs each
```

Every candidate is scored on held-out EV with a purged, embargoed split, and any
config with fewer than 300 test signals is rejected — 100% precision on 40
signals is overfitting wearing a nice suit. Winners are written to
`data/tuned.json` and picked up automatically on the next train.

The 2026-09-08 sweep moved stocks from +0.278% to **+1.059%** EV/trade. Two
things drove it: shallower, more heavily regularised trees (`max_depth=3`,
`l2=2.0` beat every deeper config — the deep ones were overfitting), and a wider
universe (80 symbols → 289k rows vs 30 symbols → 112k). Since 5m history is
capped at 60 days, breadth is the only way to add data.

**Caveat worth keeping in mind:** 92.6% precision is measured on a single 60-day
window at a high selection bar. Walk-forward validation across several windows
is the real test, and is not yet implemented.

### Paper account

Paper mode models one **$500 retail cash brokerage account** (Robinhood-style:
zero stock commissions, fractional shares, a per-side crypto markup) closely
enough that two months of results are a fair basis for a go/no-go decision on
real money. Every assumption is a setting (table at the end of this section).

**Account.** One row in the `account` table, seeded from `STARTING_CASH` on the
first start and never reset — there is no setter, because changing starting
cash mid-run would corrupt the all-time return. Cash falls on BUY by
fill × qty + fees and rises on SELL by fill × qty − fees. Equity = cash + market
value of open positions. Every report counts only trades with
`created_at >= account.opened_at`; older rows stay in the DB but are ignored.

**Buying power** = cash − unsettled proceeds − pending BUYs. With
`ACCOUNT_TYPE=cash` (default) a stock sale's proceeds are unsettled until the
next trading day's 09:30 ET (T+1, weekends and `US_HOLIDAYS_2026` skipped);
crypto settles immediately. `ACCOUNT_TYPE=margin` removes the wait and changes
nothing else.

**Fills and costs** (`src/costs.py`, applied in exactly one place,
`BudgetTracker.log_trade`):

| | Fill | Fees |
|---|---|---|
| Stock | ref × (1 ± `STOCK_SLIPPAGE_BPS`/1e4), 5 bps default | BUY none; SELL `SEC_FEE_RATE` × proceeds + min(`FINRA_TAF_PER_SHARE` × qty, `FINRA_TAF_CAP`) |
| Crypto | ref × (1 ± `CRYPTO_SPREAD_BPS`/1e4) per side, 60 bps default | none |

Round-trip cost: stocks ≈ 0.1%, crypto 1.2%. The crypto take-profit (0.6%) is
below its round-trip cost, so the cost gate blocks crypto entries at the
defaults; lower `CRYPTO_SPREAD_BPS` for a cheaper venue and it re-enables.

**Sizing.** Every entry is `min(buying power, equity / FAST_MAX_POSITIONS)`
dollars as a fractional quantity, skipped below `MIN_ORDER_USD`. Wins compound;
losses shrink the next order.

**Barriers are measured reference-to-reference** — the quote at entry against
the quote now, the same move the labels use — so costs show up in cash and P&L,
never in the stop trigger. Max hold is the label horizon × bar interval
(24 × 5m = 120 min); `FAST_MAX_HOLD_MIN` no longer exists.

**Daily loss limit.** Once equity is down `DAILY_LOSS_LIMIT_PCT` (3%) from the
day's starting equity, no new positions open for the rest of the ET day. Exits
still run. Announced once on Discord.

**Daily report.** At 16:05 ET every day (weekends included) the bot labels the
day's logged signals, writes the day's `equity_history` row, and posts one
scorecard: all-time P&L with n and a 95% CI, a GO / NO-GO / EXTEND verdict per
class, gross P&L and fees, balance, buying power, unsettled cash, today's
trades, open positions, win rate with a Wilson CI, profit factor, max drawdown,
per-class realised vs backtested precision, and SPY buy-and-hold on the same
starting cash as context. `/pnl` and `/summary` show the same scorecard on
demand.

**Decision rule** (`src/scorecard.py: verdict`, pre-registered and frozen at
`account.opened_at`; review 2026-11-13). Per class, with cost-adjusted
breakeven `be_c = (sl + c) / (tp + sl)` and `[lo, hi]` the Wilson 95% CI of the
live signal TP-first rate (n ≈ 40 signals/day — the executed-trade sample is
too small to decide anything):

- **NO-GO** if `hi < be_c`, or the cost gate blocks the class.
- **GO** if `lo > be_c`, the day-clustered lower bound also clears `be_c`, at
  least 60 trades executed, and the realised mean net return is within one SE
  of the backtested EV net of costs.
- **EXTEND** otherwise — treated as NO-GO for real money until it turns GO.

`FAST_IGNORE_EV=1` (set for the paper run) lets a class trade on paper even
when its backtested EV is below `MIN_EV_TO_TRADE`: the point of the run is to
test whether live behaviour matches the backtest, not to overturn it. The cost
gate is never bypassed.

| Variable | Default | Notes |
|---|---|---|
| `STARTING_CASH` | `500` | seeded once on first start; no setter |
| `ACCOUNT_TYPE` | `cash` | `cash` = T+1 stock settlement; `margin` = none |
| `STOCK_SLIPPAGE_BPS` | `5` | per side |
| `CRYPTO_SPREAD_BPS` | `60` | per side; round trip 120 bps |
| `SEC_FEE_RATE` | `0.0000206` | stock sells, on proceeds |
| `FINRA_TAF_PER_SHARE` | `0.000195` | stock sells, capped by `FINRA_TAF_CAP` (`9.79`) |
| `MIN_ORDER_USD` | `1` | smallest order |
| `DAILY_LOSS_LIMIT_PCT` | `3` | account-wide, per ET day |
| `FAST_IGNORE_EV` | `0` | `1` on the paper run |

## ⛔ Read this first: this model has no measured edge

An external review on 2026-09-08 found a **look-ahead leak** that produced every
impressive number this project previously reported. `build_features()` computed

```python
prev_close = close.groupby(session).transform('last').shift(1)   # BUG
```

`transform('last')` stamps every bar with its **own session's final close**, so
for all but the first bar `gap_open = day_open / today's_final_close - 1`.
Combined with `from_open`, the model was handed `final_close / current_close` —
the answer. Permutation importance after the fact: `gap_open` 0.36, `from_open`
0.28, everything else ≤0.016.

**With the leak removed, on walk-forward validation:**

| | claimed (leaked) | honest |
|---|---|---|
| stock | 92.6% precision, EV +1.059%/trade | **31.5%** vs 37.5% break-even, EV **−0.096%** |
| crypto | 66.0% precision, EV +0.358%/trade | **40.0%** vs 40.0% break-even, EV **+0.000%** |

Against a real round-trip cost of 5–40 bps (stocks) and 22–100+ bps (retail
crypto), **no configuration in this feature set is profitable.** Both classes are
auto-blocked by `MIN_EV_TO_TRADE`. The bot runs, scans, and reports — and
correctly declines to trade.

### The methodological lesson

**Walk-forward validation would NOT have caught this.** The leaky model scored
88.8–92.9% across six independent windows with a standard deviation of 1.3
points. A leak is *consistent*, not lucky — cross-validation made the fake number
look **more** credible, not less. What exposed it was feature-level inspection
and permutation importance. Suspiciously good results should trigger a leak
audit before a validation victory lap.

A second, subtler version of the same class of bug was also found: 44% of
take-profit labels were hit in a **later session**, but the executor flattens
before the bell — so the model was trained to chase outcomes it is forbidden to
capture. `triple_barrier(..., session=...)` now stops the forward walk at the
session boundary for stocks (crypto, which holds overnight, runs the full
horizon).

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
| `/status` | Balance, buying power and open positions |
| `/scan [top]` | Rank the whole universe right now |
| `/pnl` | The scorecard now: all-time P&L, verdicts, positions marked to market |
| `/account` | Balance, cash, unsettled, buying power, starting cash, opened |
| `/daily_brief` | Claude market commentary (optional) |
| `/risk_check` | Claude risk read on open positions (optional) |
| `/retrain` | Refresh all sources and retrain (~4 min) |
| `/fast` | Intraday model stats, exit rules, gates, and live scores |
| `/summary` | The same scorecard the 16:05 ET report posts |
| `/sources` | Where market data is coming from, and last refresh counts |
| `/pause` · `/resume` | Stop or restart the loop |

### Hourly heartbeat

Every hour it posts a plain-English report, so silence is never ambiguous:

- **💰 Your money** — balance vs starting cash, buying power, unsettled cash,
  what your holdings are worth, up/down in dollars and percent
- **📊 What you own** — per position: what you paid, what it's worth now, gain/loss
- **👉 What to do** — in plain words: react to an alert, or nothing, and why
- **🔍 What I checked** — how many symbols, from where, how many were buys vs sells

Set `HEARTBEAT=0` to turn it off.

### Daily report

At 16:05 ET every day (weekends included, so a quiet Saturday still gets an
equity row) it labels the day's logged signals, records the day's equity, and
posts the scorecard described under "Paper account". A failed post is retried
every 5 minutes; `day_state.report_posted_at` guarantees at most one post per
date, even across the nightly restart. `/summary` shows it on demand.

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
| `STARTING_CASH` | `500` | seeded into the paper account on first start; no setter |
| `ACCOUNT_TYPE` | `cash` | `cash` = T+1 stock settlement; `margin` = none |
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
- Pending trades hold buying power. They are rolled back if the alert fails to
  send, and cleared at startup, since their approval watcher lives in memory.

## License

MIT
