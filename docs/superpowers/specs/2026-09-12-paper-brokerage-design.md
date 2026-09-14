# Paper brokerage account — design (REVISED 2026-09-12)

## Goal

Make paper mode mimic a real $500 retail cash brokerage account closely enough
that two months of results (account opens Mon 2026-09-14, review Fri
2026-11-13, 45 trading days) are a fair basis for a go/no-go decision on real
money. The user has not chosen a broker; assume a Robinhood-style starter
account (zero stock commissions, fractional shares, ~0.6% per-side crypto
markup) and make every assumption a setting.

Go-live assumes a cash account (at Robinhood: switch the default Instant
account to Cash before funding; Fidelity and Schwab default to cash).

Also: one Discord message every day with all-time paper P&L and the current
go/no-go verdict (section 8).

The current retrain (2026-09-12) already scores both classes below breakeven on
11,566 (stock) / 2,636 (crypto) walk-forward signals. The paper run's job is to
test whether live behaviour matches that backtest, not to overturn it with ~100
executed trades. Section 8 says how.

## Non-goals

- No broker integration. Still paper only.
- No tax, dividend or cash-interest modelling.
- No change to the ML models, features, labels or barrier values.
- No PDT simulation (the run is a cash account; margin mode only removes the
  settlement wait).

## 1. Account (replaces the weekly budget)

New `account` table (single row):

```
account(id INTEGER PRIMARY KEY CHECK (id = 1),
        opened_at TEXT NOT NULL,          -- UTC ISO
        starting_cash REAL NOT NULL,
        cash REAL NOT NULL,
        account_type TEXT NOT NULL)       -- 'cash' | 'margin'
```

Seeded by `Database._migrate()` with `INSERT OR IGNORE ... VALUES (1, now,
STARTING_CASH, STARTING_CASH, ACCOUNT_TYPE)`. `STARTING_CASH=500`,
`ACCOUNT_TYPE=cash` in `.env`. There is no setter; changing starting cash later
would corrupt the all-time return.

Cash decreases on BUY by `amount` (fill × qty + fees), increases on SELL by
`amount` (fill × qty − fees). Never resets. Equity = cash + market value of open
positions. Position size is a fraction of equity (section 2), so wins compound
and losses shrink the account.

Buying power = cash − unsettled proceeds − `amount` of PENDING BUY rows.

`ACCOUNT_TYPE=cash` (default):
- A stock SELL's proceeds are unsettled until the next trading day's 09:30 ET
  (T+1, skipping weekends and `US_HOLIDAYS_2026`). Unsettled cash cannot be
  spent.
- Crypto proceeds settle immediately.

`ACCOUNT_TYPE=margin`: no settlement wait. Nothing else.

Settlement is stored, not recomputed: `trades.available_at` (UTC ISO) is set
when a SELL executes, to `next_trading_day_open(created_at)` for stocks and to
`created_at` for crypto. Unsettled = `SUM(amount) FROM trades WHERE side='SELL'
AND status='EXECUTED' AND available_at > now_utc AND created_at >=
account.opened_at`. The existing `trades.settled_at` column keeps its current
meaning (status-decided timestamp) and is not reused.

New helpers in `src/intraday_engine.py`: `is_trading_day(d: date) -> bool`
(weekday < 5 and not in `US_HOLIDAYS_2026`) and `next_trading_day_open(ts) ->
datetime` (first trading day strictly after `ts` in ET, at 09:30 ET, returned
tz-aware). `market_state()` and the "Stocks resume" line in
`_daily_summary_embed` use them. Add `'2027-01-01'` to `US_HOLIDAYS_2026`.

`WEEKLY_BUDGET`, `BUDGET_MODE`, `BudgetTracker.weekly_budget`,
`set_weekly_budget`, `get_weekly_spent`, `get_turnover`,
`get_remaining_budget`, `can_trade`, `_sum`, `_deployed`, `_committed` are
deleted. `configure.sh` prompts "Starting cash [500]" and writes
`STARTING_CASH`. The `/budget` command is replaced by `/account` (balance,
cash, unsettled, buying power, starting cash, opened_at; no parameters).

New `BudgetTracker` reads: `get_cash()`, `get_unsettled(now=None)`,
`get_buying_power(now=None)`, `get_equity(prices: dict) -> float` (cash +
Σ shares × prices[symbol]; a symbol missing from `prices` is valued at
`avg_price` and listed as stale).

Trades executed before `account.opened_at` stay in the DB. Every report, sum
and count filters `created_at >= account.opened_at`. The one exception is
`FastTrader._recover_entry_times`, which is not filtered (a held position must
keep its max-hold clock). The positions table is flat at open (it already is).

Every mutating `BudgetTracker` method (`log_trade`, `execute_trade`,
`reject_trade`, `ensure_day_state`, `set_day_flag`, `record_equity`) takes one
`threading.Lock` and runs its SQL in one transaction opened with
`BEGIN IMMEDIATE`, so a buying-power check and the cash debit are atomic
across the fast-cycle worker thread and the event-loop thread.

## 2. Fills and costs

New module `src/costs.py`:

```
fill(symbol, side, ref_price, qty) -> {'fill_price', 'fees', 'gross', 'net'}
round_trip_cost(cls) -> float          # fraction of notional, see below
```

- Stocks: commission 0. Fill = ref × (1 + STOCK_SLIPPAGE_BPS/1e4) on BUY,
  ref × (1 − STOCK_SLIPPAGE_BPS/1e4) on SELL. `STOCK_SLIPPAGE_BPS=5`.
  BUY fees 0. SELL fees = `SEC_FEE_RATE × proceeds + min(FINRA_TAF_PER_SHARE
  × qty, FINRA_TAF_CAP)`, unrounded. Defaults `SEC_FEE_RATE=0.0000206`,
  `FINRA_TAF_PER_SHARE=0.000195`, `FINRA_TAF_CAP=9.79` (2026 schedules).
- Crypto: fill = ref × (1 ± CRYPTO_SPREAD_BPS/1e4) against the trader, per
  side. `CRYPTO_SPREAD_BPS=60` is the PER-SIDE markup (Coinbase Advanced base
  taker tier; Robinhood's lowest tier is 0.55–0.6%). Round trip = 120 bps. No
  other fee.
- `gross` = fill × qty. `net` = gross + fees (BUY) or gross − fees (SELL).
- `round_trip_cost('stock')` = 2 × STOCK_SLIPPAGE_BPS/1e4 + SEC_FEE_RATE;
  `round_trip_cost('crypto')` = 2 × CRYPTO_SPREAD_BPS/1e4.

`costs.fill` is a pure function; it is called in exactly one place,
`BudgetTracker.log_trade`. No caller applies costs itself.

Trade row (new columns in section 6): `ref_price` (the quote the caller
passed), `price` (= fill), `fees`, `amount` (net cash movement), `shares`
(REAL, rounded to 6 dp), `trade_date` (ET date of `created_at`),
`entry_probability` (BUY rows, from `sig['probability']`), `exit_reason`
(SELL rows: `'tp' | 'sl' | 'timeout' | 'eod' | 'manual'`), `available_at`
(SELL rows, section 1), `gross_pnl` and `realized_pnl` (SELL rows).

New signature:
`log_trade(symbol, side, ref_price, qty, *, probability=None, exit_reason=None) -> int`.
It rounds `qty` to 6 dp, calls `costs.fill`, and inserts the row as PENDING
with `week_key` still stamped (legacy NOT NULL column, unread).

`execute_trade(trade_id)`, inside the locked transaction: set
`status='EXECUTED'`, `settled_at=now`; apply the position; on BUY
`UPDATE account SET cash = cash - amount`; on SELL `cash = cash + amount`,
set `available_at`, `realized_pnl = amount − closed × avg_price`,
`gross_pnl = realized_pnl + fees` (buy-side fees are zero in this model, so
`avg_price` equals the average buy fill). Returns the executed row (sqlite
Row) so callers report the ledger's fill, amount, fees and P&L instead of
recomputing them. `FastTrader.cycle` summary entries carry `cost=row['amount']`,
`price=row['price']`; exits carry `pnl=row['realized_pnl']`,
`gross=row['gross_pnl']`, `fees=row['fees']`, `price=row['price']`.

Positions: `positions.shares` REAL; new `positions.entry_ref REAL NOT NULL
DEFAULT 0` = quantity-weighted `ref_price` of the open lots. `_apply_position`
on BUY: `entry_ref = (held × entry_ref + qty × ref_price) / new_shares`; on
partial SELL unchanged; when flat, 0. `avg_price` stays the net cost basis
(from `amount`). After every update, `|shares| < 1e-6` is written as 0 and a
negative result logs the existing warning and writes 0. `get_positions()`
selects `shares > 1e-9` and returns `shares` as float, plus `entry_ref` and
`avg_price` un-rounded.

**Barriers are measured reference-to-reference**, the same move the labels
use (`labeling.triple_barrier` places barriers at entry close × (1 ± barrier)).
`FastTrader.cycle` computes `change = (ref_quote − pos['entry_ref']) /
pos['entry_ref']` for the stop/take-profit test and for the reported `pct`.
`avg_price` is used only for realised/unrealised P&L, cost basis, "paid vs
now" and equity. Costs therefore show up in cash and P&L, never in the
trigger. (Without this, a crypto BUY at ref R has avg_price 1.006R and the
0.4% crypto stop fires on the next 60 s cycle at an unchanged price.)

Max hold is derived from the label horizon per class:
`max_hold_min(cls) = barriers(cls)[2] × interval_minutes` (24 × 5 = 120 today).
`FAST_MAX_HOLD_MIN` is deleted from `.env` and the code. The exit loop passes
`exit_reason` from the branch that fired: EOD flatten `'eod'`, stop `'sl'`,
take-profit `'tp'`, max hold `'timeout'`.

Sizing (`BudgetTracker.size_order(symbol, ref_price) -> (qty, size_usd, est_fill)`):
`size_usd = min(buying_power, equity / FAST_MAX_POSITIONS)`; return
`qty = 0` if `size_usd < MIN_ORDER_USD` (`MIN_ORDER_USD=1`);
`est_fill = costs.fill(symbol, 'BUY', ref_price, 1)['fill_price']`;
`qty = round(size_usd / est_fill, 6)`. Orders are dollar-based, so the cash
debit equals `size_usd ≤ buying_power` by construction. Equity for sizing is
`get_equity(prices)` where `prices` is the dict of quotes the same cycle's
exit pass fetched for held symbols (no second quote). This deliberately
replaces `cash / free_slots`: every entry is the same fraction of equity
regardless of free slots. The `int()` casts and the 1-share floor are deleted
in `fast_trader.py` and `discord_bot.send_trade_alert`; the latter's
20%-of-budget slice and `100000 * 0.1` cap are deleted, it calls
`size_order`, and its SELL sells the whole held quantity (skip if none held)
with `exit_reason='manual'`. Skip reasons in the cycle summary: `'below min
order'`, `'insufficient buying power'`, `'daily loss limit'`, `'class gated:
<text>'`.

One quantity formatter `qty_str(x)` in `src/costs.py`
(`f"{x:.6f}".rstrip('0').rstrip('.')`) is used by every log line and embed
that prints a quantity (budget_tracker 112; fast_trader 182, 246; discord_bot
103, 107, 312, 509, 614, 623, 713, 719, 756, 887, 900, 907, 917, 997, 1128).

## 3. Trade regardless of the EV gate; hard gates that remain

`FAST_IGNORE_EV=1` in `.env`. It changes only whether the EV gate blocks
entries; every place that renders an EV verdict must honour it.

One helper in `src/fast_trader.py`:
`class_gate(cls, metrics) -> (tradeable: bool, text: str)`, evaluated in this
order:

1. Cost gate (not bypassed by `FAST_IGNORE_EV`): `tp, sl, _ = barriers(cls)`;
   if `tp <= costs.round_trip_cost(cls)` → `(False, "blocked: take-profit
   {tp:.2%} is below the {cost:.2%} round-trip cost")`. With defaults this
   blocks crypto (0.6% ≤ 1.2%) and allows stocks (1.0% > 0.1%). A user on a
   cheaper venue lowers `CRYPTO_SPREAD_BPS` and crypto re-enables.
2. EV gate: `ev < MIN_EV_TO_TRADE` and `FAST_IGNORE_EV` unset →
   `(False, existing "Not trading" text)`.
3. `ev < MIN_EV_TO_TRADE` and `FAST_IGNORE_EV` set → `(True, "trading on paper
   despite EV {ev*100:+.3f}% below the {floor:.2%} floor (FAST_IGNORE_EV on)")`.
4. Otherwise `(True, "Trading. Edge clears the {floor:.2%} cost floor")`.

Used at: `FastTrader.cycle` (`tradeable()` and the skip reason), the on_ready
log (discord_bot.py:422-429), `_send_startup_notice` (:468-491, including the
`will_trade` list that picks headline and colour; "Nothing will be traded"
only when every class is gated), and the daily report's per-class block. The
gate result is logged once at startup per class. Exits always run for a
gated class. `TRADE_CRYPTO=1` stays so crypto signals are still scored and
logged (section 7) while the cost gate keeps crypto out of the executed
record.

Daily loss limit `DAILY_LOSS_LIMIT_PCT=3`, account-wide. New table

```
day_state(date TEXT PRIMARY KEY,          -- ET date
          start_equity REAL NOT NULL,
          loss_tripped_at TEXT, loss_announced_at TEXT, report_posted_at TEXT)
```

At the top of each `FastTrader.cycle`, after the exit pass has fetched quotes:
`INSERT OR IGNORE INTO day_state(date, start_equity) VALUES (today_ET,
equity)`, so a same-date restart reuses the original baseline. Before the
entry loop: if `equity <= start_equity × (1 − DAILY_LOSS_LIMIT_PCT/100)` and
`loss_tripped_at IS NULL`, write it. While `loss_tripped_at` is set for
today's ET date, every entry is skipped with reason `'daily loss limit'`.
The cycle summary carries `loss_tripped: bool`; `discord_bot` announces it
once (sends, then sets `loss_announced_at`; skipped when already set). The
block ends by virtue of the new date key. Note: unrealised P&L is measured
against `avg_price`, so a freshly opened position contributes ≈ −(per-side
cost) × its value immediately; that is expected, not a barrier hit.

## 4. Daily report and scorecard

`equity_history` table: one row per ET date, written by the 16:05 ET tick
only (not by `/pnl`):

```
equity_history(date TEXT PRIMARY KEY, cash REAL NOT NULL,
               positions_value REAL NOT NULL, equity REAL NOT NULL,
               fees_to_date REAL NOT NULL, realized_to_date REAL NOT NULL,
               recorded_at TEXT NOT NULL)
```

Scheduler: the existing `daily_summary` 5-minute loop, started in `on_ready`
unconditionally (not only when intraday training succeeds). Each tick, with
`today = datetime.now(ET).date()`: if `now_ET.time() >= 16:05` and
`day_state[today].report_posted_at IS NULL` (row created with
`INSERT OR IGNORE` if absent, `start_equity` = current equity):

1. `signal_log.label_pending(conn)` (section 7).
2. Upsert today's `equity_history` row. This happens before any Discord call.
3. Build the scorecard embed(s) and send. On success set `report_posted_at`.
   `_destination()` returning `None` and any exception are logged at ERROR
   with the date; the next tick retries. There is no 16:30 upper bound, no
   weekday check, and `_last_daily_summary` is deleted.
4. On `discord.HTTPException`, send a one-line text message with the all-time
   headline and the error, and still set `report_posted_at`.

`build_scorecard(budget, engine, intraday, day) -> dict` lives in new
`src/scorecard.py` with `wilson_ci(hits, n)` and `mean_ci(values)` (mean,
SE, 95% normal interval with z = 1.96 — scipy is not a declared dependency, and
at the n ≥ 60 the GO rule requires t(0.975, 59) = 2.00 differs by 2%).
`discord_bot._scorecard_embed(day)` renders it and is
used by the 16:05 report, `/pnl` and `/summary`. `/stats` is removed
(`get_statistics` deleted). The "Registered N slash commands" log count is
updated. "Today" everywhere means the ET date (`trades.trade_date`).

Report contents, in order (each a field; list fields truncated to 1000 chars
with "… and N more"):

- **All-time: +$X.XX (+Y.YY%)** = equity − starting_cash, since `opened_at`,
  with `n=<closed trades>, 95% CI ±$<1.96 × SE × n>`.
- **Verdict** per class from section 8: GO / NO-GO / EXTEND, with the
  numbers that decided it.
- Gross P&L, total fees paid, balance (equity) vs starting cash, cash,
  buying power, unsettled cash with "settles <date> 09:30 ET".
- Today: realised, unrealised, trades, wins/losses; daily-loss block tripped
  or not.
- Closed today: `qty symbol @ fill → net (gross, fees) [exit_reason]`, max 10.
- Open positions: `qty symbol @ avg → now (pct vs entry_ref)`.
- Scorecard since open: trades closed, win rate with Wilson 95% CI, mean net
  per trade ± 95% CI, profit factor, max drawdown (from `equity_history` plus
  `starting_cash` as day 0), days running (calendar days since `opened_at`).
- Per class: n closed; exits by reason (count, mean net); realised TP-first
  rate = tp/(tp+sl+timeout+eod) with Wilson CI beside walk-forward precision
  `metrics[cls]['precision']` (n=`test_signals`) — the like-for-like pair;
  backtested EV (`metrics[cls]['ev']`, printed net of `round_trip_cost`)
  beside realised mean net per trade (%) ± CI; `class_gate` text; the signal
  calibration line from section 7.
- SPY buy-and-hold on the same starting cash over the same period, labelled
  "context only, not risk-matched: SPY is exposed 24/7, the bot is flat
  overnight". Computed at report time from the `prices` table: first SPY
  close with `date >= opened_at[:10]` vs `engine.stored_close('SPY')`.
- Footer caveat: "Backtest scores timeouts/EOD as −sl, assumes exact-barrier
  fills, and samples intrabar highs/lows; live exits are checked on one quote
  every FAST_POLL_SECONDS."

SPY: `BENCHMARK_SYMBOLS = ['SPY']` in `src/ml_engine.py`;
`fetch_and_store_data` builds `pending = [s for s in BENCHMARK_SYMBOLS if s not
in self.symbols] + list(self.symbols)` without touching `self.symbols`;
`_stored_symbols()` excludes `BENCHMARK_SYMBOLS`.

Label changes in every embed: "Cash left", "Remaining", "Remaining Budget",
"Budget Remaining" → "Buying power"; add "Balance" (equity) beside it in
`_send_startup_notice`, `_fast_action_embed` (plus "Unsettled"),
`send_trade_alert`, `execute_approved_trade`, `_embed_status`. "Weekly
Budget"/"Spent" fields are removed. The SKIPPED embed in `send_trade_alert`
becomes "Insufficient buying power" with Balance / Buying power / Unsettled.
`/scan` warns only when `buying_power < MIN_ORDER_USD`; footer "Buying power
$X". Heartbeat copy (discord_bot 86-95, 126) reads "Balance $equity (started
with $starting_cash)", "Buying power $bp", "Unsettled $x". BOUGHT/SOLD and
approval embeds show the ledger row's fill, amount and fees (approval embed
shows the estimated fill from `costs.fill`, labelled "est."). `get_pnl`
gains keys `cash`, `unsettled`, `buying_power`, `equity`, `starting_cash`,
`all_time_net`, `all_time_pct`, `fees_paid`, `gross_pnl`; `return_pct` is
removed.

## 5. Testing

pytest under `tests/`, run via `dev/ct-test.sh` (pushes the tree to LXC 200,
runs in the production venv with `DB_PATH=:memory:`). Because each
`connect(':memory:')` is a separate empty database, every test builds a file
DB under `tmp_path` and passes the same path to `Database(path)`,
`BudgetTracker(path)` and any engine. Clock-dependent code takes a `now=`
argument; tests never rely on wall-clock time.

- costs: buy/sell fill direction, fee math (stocks sell-side only, crypto
  none), `round_trip_cost` per class, `qty_str`.
- settlement: stock SELL on a Friday 15:55 ET → `available_at` = Monday
  09:30 ET (UTC), buying power excludes proceeds until then; a SELL on
  2026-11-25 settles 2026-11-27; crypto immediate; margin immediate.
- sizing: fractional qty; `size_usd < MIN_ORDER_USD` → no order; from a
  fresh $500 account filling all `FAST_MAX_POSITIONS` slots, every net debit
  ≤ buying power with slippage applied and cash ≥ 0 (stock and crypto);
  an entry when buying power binds (unsettled proceeds leave it below
  equity / FAST_MAX_POSITIONS) fills in full at buying power; compounding
  after a win.
- barriers ref-to-ref: crypto BUY at ref 100 (fill 100.60) → cycle at ref
  100 no exit; at 99.61 no exit; at 99.60 stop fires, fills 99.60 × 0.994,
  `realized_pnl` per unit = fill − 100.60 (≈ −1.19% of cost). Stock BUY at
  ref 100 (fill 100.05) → 99.45 no stop; 99.40 stop; 100.95 no TP; 101.00
  TP. A flat-ref position survives `max_hold_min` worth of cycles without a
  barrier exit and then exits with `exit_reason='timeout'`.
- realised P&L net vs gross; `gross_pnl = realized_pnl + fees`; `entry_ref`
  weighting across two BUYs; partial SELL leaves `entry_ref` unchanged.
- class gate: with `FAST_IGNORE_EV=1`, `CRYPTO_SPREAD_BPS=60` and crypto tp
  0.006 the crypto class is refused; at `CRYPTO_SPREAD_BPS=10` it is allowed;
  stocks allowed at defaults. With `FAST_IGNORE_EV=1`, `MIN_EV_TO_TRADE=0.003`
  and metrics `{stock: ev=-0.0003, crypto: ev=0.0006}` the startup embed
  says "LIVE", no field contains "Not trading", colour green; with the flag
  unset the same inputs yield "Nothing will be traded".
- daily loss limit: trips at −3% of `start_equity`, blocks entries, exits
  still run; a new `BudgetTracker`/`FastTrader` on the same file stays
  blocked and does not re-announce; resets on the next ET date; a restart on
  the same date keeps the original `start_equity`.
- equity history + max drawdown (including `starting_cash` as day 0).
- daily report: fires on a Saturday; posts once per date across a restart
  inside the window; a failed send leaves `report_posted_at` NULL and the
  next tick posts; the `equity_history` row exists for a date whose send
  failed; `/pnl` before 16:05 does not suppress the report.
- migration: a file DB written with the OLD schema (INTEGER shares, no
  account/day_state/equity_history/signals tables, no new columns) and 21
  legacy trades upgrades in place; a fractional BUY then round-trips as
  float; old trades are excluded from every report and from unsettled; the
  account row exists once after two `Database()` constructions.
- signal log: two cycles on the same 5m bar write one row per symbol;
  `label_pending` labels a stock signal 1 when TP is touched first, 0 on
  stop/horizon/session end, leaves NULL when bars are insufficient.
- scorecard: `wilson_ci`; `mean_ci` on a synthetic +0.5%/trade ledger of 100
  trades excludes zero and shrinks with n; verdict function returns GO /
  NO-GO / EXTEND on constructed inputs.

Then a live check on the container: first cycle enters, exits, settles, the
signals table fills, and the 16:05 report posts with a verdict line.

## 6. Migration and restart recovery

Schema changes in `src/database.py` (`SCHEMA` text updated for fresh DBs;
`_migrate()` upgrades existing ones):

- `trades.shares` and `positions.shares` are declared REAL in `SCHEMA`.
  Existing databases are NOT rebuilt: SQLite INTEGER affinity stores 0.5 as
  REAL losslessly (verified on 3.46.1), and the only truncation was the
  Python `int()` casts, which are deleted. Readers wrap `shares` in
  `float()`. `week_key` keeps its NOT NULL and keeps being stamped.
- Additive, PRAGMA-guarded `ALTER TABLE ... ADD COLUMN`:
  `trades.ref_price REAL`, `trades.fees REAL NOT NULL DEFAULT 0`,
  `trades.gross_pnl REAL`, `trades.available_at TEXT`,
  `trades.trade_date TEXT`, `trades.entry_probability REAL`,
  `trades.exit_reason TEXT`, `positions.entry_ref REAL NOT NULL DEFAULT 0`.
  Backfill `positions.entry_ref = avg_price WHERE shares > 0 AND entry_ref = 0`.
- New tables `account`, `day_state`, `equity_history`, `signals` via
  `CREATE TABLE IF NOT EXISTS` in `SCHEMA`; `account` seeded in `_migrate()`.
- `Database()` runs before any other connection (main.py:63), so
  `BudgetTracker` assumes the tables exist.

Restart safety. The service restarts nightly at 03:30 CT
(`trading-bot-restart.timer`) and on OOM (`MemoryMax=1200M`). Every piece of
account state has a durable home: settlement in `trades.available_at`;
start-of-day equity, the loss-limit latch, its announcement and the report
guard in `day_state`; equity in `account.cash` + `positions`; entry times
recovered by `_recover_entry_times` (unchanged); signal log in `signals`.
Buying power is always derived from the ledger, never cached.
`FastTrader.cooldown` stays memory-only (accepted).

## 7. Signal log

New table, written by `FastTrader.cycle` on every cycle for every scored
symbol (call `engine.scan_all()`; candidates = rows with `above_bar`, ranked
by `margin` as today):

```
signals(id INTEGER PRIMARY KEY AUTOINCREMENT,
        bar_ts TEXT NOT NULL,        -- UTC ISO of the 5m bar scored
        symbol TEXT NOT NULL, asset_class TEXT NOT NULL,
        probability REAL NOT NULL, bar REAL NOT NULL,
        above_bar INTEGER NOT NULL, ref_price REAL NOT NULL,   -- bar close
        trade_date TEXT NOT NULL,    -- ET date
        executed_trade_id INTEGER,   -- BUY row id when this signal was traded
        label INTEGER, labeled_at TEXT,
        UNIQUE (symbol, bar_ts))
```

`IntradayEngine.signal()` adds `bar_ts` (the newest bar's UTC ISO). Inserts
use `INSERT OR IGNORE` on `(symbol, bar_ts)`, so the 60 s poll cannot inflate
n while the same 5m bar is current.

`src/signal_log.py`: `record(conn, signals)` and `label_pending(conn, now)`.
`label_pending` takes every row with `label IS NULL`, loads `prices_intraday`
bars for the symbol from `bar_ts` forward, and applies
`labeling.triple_barrier` with the class's `barriers()` tp/sl/horizon and,
for stocks, `session` = ET date (same rule as `tune.py dataset()`). A label
that is NaN (window runs past the last stored bar) stays NULL. Runs at step 1
of the 16:05 tick.

Report line per class (section 4): `n` above-bar signals labelled, TP-first
rate with Wilson 95% CI on raw n, and a day-clustered interval (mean of
per-day precision ± 1.96 × SD/√days), against the cost-adjusted breakeven.

## 8. Decision rule (pre-registered, frozen at `account.opened_at`)

Definitions per class, using the tuned barriers at the time of the report:
- `c` = `costs.round_trip_cost(cls)`.
- Cost-adjusted breakeven `be_c = (sl + c) / (tp + sl)`. Stocks today
  (tp 1.0%, sl 0.6%, c 0.1%): 43.75% (naive 37.5%). Crypto: c = 1.2% >
  tp = 0.6%, so no crypto trade can be net positive under this cost model.
- Primary endpoint: live signal precision `p_sig` = TP-first rate of
  labelled above-bar signals since `opened_at` (section 7), with its Wilson
  95% CI `[lo, hi]` on raw n and the day-clustered lower bound `lo_day`.
  Expected n ≈ 40/day × 45 ≈ 1800 → CI ≈ ±2.2 pp, which resolves the gap to
  breakeven; the executed-trade sample (≤ 4 stock round trips/day, realistic
  60–130) cannot (Wilson ±8–12 pp).
- Secondary endpoint: executed trades — `n_exec`, mean net return per trade
  `r̄` with SE, compared with `ev_bt − c` where `ev_bt` is the current
  `metrics[cls]['ev']`. Purpose: verify fills, settlement and cost
  assumptions, not to estimate EV.

Verdict per class, printed under the all-time headline in every report:
- **NO-GO** if `hi < be_c`, or if the cost gate blocks the class (crypto is
  NO-GO by construction with defaults).
- **GO** if `lo > be_c` and `lo_day > be_c` and `n_exec >= 60` and
  `r̄ >= (ev_bt − c) − SE`.
- **EXTEND** otherwise. On 2026-11-13 an EXTEND is treated as NO-GO for real
  money until a further paper period turns it into GO.

Every rate and dollar figure in the report carries its n and CI; the
all-time $ headline is never read without them.

`src/scorecard.py: verdict(cls, stats) -> (str, str)` implements this and is
the only place the rule lives.
