# Paper brokerage — interface contract (binding for every plan task)

Spec: `docs/superpowers/specs/2026-09-12-paper-brokerage-design.md`. This file
fixes the names every task must use so sections written independently agree.
If a task needs something not listed here, it must add it to its own section
under a heading "Contract additions" — never rename anything below.

Repo: `/home/gdhughey/hugheylab-trading-bot`. Tests: `dev/ct-test.sh [pytest args]`
(pushes the tree to LXC 200 and runs pytest in the production venv; the pve
host has no sklearn/pandas/discord). Tests use a file DB under `tmp_path`.

## Task order and dependencies

| # | Task | Creates / modifies | Depends on |
|---|------|--------------------|-----------|
| 1 | costs | `src/costs.py`, `tests/test_costs.py` | — |
| 2 | calendar | `src/intraday_engine.py` (helpers only), `tests/test_calendar.py` | — |
| 3 | schema | `src/database.py`, `tests/test_migration.py`, `tests/conftest.py` | — |
| 4 | account ledger | `src/budget_tracker.py`, `tests/test_budget_tracker.py` | 1, 2, 3 |
| 5 | signal log | `src/signal_log.py`, `src/intraday_engine.py` (`signal()` bar_ts), `tests/test_signal_log.py` | 3 |
| 6 | fast trader | `src/fast_trader.py`, `tests/test_fast_trader.py` | 1, 2, 4, 5 |
| 7 | scorecard | `src/scorecard.py`, `tests/test_scorecard.py` | 4, 6 (8 for the live report) |
| 8 | benchmark | `src/ml_engine.py`, `tests/test_benchmark.py` | — |
| 9 | discord | `src/discord_bot.py`, `tests/test_discord_embeds.py` | 4, 6, 7 |
| 10 | config + deploy | `.env.example`, `configure.sh`, `README.md`, live `.env`, restart | all |

## Environment variables (final set; defaults in parentheses)

Added: `STARTING_CASH` (500), `ACCOUNT_TYPE` (cash | margin; cash),
`STOCK_SLIPPAGE_BPS` (5), `CRYPTO_SPREAD_BPS` (60, per side),
`SEC_FEE_RATE` (0.0000206), `FINRA_TAF_PER_SHARE` (0.000195),
`FINRA_TAF_CAP` (9.79), `MIN_ORDER_USD` (1), `DAILY_LOSS_LIMIT_PCT` (3),
`FAST_IGNORE_EV` (0; set to 1 in the live .env).

Removed: `WEEKLY_BUDGET`, `BUDGET_MODE`, `FAST_MAX_HOLD_MIN`.

Unchanged and still read: `FAST_MAX_POSITIONS`, `FAST_EOD_FLATTEN_MIN`,
`FAST_COOLDOWN_MIN`, `MIN_EV_TO_TRADE`, `FAST_POLL_SECONDS`, `TRADE_CRYPTO`,
`DB_PATH`.

## `src/costs.py` (Task 1)

```python
def fill(symbol: str, side: str, ref_price: float, qty: float) -> dict
    # side in ('BUY', 'SELL'); returns
    # {'fill_price': float, 'fees': float, 'gross': float, 'net': float}
    # gross = fill_price * qty; net = gross + fees (BUY) | gross - fees (SELL)
    # stock: fill = ref*(1 ± STOCK_SLIPPAGE_BPS/1e4); BUY fees 0;
    #        SELL fees = SEC_FEE_RATE*gross + min(FINRA_TAF_PER_SHARE*qty, FINRA_TAF_CAP)
    # crypto: fill = ref*(1 ± CRYPTO_SPREAD_BPS/1e4); fees 0
def round_trip_cost(cls: str) -> float
    # 'stock': 2*STOCK_SLIPPAGE_BPS/1e4 + SEC_FEE_RATE ; 'crypto': 2*CRYPTO_SPREAD_BPS/1e4
def qty_str(x: float) -> str
    # f"{x:.6f}".rstrip('0').rstrip('.'); qty_str(1.0) == '1'; qty_str(0.5) == '0.5'
```
Env is read at call time via `os.getenv` (tests use `monkeypatch.setenv`).
`asset_class`/`is_crypto` are imported from `src.intraday_engine`.

## `src/intraday_engine.py` calendar helpers (Task 2)

```python
US_HOLIDAYS_2026  # add '2027-01-01'
def is_trading_day(d: datetime.date) -> bool          # weekday < 5 and iso date not in US_HOLIDAYS_2026
def next_trading_day_open(ts: datetime) -> datetime   # ts tz-aware (any tz); first trading day STRICTLY after ts's ET date, at 09:30 ET, tz=ET
```
`market_state()` replaces its weekend + holiday checks with `is_trading_day`.
Behaviour otherwise unchanged.

## `src/database.py` (Task 3)

`SCHEMA` declares `trades.shares REAL NOT NULL`, `positions.shares REAL NOT NULL DEFAULT 0`,
and adds:

```sql
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    opened_at TEXT NOT NULL, starting_cash REAL NOT NULL,
    cash REAL NOT NULL, account_type TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS day_state (
    date TEXT PRIMARY KEY, start_equity REAL NOT NULL,
    loss_tripped_at TEXT, loss_announced_at TEXT, report_posted_at TEXT);
CREATE TABLE IF NOT EXISTS equity_history (
    date TEXT PRIMARY KEY, cash REAL NOT NULL, positions_value REAL NOT NULL,
    equity REAL NOT NULL, fees_to_date REAL NOT NULL, realized_to_date REAL NOT NULL,
    recorded_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, bar_ts TEXT NOT NULL, symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL, probability REAL NOT NULL, bar REAL NOT NULL,
    above_bar INTEGER NOT NULL, ref_price REAL NOT NULL, trade_date TEXT NOT NULL,
    executed_trade_id INTEGER, label INTEGER, labeled_at TEXT,
    UNIQUE (symbol, bar_ts));
CREATE INDEX IF NOT EXISTS idx_signals_label ON signals (label, symbol);
```

`_migrate()` adds (PRAGMA-guarded) `trades.ref_price REAL`, `trades.fees REAL NOT NULL DEFAULT 0`,
`trades.gross_pnl REAL`, `trades.available_at TEXT`, `trades.trade_date TEXT`,
`trades.entry_probability REAL`, `trades.exit_reason TEXT`,
`positions.entry_ref REAL NOT NULL DEFAULT 0`; backfills
`UPDATE positions SET entry_ref = avg_price WHERE shares > 0 AND entry_ref = 0`;
seeds `INSERT OR IGNORE INTO account VALUES (1, <now utc iso>, STARTING_CASH, STARTING_CASH, ACCOUNT_TYPE)`.
Existing INTEGER `shares` columns are NOT rebuilt (SQLite affinity stores REAL).

`tests/conftest.py` provides fixture `db_path(tmp_path)` → str path with
`Database(path)` already constructed, and fixture `old_db_path(tmp_path)` that
writes the PRE-change schema (INTEGER shares, no new tables/columns) with a
few legacy rows.

## `src/budget_tracker.py` (Task 4) — class `BudgetTracker(db_path=None)`

Time helpers: module-level `_now()` (UTC ISO seconds, existing) and
`_et_date(ts_iso: str) -> str` (ET calendar date of a UTC ISO string).
All methods that depend on "now" accept `now: datetime | None = None` (UTC, tz-aware).

```python
# account
def opened_at(self) -> str                     # account.opened_at
def starting_cash(self) -> float
def account_type(self) -> str
def get_cash(self) -> float
def get_unsettled(self, now=None) -> float     # SUM(amount) of EXECUTED SELLs with available_at > now, created_at >= opened_at
def get_buying_power(self, now=None) -> float  # cash - unsettled - SUM(amount) of PENDING BUYs; never < 0
def get_equity(self, prices: dict) -> float    # cash + Σ shares*prices.get(sym, avg_price)
def get_fees_paid(self) -> float               # SUM(fees) EXECUTED since opened_at
def get_realized_pnl(self) -> float            # SUM(realized_pnl) EXECUTED since opened_at
def get_gross_pnl(self) -> float               # SUM(gross_pnl) EXECUTED since opened_at

# sizing
def size_order(self, symbol, ref_price, prices=None, now=None) -> tuple[float, float, float]
    # (qty, size_usd, est_fill); qty == 0.0 when size_usd < MIN_ORDER_USD
    # size_usd = min(buying_power, equity / FAST_MAX_POSITIONS)
    # est_fill = costs.fill(symbol,'BUY',ref_price,1)['fill_price']; qty = round(size_usd/est_fill, 6)

# trade lifecycle (all under self._lock, BEGIN IMMEDIATE)
def log_trade(self, symbol, side, ref_price, qty, *, probability=None, exit_reason=None, now=None) -> int
    # rounds qty to 6dp, calls costs.fill, inserts PENDING row with ref_price, price=fill, fees,
    # amount=net, shares=qty, trade_date=_et_date(created_at), entry_probability, exit_reason, week_key
def execute_trade(self, trade_id, now=None) -> sqlite3.Row | None
    # PENDING -> EXECUTED; BUY: cash -= amount; SELL: cash += amount, available_at
    # (stock: next_trading_day_open(created_at) as UTC ISO; crypto: created_at),
    # realized_pnl = amount - closed*avg_price, gross_pnl = realized_pnl + fees;
    # applies position (entry_ref weighting, |shares|<1e-6 -> 0). Returns the executed row.
def reject_trade(self, trade_id, now=None) -> bool

# positions / reporting
def get_positions(self) -> list[dict]          # shares > 1e-9; keys symbol, shares(float), avg_price, entry_ref, cost_basis, updated_at
def get_pnl(self, price_fn) -> dict
    # keys: positions, stale, realized, unrealized, total, cost_basis, market_value,
    #       cash, unsettled, buying_power, equity, starting_cash, all_time_net, all_time_pct,
    #       fees_paid, gross_pnl   (return_pct REMOVED)
def get_trades_since_open(self, day: str | None = None) -> list[sqlite3.Row]  # EXECUTED, created_at >= opened_at, optional trade_date filter

# day state / equity history
def get_day_state(self, date_et: str) -> sqlite3.Row | None
def ensure_day_state(self, date_et: str, equity: float) -> sqlite3.Row   # INSERT OR IGNORE then return
def set_day_flag(self, date_et: str, column: str, ts: str | None = None) -> None  # column in ('loss_tripped_at','loss_announced_at','report_posted_at')
def record_equity(self, date_et: str, prices: dict, now=None) -> dict    # upserts equity_history; returns the row as dict
def equity_series(self) -> list[dict]          # equity_history rows ordered by date, prefixed by {'date': opened_at[:10], 'equity': starting_cash}
```
Deleted: `weekly_budget`, `set_weekly_budget`, `get_weekly_spent`, `get_turnover`,
`get_remaining_budget`, `can_trade`, `get_statistics`, `_sum`, `_deployed`, `_committed`.
`_week_key()` stays (column is NOT NULL).

## `src/signal_log.py` (Task 5)

```python
def record(conn, signals: list[dict], now=None) -> int     # INSERT OR IGNORE rows from scan_all() output; returns rows inserted
def mark_executed(conn, symbol: str, bar_ts: str, trade_id: int) -> None
def label_pending(conn, now=None) -> int                   # labels rows with label IS NULL; returns rows labelled
```
`record` expects each signal dict to carry `symbol, asset_class, probability, bar, above_bar, price, bar_ts`
(`IntradayEngine.signal()` gains `'bar_ts': h.index[-1].isoformat()` — UTC ISO).
`trade_date` = ET date of `bar_ts`. `label_pending` loads `prices_intraday`
bars for the symbol with `ts >= bar_ts` (interval = `INTERVAL`), applies
`labeling.triple_barrier(high, low, close, tp, sl, horizon, session=<ET date per bar> for stocks else None)`
and stores `label` for the first row (the signal's bar); NaN stays NULL.

## `src/fast_trader.py` (Task 6) — class `FastTrader(engine, budget_tracker, quote_fn=None)`

```python
def class_gate(cls: str, metrics: dict | None) -> tuple[bool, str]   # module-level; order: cost gate, EV gate, FAST_IGNORE_EV text, ok text
def max_hold_min(cls: str) -> float                                   # module-level; barriers(cls)[2] * interval_minutes (INTERVAL '5m' -> 5)
```
`FastTrader.max_hold_min` property and `FAST_MAX_HOLD_MIN` are deleted.
`cycle()` summary dict keys (superset of today's): `state, desc, exits, entries, skipped,
candidates, ts, stocks_open, crypto, rows, minutes_to_close, bars, bar, note,
gates: {cls: (tradeable, text)}, blocked_classes: [cls with tradeable False],
loss_tripped: bool, loss_announce: bool (True on the cycle that first trips),
equity: float, buying_power: float, unsettled: float`.
Entry dict: `symbol, shares, price (fill), ref_price, probability, cost (=amount), fees, trade_id`.
Exit dict: `symbol, shares, price (fill), ref_price, reason (text), exit_reason ('tp'|'sl'|'timeout'|'eod'),
pnl (=realized_pnl), gross (=gross_pnl), fees, pct (ref-to-ref), trade_id`.
Skip reasons (exact strings): `'market closed'`, `'too close to the bell'`, `'cooling down'`,
`'below min order'`, `'insufficient buying power'`, `'daily loss limit'`, `'class gated: <text>'`.
Cycle order: market state → gates → `engine.fetch()` → quotes for held symbols (dict `prices`)
→ `ensure_day_state(today_et, equity)` → exits (ref-to-ref vs `pos['entry_ref']`) → loss-limit check
→ `scan_all()` → `signal_log.record` → entries via `size_order` → `signal_log.mark_executed`.

## `src/scorecard.py` (Task 7)

```python
def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]     # (0.0, 1.0) when n == 0
def mean_ci(values: list[float]) -> tuple[float, float, float]               # (mean, se, half_width_95) ; (0,0,0) when n < 2
def cost_breakeven(cls: str) -> float                                        # (sl + round_trip_cost(cls)) / (tp + sl)
def verdict(cls: str, stats: dict) -> tuple[str, str]                        # ('GO'|'NO-GO'|'EXTEND', reason text)
    # stats keys: gated(bool), sig_n, sig_lo, sig_hi, sig_lo_day, exec_n, exec_mean, exec_se, ev_bt, cost
def build_scorecard(budget, engine, intraday, day_et: str, now=None) -> dict
    # keys: headline{all_time_net, all_time_pct, n_closed, ci_dollars}, account{equity, cash, buying_power,
    #  unsettled, unsettled_until, starting_cash, gross_pnl, fees_paid}, today{realized, unrealized, n_trades,
    #  wins, losses, loss_tripped}, closed_today[list], positions[list], since_open{n_closed, win_rate, win_lo,
    #  win_hi, mean_net, mean_ci, profit_factor, max_drawdown_pct, days_running}, classes{cls: {...verdict inputs,
    #  verdict, verdict_text, exits_by_reason, tp_first_rate, tp_lo, tp_hi, bt_precision, bt_n, ev_bt_net,
    #  exec_mean_pct, exec_ci, gate_text, sig_n, sig_rate, sig_lo, sig_hi, sig_lo_day}}, spy{start_close, last_close, pct, value} | None
```

## `src/ml_engine.py` (Task 8)

`BENCHMARK_SYMBOLS = ['SPY']`. `fetch_and_store_data(symbols)` fetches
`[s for s in BENCHMARK_SYMBOLS if s not in symbols] + list(symbols)`; the
symbol list used for training/scanning (`self.symbols`, `_stored_symbols()`)
never includes benchmark symbols. `stored_close(symbol)` works for `'SPY'`.
New: `first_close_on_or_after(symbol, date_iso) -> float | None`.

## `src/discord_bot.py` (Task 9)

- `_send_startup_notice`, on_ready log, `_fast_action_embed`, `_fast_idle_embed`, daily report use `class_gate`.
- `daily_summary` loop: every 5 min, unconditional start in `on_ready`; fires when `now_et.time() >= 16:05`
  and `day_state[today].report_posted_at IS NULL`; steps: `signal_log.label_pending` → `record_equity` →
  `_scorecard_embed(today)` → send → `set_day_flag(today,'report_posted_at')`; `HTTPException` → one-line text
  fallback then flag; other exceptions → ERROR log, retry next tick. `_last_daily_summary` deleted.
- `_scorecard_embed(day_et) -> discord.Embed` renders `build_scorecard`; used by the daily report, `/pnl`, `/summary`.
- `/stats` removed; `/budget` replaced by `/account` (`_embed_account`).
- All "Cash left"/"Remaining"/"Remaining Budget"/"Budget Remaining" → "Buying power"; "Balance" added.
- Loss-limit announcement: when a cycle summary has `loss_announce`, send one embed then `set_day_flag(today,'loss_announced_at')`.
- `send_trade_alert` uses `budget.size_order`; SELL sells the whole held qty with `exit_reason='manual'`.
- Every quantity display uses `costs.qty_str`.

## Task 10 (config + deploy)

`.env.example` and `configure.sh` updated; README section "Paper account".
Live `.env` on LXC 200: add `STARTING_CASH=500`, `ACCOUNT_TYPE=cash`, `FAST_IGNORE_EV=1`,
`STOCK_SLIPPAGE_BPS=5`, `CRYPTO_SPREAD_BPS=60`, `DAILY_LOSS_LIMIT_PCT=3`, `MIN_ORDER_USD=1`;
remove `WEEKLY_BUDGET`, `BUDGET_MODE`, `FAST_MAX_HOLD_MIN`. Deploy = `tar` the tree into
`/opt/trading-bot` (excluding data/, logs/, venv/, .env), `systemctl restart trading-bot`,
watch `journalctl -u trading-bot -f` for the startup notice, and confirm the `account` row.
