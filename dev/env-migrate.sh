#!/usr/bin/env bash
# Bring a pre-paper-account .env up to the paper-brokerage contract, in place.
# Usage: dev/env-migrate.sh <env-file>
#
# Adds the paper-run keys that are missing (an operator's existing value is
# kept, never overwritten), deletes the keys the bot no longer reads, and
# leaves every other line byte-for-byte where it was. Idempotent: a second
# run changes nothing. Writes <env-file>.bak (same mode) before the first
# change. Prints only the keys it touched - never a token.
#
# dev/deploy.sh streams this into LXC 200 against /opt/trading-bot/.env, so
# the deploy cannot forget it. Why it matters: every new key has a code
# default EXCEPT FAST_IGNORE_EV, which defaults to 0 - and with both classes
# backtesting below MIN_EV_TO_TRADE that gate blocks every entry, so the paper
# account would open with "Nothing will be traded" and stay empty.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: dev/env-migrate.sh <env-file>" >&2; exit 2; }
f=$1
[[ -f "$f" ]] || { echo "ERROR: $f does not exist" >&2; exit 1; }

# Contract Task 10, in this order. One KEY=VALUE per line, NO trailing
# comment: systemd's EnvironmentFile keeps "   # text" as part of the value.
ADD=(
  STARTING_CASH=500
  ACCOUNT_TYPE=cash
  FAST_IGNORE_EV=1
  STOCK_SLIPPAGE_BPS=5
  CRYPTO_SPREAD_BPS=60
  DAILY_LOSS_LIMIT_PCT=3
  MIN_ORDER_USD=1
)
REMOVE=(WEEKLY_BUDGET BUDGET_MODE FAST_MAX_HOLD_MIN)

has_key() { grep -q -E "^$1=" "$f"; }

changed=0
backup() {
  if [[ $changed -eq 0 ]]; then
    cp -p "$f" "$f.bak"
    changed=1
  fi
}

for key in "${REMOVE[@]}"; do
  if has_key "$key"; then
    backup
    sed -i -E "/^$key=/d" "$f"      # sed -i keeps the file's mode and owner
    echo "removed $key"
  fi
done

for kv in "${ADD[@]}"; do
  key=${kv%%=*}
  if ! has_key "$key"; then
    backup
    # Appending keeps the inode, so mode 600 and ownership survive; just make
    # sure the previous last line is terminated first.
    if [[ -s "$f" && $(tail -c 1 "$f" | od -An -c | tr -d ' ') != '\n' ]]; then
      printf '\n' >> "$f"
    fi
    printf '%s\n' "$kv" >> "$f"
    echo "added $kv"
  fi
done

[[ $changed -eq 1 ]] || echo "unchanged: $f already matches the contract"

ev=$(grep -E '^FAST_IGNORE_EV=' "$f" | tail -n 1 | cut -d= -f2-)
case "$ev" in
  1|true|yes) ;;
  *) echo "WARNING: FAST_IGNORE_EV=$ev - the EV gate stays on, so with both" \
          "classes below MIN_EV_TO_TRADE nothing will be traded on paper" >&2 ;;
esac
