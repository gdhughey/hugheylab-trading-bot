#!/bin/bash
# Interactive configurator for the trading bot .env - run from the Proxmox
# console so the token is never echoed or captured in a transcript.
# ENV_FILE is overridable so the script can be exercised against a temp file.
set -euo pipefail
ENV_FILE=${ENV_FILE:-/opt/trading-bot/.env}

echo "=== Trading bot configuration ==="
read -rsp "Discord bot token (hidden, paste + Enter): " TOKEN; echo
echo
echo "Delivery: leave Channel ID BLANK to get alerts as direct messages (recommended)."
read -rp  "Channel ID [blank = DM you]: " CHANNEL
read -rp  "Your user ID  (right-click yourself  -> Copy User ID): " USERID
read -rp  "Starting cash [500]: " CASH; CASH=${CASH:-500}
read -rsp "Anthropic API key for Claude analysis (optional, Enter to skip): " CKEY; echo

[[ -z "$TOKEN"   ]] && { echo "ERROR: token is required"; exit 1; }
if [[ -n "$CHANNEL" ]]; then
  [[ "$CHANNEL" =~ ^[0-9]{17,20}$ ]] || { echo "ERROR: channel ID must be 17-20 digits (or blank for DM)"; exit 1; }
fi
[[ "$USERID"  =~ ^[0-9]{17,20}$ ]] || { echo "ERROR: user ID must be 17-20 digits"; exit 1; }
# The account row is seeded from this value on the first start and never
# reset (there is no setter), so a typo here corrupts the all-time return.
[[ "$CASH" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "ERROR: starting cash must be a number"; exit 1; }

umask 077
cat > "$ENV_FILE" <<ENVEOF
DISCORD_TOKEN=$TOKEN
CHANNEL_ID=$CHANNEL
USER_ID=$USERID

STARTING_CASH=$CASH
ACCOUNT_TYPE=cash
LOOKBACK_PERIOD=2y

CLAUDE_API_KEY=$CKEY
CLAUDE_MODEL=claude-opus-5

DB_PATH=data/trading_bot.db
MODEL_PATH=data/model.joblib
ENVEOF
chmod 600 "$ENV_FILE"

echo
echo "Wrote $ENV_FILE (mode 600). Values recorded:"
sed -E 's/^(DISCORD_TOKEN=).*/\1<hidden>/; s/^(CLAUDE_API_KEY=).+/\1<hidden>/' "$ENV_FILE"
echo
if [[ -z "$CHANNEL" ]]; then
  echo "Delivery mode: DIRECT MESSAGE to user $USERID"
else
  echo "Delivery mode: channel $CHANNEL"
fi
echo
echo "Now start it:  systemctl enable --now trading-bot"
echo "Watch it:      journalctl -u trading-bot -f"
