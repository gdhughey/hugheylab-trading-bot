#!/bin/bash
# Interactive configurator for the trading bot .env - run from the Proxmox
# console so the token is never echoed or captured in a transcript.
set -euo pipefail
ENV=/opt/trading-bot/.env

echo "=== Trading bot configuration ==="
read -rsp "Discord bot token (hidden, paste + Enter): " TOKEN; echo
echo
echo "Delivery: leave Channel ID BLANK to get alerts as direct messages (recommended)."
read -rp  "Channel ID [blank = DM you]: " CHANNEL
read -rp  "Your user ID  (right-click yourself  -> Copy User ID): " USERID
read -rp  "Weekly budget in dollars [5000]: " BUDGET; BUDGET=${BUDGET:-5000}
read -rsp "Anthropic API key for Claude analysis (optional, Enter to skip): " CKEY; echo

[[ -z "$TOKEN"   ]] && { echo "ERROR: token is required"; exit 1; }
if [[ -n "$CHANNEL" ]]; then
  [[ "$CHANNEL" =~ ^[0-9]{17,20}$ ]] || { echo "ERROR: channel ID must be 17-20 digits (or blank for DM)"; exit 1; }
fi
[[ "$USERID"  =~ ^[0-9]{17,20}$ ]] || { echo "ERROR: user ID must be 17-20 digits"; exit 1; }

umask 077
cat > "$ENV" <<ENVEOF
DISCORD_TOKEN=$TOKEN
CHANNEL_ID=$CHANNEL
USER_ID=$USERID

WEEKLY_BUDGET=$BUDGET
LOOKBACK_PERIOD=2y

CLAUDE_API_KEY=$CKEY
CLAUDE_MODEL=claude-opus-5

DB_PATH=data/trading_bot.db
MODEL_PATH=data/model.joblib
ENVEOF
chmod 600 "$ENV"

echo
echo "Wrote $ENV (mode 600). Values recorded:"
sed -E 's/^(DISCORD_TOKEN=).*/\1<hidden>/; s/^(CLAUDE_API_KEY=).+/\1<hidden>/' "$ENV"
echo
if [[ -z "$CHANNEL" ]]; then
  echo "Delivery mode: DIRECT MESSAGE to user $USERID"
else
  echo "Delivery mode: channel $CHANNEL"
fi
echo
echo "Now start it:  systemctl enable --now trading-bot"
echo "Watch it:      journalctl -u trading-bot -f"
