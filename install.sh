#!/usr/bin/env bash
# Install the trading bot on a Debian/Ubuntu host (or inside an LXC).
# Run as root:  bash install.sh
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/trading-bot}"
REPO="${REPO:-https://github.com/gdhughey/hugheylab-trading-bot.git}"

log() { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root"

log "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git ca-certificates tzdata >/dev/null

log "Fetching application to $APP_DIR"
if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" pull --ff-only
elif [[ -f "$(dirname "$0")/main.py" ]]; then
  mkdir -p "$APP_DIR"
  cp -r "$(dirname "$0")/." "$APP_DIR/"
else
  git clone --depth 1 "$REPO" "$APP_DIR"
fi
mkdir -p "$APP_DIR/data" "$APP_DIR/logs"

log "Creating virtualenv and installing dependencies (takes a few minutes)"
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install -q --upgrade pip
"$APP_DIR/venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

log "Installing systemd unit"
install -m 644 "$APP_DIR/proxmox/trading-bot.service" /etc/systemd/system/trading-bot.service
systemctl daemon-reload

if [[ ! -f "$APP_DIR/.env" ]]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  chmod 600 "$APP_DIR/.env"
fi

cat <<EOF

============================================================
 Installed to $APP_DIR
============================================================

NEXT STEPS

1. Configure it:
     $APP_DIR/configure.sh
   (or edit $APP_DIR/.env by hand)

   You need, from https://discord.com/developers/applications:
     DISCORD_TOKEN  - Bot tab -> Reset Token
     USER_ID        - your own Discord user ID (enable Developer Mode,
                      right-click yourself -> Copy User ID)
   CHANNEL_ID is OPTIONAL - leave it blank to receive alerts as DMs.

2. Invite the bot to a server you are in. Discord will NOT deliver a DM
   from a bot you share no server with. Permissions 85056:
     https://discord.com/api/oauth2/authorize?client_id=YOUR_APP_ID&permissions=85056&scope=bot%20applications.commands

3. Start it:
     systemctl enable --now trading-bot
     journalctl -u trading-bot -f

First start takes ~4 minutes: it downloads ~250k price rows and trains
the model before the first scan.

THIS IS PAPER TRADING. No broker is connected. Approving a trade writes
a row to a local SQLite ledger and nothing else.
============================================================
EOF
