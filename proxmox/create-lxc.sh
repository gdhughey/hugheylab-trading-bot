#!/usr/bin/env bash
# Create a Proxmox LXC and install the bot into it.
# Run on the Proxmox host:  bash create-lxc.sh [CTID] [IP/CIDR] [GATEWAY]
set -euo pipefail

CTID="${1:-200}"
IPCIDR="${2:-dhcp}"
GATEWAY="${3:-}"
TEMPLATE="${TEMPLATE:-local:vztmpl/debian-13-standard_13.1-2_amd64.tar.zst}"
STORAGE="${STORAGE:-local-lvm}"
REPO="${REPO:-https://github.com/gdhughey/hugheylab-trading-bot.git}"

command -v pct >/dev/null || { echo "pct not found - run this on a Proxmox host"; exit 1; }
pct config "$CTID" &>/dev/null && { echo "CTID $CTID already exists"; exit 1; }

if [[ "$IPCIDR" == "dhcp" ]]; then
  NET="name=eth0,bridge=vmbr0,ip=dhcp,type=veth"
else
  NET="name=eth0,bridge=vmbr0,ip=${IPCIDR},type=veth${GATEWAY:+,gw=$GATEWAY}"
fi

echo "==> Creating LXC $CTID (2 cores / 1024 MB / 4 G)"
pct create "$CTID" "$TEMPLATE" \
  --hostname tradingbot \
  --cores 2 --memory 1024 --swap 512 \
  --rootfs "${STORAGE}:4" \
  --net0 "$NET" \
  --ostype debian --unprivileged 1 --features nesting=1 \
  --onboot 1 --description "Hybrid trading bot (paper trading)"

pct start "$CTID"
echo "==> Waiting for network"
for _ in $(seq 1 30); do
  pct exec "$CTID" -- getent hosts github.com &>/dev/null && break
  sleep 2
done

echo "==> Installing inside the container"
pct exec "$CTID" -- bash -c "apt-get update -qq && apt-get install -y -qq curl git >/dev/null"
pct exec "$CTID" -- bash -c "git clone --depth 1 $REPO /tmp/bot && bash /tmp/bot/install.sh"

echo
echo "Done. Configure and start it with:"
echo "    pct enter $CTID"
echo "    /opt/trading-bot/configure.sh"
echo "    systemctl enable --now trading-bot"
