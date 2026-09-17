#!/usr/bin/env bash
# Deploy the committed working tree to the LIVE bot in LXC 200 and restart it.
# Usage: dev/deploy.sh [--dry-run]
#   --dry-run  print the list of files that would be shipped; touches nothing.
#
# Sibling of dev/ct-test.sh, but aimed at /opt/trading-bot (the production
# copy, not /opt/trading-bot-dev). The container has no git, so this is the
# only way code reaches it. Untarring over the live tree never deletes files,
# and the container's own data/, logs/, venv/ and .env are excluded here, so a
# deploy can never clobber the ledger, the models or the secrets.
set -euo pipefail
cd "$(dirname "$0")/.."

CT=200
APP=/opt/trading-bot
EXCLUDES=(--exclude=.git --exclude=venv --exclude=data --exclude=logs --exclude=.env
          --exclude=__pycache__ --exclude=tests --exclude=dev --exclude=docs)

if [[ "${1:-}" == "--dry-run" ]]; then
  tar cf - "${EXCLUDES[@]}" . | tar tf - | sort
  exit 0
fi

# Only ship a commit: a rollback is "git checkout <tag> and deploy again",
# which is meaningless if what was running never existed in git.
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: uncommitted changes in tracked files - commit first" >&2
  exit 1
fi
rev=$(git rev-parse --short HEAD)

# The .env is excluded from the tar, so the paper-account keys the contract
# adds (and the three it removes) have to be edited in place. The script only
# appends what is missing and deletes what is dead; it never touches a
# secret, and it prints the effective FAST_IGNORE_EV because that is the one
# key whose code default (0) would leave the paper account never trading.
echo "==> migrating $APP/.env"
sudo pct exec "$CT" -- bash -s "$APP/.env" < dev/env-migrate.sh

echo "==> shipping $rev to CT $CT:$APP"
tar czf - "${EXCLUDES[@]}" . | sudo pct exec "$CT" -- tar xzf - -C "$APP"

# Unit files ship inside the tar but systemd reads /etc; re-install them so a
# unit change in git is a unit change in the container.
echo "==> installing systemd units"
sudo pct exec "$CT" -- bash -c "install -m 644 $APP/proxmox/trading-bot.service $APP/proxmox/trading-collector.service $APP/proxmox/trading-collector.timer /etc/systemd/system/ && systemctl daemon-reload && systemctl enable -q trading-collector.timer && systemctl start trading-collector.timer"

echo "==> restarting trading-bot"
sudo pct exec "$CT" -- systemctl restart trading-bot
sudo pct exec "$CT" -- systemctl is-active trading-bot
sudo pct exec "$CT" -- systemctl is-active trading-collector.timer
cat <<EOF
==> follow it with: sudo pct exec $CT -- journalctl -u trading-bot -f
    expect the startup notice to say "Trading is LIVE" with the stock line
    "trading on paper despite EV ... (FAST_IGNORE_EV on)"
==> then confirm the ledger opened: dev/ct-sql.sh "select * from account"
    one row: starting_cash 500 / cash 500 / account_type cash
EOF
