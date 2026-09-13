#!/usr/bin/env bash
# Push the working tree to LXC 200's dev copy and run the test suite there,
# using the production venv (the pve host has no sklearn/discord/pandas).
# Usage: dev/ct-test.sh [pytest args...]
set -euo pipefail
cd "$(dirname "$0")/.."
tar czf - --exclude=venv --exclude=__pycache__ --exclude=.git --exclude=data --exclude=logs . \
  | sudo pct exec 200 -- bash -c 'rm -rf /opt/trading-bot-dev && mkdir -p /opt/trading-bot-dev && tar xzf - -C /opt/trading-bot-dev'
sudo pct exec 200 -- bash -c "cd /opt/trading-bot-dev && DB_PATH=:memory: /opt/trading-bot/venv/bin/python -m pytest -q $* 2>&1"
