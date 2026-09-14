#!/usr/bin/env bash
# Run one read-only SQL statement against the LIVE ledger in LXC 200.
# The container has no sqlite3 CLI, so this goes through the venv's python.
# Usage: dev/ct-sql.sh "select * from account"
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: dev/ct-sql.sh \"<sql>\"" >&2; exit 2; }
sudo pct exec 200 -- /opt/trading-bot/venv/bin/python - "$1" <<'PY'
import sqlite3, sys
# mode=ro: this tool must never take a write lock under the running bot.
c = sqlite3.connect('file:/opt/trading-bot/data/trading_bot.db?mode=ro', uri=True)
cur = c.execute(sys.argv[1])
if cur.description:
    print(' | '.join(d[0] for d in cur.description))
for row in cur:
    print(' | '.join(str(v) for v in row))
PY
