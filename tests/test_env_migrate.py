"""dev/env-migrate.sh: brings a pre-paper-account .env up to the contract.

Runs the real script with bash against a temp file shaped like the live
/opt/trading-bot/.env was before the paper-account deploy (removed keys
present, new keys absent). No container needed: pure bash on a file.
"""
import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / 'dev' / 'env-migrate.sh'

LIVE_SHAPE = (
    "DISCORD_TOKEN=abc.def.ghi\n"
    "CHANNEL_ID=\n"
    "USER_ID=123456789012345678\n"
    "WEEKLY_BUDGET=500\n"
    "LOOKBACK_PERIOD=2y\n"
    "FAST_MODE=1\n"
    "FAST_MAX_HOLD_MIN=240\n"
    "FAST_COOLDOWN_MIN=15\n"
    "BUDGET_MODE=deployed\n"
    "MIN_EV_TO_TRADE=0.003\n"
)

# The paper-run values the contract's Task 10 puts in the live .env.
REQUIRED = {
    'STARTING_CASH': '500',
    'ACCOUNT_TYPE': 'cash',
    'FAST_IGNORE_EV': '1',
    'STOCK_SLIPPAGE_BPS': '5',
    'CRYPTO_SPREAD_BPS': '60',
    'DAILY_LOSS_LIMIT_PCT': '3',
    'MIN_ORDER_USD': '1',
}
REMOVED = ('WEEKLY_BUDGET', 'BUDGET_MODE', 'FAST_MAX_HOLD_MIN')


def run(env_file: Path) -> subprocess.CompletedProcess:
    return subprocess.run(['bash', str(SCRIPT), str(env_file)],
                          capture_output=True, text=True)


def parse(text: str) -> dict:
    """KEY -> list of raw values (a list so duplicates are visible)."""
    out: dict = {}
    for line in text.splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            out.setdefault(k, []).append(v)
    return out


@pytest.fixture
def env_file(tmp_path):
    p = tmp_path / '.env'
    p.write_text(LIVE_SHAPE)
    p.chmod(0o600)
    return p


def test_adds_required_keys_and_drops_removed_ones(env_file):
    res = run(env_file)
    assert res.returncode == 0, res.stderr
    text = env_file.read_text()
    kv = parse(text)
    for key, val in REQUIRED.items():
        assert kv.get(key) == [val], (key, kv.get(key))
    for key in REMOVED:
        assert key not in kv
    # Each new key sits on its own line with NO trailing comment: systemd's
    # EnvironmentFile keeps "   # text" as part of the value.
    for key, val in REQUIRED.items():
        assert f"{key}={val}\n" in text
    # Everything else is untouched, in its original order.
    kept = [l for l in LIVE_SHAPE.splitlines()
            if l.split('=', 1)[0] not in REMOVED]
    assert [l for l in text.splitlines() if l.split('=', 1)[0] not in REQUIRED] == kept
    assert kv['DISCORD_TOKEN'] == ['abc.def.ghi']


def test_reports_what_changed(env_file):
    res = run(env_file)
    for key, val in REQUIRED.items():
        assert f"{key}={val}" in res.stdout
    for key in REMOVED:
        assert key in res.stdout
    assert 'abc.def.ghi' not in res.stdout + res.stderr   # never echo secrets


def test_idempotent(env_file):
    run(env_file)
    first = env_file.read_text()
    res = run(env_file)
    assert res.returncode == 0, res.stderr
    assert env_file.read_text() == first
    kv = parse(first)
    assert all(len(v) == 1 for v in kv.values()), kv


def test_keeps_an_operator_value_and_warns_when_ev_gate_stays_on(env_file):
    env_file.write_text(LIVE_SHAPE + "STARTING_CASH=1000\nFAST_IGNORE_EV=0\n")
    res = run(env_file)
    assert res.returncode == 0, res.stderr
    kv = parse(env_file.read_text())
    assert kv['STARTING_CASH'] == ['1000']
    assert kv['FAST_IGNORE_EV'] == ['0']
    assert 'FAST_IGNORE_EV' in res.stderr and 'WARNING' in res.stderr


def test_preserves_mode_and_writes_backup(env_file):
    run(env_file)
    assert stat.S_IMODE(os.stat(env_file).st_mode) == 0o600
    bak = env_file.with_name('.env.bak')
    assert bak.exists()
    assert bak.read_text() == LIVE_SHAPE
    assert stat.S_IMODE(os.stat(bak).st_mode) == 0o600


def test_handles_missing_trailing_newline(tmp_path):
    p = tmp_path / '.env'
    p.write_text("DISCORD_TOKEN=x\nWEEKLY_BUDGET=500")
    res = run(p)
    assert res.returncode == 0, res.stderr
    kv = parse(p.read_text())
    assert kv['DISCORD_TOKEN'] == ['x']
    assert kv['STARTING_CASH'] == ['500']
    assert 'WEEKLY_BUDGET' not in kv


def test_refuses_missing_file(tmp_path):
    res = run(tmp_path / 'nope.env')
    assert res.returncode != 0
