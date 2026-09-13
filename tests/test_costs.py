"""Tests for src/costs.py — the paper-fill cost model.

Every function under test is pure and reads its parameters from the
environment at call time, so each test pins the env it needs via monkeypatch
and never touches a database or the clock.
"""
import pytest

from src import costs


# Contract defaults (docs/superpowers/plans/2026-09-13-paper-brokerage-contract.md).
DEFAULT_ENV = {
    'STOCK_SLIPPAGE_BPS': '5',
    'CRYPTO_SPREAD_BPS': '60',
    'SEC_FEE_RATE': '0.0000206',
    'FINRA_TAF_PER_SHARE': '0.000195',
    'FINRA_TAF_CAP': '9.79',
}


@pytest.fixture(autouse=True)
def pin_default_env(monkeypatch):
    # The container's shell or a loaded .env may carry different values;
    # pin the documented defaults so every assertion below is deterministic.
    for k, v in DEFAULT_ENV.items():
        monkeypatch.setenv(k, v)


# ---------------------------------------------------------------- fill(): stocks

def test_stock_buy_fills_above_ref_with_no_fees():
    r = costs.fill('AAPL', 'BUY', 100.0, 10.0)
    assert set(r) == {'fill_price', 'fees', 'gross', 'net'}
    assert r['fill_price'] == pytest.approx(100.05)      # 100 * (1 + 5/1e4)
    assert r['gross'] == pytest.approx(1000.5)
    assert r['fees'] == 0.0
    assert r['net'] == pytest.approx(1000.5)             # BUY: net = gross + fees


def test_stock_sell_fills_below_ref_and_charges_sec_plus_taf():
    r = costs.fill('AAPL', 'SELL', 100.0, 10.0)
    assert r['fill_price'] == pytest.approx(99.95)       # 100 * (1 - 5/1e4)
    assert r['gross'] == pytest.approx(999.5)
    expected_fees = 0.0000206 * 999.5 + 0.000195 * 10    # SEC on proceeds + TAF per share
    assert expected_fees == pytest.approx(0.0225397)     # anchor the literal
    assert r['fees'] == pytest.approx(expected_fees)
    assert r['net'] == pytest.approx(999.5 - expected_fees)   # SELL: net = gross - fees
    assert r['net'] == pytest.approx(999.4774603)


def test_stock_sell_taf_is_capped():
    # 100,000 shares * 0.000195 = 19.50 TAF, which the cap pulls down to 9.79.
    r = costs.fill('AAPL', 'SELL', 1.0, 100_000.0)
    assert r['fill_price'] == pytest.approx(0.9995)
    assert r['gross'] == pytest.approx(99_950.0)
    assert r['fees'] == pytest.approx(0.0000206 * 99_950.0 + 9.79)
    assert r['fees'] == pytest.approx(11.84897)
    assert r['net'] == pytest.approx(99_938.15103)


def test_stock_sell_fees_are_unrounded():
    # Tiny order: fees must not be rounded to cents or zeroed.
    # fill = 10 * (1 - 5/1e4) = 9.995, gross = 9.995 * 0.1 = 0.9995; the SEC
    # fee is on that slipped gross, not on ref * qty.
    r = costs.fill('AAPL', 'SELL', 10.0, 0.1)
    assert r['gross'] == pytest.approx(0.9995)
    assert r['fees'] == pytest.approx(0.0000206 * 0.9995 + 0.000195 * 0.1)
    assert r['fees'] == pytest.approx(0.0000400897)
    assert r['fees'] > 0


def test_stock_fee_env_overrides(monkeypatch):
    monkeypatch.setenv('SEC_FEE_RATE', '0')
    monkeypatch.setenv('FINRA_TAF_PER_SHARE', '0')
    r = costs.fill('AAPL', 'SELL', 100.0, 10.0)
    assert r['fees'] == 0.0
    assert r['net'] == pytest.approx(r['gross'])

    monkeypatch.setenv('FINRA_TAF_PER_SHARE', '1')      # 1 $/share -> 10, capped to 1
    monkeypatch.setenv('FINRA_TAF_CAP', '1')
    r = costs.fill('AAPL', 'SELL', 100.0, 10.0)
    assert r['fees'] == pytest.approx(1.0)


def test_stock_slippage_env_override(monkeypatch):
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '20')
    assert costs.fill('AAPL', 'BUY', 100.0, 1.0)['fill_price'] == pytest.approx(100.2)
    assert costs.fill('AAPL', 'SELL', 100.0, 1.0)['fill_price'] == pytest.approx(99.8)

    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '0')
    assert costs.fill('AAPL', 'BUY', 100.0, 1.0)['fill_price'] == pytest.approx(100.0)


# ---------------------------------------------------------------- fill(): crypto

def test_crypto_buy_fills_above_ref_no_fees():
    r = costs.fill('BTC-USD', 'BUY', 100.0, 0.5)
    assert r['fill_price'] == pytest.approx(100.6)       # 100 * (1 + 60/1e4)
    assert r['gross'] == pytest.approx(50.3)
    assert r['fees'] == 0.0
    assert r['net'] == pytest.approx(50.3)


def test_crypto_sell_fills_below_ref_no_fees():
    r = costs.fill('BTC-USD', 'SELL', 100.0, 0.5)
    assert r['fill_price'] == pytest.approx(99.4)        # 100 * (1 - 60/1e4)
    assert r['gross'] == pytest.approx(49.7)
    assert r['fees'] == 0.0                              # SEC/TAF never apply to crypto
    assert r['net'] == pytest.approx(49.7)


def test_crypto_ignores_stock_slippage_and_stock_fees(monkeypatch):
    # Even with absurd stock parameters, crypto is priced only by CRYPTO_SPREAD_BPS.
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '500')
    monkeypatch.setenv('SEC_FEE_RATE', '0.5')
    monkeypatch.setenv('FINRA_TAF_PER_SHARE', '5')
    r = costs.fill('ETH-USD', 'SELL', 100.0, 1.0)
    assert r['fill_price'] == pytest.approx(99.4)
    assert r['fees'] == 0.0


def test_crypto_spread_env_override(monkeypatch):
    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '10')
    assert costs.fill('BTC-USD', 'BUY', 100.0, 1.0)['fill_price'] == pytest.approx(100.1)
    assert costs.fill('BTC-USD', 'SELL', 100.0, 1.0)['fill_price'] == pytest.approx(99.9)


def test_fill_rejects_unknown_side():
    with pytest.raises(ValueError):
        costs.fill('AAPL', 'buy', 100.0, 1.0)      # case-sensitive by design
    with pytest.raises(ValueError):
        costs.fill('AAPL', 'SHORT', 100.0, 1.0)


# ---------------------------------------------------------------- round_trip_cost()

def test_round_trip_cost_stock_default():
    # 2 * 5 bps slippage + SEC rate (TAF is per share, not a fraction of notional,
    # so it is deliberately excluded from the fractional estimate).
    assert costs.round_trip_cost('stock') == pytest.approx(2 * 5 / 1e4 + 0.0000206)
    assert costs.round_trip_cost('stock') == pytest.approx(0.0010206)


def test_round_trip_cost_crypto_default():
    assert costs.round_trip_cost('crypto') == pytest.approx(2 * 60 / 1e4)
    assert costs.round_trip_cost('crypto') == pytest.approx(0.012)


def test_round_trip_cost_reads_env_at_call_time(monkeypatch):
    monkeypatch.setenv('STOCK_SLIPPAGE_BPS', '10')
    monkeypatch.setenv('SEC_FEE_RATE', '0')
    assert costs.round_trip_cost('stock') == pytest.approx(0.002)

    monkeypatch.setenv('CRYPTO_SPREAD_BPS', '10')
    assert costs.round_trip_cost('crypto') == pytest.approx(0.002)


def test_round_trip_cost_rejects_unknown_class():
    with pytest.raises(ValueError):
        costs.round_trip_cost('forex')


# ---------------------------------------------------------------- qty_str()

@pytest.mark.parametrize('x, expected', [
    (1.0, '1'),
    (0.5, '0.5'),
    (10.0, '10'),           # rstrip('0') must not eat the integer zero
    (100.0, '100'),
    (0.0, '0'),
    (0.123457, '0.123457'), # a 6dp-rounded qty (what log_trade stores) round-trips
    (0.123456789, '0.123457'),  # unrounded input is formatted to 6dp; no extra digits
    (2.5e-06, '0.000003'),
    (1234.5, '1234.5'),
])
def test_qty_str(x, expected):
    assert costs.qty_str(x) == expected
