"""Trading-calendar helpers in src.intraday_engine.

Every call passes now=/ts= explicitly; nothing here touches the wall clock.
Dates used (all 2026 unless noted):
  Mon 09-14 .. Fri 09-18 is the account's first week; Sat 09-19 / Sun 09-20 weekend
  Thu 11-26 Thanksgiving (holiday), Fri 11-27 early close 13:00 ET
  Thu 12-31, Fri 2027-01-01 New Year's Day (holiday), Mon 2027-01-04
"""
from datetime import date, datetime, timedelta, timezone

import pytest

from src.intraday_engine import (ET, US_HOLIDAYS_2026, is_trading_day,
                                 market_state, next_trading_day_open)


def et(y, m, d, hh, mm, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=ET)


# --- market_state: behaviour must be unchanged by the refactor -------------

@pytest.mark.parametrize('now, expected', [
    (et(2026, 9, 15, 10, 0), ('open', 'regular session')),          # Tue mid-session
    (et(2026, 9, 15, 8, 0), ('premarket', 'pre-market')),            # Tue 08:00
    (et(2026, 9, 15, 17, 0), ('afterhours', 'after hours')),         # Tue 17:00
    (et(2026, 9, 15, 2, 0), ('closed', 'overnight')),                # Tue 02:00
    (et(2026, 9, 19, 12, 0), ('closed', 'weekend')),                 # Sat
    (et(2026, 9, 20, 12, 0), ('closed', 'weekend')),                 # Sun
    (et(2026, 11, 26, 12, 0), ('closed', 'market holiday')),         # Thanksgiving
    (et(2027, 1, 1, 12, 0), ('closed', 'market holiday')),           # New Year's Day 2027
    (et(2026, 11, 27, 12, 0), ('open', 'early close 13:00 ET')),     # early-close Friday
    # 13:00 on an early-close day: not open, and the after-hours branch only
    # starts at 16:00, so the existing code falls through to 'overnight'.
    # Pinned as-is (pre-existing behaviour); changing it is out of scope.
    (et(2026, 11, 27, 13, 0), ('closed', 'overnight')),
])
def test_market_state_cases(now, expected):
    assert market_state(now=now) == expected


def test_market_state_converts_utc_input():
    # 14:00 UTC on Tue 09-15 is 10:00 EDT -> regular session
    assert market_state(now=datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc)) == ('open', 'regular session')


def test_market_state_crypto_always_open():
    assert market_state(now=et(2026, 9, 19, 12, 0), symbol='BTC-USD') == ('open', '24/7 crypto')


# --- is_trading_day -------------------------------------------------------

def test_2027_new_years_day_is_in_holiday_set():
    assert '2027-01-01' in US_HOLIDAYS_2026


@pytest.mark.parametrize('d, expected', [
    (date(2026, 9, 14), True),    # Mon
    (date(2026, 9, 18), True),    # Fri
    (date(2026, 9, 19), False),   # Sat
    (date(2026, 9, 20), False),   # Sun
    (date(2026, 11, 26), False),  # Thanksgiving
    (date(2026, 11, 27), True),   # early-close day is still a trading day
    (date(2026, 12, 25), False),  # Christmas
    (date(2027, 1, 1), False),    # New Year's Day 2027 (the added entry)
    (date(2027, 1, 4), True),     # first trading day of 2027
])
def test_is_trading_day(d, expected):
    assert is_trading_day(d) is expected


# --- next_trading_day_open ------------------------------------------------

def test_friday_sell_settles_monday_0930_et():
    got = next_trading_day_open(et(2026, 9, 18, 15, 55))
    assert got == et(2026, 9, 21, 9, 30)
    assert got.strftime('%a') == 'Mon'


def test_wednesday_before_thanksgiving_settles_friday():
    # Thu 11-26 is a holiday, so T+1 from Wed 11-25 is Fri 11-27.
    assert next_trading_day_open(et(2026, 11, 25, 12, 0)) == et(2026, 11, 27, 9, 30)


def test_new_years_eve_settles_first_trading_day_of_2027():
    # Fri 2027-01-01 is a holiday, then Sat/Sun -> Mon 2027-01-04.
    assert next_trading_day_open(et(2026, 12, 31, 15, 0)) == et(2027, 1, 4, 9, 30)


def test_strictly_after_the_input_date():
    # Monday 09:00 ET (pre-market) must NOT return the same Monday's 09:30.
    assert next_trading_day_open(et(2026, 9, 14, 9, 0)) == et(2026, 9, 15, 9, 30)


def test_utc_input_is_converted_to_et_date():
    # 02:00 UTC on Tue 09-15 is still Mon 09-14 22:00 EDT, so the answer is Tue 09-15,
    # not Wed 09-16 (which is what a naive .date() on the UTC value would give).
    got = next_trading_day_open(datetime(2026, 9, 15, 2, 0, tzinfo=timezone.utc))
    assert got == et(2026, 9, 15, 9, 30)


def test_result_is_tz_aware_et():
    got = next_trading_day_open(datetime(2026, 9, 18, 19, 55, tzinfo=timezone.utc))
    assert got.tzinfo is ET
    assert got.hour == 9 and got.minute == 30 and got.second == 0 and got.microsecond == 0
    # EDT on 09-21: the UTC rendering Task 4 stores must be 13:30Z
    assert got.astimezone(timezone.utc) == datetime(2026, 9, 21, 13, 30, tzinfo=timezone.utc)
    assert got.utcoffset() == timedelta(hours=-4)


def test_naive_datetime_is_rejected():
    # A naive value would be read as system-local time (CST/CDT on LXC 200) and
    # silently land an evening SELL on the wrong settlement date. Fail loud.
    with pytest.raises(ValueError):
        next_trading_day_open(datetime(2026, 9, 18, 15, 55))


# --- completed bars + minutes since open (2026-09-17) ------------------------

def test_minutes_since_open():
    from src.intraday_engine import minutes_since_open
    assert minutes_since_open(et(2026, 9, 17, 9, 31)) == pytest.approx(1.0)
    assert minutes_since_open(et(2026, 9, 17, 9, 0)) == pytest.approx(-30.0)
    assert minutes_since_open(datetime(2026, 9, 17, 13, 46, tzinfo=timezone.utc)) == pytest.approx(16.0)


def test_completed_bars_drops_only_a_forming_last_bar():
    import pandas as pd
    from src.intraday_engine import completed_bars
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz='America/New_York')
                            for t in ('2026-09-17 09:25', '2026-09-17 09:30')])
    h = pd.DataFrame({'close': [1.0, 2.0]}, index=idx)
    # 09:31:34 ET: the 09:30 5m bar runs until 09:35 - still forming
    assert list(completed_bars(h, now=et(2026, 9, 17, 9, 31, 34), interval='5m')['close']) == [1.0]
    # 09:35:01 ET: it is finished
    assert list(completed_bars(h, now=et(2026, 9, 17, 9, 35, 1), interval='5m')['close']) == [1.0, 2.0]
    assert completed_bars(h.iloc[:0], now=et(2026, 9, 17, 9, 31, 34)).empty
