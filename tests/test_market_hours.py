import os
from datetime import datetime
from zoneinfo import ZoneInfo

from papertrader.engine.market_hours import MarketCalendar

HOLIDAYS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nse_holidays.csv")


def _cal():
    return MarketCalendar("Asia/Kolkata", "09:15", "15:30", HOLIDAYS_FILE)


def test_weekday_during_hours_is_open():
    cal = _cal()
    dt = datetime(2026, 8, 25, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # Tuesday
    assert cal.is_market_open(dt)


def test_weekend_is_closed():
    cal = _cal()
    dt = datetime(2026, 8, 23, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # Sunday
    assert not cal.is_market_open(dt)


def test_before_open_is_closed():
    cal = _cal()
    dt = datetime(2026, 8, 25, 8, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert not cal.is_market_open(dt)


def test_after_close_is_closed():
    cal = _cal()
    dt = datetime(2026, 8, 25, 16, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert not cal.is_market_open(dt)


def test_holiday_is_closed():
    cal = _cal()
    dt = datetime(2026, 8, 15, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # Independence Day
    assert not cal.is_market_open(dt)
