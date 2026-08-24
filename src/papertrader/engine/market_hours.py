"""NSE market-hours and holiday-calendar helpers."""
from __future__ import annotations

import csv
import os
from datetime import datetime, time
from zoneinfo import ZoneInfo


def load_holidays(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return {row["date"].strip() for row in reader if row.get("date")}


class MarketCalendar:
    def __init__(self, timezone: str, market_open: str, market_close: str, holidays_file: str):
        self.tz = ZoneInfo(timezone)
        self.open_time = time.fromisoformat(market_open)
        self.close_time = time.fromisoformat(market_close)
        self.holidays = load_holidays(holidays_file)

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def is_holiday(self, dt: datetime) -> bool:
        return dt.strftime("%Y-%m-%d") in self.holidays

    def is_market_open(self, dt: datetime | None = None) -> bool:
        dt = dt or self.now()
        if dt.weekday() >= 5:  # Sat/Sun
            return False
        if self.is_holiday(dt):
            return False
        return self.open_time <= dt.time() <= self.close_time
