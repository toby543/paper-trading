import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


import pytest


@pytest.fixture(autouse=True)
def _no_live_end_of_day_files(monkeypatch):
    """The end-of-day price fallback fetches NSE/BSE files over the network.
    Tests must not depend on that, so it finds nothing unless a test provides
    its own table via eod_prices._load."""
    from papertrader.data import eod_prices

    eod_prices._cache.clear()
    monkeypatch.setattr(eod_prices, "_load", lambda exchange, day, timeout: None)
