"""Crypto quotes arrive in USD; an INR book must convert them."""
import pandas as pd
import pytest

from papertrader.data import fx
from papertrader.data.nse_client import MarketDataClient


def test_conversion_rate_is_identity_for_same_currency():
    # Callers apply the rate unconditionally, so same-currency must be a
    # no-op rather than an error or a network call.
    assert fx.conversion_rate("USD", "USD") == 1.0
    assert fx.conversion_rate("INR", "INR") == 1.0


def test_conversion_rate_usd_inr_round_trips(monkeypatch):
    monkeypatch.setattr(fx, "usd_to_inr_rate", lambda timeout=10: 90.0)
    assert fx.conversion_rate("USD", "INR") == 90.0
    assert fx.conversion_rate("INR", "USD") == pytest.approx(1 / 90.0)


def test_unsupported_pair_raises():
    with pytest.raises(ValueError):
        fx.conversion_rate("USD", "EUR")


def test_usd_book_leaves_crypto_prices_untouched(monkeypatch):
    monkeypatch.setattr(fx, "usd_to_inr_rate", lambda timeout=10: 90.0)
    client = MarketDataClient(quote_currency="USD")
    assert client._crypto_fx_rate() == 1.0


def test_inr_book_converts_crypto_prices(monkeypatch):
    monkeypatch.setattr(fx, "usd_to_inr_rate", lambda timeout=10: 90.0)
    client = MarketDataClient(quote_currency="INR")
    assert client._crypto_fx_rate() == 90.0


def test_history_conversion_scales_prices_but_not_volume(monkeypatch):
    """Volume is a coin count, not a price. Scaling it would inflate
    turnover (Close*Volume) by the square of the FX rate."""
    monkeypatch.setattr(fx, "usd_to_inr_rate", lambda timeout=10: 90.0)

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    usd_frame = pd.DataFrame(
        {"Open": [1.0, 2.0, 3.0], "High": [1.0, 2.0, 3.0],
         "Low": [1.0, 2.0, 3.0], "Close": [1.0, 2.0, 3.0],
         "Volume": [10.0, 10.0, 10.0]}, index=idx)

    class _StubExchange:
        broken = False
        def get_history(self, symbol, days=365, ttl_seconds=900):
            return usd_frame

    client = MarketDataClient(quote_currency="INR")
    client._binance = _StubExchange()

    out = client.get_history("BTC-USD")
    assert list(out["Close"]) == [90.0, 180.0, 270.0]
    assert list(out["Volume"]) == [10.0, 10.0, 10.0]
    # The stub's frame is its own cache; converting must not mutate it.
    assert list(usd_frame["Close"]) == [1.0, 2.0, 3.0]


def test_equity_symbols_are_never_converted(monkeypatch):
    """NSE equities are already INR -- applying FX to them would be a
    ~90x error in the opposite direction."""
    monkeypatch.setattr(fx, "usd_to_inr_rate", lambda timeout=10: 90.0)
    client = MarketDataClient(quote_currency="INR")
    assert client._is_crypto_symbol("RELIANCE") is False
    assert client._is_crypto_symbol("BTC-USD") is True
