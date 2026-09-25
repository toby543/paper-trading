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


# --------------------------------------------------------- backtest parity
# The live data layer converting crypto to the book's currency is only half
# the requirement: the BACKTEST fetches its own history straight from
# yfinance, so it needs the same conversion or the two disagree. It didn't,
# and the liquidity filter was the visible casualty -- USD turnover compared
# against min_avg_daily_turnover_inr is a bar ~96x too high, which silently
# rejected roughly half the universe every day as illiquid while the live
# engine traded those same coins.


def test_backtester_converts_crypto_history_to_book_currency():
    from papertrader.backtest.engine import Backtester

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    usd_frame = pd.DataFrame(
        {"Open": [1.0, 2.0, 3.0], "High": [1.0, 2.0, 3.0],
         "Low": [1.0, 2.0, 3.0], "Close": [1.0, 2.0, 3.0],
         "Volume": [10.0, 10.0, 10.0]}, index=idx)

    out = Backtester._to_book_currency(usd_frame, 90.0)

    assert list(out["Close"]) == [90.0, 180.0, 270.0]
    assert list(out["High"]) == [90.0, 180.0, 270.0]
    # Volume stays a coin count, exactly as in the live path -- so
    # Close*Volume turnover lands in the book's currency on its own.
    assert list(out["Volume"]) == [10.0, 10.0, 10.0]
    # Frames come from the shared price cache; converting must copy, not
    # mutate, or every later read in the same run compounds the rate.
    assert list(usd_frame["Close"]) == [1.0, 2.0, 3.0]


def test_backtester_conversion_is_a_noop_for_a_usd_book():
    """fx == 1.0 must return the frame untouched, so a USD-denominated
    book (or an equity universe) costs nothing and copies nothing."""
    from papertrader.backtest.engine import Backtester

    frame = pd.DataFrame({"Close": [1.0, 2.0], "Volume": [10.0, 10.0]})
    assert Backtester._to_book_currency(frame, 1.0) is frame


def test_backtest_and_live_agree_on_converted_turnover(monkeypatch):
    """The actual regression: the same coin must clear (or fail) the
    liquidity bar identically in a backtest and in live trading."""
    monkeypatch.setattr(fx, "usd_to_inr_rate", lambda timeout=10: 90.0)
    from papertrader.backtest.engine import Backtester

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    usd_frame = pd.DataFrame(
        {"Open": [100.0] * 3, "High": [100.0] * 3, "Low": [100.0] * 3,
         "Close": [100.0] * 3, "Volume": [1000.0] * 3}, index=idx)

    class _StubExchange:
        broken = False
        def get_history(self, symbol, days=365, ttl_seconds=900):
            return usd_frame

    client = MarketDataClient(quote_currency="INR")
    client._binance = _StubExchange()
    live = client.get_history("BTC-USD")
    backtested = Backtester._to_book_currency(usd_frame, fx.conversion_rate("USD", "INR"))

    live_turnover = float((live["Close"] * live["Volume"]).mean())
    backtest_turnover = float((backtested["Close"] * backtested["Volume"]).mean())
    assert backtest_turnover == pytest.approx(live_turnover)
    # And both are in INR, i.e. 90x the raw USD figure -- the thing the
    # INR-denominated threshold is actually comparable against.
    assert backtest_turnover == pytest.approx(100.0 * 1000.0 * 90.0)
