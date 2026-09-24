"""Crypto exchange data access layer.

Pulls live quotes and historical daily klines from crypto exchanges'
public REST APIs (no API key required for market data -- these are
unauthenticated endpoints, matching this project's paper-trading/
read-only use case). This gives crypto profiles real exchange data
instead of routing through Yahoo Finance, which has thin/delayed crypto
coverage.

Two independent exchanges are wired in so a single exchange outage (or a
delisted/renamed pair on one of them) doesn't fall all the way back to
Yahoo Finance: MarketDataClient tries Binance first, then Kraken, then
Yahoo Finance as the last resort. See BinanceClient / KrakenClient below.

Universe symbols use the Yahoo-Finance-style "<ASSET>-USD" convention
(e.g. "BTC-USD") so the same universe CSV works across all three sources;
each client maps that to its own exchange's pair convention internally.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger(__name__)

BINANCE_BASE = "https://api.binance.com"
BINANCE_TICKER_24H_URL = BINANCE_BASE + "/api/v3/ticker/24hr"
BINANCE_KLINES_URL = BINANCE_BASE + "/api/v3/klines"
BINANCE_EXCHANGE_INFO_URL = BINANCE_BASE + "/api/v3/exchangeInfo"

KRAKEN_BASE = "https://api.kraken.com/0/public"
KRAKEN_TICKER_URL = KRAKEN_BASE + "/Ticker"
KRAKEN_OHLC_URL = KRAKEN_BASE + "/OHLC"

# Kraken uses its own legacy asset codes for a handful of coins instead of
# the usual ticker (e.g. Bitcoin is "XBT", not "BTC"). Anything not listed
# here is assumed to match its universe symbol as-is.
KRAKEN_ASSET_ALIASES = {
    "BTC": "XBT",
    "DOGE": "XDG",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


class CryptoDataUnavailableError(RuntimeError):
    """Raised when the Binance exchange API could not serve a symbol."""


@dataclass
class CryptoQuote:
    symbol: str  # original "<ASSET>-USD" universe symbol
    ltp: float
    prev_close: float
    week52_high: float
    week52_low: float
    volume: float
    timestamp: datetime
    source: str = "binance"


def to_binance_pair(symbol: str) -> str:
    """"BTC-USD" -> "BTCUSDT". Binance quotes most pairs in USDT, not USD;
    USDT is a USD-pegged stablecoin so this is a like-for-like substitution
    for paper-trading purposes."""
    asset = symbol.split("-")[0]
    return f"{asset}USDT"


class BinanceClient:
    """Thin wrapper around Binance's public (unauthenticated) market-data
    REST endpoints. Read-only -- never places orders; this project's
    paper trading simulates fills internally against these quotes."""

    _MIN_INTERVAL_SECONDS = 0.1  # stay well under Binance's public rate limits

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self._history_cache: dict[str, pd.DataFrame] = {}
        self._last_call = 0.0
        self._known_symbols: Optional[set[str]] = None
        # Once Binance is unreachable (blocked network, outage), stop
        # retrying it for the rest of this process -- same circuit-breaker
        # pattern as NSESession, so a dead endpoint doesn't add per-symbol
        # retry latency across an entire scan.
        self.broken = False

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self._MIN_INTERVAL_SECONDS:
            time.sleep(self._MIN_INTERVAL_SECONDS - elapsed)
        self._last_call = time.time()

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((requests.RequestException,)),
    )
    def _get_json(self, url: str, params: dict | None = None):
        self._throttle()
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def is_supported(self, symbol: str) -> bool:
        """Checks the pair exists on Binance (cached once per process)."""
        if self.broken:
            return False
        if self._known_symbols is None:
            try:
                data = self._get_json(BINANCE_EXCHANGE_INFO_URL)
                self._known_symbols = {s["symbol"] for s in data.get("symbols", [])}
            except Exception as exc:  # noqa: BLE001
                log.warning("Binance exchangeInfo unavailable (%s); disabling Binance for this run", exc)
                self.broken = True
                return False
        return to_binance_pair(symbol) in self._known_symbols

    def get_quote(self, symbol: str) -> CryptoQuote:
        pair = to_binance_pair(symbol)
        try:
            data = self._get_json(BINANCE_TICKER_24H_URL, params={"symbol": pair})
        except Exception as exc:  # noqa: BLE001
            self.broken = True
            raise CryptoDataUnavailableError(f"Binance ticker unavailable for {symbol}: {exc}") from exc
        # The 24hr ticker only carries a 24h high/low, not a true 52-week
        # range. Reuse the (likely already-cached, from momentum/MA calcs)
        # 1y daily history for a real 52-week high/low when we have it;
        # otherwise fall back to the 24h range rather than fail the quote.
        week_high, week_low = float(data["highPrice"]), float(data["lowPrice"])
        cache_key = f"{pair}:365"
        cached_hist = self._history_cache.get(cache_key)
        if cached_hist is not None and not cached_hist.empty:
            week_high = float(cached_hist["High"].max())
            week_low = float(cached_hist["Low"].min())
        return CryptoQuote(
            symbol=symbol,
            ltp=float(data["lastPrice"]),
            prev_close=float(data["prevClosePrice"]),
            week52_high=week_high,
            week52_low=week_low,
            volume=float(data["volume"]),
            timestamp=datetime.now(timezone.utc),
        )

    def get_history(self, symbol: str, days: int = 365, ttl_seconds: int = 900) -> pd.DataFrame:
        """Daily OHLCV klines, shaped like the yfinance history frame this
        project already consumes (Open/High/Low/Close/Volume columns,
        DatetimeIndex) so downstream strategy/momentum code needs no
        source-specific branching."""
        pair = to_binance_pair(symbol)
        cache_key = f"{pair}:{days}"
        cached = self._history_cache.get(cache_key)
        now = time.time()
        if cached is not None and (now - cached.attrs.get("_fetched_at", 0)) < ttl_seconds:
            return cached
        try:
            raw = self._get_json(
                BINANCE_KLINES_URL,
                params={"symbol": pair, "interval": "1d", "limit": min(days, 1000)},
            )
        except Exception as exc:  # noqa: BLE001
            self.broken = True
            raise CryptoDataUnavailableError(f"Binance klines unavailable for {symbol}: {exc}") from exc
        if not raw:
            raise CryptoDataUnavailableError(f"Binance returned no klines for {symbol}")
        # Each kline: [openTime, open, high, low, close, volume, closeTime, ...]
        df = pd.DataFrame(
            raw,
            columns=[
                "OpenTime", "Open", "High", "Low", "Close", "Volume", "CloseTime",
                "QuoteVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore",
            ],
        )
        for col in ("Open", "High", "Low", "Close", "Volume"):
            df[col] = df[col].astype(float)
        df.index = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.attrs["_fetched_at"] = now
        self._history_cache[cache_key] = df
        return df


def to_kraken_pair(symbol: str) -> str:
    """"BTC-USD" -> "XBTUSD" (via KRAKEN_ASSET_ALIASES); "SOL-USD" -> "SOLUSD"."""
    asset = symbol.split("-")[0]
    asset = KRAKEN_ASSET_ALIASES.get(asset, asset)
    return f"{asset}USD"


class KrakenClient:
    """Second-exchange fallback, tried after Binance and before Yahoo
    Finance. Same shape as BinanceClient so MarketDataClient can treat
    them interchangeably."""

    _MIN_INTERVAL_SECONDS = 0.2  # Kraken's public API is stricter than Binance's

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self._history_cache: dict[str, pd.DataFrame] = {}
        self._last_call = 0.0
        # Same circuit-breaker pattern as BinanceClient/NSESession: once
        # Kraken is unreachable, stop retrying it for the rest of this run.
        self.broken = False

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self._MIN_INTERVAL_SECONDS:
            time.sleep(self._MIN_INTERVAL_SECONDS - elapsed)
        self._last_call = time.time()

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((requests.RequestException,)),
    )
    def _get_json(self, url: str, params: dict | None = None):
        self._throttle()
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # Kraken returns HTTP 200 even for pair-not-found / rate-limit
        # errors, with the problem reported in the "error" array instead.
        if data.get("error"):
            raise CryptoDataUnavailableError(f"Kraken API error: {data['error']}")
        return data

    def get_quote(self, symbol: str) -> CryptoQuote:
        pair = to_kraken_pair(symbol)
        try:
            data = self._get_json(KRAKEN_TICKER_URL, params={"pair": pair})
        except Exception as exc:  # noqa: BLE001
            raise CryptoDataUnavailableError(f"Kraken ticker unavailable for {symbol}: {exc}") from exc
        result = data.get("result", {})
        if not result:
            raise CryptoDataUnavailableError(f"Kraken has no ticker for {symbol} ({pair})")
        # Kraken echoes the pair back under its own internal name (which
        # can differ slightly from the request, e.g. "XXBTZUSD"), so just
        # take the single entry rather than matching the key.
        t = next(iter(result.values()))
        ltp = float(t["c"][0])  # last trade: [price, lot volume]
        week_high, week_low = float(t["h"][1]), float(t["l"][1])  # today+yesterday high/low
        cache_key = f"{pair}:365"
        cached_hist = self._history_cache.get(cache_key)
        if cached_hist is not None and not cached_hist.empty:
            week_high = float(cached_hist["High"].max())
            week_low = float(cached_hist["Low"].min())
        return CryptoQuote(
            symbol=symbol,
            ltp=ltp,
            prev_close=float(t["o"]),  # today's opening price, closest analogue Kraken exposes
            week52_high=week_high,
            week52_low=week_low,
            volume=float(t["v"][1]),  # 24h volume
            timestamp=datetime.now(timezone.utc),
            source="kraken",
        )

    def get_history(self, symbol: str, days: int = 365, ttl_seconds: int = 900) -> pd.DataFrame:
        pair = to_kraken_pair(symbol)
        cache_key = f"{pair}:{days}"
        cached = self._history_cache.get(cache_key)
        now = time.time()
        if cached is not None and (now - cached.attrs.get("_fetched_at", 0)) < ttl_seconds:
            return cached
        try:
            # interval=1440 minutes == daily candles.
            data = self._get_json(KRAKEN_OHLC_URL, params={"pair": pair, "interval": 1440})
        except Exception as exc:  # noqa: BLE001
            raise CryptoDataUnavailableError(f"Kraken OHLC unavailable for {symbol}: {exc}") from exc
        result = data.get("result", {})
        rows = None
        for key, value in result.items():
            if key != "last":
                rows = value
                break
        if not rows:
            raise CryptoDataUnavailableError(f"Kraken returned no OHLC for {symbol}")
        # Each row: [time, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(
            rows,
            columns=["Time", "Open", "High", "Low", "Close", "VWAP", "Volume", "Count"],
        )
        for col in ("Open", "High", "Low", "Close", "Volume"):
            df[col] = df[col].astype(float)
        df.index = pd.to_datetime(df["Time"], unit="s", utc=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].tail(days)
        df.attrs["_fetched_at"] = now
        self._history_cache[cache_key] = df
        return df
