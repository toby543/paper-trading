"""Market data access layer.

Real-time quotes and 52-week high/low are pulled from the official (but
unofficial/undocumented) NSE website JSON API. Historical daily bars used
for moving averages and momentum-return calculations come from Yahoo
Finance (``yfinance``), which mirrors NSE prices under the ``.NS`` suffix.

If the NSE endpoint is unreachable (common from data-center / CI IPs,
which NSE's anti-bot layer frequently blocks), everything transparently
falls back to Yahoo Finance so the system keeps working end-to-end.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger(__name__)

NSE_BASE = "https://www.nseindia.com"
NSE_QUOTE_URL = NSE_BASE + "/api/quote-equity?symbol={symbol}"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


class DataUnavailableError(RuntimeError):
    """Raised when neither NSE nor the fallback source could serve a quote."""


@dataclass
class Quote:
    symbol: str
    ltp: float
    prev_close: float
    week52_high: float
    week52_low: float
    volume: float
    timestamp: datetime
    source: str  # "nse" or "yfinance"


class NSESession:
    """Thin wrapper that warms up cookies against nseindia.com."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self._warmed = False

    def _warm_up(self) -> None:
        if self._warmed:
            return
        self.session.get(NSE_BASE, timeout=self.timeout)
        self.session.get(NSE_BASE + "/get-quotes/equity?symbol=RELIANCE", timeout=self.timeout)
        self._warmed = True

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((requests.RequestException,)),
    )
    def get_json(self, url: str) -> dict:
        self._warm_up()
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code == 401 or resp.status_code == 403:
            # Session likely stale; force a fresh warm-up on next attempt.
            self._warmed = False
            resp.raise_for_status()
        resp.raise_for_status()
        return resp.json()


class MarketDataClient:
    """Facade combining NSE (preferred) and Yahoo Finance (fallback)."""

    # Minimum spacing between consecutive Yahoo Finance requests. With a
    # 100-500 symbol universe, firing requests back-to-back with no pacing
    # is exactly the pattern that trips Yahoo's rate limiting/anti-abuse
    # throttling -- which then looks indistinguishable from real symbols
    # being "delisted" (both surface as a 404) unless you know to suspect
    # rate limiting first when several genuinely-listed, actively-traded
    # stocks fail in the same run.
    _YFINANCE_MIN_INTERVAL_SECONDS = 0.2

    def __init__(self, preferred: str = "nse", fallback: str = "yfinance", timeout: int = 10):
        self.preferred = preferred
        self.fallback = fallback
        self.timeout = timeout
        self._nse = NSESession(timeout=timeout)
        self._history_cache: dict[str, pd.DataFrame] = {}
        # Sector and earnings-date lookups hit yfinance's slower `.info` /
        # calendar endpoints (nothing NSE-native serves this), and both are
        # effectively static for the lifetime of one process run -- cache
        # indefinitely per symbol rather than re-fetching every scan.
        self._sector_cache: dict[str, Optional[str]] = {}
        self._earnings_cache: dict[str, Optional[datetime]] = {}
        self._last_yfinance_call = 0.0
        # Once NSE fails once, its anti-bot layer is almost always blocking
        # this network for the rest of the run too. Retrying it per-symbol
        # (with backoff) across a large universe is what makes scans take
        # minutes, so trip a circuit breaker and go straight to the
        # fallback for the remainder of this process's lifetime.
        self._nse_broken = False

    def _throttle_yfinance(self) -> None:
        elapsed = time.time() - self._last_yfinance_call
        if elapsed < self._YFINANCE_MIN_INTERVAL_SECONDS:
            time.sleep(self._YFINANCE_MIN_INTERVAL_SECONDS - elapsed)
        self._last_yfinance_call = time.time()

    # ---- live quotes -----------------------------------------------
    def get_quote(self, symbol: str) -> Quote:
        if self.preferred == "nse" and not self._nse_broken:
            try:
                return self._quote_from_nse(symbol)
            except Exception as exc:  # noqa: BLE001 - deliberately broad, we fall back
                log.warning("NSE quote failed for %s (%s); falling back to yfinance for the rest of this run", symbol, exc)
                self._nse_broken = True
        try:
            return self._quote_from_yfinance(symbol)
        except Exception as exc:  # noqa: BLE001
            raise DataUnavailableError(f"No data source available for {symbol}: {exc}") from exc

    def _quote_from_nse(self, symbol: str) -> Quote:
        data = self._nse.get_json(NSE_QUOTE_URL.format(symbol=symbol))
        price_info = data["priceInfo"]
        week = price_info.get("weekHighLow", {})
        return Quote(
            symbol=symbol,
            ltp=float(price_info["lastPrice"]),
            prev_close=float(price_info["previousClose"]),
            week52_high=float(week.get("max") or price_info["lastPrice"]),
            week52_low=float(week.get("min") or price_info["lastPrice"]),
            volume=float(data.get("marketDeptOrderBook", {}).get("tradeInfo", {}).get("totalTradedVolume", 0) or 0),
            timestamp=datetime.now(),
            source="nse",
        )

    def _quote_from_yfinance(self, symbol: str) -> Quote:
        import yfinance as yf

        self._throttle_yfinance()
        ticker = yf.Ticker(symbol + ".NS")
        hist = ticker.history(period="1y", interval="1d")
        if hist.empty:
            raise DataUnavailableError(f"yfinance returned no history for {symbol}")
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else last
        return Quote(
            symbol=symbol,
            ltp=float(last["Close"]),
            prev_close=float(prev["Close"]),
            week52_high=float(hist["High"].max()),
            week52_low=float(hist["Low"].min()),
            volume=float(last["Volume"]),
            timestamp=datetime.now(),
            source="yfinance",
        )

    # ---- historical bars (for MAs / momentum returns) ----------------
    def get_history(self, symbol: str, period: str = "1y", ttl_seconds: int = 900) -> pd.DataFrame:
        cache_key = f"{symbol}:{period}"
        cached = self._history_cache.get(cache_key)
        now = time.time()
        if cached is not None and (now - cached.attrs.get("_fetched_at", 0)) < ttl_seconds:
            return cached
        import yfinance as yf

        self._throttle_yfinance()
        df = yf.Ticker(symbol + ".NS").history(period=period, interval="1d")
        if df.empty:
            raise DataUnavailableError(f"No history for {symbol}")
        df.attrs["_fetched_at"] = now
        self._history_cache[cache_key] = df
        return df

    def get_index_history(self, index_symbol: str, period: str = "1y", ttl_seconds: int = 900) -> pd.DataFrame:
        """Like get_history, but for a raw Yahoo Finance ticker (e.g. the
        "^NSEI" Nifty 50 index) that must NOT get the ".NS" equity suffix."""
        cache_key = f"index:{index_symbol}:{period}"
        cached = self._history_cache.get(cache_key)
        now = time.time()
        if cached is not None and (now - cached.attrs.get("_fetched_at", 0)) < ttl_seconds:
            return cached
        import yfinance as yf

        self._throttle_yfinance()
        df = yf.Ticker(index_symbol).history(period=period, interval="1d")
        if df.empty:
            raise DataUnavailableError(f"No history for index {index_symbol}")
        df.attrs["_fetched_at"] = now
        self._history_cache[cache_key] = df
        return df

    # ---- sector / earnings (used by portfolio-construction filters) --
    def get_sector(self, symbol: str) -> Optional[str]:
        """GICS-style sector name for `symbol`, or None if it can't be
        determined -- callers should fail open (never block a trade) on
        None, since this is a diversification nicety, not a core signal."""
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]
        sector: Optional[str] = None
        try:
            import yfinance as yf

            self._throttle_yfinance()
            info = yf.Ticker(symbol + ".NS").info
            sector = info.get("sector") or None
        except Exception as exc:  # noqa: BLE001 - best-effort classification only
            log.debug("Could not fetch sector for %s: %s", symbol, exc)
        self._sector_cache[symbol] = sector
        return sector

    def get_next_earnings_date(self, symbol: str) -> Optional[datetime]:
        """The next scheduled earnings date for `symbol`, or None if it's
        unknown/unavailable. Best-effort: yfinance's earnings calendar is
        itself sourced from estimates and can be missing, stale, or absent
        for smaller names -- callers should fail open (never block a trade)
        on None, same as every other optional filter in this codebase."""
        if symbol in self._earnings_cache:
            return self._earnings_cache[symbol]
        next_date: Optional[datetime] = None
        try:
            import yfinance as yf

            self._throttle_yfinance()
            ticker = yf.Ticker(symbol + ".NS")
            dates_df = ticker.get_earnings_dates(limit=4)
            if dates_df is not None and not dates_df.empty:
                now = pd.Timestamp.now(tz=dates_df.index.tz)
                future = dates_df.index[dates_df.index >= now]
                if len(future):
                    next_date = future.min().to_pydatetime()
        except Exception as exc:  # noqa: BLE001 - best-effort, never fatal
            log.debug("Could not fetch earnings date for %s: %s", symbol, exc)
        self._earnings_cache[symbol] = next_date
        return next_date

    def get_avg_daily_turnover(self, symbol: str, days: int = 20, history: pd.DataFrame | None = None) -> float:
        # Reuse an already-fetched history frame when the caller has one
        # (the 1y frame pulled for moving averages/momentum easily covers
        # the last 20 sessions) instead of making a second network round
        # trip per symbol just for this filter.
        df = history if history is not None else self.get_history(symbol, period="3mo")
        recent = df.tail(days)
        turnover = (recent["Close"] * recent["Volume"]).mean()
        return float(turnover) if not pd.isna(turnover) else 0.0
