"""FX conversion for instruments quoted in a currency other than the
book's own.

Crypto pairs come off Binance/Kraken quoted in USD ("BTC-USD"), but a
book denominated in rupees has to hold them in rupees -- otherwise a
$84,000 coin sits in an INR ledger as "84,000", understating it by the
whole exchange rate and making equity, P&L and position sizing wrong.
This module supplies the USD->INR rate used to convert those quotes at
the data layer, so everything downstream (strategy thresholds, position
sizing, the broker, stored ledger values, the dashboard) sees one
consistent currency and needs no conversion logic of its own.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

# Yahoo Finance's spot USD/INR. Same dependency the rest of the data
# layer already falls back to, so this adds no new integration.
_USD_INR_TICKER = "USDINR=X"

# FX moves on a far slower timescale than the crypto quotes it's applied
# to, so an hour-old rate is perfectly good and avoids a network round
# trip on every single symbol of every scan.
_TTL_SECONDS = 3600

# Used only if the rate cannot be fetched at all (no network, Yahoo down)
# AND nothing has been cached yet this process. Deliberately a plausible
# recent rate rather than 1.0: a 1.0 fallback would silently price every
# crypto holding at 1/90th of its real value, which looks like a
# catastrophic loss rather than like the data problem it actually is.
_FALLBACK_USD_INR = 90.0


class _RateCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rate: float | None = None
        self._fetched_at = 0.0

    def get(self, timeout: int = 10) -> float:
        with self._lock:
            now = time.time()
            if self._rate is not None and (now - self._fetched_at) < _TTL_SECONDS:
                return self._rate
            try:
                import yfinance as yf

                hist = yf.Ticker(_USD_INR_TICKER).history(period="5d", interval="1d", timeout=timeout)
                if hist.empty:
                    raise ValueError("no USD/INR history returned")
                rate = float(hist["Close"].iloc[-1])
                if not (rate > 0):
                    raise ValueError(f"implausible USD/INR rate {rate}")
                self._rate = rate
                self._fetched_at = now
                log.info("USD/INR rate refreshed: %.4f", rate)
                return rate
            except Exception as exc:  # noqa: BLE001 - never let FX break a scan
                if self._rate is not None:
                    # A stale rate is far better than no rate: it keeps the
                    # book in the right order of magnitude while the fetch
                    # recovers on a later cycle.
                    log.warning("USD/INR refresh failed (%s); reusing cached %.4f", exc, self._rate)
                    return self._rate
                log.warning("USD/INR unavailable (%s); falling back to %.2f", exc, _FALLBACK_USD_INR)
                return _FALLBACK_USD_INR


_cache = _RateCache()


def usd_to_inr_rate(timeout: int = 10) -> float:
    """Current USD->INR rate, cached for an hour across all callers."""
    return _cache.get(timeout=timeout)


def conversion_rate(from_currency: str, to_currency: str, timeout: int = 10) -> float:
    """Multiplier taking a price quoted in `from_currency` into
    `to_currency`. 1.0 when they match, so callers can apply it
    unconditionally without branching."""
    src = (from_currency or "").upper()
    dst = (to_currency or "").upper()
    if src == dst:
        return 1.0
    if src == "USD" and dst == "INR":
        return usd_to_inr_rate(timeout=timeout)
    if src == "INR" and dst == "USD":
        return 1.0 / usd_to_inr_rate(timeout=timeout)
    raise ValueError(f"No conversion available for {src}->{dst}")
