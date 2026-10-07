"""Official end-of-day prices from the exchanges' own daily files.

The last line of defence for a stock's price. Yahoo Finance sometimes ends a
thin stock's history with a bar that has a volume but no prices, or has no
data for a symbol at all; NSE and BSE publish every listed stock's open, high,
low and close in a daily file after the close, with no key and no throttle.

    NSE   sec_bhavdata_full_DDMMYYYY.csv      (symbols as listed, e.g. RPEL)
    BSE   BhavCopy_BSE_CM_0_0_0_YYYYMMDD_F_0000.CSV   (symbols as "ABB.BO")
"""
from __future__ import annotations

import io
import logging
import threading
import time
from datetime import date, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

NSE_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"
BSE_URL = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{d}_F_0000.CSV"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bseindia.com/"}
BSE_SUFFIX = ".BO"
# A published file never changes, so it is cached for good; a miss (not yet
# published, a holiday, the site blocking us) is retried after this long.
_RETRY_AFTER_SECONDS = 300
# How far back to look for the latest published file (weekends, holidays).
LOOKBACK_DAYS = 7

_lock = threading.Lock()
_cache: dict[tuple[str, date], tuple[float, dict | None]] = {}


def _num(value) -> float:
    return float(pd.to_numeric(value, errors="coerce"))


def parse_nse(text: str) -> dict[str, dict]:
    """symbol -> prices; the EQ series wins when a symbol trades in several."""
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df = df.sort_values("SERIES", key=lambda s: (s != "EQ"))  # EQ first
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        symbol = str(r["SYMBOL"]).strip().upper()
        close = _num(r["CLOSE_PRICE"])
        if symbol in out or pd.isna(close) or close <= 0:
            continue
        out[symbol] = {
            "open": _num(r["OPEN_PRICE"]), "high": _num(r["HIGH_PRICE"]), "low": _num(r["LOW_PRICE"]),
            "close": close, "prev_close": _num(r["PREV_CLOSE"]), "volume": _num(r["TTL_TRD_QNTY"]),
        }
    return out


def parse_bse(text: str) -> dict[str, dict]:
    """ticker (without the .BO suffix) -> prices."""
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        symbol = str(r["TckrSymb"]).strip().upper()
        close = _num(r["ClsPric"])
        if symbol in out or pd.isna(close) or close <= 0:
            continue
        out[symbol] = {
            "open": _num(r["OpnPric"]), "high": _num(r["HghPric"]), "low": _num(r["LwPric"]),
            "close": close, "prev_close": _num(r["PrvsClsgPric"]), "volume": _num(r["TtlTradgVol"]),
        }
    return out


def _load(exchange: str, day: date, timeout: float) -> dict | None:
    key = (exchange, day)
    with _lock:
        hit = _cache.get(key)
        if hit and (hit[1] is not None or time.time() - hit[0] < _RETRY_AFTER_SECONDS):
            return hit[1]
    table = None
    try:
        url = (NSE_URL if exchange == "NSE" else BSE_URL).format(
            d=day.strftime("%d%m%Y" if exchange == "NSE" else "%Y%m%d"))
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        if resp.status_code == 200 and resp.text:
            table = (parse_nse if exchange == "NSE" else parse_bse)(resp.text)
    except Exception as exc:  # noqa: BLE001 - a fallback source must never raise
        log.debug("%s end-of-day file for %s unavailable: %s", exchange, day, exc)
    with _lock:
        _cache[key] = (time.time(), table)
    return table


def _split(symbol: str) -> tuple[str, str]:
    sym = symbol.strip().upper()
    return ("BSE", sym[: -len(BSE_SUFFIX)]) if sym.endswith(BSE_SUFFIX) else ("NSE", sym)


def lookup(symbol: str, day: date, timeout: float = 10) -> dict | None:
    """Official prices for `symbol` on `day`, or None when not available."""
    exchange, ticker = _split(symbol)
    table = _load(exchange, day, timeout)
    return table.get(ticker) if table else None


def latest(symbol: str, on_or_before: date, timeout: float = 10) -> tuple[date, dict] | None:
    """The most recent published day (weekends skipped) that has prices for
    `symbol`, searching back up to LOOKBACK_DAYS."""
    for back in range(LOOKBACK_DAYS + 1):
        day = on_or_before - timedelta(days=back)
        if day.weekday() >= 5:
            continue
        row = lookup(symbol, day, timeout)
        if row:
            return day, row
    return None
