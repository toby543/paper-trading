"""On-disk cache for historical daily OHLCV bars, so repeat backtests
over already-fetched symbols/date-ranges don't need to hit Yahoo
Finance again. One CSV per symbol under data/price_cache/ -- readable
in a text editor if something looks off, and cheap enough in total size
(a few hundred KB per symbol) that keeping it in `data/` alongside
everything else is fine.

Cache validity: a cached file only satisfies a fetch request if it
already covers the ENTIRE requested [start, end) range. Anything
narrower triggers a full re-fetch of that range from the network, which
is then merged into (and extends) the cache for next time -- there's no
partial-range stitching, which keeps this simple and correct at the
cost of occasionally re-fetching a day or two of overlap.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from ..config import REPO_ROOT

log = logging.getLogger(__name__)

CACHE_DIR = os.path.join(REPO_ROOT, "data", "price_cache")


def _cache_path(symbol: str) -> str:
    # Index symbols like "^NSEI" contain characters that aren't safe in
    # filenames on Windows -- replace them rather than reject them.
    safe = symbol.replace("^", "_idx_").replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def load(symbol: str) -> pd.DataFrame | None:
    """Full cached history for `symbol`, or None if there's no cache
    file yet (or it can't be read)."""
    path = _cache_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as exc:  # noqa: BLE001 - a corrupt cache file must not abort a backtest
        log.warning("Could not read price cache for %s (%s); will re-fetch", symbol, exc)
        return None


def covers(df: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True if `df` already has bars spanning [start, end) -- i.e. a
    fetch for that exact window can be served entirely from the cache
    without touching the network."""
    if df is None or df.empty:
        return False
    return bool(df.index.min() <= start and df.index.max() >= end - pd.Timedelta(days=1))


def slice_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index < end)]


def save(symbol: str, fresh: pd.DataFrame, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge freshly-fetched bars into whatever was already cached,
    write the combined history back to disk, and return it -- the
    cache only ever grows to cover more of the timeline, never shrinks."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = fresh.sort_index()
    combined.to_csv(_cache_path(symbol))
    return combined
