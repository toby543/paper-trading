"""IPO base breakout strategy.

A stock that IPO'd recently has no long-term shareholders sitting on
losses at higher prices -- the "overhead resistance" that makes an
established large-cap's first breakout attempt often fail and retest.
The classic pattern (O'Neil/Minervini-style "IPO base" trading) is: let
the initial post-listing volatility settle, wait for the stock to carve
out its first tight consolidation, then buy the breakout above that base
on strong volume. The entry logic here is deliberately close to
consolidation_breakout.py's -- same base-detection/breakout mechanics --
with one structural difference that defines this strategy: it actively
REQUIRES a short trading history (a recent listing) rather than rejecting
one, and uses shorter moving averages to match, since a stock 4-18 months
old cannot support a 50/200-day MA pair the way an established name can.

Entry filters:
  1. Recent listing: available history is between min_listing_days and
     max_listing_days -- old enough to have formed a real base, young
     enough that "no overhead resistance" is still true. A stock with
     more history than max_listing_days is no longer a fresh IPO for
     this strategy's purposes, even if it still trades like one.
  2. Uptrend: price > fast MA > slow MA (10/30 by default, not the
     established-stock 50/200 pair -- there usually isn't 200 days of
     history to compute that from this early).
  3. Tight consolidation base over `base_days`, then a breakout above it
     on volume >= `volume_multiple` the baseline.
  4. Liquidity and minimum momentum, same convention as every other
     strategy here.

Exit rules mirror consolidation_breakout.py exactly: hard stop-loss,
trailing stop from the peak since entry, optional take-profit, and a
momentum-breakdown exit on a close back below the fast MA.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ..data.nse_client import Quote
from ..portfolio.models import Position


@dataclass
class Candidate:
    symbol: str
    ltp: float
    consolidation_high: float
    consolidation_low: float
    breakout_volume: float
    avg_volume: float
    momentum_return_pct: float
    listing_age_days: int
    score: float


def _moving_average(history: pd.DataFrame, days: int) -> float | None:
    if len(history) < days:
        return None
    return float(history["Close"].tail(days).mean())


def _momentum_return_pct(history: pd.DataFrame, lookback_days: int) -> float | None:
    if len(history) < lookback_days + 1:
        return None
    closes = history["Close"]
    past = float(closes.iloc[-(lookback_days + 1)])
    now = float(closes.iloc[-1])
    if past <= 0:
        return None
    return (now - past) / past * 100.0


def _avg_volume(history: pd.DataFrame, days: int) -> float | None:
    if len(history) < days:
        return None
    try:
        return float(history["Volume"].tail(days).mean())
    except (KeyError, TypeError):
        return None


def _detect_consolidation(
    history: pd.DataFrame, consolidation_days: int, max_range_pct: float
) -> tuple[float, float, bool] | None:
    """Same mechanics as consolidation_breakout._detect_consolidation: the
    base window excludes the latest bar (that bar is what's tested against
    this high), and tightness is measured against the base's own average
    close so a big breakout move on the final bar can't retroactively make
    a wide base look narrow."""
    if len(history) < consolidation_days + 1:
        return None

    base = history.iloc[-(consolidation_days + 1):-1]
    high = float(base["High"].max())
    low = float(base["Low"].min())
    reference = float(base["Close"].mean())

    if not (reference > 0) or math.isnan(high) or math.isnan(low):
        return None

    range_pct = (high - low) / reference * 100.0
    is_tight = range_pct < max_range_pct

    return (high, low, is_tight)


def _detect_breakout(history: pd.DataFrame, consolidation_high: float, volume_multiple: float) -> bool:
    if len(history) < 2:
        return False

    current_close = float(history["Close"].iloc[-1])
    current_volume = float(history["Volume"].iloc[-1])

    if not (current_close > consolidation_high):
        return False

    avg_vol = _avg_volume(history.iloc[:-1], 20)
    if avg_vol is None or avg_vol <= 0 or math.isnan(avg_vol):
        return False

    return current_volume >= avg_vol * volume_multiple


def evaluate_candidate(
    symbol: str,
    quote: Quote,
    history: pd.DataFrame,
    daily_turnover_inr: float,
    config: dict,
    reasons: dict[str, int] | None = None,
) -> Candidate | None:
    if reasons is None:
        reasons = {}

    def reject(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    ipo_config = config.get("ipo_base_breakout") or {}

    # Recent-listing gate: this is the filter that makes this strategy an
    # "IPO base" strategy rather than a second copy of consolidation_breakout.
    # A stock still needs enough history to have formed a real base --
    # min_listing_days below -- but too much history (max_listing_days)
    # means whatever overhead-resistance-free advantage a fresh IPO has is
    # long gone, even if the stock still happens to be trading well.
    min_listing_days = int(ipo_config.get("min_listing_days", 60))
    max_listing_days = int(ipo_config.get("max_listing_days", 500))
    listing_age_days = len(history)
    if listing_age_days < min_listing_days:
        reject("too_new_no_base_yet")
        return None
    if listing_age_days > max_listing_days:
        reject("not_a_recent_listing")
        return None

    # Uptrend check: shorter MAs than an established stock's 50/200 pair --
    # a name with under max_listing_days (500) of history usually cannot
    # support a 200-day average reliably this early.
    fast_ma = _moving_average(history, config.get("fast_ma_days", 10))
    slow_ma = _moving_average(history, config.get("slow_ma_days", 30))
    if fast_ma is None or slow_ma is None:
        reject("insufficient_history")
        return None
    if not (quote.ltp > fast_ma > 0 and quote.ltp > slow_ma and fast_ma >= slow_ma):
        reject("not_in_uptrend")
        return None

    # Liquidity check
    _min_turnover = config.get("min_avg_daily_turnover_inr", 500000)
    if daily_turnover_inr < _min_turnover:
        reject("illiquid")
        return None

    # Price band: a fresh IPO priced at a few rupees is exactly the kind
    # of thin, easily-manipulated listing this filter exists to exclude
    # for every other equity strategy here.
    min_ltp = config.get("min_ltp_inr")
    if min_ltp and quote.ltp < min_ltp:
        reject("ltp_below_min")
        return None
    max_ltp = config.get("max_ltp_inr")
    if max_ltp and quote.ltp > max_ltp:
        reject("ltp_above_max")
        return None

    # Base detection: wider default tolerance than the established-stock
    # strategy's 3% -- a stock still finding its footing post-listing
    # naturally swings wider even while consolidating.
    consolidation_days = int(ipo_config.get("base_days", 15))
    max_range_pct = ipo_config.get("max_base_range_pct", 15.0)
    consolidation = _detect_consolidation(history, consolidation_days, max_range_pct)
    if consolidation is None:
        reject("insufficient_history")
        return None

    consolidation_high, consolidation_low, is_tight = consolidation
    if not is_tight:
        reject("base_not_tight")
        return None

    volume_multiple = ipo_config.get("volume_multiple", 1.5)
    if not _detect_breakout(history, consolidation_high, volume_multiple):
        reject("no_breakout")
        return None

    momentum = _momentum_return_pct(history, config.get("momentum_lookback_days", 21))
    if momentum is None or momentum < config.get("min_momentum_return_pct", 5.0):
        reject("weak_momentum")
        return None

    # Score: higher momentum and closer to the breakout level score
    # higher, same convention as consolidation_breakout, plus a mild bonus
    # for a younger listing -- closer to the "freshest" part of this
    # strategy's edge, all else equal.
    distance_from_breakout = max(0, consolidation_high - quote.ltp)
    score = momentum
    if consolidation_high > 0:
        score -= (distance_from_breakout / consolidation_high) * 10.0
    score += max(0.0, (max_listing_days - listing_age_days) / max_listing_days) * 5.0

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        consolidation_high=consolidation_high,
        consolidation_low=consolidation_low,
        breakout_volume=float(history["Volume"].iloc[-1]),
        avg_volume=_avg_volume(history, 20) or 0.0,
        momentum_return_pct=momentum,
        listing_age_days=listing_age_days,
        score=score,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def check_exit(position: Position, quote: Quote, history: pd.DataFrame, config: dict) -> tuple[bool, str]:
    """Mirrors consolidation_breakout.check_exit exactly: hard stop-loss,
    trailing stop, optional take-profit, momentum breakdown below the
    fast MA."""
    stop_loss_pct = config.get("stop_loss_pct", 8.0)
    stop_loss_price = position.avg_price * (1 - stop_loss_pct / 100.0)
    if quote.ltp <= stop_loss_price:
        return True, f"stop_loss ({stop_loss_pct}% below entry {position.avg_price:.2f})"

    trailing_stop_pct = config.get("trailing_stop_pct", 12.0)
    trailing_stop_price = position.highest_close_since_entry * (1 - trailing_stop_pct / 100.0)
    if quote.ltp <= trailing_stop_price:
        return True, f"trailing_stop ({trailing_stop_pct}% below peak {position.highest_close_since_entry:.2f})"

    take_profit_pct = config.get("take_profit_pct")
    if take_profit_pct:
        take_profit_price = position.avg_price * (1 + take_profit_pct / 100.0)
        if quote.ltp >= take_profit_price:
            return True, f"take_profit (+{take_profit_pct}% above entry {position.avg_price:.2f})"

    if config.get("exit_below_fast_ma", True):
        fast_days = int(config.get("fast_ma_days", 10))
        fast_ma = _moving_average(history, fast_days)
        if fast_ma is not None and quote.ltp < fast_ma:
            return True, f"momentum_breakdown (close below {fast_days}DMA {fast_ma:.2f})"

    return False, ""
