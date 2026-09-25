"""Consolidation breakout strategy for crypto markets.

Designed for 24/7 crypto trading with continuous price action.

Entry filters:
  1. Coin in clear uptrend (price > 20-day MA > 50-day MA)
  2. Forms tight consolidation/base for several days
  3. Breaks out above base high on strong volume (≥ 1.5× average)
  4. Entry: At or just above breakout level
  5. Adequate liquidity (daily_turnover >= min_avg_daily_turnover_inr)

Exit rules:
  1. Hard stop-loss (8% below entry)
  2. Trailing stop from highest close since entry (12%)
  3. Momentum breakdown (close below 20-day MA)
  4. (optional) Take profit at fixed level
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from papertrader.data.nse_client import Quote
from papertrader.portfolio.models import Position


@dataclass
class Candidate:
    symbol: str
    ltp: float
    consolidation_high: float
    consolidation_low: float
    breakout_volume: float
    avg_volume: float
    momentum_return_pct: float
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
    history: pd.DataFrame, consolidation_days: int, max_range_pct: float = 20.0
) -> tuple[float, float, bool] | None:
    """Detect a tight consolidation base in the days *before* the latest bar.

    Returns (high, low, is_tight) for the base window, or None if there is
    not enough history.

    The base deliberately EXCLUDES the most recent bar, because that bar is
    the one `_detect_breakout` tests against this high. Including it made
    the breakout test unsatisfiable: `high` would be the max High over a
    window containing today, and a bar's Close can never exceed its own
    High, so `close > high` was false by construction on every symbol,
    every day.

    Tightness is measured against the base's own average close rather than
    the latest close, so a large breakout move on the final bar can't make
    a wide base look narrow (or vice versa).
    """
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
    """Detect if the latest bar closed above the base high on strong volume."""
    if len(history) < 2:
        return False

    current_close = float(history["Close"].iloc[-1])
    current_volume = float(history["Volume"].iloc[-1])

    # Broke above the consolidation high. `consolidation_high` comes from
    # the bars before this one (see _detect_consolidation), so this is a
    # real comparison rather than a bar against its own high.
    if not (current_close > consolidation_high):
        return False

    # Volume check: today's volume >= a multiple of the baseline average.
    # The baseline excludes the breakout bar itself -- otherwise a big
    # volume day inflates the very average it is being measured against,
    # making the breakout look weaker the stronger it actually is.
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
    """Evaluate a crypto coin for consolidation breakout entry.

    Returns Candidate if symbol qualifies, None otherwise.
    `reasons` is an optional counter the caller can pass in to track
    rejection reasons for diagnostics.
    """
    if reasons is None:
        reasons = {}

    def reject(reason: str) -> None:
        if reasons is not None:
            reasons[reason] = reasons.get(reason, 0) + 1

    # Uptrend check: price > 20-day MA > 50-day MA (crypto-specific defaults)
    fast_ma = _moving_average(history, config.get("fast_ma_days", 20))
    slow_ma = _moving_average(history, config.get("slow_ma_days", 50))

    if fast_ma is None or slow_ma is None:
        reject("insufficient_history")
        return None
    if not (quote.ltp > fast_ma > 0 and quote.ltp > slow_ma and fast_ma >= slow_ma):
        reject("not_in_uptrend")
        return None

    # Liquidity check
    # Crypto quotes are converted from USD to INR at the data layer
    # (nse_client.py), so turnover = Close (INR) * Volume (coins) arrives
    # in INR. Prefer min_avg_daily_turnover_usd if set (for legacy/explicit
    # USD thresholds), else fall back to the INR figure in config.
    _min_turnover = config.get("min_avg_daily_turnover_usd")
    if _min_turnover is None:
        _min_turnover = config.get("min_avg_daily_turnover_inr", 480000000)
    if daily_turnover_inr < _min_turnover:
        reject("illiquid")
        return None

    # Consolidation parameters live under strategy.crypto_breakout in config
    cb_config = config.get("crypto_breakout") or {}

    # Detect consolidation
    consolidation_days = cb_config.get("consolidation_days", 7)
    max_range_pct = cb_config.get("max_consolidation_range_pct", 20.0)
    consolidation = _detect_consolidation(history, consolidation_days, max_range_pct)

    if consolidation is None:
        reject("insufficient_history")
        return None

    consolidation_high, consolidation_low, is_tight = consolidation
    if not is_tight:
        reject("base_not_tight")
        return None

    # Detect breakout
    volume_multiple = cb_config.get("volume_multiple", 1.5)
    if not _detect_breakout(history, consolidation_high, volume_multiple):
        reject("no_breakout")
        return None

    # Momentum check
    momentum = _momentum_return_pct(history, config.get("momentum_lookback_days", 21))
    if momentum is None or momentum < config.get("min_momentum_return_pct", 3.0):
        reject("weak_momentum")
        return None

    # Score: higher momentum and closer to breakout level score higher
    distance_from_breakout = max(0, consolidation_high - quote.ltp)
    score = momentum
    if consolidation_high > 0:
        score -= (distance_from_breakout / consolidation_high) * 10.0

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        consolidation_high=consolidation_high,
        consolidation_low=consolidation_low,
        breakout_volume=float(history["Volume"].iloc[-1]),
        avg_volume=_avg_volume(history, 20) or 0.0,
        momentum_return_pct=momentum,
        score=score,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def check_exit(position: Position, quote: Quote, history: pd.DataFrame, config: dict) -> tuple[bool, str]:
    """Determine if a crypto position should exit."""

    # Hard stop-loss (8% below entry for crypto)
    stop_loss_pct = config.get("stop_loss_pct", 8.0)
    stop_loss_price = position.avg_price * (1 - stop_loss_pct / 100.0)
    if quote.ltp <= stop_loss_price:
        return True, f"stop_loss ({stop_loss_pct}% below entry {position.avg_price:.2f})"

    # Trailing stop (12% below peak)
    trailing_stop_pct = config.get("trailing_stop_pct", 12.0)
    trailing_stop_price = position.highest_close_since_entry * (1 - trailing_stop_pct / 100.0)
    if quote.ltp <= trailing_stop_price:
        return True, f"trailing_stop ({trailing_stop_pct}% below peak {position.highest_close_since_entry:.2f})"

    # Take profit (optional)
    take_profit_pct = config.get("take_profit_pct")
    if take_profit_pct:
        take_profit_price = position.avg_price * (1 + take_profit_pct / 100.0)
        if quote.ltp >= take_profit_price:
            return True, f"take_profit (+{take_profit_pct}% above entry {position.avg_price:.2f})"

    # Momentum breakdown (close below fast MA)
    fast_days = int(config.get("fast_ma_days", 20))
    fast_ma = _moving_average(history, fast_days)
    if fast_ma is not None and quote.ltp < fast_ma:
        return True, f"momentum_breakdown (close below {fast_days}-day MA)"

    return False, ""
