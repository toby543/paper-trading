"""Institutional swing trading strategy for crypto bull markets.

Designed for 2026+ post-halving bull cycles with institutional participation.
Entry: Pullback to 20-day MA during strong uptrend, confirmed by volume.
Exit: Swing profit target (+22%) or time stop (7 days), or trend break.

Characteristics:
  - Trend-following (lower whipsaw risk)
  - Pullback entries (better risk/reward than breakouts)
  - Volume confirmation (institutional capital flows)
  - Volatile-adjusted stops (8% for crypto)
  - Swing-sized holds (5-10 days optimal)
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
    ma20: float
    ma50: float
    ma200: float
    volume_multiple: float
    momentum_return_pct: float
    score: float


def _moving_average(history: pd.DataFrame, days: int) -> float | None:
    """Calculate simple moving average. Returns None if insufficient data."""
    if len(history) < days:
        return None
    return float(history["Close"].tail(days).mean())


def _rsi(history: pd.DataFrame, period: int = 14) -> float | None:
    """Calculate RSI(14). Returns None if it cannot be computed.

    A completely flat window has zero average gain AND zero average loss,
    so rs = 0/0 = NaN and the RSI comes out NaN. Return None for this case.
    """
    if len(history) < period + 1:
        return None
    closes = history["Close"]
    deltas = closes.diff()
    gains = (deltas.where(deltas > 0, 0)).rolling(period).mean()
    losses = (-deltas.where(deltas < 0, 0)).rolling(period).mean()
    rs = gains / losses
    rsi = 100 - (100 / (1 + rs))
    value = float(rsi.iloc[-1])
    if math.isnan(value):
        return None
    return value


def _momentum_return_pct(history: pd.DataFrame, lookback_days: int) -> float | None:
    """Return % gain over the past lookback_days."""
    if len(history) <= lookback_days:
        return None
    lookback_idx = -lookback_days - 1
    past_close = history["Close"].iloc[lookback_idx]
    current_close = history["Close"].iloc[-1]
    if past_close <= 0:
        return None
    return ((current_close - past_close) / past_close) * 100


def _avg_volume(history: pd.DataFrame, days: int) -> float | None:
    """Calculate average volume over recent bars."""
    if len(history) < days:
        return None
    try:
        return float(history["Volume"].tail(days).mean())
    except (KeyError, TypeError):
        return None


def evaluate_candidate(
    symbol: str,
    quote: Quote,
    history: pd.DataFrame,
    daily_turnover_inr: float,
    config: dict,
    reasons: dict[str, int] | None = None,
) -> Candidate | None:
    """Evaluate a crypto coin for institutional swing entry.

    Institutional swing strategy: Buy pullbacks in confirmed uptrends,
    confirmed by volume. Targets 2026 bull market conditions.

    Entry criteria:
    - Long-term uptrend: price > 200-day MA
    - Medium-term uptrend: 50-day MA > 200-day MA
    - Swing pullback: price near 20-day MA (support)
    - RSI bounce: 35-50 (not oversold, bouncing)
    - Volume confirmation: 1.5x+ average volume
    - Adequate liquidity: daily_turnover >= min threshold
    """
    if reasons is None:
        reasons = {}

    def reject(reason: str) -> None:
        if reasons is not None:
            reasons[reason] = reasons.get(reason, 0) + 1

    # Liquidity check first
    _min_turnover = config.get("min_avg_daily_turnover_usd")
    if _min_turnover is None:
        _min_turnover = config.get("min_avg_daily_turnover_inr", 480000000)
    if daily_turnover_inr < _min_turnover:
        reject("illiquid")
        return None

    # History sufficiency
    if len(history) < 250:  # Need 200-day MA + buffer
        reject("insufficient_history")
        return None

    # Calculate moving averages
    ma20 = _moving_average(history, 20)
    ma50 = _moving_average(history, 50)
    ma200 = _moving_average(history, 200)

    if ma20 is None or ma50 is None or ma200 is None:
        reject("insufficient_history")
        return None

    # Long-term uptrend: price > 200-day MA
    if quote.ltp <= ma200:
        reject("not_in_long_uptrend")
        return None

    # Medium-term uptrend: 50-day > 200-day (trend confirmed)
    if ma50 <= ma200:
        reject("not_in_medium_uptrend")
        return None

    # Swing pullback setup: price within 5% of 20-day MA (at support)
    # This catches the bounce without being too strict
    distance_to_ma20 = abs(quote.ltp - ma20) / ma20 * 100
    if distance_to_ma20 > config.get("pullback_tolerance_pct", 5.0):
        reject("not_at_pullback")
        return None

    # RSI bounce confirmation: 35-50 (bouncing from dip, not crashed)
    rsi14 = _rsi(history, 14)
    min_rsi = config.get("min_rsi", 35.0)
    max_rsi = config.get("max_rsi", 50.0)
    if rsi14 is None or rsi14 < min_rsi or rsi14 > max_rsi:
        reject("rsi_out_of_range")
        return None

    # Volume confirmation: current volume > 1.5x baseline
    avg_vol_20 = _avg_volume(history.iloc[:-1], 20)  # Exclude today
    current_volume = float(history["Volume"].iloc[-1])
    volume_multiple = 1.0
    if avg_vol_20 is not None and avg_vol_20 > 0:
        volume_multiple = current_volume / avg_vol_20
        if volume_multiple < config.get("min_volume_multiple", 1.5):
            reject("low_volume")
            return None

    # Momentum check: at least some recent gain (validates uptrend)
    momentum_days = config.get("momentum_lookback_days", 30)
    momentum_pct = _momentum_return_pct(history, momentum_days)
    if momentum_pct is None or momentum_pct < config.get("min_momentum_return_pct", 3.0):
        reject("weak_momentum")
        return None

    # Score: higher RSI strength + higher volume = better setup
    # RSI 35-50 range maps to score 0-1
    rsi_score = (rsi14 - min_rsi) / (max_rsi - min_rsi) * 100
    volume_score = min(volume_multiple / 2.5 * 100, 100)  # Cap at 2.5x
    momentum_score = min(momentum_pct / 10 * 100, 100)  # Cap at 10% return
    distance_score = max(0, 100 - distance_to_ma20 * 10)  # Closer to MA20 = better

    score = (rsi_score * 0.3 + volume_score * 0.4 + momentum_score * 0.2 + distance_score * 0.1)

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        ma20=ma20,
        ma50=ma50,
        ma200=ma200,
        volume_multiple=volume_multiple,
        momentum_return_pct=momentum_pct,
        score=score,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Rank candidates by score, highest first."""
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def check_exit(position: Position, quote: Quote, history: pd.DataFrame, config: dict) -> tuple[bool, str]:
    """Determine if a swing position should exit.

    Exit conditions (in order):
    1. Profit target: +22% gain (swing objective)
    2. Time stop: 7 days (don't hold too long)
    3. Hard stop: -8% loss (trend break)
    4. Trend break: price closes below 20-day MA
    5. Trailing stop: -6% below peak (lock in gains)
    """
    profit_target_pct = config.get("profit_target_pct", 22.0)
    time_stop_days = config.get("time_stop_days", 7)
    # "stop_loss_pct", not "hard_stop_pct": that's the shared risk-section
    # key Settings actually writes to (see settings_schema.py). The old
    # "hard_stop_pct" name matched nothing in risk_cfg/strategy_cfg, so
    # this always silently fell back to the 8.0 default no matter what
    # the user set the stop-loss to in Edit Settings.
    hard_stop_pct = config.get("stop_loss_pct", 8.0)
    trailing_stop_pct = config.get("trailing_stop_pct", 6.0)

    # Profit target: +22% gain (swing objective)
    target_price = position.avg_price * (1 + profit_target_pct / 100.0)
    if quote.ltp >= target_price:
        return True, f"profit_target (+{profit_target_pct}% = {quote.ltp:.2f})"

    # Hard stop-loss: -8% (trend break, exit before bigger loss)
    hard_stop_price = position.avg_price * (1 - hard_stop_pct / 100.0)
    if quote.ltp <= hard_stop_price:
        return True, f"hard_stop (-{hard_stop_pct}% at {quote.ltp:.2f})"

    # Time stop: Exit after 7 days (don't hold through regime change)
    # Position has no "entry_time" attribute -- only "entry_date" (an ISO
    # string). hasattr(position, 'entry_time') was always False, so
    # days_held was always 0 and this exit could never fire.
    days_held = (pd.Timestamp.now() - pd.Timestamp(position.entry_date)).days
    if days_held >= time_stop_days:
        return True, f"time_stop ({days_held} days, current {quote.ltp:.2f})"

    # Trend break: Price closes below 20-day MA (exit if pullback fails)
    fast_days = int(config.get("fast_ma_days", 20))
    fast_ma = _moving_average(history, fast_days)
    if fast_ma is not None and quote.ltp < fast_ma:
        return True, f"trend_break (below {fast_days}-day MA at {quote.ltp:.2f})"

    # Trailing stop: -6% below peak (protect profits after +10% gain)
    current_gain_pct = (quote.ltp - position.avg_price) / position.avg_price * 100.0
    if current_gain_pct >= 10.0:  # Only activate after +10%
        trail_price = position.highest_close_since_entry * (1 - trailing_stop_pct / 100.0)
        if quote.ltp <= trail_price:
            return True, f"trailing_stop (-{trailing_stop_pct}% from peak {position.highest_close_since_entry:.2f})"

    return False, ""
