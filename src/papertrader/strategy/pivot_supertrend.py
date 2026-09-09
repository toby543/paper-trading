"""Pivot Point + SuperTrend swing strategy.

A popular retail day/swing-trading combo: classical floor-trader pivot
points set the day's directional bias, and the ATR-based SuperTrend
indicator triggers the actual entry/exit.

  1. Pivot point: P = (prior day's High + Low + Close) / 3. Trading above
     P is the bullish-bias zone; this strategy only goes long there.
  2. SuperTrend: an ATR-based trailing band (Wilder-smoothed ATR, the
     standard SuperTrend definition) that flips sides whenever price
     closes through it. A fresh flip from below the line to above it is
     the entry trigger -- not merely "currently above the line", which
     would re-qualify every day of an already-established uptrend.
  3. Liquidity and price-range filters, same convention as the other
     three strategies.

This app only fetches daily bars and scans on a multi-minute cadence, so
this is the daily-bar SWING adaptation of the combo (pivots/SuperTrend
computed from daily candles), not a true intraday day-trading system --
that would need a different data source and a much shorter scan cadence
across the whole engine, not just a new strategy module.

Exits (handled by `check_exit`) are the same universal risk-management
safety net every strategy in this app shares -- a hard stop-loss, a
trailing stop from the highest close since entry, and an optional take
profit -- plus this strategy's own specific trigger: the SuperTrend line
flipping back to a downtrend, which plays the same role here that a
momentum breakdown (close below the fast MA) plays for the other
strategies.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.nse_client import Quote
from ..portfolio.models import Position


@dataclass
class Candidate:
    symbol: str
    ltp: float
    pivot: float
    pct_above_pivot: float
    supertrend_value: float
    atr: float
    score: float


def _true_range(history: pd.DataFrame) -> pd.Series:
    high, low, close = history["High"], history["Low"], history["Close"]
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def _atr(history: pd.DataFrame, period: int) -> pd.Series | None:
    """Wilder-smoothed Average True Range -- the ATR definition SuperTrend
    is conventionally built on (matches TradingView's built-in indicator),
    not a plain rolling mean. None if there isn't enough history yet."""
    if len(history) < period + 1:
        return None
    tr = _true_range(history)
    atr = tr.astype(float).copy()
    atr.iloc[: period - 1] = float("nan")
    atr.iloc[period - 1] = tr.iloc[:period].mean()
    for i in range(period, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period
    return atr


def _supertrend(history: pd.DataFrame, period: int, multiplier: float):
    """Standard SuperTrend (Wilder ATR bands, ratcheted and flipped per
    the canonical definition -- the "up"/lower band only ever ratchets
    up while price holds above it, the "dn"/upper band only ever
    ratchets down while price holds below it, each resetting to the
    fresh basic band the moment price closes through it).

    Returns (line, in_uptrend) as aligned pd.Series, or (None, None) if
    there isn't enough history for a valid ATR yet. in_uptrend[i] is True
    while the close is riding above the line (the line trails below
    price, like a trailing stop); False in a downtrend (the line caps
    price from above).
    """
    atr = _atr(history, period)
    if atr is None:
        return None, None

    high, low, close = history["High"], history["Low"], history["Close"]
    mid = (high + low) / 2.0
    n = len(history)
    start = period - 1  # first index with a non-NaN ATR (see _atr)

    up = [float("nan")] * n
    dn = [float("nan")] * n
    trend = [1] * n  # 1 = uptrend (line follows "up"), -1 = downtrend (line follows "dn")

    up[start] = float(mid.iloc[start] - multiplier * atr.iloc[start])
    dn[start] = float(mid.iloc[start] + multiplier * atr.iloc[start])

    for i in range(start + 1, n):
        basic_up = float(mid.iloc[i] - multiplier * atr.iloc[i])
        basic_dn = float(mid.iloc[i] + multiplier * atr.iloc[i])
        prev_close = float(close.iloc[i - 1])
        up[i] = max(basic_up, up[i - 1]) if prev_close > up[i - 1] else basic_up
        dn[i] = min(basic_dn, dn[i - 1]) if prev_close < dn[i - 1] else basic_dn

        trend[i] = trend[i - 1]
        cur_close = float(close.iloc[i])
        if trend[i - 1] == -1 and cur_close > dn[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and cur_close < up[i - 1]:
            trend[i] = -1

    line = [up[i] if trend[i] == 1 else dn[i] for i in range(n)]
    return (
        pd.Series(line, index=history.index, dtype=float),
        pd.Series([t == 1 for t in trend], index=history.index),
    )


def _prior_day_pivot(history: pd.DataFrame) -> float | None:
    """Standard floor-trader pivot from the PRIOR bar's High/Low/Close.
    Deliberately excludes today's own bar -- using today's own high/low
    to judge whether today is trading above/below its own pivot would be
    circular."""
    if len(history) < 2:
        return None
    prev = history.iloc[-2]
    if pd.isna(prev["High"]) or pd.isna(prev["Low"]) or pd.isna(prev["Close"]):
        return None
    return float((prev["High"] + prev["Low"] + prev["Close"]) / 3.0)


def evaluate_candidate(
    symbol: str,
    quote: Quote,
    history: pd.DataFrame,
    avg_daily_turnover: float,
    cfg: dict,
    reasons: dict[str, int] | None = None,
) -> Candidate | None:
    """Return a Candidate if symbol currently qualifies for a pivot+SuperTrend entry."""

    def reject(reason: str) -> None:
        if reasons is not None:
            reasons[reason] = reasons.get(reason, 0) + 1

    if avg_daily_turnover < cfg.get("min_avg_daily_turnover_inr", 500000):
        reject("illiquid")
        return None

    min_ltp = cfg.get("min_ltp_inr")
    if min_ltp and quote.ltp < min_ltp:
        reject("ltp_below_min")
        return None
    max_ltp = cfg.get("max_ltp_inr")
    if max_ltp and quote.ltp > max_ltp:
        reject("ltp_above_max")
        return None

    ps_cfg = cfg.get("pivot_supertrend") or {}
    atr_period = ps_cfg.get("atr_period", 10)
    multiplier = ps_cfg.get("supertrend_multiplier", 3.0)

    line, in_uptrend = _supertrend(history, atr_period, multiplier)
    if line is None or len(in_uptrend) < 2:
        reject("insufficient_history")
        return None

    # A FLIP -- yesterday down, today up -- not merely "currently above
    # the line", which would re-qualify every single day of an already-
    # established uptrend as a fresh buy signal.
    if not (bool(in_uptrend.iloc[-1]) and not bool(in_uptrend.iloc[-2])):
        reject("no_supertrend_flip")
        return None

    pivot = _prior_day_pivot(history)
    if pivot is None or pivot <= 0:
        reject("insufficient_history")
        return None

    pct_above_pivot = (quote.ltp - pivot) / pivot * 100.0
    min_pct_above_pivot = ps_cfg.get("min_pct_above_pivot", 0.0)
    if pct_above_pivot < min_pct_above_pivot:
        reject("below_pivot")
        return None

    atr_series = _atr(history, atr_period)
    atr_value = float(atr_series.iloc[-1]) if atr_series is not None else 0.0
    supertrend_value = float(line.iloc[-1])

    # Score: how decisively price cleared the SuperTrend line, in ATR
    # units so it's comparable across stocks at very different price
    # levels, plus a smaller bonus for trading further above the day's
    # pivot -- a flip right at the pivot is a weaker bullish-bias signal
    # than one with real room above it.
    breakout_strength = (quote.ltp - supertrend_value) / atr_value if atr_value > 0 else 0.0
    score = breakout_strength * 10.0 + pct_above_pivot

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        pivot=round(pivot, 2),
        pct_above_pivot=round(pct_above_pivot, 2),
        supertrend_value=round(supertrend_value, 2),
        atr=round(atr_value, 2),
        score=score,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def check_exit(position: Position, quote: Quote, history: pd.DataFrame, cfg: dict) -> tuple[bool, str]:
    """Return (should_exit, reason)."""

    stop_loss_pct = cfg.get("stop_loss_pct", 7.0)
    stop_loss_price = position.avg_price * (1 - stop_loss_pct / 100.0)
    if quote.ltp <= stop_loss_price:
        return True, f"stop_loss ({stop_loss_pct}% below entry {position.avg_price:.2f})"

    trailing_stop_pct = cfg.get("trailing_stop_pct", 12.0)
    trailing_stop_price = position.highest_close_since_entry * (1 - trailing_stop_pct / 100.0)
    if quote.ltp <= trailing_stop_price:
        return True, f"trailing_stop ({trailing_stop_pct}% below peak {position.highest_close_since_entry:.2f})"

    take_profit_pct = cfg.get("take_profit_pct")
    if take_profit_pct:
        take_profit_price = position.avg_price * (1 + take_profit_pct / 100.0)
        if quote.ltp >= take_profit_price:
            return True, f"take_profit (+{take_profit_pct}% above entry {position.avg_price:.2f})"

    # Strategy-specific exit: the SuperTrend line flipping back to a
    # downtrend plays the same role here that a momentum-breakdown
    # (close below the fast MA) plays for the other strategies -- the
    # signal that got us in has reversed.
    ps_cfg = cfg.get("pivot_supertrend") or {}
    atr_period = ps_cfg.get("atr_period", 10)
    multiplier = ps_cfg.get("supertrend_multiplier", 3.0)
    line, in_uptrend = _supertrend(history, atr_period, multiplier)
    if line is not None and len(in_uptrend) >= 1 and not bool(in_uptrend.iloc[-1]):
        return True, f"supertrend_flip (close below SuperTrend line {float(line.iloc[-1]):.2f})"

    return False, ""
