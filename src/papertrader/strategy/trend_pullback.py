"""Trend Pullback swing strategy.

The "buy strength on a dip" style popularised by trend-following swing
traders (e.g. Adam Khoo's framework): rather than chasing a breakout or
buying right at a 52-week high, wait for an already-established uptrend
to pull back a shallow, controlled amount, then buy the first sign it's
turning back up. The premise is a better average entry price and a
tighter, more logical stop (just below the pullback) than either a
breakout or a proximity-to-highs entry gets you -- at the cost of missing
stocks that never pull back at all.

  1. Uptrend confirmation: price > fast MA > slow MA, same convention as
     the other three strategies.
  2. Pullback: today's close sits between `min_pullback_pct` and
     `max_pullback_pct` below the highest close of the last
     `pullback_lookback_days` -- shallow enough that the trend is still
     clearly intact, deep enough that it's a real pullback rather than
     noise. Too shallow and there's no discount; too deep and the trend
     may already be broken (that's what `check_exit`'s trend_break below
     is for).
  3. Turn confirmation: today must close above yesterday's close -- the
     first sign of the pullback resolving back up, not a dip still in
     progress. Buying mid-fall would defeat the point of waiting for a
     pullback at all.
  4. Liquidity and price-range filters, same convention as the other
     three strategies.

Exits (`check_exit`) share the universal stop-loss / trailing-stop /
take-profit safety net every strategy in this app uses, plus this
strategy's own specific trigger: a close below the SLOW moving average
(not the fast one, unlike 52w_high/consolidation_breakout) -- a pullback
entry is often already sitting close to or below the fast MA by design,
so using that as the trend-broken signal would exit positions
immediately after entry. The slow MA is the one that actually represents
this strategy's uptrend premise.
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
    recent_high: float
    pct_from_high: float
    day_change_pct: float
    score: float


def _moving_average(history: pd.DataFrame, window: int) -> float | None:
    if len(history) < window:
        return None
    return float(history["Close"].tail(window).mean())


def _recent_high(history: pd.DataFrame, lookback_days: int) -> float | None:
    """Highest close over the last `lookback_days` bars, INCLUDING today --
    a pullback is measured from the peak the stock actually reached, which
    may well be today if it's still climbing (in which case pct_from_high
    is ~0 and the min_pullback_pct filter below naturally rejects it)."""
    if len(history) < lookback_days:
        return None
    return float(history["Close"].tail(lookback_days).max())


def evaluate_candidate(
    symbol: str,
    quote: Quote,
    history: pd.DataFrame,
    avg_daily_turnover: float,
    cfg: dict,
    reasons: dict[str, int] | None = None,
) -> Candidate | None:
    """Return a Candidate if symbol currently qualifies for a trend-pullback entry."""

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

    fast_ma = _moving_average(history, cfg.get("fast_ma_days", 20))
    slow_ma = _moving_average(history, cfg.get("slow_ma_days", 50))
    if fast_ma is None or slow_ma is None:
        reject("insufficient_history")
        return None
    if not (quote.ltp > fast_ma > 0 and fast_ma > slow_ma):
        reject("not_in_uptrend")
        return None

    tp_cfg = cfg.get("trend_pullback") or {}
    lookback_days = tp_cfg.get("pullback_lookback_days", 20)
    recent_high = _recent_high(history, lookback_days)
    if recent_high is None or recent_high <= 0:
        reject("insufficient_history")
        return None

    pct_from_high = (recent_high - quote.ltp) / recent_high * 100.0
    min_pullback_pct = tp_cfg.get("min_pullback_pct", 3.0)
    max_pullback_pct = tp_cfg.get("max_pullback_pct", 12.0)
    if pct_from_high < min_pullback_pct:
        reject("pullback_too_shallow")
        return None
    if pct_from_high > max_pullback_pct:
        reject("pullback_too_deep")
        return None

    if len(history) < 2:
        reject("insufficient_history")
        return None
    prev_close = float(history["Close"].iloc[-2])
    if prev_close <= 0:
        reject("insufficient_history")
        return None
    day_change_pct = (quote.ltp - prev_close) / prev_close * 100.0
    if day_change_pct <= 0:
        reject("not_turning_up")
        return None

    # Score: reward a shallower pullback (closer to the trend, less
    # ground already given up) and a stronger turn-up day (more
    # conviction that the dip is actually over), rather than the deepest
    # pullback or the biggest single-day pop in isolation.
    shallowness = max_pullback_pct - pct_from_high
    score = shallowness + day_change_pct * 2.0

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        recent_high=round(recent_high, 2),
        pct_from_high=round(pct_from_high, 2),
        day_change_pct=round(day_change_pct, 2),
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

    # Strategy-specific exit: a close below the SLOW moving average means
    # the uptrend this pullback was supposed to be a dip *within* has
    # actually broken -- the fast MA isn't used here (unlike the other
    # strategies) because a pullback entry is often already sitting near
    # or below it by construction, which would exit positions almost
    # immediately after entry.
    slow_ma_days = cfg.get("slow_ma_days", 50)
    slow_ma = _moving_average(history, slow_ma_days)
    if slow_ma is not None and quote.ltp < slow_ma:
        return True, f"trend_break (close below {slow_ma_days}DMA {slow_ma:.2f})"

    return False, ""
