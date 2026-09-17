"""Long Term Trend strategy.

The buy-and-hold-oriented counterpart to this app's swing strategies:
instead of timing a breakout, a pullback, or an indicator flip, it just
asks "is this stock in a well-established, multi-year uptrend?" and, if
so, holds through the ordinary volatility that comes with staying in a
position for a long time rather than exiting on short-term noise.

This app only has price/volume data (no fundamentals feed), so "long
term" here means genuinely long lookback windows and wide risk
tolerances, not a fundamentals-based quality screen:

  1. Trend confirmation: price above BOTH a long moving average
     (`fast_ma_days`, e.g. 200 days) and an even longer one
     (`slow_ma_days`, e.g. 400 days), with the long average itself above
     the very long one -- the same "golden cross" shape the other
     strategies use, just stretched from weeks to years.
  2. Participation: the stock must have actually delivered a real return
     over a long lookback (`momentum_lookback_days`, e.g. 252 days) --
     merely sitting above a long-term average isn't enough; a stock that
     has drifted sideways for a year while barely holding above its
     200-day average isn't the "already compounding" story this
     strategy is looking for.
  3. Liquidity and price-range filters, same convention as every other
     strategy in this app.

Deliberately NO fast-moving-average filter, no proximity-to-high check,
no breakout/pullback timing -- entry timing precision matters far less
over a years-long holding period than being in the right names at all.

Exits (`check_exit`) use this app's usual stop-loss/trailing-stop/take-
profit shape, but are expected to be configured much wider than the
swing strategies' (via this profile's own `risk:` block -- risk is
profile-scoped, same as strategy) so an ordinary multi-week pullback
within a multi-year uptrend doesn't stop the position out. The
strategy-specific exit is a close below the LONG moving average
(`fast_ma_days`, the same one used for entry) -- the signal that the
multi-year uptrend itself has actually broken, not just that price has
had a rough month.
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
    fast_ma: float
    slow_ma: float
    momentum_return_pct: float
    score: float


def _moving_average(history: pd.DataFrame, window: int) -> float | None:
    if len(history) < window:
        return None
    return float(history["Close"].tail(window).mean())


def _momentum_return_pct(history: pd.DataFrame, lookback_days: int) -> float | None:
    if len(history) < lookback_days + 1:
        return None
    closes = history["Close"]
    past = float(closes.iloc[-(lookback_days + 1)])
    now = float(closes.iloc[-1])
    if past <= 0:
        return None
    return (now - past) / past * 100.0


def evaluate_candidate(
    symbol: str,
    quote: Quote,
    history: pd.DataFrame,
    avg_daily_turnover: float,
    cfg: dict,
    reasons: dict[str, int] | None = None,
) -> Candidate | None:
    """Return a Candidate if symbol currently qualifies for a long-term entry."""

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

    fast_ma = _moving_average(history, cfg.get("fast_ma_days", 200))
    slow_ma = _moving_average(history, cfg.get("slow_ma_days", 400))
    if fast_ma is None or slow_ma is None:
        reject("insufficient_history")
        return None
    if not (quote.ltp > fast_ma > 0 and fast_ma > slow_ma):
        reject("not_in_uptrend")
        return None

    momentum = _momentum_return_pct(history, cfg.get("momentum_lookback_days", 252))
    if momentum is None or momentum < cfg.get("min_momentum_return_pct", 12.0):
        reject("weak_participation")
        return None

    # Score: reward genuine participation in the trend (the long lookback
    # return) plus how much room the long average has over the very long
    # one -- a wide, well-established gap is a more mature, decisive
    # uptrend than a long average that has only just crossed above the
    # very long one.
    ma_spread_pct = (fast_ma - slow_ma) / slow_ma * 100.0 if slow_ma > 0 else 0.0
    score = momentum + ma_spread_pct

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        fast_ma=round(fast_ma, 2),
        slow_ma=round(slow_ma, 2),
        momentum_return_pct=round(momentum, 2),
        score=score,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def check_exit(position: Position, quote: Quote, history: pd.DataFrame, cfg: dict) -> tuple[bool, str]:
    """Return (should_exit, reason). `cfg` here is expected to carry this
    profile's own (wide) risk.* values plus its strategy.* fast_ma_days --
    see check_exits()'s {**risk_cfg, **strategy_cfg} merge convention."""

    stop_loss_pct = cfg.get("stop_loss_pct", 20.0)
    stop_loss_price = position.avg_price * (1 - stop_loss_pct / 100.0)
    if quote.ltp <= stop_loss_price:
        return True, f"stop_loss ({stop_loss_pct}% below entry {position.avg_price:.2f})"

    trailing_stop_pct = cfg.get("trailing_stop_pct", 25.0)
    trailing_stop_price = position.highest_close_since_entry * (1 - trailing_stop_pct / 100.0)
    if quote.ltp <= trailing_stop_price:
        return True, f"trailing_stop ({trailing_stop_pct}% below peak {position.highest_close_since_entry:.2f})"

    take_profit_pct = cfg.get("take_profit_pct")
    if take_profit_pct:
        take_profit_price = position.avg_price * (1 + take_profit_pct / 100.0)
        if quote.ltp >= take_profit_price:
            return True, f"take_profit (+{take_profit_pct}% above entry {position.avg_price:.2f})"

    # Strategy-specific exit: a close below the LONG moving average means
    # the multi-year uptrend this position was entered on has actually
    # broken -- not merely that price has had a rough stretch, which the
    # wide trailing stop above already tolerates.
    fast_ma_days = cfg.get("fast_ma_days", 200)
    fast_ma = _moving_average(history, fast_ma_days)
    if fast_ma is not None and quote.ltp < fast_ma:
        return True, f"trend_break (close below {fast_ma_days}DMA {fast_ma:.2f})"

    return False, ""
