"""Mean-reversion strategy for crypto markets.

Every other crypto strategy in this app (crypto_momentum, crypto_breakout,
crypto_institutional_swing) is trend-following -- they all require price
above rising moving averages and go quiet the moment the market chops
sideways instead of trending. This one is deliberately the opposite: buy a
short-term oversold dip and sell the bounce back to the mean, which is
exactly the condition where the trend-following strategies find nothing.

Entry:
  1. Coin is NOT in a structural downtrend -- price above its own
     `long_ma_days` (default 100) moving average. Mean reversion bets on a
     snap-back, not a falling knife; this filter keeps it out of coins
     that are simply dying.
  2. Short-term oversold: today's close sits `min_deviation_pct` or more
     below its own `mean_ma_days` (default 20) moving average -- the
     "mean" this strategy reverts to.
  3. RSI(14) below `max_rsi` (default 35) -- momentum confirmation that
     the dip is real, not just a shallow, ordinary daily wiggle.
  4. Adequate liquidity, same convention as every other crypto strategy.

Exit:
  1. Reached the mean: close >= its own mean_ma -- the reversion this
     trade was betting on has happened, take the win.
  2. Hard stop-loss -- protects against the dip continuing instead of
     reverting (mean reversion is wrong roughly as often as it's right).
  3. Time stop (default 5 days) -- if the bounce hasn't happened by now,
     the setup has stopped being a short-term dip and holding longer just
     turns this into an unplanned trend-following bet.
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
    mean_ma: float
    deviation_pct: float
    rsi: float
    score: float


def _moving_average(history: pd.DataFrame, days: int) -> float | None:
    if len(history) < days:
        return None
    return float(history["Close"].tail(days).mean())


def _rsi(history: pd.DataFrame, period: int = 14) -> float | None:
    """Same NaN-safe RSI(14) as every other crypto strategy here -- see
    crypto_momentum._rsi for why a flat window must return None, not NaN."""
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

    min_turnover = config.get("min_avg_daily_turnover_usd")
    if min_turnover is None:
        min_turnover = config.get("min_avg_daily_turnover_inr", 480000000)
    if daily_turnover_inr < min_turnover:
        reject("illiquid")
        return None

    mr_cfg = config.get("crypto_mean_reversion") or {}
    long_ma_days = int(mr_cfg.get("long_ma_days", 100))
    mean_ma_days = int(mr_cfg.get("mean_ma_days", 20))

    if len(history) < long_ma_days:
        reject("insufficient_history")
        return None

    long_ma = _moving_average(history, long_ma_days)
    mean_ma = _moving_average(history, mean_ma_days)
    if long_ma is None or mean_ma is None or mean_ma <= 0:
        reject("insufficient_history")
        return None

    # Not a falling knife: stay above the long-run trend even while
    # dipping short-term.
    if quote.ltp <= long_ma:
        reject("structural_downtrend")
        return None

    deviation_pct = (mean_ma - quote.ltp) / mean_ma * 100.0
    min_deviation_pct = mr_cfg.get("min_deviation_pct", 6.0)
    if deviation_pct < min_deviation_pct:
        reject("not_oversold_enough")
        return None

    rsi14 = _rsi(history, 14)
    max_rsi = mr_cfg.get("max_rsi", 35.0)
    if rsi14 is None or rsi14 > max_rsi:
        reject("rsi_not_oversold")
        return None

    # Score: deeper (but not runaway) dips with lower RSI score higher --
    # both are read as "more compressed spring", within the bounds the
    # filters above already enforce.
    score = deviation_pct * 2.0 + (max_rsi - rsi14)

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        mean_ma=round(mean_ma, 8),
        deviation_pct=round(deviation_pct, 2),
        rsi=round(rsi14, 2),
        score=score,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def check_exit(position: Position, quote: Quote, history: pd.DataFrame, config: dict) -> tuple[bool, str]:
    stop_loss_pct = config.get("stop_loss_pct", 8.0)
    stop_price = position.avg_price * (1 - stop_loss_pct / 100.0)
    if quote.ltp <= stop_price:
        return True, f"stop_loss ({stop_loss_pct}% below entry {position.avg_price:.2f})"

    # Reversion target: back at (or above) the mean this trade bet on.
    mean_ma_days = int(config.get("mean_ma_days", 20))
    mean_ma = _moving_average(history, mean_ma_days)
    if mean_ma is not None and quote.ltp >= mean_ma:
        return True, f"reverted_to_mean ({mean_ma_days}DMA {mean_ma:.2f})"

    # Time stop: a mean-reversion trade that hasn't reverted within a few
    # days has stopped being what this strategy is designed to hold.
    time_stop_days = config.get("time_stop_days", 5)
    entry_ts = pd.Timestamp(position.entry_date)
    days_held = (pd.Timestamp.now() - entry_ts).days
    if days_held >= time_stop_days:
        return True, f"time_stop ({days_held} days, current {quote.ltp:.2f})"

    return False, ""
