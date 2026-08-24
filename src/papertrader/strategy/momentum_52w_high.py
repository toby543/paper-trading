"""52-week-high momentum swing strategy.

Classic momentum-at-new-highs approach ("it's not too late to buy a stock
making a new high"): a stock is a BUY candidate when

  1. It is trading within `proximity_to_52w_high_pct` of its 52-week high
     (i.e. it is at, or close to breaking out to, a fresh high).
  2. It has delivered strong trailing momentum: at least
     `min_momentum_return_pct` return over `momentum_lookback_days`.
  3. It confirms trend: price is above both the fast and slow moving
     averages (fast above slow too, i.e. no death-cross).
  4. It is liquid enough to trade in size (average daily turnover filter).

Exits (handled by `check_exit`) are a hard stop-loss from entry, a
trailing stop from the highest close recorded since entry, or a momentum
breakdown (close crosses below the fast moving average).
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
    week52_high: float
    pct_from_52w_high: float
    momentum_return_pct: float
    avg_daily_turnover: float
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


def evaluate_candidate(symbol: str, quote: Quote, history: pd.DataFrame, avg_daily_turnover: float, cfg: dict) -> Candidate | None:
    """Return a Candidate if `symbol` currently qualifies as a BUY, else None."""
    if quote.week52_high <= 0:
        return None

    pct_from_high = (quote.week52_high - quote.ltp) / quote.week52_high * 100.0
    if pct_from_high > cfg["proximity_to_52w_high_pct"]:
        return None

    momentum = _momentum_return_pct(history, cfg["momentum_lookback_days"])
    if momentum is None or momentum < cfg["min_momentum_return_pct"]:
        return None

    fast_ma = _moving_average(history, cfg["fast_ma_days"])
    slow_ma = _moving_average(history, cfg["slow_ma_days"])
    if fast_ma is None or slow_ma is None:
        return None
    if not (quote.ltp > fast_ma > 0 and quote.ltp > slow_ma and fast_ma >= slow_ma):
        return None

    if avg_daily_turnover < cfg["min_avg_daily_turnover_inr"]:
        return None

    # Higher momentum and closer proximity to the 52-week high both
    # increase the score; proximity contributes as a bounded bonus so a
    # single mega-momentum outlier can't dominate purely on that axis.
    proximity_bonus = max(0.0, cfg["proximity_to_52w_high_pct"] - pct_from_high)
    score = momentum + proximity_bonus

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        week52_high=quote.week52_high,
        pct_from_52w_high=pct_from_high,
        momentum_return_pct=momentum,
        avg_daily_turnover=avg_daily_turnover,
        score=score,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def check_exit(position: Position, quote: Quote, history: pd.DataFrame, cfg: dict) -> tuple[bool, str]:
    """Return (should_exit, reason)."""
    stop_loss_price = position.avg_price * (1 - cfg["stop_loss_pct"] / 100.0)
    if quote.ltp <= stop_loss_price:
        return True, f"stop_loss ({cfg['stop_loss_pct']}% below entry {position.avg_price:.2f})"

    trailing_stop_price = position.highest_close_since_entry * (1 - cfg["trailing_stop_pct"] / 100.0)
    if quote.ltp <= trailing_stop_price:
        return True, f"trailing_stop ({cfg['trailing_stop_pct']}% below peak {position.highest_close_since_entry:.2f})"

    if cfg.get("exit_below_fast_ma", True):
        fast_ma = _moving_average(history, cfg.get("fast_ma_days", 50))
        if fast_ma is not None and quote.ltp < fast_ma:
            return True, f"momentum_breakdown (close below {cfg.get('fast_ma_days', 50)}DMA {fast_ma:.2f})"

    return False, ""
