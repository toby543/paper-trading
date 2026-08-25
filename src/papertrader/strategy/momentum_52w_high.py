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
  5. (optional) It is outperforming the broader index over the same
     lookback window by at least `min_relative_strength_pct` -- a stock
     merely drifting up with a rising market isn't the same signal as
     one genuinely leading it.
  6. (optional) Its recent trading volume is running hot relative to its
     own baseline (`volume_confirmation`) -- a breakout near a 52-week
     high on light volume is a weaker signal than one with real
     participation behind it.

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
    relative_strength_pct: float | None = None
    volume_multiple: float | None = None


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


def _relative_strength_pct(history: pd.DataFrame, index_history: pd.DataFrame, lookback_days: int) -> float | None:
    """How much more (or less) the stock has returned than the index over
    the same window, in percentage points. None if either return can't be
    computed (not enough history yet)."""
    stock_return = _momentum_return_pct(history, lookback_days)
    index_return = _momentum_return_pct(index_history, lookback_days)
    if stock_return is None or index_return is None:
        return None
    return stock_return - index_return


def _volume_multiple(history: pd.DataFrame, recent_days: int, baseline_days: int) -> float | None:
    """Recent average volume as a multiple of the longer-run baseline
    average -- >1 means trading activity has picked up. None if volume
    data isn't available or a baseline can't be computed."""
    try:
        baseline_avg = float(history["Volume"].tail(baseline_days).mean())
        if not baseline_avg or baseline_avg <= 0:
            return None
        recent_avg = float(history["Volume"].tail(recent_days).mean())
    except (KeyError, TypeError, ZeroDivisionError):
        return None
    return recent_avg / baseline_avg


def evaluate_candidate(
    symbol: str,
    quote: Quote,
    history: pd.DataFrame,
    avg_daily_turnover: float,
    cfg: dict,
    index_history: pd.DataFrame | None = None,
) -> Candidate | None:
    """Return a Candidate if `symbol` currently qualifies as a BUY, else None.

    `index_history` is optional: pass the broader index's daily bars to
    enable the relative-strength filter/score component. Without it, that
    check is simply skipped (fails open), same as any other filter whose
    config key is absent.
    """
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

    relative_strength = None
    min_rs = cfg.get("min_relative_strength_pct")
    if index_history is not None:
        relative_strength = _relative_strength_pct(history, index_history, cfg["momentum_lookback_days"])
        if min_rs is not None and relative_strength is not None and relative_strength < min_rs:
            return None

    volume_multiple = None
    volume_cfg = cfg.get("volume_confirmation") or {}
    if volume_cfg.get("enabled", False):
        volume_multiple = _volume_multiple(
            history,
            recent_days=volume_cfg.get("recent_days", 10),
            baseline_days=volume_cfg.get("baseline_days", 50),
        )
        min_multiple = volume_cfg.get("min_volume_multiple", 1.0)
        if volume_multiple is not None and volume_multiple < min_multiple:
            return None

    # Higher momentum, closer proximity to the 52-week high, and stronger
    # relative strength vs. the index all increase the score; proximity
    # and relative strength contribute as bounded/secondary terms so raw
    # momentum can't be swamped by either single axis.
    proximity_bonus = max(0.0, cfg["proximity_to_52w_high_pct"] - pct_from_high)
    score = momentum + proximity_bonus
    if relative_strength is not None:
        score += max(0.0, relative_strength) * 0.5

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        week52_high=quote.week52_high,
        pct_from_52w_high=pct_from_high,
        momentum_return_pct=momentum,
        avg_daily_turnover=avg_daily_turnover,
        score=score,
        relative_strength_pct=relative_strength,
        volume_multiple=volume_multiple,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def is_market_in_uptrend(index_history: pd.DataFrame, ma_days: int) -> bool:
    """Market regime filter: true when the index's last close is above its
    own `ma_days` moving average. Momentum-at-new-highs strategies tend to
    whipsaw badly when the broader market is in a downtrend, so this gates
    new entries (never exits -- risk management still applies regardless
    of regime)."""
    ma = _moving_average(index_history, ma_days)
    if ma is None:
        return True  # not enough index history yet; fail open rather than freezing entries
    last_close = float(index_history["Close"].iloc[-1])
    return last_close > ma


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
