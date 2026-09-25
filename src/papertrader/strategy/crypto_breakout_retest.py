"""Breakout-and-retest strategy for crypto markets.

crypto_breakout.py buys the breakout candle itself -- the moment price
closes above a consolidation base on strong volume. That entry is early
but it's also buying into an already-extended move, at whatever price the
breakout happened to close at. This strategy is the more conservative
sibling: it waits for price to break out, then pull back down to retest
the former resistance level (now expected to act as support), and buys
the bounce off that retest -- a better average entry price and a much
tighter, more logical stop (just below the retested level) than buying
the breakout bar outright, at the cost of missing every breakout that
never comes back to retest at all (a strong minority of them, historically).

  1. Establish a base: the highest close over `base_lookback_days`,
     measured in the window *before* the breakout search window (see
     `_find_base_and_breakout` for why the two windows can't overlap).
  2. Confirm a breakout happened: within the last `retest_window_days`
     bars, some bar closed above the base level by at least
     `breakout_confirm_pct`.
  3. Confirm the retest: today's close is within `retest_tolerance_pct`
     of the base level (back testing the old resistance as support) and
     closes above yesterday's close (turning back up off the level, not
     still falling through it).
  4. Liquidity, same convention as every other crypto strategy.

Exit: universal stop-loss / trailing-stop / take-profit, plus a
trend-break exit on a close back below the retested level itself -- a
failed retest invalidates this trade's whole premise.
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
    breakout_level: float
    pct_from_level: float
    day_change_pct: float
    score: float


def _avg_volume(history: pd.DataFrame, days: int) -> float | None:
    if len(history) < days:
        return None
    try:
        return float(history["Volume"].tail(days).mean())
    except (KeyError, TypeError):
        return None


def _find_base_and_breakout(
    history: pd.DataFrame, base_lookback_days: int, retest_window_days: int, breakout_confirm_pct: float,
) -> float | None:
    """Highest close of the base window if a breakout above it actually
    happened within the retest window, else None.

    The base window and the retest/breakout-search window must not
    overlap: the base has to be a level set *before* the breakout, or a
    breakout bar would inflate the very base level it's supposed to have
    broken out of, making a genuine breakout look like it never
    happened (the same bug class _detect_consolidation in
    crypto_breakout.py guards against, for the same reason).
    """
    total_needed = base_lookback_days + retest_window_days + 1  # +1 excludes today from both windows
    if len(history) < total_needed:
        return None

    # Today (index -1) is excluded from both windows: it's evaluated
    # separately, as the retest bar itself.
    search_window = history.iloc[-(retest_window_days + 1):-1]
    base_window = history.iloc[-(retest_window_days + base_lookback_days + 1):-(retest_window_days + 1)]
    if base_window.empty or search_window.empty:
        return None

    base_level = float(base_window["Close"].max())
    if not (base_level > 0) or math.isnan(base_level):
        return None

    breakout_threshold = base_level * (1 + breakout_confirm_pct / 100.0)
    broke_out = bool((search_window["Close"] > breakout_threshold).any())
    if not broke_out:
        return None

    return base_level


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

    br_cfg = config.get("crypto_breakout_retest") or {}
    base_lookback_days = int(br_cfg.get("base_lookback_days", 20))
    retest_window_days = int(br_cfg.get("retest_window_days", 10))
    breakout_confirm_pct = br_cfg.get("breakout_confirm_pct", 3.0)

    breakout_level = _find_base_and_breakout(history, base_lookback_days, retest_window_days, breakout_confirm_pct)
    if breakout_level is None:
        reject("no_confirmed_breakout")
        return None

    pct_from_level = abs(quote.ltp - breakout_level) / breakout_level * 100.0
    retest_tolerance_pct = br_cfg.get("retest_tolerance_pct", 4.0)
    if pct_from_level > retest_tolerance_pct:
        reject("not_at_retest_level")
        return None

    # Must be holding above the retested level, not failing back below
    # it -- a retest that's breaking back down through support isn't a
    # bounce, it's the breakout failing.
    if quote.ltp < breakout_level:
        reject("retest_failing")
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

    # Score: closer to the level (a tighter, more logical stop) and a
    # stronger turn-up day both score higher.
    tightness = retest_tolerance_pct - pct_from_level
    score = tightness * 2.0 + day_change_pct * 2.0

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        breakout_level=round(breakout_level, 8),
        pct_from_level=round(pct_from_level, 2),
        day_change_pct=round(day_change_pct, 2),
        score=score,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def check_exit(position: Position, quote: Quote, history: pd.DataFrame, config: dict) -> tuple[bool, str]:
    stop_loss_pct = config.get("stop_loss_pct", 8.0)
    stop_price = position.avg_price * (1 - stop_loss_pct / 100.0)
    if quote.ltp <= stop_price:
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

    # A failed retest invalidates the whole premise of this trade: the
    # level it bounced off entry is meant to hold as support. Re-derive
    # it the same way entry did (the base lookback/retest window before
    # today), and exit if price has closed back below it.
    br_cfg = config.get("crypto_breakout_retest") or {}
    base_lookback_days = int(br_cfg.get("base_lookback_days", 20))
    retest_window_days = int(br_cfg.get("retest_window_days", 10))
    if len(history) >= base_lookback_days + retest_window_days + 1:
        base_window = history.iloc[-(retest_window_days + base_lookback_days + 1):-(retest_window_days + 1)]
        if not base_window.empty:
            level = float(base_window["Close"].max())
            if level > 0 and quote.ltp < level:
                return True, f"retest_failed (below level {level:.2f})"

    return False, ""
