"""Trend Pullback strategy, crypto-native variant.

Same premise as trend_pullback.py (the equity version): wait for an
already-established uptrend to pull back a shallow, controlled amount,
then buy the first sign it's turning back up, rather than chasing a
breakout or buying at a fresh high. This is a separate module -- not a
shared one with trend_pullback.py -- for the same reason crypto_breakout
is separate from consolidation_breakout: the equity version's
min_ltp_inr/max_ltp_inr price-band filters don't apply to crypto (a coin's
absolute price is meaningless -- SHIB-USD at $0.00001 and BTC-USD at
$80,000 are both perfectly tradeable), and crypto's fast/slow MA windows
(20/50) and turnover thresholds (INR, converted from the coin's USD
quote) follow the convention every other crypto strategy here uses,
not the equity defaults (50/200, INR-native).

  1. Uptrend confirmation: price > fast MA > slow MA.
  2. Pullback: today's close sits between `min_pullback_pct` and
     `max_pullback_pct` below the highest close of the last
     `pullback_lookback_days` -- shallow enough the trend is intact,
     deep enough to be a real pullback rather than noise.
  3. Turn confirmation: today closes above yesterday's close -- the
     first sign of the pullback resolving back up.
  4. Liquidity, same convention as every other crypto strategy.

Exit: universal stop-loss / trailing-stop / take-profit, plus a
trend-break exit on a close below the SLOW MA (not the fast one -- a
pullback entry sits close to or below the fast MA by construction, so
using it as the trend-broken signal would exit almost immediately).
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


def _moving_average(history: pd.DataFrame, days: int) -> float | None:
    if len(history) < days:
        return None
    return float(history["Close"].tail(days).mean())


def _recent_high(history: pd.DataFrame, lookback_days: int) -> float | None:
    if len(history) < lookback_days:
        return None
    return float(history["Close"].tail(lookback_days).max())


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

    fast_days = int(config.get("fast_ma_days", 20))
    slow_days = int(config.get("slow_ma_days", 50))
    fast_ma = _moving_average(history, fast_days)
    slow_ma = _moving_average(history, slow_days)
    if fast_ma is None or slow_ma is None:
        reject("insufficient_history")
        return None
    if not (quote.ltp > fast_ma > 0 and fast_ma > slow_ma):
        reject("not_in_uptrend")
        return None

    tp_cfg = config.get("crypto_trend_pullback") or {}
    lookback_days = tp_cfg.get("pullback_lookback_days", 20)
    recent_high = _recent_high(history, lookback_days)
    if recent_high is None or recent_high <= 0:
        reject("insufficient_history")
        return None

    pct_from_high = (recent_high - quote.ltp) / recent_high * 100.0
    min_pullback_pct = tp_cfg.get("min_pullback_pct", 5.0)
    max_pullback_pct = tp_cfg.get("max_pullback_pct", 20.0)
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

    shallowness = max_pullback_pct - pct_from_high
    score = shallowness + day_change_pct * 2.0

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        recent_high=round(recent_high, 8),
        pct_from_high=round(pct_from_high, 2),
        day_change_pct=round(day_change_pct, 2),
        score=score,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def check_exit(position: Position, quote: Quote, history: pd.DataFrame, config: dict) -> tuple[bool, str]:
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

    slow_ma_days = int(config.get("slow_ma_days", 50))
    slow_ma = _moving_average(history, slow_ma_days)
    if slow_ma is not None and quote.ltp < slow_ma:
        return True, f"trend_break (close below {slow_ma_days}DMA {slow_ma:.2f})"

    return False, ""
