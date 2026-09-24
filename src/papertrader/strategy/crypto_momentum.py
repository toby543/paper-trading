"""Crypto momentum strategy using RSI and moving average crossovers.

Designed for 24/7 crypto markets with higher volatility and faster price moves.
Entry: Price above 20-day MA, RSI(14) > 50 (momentum phase), 20-day > 50-day MA (uptrend)
Exit: RSI drops below 40 (momentum loss), price closes below 20-day MA (trend break),
      or stop-loss/trailing-stop thresholds hit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from papertrader.data.nse_client import Quote
from papertrader.portfolio.models import Position


@dataclass
class Candidate:
    """One qualifying coin from a scan.

    Must be an object with .symbol/.ltp/.score attributes, NOT a plain
    dict: the shared entry path in TradingEngine.scan_for_entries()
    reads cand.ltp/cand.symbol off whatever rank_candidates() returns,
    exactly as it does for every other strategy's Candidate. Returning
    dicts here raised AttributeError on every scan that found anything,
    which run_forever()'s catch-all then swallowed -- so this strategy
    logged candidates every cycle but could never actually buy one.
    """
    symbol: str
    ltp: float
    ma20: float
    ma50: float
    rsi: float
    momentum_return_pct: float
    score: float


def _moving_average(history: pd.DataFrame, days: int) -> float | None:
    """Calculate simple moving average. Returns None if insufficient data."""
    if len(history) < days:
        return None
    return float(history["Close"].tail(days).mean())


def _rsi(history: pd.DataFrame, period: int = 14) -> float | None:
    """Calculate RSI(14). Returns None if it cannot be computed.

    A completely flat window has zero average gain AND zero average
    loss, so rs = 0/0 = NaN and the RSI comes out NaN. That must be
    reported as None, not handed back as a float: every comparison
    against NaN is False, so a NaN RSI would sail through the
    `rsi14 < min_rsi` entry gate and a dead, non-moving coin would be
    bought as though it had momentum. Callers already treat None as
    "insufficient data" and reject, which is the correct outcome.
    An all-gains window (zero losses) is a different case -- rs = inf
    gives RSI 100, which is genuinely correct and must be preserved.
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
    """Return % gain over the past lookback_days. Crypto typically uses 30-90 day lookback."""
    if len(history) <= lookback_days:
        return None
    lookback_idx = -lookback_days - 1
    past_close = history["Close"].iloc[lookback_idx]
    current_close = history["Close"].iloc[-1]
    if past_close <= 0:
        return None
    return ((current_close - past_close) / past_close) * 100


def evaluate_candidate(symbol: str, quote: Quote, history: pd.DataFrame, daily_turnover_inr: float,
                       config: dict, reasons: dict[str, int] | None = None) -> Candidate | None:
    """
    Evaluate a crypto candidate for entry.

    Returns dict with symbol, ltp, ma20, ma50, rsi, momentum_return_pct, score
    or None if doesn't qualify.

    Entry criteria:
    - Sufficient history (50 days for MAs)
    - Price > 20-day MA (above short-term support)
    - 20-day MA > 50-day MA (uptrend confirmation)
    - RSI(14) > 50 (momentum phase)
    - 30-day return >= config["min_momentum_return_pct"] (participation)
    - Adequate liquidity (daily_turnover >= min_avg_daily_turnover_inr)
    """
    if reasons is None:
        reasons = {}

    # Liquidity check first
    # Crypto turnover comes back in the pair's quote currency (USD for
    # every "<ASSET>-USD" symbol), so it must be compared against a USD
    # threshold. min_avg_daily_turnover_inr is an NSE rupee figure --
    # inheriting it here silently demanded $50M/day instead of the
    # ~₹5cr ($600K) it means on the equity side, ~83x too strict, which
    # rejected 16 of 24 coins (including LINK and ARB) as "illiquid".
    min_turnover = config.get("min_avg_daily_turnover_usd")
    if min_turnover is None:
        min_turnover = config.get("min_avg_daily_turnover_inr", 5_000_000)
    if daily_turnover_inr < min_turnover:
        reasons["illiquid"] = reasons.get("illiquid", 0) + 1
        return None

    # History sufficiency
    # Enough bars for the slower of the two averages actually configured,
    # not a fixed 50 -- otherwise raising slow_ma_days silently yields a
    # None MA and every symbol gets rejected as insufficient_history.
    if len(history) < int(config.get("slow_ma_days", 50)):
        reasons["insufficient_history"] = reasons.get("insufficient_history", 0) + 1
        return None

    # Read from config rather than hardcoded 20/50. These were fixed
    # constants while the Edit Settings panel displayed the inherited
    # equity values (50/200) for them -- so the dashboard showed two
    # numbers this strategy did not use and offered no way to change the
    # two it did. Defaults preserve the documented 20/50 crypto windows.
    fast_days = int(config.get("fast_ma_days", 20))
    slow_days = int(config.get("slow_ma_days", 50))
    ma20 = _moving_average(history, fast_days)
    ma50 = _moving_average(history, slow_days)
    rsi14 = _rsi(history, 14)
    momentum_days = config.get("momentum_lookback_days", 30)
    momentum_pct = _momentum_return_pct(history, momentum_days)

    # Validate all required indicators
    if ma20 is None or ma50 is None or momentum_pct is None:
        reasons["insufficient_history"] = reasons.get("insufficient_history", 0) + 1
        return None
    # Reported separately from insufficient_history: there IS enough
    # history here, the coin simply hasn't moved at all, so RSI is 0/0
    # and undefined. Folding it into insufficient_history would make the
    # dashboard's rejection breakdown claim a data-coverage problem for
    # what is really a dead market.
    if rsi14 is None:
        reasons["no_price_movement"] = reasons.get("no_price_movement", 0) + 1
        return None

    # Uptrend check: price > MA20 > MA50
    if quote.ltp <= ma20 or ma20 <= ma50:
        reasons["not_in_uptrend"] = reasons.get("not_in_uptrend", 0) + 1
        return None

    # RSI momentum check
    min_rsi = config.get("min_rsi", 50.0)
    if rsi14 < min_rsi:
        reasons["weak_momentum"] = reasons.get("weak_momentum", 0) + 1
        return None

    # Participation check
    min_momentum = config.get("min_momentum_return_pct", 5.0)
    if momentum_pct < min_momentum:
        reasons["weak_participation"] = reasons.get("weak_participation", 0) + 1
        return None

    # Score based on how far above MAs and RSI strength
    ma_score = ((quote.ltp - ma50) / ma50) * 100  # How far above 50-day MA
    rsi_score = (rsi14 - min_rsi) / (100 - min_rsi) * 100  # RSI strength
    momentum_score = min(momentum_pct / min_momentum * 100, 100)  # Cap momentum contribution

    score = (ma_score * 0.4 + rsi_score * 0.4 + momentum_score * 0.2)

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        ma20=ma20,
        ma50=ma50,
        rsi=rsi14,
        momentum_return_pct=momentum_pct,
        score=score,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Rank candidates by score, highest first."""
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def check_exit(position: Position, quote: Quote, history: pd.DataFrame, config: dict) -> tuple[bool, str]:
    """
    Determine if a crypto position should exit.

    Exit conditions (in order):
    1. Stop-loss: price < avg_price * (1 - stop_loss_pct/100)
    2. Trailing-stop: price < highest_close_since_entry * (1 - trailing_stop_pct/100)
    3. RSI < 40: momentum loss signal
    4. Price closes below 20-day MA: trend break
    """
    stop_loss_pct = config.get("stop_loss_pct", 10.0)
    trailing_stop_pct = config.get("trailing_stop_pct", 15.0)
    take_profit_pct = config.get("take_profit_pct", 0)

    # Stop-loss
    stop_price = position.avg_price * (1 - stop_loss_pct / 100)
    if quote.ltp < stop_price:
        return True, f"stop_loss ({stop_loss_pct}%)"

    # Trailing-stop
    trail_price = position.highest_close_since_entry * (1 - trailing_stop_pct / 100)
    if quote.ltp < trail_price:
        return True, f"trailing_stop ({trailing_stop_pct}%)"

    # Take-profit (only if configured > 0)
    if take_profit_pct > 0:
        target_price = position.avg_price * (1 + take_profit_pct / 100)
        if quote.ltp > target_price:
            return True, f"take_profit ({take_profit_pct}%)"

    # RSI momentum loss
    rsi14 = _rsi(history, 14)
    if rsi14 is not None and rsi14 < 40:
        return True, "rsi_momentum_loss (RSI < 40)"

    # Trend break: price closes below the fast MA (same window the entry
    # uptrend check uses, so an exit can't disagree with its own entry).
    fast_days = int(config.get("fast_ma_days", 20))
    ma_fast = _moving_average(history, fast_days)
    if ma_fast is not None and quote.ltp < ma_fast:
        return True, f"trend_break (below {fast_days}-day MA)"

    return False, ""
