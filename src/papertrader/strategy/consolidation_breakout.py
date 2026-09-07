"""Consolidation breakout momentum strategy with Phase 2 screening.

Entry filters:
  1. Stock in clear uptrend (price > 50 DMA > 200 DMA)
  2. Forms tight consolidation/base for several days
  3. Breaks out above base high on strong volume (≥ 2× average)
  4. Entry: At or just above breakout level
  5. (Phase 2) Market cap > ₹20,000 crore
  6. (Phase 2) Beta ≤ 1.0 (avoid wild volatility)
  7. (Phase 2) Part of Nifty 50 / Nifty Next 50 / sector leaders
  
Exit rules:
  1. Hard stop-loss at or below consolidation low
  2. Trailing stop from highest close since entry
  3. Momentum breakdown (close below fast MA)
  4. (optional) Take profit at fixed level
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from ..data.nse_client import Quote
from ..portfolio.models import Position


@dataclass
class Candidate:
    symbol: str
    ltp: float
    consolidation_high: float
    consolidation_low: float
    breakout_volume: float
    avg_volume: float
    momentum_return_pct: float
    score: float
    beta: float | None = None
    market_cap_cr: float | None = None
    in_index: bool | None = None


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


def _avg_volume(history: pd.DataFrame, window: int) -> float | None:
    if len(history) < window:
        return None
    try:
        return float(history["Volume"].tail(window).mean())
    except (KeyError, TypeError):
        return None


def _calculate_beta(history: pd.DataFrame, market_history: pd.DataFrame, window: int = 252) -> float | None:
    """Calculate stock beta relative to market index (e.g., Nifty 50).
    
    Beta = covariance(stock returns, market returns) / variance(market returns)
    """
    if len(history) < window or len(market_history) < window:
        return None
    
    try:
        # Calculate daily returns
        stock_closes = history["Close"].tail(window)
        market_closes = market_history["Close"].tail(window)
        
        if len(stock_closes) < 2 or len(market_closes) < 2:
            return None
        
        stock_returns = stock_closes.pct_change().dropna()
        market_returns = market_closes.pct_change().dropna()
        
        if len(stock_returns) < 2 or len(market_returns) < 2:
            return None
        
        # Align lengths
        min_len = min(len(stock_returns), len(market_returns))
        stock_returns = stock_returns.iloc[-min_len:]
        market_returns = market_returns.iloc[-min_len:]
        
        # Calculate covariance and variance
        covariance = (stock_returns * market_returns).mean() - (stock_returns.mean() * market_returns.mean())
        variance = market_returns.var()
        
        if variance == 0:
            return None
        
        beta = covariance / variance
        return float(beta) if not math.isnan(beta) else None
    except (KeyError, TypeError, ValueError):
        return None


def _detect_consolidation(history: pd.DataFrame, consolidation_days: int) -> tuple[float, float, bool] | None:
    """Detect tight consolidation over recent days.
    
    Returns (high, low, is_consolidation) if consolidation found, else None.
    A consolidation is tight if the range (high-low)/average_price is small."""
    if len(history) < consolidation_days:
        return None
    
    recent = history.tail(consolidation_days)
    high = float(recent["High"].max())
    low = float(recent["Low"].min())
    close = float(history["Close"].iloc[-1])
    
    # Tight range check: range < 3% of current price
    range_pct = (high - low) / close * 100.0 if close > 0 else 100.0
    is_tight = range_pct < 3.0
    
    return (high, low, is_tight)


def _detect_breakout(history: pd.DataFrame, consolidation_high: float, volume_multiple: float) -> bool:
    """Detect if price broke above consolidation high on strong volume."""
    if len(history) < 2:
        return False
    
    current_close = float(history["Close"].iloc[-1])
    current_volume = float(history["Volume"].iloc[-1])
    
    # Broke above consolidation high
    if current_close <= consolidation_high:
        return False
    
    # Volume check: current volume >= multiple of baseline average
    avg_vol = _avg_volume(history, 20)
    if avg_vol is None or avg_vol <= 0:
        return False
    
    return current_volume >= avg_vol * volume_multiple


def evaluate_candidate(
    symbol: str,
    quote: Quote,
    history: pd.DataFrame,
    avg_daily_turnover: float,
    cfg: dict,
    index_history: pd.DataFrame | None = None,
    market_cap_cr: float | None = None,
    in_nifty_index: bool | None = None,
) -> Candidate | None:
    """Return a Candidate if symbol currently qualifies for consolidation breakout entry."""
    
    # Uptrend check: price > 50 DMA > 200 DMA
    fast_ma = _moving_average(history, cfg.get("fast_ma_days", 50))
    slow_ma = _moving_average(history, cfg.get("slow_ma_days", 200))
    
    if fast_ma is None or slow_ma is None:
        return None
    if not (quote.ltp > fast_ma > 0 and quote.ltp > slow_ma and fast_ma >= slow_ma):
        return None
    
    # Liquidity check
    if avg_daily_turnover < cfg.get("min_avg_daily_turnover_inr", 500000):
        return None
    
    # consolidation_days/volume_multiple live under strategy.consolidation_breakout
    # in config.yaml, same as the Phase 2 fields below -- read them from the
    # same nested sub-dict rather than the top level, where they'd silently
    # always fall back to the hardcoded default no matter what's configured.
    cb_cfg = cfg.get("consolidation_breakout") or {}

    # Detect consolidation
    consolidation_days = cb_cfg.get("consolidation_days", 10)
    consolidation = _detect_consolidation(history, consolidation_days)

    if consolidation is None:
        return None

    consolidation_high, consolidation_low, is_tight = consolidation
    if not is_tight:
        return None

    # Detect breakout
    volume_multiple = cb_cfg.get("volume_multiple", 2.0)
    if not _detect_breakout(history, consolidation_high, volume_multiple):
        return None

    # Momentum check
    momentum = _momentum_return_pct(history, cfg.get("momentum_lookback_days", 21))
    if momentum is None or momentum < cfg.get("min_momentum_return_pct", 5.0):
        return None

    # Phase 2: Market cap filter
    market_cap_min = cb_cfg.get("market_cap_min_cr")
    if market_cap_min and market_cap_cr and market_cap_cr < market_cap_min:
        return None
    
    # Phase 2: Beta filter
    beta = None
    beta_max = cb_cfg.get("beta_max")
    if index_history is not None and beta_max:
        beta = _calculate_beta(history, index_history)
        if beta is not None and beta > beta_max:
            return None
    
    # Phase 2: Index membership filter
    index_list = cb_cfg.get("index_list")
    if index_list and in_nifty_index is False:
        return None
    
    # Score: higher momentum and closer to breakout level score higher
    distance_from_breakout = max(0, consolidation_high - quote.ltp)
    score = momentum - (distance_from_breakout / consolidation_high) * 10.0
    
    # Bonus for large-cap, low-beta stocks
    if market_cap_cr and market_cap_cr > market_cap_min:
        score += min(2.0, (market_cap_cr - market_cap_min) / market_cap_min)
    if beta is not None and beta < 1.0:
        score += 1.0
    
    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        consolidation_high=consolidation_high,
        consolidation_low=consolidation_low,
        breakout_volume=float(history["Volume"].iloc[-1]),
        avg_volume=_avg_volume(history, 20) or 0.0,
        momentum_return_pct=momentum,
        score=score,
        beta=beta,
        market_cap_cr=market_cap_cr,
        in_index=in_nifty_index,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def check_exit(position: Position, quote: Quote, history: pd.DataFrame, cfg: dict) -> tuple[bool, str]:
    """Return (should_exit, reason)."""
    
    # Hard stop-loss (below consolidation low or fixed percentage)
    stop_loss_pct = cfg.get("stop_loss_pct", 4.0)
    stop_loss_price = position.avg_price * (1 - stop_loss_pct / 100.0)
    if quote.ltp <= stop_loss_price:
        return True, f"stop_loss ({stop_loss_pct}% below entry {position.avg_price:.2f})"
    
    # Trailing stop
    trailing_stop_pct = cfg.get("trailing_stop_pct", 8.0)
    trailing_stop_price = position.highest_close_since_entry * (1 - trailing_stop_pct / 100.0)
    if quote.ltp <= trailing_stop_price:
        return True, f"trailing_stop ({trailing_stop_pct}% below peak {position.highest_close_since_entry:.2f})"
    
    # Take profit (optional)
    take_profit_pct = cfg.get("take_profit_pct")
    if take_profit_pct:
        take_profit_price = position.avg_price * (1 + take_profit_pct / 100.0)
        if quote.ltp >= take_profit_price:
            return True, f"take_profit (+{take_profit_pct}% above entry {position.avg_price:.2f})"
    
    # Momentum breakdown (close below fast MA)
    if cfg.get("exit_below_fast_ma", True):
        fast_ma = _moving_average(history, cfg.get("fast_ma_days", 50))
        if fast_ma is not None and quote.ltp < fast_ma:
            return True, f"momentum_breakdown (close below {cfg.get('fast_ma_days', 50)}DMA {fast_ma:.2f})"
    
    return False, ""
