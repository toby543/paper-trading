"""Pairs-trading strategy for crypto markets, adapted for a long-only spot book.

Classic pairs trading is market-neutral: long the cheap leg, short the
expensive leg of two correlated instruments, profiting from their ratio
reverting regardless of which direction the market as a whole moves. This
app's broker/risk layer has no short-selling anywhere in it (see
PaperBroker/RiskManager) -- it is a long-only spot paper book, matching
how a retail crypto exchange account actually works. A literal
long/short pairs trade isn't representable here.

What this strategy does instead is the long-only adaptation that's
actually common in crypto: treat BTC-USD as the market benchmark every
altcoin is "paired" against, and buy an altcoin when its price *relative
to BTC* is statistically far below its own recent average -- i.e. it has
lagged Bitcoin's move by an unusual amount -- on the expectation that the
ratio reverts (the altcoin catches up, or BTC cools off relative to it).
This is a real, common form of crypto relative-value trading ("alt/BTC
mean reversion"), just expressed as a single long position in the lagging
coin rather than a simultaneous long/short pair.

  1. Compute the coin's price ratio to BTC-USD over the shared history,
     and that ratio's rolling mean/std over `pairs_lookback_days`.
  2. Entry: the ratio's current z-score is at or below
     `-entry_z_threshold` -- the coin is unusually cheap versus BTC right
     now, relative to its own recent relationship with BTC.
  3. Neither leg can be in an outright structural downtrend: both the
     coin and BTC itself must be above their own `trend_ma_days` moving
     average. Without this, "cheap versus BTC" could just mean "crashing
     alongside BTC in a bear market", which is not a relative-value setup.
  4. Liquidity, same convention as every other crypto strategy.

Exit:
  1. Reversion target: the ratio's z-score has recovered to
     `exit_z_threshold` (default 0 -- back to its own recent average).
  2. Universal stop-loss (the ratio can just keep falling instead of
     reverting) and time stop (default 10 days -- if the ratio hasn't
     reverted by then, the relationship has likely structurally shifted,
     not just temporarily diverged).

`evaluate_candidate`/`check_exit` both take an extra `benchmark_history`
keyword the other crypto strategies don't have -- the scheduler fetches
BTC-USD's history once per scan (see the "Fetch the benchmark index once
per scan" pattern already used for the 52w_high strategy's relative
strength filter) rather than once per candidate coin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ..data.nse_client import Quote
from ..portfolio.models import Position

BENCHMARK_SYMBOL_DEFAULT = "BTC-USD"


@dataclass
class Candidate:
    symbol: str
    ltp: float
    ratio: float
    ratio_mean: float
    z_score: float
    score: float


def _moving_average(history: pd.DataFrame, days: int) -> float | None:
    if len(history) < days:
        return None
    return float(history["Close"].tail(days).mean())


def _ratio_series(coin_history: pd.DataFrame, benchmark_history: pd.DataFrame) -> pd.Series | None:
    """Coin close / benchmark close, aligned on their shared dates.

    The two histories can come from different exchange sources with
    slightly different bar timestamps/coverage (see nse_client.py's
    Binance-then-Kraken-then-Yahoo fallback chain), so this aligns on the
    intersection of their date-only indices rather than assuming the two
    frames line up row-for-row.
    """
    coin = coin_history["Close"].copy()
    bench = benchmark_history["Close"].copy()
    coin.index = pd.to_datetime(coin.index).normalize()
    bench.index = pd.to_datetime(bench.index).normalize()
    common = coin.index.intersection(bench.index)
    if len(common) < 20:
        return None
    coin = coin.loc[common]
    bench = bench.loc[common]
    bench = bench.replace(0, pd.NA)
    ratio = (coin / bench).dropna()
    return ratio if len(ratio) >= 20 else None


def _ratio_zscore(ratio: pd.Series, lookback_days: int) -> tuple[float, float, float] | None:
    """(current ratio, rolling mean, z-score) over the last lookback_days,
    or None if there isn't enough ratio history yet."""
    if len(ratio) < lookback_days:
        return None
    window = ratio.tail(lookback_days)
    mean = float(window.mean())
    std = float(window.std())
    if not (std > 0) or math.isnan(std):
        return None
    current = float(ratio.iloc[-1])
    z = (current - mean) / std
    return current, mean, z


def evaluate_candidate(
    symbol: str,
    quote: Quote,
    history: pd.DataFrame,
    daily_turnover_inr: float,
    config: dict,
    benchmark_history: pd.DataFrame | None = None,
    reasons: dict[str, int] | None = None,
) -> Candidate | None:
    if reasons is None:
        reasons = {}

    def reject(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    if benchmark_history is None or benchmark_history.empty:
        reject("no_benchmark_data")
        return None

    min_turnover = config.get("min_avg_daily_turnover_usd")
    if min_turnover is None:
        min_turnover = config.get("min_avg_daily_turnover_inr", 480000000)
    if daily_turnover_inr < min_turnover:
        reject("illiquid")
        return None

    pt_cfg = config.get("crypto_pairs_trading") or {}
    trend_ma_days = int(pt_cfg.get("trend_ma_days", 100))
    lookback_days = int(pt_cfg.get("pairs_lookback_days", 30))

    coin_trend_ma = _moving_average(history, trend_ma_days)
    bench_trend_ma = _moving_average(benchmark_history, trend_ma_days)
    if coin_trend_ma is None or bench_trend_ma is None:
        reject("insufficient_history")
        return None

    # Neither leg may be in an outright structural downtrend -- otherwise
    # "cheap versus BTC" is indistinguishable from "crashing with BTC".
    if quote.ltp <= coin_trend_ma:
        reject("coin_structural_downtrend")
        return None
    bench_last = float(benchmark_history["Close"].iloc[-1])
    if bench_last <= bench_trend_ma:
        reject("benchmark_structural_downtrend")
        return None

    ratio = _ratio_series(history, benchmark_history)
    if ratio is None:
        reject("insufficient_history")
        return None

    z_result = _ratio_zscore(ratio, lookback_days)
    if z_result is None:
        reject("insufficient_history")
        return None
    current_ratio, ratio_mean, z_score = z_result

    entry_z_threshold = pt_cfg.get("entry_z_threshold", 2.0)
    if z_score > -entry_z_threshold:
        reject("ratio_not_stretched")
        return None

    # Score: the more stretched (more negative) the z-score, the more
    # unusual -- and therefore more attractive -- the relative-value
    # setup, within the bound the filters above already enforce.
    score = -z_score * 10.0

    return Candidate(
        symbol=symbol,
        ltp=quote.ltp,
        ratio=round(current_ratio, 10),
        ratio_mean=round(ratio_mean, 10),
        z_score=round(z_score, 2),
        score=score,
    )


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def check_exit(
    position: Position,
    quote: Quote,
    history: pd.DataFrame,
    config: dict,
    benchmark_history: pd.DataFrame | None = None,
) -> tuple[bool, str]:
    stop_loss_pct = config.get("stop_loss_pct", 10.0)
    stop_price = position.avg_price * (1 - stop_loss_pct / 100.0)
    if quote.ltp <= stop_price:
        return True, f"stop_loss ({stop_loss_pct}% below entry {position.avg_price:.2f})"

    # Reversion target, if the benchmark history fetch succeeded this
    # cycle -- gracefully falls through to the time/stop-loss safety net
    # below if it didn't, rather than failing the whole exit check.
    if benchmark_history is not None and not benchmark_history.empty:
        pt_cfg = config.get("crypto_pairs_trading") or {}
        lookback_days = int(pt_cfg.get("pairs_lookback_days", 30))
        ratio = _ratio_series(history, benchmark_history)
        if ratio is not None:
            z_result = _ratio_zscore(ratio, lookback_days)
            if z_result is not None:
                _current_ratio, _ratio_mean, z_score = z_result
                exit_z_threshold = pt_cfg.get("exit_z_threshold", 0.0)
                if z_score >= exit_z_threshold:
                    return True, f"reverted_to_mean (z={z_score:.2f})"

    time_stop_days = config.get("time_stop_days", 10)
    entry_ts = pd.Timestamp(position.entry_date)
    days_held = (pd.Timestamp.now() - entry_ts).days
    if days_held >= time_stop_days:
        return True, f"time_stop ({days_held} days, current {quote.ltp:.2f})"

    return False, ""
