"""Cross-sectional momentum with an absolute-return gate ("dual momentum").

Ranks the *whole universe* against each other by trailing return each scan
and buys only the top percentile -- as opposed to `momentum_52w_high.py`,
which accepts/rejects each stock independently against fixed thresholds
(proximity to its own 52-week high, an absolute momentum floor). A stock
can qualify here with only modest momentum, as long as it's leading its
peers; conversely a stock with strong absolute momentum can still miss out
if the rest of the universe is running hotter still.

This is the "relative strength" leg of Gary Antonacci-style dual
momentum. The "absolute" leg -- only invest at all while the asset class
itself is trending up, otherwise sit in cash -- is provided by the
existing market regime filter (`regime.enabled`, checked by the engine
before any entry scan runs, both strategy modes share it) plus a
same-stock sanity check here (never buy something with negative trailing
return just because it beat its peers while still losing money).

Momentum is computed "12-1" style by default: trailing return over
`lookback_days`, ending `skip_recent_days` before the most recent close.
Skipping the most recent ~month is standard in academic momentum
research to avoid the well-documented short-term reversal effect
distorting the ranking -- pass `skip_recent_days: 0` to disable that and
use the raw trailing return through the latest close instead.

Trend confirmation (price above both moving averages) and the liquidity
and price-range filters are reused unchanged from the 52-week-high
strategy's config keys (`strategy.fast_ma_days` etc.) -- "leading the
pack" should still mean a stock in a genuine uptrend, not just the
least-worst decliner during a broad selloff.

Exits are unaffected: `momentum_52w_high.check_exit` (stop-loss, trailing
stop, momentum breakdown, optional take-profit) is generic risk
management, not specific to how a position was selected, and applies the
same way regardless of which strategy mode picked it.
"""
from __future__ import annotations

import pandas as pd

from ..data.nse_client import Quote
from .momentum_52w_high import Candidate, _moving_average


def _lookback_return_pct(history: pd.DataFrame, lookback_days: int, skip_recent_days: int = 0) -> float | None:
    """Trailing return over `lookback_days`, ending `skip_recent_days`
    before the last available close. None if there isn't enough history."""
    closes = history["Close"]
    end_idx = len(closes) - 1 - skip_recent_days
    start_idx = end_idx - lookback_days
    if start_idx < 0 or end_idx < 0:
        return None
    past = float(closes.iloc[start_idx])
    now = float(closes.iloc[end_idx])
    if past <= 0:
        return None
    return (now - past) / past * 100.0


def select_cross_sectional_candidates(
    universe_data: list[tuple[str, Quote, pd.DataFrame, float]],
    cfg: dict,
    reasons: dict[str, int] | None = None,
) -> list[Candidate]:
    """`universe_data` is (symbol, quote, history, avg_daily_turnover) for
    every not-currently-held symbol that had usable data this scan.

    Returns the top `cross_sectional.top_pct` percentile by trailing
    return as a ranked list of Candidate (best first, `score` set to the
    percentile rank 0-100) -- a drop-in replacement for
    `rank_candidates(...)` in the 52w-high strategy, ready for the same
    position-sizing/buy loop.

    `reasons` is an optional counter, matching the other two strategy
    modules: each screened-out symbol increments the filter that dropped
    it, plus `outside_top_pct` for those that passed every filter but
    didn't make the percentile cut. Without it a scan that selects nothing
    gives no clue whether the filters or the ranking were responsible.
    """
    def reject(reason: str, count: int = 1) -> None:
        if reasons is not None and count > 0:
            reasons[reason] = reasons.get(reason, 0) + count

    cs_cfg = cfg.get("cross_sectional") or {}
    lookback_days = cs_cfg.get("lookback_days", 252)
    skip_recent_days = cs_cfg.get("skip_recent_days", 21)
    top_pct = cs_cfg.get("top_pct", 10.0)

    fast_ma_days = cfg.get("fast_ma_days", 50)
    slow_ma_days = cfg.get("slow_ma_days", 200)
    min_turnover = cfg.get("min_avg_daily_turnover_inr", 0)
    min_ltp = cfg.get("min_ltp_inr")
    max_ltp = cfg.get("max_ltp_inr")

    scored: list[tuple[str, Quote, float, float]] = []  # symbol, quote, turnover, momentum
    for symbol, quote, history, turnover in universe_data:
        if min_ltp and quote.ltp < min_ltp:
            reject("ltp_below_min")
            continue
        if max_ltp and quote.ltp > max_ltp:
            reject("ltp_above_max")
            continue
        if turnover < min_turnover:
            reject("illiquid")
            continue

        fast_ma = _moving_average(history, fast_ma_days)
        slow_ma = _moving_average(history, slow_ma_days)
        if fast_ma is None or slow_ma is None:
            reject("insufficient_history")
            continue
        if not (quote.ltp > fast_ma > 0 and quote.ltp > slow_ma and fast_ma >= slow_ma):
            reject("not_in_uptrend")
            continue

        momentum = _lookback_return_pct(history, lookback_days, skip_recent_days)
        if momentum is None:
            reject("insufficient_history")
            continue
        if momentum <= 0:
            # Absolute-return sanity check: never buy something merely for
            # beating its peers while it's itself still losing money.
            reject("negative_trailing_return")
            continue

        scored.append((symbol, quote, turnover, momentum))

    if not scored:
        return []

    scored.sort(key=lambda t: t[3], reverse=True)
    n = len(scored)
    cutoff = max(1, round(n * top_pct / 100.0))
    top = scored[:cutoff]
    reject("outside_top_pct", n - len(top))

    candidates = []
    for rank, (symbol, quote, turnover, momentum) in enumerate(top):
        percentile = (n - rank) / n * 100.0
        pct_from_high = (quote.week52_high - quote.ltp) / quote.week52_high * 100.0 if quote.week52_high > 0 else 0.0
        candidates.append(Candidate(
            symbol=symbol,
            ltp=quote.ltp,
            week52_high=quote.week52_high,
            pct_from_52w_high=pct_from_high,
            momentum_return_pct=momentum,
            avg_daily_turnover=turnover,
            score=percentile,
        ))
    return candidates
