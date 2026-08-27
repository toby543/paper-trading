"""Pure statistics helpers for backtest results -- no pandas/network
dependency, so these are unit-testable in isolation from the walk-forward
simulation itself."""
from __future__ import annotations


def max_drawdown_pct(equity_curve: list[float]) -> float:
    """Largest peak-to-trough decline in the equity curve, as a positive
    percentage (e.g. 18.4 means an 18.4% drawdown occurred at some point).
    0.0 for an empty or ever-rising curve."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        elif peak > 0:
            dd = (peak - value) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


def cagr_pct(start_value: float, end_value: float, days: int) -> float:
    """Compound annual growth rate, as a percentage. 0.0 for degenerate
    inputs (non-positive values or non-positive duration)."""
    if start_value <= 0 or end_value <= 0 or days <= 0:
        return 0.0
    years = days / 365.25
    return ((end_value / start_value) ** (1.0 / years) - 1.0) * 100.0


def win_rate_pct(trade_pnls: list[float]) -> float:
    """Percentage of closed round-trip trades (by realized P&L) that were
    profitable. 0.0 if there were no closed trades."""
    if not trade_pnls:
        return 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls) * 100.0


def avg_value(values: list[float]) -> float:
    """Plain average, 0.0 for an empty list (avoids a ZeroDivisionError
    at every call site that reports avg win/loss when there are none)."""
    return sum(values) / len(values) if values else 0.0
