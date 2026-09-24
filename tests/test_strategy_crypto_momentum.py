"""Tests for the crypto momentum strategy."""
from datetime import datetime

import pandas as pd

from papertrader.data.nse_client import Quote
from papertrader.portfolio.models import Position
from papertrader.strategy.crypto_momentum import (
    _momentum_return_pct,
    _moving_average,
    _rsi,
    check_exit,
    evaluate_candidate,
    rank_candidates,
)

# Crypto-focused config with 30-day lookback and lower momentum threshold
CFG = {
    "min_avg_daily_turnover_inr": 50_000_000,
    "momentum_lookback_days": 30,
    "min_momentum_return_pct": 5.0,
    "min_rsi": 50.0,
}


def _history_from_closes(closes: list[float]) -> pd.DataFrame:
    """closes[-1] is "today" (current/reference bar)."""
    idx = pd.date_range(end=pd.Timestamp.today(), periods=len(closes), freq="B")
    high = [c * 1.02 for c in closes]  # Crypto has wider wicks
    low = [c * 0.98 for c in closes]
    df = pd.DataFrame({"Close": closes, "High": high, "Low": low}, index=idx)
    df["Volume"] = 1_000_000.0
    return df


def _quote(ltp: float) -> Quote:
    return Quote(symbol="BTCUSD", ltp=ltp, prev_close=ltp, week52_high=ltp * 1.5,
                 week52_low=ltp * 0.5, volume=10_000_000, timestamp=datetime.now(), source="test")


def test_moving_average_needs_the_full_window():
    hist = _history_from_closes([100.0] * 19)
    assert _moving_average(hist, 20) is None
    hist = _history_from_closes([100.0] * 20)
    assert _moving_average(hist, 20) == 100.0


def test_rsi_needs_sufficient_data():
    # Non-flat closes, so this isolates the bar-count condition: a FLAT
    # window is undefined (0/0) regardless of length and also returns
    # None -- covered by test_flat_window_gives_undefined_rsi_not_nan.
    rising = [100.0 + i for i in range(15)]
    assert _rsi(_history_from_closes(rising[:14]), 14) is None
    assert _rsi(_history_from_closes(rising), 14) is not None


def test_momentum_return_pct_uses_the_lookback_window():
    closes = [100.0] * 30 + [105.0]  # 30-day return is 5%
    hist = _history_from_closes(closes)
    assert _momentum_return_pct(hist, 30) == 5.0


def test_uptrend_with_rsi_momentum_qualifies():
    """Steady uptrend with strong RSI should qualify."""
    closes = [100.0]
    for _ in range(50):
        closes.append(closes[-1] * 1.01)  # 1% daily gains
    hist = _history_from_closes(closes)
    cand = evaluate_candidate("BTCUSD", _quote(closes[-1]), hist, 100_000_000, CFG)
    assert cand is not None
    assert cand.ma20 > cand.ma50
    assert cand.rsi > CFG["min_rsi"]
    assert cand.momentum_return_pct >= CFG["min_momentum_return_pct"]


def test_flat_price_action_rejected():
    """Sideways movement lacks momentum."""
    closes = [100.0] * 60
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("BTCUSD", _quote(100.0), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    # Flat closes make RSI undefined (0/0), which is its own reason --
    # reported distinctly from insufficient_history so the dashboard
    # doesn't blame data coverage for what is really a dead market.
    assert "no_price_movement" in reasons


def test_declining_trend_rejected():
    """Downtrend should not qualify even with liquidity."""
    closes = [100.0]
    for _ in range(60):
        closes.append(closes[-1] * 0.99)  # 1% daily losses
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("BTCUSD", _quote(closes[-1]), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("not_in_uptrend") == 1


def test_illiquid_pair_rejected():
    """Low daily turnover disqualifies even with great trend."""
    closes = [100.0]
    for _ in range(50):
        closes.append(closes[-1] * 1.01)
    hist = _history_from_closes(closes)
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("BTCUSD", _quote(closes[-1]), hist, 1000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("illiquid") == 1


def test_insufficient_history_rejected():
    """Need at least 50 bars for MA calculation."""
    hist = _history_from_closes([100.0, 101.0])
    reasons: dict[str, int] = {}
    cand = evaluate_candidate("BTCUSD", _quote(101.0), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None
    assert reasons.get("insufficient_history") == 1


def _position(avg_price: float, highest_close: float) -> Position:
    return Position(symbol="BTCUSD", quantity=0.5, avg_price=avg_price,
                    entry_date="2024-01-01", highest_close_since_entry=highest_close)


RISK_CFG = {"stop_loss_pct": 10.0, "trailing_stop_pct": 15.0, "take_profit_pct": 0}


def test_stop_loss_triggers():
    """Stop-loss should trigger at configured threshold."""
    pos = _position(avg_price=100.0, highest_close=100.0)
    hist = _history_from_closes([100.0] * 60)
    should_exit, reason = check_exit(pos, _quote(89.0), hist, RISK_CFG)  # 11% below entry
    assert should_exit
    assert "stop_loss" in reason


def test_trailing_stop_triggers():
    """Trailing-stop should trigger on retracement from peak."""
    pos = _position(avg_price=100.0, highest_close=150.0)  # Peak at 150
    hist = _history_from_closes([100.0] * 60)
    should_exit, reason = check_exit(pos, _quote(126.0), hist, RISK_CFG)  # 16% below peak
    assert should_exit
    assert "trailing_stop" in reason


def test_ordinary_pullback_within_stops():
    """Small pullback within stop thresholds should not exit."""
    pos = _position(avg_price=100.0, highest_close=120.0)
    hist = _history_from_closes([100.0] * 60)
    should_exit, reason = check_exit(pos, _quote(110.0), hist, RISK_CFG)  # 8.3% below peak
    assert not should_exit


def test_rsi_momentum_loss_triggers_exit():
    """RSI < 40 signals momentum loss and exit.

    Needs a genuinely DECLINING window, not a flat one: a flat window has
    zero average gain and zero average loss, so RSI is undefined (NaN ->
    None) rather than low, and the exit that fires is the 20-day-MA
    trend break instead.
    """
    pos = _position(avg_price=100.0, highest_close=100.0)
    closes = [100.0] * 30 + [100.0 - i * 0.5 for i in range(1, 21)]
    hist = _history_from_closes(closes)
    # Price held above the 20-day MA so only the RSI condition can fire.
    ma20 = _moving_average(hist, 20)
    should_exit, reason = check_exit(pos, _quote(ma20 + 1.0), hist, RISK_CFG)
    assert should_exit
    assert "rsi_momentum_loss" in reason


def test_trend_break_below_20ma():
    """Close below the 20-day MA signals a trend break.

    RSI is evaluated before the trend break, so the window has to keep
    RSI healthy (a rising trend) and then drop price below the MA --
    otherwise the RSI condition fires first and this asserts nothing
    about the trend break at all.
    """
    pos = _position(avg_price=100.0, highest_close=110.0)
    closes = [100.0 + i for i in range(30)]  # steady climb keeps RSI high
    hist = _history_from_closes(closes)
    ma20 = _moving_average(hist, 20)
    assert _rsi(hist, 14) >= 40, "setup must not trip the RSI exit"
    should_exit, reason = check_exit(pos, _quote(ma20 - 1.0), hist, RISK_CFG)
    assert should_exit
    assert "trend_break" in reason


def test_flat_window_gives_undefined_rsi_not_nan():
    """A flat window has zero gains AND zero losses, so RSI is 0/0.

    It must come back as None, never a NaN float: every comparison
    against NaN is False, so a NaN would slip past the `rsi < min_rsi`
    entry gate and a dead, non-moving coin would be bought as though it
    had momentum.
    """
    hist = _history_from_closes([100.0] * 40)
    assert _rsi(hist, 14) is None

    reasons: dict[str, int] = {}
    cand = evaluate_candidate("DEAD-USD", _quote(100.0), hist, 100_000_000, CFG, reasons=reasons)
    assert cand is None, "a coin with no price movement must never qualify"


def test_pure_uptrend_rsi_is_100_not_none():
    """Zero losses is rs = inf -> RSI 100, which is correct and must be
    preserved -- only the 0/0 flat case is undefined."""
    closes = [100.0 * (1.01 ** i) for i in range(40)]
    hist = _history_from_closes(closes)
    assert _rsi(hist, 14) == 100.0


def test_candidate_exposes_attributes_not_dict_keys():
    """The shared entry path in TradingEngine.scan_for_entries() reads
    cand.symbol/cand.ltp off whatever rank_candidates() returns. This
    strategy originally returned plain dicts, so every scan that found a
    candidate raised AttributeError -- swallowed by run_forever()'s
    catch-all, which made the profile look idle while it was in fact
    crashing every cycle and could never place a single trade. Assert the
    attribute access the engine actually performs."""
    closes = [100.0]
    for _ in range(50):
        closes.append(closes[-1] * 1.01)
    hist = _history_from_closes(closes)
    cand = evaluate_candidate("BTC-USD", _quote(closes[-1]), hist, 100_000_000, CFG)
    assert cand is not None

    ranked = rank_candidates([cand])
    assert ranked, "a qualifying candidate must survive ranking"
    # Exactly what the engine does -- not subscripting.
    assert ranked[0].symbol == "BTC-USD"
    assert ranked[0].ltp > 0
    assert ranked[0].score is not None


def test_usd_turnover_threshold_preferred_over_inr():
    """Crypto turnover is quoted in the pair's USD quote currency, so a
    crypto profile sets min_avg_daily_turnover_usd. It must take
    precedence over the NSE rupee field, whose value compared against USD
    was ~83x too strict and rejected most of the universe as illiquid."""
    closes = [100.0]
    for _ in range(50):
        closes.append(closes[-1] * 1.01)
    hist = _history_from_closes(closes)
    cfg = {**CFG, "min_avg_daily_turnover_inr": 50_000_000,
           "min_avg_daily_turnover_usd": 5_000_000}

    # $10M/day clears the USD floor but would fail the stale INR one.
    assert evaluate_candidate("LINK-USD", _quote(closes[-1]), hist, 10_000_000, cfg) is not None
    # Genuinely thin -- rejected under the USD floor.
    assert evaluate_candidate("GMX-USD", _quote(closes[-1]), hist, 300_000, cfg) is None
