"""Tests for the "why isn't this profile buying anything" insights logic.

Needs pandas transitively (data_api.py imports momentum_52w_high, which
does), so this can't run in a pandas-free environment -- see
test_strategy_consolidation_breakout.py for the same situation. The four
scan-diagnostics states below were hand-verified against a transplanted
copy of this exact logic before being written here; this file is the
executable version of that check.
"""
from papertrader.web.data_api import _build_insights


class FakeRisk:
    def __init__(self, max_open_positions=10):
        self.max_open_positions = max_open_positions


class FakeStorage:
    """get_trades() is the only method _build_insights calls on storage
    (for the recent-realised-loss-streak insight, unrelated to what these
    tests exercise) -- always return no trades so that section is a no-op."""
    def get_trades(self, limit=None):
        return []


class FakeEngine:
    def __init__(self, diagnostics=None, max_open_positions=10):
        self._diagnostics = diagnostics
        self.risk_cfg = {"stop_loss_pct": 7.0, "trailing_stop_pct": 12.0}
        self.regime_cfg = {"index_symbol": "^NSEI", "ma_days": 200}
        self.risk = FakeRisk(max_open_positions)
        self.storage = FakeStorage()
        self.cfg = _FakeCfg()

    def get_scan_diagnostics(self):
        return self._diagnostics


class _FakeCfg:
    def get(self, *keys, default=None):
        # Only "engine"/"scan_interval_minutes" is read by the engine-health
        # check, which these tests don't exercise (last_scan_at handles that).
        return default


def _titles(insights):
    return [i["title"] for i in insights]


def test_regime_blocked_gives_the_specific_reason():
    engine = FakeEngine(diagnostics={
        "status": "regime_blocked", "mode": "52w_high", "scanned": 0, "candidates": 0, "rejections": {},
    })
    insights = _build_insights(engine, position_rows=[], cash=90_000, total_equity=100_000,
                               market_open=True, last_scan_at="2026-09-09T10:00:00")
    assert "Market regime filter blocked the last scan" in _titles(insights)
    assert not any("in cash" in t for t in _titles(insights))  # generic fallback must not also fire


def test_scan_failed_is_a_warning():
    engine = FakeEngine(diagnostics={
        "status": "scan_failed", "mode": "cross_sectional_momentum", "scanned": 50, "candidates": 0, "rejections": {},
    })
    insights = _build_insights(engine, position_rows=[], cash=90_000, total_equity=100_000,
                               market_open=True, last_scan_at="2026-09-09T10:00:00")
    matches = [i for i in insights if "scan failed" in i["title"]]
    assert len(matches) == 1
    assert matches[0]["level"] == "warn"
    assert "cross_sectional_momentum" in matches[0]["title"]


def test_zero_candidates_surfaces_top_rejections():
    engine = FakeEngine(diagnostics={
        "status": "scanned", "mode": "52w_high", "scanned": 435, "candidates": 0,
        "rejections": {"far_from_52w_high": 18471, "weak_volume": 723, "weak_momentum": 640, "illiquid": 15},
    })
    insights = _build_insights(engine, position_rows=[], cash=90_000, total_equity=100_000,
                               market_open=True, last_scan_at="2026-09-09T10:00:00")
    match = next(i for i in insights if "0 candidates" in i["title"])
    assert "435 symbols evaluated" in match["title"]
    # Only the top 3 rejections, in descending order -- not all 4.
    assert "far_from_52w_high (18,471)" in match["detail"]
    assert "weak_volume (723)" in match["detail"]
    assert "weak_momentum (640)" in match["detail"]
    assert "illiquid" not in match["detail"]


def test_no_diagnostics_yet_falls_back_to_generic_idle_capital_message():
    """Before this profile's first scan (e.g. right after a restart),
    get_scan_diagnostics() returns None -- the dashboard must still say
    something rather than nothing."""
    engine = FakeEngine(diagnostics=None)
    insights = _build_insights(engine, position_rows=[], cash=90_000, total_equity=100_000,
                               market_open=True, last_scan_at=None)
    assert any("in cash" in t and "position slot" in t for t in _titles(insights))


def test_portfolio_full_shows_fully_invested_not_a_scan_reason():
    """room<=0 short-circuits before any symbol is evaluated, so status is
    "portfolio_full" with scanned=0 -- that must produce the existing
    "Fully invested" message, not a confusing "0 candidates" one."""
    engine = FakeEngine(diagnostics={
        "status": "portfolio_full", "mode": "52w_high", "scanned": 0, "candidates": 0, "rejections": {},
    }, max_open_positions=3)
    position_rows = [
        {"symbol": s, "market_value": 1000, "avg_price": 100, "ltp": 100,
         "highest_close_since_entry": 100, "unrealized_pnl_pct": 0.0}
        for s in ("A", "B", "C")
    ]
    insights = _build_insights(engine, position_rows=position_rows, cash=0, total_equity=3000,
                               market_open=True, last_scan_at="2026-09-09T10:00:00")
    assert "Fully invested" in _titles(insights)
    assert not any("candidates" in t for t in _titles(insights))
