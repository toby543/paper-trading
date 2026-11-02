"""Breadth regime rule, the Momentum entry gap, and pausing a strategy's entries."""
import pandas as pd
import pytest

from papertrader.data.nse_client import Quote
from papertrader.strategy import crypto_momentum
from papertrader.strategy.momentum_52w_high import (
    describe_regime_rule, regime_allows_entries, universe_breadth_pct,
)

RISING = pd.DataFrame({"Close": [50.0 + i for i in range(120)]})
FALLING = pd.DataFrame({"Close": [200.0 - i for i in range(120)]})
SHORT = pd.DataFrame({"Close": [1.0, 2.0]})


def test_breadth_counts_only_coins_with_enough_history():
    assert universe_breadth_pct([RISING, FALLING, SHORT], 50) == 50.0
    assert universe_breadth_pct([RISING, RISING, FALLING, None], 50) == pytest.approx(200 / 3)
    assert universe_breadth_pct([SHORT], 50) is None


def test_breadth_blocks_entries_when_too_few_coins_are_in_an_uptrend():
    cfg = {"enabled": True, "ma_days": 100, "breadth_min_pct": 60, "breadth_ma_days": 50}
    weak = lambda: [RISING, FALLING, FALLING]       # noqa: E731 - 33% above
    strong = lambda: [RISING, RISING, FALLING]      # noqa: E731 - 67% above
    assert regime_allows_entries(cfg, RISING, weak) is False      # Bitcoin fine, coins not
    assert regime_allows_entries(cfg, RISING, strong) is True
    assert regime_allows_entries(cfg, FALLING, strong) is False   # the index rule still applies


def test_breadth_is_off_unless_configured_and_only_fetches_when_on():
    called = []

    def histories():
        called.append(1)
        return [FALLING]
    assert regime_allows_entries({"enabled": True, "ma_days": 100}, RISING, histories) is True
    assert regime_allows_entries({"enabled": False, "breadth_min_pct": 90}, None, histories) is True
    assert not called
    # ma_days 0 means breadth only; no data at all fails open
    assert regime_allows_entries({"enabled": True, "ma_days": 0, "breadth_min_pct": 60}, None, histories) is False
    assert regime_allows_entries({"enabled": True, "ma_days": 0, "breadth_min_pct": 60}, None, lambda: [SHORT]) is True


def test_regime_rule_is_described_in_words():
    text = describe_regime_rule({"index_symbol": "BTC-USD", "ma_days": 100, "breadth_min_pct": 60, "breadth_ma_days": 50})
    assert "BTC-USD must be above its 100-day average" in text and "at least 60% of the universe" in text


def _quote(price):
    return Quote(symbol="X-USD", ltp=price, prev_close=price, week52_high=price, week52_low=price,
                 volume=1.0, timestamp=pd.Timestamp("2026-11-01").to_pydatetime(), source="test")


def test_momentum_can_require_room_above_its_20_day_average():
    history = pd.DataFrame({"Close": [100.0 + i * 0.5 for i in range(60)]})
    ma20 = float(history["Close"].tail(20).mean())
    cfg = {"min_avg_daily_turnover_inr": 0, "min_momentum_return_pct": 1.0, "min_rsi": 50.0}
    just_above = _quote(ma20 * 1.01)
    assert crypto_momentum.evaluate_candidate("X-USD", just_above, history, 1e12, cfg) is not None
    reasons = {}
    assert crypto_momentum.evaluate_candidate("X-USD", just_above, history, 1e12,
                                              {**cfg, "min_pct_above_fast_ma": 3.0}, reasons) is None
    assert reasons == {"too_close_to_fast_ma": 1}


def test_a_paused_profile_buys_nothing_but_says_why(tmp_path, monkeypatch):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine

    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / "p.db"))
    engine = TradingEngine(cfg, profile_name="crypto_mean_reversion")
    assert engine.strategy_cfg.get("pause_new_entries") is True

    def no_scan(*a, **k):
        raise AssertionError("a paused profile must not scan")
    monkeypatch.setattr(engine, "market_regime_ok", no_scan)
    engine.scan_for_entries()
    assert engine.get_scan_diagnostics()["status"] == "paused"
    assert not engine.broker.positions()


def test_the_regime_says_which_part_failed():
    from papertrader.strategy.momentum_52w_high import evaluate_regime
    cfg = {"enabled": True, "index_symbol": "BTC-USD", "ma_days": 100, "breadth_min_pct": 60, "breadth_ma_days": 50}
    ev = evaluate_regime(cfg, RISING, lambda: [RISING, FALLING, FALLING, FALLING])
    assert ev["ok"] is False and ev["index_ok"] is True and ev["breadth_ok"] is False
    assert ev["breadth_pct"] == 25.0
    assert ev["reason"] == "only 25% of the universe is above its 50-day average (needs 60%)"


def _crypto_engine(tmp_path, monkeypatch, profile="crypto_momentum"):
    from papertrader.config import Config
    from papertrader.engine.scheduler import TradingEngine
    cfg = Config.load()
    monkeypatch.setattr(cfg, "get_profile_state_file", lambda name=None: str(tmp_path / "r.db"))
    return TradingEngine(cfg, profile_name=profile)


def test_breadth_still_applies_when_the_index_cannot_be_fetched(tmp_path, monkeypatch):
    from papertrader.data.nse_client import DataUnavailableError
    engine = _crypto_engine(tmp_path, monkeypatch)

    def no_index(*a, **k):
        raise DataUnavailableError("yahoo down")
    engine.data.get_index_history = no_index
    engine.data.get_history = lambda symbol, *a, **k: FALLING
    assert engine.market_regime_ok() is False
    assert engine.last_regime["index_ok"] is None and engine.last_regime["breadth_ok"] is False
    engine.data.get_history = lambda symbol, *a, **k: RISING
    assert engine.market_regime_ok() is True


def test_live_breadth_ignores_todays_unfinished_bar(tmp_path, monkeypatch):
    engine = _crypto_engine(tmp_path, monkeypatch)
    days = pd.date_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=121, freq="D")
    closes = [50.0 + i for i in range(120)] + [1.0]          # today's partial bar has crashed
    history = pd.DataFrame({"Close": closes}, index=days)
    engine.data.get_index_history = lambda *a, **k: RISING
    engine.data.get_history = lambda symbol, *a, **k: history
    assert engine.market_regime_ok() is True                  # judged on completed closes


def test_dashboard_regime_reflects_weak_breadth():
    from papertrader.web.data_api import _market_regime

    class Data:
        def get_index_history(self, *a, **k):
            return RISING

    class Engine:
        is_manual = False
        data = Data()
        regime_cfg = {"enabled": True, "index_symbol": "BTC-USD", "ma_days": 100,
                      "breadth_min_pct": 60, "breadth_ma_days": 50}
        last_regime = {"breadth_pct": 4.3}
    r = _market_regime(Engine())
    assert r["status"] == "down" and r["label"].startswith("Weak breadth 4%")
    assert "only 4% of the universe" in r["reason"]
    Engine.last_regime = {"breadth_pct": 75.0}
    assert _market_regime(Engine())["status"] == "up"
