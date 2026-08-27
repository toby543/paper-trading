"""Backtest engine: replays the exact same strategy/exit logic used live
(evaluate_candidate, check_exit, rank_candidates, is_market_in_uptrend)
against historical daily bars, day by day, so a strategy or config change
can be validated against the past in minutes instead of waiting months of
live trading for a statistically meaningful read.

Deliberately reuses PaperBroker/Storage/RiskManager unchanged (against a
throwaway temp-file SQLite database, never the live ledger) so position
sizing, slippage, and charges match live trading exactly. The only real
difference from `python main.py run`: all price history is fetched once
up front instead of live per-scan network calls, and time is simulated
day-by-day instead of wall-clock -- the strategy/exit functions themselves
are the identical code paths, so there is no risk of backtest and live
behavior silently diverging.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from ..config import Config
from ..data.nse_client import Quote
from ..data.universe import load_universe
from ..portfolio.broker import InsufficientFundsError, PaperBroker
from ..portfolio.storage import Storage
from ..risk.risk_manager import RiskManager
from ..strategy.cross_sectional_momentum import select_cross_sectional_candidates
from ..strategy.momentum_52w_high import (
    Candidate,
    check_exit,
    evaluate_candidate,
    is_market_in_uptrend,
    rank_candidates,
)
from .metrics import avg_value, cagr_pct, max_drawdown_pct, win_rate_pct

log = logging.getLogger(__name__)

# Trading-day approximation of "52 weeks", matching how a ~1y lookback
# behaves elsewhere in this codebase (e.g. yfinance's period="1y").
_WEEK52_TRADING_DAYS = 252


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    trading_days: int
    symbols_with_data: int
    starting_capital: float
    ending_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    num_round_trips: int
    win_rate_pct: float
    avg_win_inr: float
    avg_loss_inr: float
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)


def _avg_daily_turnover(history: pd.DataFrame, days: int = 20) -> float:
    recent = history.tail(days)
    turnover = (recent["Close"] * recent["Volume"]).mean()
    return float(turnover) if pd.notna(turnover) else 0.0


class Backtester:
    def __init__(
        self,
        cfg: Config,
        start: str,
        end: str,
        lookback_buffer_days: int = 420,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.cfg = cfg
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        if self.end <= self.start:
            raise ValueError(f"end ({end}) must be after start ({start})")
        self.lookback_buffer_days = lookback_buffer_days
        # Optional (stage, current, total) callback -- lets a caller (e.g.
        # the web dashboard, running this in a background thread) surface
        # progress without coupling this module to Flask/threading at all.
        self._on_progress = on_progress or (lambda stage, current, total: None)

        self.strategy_cfg = cfg.get("strategy", default={})
        self.risk_cfg = cfg.get("risk", default={})
        self.regime_cfg = cfg.get("regime", default={})
        self.universe = load_universe(cfg.universe_file)

        if self.strategy_cfg.get("mode") == "cross_sectional_momentum":
            # This mode's lookback (default 252 trading days + 21 skipped)
            # can exceed the generic 420-day buffer for a bigger
            # lookback_days config -- make sure the fetch window is always
            # wide enough for it, on top of whatever the caller passed.
            cs_cfg = self.strategy_cfg.get("cross_sectional") or {}
            needed = cs_cfg.get("lookback_days", 252) + cs_cfg.get("skip_recent_days", 21) + 30
            self.lookback_buffer_days = max(self.lookback_buffer_days, needed)

        self.risk = RiskManager(
            max_open_positions=cfg.get("risk", "max_open_positions", default=10),
            position_size_pct_of_equity=cfg.get("risk", "position_size_pct_of_equity", default=8.0),
            max_cash_deployed_per_scan_pct=cfg.get("risk", "max_cash_deployed_per_scan_pct", default=40.0),
        )
        self.starting_capital = float(cfg.get("account", "starting_capital", default=1_000_000.0))

        fd, self._tmp_db = tempfile.mkstemp(suffix=".db", prefix="papertrader-backtest-")
        os.close(fd)
        self.storage = Storage(self._tmp_db, self.starting_capital)
        self.broker = PaperBroker(
            self.storage,
            slippage_bps=cfg.get("execution", "slippage_bps", default=5.0),
            flat_charges_inr=cfg.get("execution", "flat_charges_inr", default=20.0),
        )

        self._history: dict[str, pd.DataFrame] = {}
        self._index_history: pd.DataFrame | None = None

    def __del__(self):
        try:
            os.remove(self._tmp_db)
        except OSError:
            pass

    # Minimum spacing between consecutive Yahoo Finance requests. Firing
    # requests for a 100-500 symbol universe back-to-back with no pacing
    # is exactly the pattern that trips Yahoo's rate limiting/anti-abuse
    # throttling, which then surfaces as a 404 -- indistinguishable from a
    # genuinely delisted symbol unless several real, actively-traded
    # stocks failing in the same run tips you off to suspect throttling.
    _YFINANCE_MIN_INTERVAL_SECONDS = 0.2

    def _fetch(self) -> None:
        import time as _time

        import yfinance as yf

        fetch_start = self.start - pd.Timedelta(days=self.lookback_buffer_days)
        fetch_end = self.end + pd.Timedelta(days=1)  # yfinance's `end` is exclusive
        log.info("Fetching %d symbols from %s to %s (includes lookback buffer for MAs/momentum)...",
                  len(self.universe), fetch_start.date(), self.end.date())

        last_call = 0.0
        for i, symbol in enumerate(self.universe):
            elapsed = _time.time() - last_call
            if elapsed < self._YFINANCE_MIN_INTERVAL_SECONDS:
                _time.sleep(self._YFINANCE_MIN_INTERVAL_SECONDS - elapsed)
            last_call = _time.time()
            try:
                df = yf.Ticker(symbol + ".NS").history(start=fetch_start, end=fetch_end)
                if not df.empty:
                    df.index = df.index.tz_localize(None)
                    self._history[symbol] = df
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not abort the whole backtest
                log.debug("Skipping %s: %s", symbol, exc)
            self._on_progress("fetch", i + 1, len(self.universe))
            if (i + 1) % 50 == 0:
                log.info("Fetch progress: %d/%d symbols (%d resolved so far)", i + 1, len(self.universe), len(self._history))

        index_symbol = self.regime_cfg.get("index_symbol", "^NSEI")
        try:
            idx = yf.Ticker(index_symbol).history(start=fetch_start, end=fetch_end)
            if not idx.empty:
                idx.index = idx.index.tz_localize(None)
                self._index_history = idx
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not fetch benchmark index %s (%s); regime/relative-strength checks will fail open", index_symbol, exc)

        log.info("Fetched history for %d/%d symbols.", len(self._history), len(self.universe))

    # ------------------------------------------------------------------
    def _quote_for(self, symbol: str, history_upto: pd.DataFrame) -> Quote | None:
        if len(history_upto) < 2:
            return None
        last = history_upto.iloc[-1]
        prev = history_upto.iloc[-2]
        window = history_upto.tail(_WEEK52_TRADING_DAYS)
        return Quote(
            symbol=symbol,
            ltp=float(last["Close"]),
            prev_close=float(prev["Close"]),
            week52_high=float(window["High"].max()),
            week52_low=float(window["Low"].min()),
            volume=float(last["Volume"]),
            timestamp=history_upto.index[-1].to_pydatetime(),
            source="backtest",
        )

    def _market_regime_ok(self, index_upto: pd.DataFrame | None) -> bool:
        if not self.regime_cfg.get("enabled", False):
            return True
        if index_upto is None or len(index_upto) < 2:
            return True  # fail open, matching live behavior when index data is unavailable
        return is_market_in_uptrend(index_upto, self.regime_cfg.get("ma_days", 200))

    # ------------------------------------------------------------------
    def _run_exits(self, day: pd.Timestamp, trade_pnls: list[float], trade_log: list[dict]) -> None:
        for symbol, pos in list(self.broker.positions().items()):
            hist = self._history.get(symbol)
            if hist is None:
                continue
            history_upto = hist.loc[:day]
            quote = self._quote_for(symbol, history_upto)
            if quote is None:
                continue

            self.broker.update_trailing_high(symbol, quote.ltp)
            should_exit, reason = check_exit(pos, quote, history_upto, {**self.risk_cfg, **self.strategy_cfg})
            if not should_exit:
                continue

            pre_qty, pre_avg = pos.quantity, pos.avg_price
            try:
                trade = self.broker.sell(symbol, pre_qty, quote.ltp, reason)
            except ValueError:
                continue
            realized = pre_qty * (trade.price - pre_avg) - trade.charges
            trade_pnls.append(realized)
            trade_log.append({
                "date": str(day.date()), "side": "SELL", "symbol": symbol,
                "qty": pre_qty, "price": round(trade.price, 2), "reason": reason, "pnl": round(realized, 2),
            })

    def _run_entries(self, day: pd.Timestamp, trade_log: list[dict]) -> None:
        positions = self.broker.positions()
        room = self.risk.room_for_new_positions(len(positions))
        if room <= 0:
            return

        index_upto = self._index_history.loc[:day] if self._index_history is not None else None
        if not self._market_regime_ok(index_upto):
            return

        mode = self.strategy_cfg.get("mode", "52w_high")

        if mode == "cross_sectional_momentum":
            universe_data = []
            for symbol in self.universe:
                if symbol in positions:
                    continue
                hist = self._history.get(symbol)
                if hist is None:
                    continue
                history_upto = hist.loc[:day]
                quote = self._quote_for(symbol, history_upto)
                if quote is None:
                    continue
                turnover = _avg_daily_turnover(history_upto)
                universe_data.append((symbol, quote, history_upto, turnover))
            ranked = select_cross_sectional_candidates(universe_data, self.strategy_cfg)
        else:
            candidates: list[Candidate] = []
            for symbol in self.universe:
                if symbol in positions:
                    continue
                hist = self._history.get(symbol)
                if hist is None:
                    continue
                history_upto = hist.loc[:day]
                quote = self._quote_for(symbol, history_upto)
                if quote is None:
                    continue
                turnover = _avg_daily_turnover(history_upto)
                cand = evaluate_candidate(symbol, quote, history_upto, turnover, self.strategy_cfg, index_history=index_upto)
                if cand:
                    candidates.append(cand)
            ranked = rank_candidates(candidates)

        max_new = min(room, self.strategy_cfg.get("max_new_positions_per_scan", 3))
        ranked = ranked[:max_new]
        if not ranked:
            return

        free_cash = self.broker.cash()
        scan_budget = self.risk.scan_cash_budget(free_cash)
        equity_now = self.broker.equity({})
        spent = 0.0
        for cand in ranked:
            qty = self.risk.position_size_shares(equity_now, cand.ltp)
            if qty <= 0:
                continue
            cost_estimate = qty * cand.ltp
            if spent + cost_estimate > scan_budget:
                continue
            try:
                trade = self.broker.buy(cand.symbol, qty, cand.ltp, reason=f"score={cand.score:.1f}")
            except InsufficientFundsError:
                continue
            spent += cost_estimate
            trade_log.append({
                "date": str(day.date()), "side": "BUY", "symbol": cand.symbol,
                "qty": qty, "price": round(trade.price, 2), "reason": trade.reason, "pnl": None,
            })

    def _mark_to_market(self, day: pd.Timestamp) -> float:
        quotes = {}
        for symbol in self.broker.positions():
            hist = self._history.get(symbol)
            if hist is None:
                continue
            history_upto = hist.loc[:day]
            if len(history_upto):
                quotes[symbol] = float(history_upto.iloc[-1]["Close"])
        positions_value = sum(p.quantity * quotes.get(s, p.avg_price) for s, p in self.broker.positions().items())
        return self.broker.cash() + positions_value

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        self._fetch()
        if not self._history:
            raise RuntimeError(
                "No historical data could be fetched for any symbol in the universe. "
                "Check network access and that the universe file has valid NSE symbols."
            )

        trading_days = pd.bdate_range(self.start, self.end)
        equity_curve: list[tuple[str, float]] = []
        trade_pnls: list[float] = []
        trade_log: list[dict] = []

        for i, day in enumerate(trading_days):
            self._run_exits(day, trade_pnls, trade_log)
            self._run_entries(day, trade_log)
            equity_curve.append((str(day.date()), self._mark_to_market(day)))
            self._on_progress("simulate", i + 1, len(trading_days))
            if (i + 1) % 50 == 0:
                log.info("Backtest progress: %d/%d trading days simulated", i + 1, len(trading_days))

        ending_equity = equity_curve[-1][1] if equity_curve else self.starting_capital
        equity_values = [v for _, v in equity_curve]
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        elapsed_days = (self.end - self.start).days

        return BacktestResult(
            start_date=str(self.start.date()),
            end_date=str(self.end.date()),
            trading_days=len(trading_days),
            symbols_with_data=len(self._history),
            starting_capital=self.starting_capital,
            ending_equity=ending_equity,
            total_return_pct=((ending_equity - self.starting_capital) / self.starting_capital * 100.0)
            if self.starting_capital else 0.0,
            cagr_pct=cagr_pct(self.starting_capital, ending_equity, elapsed_days),
            max_drawdown_pct=max_drawdown_pct(equity_values),
            num_round_trips=len(trade_pnls),
            win_rate_pct=win_rate_pct(trade_pnls),
            avg_win_inr=avg_value(wins),
            avg_loss_inr=avg_value(losses),
            equity_curve=equity_curve,
            trade_log=trade_log,
        )
