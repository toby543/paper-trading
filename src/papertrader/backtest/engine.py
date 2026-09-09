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
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from ..config import Config
from ..data import price_cache
from ..data.nse_client import Quote
from ..data.universe import load_universe
from ..portfolio.broker import InsufficientFundsError, PaperBroker
from ..portfolio.storage import Storage
from ..risk.risk_manager import RiskManager
from ..strategy.cross_sectional_momentum import select_cross_sectional_candidates
from ..strategy.momentum_52w_high import (
    Candidate as Candidate52w,
    check_exit as exit_52w,
    evaluate_candidate as eval_52w,
    is_market_in_uptrend,
    rank_candidates as rank_52w,
)
from ..strategy import consolidation_breakout
from .metrics import avg_value, cagr_pct, max_drawdown_pct, win_rate_pct

log = logging.getLogger(__name__)

# Trading-day approximation of "52 weeks", matching how a ~1y lookback
# behaves elsewhere in this codebase (e.g. yfinance's period="1y").
_WEEK52_TRADING_DAYS = 252

# NSE trades roughly 250 days a year, so N trading days span appreciably
# more than N calendar days. Conflating the two silently under-fetches.
_CALENDAR_DAYS_PER_TRADING_DAY = 365.0 / 250.0


def _calendar_days_for(trading_days: int) -> int:
    """Calendar-day span that reliably contains `trading_days` sessions,
    with a margin for holiday clusters (Diwali/Holi weeks, etc)."""
    return int(math.ceil(trading_days * _CALENDAR_DAYS_PER_TRADING_DAY)) + 30


def _required_trading_days(strategy_cfg: dict) -> int:
    """Longest lookback any filter in this strategy config needs before it
    can return a value at all."""
    needed = max(
        int(strategy_cfg.get("slow_ma_days", 200) or 0),
        int(strategy_cfg.get("fast_ma_days", 50) or 0),
        _WEEK52_TRADING_DAYS,  # _quote_for's 52-week high/low window
    )
    mode = strategy_cfg.get("mode", "52w_high")
    if mode == "cross_sectional_momentum":
        cs_cfg = strategy_cfg.get("cross_sectional") or {}
        needed = max(needed, int(cs_cfg.get("lookback_days", 252))
                     + int(cs_cfg.get("skip_recent_days", 21)) + 1)
    else:
        needed = max(needed, int(strategy_cfg.get("momentum_lookback_days", 252) or 0) + 1)
    if mode == "consolidation_breakout":
        cb_cfg = strategy_cfg.get("consolidation_breakout") or {}
        needed = max(needed, int(cb_cfg.get("consolidation_days", 10)) + 1)
    return needed


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
    benchmark_symbol: str = ""
    # None (not a genuine 0.0) when the benchmark index couldn't be
    # fetched or didn't have at least two priced days in the window --
    # distinguishes "no data" from "a real 0% return".
    benchmark_total_return_pct: float | None = None
    benchmark_cagr_pct: float | None = None
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    # Why entries didn't happen. A backtest that trades nothing is a valid
    # result, but without this it is indistinguishable from one that is
    # broken or misconfigured -- and the reader has no way to tell which
    # filter to loosen. `entry_rejections` tallies, across every simulated
    # day and symbol, which filter rejected the symbol.
    profile: str = ""
    days_regime_blocked: int = 0
    days_portfolio_full: int = 0
    days_with_candidates: int = 0
    entry_rejections: dict[str, int] = field(default_factory=dict)


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
        refresh_cache: bool = False,
        profile_name: str | None = None,
    ):
        self.cfg = cfg
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        if self.end <= self.start:
            raise ValueError(f"end ({end}) must be after start ({start})")
        self.lookback_buffer_days = lookback_buffer_days
        # Which profile's strategy this backtest replays. Defaults to the
        # active one so a backtest always mirrors what that profile is
        # actually trading live -- including any parameters edited from the
        # dashboard, which are stored per-profile rather than in the shared
        # top-level strategy: block.
        self.profile_name = profile_name or cfg.get("active_profile", default="52w_high")
        # Ignore any on-disk cache and re-fetch everything from Yahoo
        # Finance fresh -- for when you suspect the cached bars are
        # stale or wrong, rather than the normal "reuse what we have"
        # path every other backtest run takes.
        self.refresh_cache = refresh_cache
        # Optional (stage, current, total) callback -- lets a caller (e.g.
        # the web dashboard, running this in a background thread) surface
        # progress without coupling this module to Flask/threading at all.
        self._on_progress = on_progress or (lambda stage, current, total: None)

        # Profile-scoped, exactly like TradingEngine does it -- otherwise a
        # backtest would replay the shared fallback block that no profile
        # actually runs verbatim, silently testing the wrong strategy with
        # the wrong parameters. risk/regime/universe stay global, matching
        # live trading.
        self.strategy_cfg = cfg.get_profile_strategy_config(self.profile_name)
        self.risk_cfg = cfg.get("risk", default={})
        self.regime_cfg = cfg.get("regime", default={})
        # Bounds every yfinance .history() call below -- without it, a
        # stalled connection on any one symbol hangs the whole fetch loop
        # indefinitely with no exception raised (the surrounding try/except
        # only helps once something actually raises).
        self._fetch_timeout = cfg.get("data_source", "request_timeout_seconds", default=15)
        self.universe = load_universe(cfg.universe_file)

        # Make sure the fetch window is wide enough that every lookback this
        # profile needs is already satisfied on the FIRST simulated day --
        # otherwise the opening stretch of the backtest silently evaluates
        # nothing (every _moving_average/_momentum_return_pct returns None)
        # and the run quietly understates how often the strategy would have
        # traded. Requirements are in TRADING days; the fetch window is in
        # calendar days, so they have to be converted, not compared directly.
        self.lookback_buffer_days = max(
            self.lookback_buffer_days,
            _calendar_days_for(_required_trading_days(self.strategy_cfg)),
        )

        self.risk = RiskManager(
            max_open_positions=cfg.get("risk", "max_open_positions", default=10),
            position_size_pct_of_equity=cfg.get("risk", "position_size_pct_of_equity", default=8.0),
            max_cash_deployed_per_scan_pct=cfg.get("risk", "max_cash_deployed_per_scan_pct", default=40.0),
        )
        self.starting_capital = cfg.get_profile_starting_capital(self.profile_name)

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
        # Simulated-day of each symbol's last sell, for the re-entry
        # cooldown below. Trade.timestamp itself always stores real
        # wall-clock time (see PaperBroker.sell), not the simulated day, so
        # it can't be used for this -- tracked separately here instead.
        self._sold_on: dict[str, pd.Timestamp] = {}

        # Entry diagnostics, accumulated across the whole simulation and
        # reported on BacktestResult (see the fields there).
        self._days_regime_blocked = 0
        self._days_portfolio_full = 0
        self._days_with_candidates = 0
        self._entry_rejections: dict[str, int] = {}

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
        cache_hits = 0
        for i, symbol in enumerate(self.universe):
            cached = None if self.refresh_cache else price_cache.load(symbol)
            if price_cache.covers(cached, fetch_start, fetch_end):
                self._history[symbol] = price_cache.slice_range(cached, fetch_start, fetch_end)
                cache_hits += 1
            else:
                elapsed = _time.time() - last_call
                if elapsed < self._YFINANCE_MIN_INTERVAL_SECONDS:
                    _time.sleep(self._YFINANCE_MIN_INTERVAL_SECONDS - elapsed)
                last_call = _time.time()
                try:
                    df = yf.Ticker(symbol + ".NS").history(start=fetch_start, end=fetch_end, timeout=self._fetch_timeout)
                    if not df.empty:
                        df.index = df.index.tz_localize(None)
                        merged = price_cache.save(symbol, df, existing=cached)
                        self._history[symbol] = price_cache.slice_range(merged, fetch_start, fetch_end)
                except Exception as exc:  # noqa: BLE001 - one bad symbol must not abort the whole backtest
                    log.debug("Skipping %s: %s", symbol, exc)
            self._on_progress("fetch", i + 1, len(self.universe))
            if (i + 1) % 50 == 0:
                log.info("Fetch progress: %d/%d symbols (%d resolved so far, %d from cache)",
                          i + 1, len(self.universe), len(self._history), cache_hits)

        index_symbol = self.regime_cfg.get("index_symbol", "^NSEI")
        cached_idx = None if self.refresh_cache else price_cache.load(index_symbol)
        if price_cache.covers(cached_idx, fetch_start, fetch_end):
            self._index_history = price_cache.slice_range(cached_idx, fetch_start, fetch_end)
        else:
            try:
                idx = yf.Ticker(index_symbol).history(start=fetch_start, end=fetch_end, timeout=self._fetch_timeout)
                if not idx.empty:
                    idx.index = idx.index.tz_localize(None)
                    merged_idx = price_cache.save(index_symbol, idx, existing=cached_idx)
                    self._index_history = price_cache.slice_range(merged_idx, fetch_start, fetch_end)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not fetch benchmark index %s (%s); regime/relative-strength checks will fail open", index_symbol, exc)

        log.info("Fetched history for %d/%d symbols (%d served from local cache, %d from network).",
                  len(self._history), len(self.universe), cache_hits, len(self._history) - cache_hits)
        missing = [s for s in self.universe if s not in self._history]
        if missing:
            # Loud, and named. Unresolvable tickers are otherwise invisible --
            # yfinance logs each as a stray ERROR line and the universe just
            # quietly shrinks, so a stale alias looks identical to a real
            # delisting and nobody notices the strategy is screening fewer
            # names than intended.
            log.warning(
                "%d/%d universe symbols had no usable price data and were skipped: %s. "
                "Some may be stale tickers rather than delistings -- run "
                "`python main.py validate-universe` to check and repair them.",
                len(missing), len(self.universe), ", ".join(missing),
            )

    # ------------------------------------------------------------------
    def _quote_for(self, symbol: str, history_upto: pd.DataFrame) -> Quote | None:
        if len(history_upto) < 2:
            return None
        last = history_upto.iloc[-1]
        prev = history_upto.iloc[-2]
        # A symbol that gets suspended/delisted partway through the backtest
        # window can have yfinance keep returning rows for it with NaN
        # OHLC instead of just stopping. NaN compares False against
        # everything (`nan <= x` and `nan > x` are both False), so it
        # silently slips past every threshold check in evaluate_candidate/
        # check_exit instead of being rejected -- and once it reaches
        # mark-to-market or a trade price, NaN poisons the running
        # cash/equity total for every day afterwards. Treat it the same as
        # "no data today" (matches DataUnavailableError handling in live
        # trading) rather than letting it through as a real quote.
        if pd.isna(last["Close"]) or pd.isna(prev["Close"]):
            return None
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

    def _benchmark_buy_and_hold(self) -> tuple[float | None, float | None]:
        """Plain buy-and-hold return/CAGR of the benchmark index over
        [start, end] -- the baseline any active strategy needs to beat to
        be worth trading at all instead of just holding the index. None,
        None if the index history is unavailable or too short (never a
        fabricated 0.0, which would read as a real flat return)."""
        if self._index_history is None:
            return None, None
        window = self._index_history.loc[self.start:self.end]
        if len(window) < 2:
            return None, None
        start_price = float(window["Close"].iloc[0])
        end_price = float(window["Close"].iloc[-1])
        if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0:
            return None, None
        total_return_pct = (end_price - start_price) / start_price * 100.0
        elapsed_days = (window.index[-1] - window.index[0]).days
        cagr = cagr_pct(start_price, end_price, elapsed_days) if elapsed_days > 0 else None
        return total_return_pct, cagr

    def _market_regime_ok(self, index_upto: pd.DataFrame | None) -> bool:
        if not self.regime_cfg.get("enabled", False):
            return True
        if index_upto is None or len(index_upto) < 2:
            return True  # fail open, matching live behavior when index data is unavailable
        return is_market_in_uptrend(index_upto, self.regime_cfg.get("ma_days", 200))

    # ------------------------------------------------------------------
    def _run_exits(self, day: pd.Timestamp, trade_pnls: list[float], trade_log: list[dict]) -> None:
        mode = self.strategy_cfg.get("mode", "52w_high")
        for symbol, pos in list(self.broker.positions().items()):
            hist = self._history.get(symbol)
            if hist is None:
                continue
            history_upto = hist.loc[:day]
            quote = self._quote_for(symbol, history_upto)
            if quote is None:
                continue

            self.broker.update_trailing_high(symbol, quote.ltp)
            cfg = {**self.risk_cfg, **self.strategy_cfg}
            if mode == "consolidation_breakout":
                should_exit, reason = consolidation_breakout.check_exit(pos, quote, history_upto, cfg)
            else:
                should_exit, reason = exit_52w(pos, quote, history_upto, cfg)
            if not should_exit:
                continue

            pre_qty, pre_avg = pos.quantity, pos.avg_price
            try:
                trade = self.broker.sell(symbol, pre_qty, quote.ltp, reason)
            except ValueError:
                continue
            self._sold_on[symbol] = day
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
            self._days_portfolio_full += 1
            return

        index_upto = self._index_history.loc[:day] if self._index_history is not None else None
        if not self._market_regime_ok(index_upto):
            self._days_regime_blocked += 1
            return

        mode = self.strategy_cfg.get("mode", "52w_high")
        cooldown_days = self.risk_cfg.get("reentry_cooldown_days", 3)
        cooldown_blocked = (
            {s for s, sold_day in self._sold_on.items() if (day - sold_day).days < cooldown_days}
            if cooldown_days else set()
        )

        if mode == "cross_sectional_momentum":
            universe_data = []
            for symbol in self.universe:
                if symbol in positions or symbol in cooldown_blocked:
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
            ranked = select_cross_sectional_candidates(universe_data, self.strategy_cfg,
                                                       reasons=self._entry_rejections)
        elif mode == "consolidation_breakout":
            candidates = []
            for symbol in self.universe:
                if symbol in positions or symbol in cooldown_blocked:
                    continue
                hist = self._history.get(symbol)
                if hist is None:
                    continue
                history_upto = hist.loc[:day]
                quote = self._quote_for(symbol, history_upto)
                if quote is None:
                    continue
                turnover = _avg_daily_turnover(history_upto)
                cand = consolidation_breakout.evaluate_candidate(
                    symbol, quote, history_upto, turnover, self.strategy_cfg,
                    index_history=index_upto, reasons=self._entry_rejections,
                )
                if cand:
                    candidates.append(cand)
            ranked = consolidation_breakout.rank_candidates(candidates)
        else:
            candidates: list[Candidate52w] = []
            for symbol in self.universe:
                if symbol in positions or symbol in cooldown_blocked:
                    continue
                hist = self._history.get(symbol)
                if hist is None:
                    continue
                history_upto = hist.loc[:day]
                quote = self._quote_for(symbol, history_upto)
                if quote is None:
                    continue
                turnover = _avg_daily_turnover(history_upto)
                cand = eval_52w(symbol, quote, history_upto, turnover, self.strategy_cfg,
                                index_history=index_upto, reasons=self._entry_rejections)
                if cand:
                    candidates.append(cand)
            ranked = rank_52w(candidates)

        if ranked:
            self._days_with_candidates += 1

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
                close = float(history_upto.iloc[-1]["Close"])
                # A NaN close (symbol suspended/delisted mid-window -- see
                # the comment in _quote_for) must never enter this dict:
                # `quotes.get(s, p.avg_price)` only falls back to avg_price
                # when the key is *missing*, not when its value is NaN, and
                # a single NaN here would permanently poison every equity
                # value (and hence every backtest day) from this point on.
                if not pd.isna(close):
                    quotes[symbol] = close
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
            equity_today = self._mark_to_market(day)
            if pd.isna(equity_today):
                # Fail fast with the exact day/holdings instead of quietly
                # finishing all remaining days with a poisoned NaN total --
                # every subsequent equity value would be NaN too once this
                # happens, so there is nothing useful left to compute.
                cash = self.broker.cash()
                positions = self.broker.positions()
                detail = ", ".join(
                    f"{sym}: qty={pos.quantity} avg_price={pos.avg_price}"
                    for sym, pos in positions.items()
                ) or "none"
                raise RuntimeError(
                    f"Equity became NaN/undefined on {day.date()}. Cash={cash!r}. "
                    f"Open positions: {detail}. This means a trade or mark-to-market "
                    f"picked up bad price data (e.g. a symbol suspended/delisted or "
                    f"missing a data point mid-window) that wasn't caught -- "
                    f"please report this with the date and symbols above."
                )
            equity_curve.append((str(day.date()), equity_today))
            self._on_progress("simulate", i + 1, len(trading_days))
            if (i + 1) % 50 == 0:
                log.info("Backtest progress: %d/%d trading days simulated", i + 1, len(trading_days))

        ending_equity = equity_curve[-1][1] if equity_curve else self.starting_capital
        equity_values = [v for _, v in equity_curve]
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        elapsed_days = (self.end - self.start).days
        benchmark_return_pct, benchmark_cagr = self._benchmark_buy_and_hold()

        log.info(
            "Backtest entry diagnostics [%s]: %d/%d days blocked by market regime, "
            "%d days with the portfolio already full, %d days with at least one candidate. "
            "Rejections: %s",
            self.profile_name, self._days_regime_blocked, len(trading_days),
            self._days_portfolio_full, self._days_with_candidates,
            ", ".join(f"{k}={v}" for k, v in sorted(self._entry_rejections.items(), key=lambda kv: -kv[1]))
            or "none",
        )

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
            benchmark_symbol=self.regime_cfg.get("index_symbol", "^NSEI"),
            benchmark_total_return_pct=benchmark_return_pct,
            benchmark_cagr_pct=benchmark_cagr,
            equity_curve=equity_curve,
            trade_log=trade_log,
            profile=self.profile_name,
            days_regime_blocked=self._days_regime_blocked,
            days_portfolio_full=self._days_portfolio_full,
            days_with_candidates=self._days_with_candidates,
            entry_rejections=dict(sorted(self._entry_rejections.items(), key=lambda kv: -kv[1])),
        )
