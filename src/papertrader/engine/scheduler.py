"""The autonomous trading loop.

During NSE market hours it periodically:
  1. Checks open positions for stop-loss / trailing-stop / momentum-breakdown
     exits and sells (paper) any that trigger.
  2. Scans the configured universe for new 52-week-high momentum
     candidates and buys (paper) the top-ranked ones, sized by the risk
     manager, subject to max open positions and per-scan cash limits.
Outside market hours it sleeps until the next open.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta

from ..config import Config
from ..data.nse_client import MarketDataClient, DataUnavailableError
from ..data.universe import load_universe
from ..portfolio.broker import PaperBroker, InsufficientFundsError
from ..portfolio.storage import Storage
from ..risk.risk_manager import RiskManager
from ..strategy.cross_sectional_momentum import select_cross_sectional_candidates
from ..strategy.momentum_52w_high import Candidate as Candidate52w, evaluate_candidate as eval_52w, rank_candidates as rank_52w, check_exit as exit_52w, is_market_in_uptrend
from ..strategy import consolidation_breakout, pivot_supertrend, trend_pullback
from .market_hours import MarketCalendar

log = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, cfg: Config, profile_name: str | None = None):
        self.cfg = cfg
        self.profile_name = profile_name or cfg.get("active_profile", default="52w_high")
        # Guards storage/broker/strategy_cfg against a profile switch
        # (reload_profile(), called from a Flask request thread) racing
        # with this engine's own run_forever() loop reading/writing them
        # from a background thread at the same time.
        self._state_lock = threading.Lock()
        # Snapshot of the most recent scan_for_entries()/find_candidates()
        # call, read by the dashboard's Insights panel so "why isn't this
        # profile buying anything" is answerable at a glance instead of
        # only by grepping data/papertrader.log. In-memory only -- it
        # describes "as of the last scan this process ran", which is
        # meaningless to carry across a restart; the next scan after one
        # repopulates it within a cycle.
        #
        # Deliberately guarded by its OWN lock, never _state_lock:
        # run_forever() holds _state_lock for the whole
        # check_exits()+mark_to_market()+scan_for_entries() block, and
        # _record_scan_diagnostics() is called from inside that same call
        # chain on that same thread. threading.Lock is not reentrant, so
        # reusing _state_lock here would make the very first scan after
        # every restart deadlock immediately and permanently on that
        # thread -- and separately, every dashboard page load's
        # get_scan_diagnostics() would then block for the full duration of
        # whatever scan happens to be in progress, which is exactly the
        # blocking behavior every build_* function in data_api.py is
        # written to avoid. A dedicated lock only ever needs to be held for
        # the instant it takes to copy a small dict, so contention here is
        # never a real concern the way it is for _state_lock.
        self._scan_diagnostics_lock = threading.Lock()
        self._last_scan_diagnostics: dict | None = None
        self.calendar = MarketCalendar(
            timezone=cfg.get("engine", "timezone", default="Asia/Kolkata"),
            market_open=cfg.get("engine", "market_open", default="09:15"),
            market_close=cfg.get("engine", "market_close", default="15:30"),
            holidays_file=cfg.holidays_file,
        )
        # Use profile-specific starting capital and ledger path -- must be
        # resolved against self.profile_name explicitly (not cfg.state_file,
        # which depends on the config's single global active_profile and
        # would give every engine the same ledger in multi_profile_mode).
        starting_capital = cfg.get_profile_starting_capital(self.profile_name)
        self.storage = Storage(cfg.get_profile_state_file(self.profile_name), starting_capital)
        self.broker = PaperBroker(
            self.storage,
            slippage_bps=cfg.get("execution", "slippage_bps", default=5.0),
            flat_charges_inr=cfg.get("execution", "flat_charges_inr", default=20.0),
        )
        self.risk = RiskManager(
            max_open_positions=cfg.get("risk", "max_open_positions", default=10),
            position_size_pct_of_equity=cfg.get("risk", "position_size_pct_of_equity", default=8.0),
            max_cash_deployed_per_scan_pct=cfg.get("risk", "max_cash_deployed_per_scan_pct", default=40.0),
        )
        self.data = MarketDataClient(
            preferred=cfg.get("data_source", "preferred", default="nse"),
            fallback=cfg.get("data_source", "fallback", default="yfinance"),
            timeout=cfg.get("data_source", "request_timeout_seconds", default=10),
        )
        self.universe = load_universe(cfg.universe_file)
        # Profile-specific strategy parameters (mode + any overrides that
        # profile's own strategy: block sets) merged over the shared
        # base. Always a fresh dict -- safe even with several
        # TradingEngine instances sharing this same Config object in
        # multi_profile_mode.
        self.strategy_cfg = cfg.get_profile_strategy_config(self.profile_name)
        self.risk_cfg = cfg.get("risk", default={})
        self.regime_cfg = cfg.get("regime", default={})

    def reload_profile(self) -> str:
        """Reload engine configuration from disk after profile has changed.
        Called when user switches profiles via API. Returns new profile name."""
        try:
            # Reload config from disk
            log.info("Reloading configuration from disk...")
            reloaded = self.cfg.reload()
            log.info("Config reload result: %s", reloaded)

            # Get updated active profile
            new_profile = self.cfg.get("active_profile", default="52w_high")
            log.info("Active profile from config: %s (current profile_name: %s)", new_profile, self.profile_name)

            if new_profile == self.profile_name:
                log.info("No profile change needed (already on %s)", new_profile)
                return self.profile_name

            # Profile changed - reinitialize with new profile. Build
            # everything first, then swap it all in under the lock so
            # run_forever() never sees a half-updated engine.
            log.info("Profile changed to: %s, reinitializing engine...", new_profile)

            starting_capital = self.cfg.get_profile_starting_capital(new_profile)
            state_file = self.cfg.get_profile_state_file(new_profile)
            log.info("Loading profile %s: state_file=%s, starting_capital=%.0f",
                     new_profile, state_file, starting_capital)

            new_storage = Storage(state_file, starting_capital)
            new_broker = PaperBroker(
                new_storage,
                slippage_bps=self.cfg.get("execution", "slippage_bps", default=5.0),
                flat_charges_inr=self.cfg.get("execution", "flat_charges_inr", default=20.0),
            )

            new_strategy_cfg = self.cfg.get_profile_strategy_config(new_profile)
            new_risk_cfg = self.cfg.get("risk", default={})
            new_regime_cfg = self.cfg.get("regime", default={})

            with self._state_lock:
                self.profile_name = new_profile
                self.storage = new_storage
                self.broker = new_broker
                self.strategy_cfg = new_strategy_cfg
                self.risk_cfg = new_risk_cfg
                self.regime_cfg = new_regime_cfg

            log.info("✓ Engine reloaded successfully for profile: %s (strategy: %s, ledger: %s)",
                     new_profile, self.strategy_cfg.get("mode"), state_file)
            return new_profile
        except Exception as e:
            log.error("✗ Failed to reload profile: %s", e)
            raise

    # ------------------------------------------------------------------
    def market_regime_ok(self) -> bool:
        """True if new entries are allowed under the market regime filter
        (or the filter is disabled / its data is unavailable, in which
        case we fail open rather than freezing the whole system)."""
        if not self.regime_cfg.get("enabled", False):
            return True
        index_symbol = self.regime_cfg.get("index_symbol", "^NSEI")
        try:
            index_history = self.data.get_index_history(index_symbol)
        except DataUnavailableError as exc:
            log.warning("Could not evaluate market regime (%s); proceeding without the filter this scan", exc)
            return True
        return is_market_in_uptrend(index_history, self.regime_cfg.get("ma_days", 200))

    def check_exits(self) -> None:
        positions = self.broker.positions()
        mode = self.strategy_cfg.get("mode", "52w_high")
        for symbol, pos in positions.items():
            try:
                quote = self.data.get_quote(symbol)
                history = self.data.get_history(symbol, period="1y")
            except DataUnavailableError as exc:
                log.warning("Skipping exit-check for %s: %s", symbol, exc)
                continue

            self.broker.update_trailing_high(symbol, quote.ltp)
            cfg = {**self.risk_cfg, **self.strategy_cfg}
            if mode == "consolidation_breakout":
                should_exit, reason = consolidation_breakout.check_exit(pos, quote, history, cfg)
            elif mode == "pivot_supertrend":
                should_exit, reason = pivot_supertrend.check_exit(pos, quote, history, cfg)
            elif mode == "trend_pullback":
                should_exit, reason = trend_pullback.check_exit(pos, quote, history, cfg)
            else:
                should_exit, reason = exit_52w(pos, quote, history, cfg)
            if should_exit:
                try:
                    self.broker.sell(symbol, pos.quantity, quote.ltp, reason)
                except ValueError as exc:
                    log.error("Failed to sell %s: %s", symbol, exc)

    def _record_scan_diagnostics(
        self, mode: str, status: str, scanned: int = 0,
        candidates: int = 0, reasons: dict[str, int] | None = None,
    ) -> None:
        """Record a snapshot of the scan that just ran/short-circuited.
        `status` is one of "portfolio_full", "regime_blocked", "scanned",
        or "scan_failed" -- see get_scan_diagnostics()'s docstring for how
        the dashboard uses each."""
        with self._scan_diagnostics_lock:
            self._last_scan_diagnostics = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": mode,
                "status": status,
                "scanned": scanned,
                "candidates": candidates,
                "rejections": dict(sorted((reasons or {}).items(), key=lambda kv: -kv[1])),
            }

    def get_scan_diagnostics(self) -> dict | None:
        """The most recent scan snapshot recorded by _record_scan_diagnostics,
        or None before this engine has completed its first scan. Read by
        data_api.py's _build_insights() to surface "why isn't this profile
        buying anything" -- a barren scan (status="scanned", candidates=0)
        and a regime-blocked one look identical from the outside otherwise,
        and both used to be visible only in the log file."""
        with self._scan_diagnostics_lock:
            return dict(self._last_scan_diagnostics) if self._last_scan_diagnostics else None

    def find_candidates(self, exclude_symbols: set[str] | None = None) -> list:
        """Evaluate the whole universe against the strategy right now and
        return every currently-qualifying candidate, ranked. Read-only --
        places no trades, and does NOT apply the room/regime short-circuits
        scan_for_entries() uses (those exist purely to skip needless
        network calls when we already know nothing can be bought; a
        preview should still show what qualifies even if regime or room
        currently blocks acting on it). Shared by scan_for_entries() below
        and the dashboard's on-demand candidate-preview endpoint.
        """
        exclude_symbols = exclude_symbols if exclude_symbols is not None else set(self.broker.positions())
        mode = self.strategy_cfg.get("mode", "52w_high")

        if mode == "cross_sectional_momentum":
            # Needs a longer lookback (default 252 + 21 skip days) than the
            # "1y" period used elsewhere -- fetch a roomier window just for
            # this mode rather than changing what exit-checks etc. pull.
            universe_data = []
            for symbol in self.universe:
                if symbol in exclude_symbols:
                    continue
                try:
                    quote = self.data.get_quote(symbol)
                    history = self.data.get_history(symbol, period="2y")
                    turnover = self.data.get_avg_daily_turnover(symbol, history=history)
                except DataUnavailableError as exc:
                    log.debug("Skipping %s: %s", symbol, exc)
                    continue
                universe_data.append((symbol, quote, history, turnover))
            reasons: dict[str, int] = {}
            try:
                ranked = select_cross_sectional_candidates(universe_data, self.strategy_cfg,
                                                           reasons=reasons)
                log.info(
                    "cross_sectional_momentum scan: %d symbols evaluated, %d candidates. Rejections: %s",
                    len(universe_data), len(ranked),
                    ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])) or "none",
                )
                self._record_scan_diagnostics(mode, "scanned", scanned=len(universe_data),
                                              candidates=len(ranked), reasons=reasons)
                return ranked
            except Exception as exc:  # noqa: BLE001 - see the consolidation_breakout branch below:
                # a bug or malformed data here must not kill this engine's entire background
                # thread -- fail this one scan and let the next one try again instead.
                log.error("cross_sectional_momentum: scan failed, skipping this cycle: %s", exc)
                self._record_scan_diagnostics(mode, "scan_failed", scanned=len(universe_data))
                return []

        if mode == "consolidation_breakout":
            # Fetch index history for beta calculation if Phase 2 screening is enabled
            index_history = None
            cb_cfg = self.strategy_cfg.get("consolidation_breakout") or {}
            if cb_cfg.get("beta_max"):
                try:
                    index_history = self.data.get_index_history(self.regime_cfg.get("index_symbol", "^NSEI"))
                except DataUnavailableError as exc:
                    log.debug("Could not fetch index history for beta calculation: %s", exc)

            candidates = []
            # Which filter rejected how many symbols this scan. A breakout
            # setup is genuinely rare, so zero candidates is often correct --
            # but "correctly found nothing" and "silently broken" look
            # identical without this, so log the tally either way.
            reasons: dict[str, int] = {}
            scanned = 0
            for symbol in self.universe:
                if symbol in exclude_symbols:
                    continue
                try:
                    quote = self.data.get_quote(symbol)
                    history = self.data.get_history(symbol, period="1y")
                    turnover = self.data.get_avg_daily_turnover(symbol, history=history)
                except DataUnavailableError as exc:
                    log.debug("Skipping %s: %s", symbol, exc)
                    reasons["no_data"] = reasons.get("no_data", 0) + 1
                    continue

                scanned += 1
                # Phase 2: Market cap and index membership would be fetched here
                # For now, pass None and they default to no filter
                try:
                    cand = consolidation_breakout.evaluate_candidate(
                        symbol, quote, history, turnover, self.strategy_cfg,
                        index_history=index_history,
                        market_cap_cr=None,  # TODO: fetch from NSE metadata
                        in_nifty_index=None,  # TODO: check against Nifty 50/Next 50 lists
                        reasons=reasons,
                    )
                except Exception as exc:  # noqa: BLE001 - one symbol's malformed data (e.g. an
                    # unexpected column shape from the data source) must never take down the
                    # whole scan -- let alone the engine's entire background thread, which has
                    # no other safety net if this call is left unguarded.
                    log.warning("consolidation_breakout: skipping %s after evaluation error: %s", symbol, exc)
                    reasons["evaluation_error"] = reasons.get("evaluation_error", 0) + 1
                    continue
                if cand:
                    candidates.append(cand)

            log.info(
                "consolidation_breakout scan: %d symbols evaluated, %d candidates. Rejections: %s",
                scanned, len(candidates),
                ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])) or "none",
            )
            self._record_scan_diagnostics(mode, "scanned", scanned=scanned,
                                          candidates=len(candidates), reasons=reasons)
            return consolidation_breakout.rank_candidates(candidates)

        if mode == "pivot_supertrend":
            candidates = []
            reasons: dict[str, int] = {}
            scanned = 0
            for symbol in self.universe:
                if symbol in exclude_symbols:
                    continue
                try:
                    quote = self.data.get_quote(symbol)
                    history = self.data.get_history(symbol, period="1y")
                    turnover = self.data.get_avg_daily_turnover(symbol, history=history)
                except DataUnavailableError as exc:
                    log.debug("Skipping %s: %s", symbol, exc)
                    reasons["no_data"] = reasons.get("no_data", 0) + 1
                    continue

                scanned += 1
                try:
                    cand = pivot_supertrend.evaluate_candidate(
                        symbol, quote, history, turnover, self.strategy_cfg, reasons=reasons,
                    )
                except Exception as exc:  # noqa: BLE001 - one symbol's malformed data must never
                    # take down the whole scan, let alone this engine's entire background thread.
                    log.warning("pivot_supertrend: skipping %s after evaluation error: %s", symbol, exc)
                    reasons["evaluation_error"] = reasons.get("evaluation_error", 0) + 1
                    continue
                if cand:
                    candidates.append(cand)

            log.info(
                "pivot_supertrend scan: %d symbols evaluated, %d candidates. Rejections: %s",
                scanned, len(candidates),
                ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])) or "none",
            )
            self._record_scan_diagnostics(mode, "scanned", scanned=scanned,
                                          candidates=len(candidates), reasons=reasons)
            return pivot_supertrend.rank_candidates(candidates)

        if mode == "trend_pullback":
            candidates = []
            reasons: dict[str, int] = {}
            scanned = 0
            for symbol in self.universe:
                if symbol in exclude_symbols:
                    continue
                try:
                    quote = self.data.get_quote(symbol)
                    history = self.data.get_history(symbol, period="1y")
                    turnover = self.data.get_avg_daily_turnover(symbol, history=history)
                except DataUnavailableError as exc:
                    log.debug("Skipping %s: %s", symbol, exc)
                    reasons["no_data"] = reasons.get("no_data", 0) + 1
                    continue

                scanned += 1
                try:
                    cand = trend_pullback.evaluate_candidate(
                        symbol, quote, history, turnover, self.strategy_cfg, reasons=reasons,
                    )
                except Exception as exc:  # noqa: BLE001 - one symbol's malformed data must never
                    # take down the whole scan, let alone this engine's entire background thread.
                    log.warning("trend_pullback: skipping %s after evaluation error: %s", symbol, exc)
                    reasons["evaluation_error"] = reasons.get("evaluation_error", 0) + 1
                    continue
                if cand:
                    candidates.append(cand)

            log.info(
                "trend_pullback scan: %d symbols evaluated, %d candidates. Rejections: %s",
                scanned, len(candidates),
                ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])) or "none",
            )
            self._record_scan_diagnostics(mode, "scanned", scanned=scanned,
                                          candidates=len(candidates), reasons=reasons)
            return trend_pullback.rank_candidates(candidates)

        # Default: 52w_high strategy
        # Fetch the benchmark index once per scan (cached) so every
        # candidate's relative strength is judged against the same frame,
        # instead of a per-symbol network round trip.
        index_history = None
        if self.strategy_cfg.get("min_relative_strength_pct") is not None:
            try:
                index_history = self.data.get_index_history(self.regime_cfg.get("index_symbol", "^NSEI"))
            except DataUnavailableError as exc:
                log.warning("Could not fetch index history for relative-strength scoring (%s); skipping that filter this scan", exc)

        candidates = []
        reasons_52w: dict[str, int] = {}
        scanned = 0
        for symbol in self.universe:
            if symbol in exclude_symbols:
                continue
            try:
                quote = self.data.get_quote(symbol)
                history = self.data.get_history(symbol, period="1y")
                turnover = self.data.get_avg_daily_turnover(symbol, history=history)
            except DataUnavailableError as exc:
                log.debug("Skipping %s: %s", symbol, exc)
                reasons_52w["no_data"] = reasons_52w.get("no_data", 0) + 1
                continue

            scanned += 1
            try:
                cand = eval_52w(symbol, quote, history, turnover, self.strategy_cfg,
                                index_history=index_history, reasons=reasons_52w)
            except Exception as exc:  # noqa: BLE001 - one symbol's malformed data must never
                # take down the whole scan, let alone this engine's entire background thread.
                log.warning("52w_high: skipping %s after evaluation error: %s", symbol, exc)
                reasons_52w["evaluation_error"] = reasons_52w.get("evaluation_error", 0) + 1
                continue
            if cand:
                candidates.append(cand)

        log.info(
            "52w_high scan: %d symbols evaluated, %d candidates. Rejections: %s",
            scanned, len(candidates),
            ", ".join(f"{k}={v}" for k, v in sorted(reasons_52w.items(), key=lambda kv: -kv[1])) or "none",
        )
        self._record_scan_diagnostics(mode, "scanned", scanned=scanned,
                                      candidates=len(candidates), reasons=reasons_52w)
        return rank_52w(candidates)

    def _recently_sold_symbols(self) -> set[str]:
        """Symbols sold within risk.reentry_cooldown_days -- blocks a stock
        stopped out on a dip from being immediately rebought this scan or
        the next one, which otherwise looks like the engine buying and
        selling the same names back-to-back. 0/absent disables the cooldown."""
        cooldown_days = self.risk_cfg.get("reentry_cooldown_days", 3)
        if not cooldown_days:
            return set()
        cutoff = (datetime.now() - timedelta(days=cooldown_days)).isoformat(timespec="seconds")
        return self.storage.get_recently_sold_symbols(cutoff)

    def scan_for_entries(self) -> None:
        # Recorded unconditionally, before any early-exit below: this marks
        # "the engine attempted a scan cycle just now" (i.e. it's alive and
        # running), not narrowly "the universe was actually searched" --
        # otherwise a fully-invested account (no room for new positions)
        # would show a permanently stale/"not yet scanned" timestamp on the
        # dashboard even while the engine keeps running normally.
        self.storage.set_last_scan_at(datetime.now().isoformat(timespec="seconds"))
        mode = self.strategy_cfg.get("mode", "52w_high")

        positions = self.broker.positions()
        room = self.risk.room_for_new_positions(len(positions))
        if room <= 0:
            log.info("Max open positions reached (%d); skipping entry scan", self.risk.max_open_positions)
            self._record_scan_diagnostics(mode, "portfolio_full")
            return

        if not self.market_regime_ok():
            log.info(
                "Market regime filter: %s below its %sd average; skipping new entries this scan",
                self.regime_cfg.get("index_symbol", "^NSEI"), self.regime_cfg.get("ma_days", 200),
            )
            self._record_scan_diagnostics(mode, "regime_blocked")
            return

        exclude = set(positions) | self._recently_sold_symbols()
        ranked = self.find_candidates(exclude_symbols=exclude)
        max_new = min(room, self.strategy_cfg.get("max_new_positions_per_scan", 3))
        ranked = ranked[:max_new]

        if not ranked:
            log.info("No qualifying %s candidates this scan.", mode)
            return

        free_cash = self.broker.cash()
        scan_budget = self.risk.scan_cash_budget(free_cash)
        # Approximate total equity once per scan (existing positions valued
        # at their average cost) for consistent position sizing across the
        # candidates picked in this pass.
        equity = self.broker.equity({})
        spent = 0.0
        for cand in ranked:
            qty = self.risk.position_size_shares(equity, cand.ltp)
            if qty <= 0:
                continue
            cost_estimate = qty * cand.ltp
            if spent + cost_estimate > scan_budget:
                log.info("Per-scan cash budget reached; deferring %s to next scan", cand.symbol)
                continue
            if mode == "cross_sectional_momentum":
                cs_cfg = self.strategy_cfg.get("cross_sectional") or {}
                reason = (
                    f"cross_sectional_momentum percentile={cand.score:.0f} "
                    f"{cand.momentum_return_pct:.1f}% {cs_cfg.get('lookback_days', 252)}d return "
                    f"(skip last {cs_cfg.get('skip_recent_days', 21)}d)"
                )
            elif mode == "consolidation_breakout":
                reason = (
                    f"consolidation_breakout score={cand.score:.1f} "
                    f"breakout high ₹{cand.consolidation_high:.2f}, "
                    f"{cand.momentum_return_pct:.1f}% {self.strategy_cfg.get('momentum_lookback_days', 21)}d momentum, "
                    f"volume {cand.breakout_volume/cand.avg_volume:.1f}x baseline"
                )
            else:
                reason = (
                    f"momentum_52w_high score={cand.score:.1f} "
                    f"{cand.pct_from_52w_high:.1f}% off 52w-high, "
                    f"{cand.momentum_return_pct:.1f}% {self.strategy_cfg.get('momentum_lookback_days')}d return"
                )
                if cand.relative_strength_pct is not None:
                    reason += f", RS {cand.relative_strength_pct:+.1f}pp vs index"
                if cand.volume_multiple is not None:
                    reason += f", volume {cand.volume_multiple:.1f}x baseline"
            try:
                self.broker.buy(cand.symbol, qty, cand.ltp, reason=reason)
                spent += cost_estimate
            except InsufficientFundsError as exc:
                log.warning("Insufficient funds for %s: %s", cand.symbol, exc)

    def mark_to_market(self) -> None:
        quotes = {}
        for symbol in self.broker.positions():
            try:
                quotes[symbol] = self.data.get_quote(symbol).ltp
            except DataUnavailableError:
                continue
        positions_value = sum(p.quantity * quotes.get(s, p.avg_price) for s, p in self.broker.positions().items())
        self.storage.record_equity(self.broker.cash(), positions_value)

    def run_once(self) -> None:
        log.info("=== Scan started %s ===", datetime.now().isoformat(timespec="seconds"))
        self.check_exits()
        self.scan_for_entries()
        self.mark_to_market()
        log.info("=== Scan complete. Cash=%.2f, Open positions=%d ===", self.broker.cash(), len(self.broker.positions()))

    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        scan_interval = self.cfg.get("engine", "scan_interval_minutes", default=15) * 60
        exit_interval = self.cfg.get("engine", "exit_check_interval_minutes", default=5) * 60
        last_full_scan = 0.0
        log.info("Autonomous trading engine started. Universe size=%d", len(self.universe))
        while True:
            try:
                # Hot-reload config if it has changed (e.g., via dashboard).
                # Held under the lock alongside reload_profile()'s own swap so
                # the two can't interleave and leave stale values in place.
                if self.cfg.reload():
                    strategy_cfg = self.cfg.get_profile_strategy_config(self.profile_name)
                    with self._state_lock:
                        self.strategy_cfg = strategy_cfg
                        self.risk_cfg = self.cfg.get("risk", default={})
                        self.regime_cfg = self.cfg.get("regime", default={})
                        self.risk.max_open_positions = self.cfg.get("risk", "max_open_positions", default=10)
                        self.risk.position_size_pct_of_equity = self.cfg.get("risk", "position_size_pct_of_equity", default=8.0)
                        self.risk.max_cash_deployed_per_scan_pct = self.cfg.get("risk", "max_cash_deployed_per_scan_pct", default=40.0)
                        self.data.timeout = self.cfg.get("data_source", "request_timeout_seconds", default=10)
                    log.info("Configuration hot-reloaded during run. New strategy/risk settings active.")

                now_dt = self.calendar.now()
                if not self.calendar.is_market_open(now_dt):
                    log.info("Market closed (%s). Sleeping 5 minutes...", now_dt.strftime("%Y-%m-%d %H:%M %Z"))
                    time.sleep(300)
                    continue

                # Held for the whole iteration so a concurrent reload_profile()
                # (from a Flask request thread, single-profile mode only)
                # can't swap self.storage/self.broker/self.strategy_cfg out
                # from under a scan that's already in progress.
                with self._state_lock:
                    self.check_exits()
                    self.mark_to_market()

                    if time.time() - last_full_scan >= scan_interval:
                        self.scan_for_entries()
                        last_full_scan = time.time()

                time.sleep(exit_interval)
            except Exception:
                # This loop runs in its own background thread (one per
                # profile in multi_profile_mode) with nothing else
                # supervising it -- an uncaught exception here would
                # silently kill that profile's engine forever (the thread
                # just ends; nothing restarts it, nothing scans for that
                # profile again until the whole process is restarted),
                # while every other profile keeps trading normally. Log
                # the full traceback and keep the loop alive instead.
                log.exception("Unexpected error in trading loop for profile %s; will retry after a short pause", self.profile_name)
                time.sleep(60)
