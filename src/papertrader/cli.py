"""Command-line interface for the paper trading system."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading

from tabulate import tabulate

from .config import REPO_ROOT, Config
from .engine.scheduler import TradingEngine

log = logging.getLogger(__name__)


def _force_utf8_stdio() -> None:
    """Make stdout/stderr able to carry non-ASCII, on every platform.

    Every rupee figure this CLI prints uses the U+20B9 sign, and a Windows
    console defaults to a legacy code page (cp1252) that has no mapping for
    it. Printing one there raises UnicodeEncodeError and kills the command
    mid-report -- a backtest could finish a multi-minute run, then crash
    while printing its own first result line. `errors="replace"` is the
    safety net for any console that still can't render a glyph after the
    switch: a placeholder character is a cosmetic problem, a traceback that
    discards a completed run is not.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # a redirected/wrapped stream may not support it
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # already detached, or a stream that refuses
            pass


def _setup_logging(cfg: Config) -> None:
    log_file = cfg.log_file
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    level = getattr(logging, cfg.get("logging", "level", default="INFO"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        # encoding is explicit for the same reason as _force_utf8_stdio: the
        # log file would otherwise be opened in the platform's locale
        # encoding, and a single rupee sign in a log line would raise on
        # Windows.
        handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )


def _resolve_profiles(cfg: Config, explicit_profile: str | None) -> list[str]:
    """Which profile(s) a read-only/one-shot CLI command should cover:
    the one named by --profile if given, every configured profile if
    multi_profile_mode is on (since they're all trading independently
    and none is more "the" account than another), or just the single
    active_profile otherwise -- matching how `serve`/`run` decide which
    engine(s) to actually run."""
    all_profiles = cfg.list_profiles()
    if explicit_profile:
        if explicit_profile not in all_profiles:
            print(f"Unknown profile '{explicit_profile}'. Configured profiles: {list(all_profiles)}", file=sys.stderr)
            sys.exit(1)
        return [explicit_profile]
    if cfg.is_multi_profile_mode() and all_profiles:
        return list(all_profiles)
    return [cfg.get("active_profile", default="52w_high")]


def cmd_run(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    _setup_logging(cfg)

    if cfg.is_multi_profile_mode():
        profiles = list(cfg.list_profiles().keys())
        engines = [TradingEngine(cfg, profile_name=p) for p in profiles]
        log.info("Multi-profile mode: starting %d engines in parallel: %s", len(engines), profiles)
        for eng in engines[1:]:
            threading.Thread(target=eng.run_forever, name=f"trading-engine-{eng.profile_name}", daemon=True).start()
        # Run the first engine's loop in the main thread so Ctrl+C stops the process.
        engines[0].run_forever()
    else:
        engine = TradingEngine(cfg)
        engine.run_forever()


def cmd_once(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    _setup_logging(cfg)
    profiles = _resolve_profiles(cfg, getattr(args, "profile", None))
    for profile_name in profiles:
        if len(profiles) > 1:
            log.info("=== Running once for profile: %s ===", profile_name)
        TradingEngine(cfg, profile_name=profile_name).run_once()


def cmd_portfolio(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    display_names = cfg.list_profiles()
    profiles = _resolve_profiles(cfg, getattr(args, "profile", None))

    for profile_name in profiles:
        engine = TradingEngine(cfg, profile_name=profile_name)
        if len(profiles) > 1:
            print(f"\n=== {display_names.get(profile_name, profile_name)} ({profile_name}) ===")

        positions = engine.broker.positions()
        cash = engine.broker.cash()

        rows = []
        total_mv = 0.0
        total_pnl = 0.0
        for symbol, pos in positions.items():
            try:
                quote = engine.data.get_quote(symbol)
                ltp = quote.ltp
                week52_high = quote.week52_high
            except Exception:
                ltp = pos.avg_price
                week52_high = None
            if week52_high and week52_high > 0:
                from_high = f"-{max(0.0, (week52_high - ltp) / week52_high * 100.0):.1f}%"
            else:
                from_high = "—"
            mv = pos.market_value(ltp)
            pnl = pos.unrealized_pnl(ltp)
            pnl_pct = pos.unrealized_pnl_pct(ltp)
            total_mv += mv
            total_pnl += pnl
            rows.append([symbol, pos.quantity, f"{pos.avg_price:.2f}", f"{ltp:.2f}", from_high, f"{mv:.2f}", f"{pnl:+.2f}", f"{pnl_pct:+.2f}%"])

        print(tabulate(rows, headers=["Symbol", "Qty", "Avg Price", "LTP", "From 52W High", "Mkt Value", "Unrl. P&L", "P&L %"], tablefmt="simple"))
        print()
        print(f"Cash:              ₹{cash:,.2f}")
        print(f"Positions value:   ₹{total_mv:,.2f}")
        print(f"Unrealized P&L:    ₹{total_pnl:+,.2f}")
        print(f"Total equity:      ₹{cash + total_mv:,.2f}")


def cmd_web(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    _setup_logging(cfg)
    from .web.app import run_dashboard

    run_dashboard(cfg, host=args.host, port=args.port)


def cmd_serve(args: argparse.Namespace) -> None:
    """Run the autonomous trading loop and the web dashboard together in
    one process: the engine loop(s) run in background threads, the
    dashboard's Flask server runs (blocking) in the main thread and
    shares the same TradingEngine instance(s), so there's nothing else to
    start separately -- Ctrl+C stops everything.

    In multi_profile_mode, one TradingEngine per configured profile is
    created and each runs its own background thread concurrently, so all
    profiles trade in parallel with fully isolated ledgers. The dashboard's
    profile dropdown then just selects which engine's data to *view* --
    it does not start/stop/restart any engine."""
    cfg = Config.load(args.config)
    _setup_logging(cfg)

    engines: dict[str, TradingEngine] = {}
    if cfg.is_multi_profile_mode():
        for profile_name in cfg.list_profiles():
            eng = TradingEngine(cfg, profile_name=profile_name)
            engines[profile_name] = eng
        log.info("Multi-profile mode: starting %d engines in parallel: %s", len(engines), list(engines))
    else:
        eng = TradingEngine(cfg)
        engines[eng.profile_name] = eng

    for name, eng in engines.items():
        threading.Thread(target=eng.run_forever, name=f"trading-engine-{name}", daemon=True).start()

    from .web.app import create_app

    app = create_app(engines, cfg)
    # use_reloader must stay off: Flask's reloader forks a second process,
    # which would start a second copy of the engine thread(s) too. threaded:
    # /api/candidates can take a while (network calls across the whole
    # universe) -- without it the dev server would serve one request at a
    # time and the rest of the dashboard would appear frozen during a scan.
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    _setup_logging(cfg)
    from .backtest.engine import Backtester

    profile_name = getattr(args, "profile", None)
    if profile_name and profile_name not in cfg.list_profiles():
        print(f"Unknown profile '{profile_name}'. Configured profiles: {list(cfg.list_profiles())}", file=sys.stderr)
        sys.exit(1)

    bt = Backtester(cfg, start=args.start, end=args.end, refresh_cache=args.refresh_cache,
                    profile_name=profile_name)
    cache_note = "ignoring the local price cache, re-fetching everything" if args.refresh_cache else "reusing the local price cache where it already covers this range"
    print(f"Backtesting profile '{bt.profile_name}' (strategy: {bt.strategy_cfg.get('mode')}) "
          f"{args.start} -> {args.end} against {len(bt.universe)} symbols ({cache_note})...")
    result = bt.run()

    print()
    print(f"Backtest: {result.start_date} -> {result.end_date} ({result.trading_days} trading days, "
          f"{result.symbols_with_data}/{len(bt.universe)} symbols had usable data)")
    print(f"Starting capital:    ₹{result.starting_capital:,.2f}")
    print(f"Ending equity:       ₹{result.ending_equity:,.2f}")
    print(f"Total return:        {result.total_return_pct:+.2f}%")
    print(f"CAGR:                {result.cagr_pct:+.2f}%")
    print(f"Max drawdown:        -{result.max_drawdown_pct:.2f}%")
    print(f"Round-trip trades:   {result.num_round_trips}")
    print(f"Win rate:            {result.win_rate_pct:.1f}%")
    print(f"Avg win / avg loss:  ₹{result.avg_win_inr:+,.2f} / ₹{result.avg_loss_inr:+,.2f}")
    if result.benchmark_total_return_pct is not None:
        beat = "beat" if result.total_return_pct > result.benchmark_total_return_pct else "lagged"
        print(f"Benchmark ({result.benchmark_symbol}) buy & hold: {result.benchmark_total_return_pct:+.2f}% "
              f"total return, {result.benchmark_cagr_pct:+.2f}% CAGR -- strategy {beat} it")
    else:
        print(f"Benchmark ({result.benchmark_symbol}) buy & hold: unavailable (could not fetch index history)")

    print()
    print("Why entries did or didn't happen:")
    print(f"  Days blocked by market regime filter: {result.days_regime_blocked}/{result.trading_days}")
    print(f"  Days portfolio was already full:      {result.days_portfolio_full}")
    print(f"  Days with at least one candidate:     {result.days_with_candidates}")
    if result.entry_rejections:
        top = sorted(result.entry_rejections.items(), key=lambda kv: -kv[1])
        print("  Filter rejections (symbol-days):      "
              + ", ".join(f"{k}={v:,}" for k, v in top))
    else:
        print("  Filter rejections (symbol-days):      none recorded")

    if args.trades:
        print("\nTrade log:")
        rows = [[t["date"], t["side"], t["symbol"], t["qty"], f"{t['price']:.2f}",
                 "—" if t["pnl"] is None else f"{t['pnl']:+.2f}", t["reason"]] for t in result.trade_log]
        print(tabulate(rows, headers=["Date", "Side", "Symbol", "Qty", "Price", "P&L", "Reason"], tablefmt="simple"))

    if args.export:
        import csv
        with open(args.export, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "equity"])
            writer.writerows(result.equity_curve)
        print(f"\nEquity curve written to {args.export}")


def cmd_sync_config(args: argparse.Namespace) -> None:
    """Safely merge any settings config.default.yaml has picked up (since
    this config.yaml was created or last synced) into the live file --
    e.g. a new profile added by a `git pull`. Additive only: never
    touches a key the live file already has, customized or not. See
    config.default.yaml's own header comment and
    config_editor.find_missing_keys's docstring for the exact guarantee.
    """
    import yaml as _yaml
    from .config_editor import find_missing_keys, update_config_file

    live_path = args.config or os.path.join(REPO_ROOT, "config.yaml")
    default_path = args.default or os.path.join(os.path.dirname(live_path) or ".", "config.default.yaml")

    if not os.path.exists(default_path):
        print(f"No template found at {default_path} -- nothing to sync against.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(live_path):
        print(f"No live config at {live_path} yet -- run any normal command first "
              f"(e.g. `python main.py once`) to create it from the template, then sync-config "
              f"has nothing to do until the template changes again.", file=sys.stderr)
        sys.exit(1)

    with open(default_path, "r", encoding="utf-8") as fh:
        default = _yaml.safe_load(fh)
    with open(live_path, "r", encoding="utf-8") as fh:
        live = _yaml.safe_load(fh)

    missing = find_missing_keys(default, live)
    if not missing:
        print(f"{live_path} is already up to date with {default_path} -- nothing to add.")
        return

    print(f"{len(missing)} setting(s) in {default_path} are missing from {live_path}:")
    for path, value in missing:
        preview = repr(value)
        if len(preview) > 90:
            preview = preview[:90] + "..."
        print(f"  + {'.'.join(path)} = {preview}")

    if args.dry_run:
        print("\n(--dry-run: nothing written. Re-run without it to apply.)")
        return

    update_config_file(live_path, missing)
    print(f"\nAdded {len(missing)} setting(s) to {live_path} (backed up to {live_path}.bak first). "
          f"Nothing you'd already customized was touched.")


def cmd_validate_universe(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    _setup_logging(cfg)
    from .data.universe import load_universe
    from .data.universe_doctor import apply_report, diagnose

    path = args.file or cfg.universe_file
    symbols = load_universe(path)
    print(f"Checking {len(symbols)} symbols in {path} against Yahoo Finance "
          f"(paced, so this takes a couple of minutes)...")

    def progress(current: int, total: int) -> None:
        if current % 50 == 0 or current == total:
            print(f"  {current}/{total} checked", flush=True)

    report = diagnose(symbols, on_progress=progress)

    print()
    print(f"Resolved:    {len(report.ok)}/{len(symbols)}")
    if report.duplicates:
        print(f"\nStale aliases ({len(report.duplicates)}) -- the live ticker is ALREADY in "
              f"the universe, so these are dead entries, not missing companies:")
        for stale, existing in report.duplicates:
            print(f"  {stale:<14} -> already present as {existing}")
    if report.renamed:
        print(f"\nRenamed ({len(report.renamed)}) -- replacement verified against Yahoo:")
        for old_sym, new_sym in report.renamed:
            print(f"  {old_sym:<14} -> {new_sym}")
    if report.unresolved:
        print(f"\nUnresolved ({len(report.unresolved)}) -- no working replacement known. "
              f"Genuinely delisted/merged, or a ticker this tool doesn't have a mapping for:")
        print("  " + ", ".join(report.unresolved))

    if not report.needs_attention:
        print("\nEvery symbol resolves. Nothing to fix.")
        return

    if not args.fix:
        print("\nRe-run with --fix to apply the alias/rename changes above "
              "(add --drop-unresolved to also remove the unresolved ones).")
        return

    changed = apply_report(path, report, drop_unresolved=args.drop_unresolved)
    print(f"\nRewrote {path}: {changed} rows changed.")
    if report.unresolved and not args.drop_unresolved:
        print("Unresolved symbols were left in place -- removing a symbol on the strength "
              "of one failed probe would quietly shrink the universe on a bad network day. "
              "Use --drop-unresolved once you've confirmed they're really gone.")


def cmd_setup_auth(args: argparse.Namespace) -> None:
    import getpass

    from .web import auth

    if auth.is_configured():
        if not args.force:
            print(f"Authentication is already configured ({auth.AUTH_FILE}).")
            print("This bootstraps the FIRST admin account only -- to add more users "
                  "(admin or not), log in as an admin and use the Admin > Users page.")
            print("Re-run with --force to wipe ALL existing users and start over "
                  "(irreversible -- only do this if you're locked out).")
            return
        print("--force: deleting the existing auth store and every user in it.")
        os.remove(auth.AUTH_FILE)

    print("Bootstrapping the first (admin) dashboard account. This does NOT affect")
    print("trading -- it only gates access to the web dashboard (python main.py web/serve).\n")

    while True:
        username = input("Choose a username: ").strip()
        if username:
            break
        print("Username can't be empty.\n")

    while True:
        password = getpass.getpass("Choose a dashboard password: ")
        if len(password) < 8:
            print("Please use at least 8 characters.\n")
            continue
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords didn't match, try again.\n")
            continue
        break

    auth.bootstrap_admin(username, password)

    print(f"\nSaved to {auth.AUTH_FILE} (never commit this file -- it's already in .gitignore).")
    print(f"\nStart the dashboard (`python main.py web` or `serve`) and log in as '{username}' with")
    print("that password.")
    print("\nOnce logged in, use the Admin > Users page to create additional accounts.")


def cmd_history(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    display_names = cfg.list_profiles()
    profiles = _resolve_profiles(cfg, getattr(args, "profile", None))

    for profile_name in profiles:
        if len(profiles) > 1:
            print(f"\n=== {display_names.get(profile_name, profile_name)} ({profile_name}) ===")
        engine = TradingEngine(cfg, profile_name=profile_name)
        trades = engine.storage.get_trades(limit=args.limit)
        rows = [[t.timestamp, t.side, t.symbol, t.quantity, f"{t.price:.2f}", f"{t.charges:.2f}", t.reason] for t in trades]
        print(tabulate(rows, headers=["Timestamp", "Side", "Symbol", "Qty", "Price", "Charges", "Reason"], tablefmt="simple"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="papertrader", description="Autonomous NSE momentum paper trading simulator")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    profile_help = ("Restrict to one profile (e.g. cross_sectional). Default: every configured "
                     "profile when multi_profile_mode is on, otherwise just the active_profile.")

    once = sub.add_parser("once", help="Run a single scan/trade cycle and exit")
    once.add_argument("--profile", default=None, help=profile_help)
    once.set_defaults(func=cmd_once)

    sub.add_parser("run", help="Run the autonomous trading loop (blocks until killed)").set_defaults(func=cmd_run)

    portfolio = sub.add_parser("portfolio", help="Show current positions and P&L")
    portfolio.add_argument("--profile", default=None, help=profile_help)
    portfolio.set_defaults(func=cmd_portfolio)

    web = sub.add_parser("web", help="Launch the local read-only web dashboard")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.set_defaults(func=cmd_web)

    serve = sub.add_parser("serve", help="Run the autonomous trading loop AND the web dashboard together, in one command")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    hist = sub.add_parser("history", help="Show trade history")
    hist.add_argument("--limit", type=int, default=50)
    hist.add_argument("--profile", default=None, help=profile_help)
    hist.set_defaults(func=cmd_history)

    bt = sub.add_parser("backtest", help="Replay the strategy against historical data instead of trading live")
    bt.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    bt.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    bt.add_argument("--trades", action="store_true", help="Also print the full trade log")
    bt.add_argument("--export", default=None, help="Optional CSV path to save the daily equity curve")
    bt.add_argument("--refresh-cache", action="store_true",
                     help="Ignore data/price_cache/ and re-fetch every symbol's history fresh from Yahoo Finance")
    bt.add_argument("--profile", default=None,
                     help="Which profile's strategy to replay (default: the active_profile). "
                          "Each profile has its own strategy mode and parameters.")
    bt.set_defaults(func=cmd_backtest)

    sc = sub.add_parser("sync-config",
                        help="Merge any new settings from config.default.yaml into your live config.yaml, "
                             "without touching anything you've already customized")
    sc.add_argument("--default", default=None,
                    help="Path to the template (default: config.default.yaml next to your config.yaml)")
    sc.add_argument("--dry-run", action="store_true", help="Show what would be added without writing anything")
    sc.set_defaults(func=cmd_sync_config)

    vu = sub.add_parser("validate-universe",
                        help="Check every universe symbol against Yahoo Finance and repair stale tickers")
    vu.add_argument("--file", default=None, help="Universe CSV to check (default: the configured one)")
    vu.add_argument("--fix", action="store_true", help="Apply the verified alias/rename fixes")
    vu.add_argument("--drop-unresolved", action="store_true",
                    help="With --fix, also remove symbols that have no working replacement")
    vu.set_defaults(func=cmd_validate_universe)

    setup_auth = sub.add_parser("setup-auth", help="Bootstrap the dashboard's first admin login (username + password)")
    setup_auth.add_argument("--force", action="store_true", help="Wipe ALL existing users and start over")
    setup_auth.set_defaults(func=cmd_setup_auth)

    return parser


def main(argv: list[str] | None = None) -> None:
    _force_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
