#!/usr/bin/env bash
# Run all three profiles' backtests and capture full output + diagnostics.
set -u
START="${1:-2025-09-08}"
END="${2:-2026-09-08}"
mkdir -p backtest_out
for p in 52w_high cross_sectional consolidation_breakout; do
  echo "=============================================================="
  echo "PROFILE: $p   ($START -> $END)"
  echo "=============================================================="
  python main.py backtest --start "$START" --end "$END" --profile "$p" \
    --export "backtest_out/${p}_equity.csv" 2>&1 | tee "backtest_out/${p}.txt"
  echo
done
echo "Full output saved under backtest_out/"
