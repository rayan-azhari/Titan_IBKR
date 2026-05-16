#!/usr/bin/env bash
# Run the D2b + B4b end-to-end pipeline after IBKR data is downloaded.
#
# Steps:
#   1. Stitch all roots (M1 + M2) -> data/{ROOT}_M[12]_stitched_D.parquet
#   2. Validate stitched vs yfinance (L38 gate)
#   3. Run D2b strict-carry audit
#   4. Run B4b TSMOM-on-stitched audit
#
# Pre-Reg: directives/Pre-Reg D2b B4b IBKR Roll-Stitched Futures 2026-05-15.md
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "  D2b + B4b pipeline"
echo "============================================================"

echo
echo "[Step 1/4] Stitching all commodity roots..."
uv run python scripts/stitch_all_futures.py

echo
echo "[Step 2/4] L38 validation (stitched vs yfinance correlation)..."
uv run python scripts/validate_stitched_vs_yfinance.py --root all || true
# Note: ABORT-status roots will be flagged in stitched_validation.csv but
# the audit harness still proceeds with whatever stitched data exists.

echo
echo "[Step 3/4] D2b strict-carry audit..."
PYTHONIOENCODING=utf-8 uv run python research/futures_carry/run_d2b_audit.py

echo
echo "[Step 4/4] B4b TSMOM audit..."
PYTHONIOENCODING=utf-8 uv run python research/tsmom/run_b4b_audit.py

echo
echo "============================================================"
echo "  Pipeline complete."
echo "  D2b result log: .tmp/reports/d2b_strict_carry/result_log.md"
echo "  B4b result log: .tmp/reports/b4b_tsmom_stitched/result_log.md"
echo "============================================================"
