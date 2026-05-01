#!/usr/bin/env bash
# Refresh warmup parquets for the champion portfolio.
#
# Designed to be invoked from inside the titan-portfolio container, so it
# can reach the IB Gateway via the compose DNS name `ib-gateway`. Uses the
# existing download scripts:
#   * AUD_JPY (H1+H4+D+W) -> download_data_mtf.py (IBKR, merge-safe)
#   * CSPX_D, IHYU_D      -> download_data_yfinance.py (Yahoo, overwrites
#                            but always pulls 10y+ history so the file
#                            ends up with the full series)
#
# Usage:
#   On host:        docker compose exec -T titan-portfolio bash scripts/refresh_market_data.sh
#   In container:   bash scripts/refresh_market_data.sh
#   With a switch:  REFRESH_SOURCE_ETF=databento bash scripts/refresh_market_data.sh
#
# Returns non-zero if ANY step fails, so systemd's Restart=on-failure can
# act on it. Per-step failures still let later steps run (best effort).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

START_EPOCH=$(date +%s)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "============================================================"
echo "  market-data refresh @ ${TIMESTAMP}"
echo "============================================================"

ANY_FAILED=0
FAILED_STEPS=()

# ── Step 1: AUD/JPY all timeframes via IBKR ──────────────────────────────
# Use a high client_id (80) to dodge any half-open connection on lower IDs.
# IBKR_CLIENT_ID_DOWNLOAD is read by download_data_mtf.py.
echo ""
echo "[1/2] Refreshing AUD_JPY at H1, H4, D, W (IBKR via ib-gateway)..."
if IBKR_CLIENT_ID_DOWNLOAD="${IBKR_CLIENT_ID_DOWNLOAD:-80}" \
        uv run python scripts/download_data_mtf.py --pair AUD_JPY --years 15; then
    echo "  ✓ AUD_JPY refresh complete"
else
    echo "  ✗ AUD_JPY refresh FAILED" >&2
    ANY_FAILED=1
    FAILED_STEPS+=("AUD_JPY")
fi

# ── Step 2: CSPX_D, IHYU_D, IHYG_D via Yahoo (or Databento if requested) ─
echo ""
SOURCE="${REFRESH_SOURCE_ETF:-yfinance}"
echo "[2/2] Refreshing CSPX_D, IHYU_D, IHYG_D via ${SOURCE}..."

if [ "$SOURCE" = "databento" ]; then
    if [ -z "${DATABENTO_API_KEY:-}" ]; then
        echo "  ✗ DATABENTO_API_KEY not set; falling back to yfinance" >&2
        SOURCE="yfinance"
    fi
fi

case "$SOURCE" in
    yfinance)
        # CSPX (S&P UCITS, iShares, USD), VUSD (S&P UCITS, Vanguard, USD line),
        # IHYU ($-HY UCITS), IHYG (€-HY UCITS). All LSE-listed; Yahoo
        # serves them under the .L suffix. Use LOCAL=YAHOO mapping so
        # parquets land with clean names. CSPX and VUSD both track S&P
        # 500 but trade as separate broker symbols (avoids strategy
        # position-attribution conflict between bond_equity_ihyu_cspx
        # and bond_equity_ihyg_vusd). VUSD (not VUSA) is used because
        # IBKR's USD-denominated Vanguard line on LSEETF is VUSD; VUSA
        # is the GBP line.
        if uv run python scripts/download_data_yfinance.py \
                --symbols CSPX=CSPX.L VUSD=VUSD.L IHYU=IHYU.L IHYG=IHYG.L \
                --interval D \
                --start 2015-01-01; then
            echo "  ✓ CSPX/VUSD/IHYU/IHYG refresh complete (yfinance)"
        else
            echo "  ✗ CSPX/VUSD/IHYU/IHYG refresh FAILED (yfinance)" >&2
            ANY_FAILED=1
            FAILED_STEPS+=("CSPX/VUSD/IHYU/IHYG")
        fi
        ;;
    databento)
        if uv run python scripts/download_data_databento.py \
                --symbols CSPX VUSD IHYU IHYG \
                --start 2018-05-01; then
            echo "  ✓ CSPX/VUSD/IHYU/IHYG refresh complete (databento)"
        else
            echo "  ✗ CSPX/VUSD/IHYU/IHYG refresh FAILED (databento)" >&2
            ANY_FAILED=1
            FAILED_STEPS+=("CSPX/VUSD/IHYU/IHYG")
        fi
        ;;
    *)
        echo "  ✗ Unknown REFRESH_SOURCE_ETF='$SOURCE' (use yfinance|databento)" >&2
        ANY_FAILED=1
        FAILED_STEPS+=("CSPX/IHYU (bad source)")
        ;;
esac

# ── Summary ──────────────────────────────────────────────────────────────
END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
echo ""
echo "============================================================"
echo "  refresh finished in ${ELAPSED}s"
if [ "$ANY_FAILED" -eq 0 ]; then
    echo "  status: ALL OK"
else
    echo "  status: PARTIAL FAILURE (${FAILED_STEPS[*]})" >&2
fi
echo "============================================================"

# Show last-modified time of the key files for at-a-glance freshness
if command -v stat >/dev/null 2>&1; then
    echo ""
    echo "Parquet timestamps (UTC):"
    for f in data/AUD_JPY_H1.parquet data/AUD_JPY_D.parquet \
             data/CSPX_D.parquet data/VUSD_D.parquet \
             data/IHYU_D.parquet data/IHYG_D.parquet; do
        if [ -f "$f" ]; then
            mtime=$(stat -c "%y" "$f" 2>/dev/null || stat -f "%Sm" "$f" 2>/dev/null)
            size=$(du -h "$f" 2>/dev/null | cut -f1)
            printf "  %-30s  %s  (%s)\n" "$f" "$size" "$mtime"
        else
            printf "  %-30s  MISSING\n" "$f"
        fi
    done
fi

exit "$ANY_FAILED"
