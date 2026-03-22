"""run_equity_phase2.py -- Phase 2 signal combination for validated equity symbols.

For each of the 6 validated equity daily symbols, runs:
  Phase 1 : Full 52-signal IC/ICIR sweep (daily bars) -- saved to .tmp/reports/
  Phase 2 : Signal combination on USABLE signals at h=5,10,20

Computes rsi_21_dev IC at each target horizon directly (apples-to-apples vs
the combination composite at the same horizon) and reports whether any
composite meaningfully improves on the baseline signal.

Usage:
    uv run python research/ic_analysis/run_equity_phase2.py
    uv run python research/ic_analysis/run_equity_phase2.py --symbol NOC
    uv run python research/ic_analysis/run_equity_phase2.py --horizons 5,10,20
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.ic_analysis.run_signal_combination import run_combination  # noqa: E402
from research.ic_analysis.run_signal_sweep import (  # noqa: E402
    _load_ohlcv,
    build_all_signals,
    run_sweep,
)

from research.ic_analysis.run_ic import (  # noqa: E402
    compute_forward_returns,
    compute_ic_table,
    compute_icir,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

VALIDATED_SYMBOLS = ["HWM", "CSCO", "NOC", "WMT", "ABNB", "GL"]
DEFAULT_HORIZONS = [5, 10, 20]
BASELINE_SIGNAL = "rsi_21_dev"
ICIR_WINDOW = 60
IMPROVEMENT_DELTA = 0.01  # combo must beat baseline by this IC delta


def _baseline_ic_at_horizon(symbol: str, horizon: int) -> tuple[float, float]:
    """Compute rsi_21_dev IC and ICIR at a specific horizon (apples-to-apples)."""
    df = _load_ohlcv(symbol, "D")
    signals = build_all_signals(df)
    fwd = compute_forward_returns(df["close"], [horizon])
    valid = signals.notna().any(axis=1) & fwd.notna().any(axis=1)
    signals = signals[valid]
    fwd = fwd[valid]
    ic_df = compute_ic_table(signals, fwd)
    icir_s = compute_icir(signals, fwd, horizons=[horizon], window=ICIR_WINDOW)
    fwd_col = f"fwd_{horizon}"
    if BASELINE_SIGNAL not in ic_df.index:
        return np.nan, np.nan
    ic = float(ic_df.loc[BASELINE_SIGNAL, fwd_col])
    icir = float(icir_s.get(BASELINE_SIGNAL, np.nan))
    return ic, icir


def _best_combo(phase2_df: pd.DataFrame) -> dict:
    """Return the row with the highest |IC| from Phase 2 results."""
    if phase2_df is None or phase2_df.empty:
        return {}
    df = phase2_df.copy()
    df["abs_ic"] = df["ic"].abs()
    best = df.loc[df["abs_ic"].idxmax()]
    return dict(best)


def run_equity_phase2(
    symbols: list[str] | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    if symbols is None:
        symbols = VALIDATED_SYMBOLS
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    summary_rows: list[dict] = []

    for symbol in symbols:
        # ── Phase 1: full sweep (saves CSV, prints leaderboard) ────────────────
        print(f"\n{'='*72}")
        print(f"  Phase 1 — Signal Sweep: {symbol}")
        print(f"{'='*72}")
        try:
            run_sweep(instrument=symbol, timeframe="D")
        except Exception as e:
            print(f"[WARN] Phase 1 failed for {symbol}: {e}")

        for horizon in horizons:
            # ── Baseline: rsi_21_dev IC at this specific horizon ────────────────
            try:
                baseline_ic, baseline_icir = _baseline_ic_at_horizon(symbol, horizon)
            except Exception as e:
                print(f"[WARN] Baseline IC failed for {symbol} h={horizon}: {e}")
                baseline_ic, baseline_icir = np.nan, np.nan

            # ── Phase 2: combination ────────────────────────────────────────────
            print(f"\n--- Phase 2 Combination: {symbol} h={horizon} ---")
            try:
                p2_df = run_combination(instrument=symbol, timeframe="D", horizon=horizon)
            except Exception as e:
                print(f"[WARN] Phase 2 failed for {symbol} h={horizon}: {e}")
                p2_df = pd.DataFrame()

            best = _best_combo(p2_df)
            combo_ic = float(best.get("ic", np.nan)) if best else np.nan
            combo_icir = float(best.get("icir", np.nan)) if best else np.nan
            combo_method = str(best.get("method", "-")) if best else "-"
            combo_signals = str(best.get("signals_used", "-")) if best else "-"

            if not np.isnan(combo_ic) and not np.isnan(baseline_ic):
                delta = abs(combo_ic) - abs(baseline_ic)
            else:
                delta = np.nan

            improved = (
                not np.isnan(delta)
                and delta >= IMPROVEMENT_DELTA
            )

            summary_rows.append({
                "symbol": symbol,
                "horizon": horizon,
                "baseline_ic": baseline_ic,
                "baseline_icir": baseline_icir,
                "combo_ic": combo_ic,
                "combo_icir": combo_icir,
                "delta_ic": delta,
                "improved": improved,
                "combo_method": combo_method,
                "combo_signals": combo_signals,
            })

    summary_df = pd.DataFrame(summary_rows)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*92}")
    print("  PHASE 2 COMBINATION SUMMARY -- Equity Daily  |  baseline: rsi_21_dev")
    print(f"{'='*92}")
    print(f"  {'Symbol':<8} {'H':>3}  {'Base IC':>9} {'Base ICIR':>10}  "
          f"{'Combo IC':>9} {'Combo ICIR':>10}  {'Delta':>7}  {'Better?':>7}  Method")
    print(f"  {'-'*90}")

    for _, row in summary_df.iterrows():
        flag = "YES" if row["improved"] else "no"
        b_ic = f"{row['baseline_ic']:+.4f}" if not np.isnan(row["baseline_ic"]) else "    n/a"
        b_ir = f"{row['baseline_icir']:+.4f}" if not np.isnan(row["baseline_icir"]) else "    n/a"
        c_ic = f"{row['combo_ic']:+.4f}" if not np.isnan(row["combo_ic"]) else "    n/a"
        c_ir = f"{row['combo_icir']:+.4f}" if not np.isnan(row["combo_icir"]) else "    n/a"
        d_ic = f"{row['delta_ic']:+.4f}" if not np.isnan(row["delta_ic"]) else "    n/a"
        print(f"  {row['symbol']:<8} {int(row['horizon']):>3}  {b_ic:>9} {b_ir:>10}  "
              f"{c_ic:>9} {c_ir:>10}  {d_ic:>7}  {flag:>7}  {row['combo_method']}")

    improved_df = summary_df[summary_df["improved"]]
    print(f"\n  Symbols where combination beats rsi_21_dev by >= {IMPROVEMENT_DELTA} IC:")
    if improved_df.empty:
        print("  None -- rsi_21_dev is already optimal across all symbols and horizons.")
    else:
        for _, row in improved_df.iterrows():
            print(
                f"  {row['symbol']} h={int(row['horizon'])}: "
                f"baseline IC={row['baseline_ic']:+.4f} -> "
                f"combo IC={row['combo_ic']:+.4f} (+{row['delta_ic']:.4f})"
                f"  [{row['combo_method']}]"
            )
            print(f"    Signals: {row['combo_signals']}")

    out_dir = ROOT / ".tmp" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "equity_phase2_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 combination for equity symbols")
    parser.add_argument("--symbol", default=None, help="Single symbol (default: all 6)")
    parser.add_argument("--horizons", default="5,10,20", help="Horizons e.g. 5,10,20")
    args = parser.parse_args()
    run_equity_phase2(
        symbols=[args.symbol] if args.symbol else None,
        horizons=[int(h) for h in args.horizons.split(",")],
    )


if __name__ == "__main__":
    main()
