"""Phase CS — Cross-Sectional IC for multi-instrument universes.

For each time bar t, ranks all instruments by a chosen signal value,
then ranks by forward return, and computes Spearman(signal_ranks, return_ranks).

This is the cross-sectional equivalent of the time-series IC in Phase 1.
It answers: "At each point in time, does the signal correctly sort instruments
by next-period return?"

Usage:
    uv run python research/ic_analysis/cross_sectional_ic.py \\
        --instruments SPY QQQ AAPL MSFT AMZN \\
        --signal accel_rsi14 \\
        --timeframe D \\
        --horizon 5
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.ic_analysis.phase1_sweep import (  # noqa: E402
    _get_annual_bars,
    _load_ohlcv,
    build_all_signals,
)
from research.ic_analysis.run_ic import compute_forward_returns  # noqa: E402

logger = logging.getLogger(__name__)

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_cross_sectional_ic(
    instruments: list[str],
    signal_name: str,
    timeframe: str,
    horizon: int = 5,
    data_dir: Path | None = None,
    fmt: str = "parquet",
) -> pd.DataFrame:
    """Compute cross-sectional IC at each bar for a given signal.

    Returns DataFrame with columns: date, cs_ic, n_instruments.
    """
    # Load data and compute signal + fwd return for each instrument.
    all_data: dict[str, pd.DataFrame] = {}
    for inst in instruments:
        try:
            df = _load_ohlcv(inst, timeframe, data_dir=data_dir, fmt=fmt)
            window_1y = _get_annual_bars(timeframe)
            signals = build_all_signals(df, window_1y)
            if signal_name not in signals.columns:
                logger.warning("Signal '%s' not found for %s — skipped.", signal_name, inst)
                continue
            fwd = compute_forward_returns(df["close"], [horizon])
            fwd_col = f"fwd_{horizon}"
            combo = pd.DataFrame(
                {
                    "signal": signals[signal_name],
                    "fwd": fwd[fwd_col] if fwd_col in fwd.columns else np.nan,
                }
            )
            all_data[inst] = combo.dropna()
        except Exception as exc:
            logger.warning("Could not load %s: %s", inst, exc)
            continue

    if len(all_data) < 3:
        raise ValueError(f"Need ≥3 instruments for cross-sectional IC, got {len(all_data)}")

    # Build signal and fwd matrices: rows=dates, cols=instruments.
    sig_frames = {inst: d["signal"] for inst, d in all_data.items()}
    fwd_frames = {inst: d["fwd"] for inst, d in all_data.items()}

    sig_panel = pd.DataFrame(sig_frames)
    fwd_panel = pd.DataFrame(fwd_frames)

    # Only keep dates where ≥3 instruments have data.
    valid_mask = sig_panel.notna().sum(axis=1) >= 3
    sig_panel = sig_panel[valid_mask]
    fwd_panel = fwd_panel.reindex(sig_panel.index)

    results: list[dict] = []
    for dt in sig_panel.index:
        sig_row = sig_panel.loc[dt].dropna()
        fwd_row = fwd_panel.loc[dt].dropna()
        common = sig_row.index.intersection(fwd_row.index)
        if len(common) < 3:
            continue
        rho, _ = spearmanr(sig_row[common].values, fwd_row[common].values)
        results.append({"date": dt, "cs_ic": float(rho), "n_instruments": len(common)})

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-Sectional IC for multi-instrument universe")
    parser.add_argument(
        "--instruments",
        nargs="+",
        required=True,
        help="List of instrument names, e.g. SPY QQQ AAPL MSFT",
    )
    parser.add_argument("--signal", required=True, help="Signal name, e.g. accel_rsi14")
    parser.add_argument("--timeframe", default="D", help="Timeframe (D, H4, etc.)")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon in bars")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    parser.add_argument("--fmt", default="parquet", choices=["parquet", "csv"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None

    print(f"\nCross-Sectional IC: {args.signal} | {args.timeframe} | h={args.horizon}")
    print(f"Instruments: {args.instruments}")
    print()

    cs_df = compute_cross_sectional_ic(
        instruments=args.instruments,
        signal_name=args.signal,
        timeframe=args.timeframe,
        horizon=args.horizon,
        data_dir=data_dir,
        fmt=args.fmt,
    )

    if cs_df.empty:
        print("  No valid cross-sectional IC data. Check instrument availability.")
        return

    mean_ic = cs_df["cs_ic"].mean()
    std_ic = cs_df["cs_ic"].std()
    icir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
    n_bars = len(cs_df)

    print(f"  Bars with ≥3 instruments : {n_bars}")
    print(f"  Mean CS-IC              : {mean_ic:+.4f}")
    print(f"  Std  CS-IC              : {std_ic:.4f}")
    print(f"  CS-ICIR                 : {icir:+.3f}")
    print(f"  Hit rate (IC > 0)       : {(cs_df['cs_ic'] > 0).mean():.1%}")
    print()

    out_path = REPORTS_DIR / f"cs_ic_{args.signal}_{args.timeframe}_h{args.horizon}.csv"
    cs_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
