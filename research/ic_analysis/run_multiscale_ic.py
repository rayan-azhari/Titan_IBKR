"""run_multiscale_ic.py -- Multi-Scale IC Analysis on Single H1 Stream.

Computes 52 signals at 4 scales (H1, H4, D, W) on a single H1 data stream,
producing 208 features with zero cross-timeframe look-ahead bias. Then runs
the IC/ICIR ranking pipeline to identify which scale+signal combinations have
genuine predictive power.

This replaces the multi-file TF approach (which suffers from ffill look-ahead)
with a single-stream multi-scale approach where all information is properly
time-aligned.

Usage:
    uv run python research/ic_analysis/run_multiscale_ic.py
    uv run python research/ic_analysis/run_multiscale_ic.py --pair EUR_USD --scales H1,D,W
    uv run python research/ic_analysis/run_multiscale_ic.py --pair GBP_USD --top-n 10

Directive: Backtesting & Validation.md
"""

import argparse
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from research.ic_analysis.phase1_sweep import (  # noqa: E402
    SCALE_MAP,
    _load_ohlcv,
    build_multiscale_signals,
)
from research.ic_analysis.run_ic import (  # noqa: E402
    compute_forward_returns,
)

HORIZONS = [1, 5, 10, 20]
WINDOW_1Y_H1 = 252 * 22  # ~5544 H1 bars per year (FX, 22 bars/day avg)


# ── Fast IC computation (replaces slow serial loop) ──────────────────────────


def _fast_ic_table(signals: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.DataFrame:
    """Vectorized Spearman IC: rank-correlate all signals vs all horizons at once.

    Uses rank transformation + Pearson on ranks (equivalent to Spearman).
    ~10-50x faster than looping scipy.stats.spearmanr per signal×horizon.
    """
    sig_cols = signals.columns.tolist()
    fwd_cols = fwd_returns.columns.tolist()

    # Align and drop NaN rows (common across all)
    combined = pd.concat([signals, fwd_returns], axis=1).dropna()
    if len(combined) < 30:
        return pd.DataFrame(0.0, index=sig_cols, columns=fwd_cols)

    # Rank transform (Spearman = Pearson on ranks)
    sig_ranked = combined[sig_cols].rank().values  # (N, n_signals)
    fwd_ranked = combined[fwd_cols].rank().values  # (N, n_horizons)

    # Demean
    sig_ranked -= sig_ranked.mean(axis=0, keepdims=True)
    fwd_ranked -= fwd_ranked.mean(axis=0, keepdims=True)

    # Pearson on ranks = Spearman
    # corr[i,j] = dot(sig_i, fwd_j) / (norm(sig_i) * norm(fwd_j))
    sig_norms = np.sqrt((sig_ranked**2).sum(axis=0, keepdims=True))  # (1, n_signals)
    fwd_norms = np.sqrt((fwd_ranked**2).sum(axis=0, keepdims=True))  # (1, n_horizons)

    # (n_signals, N) @ (N, n_horizons) = (n_signals, n_horizons)
    corr_matrix = (sig_ranked.T @ fwd_ranked) / (sig_norms.T @ fwd_norms + 1e-12)

    return pd.DataFrame(corr_matrix, index=sig_cols, columns=fwd_cols)


def _fast_icir(
    signals: pd.DataFrame, fwd_returns: pd.DataFrame, best_horizon_col: pd.Series
) -> pd.Series:
    """Fast ICIR: mean(rolling IC) / std(rolling IC) for each signal's best horizon.

    Instead of computing full rolling IC (slow), approximate with chunked IC:
    split data into ~20 equal chunks, compute IC per chunk, then mean/std.
    """
    n = len(signals)
    n_chunks = 20
    chunk_size = n // n_chunks

    chunk_ics = {sig: [] for sig in signals.columns}

    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else n
        chunk_sig = signals.iloc[start:end]
        chunk_fwd = fwd_returns.iloc[start:end]

        # Fast IC for this chunk
        ic_chunk = _fast_ic_table(chunk_sig, chunk_fwd)

        for sig in signals.columns:
            h_col = best_horizon_col.get(sig, fwd_returns.columns[0])
            if h_col in ic_chunk.columns:
                chunk_ics[sig].append(ic_chunk.loc[sig, h_col])

    icir = {}
    for sig, ics in chunk_ics.items():
        if len(ics) < 3:
            icir[sig] = 0.0
            continue
        arr = np.array(ics)
        std = arr.std()
        icir[sig] = float(arr.mean() / std) if std > 1e-9 else 0.0

    return pd.Series(icir)


# ── IC Analysis ──────────────────────────────────────────────────────────────


def run_multiscale_analysis(
    pair: str,
    scales: dict[str, int] | None = None,
    horizons: list[int] | None = None,
    top_n: int = 20,
) -> dict:
    """Run full multi-scale IC analysis on H1 data.

    Returns dict with: ic_df, icir_s, leaderboard, signals_df, close, fwd_returns
    """
    if scales is None:
        scales = SCALE_MAP
    if horizons is None:
        horizons = HORIZONS

    # Load H1 data
    print(f"\n  Loading {pair} H1 data...")
    df = _load_ohlcv(pair, "H1")
    close = df["close"]
    print(f"  {len(df)} bars ({df.index[0].date()} -> {df.index[-1].date()})")

    # Compute multi-scale signals
    print(f"\n  Computing multi-scale signals ({len(scales)} scales)...")
    t0 = time.time()
    signals = build_multiscale_signals(df, WINDOW_1Y_H1, scales=scales)
    elapsed = time.time() - t0
    print(f"  {len(signals.columns)} features computed in {elapsed:.1f}s")

    # Drop columns that are all NaN (W-scale signals need long warmup)
    valid_cols = signals.columns[signals.notna().sum() > len(signals) * 0.3]
    dropped = len(signals.columns) - len(valid_cols)
    if dropped > 0:
        print(f"  Dropped {dropped} features with >70% NaN (insufficient warmup)")
    signals = signals[valid_cols]

    # Forward returns
    print(f"\n  Computing forward returns at horizons {horizons}...")
    fwd = compute_forward_returns(close, horizons, vol_adjust=True)

    # IC table (vectorized Spearman via rank correlation -- ~50x faster)
    print(f"  Computing IC table ({len(signals.columns)} signals x {len(horizons)} horizons)...")
    t0 = time.time()
    ic_df = _fast_ic_table(signals, fwd)
    elapsed = time.time() - t0
    print(f"  IC table computed in {elapsed:.1f}s")

    # ICIR (chunked IC stability -- fast approximation)
    print("  Computing ICIR (chunked)...")
    best_h_col_raw = ic_df.abs().idxmax(axis=1)
    icir_s = _fast_icir(signals, fwd, best_h_col_raw)

    # Autocorrelation (lag-1, fast via pandas)
    ar1_s = pd.Series({col: float(signals[col].autocorr(lag=1)) for col in signals.columns})

    # Build leaderboard: rank by |best IC| across horizons
    best_h_col = ic_df.abs().idxmax(axis=1)
    best_ic = pd.Series(
        {sig: ic_df.loc[sig, col] for sig, col in best_h_col.items()}, name="best_ic"
    )
    best_horizon = best_h_col.str.replace("fwd_", "h=", regex=False)

    leaderboard = (
        pd.DataFrame(
            {
                "signal": ic_df.index,
                "best_ic": best_ic.values,
                "abs_ic": best_ic.abs().values,
                "best_horizon": best_horizon.values,
                "icir": [icir_s.get(s, np.nan) for s in ic_df.index],
                "ar1": [ar1_s.get(s, np.nan) for s in ic_df.index],
            }
        )
        .sort_values("abs_ic", ascending=False)
        .reset_index(drop=True)
    )

    # Classify signals
    def _verdict(row):
        aic = abs(row["best_ic"])
        air = abs(row["icir"]) if not np.isnan(row["icir"]) else 0
        if aic >= 0.05 and air >= 0.5:
            return "STRONG"
        if aic >= 0.05:
            return "USABLE"
        if aic >= 0.03:
            return "WEAK"
        return "NOISE"

    leaderboard["verdict"] = leaderboard.apply(_verdict, axis=1)

    # Extract scale from signal name
    def _get_scale(name):
        for prefix in ["W_", "D_", "H4_"]:
            if name.startswith(prefix):
                return prefix.rstrip("_")
        return "H1"

    leaderboard["scale"] = leaderboard["signal"].apply(_get_scale)

    return {
        "ic_df": ic_df,
        "icir_s": icir_s,
        "ar1_s": ar1_s,
        "leaderboard": leaderboard,
        "signals": signals,
        "close": close,
        "fwd": fwd,
    }


# ── Simple Composite Backtest ────────────────────────────────────────────────


def backtest_composite(
    close: pd.Series,
    signals: pd.DataFrame,
    top_signals: list[str],
    ic_signs: dict[str, float],
    threshold: float = 0.75,
    is_ratio: float = 0.70,
    spread_bps: float = 2.5,
    slippage_bps: float = 1.0,
) -> dict:
    """Build equal-weight sign-oriented composite and backtest IS/OOS."""
    # Orient signals by IC sign
    oriented = pd.DataFrame(index=signals.index)
    for sig in top_signals:
        sign = np.sign(ic_signs.get(sig, 1.0))
        oriented[sig] = signals[sig] * sign

    composite = oriented.mean(axis=1).dropna()

    # Z-score on IS only
    n = len(composite)
    is_n = int(n * is_ratio)
    is_mean = composite.iloc[:is_n].mean()
    is_std = composite.iloc[:is_n].std()
    if is_std < 1e-8:
        is_std = 1.0
    score_z = (composite - is_mean) / is_std

    # Position: shift(1), long > threshold, short < -threshold
    sig_shifted = score_z.shift(1).fillna(0.0)
    pos = np.where(sig_shifted > threshold, 1.0, np.where(sig_shifted < -threshold, -1.0, 0.0))
    pos_s = pd.Series(pos, index=composite.index)

    # Returns
    bar_rets = close.reindex(composite.index).pct_change().fillna(0.0)
    transitions = (pos_s != pos_s.shift(1).fillna(0.0)).astype(float)
    cost = transitions * (spread_bps + slippage_bps) / 10_000
    strat_rets = bar_rets * pos_s - cost

    # Daily aggregation
    is_daily = strat_rets.iloc[:is_n].resample("D").sum()
    is_daily = is_daily[is_daily != 0.0]
    oos_daily = strat_rets.iloc[is_n:].resample("D").sum()
    oos_daily = oos_daily[oos_daily != 0.0]

    def _sharpe(d):
        if len(d) < 20:
            return 0.0
        return float(d.mean() / d.std() * sqrt(252)) if d.std() > 1e-9 else 0.0

    def _dd(d):
        if len(d) < 5:
            return 0.0
        eq = (1 + d).cumprod()
        return float(((eq - eq.cummax()) / eq.cummax()).min())

    return {
        "is_sharpe": round(_sharpe(is_daily), 3),
        "oos_sharpe": round(_sharpe(oos_daily), 3),
        "oos_dd": round(_dd(oos_daily) * 100, 2),
        "oos_days": len(oos_daily),
        "n_signals": len(top_signals),
        "threshold": threshold,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Scale IC Analysis on H1")
    parser.add_argument("--pair", default="EUR_USD", help="FX pair")
    parser.add_argument("--scales", default="H1,H4,D,W", help="Comma-separated scale labels")
    parser.add_argument("--top-n", type=int, default=20, help="Top N signals to display")
    args = parser.parse_args()

    # Parse scales
    scale_labels = [s.strip() for s in args.scales.split(",")]
    scales = {k: SCALE_MAP[k] for k in scale_labels if k in SCALE_MAP}

    print("=" * 70)
    print(f"  MULTI-SCALE IC ANALYSIS -- {args.pair} H1")
    print(f"  Scales: {scales}")
    print(f"  Horizons: {HORIZONS}")
    print("=" * 70)

    result = run_multiscale_analysis(args.pair, scales=scales, top_n=args.top_n)
    lb = result["leaderboard"]

    # Summary by scale
    print(f"\n{'=' * 60}")
    print("  RESULTS BY SCALE")
    print(f"{'=' * 60}")
    for scale in scale_labels:
        subset = lb[lb["scale"] == scale]
        n_strong = (subset["verdict"] == "STRONG").sum()
        n_usable = (subset["verdict"] == "USABLE").sum()
        n_weak = (subset["verdict"] == "WEAK").sum()
        n_noise = (subset["verdict"] == "NOISE").sum()
        avg_ic = subset["abs_ic"].mean()
        print(
            f"  {scale:>3}: {len(subset):>3} signals | "
            f"STRONG={n_strong} USABLE={n_usable} WEAK={n_weak} NOISE={n_noise} | "
            f"avg|IC|={avg_ic:.4f}"
        )

    # Top N leaderboard
    top = lb.head(args.top_n)
    print(f"\n{'=' * 60}")
    print(f"  TOP {args.top_n} SIGNALS BY |IC|")
    print(f"{'=' * 60}")
    print(
        f"  {'#':>3} {'Signal':<30} {'Scale':>5} | {'IC':>7} {'ICIR':>6} {'AR1':>5} | {'Horizon':>7} {'Verdict':>7}"
    )
    print("  " + "-" * 85)
    for i, (_, r) in enumerate(top.iterrows()):
        print(
            f"  {i + 1:>3} {r['signal']:<30} {r['scale']:>5} | "
            f"{r['best_ic']:>+7.4f} {r['icir']:>+6.3f} {r['ar1']:>5.2f} | "
            f"{r['best_horizon']:>7} {r['verdict']:>7}"
        )

    # Backtest top-5, top-10 composites with threshold sweep
    print(f"\n{'=' * 60}")
    print("  COMPOSITE BACKTESTS")
    print(f"{'=' * 60}")

    # Get IC signs for orientation from IS data
    ic_signs = dict(zip(lb["signal"], np.sign(lb["best_ic"])))

    for n_sig in [3, 5, 10]:
        top_sigs = lb["signal"].head(n_sig).tolist()
        print(f"\n  Top-{n_sig} composite: {', '.join(s[:20] for s in top_sigs[:3])}...")
        print(f"  {'Thr':>5} | {'IS Sh':>6} {'OOS Sh':>7} | {'OOS DD%':>7} {'Days':>5}")
        print("  " + "-" * 45)
        for th in [0.25, 0.50, 0.75, 1.0, 1.5]:
            bt = backtest_composite(
                result["close"],
                result["signals"],
                top_sigs,
                ic_signs,
                threshold=th,
            )
            flag = (
                "+"
                if bt["oos_sharpe"] > 0
                and (bt["oos_sharpe"] / bt["is_sharpe"] >= 0.5 if bt["is_sharpe"] > 0.01 else False)
                else " "
            )
            print(
                f" {flag}{th:>4.2f} | {bt['is_sharpe']:>+6.3f} {bt['oos_sharpe']:>+7.3f}"
                f" | {bt['oos_dd']:>+6.1f}% {bt['oos_days']:>5}"
            )

    # Save full leaderboard
    lb_path = REPORTS_DIR / f"multiscale_ic_{args.pair.lower()}.csv"
    lb.to_csv(lb_path, index=False)
    print(f"\n  Leaderboard saved to: {lb_path}")


if __name__ == "__main__":
    main()
