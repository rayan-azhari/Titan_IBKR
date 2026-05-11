"""run_multiscale_confluence.py -- Multi-Scale Confluence with Agreement Filter.

Tests whether requiring cross-scale AGREEMENT produces a tradeable edge.
Unlike the IC ranking approach (which picks the best individual features),
this requires the SAME signal at all scales to confirm before entering.

Two confluence modes:
  1. Weighted Sum: score = sum(w_i * signal_i) -- enters when score > threshold
  2. AND-Gated: only enters when ALL scales agree on direction AND score > threshold

Sweeps across the 52 signal families to find which ones benefit from confluence.

Usage:
    uv run python research/ic_analysis/run_multiscale_confluence.py --pair GLD
    uv run python research/ic_analysis/run_multiscale_confluence.py --pair EUR_USD --mode and_gate
"""

import argparse
import sys
import time
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
    build_all_signals,
)

WINDOW_1Y_H1 = 252 * 22  # ~5544 H1 bars/year for equities (6.5h/day * 252)

# TF weights from mtf_eurusd.toml (same principle applies to any instrument)
TF_WEIGHTS = {"H1": 0.10, "H4": 0.05, "D": 0.55, "W": 0.30}

THRESHOLDS = [0.25, 0.50, 0.75, 1.0, 1.5]


# ── Vectorized position ─────────────────────────────────────────────────────


def _vectorized_position(sig: np.ndarray, threshold: float, exit_buf: float) -> np.ndarray:
    n = len(sig)
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        s = sig[i]
        if cur == 0.0:
            if s >= threshold:
                cur = 1.0
            elif s <= -threshold:
                cur = -1.0
        elif cur > 0:
            if s < -exit_buf:
                cur = 0.0
        elif cur < 0:
            if s > exit_buf:
                cur = 0.0
        pos[i] = cur
    return pos


# ── Fast backtest ────────────────────────────────────────────────────────────


def _backtest(
    close_vals: np.ndarray,
    score_z: np.ndarray,
    threshold: float,
    exit_buffer: float,
    spread_bps: float,
    slippage_bps: float,
    is_n: int,
) -> dict:
    """Minimal backtest returning IS/OOS Sharpe."""
    n = len(close_vals)

    # Shift by 1
    sig = np.empty(n)
    sig[0] = 0.0
    sig[1:] = score_z[:-1]

    pos = _vectorized_position(sig, threshold, exit_buffer)

    bar_rets = np.empty(n)
    bar_rets[0] = 0.0
    bar_rets[1:] = (close_vals[1:] - close_vals[:-1]) / close_vals[:-1]

    transitions = np.empty(n)
    transitions[0] = abs(pos[0])
    transitions[1:] = np.abs(pos[1:] - pos[:-1])

    strat_rets = bar_rets * pos - transitions * (spread_bps + slippage_bps) / 10_000

    # Daily aggregation (group by ~7 bars for equities H1)
    def _daily_sharpe(rets):
        from titan.research.metrics import BARS_PER_YEAR as _BPY
        from titan.research.metrics import sharpe as _sh

        daily = pd.Series(rets).groupby(np.arange(len(rets)) // 7).sum()
        daily = daily[daily != 0.0]
        if len(daily) < 20:
            return 0.0, 0.0, 0
        sharpe = float(_sh(daily, periods_per_year=_BPY["D"]))
        eq = (1 + daily).cumprod()
        dd = float(((eq - eq.cummax()) / eq.cummax()).min())
        trades = int(np.sum(np.abs(np.diff(pos[: len(rets)])) > 0))
        return sharpe, dd, trades

    is_sh, is_dd, is_tr = _daily_sharpe(strat_rets[:is_n])
    oos_sh, oos_dd, oos_tr = _daily_sharpe(strat_rets[is_n:])
    parity = oos_sh / is_sh if abs(is_sh) > 0.01 else 0.0

    return {
        "is_sharpe": round(is_sh, 3),
        "oos_sharpe": round(oos_sh, 3),
        "parity": round(parity, 3),
        "oos_dd_pct": round(oos_dd * 100, 2),
        "oos_trades": oos_tr,
    }


# ── Confluence builders ──────────────────────────────────────────────────────


def build_weighted_sum_confluence(
    per_scale_signals: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    """Weighted sum: score = sum(w_i * signal_i). Standard MTF approach."""
    parts = []
    for scale, sig in per_scale_signals.items():
        w = weights.get(scale, 0.25)
        parts.append(sig * w)
    df = pd.concat(parts, axis=1).dropna()
    return df.sum(axis=1)


def build_and_gated_confluence(
    per_scale_signals: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    """AND-gated: weighted sum BUT zeroed when any scale disagrees on direction."""
    df = pd.concat(per_scale_signals, axis=1).dropna()

    # Weighted sum
    weighted = pd.Series(0.0, index=df.index)
    for scale in per_scale_signals:
        w = weights.get(scale, 0.25)
        weighted += df[scale] * w

    # AND gate: all scales must agree on sign
    signs = np.sign(df.values)  # (N, n_scales)
    all_positive = (signs > 0).all(axis=1)
    all_negative = (signs < 0).all(axis=1)
    agreement = all_positive | all_negative

    # Zero out weighted sum where scales disagree
    gated = weighted.where(agreement, 0.0)
    return gated


# ── Main sweep ───────────────────────────────────────────────────────────────


def run_confluence_sweep(
    pair: str,
    mode: str = "both",
    spread_bps: float = 2.0,
    slippage_bps: float = 1.0,
    exit_buffer: float = 0.10,
) -> pd.DataFrame:
    """Sweep all 52 signal families with multi-scale confluence.

    For each signal: compute at H1/H4/D/W scale, build confluence, backtest.
    """
    print(f"\n  Loading {pair} H1...")
    df = _load_ohlcv(pair, "H1")
    close = df["close"]
    n = len(close)
    is_n = int(n * 0.70)
    print(f"  {n} bars ({close.index[0].date()} -> {close.index[-1].date()})")

    # Compute signals at each scale
    print("  Computing signals at 4 scales...")
    t0 = time.time()
    scale_signals = {}
    for label, mult in SCALE_MAP.items():
        prefix = f"{label}_" if mult > 1 else ""
        sigs = build_all_signals(df, WINDOW_1Y_H1, period_scale=mult, name_prefix=prefix)
        scale_signals[label] = sigs
    print(f"  Done in {time.time() - t0:.1f}s")

    # Get the 52 signal base names (from H1 scale, no prefix)
    base_names = scale_signals["H1"].columns.tolist()

    # For each signal family, build confluence and backtest
    modes_to_test = []
    if mode in ("both", "weighted"):
        modes_to_test.append(("weighted", build_weighted_sum_confluence))
    if mode in ("both", "and_gate"):
        modes_to_test.append(("and_gate", build_and_gated_confluence))

    rows = []
    total = len(base_names) * len(modes_to_test) * len(THRESHOLDS)
    done = 0

    for sig_name in base_names:
        # Get this signal at each scale
        per_scale = {}
        skip = False
        for label in SCALE_MAP:
            prefix = f"{label}_" if SCALE_MAP[label] > 1 else ""
            col = f"{prefix}{sig_name}"
            if col not in scale_signals[label].columns:
                skip = True
                break
            per_scale[label] = scale_signals[label][col]
        if skip:
            done += len(modes_to_test) * len(THRESHOLDS)
            continue

        for mode_name, confluence_fn in modes_to_test:
            # Build confluence
            confluence = confluence_fn(per_scale, TF_WEIGHTS)
            confluence = confluence.dropna()

            if len(confluence) < 100:
                # Skip signals with too few valid bars (W-scale warmup)
                done += len(THRESHOLDS)
                continue

            # Z-score on IS only
            common_idx = confluence.index
            c_vals = confluence.values
            is_end = min(is_n, len(c_vals))
            if is_end < 50:
                done += len(THRESHOLDS)
                continue
            is_mean = np.nanmean(c_vals[:is_end])
            is_std = np.nanstd(c_vals[:is_end])
            if is_std < 1e-8:
                is_std = 1.0
            score_z = (c_vals - is_mean) / is_std

            # Align close to confluence index
            close_aligned = close.reindex(common_idx).values.astype(np.float64)

            for th in THRESHOLDS:
                result = _backtest(
                    close_aligned,
                    score_z,
                    th,
                    exit_buffer,
                    spread_bps,
                    slippage_bps,
                    is_end,
                )
                rows.append(
                    {
                        "signal": sig_name,
                        "mode": mode_name,
                        "threshold": th,
                        **result,
                    }
                )

                done += 1

        if done % 200 == 0 and done > 0:
            print(f"    {done}/{total} ({done / total * 100:.0f}%)...")

    return pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)


# ── Display ──────────────────────────────────────────────────────────────────


def print_results(sweep_df: pd.DataFrame, pair: str) -> None:
    # Quality gate
    passed = sweep_df[
        (sweep_df["oos_sharpe"] > 0.0)
        & (sweep_df["parity"] >= 0.5)
        & (sweep_df["oos_trades"] >= 30)
    ]

    print(f"\n{'=' * 80}")
    print(f"  MULTI-SCALE CONFLUENCE RESULTS -- {pair} H1")
    print(f"{'=' * 80}")
    print(f"  Total combos: {len(sweep_df)}")
    print(f"  Passed gates (OOS Sh>0, parity>=0.5, trades>=30): {len(passed)}")

    # Summary by mode
    for mode_name in sweep_df["mode"].unique():
        subset = sweep_df[sweep_df["mode"] == mode_name]
        best = subset.iloc[0] if len(subset) > 0 else None
        n_positive = (subset["oos_sharpe"] > 0).sum()
        print(f"\n  {mode_name.upper()}:")
        print(
            f"    Positive OOS: {n_positive}/{len(subset)} ({n_positive / len(subset) * 100:.0f}%)"
        )
        if best is not None:
            print(
                f"    Best: {best['signal']} th={best['threshold']} "
                f"OOS Sh={best['oos_sharpe']:+.3f} DD={best['oos_dd_pct']:.1f}%"
            )

    # Top 20 overall
    top = sweep_df.head(20)
    print("\n  Top 20 by OOS Sharpe:")
    print(
        f"  {'Signal':<25} {'Mode':>8} {'Thr':>5} | {'IS Sh':>6} {'OOS Sh':>7} {'Par':>6}"
        f" | {'DD%':>6} {'Trd':>4}"
    )
    print("  " + "-" * 75)
    for _, r in top.iterrows():
        flag = "+" if r["oos_sharpe"] > 0 and r["parity"] >= 0.5 and r["oos_trades"] >= 30 else " "
        print(
            f" {flag}{r['signal']:<24} {r['mode']:>8} {r['threshold']:>5.2f}"
            f" | {r['is_sharpe']:>+6.3f} {r['oos_sharpe']:>+7.3f} {r['parity']:>6.3f}"
            f" | {r['oos_dd_pct']:>+5.1f}% {r['oos_trades']:>4}"
        )

    # Compare weighted vs AND-gate for signals that appear in both
    if "weighted" in sweep_df["mode"].values and "and_gate" in sweep_df["mode"].values:
        print("\n  WEIGHTED vs AND-GATE (avg OOS Sharpe by signal, threshold=0.75):")
        th75 = sweep_df[sweep_df["threshold"] == 0.75]
        comparison = th75.pivot_table(
            index="signal", columns="mode", values="oos_sharpe", aggfunc="first"
        ).dropna()
        if (
            len(comparison) > 0
            and "weighted" in comparison.columns
            and "and_gate" in comparison.columns
        ):
            comparison["diff"] = comparison["and_gate"] - comparison["weighted"]
            comparison = comparison.sort_values("diff", ascending=False)
            better = (comparison["diff"] > 0).sum()
            print(f"    AND-gate better: {better}/{len(comparison)} signals")
            print(f"    Avg improvement: {comparison['diff'].mean():+.4f}")
            top_improved = comparison.head(5)
            for sig, row in top_improved.iterrows():
                print(
                    f"    {sig:<25} weighted={row['weighted']:+.3f} and_gate={row['and_gate']:+.3f}"
                    f" diff={row['diff']:+.3f}"
                )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Scale Confluence Test")
    parser.add_argument("--pair", default="GLD", help="Instrument")
    parser.add_argument("--mode", default="both", choices=["weighted", "and_gate", "both"])
    parser.add_argument("--spread-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  MULTI-SCALE CONFLUENCE -- {args.pair} H1")
    print(f"  Mode: {args.mode} | Costs: {args.spread_bps}+{args.slippage_bps} bps")
    print(f"  Signals: 52 families × 4 scales × {len(THRESHOLDS)} thresholds")
    print("=" * 60)

    sweep_df = run_confluence_sweep(
        args.pair,
        mode=args.mode,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
    )

    print_results(sweep_df, args.pair)

    # Save
    save_path = REPORTS_DIR / f"multiscale_confluence_{args.pair.lower()}.csv"
    sweep_df.to_csv(save_path, index=False)
    print(f"\n  Saved to: {save_path}")


if __name__ == "__main__":
    main()
