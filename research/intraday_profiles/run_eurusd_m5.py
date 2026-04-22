"""run_eurusd_m5.py — EUR/USD M5 day-profile AM/PM conditional probability pipeline.

Stages (controlled by --stage flag):
  1 — Seasonality heatmap (hourly volatility/drift/volume)
  2 — Archetype discovery (PCA K-means sweep K=4..6, AM + PM)
  3 — AM/PM signal backtest (IS/OOS + WFO, quality gates)

Usage:
    uv run python research/intraday_profiles/run_eurusd_m5.py --stage 1
    uv run python research/intraday_profiles/run_eurusd_m5.py --stage 2
    uv run python research/intraday_profiles/run_eurusd_m5.py --stage 3
    uv run python research/intraday_profiles/run_eurusd_m5.py  # runs all stages
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.  Run: uv sync")
    sys.exit(1)

from research.intraday_profiles import pca_archetypes as arch
from research.intraday_profiles import seasonality as seas
from research.intraday_profiles.am_pm_signal import AMPMPipeline, expand_signal_to_bars
from research.intraday_profiles.day_constructor import (
    compute_vol_baseline,
    engineer_features,
    normalise_tr,
)
from research.intraday_profiles.pca_archetypes import sweep_k
from titan.models.spread import build_spread_series

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PAIR = "EUR_USD"
TF = "M5"
IS_SPLIT = 0.70

N_AM_CLUSTERS = 5
N_PM_CLUSTERS = 5
THRESHOLD = 0.20  # Minimum P to generate signal (PM clusters ~10% base rate)

# Quality gates
GATE_OOS_SHARPE = 1.0
GATE_OOS_IS_RATIO = 0.5
GATE_WIN_RATE = 0.40
GATE_MAX_DD = 0.25
GATE_MIN_OOS_TRADES = 30
GATE_MC_5PCT_SHARPE = 0.5
GATE_MC_PCT_PROFIT = 0.80


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eurusd_m5() -> pd.DataFrame:
    path = DATA_DIR / f"{PAIR}_{TF}.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    print(
        f"  Loaded {PAIR}_{TF}: {len(df):,} bars  [{df.index[0].date()} -> {df.index[-1].date()}]"
    )
    return df


# ---------------------------------------------------------------------------
# Stage 1 — Seasonality
# ---------------------------------------------------------------------------


def run_stage1(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Stage 1 — Intraday Seasonality Heatmap")
    print("=" * 60)

    seasonality = seas.hourly_seasonality(df)
    seasonality = seas.t_test_drift(seasonality, df)
    print("\n  Hourly stats:")
    print(seasonality.to_string())

    sig_hours = seasonality[seasonality["p_value"] < 0.05]
    print(f"\n  Hours with significant drift (p<0.05): {sig_hours.index.tolist()}")

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=["Absolute Return (Volatility)", "Directional Drift", "Relative Volume"],
    )
    hours = seasonality.index.tolist()
    for row, col, title in [
        (1, "abs_ret", "Abs Return"),
        (2, "fwd_drift", "Fwd Drift"),
        (3, "volume_norm", "Rel Volume"),
    ]:
        fig.add_trace(
            go.Bar(
                x=hours,
                y=seasonality[col],
                name=title,
                marker_color="steelblue"
                if col != "fwd_drift"
                else ["red" if v < 0 else "green" for v in seasonality[col]],
            ),
            row=row,
            col=1,
        )
    fig.update_layout(title=f"{PAIR} {TF} — Intraday Seasonality (UTC hours)", height=800)
    out = REPORTS_DIR / "eurusd_m5_seasonality.html"
    fig.write_html(str(out))
    print(f"\n  Saved -> {out}")


# ---------------------------------------------------------------------------
# Stage 2 — Archetype Discovery
# ---------------------------------------------------------------------------


def run_stage2(df_is: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Stage 2 — Archetype Discovery (IS data only)")
    print("=" * 60)

    print("  Computing volume baseline and features...")
    vol_base = compute_vol_baseline(df_is, window_days=30)
    features_is = engineer_features(df_is, vol_base)
    is_mean_tr = float(features_is["tr"].mean())
    features_is = normalise_tr(features_is, is_mean_tr)

    for session in ["am", "pm"]:
        from research.intraday_profiles.day_constructor import build_daily_matrices

        mats, dates = build_daily_matrices(features_is, session=session)
        print(f"\n  {session.upper()} session: {len(mats)} valid days, tensor shape {mats.shape}")

        print(f"  Sweeping K = 4..6 ({session.upper()})...")
        results = sweep_k(mats, k_values=[4, 5, 6])
        for r in results:
            print(
                f"    K={r['k']}  silhouette={r['silhouette']:.4f}  "
                f"expl_var={r['explained_variance']:.1%}"
            )

        # Describe best-K archetypes
        best = results[0]
        model = best["model"]
        labels = model.predict(mats)
        descriptions = arch.describe_archetypes(mats, labels, best["k"])
        print(f"\n  Best K={best['k']} archetypes ({session.upper()}):")
        for d in descriptions:
            print(
                f"    Cluster {d['label']}: n={d['n_days']}  "
                f"CLV_start={d['mean_clv_start']:+.3f}  CLV_end={d['mean_clv_end']:+.3f}  "
                f"-> '{d['suggested_name']}'"
            )

        # Plot centroid CLV trajectories
        n_bars = mats.shape[1]
        x_axis = list(range(n_bars))
        fig = go.Figure()
        for d in descriptions:
            mask = labels == d["label"]
            mean_clv = mats[mask, :, 1].mean(axis=0)
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=mean_clv,
                    name=f"C{d['label']}:{d['suggested_name']} (n={d['n_days']})",
                    mode="lines",
                )
            )
        x_midpoint = n_bars // 2
        fig.add_vline(x=x_midpoint, line_dash="dot", line_color="grey", annotation_text="midpoint")
        fig.update_layout(
            title=f"{PAIR} {TF} — {session.upper()} Archetypes K={best['k']} (IS) — Mean CLV",
            xaxis_title="Bar index within session",
            yaxis_title="Mean Close Location Value",
            height=500,
        )
        out = REPORTS_DIR / f"eurusd_m5_archetypes_{session}_K{best['k']}.html"
        fig.write_html(str(out))
        print(f"  Saved -> {out}")


# ---------------------------------------------------------------------------
# Stage 3 — Backtest
# ---------------------------------------------------------------------------


def build_vbt_portfolio(
    close: pd.Series,
    signal_series: pd.Series,
    spread: pd.Series,
    total_cash: float = 10_000.0,
) -> object:
    """Build a single VectorBT portfolio from a +1/-1/0 signal series."""
    expanded = expand_signal_to_bars(signal_series, close.index)

    long_entries = (expanded == 1) & (expanded.shift(1) != 1)
    short_entries = (expanded == -1) & (expanded.shift(1) != -1)
    exits = expanded == 0

    pf = vbt.Portfolio.from_signals(
        close,
        entries=long_entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=exits,
        init_cash=total_cash,
        fees=spread.reindex(close.index).values,
        freq="5min",
    )
    return pf


def run_pipeline_slice(
    df_slice: pd.DataFrame,
    df_is: pd.DataFrame,
    label: str,
    spread_multiplier: float = 1.0,
) -> dict:
    """Fit IS models, score on df_slice, backtest.

    Args:
        df_slice: Data to backtest on (IS or OOS).
        df_is: IS data used to fit models.
        label: Label for printing.
        spread_multiplier: Cost multiplier (1.0 = realistic, 3.0 = stress).

    Returns:
        Dict with sharpe, max_dd, n_trades, win_rate, daily_returns.
    """
    # Fit on IS
    vol_base = compute_vol_baseline(df_is, window_days=30)
    is_mean_tr = float(engineer_features(df_is, vol_base)["tr"].mean())

    features_is = normalise_tr(engineer_features(df_is, vol_base), is_mean_tr)
    pipe = AMPMPipeline(
        n_am=N_AM_CLUSTERS,
        n_pm=N_PM_CLUSTERS,
        threshold=THRESHOLD,
        random_state=42,
    )
    pipe.fit(features_is)

    # Score on slice (using IS vol_base + is_mean_tr for normalisation)
    vol_base_slice = compute_vol_baseline(df_slice, window_days=30)
    # Merge IS baseline into slice periods without baseline (avoids look-ahead)
    features_slice = engineer_features(df_slice, vol_base_slice)
    features_slice = normalise_tr(features_slice, is_mean_tr)

    daily_signals = pipe.score(features_slice)
    active_days = int((daily_signals != 0).sum())

    # Build spread
    spread = build_spread_series(df_slice, PAIR) * spread_multiplier
    close = df_slice["close"]

    if active_days == 0:
        print(f"  {label:<12}  0 active signal days")
        return {
            "label": label,
            "sharpe": float("nan"),
            "max_dd": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "daily_returns": pd.Series(dtype=float),
        }

    pf = build_vbt_portfolio(close, daily_signals, spread)
    n_trades = int(pf.trades.count())
    win_rate = float(pf.trades.win_rate()) if n_trades > 0 else 0.0

    from titan.research.metrics import BARS_PER_YEAR as _BPY
    from titan.research.metrics import sharpe as _sh

    daily_ret = pf.returns().resample("1D").sum()
    sharpe = float(_sh(daily_ret, periods_per_year=_BPY["D"]))
    max_dd = float(pf.max_drawdown())
    weeks = (df_slice.index[-1] - df_slice.index[0]).days / 7

    print(
        f"  {label:<12}  signal_days={active_days}  "
        f"Sharpe={sharpe:+.2f}  MaxDD={max_dd:.2%}  "
        f"WR={win_rate:.2%}  n={n_trades}  "
        f"tpw={n_trades / max(weeks, 1):.1f}"
    )

    return {
        "label": label,
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 4),
        "n_trades": n_trades,
        "trades_per_week": round(n_trades / max(weeks, 1), 2),
        "win_rate": round(win_rate, 3),
        "daily_returns": daily_ret,
    }


def monte_carlo(daily_returns: pd.Series, n: int = 500, pct: float = 5.0) -> dict:
    if len(daily_returns) == 0 or daily_returns.std() == 0:
        return {"pct_sharpe": float("nan"), "pct_profitable": 0.0, "pass": False}
    from titan.research.metrics import BARS_PER_YEAR as _BPY2
    from titan.research.metrics import sharpe as _sh2

    np.random.seed(42)
    vals, sharpes = daily_returns.values.copy(), []
    for _ in range(n):
        np.random.shuffle(vals)
        s = pd.Series(vals)
        if s.std() > 0:
            sharpes.append(float(_sh2(s, periods_per_year=_BPY2["D"])))
        else:
            sharpes.append(float("nan"))
    sharpes = [s for s in sharpes if not np.isnan(s)]
    if not sharpes:
        return {"pct_sharpe": float("nan"), "pct_profitable": 0.0, "pass": False}
    p5 = float(np.percentile(sharpes, pct))
    p_pos = float(np.mean([s > 0 for s in sharpes]))
    return {
        "pct_sharpe": round(p5, 3),
        "pct_profitable": round(p_pos, 3),
        "pass": bool(p5 > GATE_MC_5PCT_SHARPE and p_pos >= GATE_MC_PCT_PROFIT),
    }


def walk_forward(
    df: pd.DataFrame,
    is_months: int = 6,
    oos_months: int = 2,
) -> tuple[pd.Series, list[dict]]:
    dates = pd.date_range(df.index[0], df.index[-1], freq="MS")
    fold_results, stitched = [], []

    for i in range(is_months, len(dates) - oos_months):
        oos_start = dates[i]
        oos_end = dates[i + oos_months]
        is_start = dates[max(0, i - is_months)]

        df_oos_fold = df[(df.index >= oos_start) & (df.index < oos_end)]
        df_is_fold = df[(df.index >= is_start) & (df.index < oos_start)]

        if len(df_oos_fold) < 500 or len(df_is_fold) < 1000:
            continue

        m = run_pipeline_slice(df_oos_fold, df_is_fold, label=f"wfo_{i}")
        fold_results.append(
            {
                "fold": i,
                "oos_start": str(df_oos_fold.index[0].date()),
                "oos_end": str(df_oos_fold.index[-1].date()),
                "sharpe": m["sharpe"],
                "n_trades": m["n_trades"],
            }
        )
        if len(m["daily_returns"]) > 0:
            stitched.append(m["daily_returns"])

    stitched_ret = pd.concat(stitched) if stitched else pd.Series(dtype=float)
    return stitched_ret, fold_results


def run_stage3(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Stage 3 — AM/PM Signal Backtest (IS/OOS + WFO)")
    print("=" * 60)

    is_end = int(len(df) * IS_SPLIT)
    df_is = df.iloc[:is_end]
    df_oos = df.iloc[is_end:]
    print(f"  IS:  {len(df_is):,} bars  [{df_is.index[0].date()} -> {df_is.index[-1].date()}]")
    print(f"  OOS: {len(df_oos):,} bars  [{df_oos.index[0].date()} -> {df_oos.index[-1].date()}]")

    # [1/5] IS and OOS
    print("\n[1/5] IS and OOS backtests...")
    is_m = run_pipeline_slice(df_is, df_is, label="IS")
    oos_m = run_pipeline_slice(df_oos, df_is, label="OOS")
    ratio = (
        oos_m["sharpe"] / is_m["sharpe"]
        if is_m["sharpe"] not in (0.0, float("nan")) and not np.isnan(is_m["sharpe"])
        else 0.0
    )
    print(f"  OOS/IS ratio: {ratio:.2f}")

    # [2/5] Monte Carlo
    print("\n[2/5] Monte Carlo (G1)...")
    mc = monte_carlo(oos_m["daily_returns"])
    print(
        f"  5th-pct Sharpe={mc['pct_sharpe']}  "
        f"% profitable={mc['pct_profitable']:.1%}  PASS={mc['pass']}"
    )

    # [3/5] 3x slippage
    print("\n[3/5] 3x slippage stress (G3)...")
    stress = run_pipeline_slice(df_oos, df_is, label="OOS_3x", spread_multiplier=3.0)
    print(f"  3x stress Sharpe={stress['sharpe']:+.2f}  PASS={stress['sharpe'] > 0.5}")

    # [4/5] WFO
    print("\n[4/5] WFO (6mo IS / 2mo OOS)...")
    wfo_ret, wfo_folds = walk_forward(df)
    neg = sum(1 for f in wfo_folds if not np.isnan(f["sharpe"]) and f["sharpe"] < 0)
    from titan.research.metrics import BARS_PER_YEAR as _BPY3
    from titan.research.metrics import sharpe as _sh3

    wfo_sharpe = (
        float(_sh3(wfo_ret, periods_per_year=_BPY3["D"]))
        if len(wfo_ret) > 0 and wfo_ret.std() > 0
        else float("nan")
    )
    print(f"  WFO stitched Sharpe={wfo_sharpe:.3f}  Negative folds={neg}/{len(wfo_folds)}")
    for f in wfo_folds:
        print(
            f"    {f['oos_start']} -> {f['oos_end']}  Sharpe={f['sharpe']:+.2f}  n={f['n_trades']}"
        )

    # [5/5] Gates
    print("\n[5/5] Quality gates...")
    gates = {
        "G_oos_sharpe": (not np.isnan(oos_m["sharpe"])) and oos_m["sharpe"] > GATE_OOS_SHARPE,
        "G_oos_is_ratio": (not np.isnan(ratio)) and ratio > GATE_OOS_IS_RATIO,
        "G_win_rate": oos_m["win_rate"] > GATE_WIN_RATE,
        "G_max_dd": oos_m["max_dd"] < GATE_MAX_DD,
        "G_min_trades": oos_m["n_trades"] >= GATE_MIN_OOS_TRADES,
        "G1_monte_carlo": mc["pass"],
        "G3_slippage": (not np.isnan(stress["sharpe"])) and stress["sharpe"] > 0.5,
        "G4_wfo_neg_folds": neg <= 2,
    }
    all_passed = all(gates.values())
    print(f"\n  {'GATE':<25} {'RESULT'}")
    print("  " + "-" * 40)
    for g, passed in gates.items():
        print(f"  {g:<25} [{'PASS' if passed else 'FAIL'}]")
    print(f"\n  Overall: {'ALL GATES PASSED' if all_passed else 'FAILED -- do not deploy'}")

    # Save report
    report = {
        "instrument": PAIR,
        "timeframe": TF,
        "n_am_clusters": N_AM_CLUSTERS,
        "n_pm_clusters": N_PM_CLUSTERS,
        "threshold": THRESHOLD,
        "is": {k: v for k, v in is_m.items() if k != "daily_returns"},
        "oos": {k: v for k, v in oos_m.items() if k != "daily_returns"},
        "oos_is_ratio": round(float(ratio), 3),
        "monte_carlo": mc,
        "stress_3x_sharpe": oos_m["sharpe"] if np.isnan(stress["sharpe"]) else stress["sharpe"],
        "wfo": {
            "stitched_sharpe": round(wfo_sharpe, 3) if not np.isnan(wfo_sharpe) else None,
            "negative_folds": neg,
            "total_folds": len(wfo_folds),
            "folds": wfo_folds,
        },
        "gates": {k: bool(v) for k, v in gates.items()},
        "all_gates_passed": bool(all_passed),
    }
    out_json = REPORTS_DIR / "eurusd_m5_ampm_validation.json"
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=4)
    print(f"\n  Saved -> {out_json}")

    # Equity curve chart
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=["IS vs OOS Cumulative Returns", "WFO Stitched OOS Equity"]
    )
    for ret, name, colour in [
        (is_m["daily_returns"], "IS", "steelblue"),
        (oos_m["daily_returns"], "OOS", "orange"),
    ]:
        if len(ret) > 0:
            cum = (1 + ret).cumprod()
            fig.add_trace(
                go.Scatter(x=cum.index, y=cum, name=name, line=dict(color=colour)), row=1, col=1
            )
    if len(wfo_ret) > 0:
        wfo_cum = (1 + wfo_ret).cumprod()
        fig.add_trace(
            go.Scatter(x=wfo_cum.index, y=wfo_cum, name="WFO OOS", line=dict(color="green")),
            row=2,
            col=1,
        )
    fig.update_layout(
        title=f"EUR/USD M5 AM/PM Profile -- {'PASSED' if all_passed else 'FAILED'}",
        height=700,
    )
    out_html = REPORTS_DIR / "eurusd_m5_ampm_equity_curve.html"
    fig.write_html(str(out_html))
    print(f"  Saved -> {out_html}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="EUR/USD M5 intraday profile pipeline")
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        help="Stage to run: 1=seasonality, 2=archetypes, 3=backtest, 0=all",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("EUR/USD M5 — Intraday Day-Profile AM/PM Signal")
    print("=" * 60)

    df = load_eurusd_m5()
    is_end = int(len(df) * IS_SPLIT)
    df_is = df.iloc[:is_end]

    if args.stage in (0, 1):
        run_stage1(df)
    if args.stage in (0, 2):
        run_stage2(df_is)
    if args.stage in (0, 3):
        run_stage3(df)


if __name__ == "__main__":
    main()
