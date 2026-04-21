"""plot_52sig_signals.py -- Visualise ML signal entries/exits + equity + drawdown.

Runs the 52-signal ML pipeline on one instrument, collects all OOS predictions
across WFO folds, and generates a 3-panel interactive Plotly chart:

  Panel 1: Price with long entries (green ^), short entries (red v), exits (grey x)
  Panel 2: Cumulative equity curve (stitched OOS only)
  Panel 3: Drawdown from high-water mark

Output: .tmp/reports/ml_52sig_chart_{instrument}_{tf}.html (interactive)

Usage
-----
    uv run python research/ml/plot_52sig_signals.py --instrument SPY --tf D
    uv run python research/ml/plot_52sig_signals.py --instrument QQQ --tf D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from research.ic_analysis.phase1_sweep import _load_ohlcv  # noqa: E402
from research.ml.run_52signal_classifier import (  # noqa: E402
    COST_BPS,
    IS_RATIO_BARS,
    OOS_RATIO_BARS,
    SIGNAL_THRESHOLD,
    TARGET_INSTRUMENTS,
    TRAIL_MAX_HOLD_D,
    TRAIL_MAX_HOLD_H1,
    TRAIL_MAX_HOLD_M5,
    TRAIL_STOP_MULT,
    XGB_PARAMS,
    build_features,
    compute_trailing_labels,
    walk_forward_splits,
)


def run_and_collect_signals(
    instrument: str, tf: str, asset_type: str, stop_mult: float, max_hold: int
) -> dict:
    """Run WFO pipeline and collect all OOS signals with timestamps."""
    from xgboost import XGBRegressor

    df = _load_ohlcv(instrument, tf)
    n_bars = len(df)
    cost = COST_BPS.get(asset_type, 1.0)

    print(f"  {n_bars:,} bars | {df.index[0].date()} to {df.index[-1].date()}")

    df.attrs["instrument"] = instrument
    features = build_features(df, tf)
    long_r, short_r = compute_trailing_labels(df, stop_mult, max_hold)
    net_r = long_r - short_r
    bar_returns = df["close"].pct_change().fillna(0.0)

    mask_valid = net_r.notna() & features.notna().all(axis=1)
    features_clean = features[mask_valid].copy()
    target_clean = net_r[mask_valid].copy()
    returns_clean = bar_returns.reindex(features_clean.index).fillna(0.0)

    is_bars_n = IS_RATIO_BARS.get(tf, 504)
    oos_bars_n = OOS_RATIO_BARS.get(tf, 126)
    folds = walk_forward_splits(len(features_clean), is_bars_n, oos_bars_n)

    X = features_clean.values
    y_arr = target_clean.values

    all_signals = []
    all_oos_rets = []

    for i, (is_idx, oos_idx) in enumerate(folds):
        X_is = np.nan_to_num(X[is_idx], nan=0.0, posinf=0.0, neginf=0.0)
        y_is = y_arr[is_idx]
        X_oos = np.nan_to_num(X[oos_idx], nan=0.0, posinf=0.0, neginf=0.0)

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_is, y_is)

        preds = model.predict(X_oos)
        signal = np.where(
            preds > SIGNAL_THRESHOLD,
            1.0,
            np.where(preds < -SIGNAL_THRESHOLD, -1.0, 0.0),
        )

        oos_ts = features_clean.index[oos_idx]
        oos_rets = returns_clean.iloc[oos_idx]

        positions = pd.Series(signal, index=oos_ts)
        transitions = (positions != positions.shift(1).fillna(0.0)).astype(float)
        strat_rets = positions * oos_rets - transitions * cost / 10_000

        sig_df = pd.DataFrame(
            {"signal": signal, "pred_r": preds, "strat_ret": strat_rets.values},
            index=oos_ts,
        )
        all_signals.append(sig_df)
        all_oos_rets.append(strat_rets)

        print(f"    Fold {i + 1}/{len(folds)} done")

    signals = pd.concat(all_signals).sort_index()
    # Remove duplicate timestamps (fold overlap) — keep last
    signals = signals[~signals.index.duplicated(keep="last")]

    stitched_rets = pd.concat(all_oos_rets).sort_index()
    stitched_rets = stitched_rets[~stitched_rets.index.duplicated(keep="last")]

    return {
        "df": df,
        "signals": signals,
        "stitched_rets": stitched_rets,
        "instrument": instrument,
        "tf": tf,
    }


def plot_signals(result: dict) -> str:
    """Generate 3-panel Plotly chart and save as HTML. Returns path."""
    df = result["df"]
    signals = result["signals"]
    stitched = result["stitched_rets"]
    instrument = result["instrument"]
    tf = result["tf"]

    # Align price data to OOS signal period
    oos_start = signals.index[0]
    oos_end = signals.index[-1]
    price = df["close"].loc[oos_start:oos_end]

    # Entry/exit detection
    sig = signals["signal"]
    prev_sig = sig.shift(1).fillna(0.0)

    long_entry = (sig == 1.0) & (prev_sig != 1.0)
    short_entry = (sig == -1.0) & (prev_sig != -1.0)
    exit_signal = (sig == 0.0) & (prev_sig != 0.0)

    # Equity curve
    equity = (1.0 + stitched).cumprod()

    # Drawdown
    hwm = equity.cummax()
    drawdown = (equity - hwm) / hwm

    # Stats
    from titan.research.metrics import BARS_PER_YEAR as _BPY
    from titan.research.metrics import sharpe as _sh

    sharpe = float(_sh(stitched, periods_per_year=_BPY["D"]))
    max_dd = float(drawdown.min())
    total_ret = float(equity.iloc[-1] - 1.0)
    n_long = int(long_entry.sum())
    n_short = int(short_entry.sum())

    # Build figure
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.5, 0.3, 0.2],
        subplot_titles=[
            f"{instrument} {tf} -- Price + ML Signals (OOS only)",
            f"Equity Curve (Sharpe={sharpe:+.2f}, Return={total_ret:+.1%})",
            f"Drawdown (Max={max_dd:.1%})",
        ],
    )

    # Panel 1: Price
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price.values,
            mode="lines",
            name="Close",
            line=dict(color="steelblue", width=1),
        ),
        row=1,
        col=1,
    )

    # Long entries
    le_idx = long_entry[long_entry].index
    fig.add_trace(
        go.Scatter(
            x=le_idx,
            y=price.reindex(le_idx).values,
            mode="markers",
            name=f"Long Entry ({n_long})",
            marker=dict(
                symbol="triangle-up", size=8, color="lime", line=dict(width=1, color="darkgreen")
            ),
        ),
        row=1,
        col=1,
    )

    # Short entries
    se_idx = short_entry[short_entry].index
    fig.add_trace(
        go.Scatter(
            x=se_idx,
            y=price.reindex(se_idx).values,
            mode="markers",
            name=f"Short Entry ({n_short})",
            marker=dict(
                symbol="triangle-down", size=8, color="red", line=dict(width=1, color="darkred")
            ),
        ),
        row=1,
        col=1,
    )

    # Exits
    ex_idx = exit_signal[exit_signal].index
    fig.add_trace(
        go.Scatter(
            x=ex_idx,
            y=price.reindex(ex_idx).values,
            mode="markers",
            name=f"Exit ({int(exit_signal.sum())})",
            marker=dict(symbol="x", size=6, color="grey"),
        ),
        row=1,
        col=1,
    )

    # Shade long/short regions
    for idx in sig.index:
        pass  # too many bars for individual shapes; entries/exits markers suffice

    # Panel 2: Equity
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Equity",
            line=dict(color="green", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", row=2, col=1)

    # Panel 3: Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="crimson", width=1),
            fillcolor="rgba(220,20,60,0.3)",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=1000,
        width=1600,
        title_text=f"ML 52-Signal Strategy -- {instrument} {tf} (OOS Walk-Forward)",
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        template="plotly_dark",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", tickformat=".0%", row=3, col=1)

    out_path = REPORTS_DIR / f"ml_52sig_chart_{instrument.replace('^', '')}_{tf}.html"
    fig.write_html(str(out_path))
    print(f"\n  Chart saved -> {out_path}")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ML signal entries/exits + equity + drawdown."
    )
    parser.add_argument("--instrument", required=True, help="Instrument to plot.")
    parser.add_argument("--tf", default="D", help="Timeframe (default: D).")
    parser.add_argument("--stop-mult", type=float, default=TRAIL_STOP_MULT)
    parser.add_argument("--max-hold", type=int, default=0)
    args = parser.parse_args()

    if args.instrument in TARGET_INSTRUMENTS:
        _, asset_type = TARGET_INSTRUMENTS[args.instrument]
    else:
        asset_type = "index"
    tf = args.tf
    max_hold = (
        args.max_hold
        if args.max_hold > 0
        else {"H1": TRAIL_MAX_HOLD_H1, "D": TRAIL_MAX_HOLD_D, "M5": TRAIL_MAX_HOLD_M5}.get(
            tf, TRAIL_MAX_HOLD_D
        )
    )

    print(f"\n  Running ML pipeline for {args.instrument} {tf} ...")
    result = run_and_collect_signals(args.instrument, tf, asset_type, args.stop_mult, max_hold)

    print("\n  Generating chart ...")
    plot_signals(result)


if __name__ == "__main__":
    main()
