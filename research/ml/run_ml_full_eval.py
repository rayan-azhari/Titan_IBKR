"""run_ml_full_eval.py -- Comprehensive ML pipeline evaluation with full statistics.

Runs the 52-signal classifier (regime+pullback labels) on Daily data for
QQQ, SPY, EUR_USD, then computes and displays:

  - Equity curve (interactive Plotly chart)
  - Total return, CAGR, Sharpe, Sortino, Calmar
  - Max drawdown, max drawdown duration
  - Number of trades (long + short), win rate
  - Average win/loss, profit factor
  - Risk of ruin (Monte Carlo)
  - Annual returns breakdown

Usage
-----
    uv run python research/ml/run_ml_full_eval.py
    uv run python research/ml/run_ml_full_eval.py --instrument QQQ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from research.ic_analysis.phase1_sweep import (  # noqa: E402
    _get_annual_bars,
    _load_ohlcv,
)
from research.ml.run_52signal_classifier import (  # noqa: E402
    COST_BPS,
    IS_RATIO_BARS,
    OOS_RATIO_BARS,
    XGB_PARAMS,
    _pred_to_position,
    build_features,
    compute_regime_pullback_labels,
    walk_forward_splits,
)

# ---------------------------------------------------------------------------
# Target instruments for daily evaluation
# ---------------------------------------------------------------------------
EVAL_INSTRUMENTS = {
    "QQQ": ("D", "index"),
    "SPY": ("D", "index"),
    "EUR_USD": ("D", "fx"),
}

# Label sweep configs (regime+pullback, same as run_52signal_classifier v3)
LABEL_CONFIGS = [
    {"rsi_oversold": 45, "rsi_overbought": 55, "confirm_bars": 10, "confirm_pct": 0.005},
    {"rsi_oversold": 45, "rsi_overbought": 55, "confirm_bars": 20, "confirm_pct": 0.005},
    {"rsi_oversold": 45, "rsi_overbought": 55, "confirm_bars": 10, "confirm_pct": 0.01},
    {"rsi_oversold": 40, "rsi_overbought": 60, "confirm_bars": 10, "confirm_pct": 0.005},
    {"rsi_oversold": 40, "rsi_overbought": 60, "confirm_bars": 20, "confirm_pct": 0.01},
    {"rsi_oversold": 50, "rsi_overbought": 50, "confirm_bars": 10, "confirm_pct": 0.003},
    {"rsi_oversold": 50, "rsi_overbought": 50, "confirm_bars": 5, "confirm_pct": 0.002},
    {"rsi_oversold": 48, "rsi_overbought": 52, "confirm_bars": 10, "confirm_pct": 0.005},
]

SIGNAL_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Comprehensive statistics
# ---------------------------------------------------------------------------


def compute_full_stats(
    stitched_rets: pd.Series,
    positions: pd.Series,
    bars_per_year: int,
    cost_bps: float,
) -> dict:
    """Compute comprehensive trading statistics from stitched OOS returns."""
    equity = (1.0 + stitched_rets).cumprod()
    total_ret = float(equity.iloc[-1] - 1.0)
    n_bars = len(stitched_rets)
    n_years = n_bars / bars_per_year

    # CAGR
    cagr = float((1 + total_ret) ** (1.0 / n_years) - 1) if n_years > 0 else 0.0

    # Sharpe
    std = float(stitched_rets.std())
    sharpe = float(stitched_rets.mean() / std * np.sqrt(bars_per_year)) if std > 1e-10 else 0.0

    # Sortino (downside deviation only)
    downside = stitched_rets[stitched_rets < 0]
    down_std = float(downside.std()) if len(downside) > 10 else std
    sortino = (
        float(stitched_rets.mean() / down_std * np.sqrt(bars_per_year)) if down_std > 1e-10 else 0.0
    )

    # Max drawdown + duration
    hwm = equity.cummax()
    dd = (equity - hwm) / hwm
    max_dd = float(dd.min())

    # Drawdown duration (in bars)
    in_dd = (dd < 0).astype(int)
    dd_groups = (in_dd != in_dd.shift()).cumsum()
    dd_durations = in_dd.groupby(dd_groups).sum()
    max_dd_duration_bars = int(dd_durations.max()) if len(dd_durations) > 0 else 0
    max_dd_duration_days = max_dd_duration_bars  # Daily bars = days

    # Calmar ratio
    calmar = abs(cagr / max_dd) if abs(max_dd) > 1e-10 else 0.0

    # Group returns by position runs (each contiguous block of same position)
    pos_groups = (positions != positions.shift()).cumsum()
    trade_returns = stitched_rets.groupby(pos_groups).sum()
    # Only count actual trades (non-zero positions)
    active_mask = positions.groupby(pos_groups).first() != 0
    active_trades = trade_returns[active_mask]

    n_trades = len(active_trades)
    n_wins = int((active_trades > 0).sum())
    win_rate = float(n_wins / n_trades) if n_trades > 0 else 0.0

    # Average win / average loss
    wins = active_trades[active_trades > 0]
    losses = active_trades[active_trades < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-10 else float("inf")

    # Profit factor
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else float("inf")

    # Long vs short trade breakdown
    long_mask = positions.groupby(pos_groups).first() > 0
    short_mask = positions.groupby(pos_groups).first() < 0
    long_trades = trade_returns[long_mask & active_mask]
    short_trades = trade_returns[short_mask & active_mask]

    n_long_trades = len(long_trades)
    n_short_trades = len(short_trades)
    long_win_rate = float((long_trades > 0).sum() / n_long_trades) if n_long_trades > 0 else 0.0
    short_win_rate = float((short_trades > 0).sum() / n_short_trades) if n_short_trades > 0 else 0.0

    # Annual returns
    annual_rets = {}
    if hasattr(stitched_rets.index, "year"):
        for year in sorted(stitched_rets.index.year.unique()):
            yr_rets = stitched_rets[stitched_rets.index.year == year]
            yr_eq = (1.0 + yr_rets).cumprod()
            annual_rets[int(year)] = float(yr_eq.iloc[-1] - 1.0)

    # Time in market / capital invested
    pct_long = float((positions > 0).mean())
    pct_short = float((positions < 0).mean())
    pct_flat = float((positions == 0).mean())
    # Capital invested = |position| (1.0 when long or short, 0.0 when flat)
    capital_invested = positions.abs()
    avg_capital_invested = float(capital_invested.mean())
    # Rolling 63-bar (~quarter) average for chart
    rolling_exposure = capital_invested.rolling(
        min(63, max(10, n_bars // 10)), min_periods=1
    ).mean()

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "max_dd_duration_days": max_dd_duration_days,
        "n_trades": n_trades,
        "n_long_trades": n_long_trades,
        "n_short_trades": n_short_trades,
        "win_rate": win_rate,
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "profit_factor": profit_factor,
        "pct_long": pct_long,
        "pct_short": pct_short,
        "pct_flat": pct_flat,
        "avg_capital_invested": avg_capital_invested,
        "rolling_exposure": rolling_exposure,
        "annual_returns": annual_rets,
        "n_bars": n_bars,
        "n_years": n_years,
        "cost_bps": cost_bps,
    }


def risk_of_ruin(
    stitched_rets: pd.Series,
    ruin_threshold: float = -0.50,
    n_sims: int = 5000,
    n_bars: int | None = None,
) -> float:
    """Monte Carlo risk-of-ruin estimation.

    Bootstrap-resamples the OOS returns and measures how often the equity
    curve drops below the ruin threshold (default -50%).

    Returns probability of ruin (0.0 to 1.0).
    """
    if n_bars is None:
        n_bars = len(stitched_rets)
    rets = stitched_rets.values
    ruin_count = 0
    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        sim_rets = rng.choice(rets, size=n_bars, replace=True)
        sim_eq = np.cumprod(1.0 + sim_rets)
        sim_hwm = np.maximum.accumulate(sim_eq)
        sim_dd = (sim_eq - sim_hwm) / sim_hwm
        if sim_dd.min() <= ruin_threshold:
            ruin_count += 1

    return ruin_count / n_sims


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------


def generate_charts(
    instrument: str,
    tf: str,
    price: pd.Series,
    equity: pd.Series,
    drawdown: pd.Series,
    positions: pd.Series,
    stats: dict,
    annual_rets: dict,
) -> str:
    """Generate interactive 4-panel Plotly chart. Returns output path."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.30, 0.20, 0.12, 0.15, 0.23],
        subplot_titles=[
            f"{instrument} {tf} -- Price + ML Positions (OOS only)",
            f"Equity (Sharpe={stats['sharpe']:+.2f}, CAGR={stats['cagr']:.1%})",
            f"Drawdown (Max={stats['max_drawdown']:.1%})",
            f"Capital Invested (Avg={stats['avg_capital_invested']:.0%})",
            "Annual Returns",
        ],
    )

    # Panel 1: Price with position coloring
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

    # Long/short regions
    sig = positions.reindex(price.index).fillna(0)
    prev = sig.shift(1).fillna(0)
    long_entry = (sig == 1.0) & (prev != 1.0)
    short_entry = (sig == -1.0) & (prev != -1.0)

    le_idx = long_entry[long_entry].index
    fig.add_trace(
        go.Scatter(
            x=le_idx,
            y=price.reindex(le_idx).values,
            mode="markers",
            name=f"Long ({stats['n_long_trades']})",
            marker=dict(
                symbol="triangle-up", size=8, color="lime", line=dict(width=1, color="darkgreen")
            ),
        ),
        row=1,
        col=1,
    )

    se_idx = short_entry[short_entry].index
    fig.add_trace(
        go.Scatter(
            x=se_idx,
            y=price.reindex(se_idx).values,
            mode="markers",
            name=f"Short ({stats['n_short_trades']})",
            marker=dict(
                symbol="triangle-down", size=8, color="red", line=dict(width=1, color="darkred")
            ),
        ),
        row=1,
        col=1,
    )

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

    # Panel 4: Capital invested over time
    rolling_exp = stats["rolling_exposure"]
    # Position direction: +1 long, -1 short, 0 flat
    pos_aligned = positions.reindex(rolling_exp.index).fillna(0)

    # Shade long (green) vs short (red) exposure
    long_exp = rolling_exp.where(pos_aligned > 0, 0.0)
    short_exp = rolling_exp.where(pos_aligned < 0, 0.0)

    fig.add_trace(
        go.Scatter(
            x=long_exp.index,
            y=long_exp.values,
            mode="lines",
            name="Long Exposure",
            fill="tozeroy",
            line=dict(color="rgba(0,200,0,0.8)", width=1),
            fillcolor="rgba(0,200,0,0.3)",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=short_exp.index,
            y=short_exp.values,
            mode="lines",
            name="Short Exposure",
            fill="tozeroy",
            line=dict(color="rgba(200,0,0,0.8)", width=1),
            fillcolor="rgba(200,0,0,0.3)",
        ),
        row=4,
        col=1,
    )
    fig.add_hline(
        y=stats["avg_capital_invested"],
        line_dash="dash",
        line_color="yellow",
        annotation_text=f"Avg {stats['avg_capital_invested']:.0%}",
        row=4,
        col=1,
    )

    # Panel 5: Annual returns bar chart
    if annual_rets:
        years = list(annual_rets.keys())
        rets = list(annual_rets.values())
        colors = ["green" if r > 0 else "red" for r in rets]
        fig.add_trace(
            go.Bar(
                x=[str(y) for y in years],
                y=[r * 100 for r in rets],
                name="Annual %",
                marker_color=colors,
            ),
            row=5,
            col=1,
        )

    fig.update_layout(
        height=1400,
        width=1600,
        title_text=(
            f"ML Strategy -- {instrument} {tf} | "
            f"Return={stats['total_return']:.1%} | "
            f"Sharpe={stats['sharpe']:.2f} | "
            f"DD={stats['max_drawdown']:.1%} | "
            f"Trades={stats['n_trades']}"
        ),
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        template="plotly_dark",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", tickformat=".0%", row=3, col=1)
    fig.update_yaxes(title_text="Invested %", tickformat=".0%", range=[0, 1.05], row=4, col=1)
    fig.update_yaxes(title_text="Return %", row=5, col=1)

    out_path = REPORTS_DIR / f"ml_eval_{instrument}_{tf}.html"
    fig.write_html(str(out_path))
    return str(out_path)


# ---------------------------------------------------------------------------
# Main pipeline for one instrument
# ---------------------------------------------------------------------------


def run_eval(instrument: str, tf: str, asset_type: str) -> dict | None:
    """Run full WFO evaluation and compute comprehensive stats."""
    from xgboost import XGBClassifier

    print(f"\n{'=' * 70}")
    print(f"  FULL EVALUATION: {instrument} {tf}")
    print(f"{'=' * 70}")

    df = _load_ohlcv(instrument, tf)
    n_bars = len(df)
    bars_yr = _get_annual_bars(tf)
    cost = COST_BPS.get(asset_type, 1.0)

    print(f"  {n_bars:,} bars | {df.index[0].date()} to {df.index[-1].date()}")

    # 1. Features
    print("  Building features ...")
    df.attrs["instrument"] = instrument
    features = build_features(df, tf)

    # 2. Label sweep (regime+pullback v3)
    print(f"  Pre-computing labels ({len(LABEL_CONFIGS)} configs) ...")
    label_cache = []
    for lp in LABEL_CONFIGS:
        labels, _regime = compute_regime_pullback_labels(df, **lp)
        n_entries = int((labels != 0).sum())
        if n_entries >= 10:
            label_cache.append((lp, labels))
    print(f"  {len(label_cache)} configs produced >= 10 labels")

    # 3. Bar returns
    bar_returns = df["close"].pct_change().fillna(0.0)

    # 4. Valid feature mask
    mask_feat_valid = features.notna().all(axis=1)
    features_all = features[mask_feat_valid].copy()
    returns_all = bar_returns.reindex(features_all.index).fillna(0.0)

    # 5. WFO
    is_bars_n = IS_RATIO_BARS.get(tf, 504)
    oos_bars_n = OOS_RATIO_BARS.get(tf, 126)
    folds = walk_forward_splits(len(features_all), is_bars_n, oos_bars_n)
    if not folds:
        print("  [SKIP] Not enough data for WFO")
        return None

    print(f"  WFO: {len(folds)} folds (IS={is_bars_n}, OOS={oos_bars_n})")

    X_all = features_all.values
    all_idx_set = features_all.index

    all_oos_returns = []
    all_oos_positions = []

    for i, (is_idx, oos_idx) in enumerate(folds):
        # Label param sweep (pick best for this fold)
        is_mask = np.zeros(len(all_idx_set), dtype=bool)
        is_mask[is_idx] = True

        best_lp = None
        best_count = 0
        best_entry_positions = None
        best_entry_y = None

        for lp, labels in label_cache:
            lab_aligned = labels.reindex(all_idx_set).fillna(0).values
            entry_positions = np.where(lab_aligned != 0)[0]
            is_entries = entry_positions[is_mask[entry_positions]]
            if len(is_entries) < 20:
                continue
            y_is = (lab_aligned[is_entries] == 1).astype(int)
            minority_pct = min(y_is.mean(), 1 - y_is.mean())
            if minority_pct < 0.15:
                continue
            if len(is_entries) > best_count:
                best_count = len(is_entries)
                best_lp = lp
                best_entry_positions = entry_positions
                best_entry_y = (lab_aligned == 1).astype(int)

        if best_lp is None:
            continue

        # Train
        is_entries = best_entry_positions[is_mask[best_entry_positions]]
        X_is_entry = np.where(np.isinf(X_all[is_entries]), 0.0, X_all[is_entries])
        y_train = best_entry_y[is_entries]
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        spw = neg_count / pos_count if pos_count > 0 else 1.0

        params = {**XGB_PARAMS, "scale_pos_weight": spw, "eval_metric": "logloss"}
        model = XGBClassifier(**params)
        model.fit(X_is_entry, y_train)

        # Predict on ALL OOS bars
        X_oos_all = np.where(np.isinf(X_all[oos_idx]), 0.0, X_all[oos_idx])
        pred_proba = model.predict_proba(X_oos_all)[:, 1]
        position = _pred_to_position(pred_proba, threshold=SIGNAL_THRESHOLD)

        oos_returns = returns_all.iloc[oos_idx]
        oos_timestamps = features_all.index[oos_idx]

        positions_series = pd.Series(position, index=oos_timestamps)
        transitions = (positions_series != positions_series.shift(1).fillna(0.0)).astype(float)
        cost_per_bar = transitions * cost / 10_000
        strategy_rets = positions_series * oos_returns - cost_per_bar

        all_oos_returns.append(strategy_rets)
        all_oos_positions.append(positions_series)

        n_long = int((position == 1.0).sum())
        n_short = int((position == -1.0).sum())
        sh = (
            float(strategy_rets.mean() / strategy_rets.std() * np.sqrt(bars_yr))
            if strategy_rets.std() > 1e-10
            else 0.0
        )
        print(
            f"    Fold {i + 1}/{len(folds)}: Sharpe={sh:+.2f} L={n_long} S={n_short} ({best_count} train)"
        )

    if not all_oos_returns:
        print("  No valid folds produced.")
        return None

    # Stitch OOS
    stitched_rets = pd.concat(all_oos_returns).sort_index()
    stitched_rets = stitched_rets[~stitched_rets.index.duplicated(keep="last")]
    stitched_pos = pd.concat(all_oos_positions).sort_index()
    stitched_pos = stitched_pos[~stitched_pos.index.duplicated(keep="last")]

    # Compute full stats
    print("\n  Computing statistics ...")
    stats = compute_full_stats(stitched_rets, stitched_pos, bars_yr, cost)

    # Risk of ruin
    print("  Running risk-of-ruin Monte Carlo (5000 sims) ...")
    ror_50 = risk_of_ruin(stitched_rets, ruin_threshold=-0.50, n_sims=5000)
    ror_30 = risk_of_ruin(stitched_rets, ruin_threshold=-0.30, n_sims=5000)
    ror_20 = risk_of_ruin(stitched_rets, ruin_threshold=-0.20, n_sims=5000)
    stats["risk_of_ruin_50pct"] = ror_50
    stats["risk_of_ruin_30pct"] = ror_30
    stats["risk_of_ruin_20pct"] = ror_20

    # Equity + drawdown for charts
    equity = (1.0 + stitched_rets).cumprod()
    hwm = equity.cummax()
    drawdown = (equity - hwm) / hwm

    # Print report
    print(f"\n  {'=' * 60}")
    print(f"  RESULTS: {instrument} {tf}")
    print(f"  {'=' * 60}")
    print()
    print("  Performance")
    print(f"    Total Return:          {stats['total_return']:>+10.2%}")
    print(f"    CAGR:                  {stats['cagr']:>+10.2%}")
    print(f"    Sharpe Ratio:          {stats['sharpe']:>10.3f}")
    print(f"    Sortino Ratio:         {stats['sortino']:>10.3f}")
    print(f"    Calmar Ratio:          {stats['calmar']:>10.3f}")
    print()
    print("  Risk")
    print(f"    Max Drawdown:          {stats['max_drawdown']:>10.2%}")
    print(f"    Max DD Duration:       {stats['max_dd_duration_days']:>7} days")
    print(f"    Risk of Ruin (50%):    {stats['risk_of_ruin_50pct']:>10.1%}")
    print(f"    Risk of Ruin (30%):    {stats['risk_of_ruin_30pct']:>10.1%}")
    print(f"    Risk of Ruin (20%):    {stats['risk_of_ruin_20pct']:>10.1%}")
    print()
    print("  Trades")
    print(f"    Total Trades:          {stats['n_trades']:>10}")
    print(f"    Long Trades:           {stats['n_long_trades']:>10}")
    print(f"    Short Trades:          {stats['n_short_trades']:>10}")
    print(f"    Win Rate (all):        {stats['win_rate']:>10.1%}")
    print(f"    Win Rate (long):       {stats['long_win_rate']:>10.1%}")
    print(f"    Win Rate (short):      {stats['short_win_rate']:>10.1%}")
    print(f"    Avg Win:               {stats['avg_win']:>+10.4f}")
    print(f"    Avg Loss:              {stats['avg_loss']:>+10.4f}")
    print(f"    Win/Loss Ratio:        {stats['win_loss_ratio']:>10.2f}")
    print(f"    Profit Factor:         {stats['profit_factor']:>10.2f}")
    print()
    print("  Exposure / Capital Invested")
    print(f"    Avg Capital Invested:  {stats['avg_capital_invested']:>10.1%}")
    print(f"    % Time Long:           {stats['pct_long']:>10.1%}")
    print(f"    % Time Short:          {stats['pct_short']:>10.1%}")
    print(f"    % Time Flat:           {stats['pct_flat']:>10.1%}")
    print(f"    OOS Bars:              {stats['n_bars']:>10}")
    print(f"    OOS Years:             {stats['n_years']:>10.1f}")
    print(f"    Cost (bps/trade):      {stats['cost_bps']:>10.1f}")
    print()
    print("  Annual Returns")
    for year, ret in stats["annual_returns"].items():
        bar = "#" * int(abs(ret) * 200)
        sign = "+" if ret > 0 else ""
        print(f"    {year}:  {sign}{ret:.2%}  {bar}")

    # Generate charts
    print("\n  Generating charts ...")
    price = df["close"].loc[stitched_rets.index[0] : stitched_rets.index[-1]]
    chart_path = generate_charts(
        instrument,
        tf,
        price,
        equity,
        drawdown,
        stitched_pos,
        stats,
        stats["annual_returns"],
    )
    print(f"  Chart saved -> {chart_path}")

    stats["instrument"] = instrument
    stats["tf"] = tf
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Full ML pipeline evaluation.")
    parser.add_argument(
        "--instrument", default=None, help="Single instrument (default: QQQ, SPY, EUR_USD)."
    )
    args = parser.parse_args()

    if args.instrument:
        if args.instrument in EVAL_INSTRUMENTS:
            instruments = {args.instrument: EVAL_INSTRUMENTS[args.instrument]}
        else:
            instruments = {args.instrument: ("D", "index")}
    else:
        instruments = EVAL_INSTRUMENTS

    print()
    print("=" * 70)
    print("  ML FULL EVALUATION -- Regime+Pullback Classifier")
    print(f"  Instruments: {list(instruments.keys())}")
    print(f"  Threshold: P > {SIGNAL_THRESHOLD} = long, P < {1 - SIGNAL_THRESHOLD:.1f} = short")
    print("=" * 70)

    all_stats = []
    for inst, (tf, asset_type) in instruments.items():
        try:
            stats = run_eval(inst, tf, asset_type)
            if stats:
                all_stats.append(stats)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {inst}: {exc}")
        except Exception as exc:
            print(f"  [ERROR] {inst}: {exc}")
            import traceback

            traceback.print_exc()

    if not all_stats:
        print("\n  No results.")
        return

    # Cross-instrument summary
    print(f"\n{'=' * 70}")
    print("  CROSS-INSTRUMENT SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"\n  {'Inst':<10} {'Return':>8} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} "
        f"{'MaxDD':>7} {'WR%':>5} {'Trades':>6} {'PF':>5} {'Invested':>8} {'RoR50':>6}"
    )
    print("  " + "-" * 83)
    for s in all_stats:
        print(
            f"  {s['instrument']:<10} {s['total_return']:>+7.1%} {s['cagr']:>+6.1%} "
            f"{s['sharpe']:>+7.2f} {s['sortino']:>+8.2f} "
            f"{s['max_drawdown']:>7.1%} {s['win_rate']:>4.0%} "
            f"{s['n_trades']:>6} {s['profit_factor']:>5.2f} "
            f"{s['avg_capital_invested']:>7.0%} "
            f"{s['risk_of_ruin_50pct']:>5.1%}"
        )

    print(f"\n  Charts saved to: {REPORTS_DIR}/ml_eval_*.html")


if __name__ == "__main__":
    main()
