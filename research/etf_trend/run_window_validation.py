"""run_window_validation.py -- Fixed-params window validation across all market regimes.

Tests the locked config on every non-overlapping 3-year window across the full
price history WITHOUT re-optimising. Shows whether the strategy is regime-robust
(works in dot-com bust, GFC, COVID, 2022 bear) or is exploiting only recent QQQ
bull runs.

Signals are computed on the FULL series (no look-ahead bias: warmup uses past data
only). Each window's signals slice from the pre-computed full-series arrays.

Usage:
    uv run python research/etf_trend/run_window_validation.py --instrument QQQ
    uv run python research/etf_trend/run_window_validation.py --instrument QQQ --window-years 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly not installed.")
    sys.exit(1)

from research.etf_trend.run_portfolio import (  # noqa: E402
    FEES_BASE,
    INIT_CASH,
    SLIPPAGE,
    build_signals,
    load_config,
    load_data,
)
from research.etf_trend.visualise_positions import compute_position, equity_curve  # noqa: E402

# ── Per-window stats ──────────────────────────────────────────────────────────


def window_stats(close: pd.Series, in_pos: pd.Series, fees: float = FEES_BASE) -> dict:
    """Compute performance stats for a single window.

    Args:
        close: Close prices for the window.
        in_pos: Boolean position series (True = long).
        fees: Fee per side.

    Returns:
        Dict with sharpe, calmar, max_drawdown, total_return, n_trades, pct_in_market.
    """
    eq = equity_curve(close, in_pos, fees)
    daily_ret = close.pct_change().fillna(0)
    entering = in_pos & ~in_pos.shift(1).fillna(True)
    strat_ret = daily_ret * in_pos.astype(float)
    strat_ret -= entering.astype(float) * (fees + SLIPPAGE)
    exiting = (~in_pos) & in_pos.shift(1).fillna(False)
    strat_ret -= exiting.astype(float) * (fees + SLIPPAGE)

    total_ret = float(eq.iloc[-1] / INIT_CASH - 1)
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max
    max_dd = float(dd.min())

    std = strat_ret.std()
    sharpe = float(strat_ret.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0

    n_bars = len(close)
    ann_ret = (1 + total_ret) ** (365 / n_bars) - 1
    calmar = ann_ret / abs(max_dd) if max_dd < -1e-9 else 0.0

    transitions = in_pos.astype(int).diff().fillna(0)
    n_trades = int((transitions == 1).sum())
    pct_in = float(in_pos.sum()) / n_bars

    return {
        "total_return": round(total_ret, 3),
        "sharpe": round(sharpe, 3),
        "calmar": round(calmar, 3),
        "max_drawdown": round(max_dd, 3),
        "n_trades": n_trades,
        "pct_in_market": round(pct_in, 3),
    }


def bah_stats(close: pd.Series) -> dict:
    """Buy-and-hold stats (no fees)."""
    total_ret = float(close.iloc[-1] / close.iloc[0] - 1)
    daily_ret = close.pct_change().fillna(0)
    eq = INIT_CASH * (close / close.iloc[0])
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max
    max_dd = float(dd.min())
    std = daily_ret.std()
    sharpe = float(daily_ret.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0
    n_bars = len(close)
    ann_ret = (1 + total_ret) ** (365 / n_bars) - 1
    calmar = ann_ret / abs(max_dd) if max_dd < -1e-9 else 0.0
    return {
        "total_return": round(total_ret, 3),
        "sharpe": round(sharpe, 3),
        "calmar": round(calmar, 3),
        "max_drawdown": round(max_dd, 3),
        "n_trades": 1,
        "pct_in_market": 1.0,
    }


# ── HTML report ────────────────────────────────────────────────────────────────


REGIME_COLORS = [
    "#2196F3",
    "#4CAF50",
    "#FF5722",
    "#9C27B0",
    "#F44336",
    "#00BCD4",
    "#FF9800",
    "#607D8B",
    "#E91E63",
]


def build_chart(
    windows: list[dict],
    close: pd.Series,
    oos_split_date: pd.Timestamp,
    instrument: str,
) -> go.Figure:
    """Build multi-panel HTML chart: equity per window + position maps."""
    n = len(windows)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
        subplot_titles=[
            f"{instrument} Price + Window Equity Curves (normalised to 1.0 at window start)",
            "Strategy Position per Window  [1 = Long  |  0 = Flat]",
        ],
    )

    # Panel 1: price (faint background)
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            name=f"{instrument} Close",
            line=dict(color="rgba(150,150,150,0.4)", width=1),
            yaxis="y1",
        ),
        row=1,
        col=1,
    )

    for i, w in enumerate(windows):
        color = REGIME_COLORS[i % len(REGIME_COLORS)]
        win_close = close.loc[w["start"] : w["end"]]
        eq = INIT_CASH * (1 + w["strat_ret"]).cumprod()
        eq_norm = eq / eq.iloc[0]
        bah_norm = win_close / win_close.iloc[0]

        label = (
            f"{w['label']}  ret={w['stats']['total_return']:.0%}  "
            f"sh={w['stats']['sharpe']:.2f}  dd={w['stats']['max_drawdown']:.0%}"
        )

        fig.add_trace(
            go.Scatter(
                x=win_close.index,
                y=eq_norm.values,
                name=label,
                line=dict(color=color, width=2),
                yaxis="y1",
            ),
            row=1,
            col=1,
        )

        # faint B&H for this window
        fig.add_trace(
            go.Scatter(
                x=win_close.index,
                y=bah_norm.values,
                name=f"B&H {w['label']}",
                line=dict(color=color, width=0.8, dash="dot"),
                yaxis="y1",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Panel 2: position  (offset each window by i*0.1 for readability)
        pos_offset = (n - 1 - i) * 1.3
        pos_vals = w["in_pos"].astype(float) + pos_offset
        fig.add_trace(
            go.Scatter(
                x=win_close.index,
                y=pos_vals.values,
                name=w["label"],
                line=dict(color=color, width=1.5, shape="hv"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_annotation(
            x=win_close.index[0],
            y=pos_offset + 0.5,
            text=w["label"],
            showarrow=False,
            font=dict(size=9, color=color),
            xanchor="left",
            row=2,
            col=1,
        )

    # IS/OOS split line
    for row in (1, 2):
        fig.add_shape(
            type="line",
            x0=oos_split_date,
            x1=oos_split_date,
            y0=0,
            y1=1,
            xref="x",
            yref="y domain" if row == 1 else "y2 domain",
            line=dict(dash="dash", color="navy", width=1.5),
            row=row,
            col=1,
        )
    fig.add_annotation(
        x=oos_split_date,
        y=1.03,
        xref="x",
        yref="paper",
        text="◄ IS    OOS ►",
        showarrow=False,
        font=dict(size=10, color="navy"),
        xanchor="center",
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{instrument} Fixed-Params Window Validation — all market regimes<br>"
                f"<sup>Solid = strategy (normalised to 1.0 at window start)  |  "
                f"Dotted = B&H benchmark</sup>"
            ),
            font_size=14,
        ),
        height=800,
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01,
            font_size=10,
        ),
        margin=dict(t=110, b=40, l=70, r=220),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(title_text="Equity (×)", row=1, col=1)
    fig.update_yaxes(title_text="Window", row=2, col=1, showticklabels=False)
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-params window validation across all market regimes."
    )
    parser.add_argument("--instrument", default="QQQ", help="Symbol (default: QQQ)")
    parser.add_argument(
        "--window-years", type=int, default=3, help="Window size in years (default: 3)"
    )
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()
    window_years = args.window_years

    print("=" * 64)
    print("  ETF Trend -- Fixed-Params Window Validation")
    print("=" * 64)
    print(f"  Instrument: {instrument}  |  Window: {window_years} years")

    config = load_config(inst_lower)
    df = load_data(instrument)
    close = df["close"]

    # IS/OOS split (for reference annotation only -- no re-optimisation)
    split = int(len(close) * 0.70)
    oos_split_date = close.index[split]

    print(
        f"  Full series: {close.index[0].date()} to {close.index[-1].date()}  ({len(close)} bars)"
    )
    print(f"  OOS starts:  {oos_split_date.date()}  (70/30 reference split)")
    print(
        f"  Config:      entry={config.get('entry_mode')}  "
        f"exit={config['exit_mode']}  sizing={config['sizing_mode']}"
    )

    # ── Build signals on FULL series (fixed params, no re-opt) ──────────────
    entries_full, exits_full, _decel_sh, _sizes = build_signals(close, df, config)

    # Forward-simulate position once on full series
    in_pos_full = compute_position(entries_full, exits_full)

    # ── Slice into non-overlapping windows ──────────────────────────────────
    window_bars = window_years * 252
    rows: list[dict] = []
    windows: list[dict] = []

    idx = close.index
    start_i = 0

    while start_i < len(close):
        # Include remaining bars as a final window if > half a window size
        remaining = len(close) - start_i
        if remaining < window_bars and remaining < window_bars // 2:
            break
        end_i = min(start_i + window_bars, len(close)) - 1
        win_start = idx[start_i]
        win_end = idx[end_i]
        win_close = close.iloc[start_i : end_i + 1]
        win_pos = in_pos_full.iloc[start_i : end_i + 1]

        strat = window_stats(win_close, win_pos)
        bah = bah_stats(win_close)

        # Whether this window overlaps OOS region
        is_oos = win_end >= oos_split_date
        is_is = win_start < oos_split_date
        region = "IS+OOS" if (is_is and is_oos) else ("OOS" if is_oos else "IS ")

        label = f"{win_start.year}-{win_end.year}"

        # strat_ret for equity curve
        daily_ret = win_close.pct_change().fillna(0)
        entering = win_pos & ~win_pos.shift(1).fillna(True)
        exiting = (~win_pos) & win_pos.shift(1).fillna(False)
        strat_ret = daily_ret * win_pos.astype(float)
        strat_ret -= entering.astype(float) * (FEES_BASE + SLIPPAGE)
        strat_ret -= exiting.astype(float) * (FEES_BASE + SLIPPAGE)

        windows.append(
            {
                "label": label,
                "start": win_start,
                "end": win_end,
                "stats": strat,
                "in_pos": win_pos,
                "strat_ret": strat_ret,
            }
        )
        rows.append(
            {
                "window": label,
                "region": region,
                "start": str(win_start.date()),
                "end": str(win_end.date()),
                "n_bars": len(win_close),
                **{f"strat_{k}": v for k, v in strat.items()},
                **{f"bah_{k}": v for k, v in bah.items()},
                "beats_bah_return": strat["total_return"] > bah["total_return"],
                "beats_bah_sharpe": strat["sharpe"] > bah["sharpe"],
                "beats_bah_dd": abs(strat["max_drawdown"]) < abs(bah["max_drawdown"]),
            }
        )
        start_i += window_bars

    df_results = pd.DataFrame(rows)

    # ── Print scoreboard ────────────────────────────────────────────────────
    print(
        f"\n  {'Window':<12}{'Rgn':<7}{'Return':>8}{'Sharpe':>8}"
        f"{'MaxDD':>8}{'Calmar':>8}{'Trades':>7}{'%InMkt':>8}"
        f"  vs B&H (ret/sh/dd)"
    )
    print("  " + "-" * 80)
    for r in rows:
        beats = (
            ("R" if r["beats_bah_return"] else ".")
            + ("S" if r["beats_bah_sharpe"] else ".")
            + ("D" if r["beats_bah_dd"] else ".")
        )
        print(
            f"  {r['window']:<12}{r['region']:<7}"
            f"{r['strat_total_return']:>7.0%} "
            f"{r['strat_sharpe']:>7.2f} "
            f"{r['strat_max_drawdown']:>7.1%} "
            f"{r['strat_calmar']:>7.2f} "
            f"{r['strat_n_trades']:>6d} "
            f"{r['strat_pct_in_market']:>7.0%}  "
            f"  {beats}"
        )

    n_beats_2of3 = sum(
        (r["beats_bah_return"] + r["beats_bah_sharpe"] + r["beats_bah_dd"]) >= 2 for r in rows
    )
    n_total = len(rows)
    print(
        f"\n  Windows beating B&H on 2/3: {n_beats_2of3} / {n_total} ({n_beats_2of3 / n_total:.0%})"
    )
    pos_count = sum(r["strat_total_return"] > 0 for r in rows)
    print(f"  Windows profitable:         {pos_count} / {n_total} ({pos_count / n_total:.0%})")

    # ── Save CSV ────────────────────────────────────────────────────────────
    csv_path = REPORTS_DIR / f"etf_trend_{inst_lower}_window_validation.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path.relative_to(PROJECT_ROOT)}")

    # ── Build and save HTML chart ────────────────────────────────────────────
    fig = build_chart(windows, close, oos_split_date, instrument)
    html_path = REPORTS_DIR / f"etf_trend_{inst_lower}_window_validation.html"
    fig.write_html(str(html_path))
    print(f"  Chart:     {html_path.relative_to(PROJECT_ROOT)}")
    print("  Open in browser to explore interactively.")
    print("=" * 64)


if __name__ == "__main__":
    main()
