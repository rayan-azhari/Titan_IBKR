"""visualise_positions.py -- Chart QQQ in/out-of-market periods.

Three-panel Plotly chart:
  Top    : QQQ close + SMA 200 + SMA 100, green/grey shading for long/flat
  Middle : Decel composite signal (entry filter)
  Bottom : Rolling 1-year drawdown of the strategy equity vs QQQ B&H

IS/OOS split marked with a dashed vertical line.

Usage:
    uv run python research/etf_trend/visualise_positions.py --instrument QQQ
"""

import argparse
import sys
from pathlib import Path

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
from research.etf_trend.run_stage3_exits import compute_decel_composite, compute_ma  # noqa: E402


def compute_position(entries: pd.Series, exits: pd.Series) -> pd.Series:
    """Forward-simulate a long-only position from entry/exit signals.

    Returns a boolean Series: True = holding long, False = flat.
    entries/exits are already .shift(1) (act-at-open convention).
    """
    pos = pd.Series(False, index=entries.index)
    held = False
    for i in range(len(entries)):
        if entries.iloc[i] and not held:
            held = True
        elif exits.iloc[i] and held:
            held = False
        pos.iloc[i] = held
    return pos


def equity_curve(close: pd.Series, in_pos: pd.Series, fees: float = FEES_BASE) -> pd.Series:
    """Compute daily equity curve for the binary strategy."""
    daily_ret = close.pct_change().fillna(0)
    # transaction cost on entry/exit transitions
    entering = in_pos & ~in_pos.shift(1).fillna(True)
    exiting = (~in_pos) & in_pos.shift(1).fillna(False)
    strat_ret = daily_ret * in_pos.astype(float)
    strat_ret -= entering.astype(float) * (fees + SLIPPAGE)
    strat_ret -= exiting.astype(float) * (fees + SLIPPAGE)
    return INIT_CASH * (1 + strat_ret).cumprod()


def bah_equity(close: pd.Series) -> pd.Series:
    """Buy-and-hold equity curve (no fees)."""
    return INIT_CASH * (close / close.iloc[0])


def add_shading(
    fig: go.Figure,
    dates: pd.DatetimeIndex,
    mask: pd.Series,
    color: str,
    row: int,
    opacity: float = 0.15,
) -> None:
    """Add vertical shaded regions to a subplot where mask is True."""
    in_block = False
    block_start = None
    for i, (date, val) in enumerate(zip(dates, mask)):
        if val and not in_block:
            block_start = date
            in_block = True
        elif not val and in_block:
            fig.add_vrect(
                x0=block_start,
                x1=date,
                fillcolor=color,
                opacity=opacity,
                line_width=0,
                row=row,
                col=1,
            )
            in_block = False
    if in_block:
        fig.add_vrect(
            x0=block_start,
            x1=dates[-1],
            fillcolor=color,
            opacity=opacity,
            line_width=0,
            row=row,
            col=1,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise QQQ in/out-of-market periods.")
    parser.add_argument("--instrument", default="QQQ", help="Symbol (default: QQQ)")
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()

    config = load_config(inst_lower)
    df = load_data(instrument)
    close = df["close"]
    split = int(len(close) * 0.70)

    # -- Build signals and position ----------------------------------------
    entries, exits, decel_sh, _ = build_signals(close, df, config)
    in_pos = compute_position(entries, exits)

    # -- Indicators --------------------------------------------------------
    ma_type = config["ma_type"]
    slow_ma = compute_ma(close, config["slow_ma"], ma_type)
    fast_ma = compute_ma(close, config["fast_reentry_ma"], ma_type)
    decel_signals = config.get("decel_signals", [])
    decel_params = {k: config[k] for k in ("d_pct_smooth", "rv_window", "macd_fast") if k in config}
    decel = compute_decel_composite(close, df, slow_ma, decel_signals, decel_params)

    # -- Equity curves (OOS only for comparison) ---------------------------
    eq_strat = equity_curve(close, in_pos)
    eq_bah = bah_equity(close)

    # Rolling 1-year drawdown
    def rolling_dd(eq: pd.Series) -> pd.Series:
        pk = eq.cummax()
        return (eq - pk) / pk

    # -- Stats -------------------------------------------------------------
    oos_pos = in_pos.iloc[split:]
    n_oos_bars = len(oos_pos)
    n_in = int(oos_pos.sum())
    pct_in = n_in / n_oos_bars
    pct_out = 1 - pct_in

    # Count distinct long periods in OOS
    transitions = oos_pos.astype(int).diff().fillna(0)
    n_entries_oos = int((transitions == 1).sum())

    # IS stats
    is_pos = in_pos.iloc[:split]
    n_is_in = int(is_pos.sum())
    pct_is_in = n_is_in / split

    oos_date_start = close.index[split].date()
    oos_date_end = close.index[-1].date()
    is_date_start = close.index[0].date()
    is_date_end = close.index[split - 1].date()

    print(f"  IS  ({is_date_start} – {is_date_end}): {pct_is_in:.0%} in market")
    print(
        f"  OOS ({oos_date_start} – {oos_date_end}): {pct_in:.0%} in market  "
        f"({n_entries_oos} long periods)"
    )
    print(f"  OOS bars flat (cash): {n_oos_bars - n_in} / {n_oos_bars} ({pct_out:.0%})")

    # -- Equity curves normalised to 1.0 at OOS start ---------------------
    eq_strat_norm = eq_strat / eq_strat.iloc[split]
    eq_bah_norm = eq_bah / eq_bah.iloc[split]

    # -- Build chart (4 panels) --------------------------------------------
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.38, 0.10, 0.32, 0.20],
        vertical_spacing=0.03,
        subplot_titles=[
            f"{instrument} Price — Long (green) / Flat (grey)",
            "Position  [1 = Long QQQ  |  0 = Flat / Cash]",
            "Equity Curve  (normalised to 1.0 at OOS start)",
            "Decel Composite  [Mode D exits when < threshold]",
        ],
    )

    dates = close.index

    # Panel 1 shading: green = long, grey = flat
    add_shading(fig, dates, in_pos, "green", row=1, opacity=0.12)
    add_shading(fig, dates, ~in_pos, "grey", row=1, opacity=0.08)

    # IS/OOS split line across all panels
    split_ts = close.index[split]
    for row in (1, 2, 3, 4):
        fig.add_shape(
            type="line",
            x0=split_ts,
            x1=split_ts,
            y0=0,
            y1=1,
            xref="x",
            yref=f"y{row} domain" if row > 1 else "y domain",
            line=dict(dash="dash", color="navy", width=1.5),
            row=row,
            col=1,
        )
    fig.add_annotation(
        x=split_ts,
        y=1.03,
        xref="x",
        yref="paper",
        text="◄ IS    OOS ►",
        showarrow=False,
        font=dict(size=10, color="navy"),
        xanchor="center",
    )

    # --- Panel 1: Price + MAs + entry/exit markers ------------------------
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=close.values,
            name=instrument,
            line=dict(color="#1f77b4", width=1.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=slow_ma.values,
            name=f"SMA {config['slow_ma']}",
            line=dict(color="black", width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=fast_ma.values,
            name=f"SMA {config['fast_reentry_ma']} (re-entry)",
            line=dict(color="darkorange", width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )

    # Entry/exit markers — only show actual position changes
    entry_dates = dates[entries & ~in_pos.shift(1).fillna(True)]
    exit_dates = dates[exits & in_pos.shift(1).fillna(False)]
    if len(entry_dates):
        fig.add_trace(
            go.Scatter(
                x=entry_dates,
                y=close.loc[entry_dates].values * 0.985,
                mode="markers",
                name="Entry (buy)",
                marker=dict(symbol="triangle-up", size=9, color="green"),
            ),
            row=1,
            col=1,
        )
    if len(exit_dates):
        fig.add_trace(
            go.Scatter(
                x=exit_dates,
                y=close.loc[exit_dates].values * 1.015,
                mode="markers",
                name="Exit (sell)",
                marker=dict(symbol="triangle-down", size=9, color="red"),
            ),
            row=1,
            col=1,
        )

    # --- Panel 2: Binary position 1/0 step chart --------------------------
    pos_vals = in_pos.astype(int).values
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pos_vals,
            name="Position (1=long)",
            line=dict(color="green", width=1.5, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(0,160,0,0.15)",
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        tickvals=[0, 1],
        ticktext=["Flat", "Long"],
        range=[-0.1, 1.3],
        row=2,
        col=1,
    )

    # --- Panel 3: Equity curves -------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=eq_strat_norm.values,
            name="Strategy equity",
            line=dict(color="steelblue", width=2),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=eq_bah_norm.values,
            name="Buy & Hold equity",
            line=dict(color="tomato", width=1.5, dash="dot"),
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=1.0, line_color="grey", line_width=0.8, line_dash="dot", row=3, col=1)
    # Shade strategy above/below B&H
    add_shading(fig, dates, eq_strat_norm > eq_bah_norm, "steelblue", row=3, opacity=0.07)
    add_shading(fig, dates, eq_strat_norm < eq_bah_norm, "tomato", row=3, opacity=0.07)

    # --- Panel 4: Decel composite -----------------------------------------
    add_shading(fig, dates, decel >= 0, "green", row=4, opacity=0.08)
    add_shading(fig, dates, decel < 0, "red", row=4, opacity=0.08)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=decel.values,
            name="Decel composite",
            line=dict(color="purple", width=1.2),
        ),
        row=4,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=4, col=1)
    fig.add_hline(
        y=config["exit_decel_thresh"],
        line_dash="dot",
        line_color="red",
        line_width=1,
        row=4,
        col=1,
    )

    # --- Layout -----------------------------------------------------------
    stats_text = (
        f"OOS {oos_date_start}–{oos_date_end}  |  "
        f"In market: {pct_in:.0%} ({n_in} bars, {n_entries_oos} in/out cycles)  |  "
        f"Flat/cash: {pct_out:.0%} ({n_oos_bars - n_in} bars)"
    )
    fig.update_layout(
        title=dict(
            text=f"{instrument} ETF Trend Strategy — Full Breakdown<br><sup>{stats_text}</sup>",
            font_size=15,
        ),
        height=980,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(t=120, b=40, l=70, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="In/Out", row=2, col=1)
    fig.update_yaxes(title_text="Equity (×)", row=3, col=1)
    fig.update_yaxes(title_text="Decel", row=4, col=1)

    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        text=f"OOS: {pct_in:.0%} long  {pct_out:.0%} cash",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="grey",
        font=dict(size=11),
    )

    out_path = REPORTS_DIR / f"etf_trend_{inst_lower}_positions.html"
    fig.write_html(str(out_path))
    print(f"\n  Chart saved: {out_path.relative_to(PROJECT_ROOT)}")
    print("  Open in browser to explore interactively.")


if __name__ == "__main__":
    main()
