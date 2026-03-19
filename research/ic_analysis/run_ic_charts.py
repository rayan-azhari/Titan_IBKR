"""run_ic_charts.py
------------------

Generate an interactive 4-panel Plotly HTML analysis chart for the IC MTF backtest.

  Panel 1: EUR/USD price (H1) with long/short position shading and entry markers (OOS)
  Panel 2: Composite z-score with ±threshold bands and entry zones
  Panel 3: OOS equity curves — long, short, combined (log scale)
  Panel 4: WFO fold-by-fold OOS Sharpe bar chart

Outputs:
  .tmp/reports/ic_mtf_{slug}_analysis.html

Usage:
  uv run python research/ic_analysis/run_ic_charts.py
  uv run python research/ic_analysis/run_ic_charts.py --instrument EUR_USD --threshold 0.75
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import vectorbt as vbt  # noqa: E402

from research.ic_analysis.run_ic_backtest import (  # noqa: E402
    DEFAULT_RISK_PCT,
    DEFAULT_SIGNALS,
    DEFAULT_SLIPPAGE_PIPS,
    DEFAULT_STOP_ATR,
    DEFAULT_TFS,
    INIT_CASH,
    IS_RATIO,
    PIP_SIZE,
    SPREAD_DEFAULTS,
    _build_and_align,
    build_composite,
    build_size_array,
    zscore_normalise,
)

REPORTS_DIR = ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG      = "#0d1117"
C_PAPER   = "#161b22"
C_GRID    = "#21262d"
C_TEXT    = "#c9d1d9"
C_PRICE   = "#8b949e"
C_LONG    = "#3fb950"    # green
C_SHORT   = "#f85149"    # red
C_SIGNAL  = "#a371f7"    # purple
C_THRESH  = "#e3b341"    # amber
C_COMB    = "#e6edf3"    # white


# ── Helpers ───────────────────────────────────────────────────────────────────

def _equity_curve(pf) -> pd.Series:
    r = pf.returns().fillna(0.0)
    return (1 + r).cumprod() - 1


def _trade_markers(pf, direction: str) -> tuple[list, list, list]:
    """Return (timestamps, prices, hover_texts) for entry fills."""
    if pf.trades.count() == 0:
        return [], [], []

    trades = pf.trades.records_readable
    close_idx = pf.close.index

    ts_list, px_list, txt_list = [], [], []
    for _, row in trades.iterrows():
        entry_col = next((c for c in row.index if "entry" in c.lower() and "idx" in c.lower()), None)
        if entry_col is None:
            entry_col = next((c for c in row.index if "Entry" in c), None)
        if entry_col is None:
            continue
        entry_ts = row[entry_col]
        if entry_ts not in close_idx:
            continue
        px = float(pf.close.loc[entry_ts])
        ret_col = next((c for c in row.index if "return" in c.lower() or "Return" in c), "Return")
        ret = float(row.get(ret_col, 0.0)) * 100
        txt = (
            f"{'Long' if direction == 'long' else 'Short'} entry<br>"
            f"Time: {entry_ts}<br>"
            f"Price: {px:.5f}<br>"
            f"Return: {ret:+.2f}%"
        )
        ts_list.append(entry_ts)
        px_list.append(px)
        txt_list.append(txt)

    return ts_list, px_list, txt_list


def _build_position_zones(sig: pd.Series, threshold: float, direction: str) -> list[dict]:
    """Return list of shape dicts for vrect position shading."""
    mask = sig > threshold if direction == "long" else sig < -threshold
    shapes = []
    in_zone = False
    start = None
    for ts, v in mask.items():
        if v and not in_zone:
            start = ts
            in_zone = True
        elif not v and in_zone:
            shapes.append({"start": start, "end": ts})
            in_zone = False
    if in_zone and start is not None:
        shapes.append({"start": start, "end": mask.index[-1]})
    return shapes


# ── Main ──────────────────────────────────────────────────────────────────────

def build_chart(instrument: str, threshold: float) -> Path:
    slug = instrument.lower()
    pip_size = PIP_SIZE.get(instrument, 0.0001)
    spread   = SPREAD_DEFAULTS.get(instrument, 0.00005)
    slippage = DEFAULT_SLIPPAGE_PIPS * pip_size

    # ── 1. Signals + composite ────────────────────────────────────────────────
    print(f"Building composite for {instrument}...")
    tfs = list(DEFAULT_TFS)
    tf_signals, base_index, base_df = _build_and_align(instrument, tfs, base_tf=tfs[-1])
    base_close = base_df["close"]

    n    = len(base_index)
    is_n = int(n * IS_RATIO)
    is_mask  = pd.Series(False, index=base_index)
    is_mask.iloc[:is_n] = True
    oos_mask = ~is_mask

    size        = build_size_array(base_df, base_close, DEFAULT_RISK_PCT, DEFAULT_STOP_ATR)
    composite   = build_composite(tf_signals, base_close, tfs, list(DEFAULT_SIGNALS), is_mask)
    composite_z = zscore_normalise(composite, is_mask)

    oos_close = base_close[oos_mask]
    oos_z     = composite_z[oos_mask]
    oos_size  = size[oos_mask]
    dates     = oos_close.index

    # ── 2. VBT OOS portfolios ─────────────────────────────────────────────────
    print(f"Running OOS backtest at threshold=±{threshold}z...")
    med_close = float(oos_close.median()) or 1.0
    vbt_fees  = spread / med_close
    vbt_slip  = slippage / med_close
    sig       = oos_z.shift(1).fillna(0.0)
    size_arr  = oos_size.reindex(oos_close.index).fillna(0.0).values

    pf_long = vbt.Portfolio.from_signals(
        oos_close,
        entries=sig > threshold,
        exits=sig <= 0.0,
        size=size_arr,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq="h",
    )
    pf_short = vbt.Portfolio.from_signals(
        oos_close,
        entries=pd.Series(False, index=oos_close.index),
        exits=pd.Series(False, index=oos_close.index),
        short_entries=sig < -threshold,
        short_exits=sig >= 0.0,
        size=size_arr,
        size_type="percent",
        init_cash=INIT_CASH,
        fees=vbt_fees,
        slippage=vbt_slip,
        freq="h",
    )

    n_long  = pf_long.trades.count()
    n_short = pf_short.trades.count()
    sh_long  = pf_long.sharpe_ratio()
    sh_short = pf_short.sharpe_ratio()
    sh_comb  = (sh_long + sh_short) / 2
    print(f"  Long trades: {n_long} | Short trades: {n_short}")
    print(f"  Sharpe — Long: {sh_long:.2f} | Short: {sh_short:.2f} | Combined: {sh_comb:.2f}")

    # ── 3. Equity curves (log-scale — convert to +1 base) ────────────────────
    eq_long  = (_equity_curve(pf_long)  + 1) * 100   # → 100 = start
    eq_short = (_equity_curve(pf_short) + 1) * 100
    eq_comb  = (eq_long + eq_short) / 2

    fin_long  = float(eq_long.iloc[-1])
    fin_short = float(eq_short.iloc[-1])
    fin_comb  = float(eq_comb.iloc[-1])

    # ── 4. Trade markers ──────────────────────────────────────────────────────
    le_ts, le_px, le_txt = _trade_markers(pf_long,  "long")
    se_ts, se_px, se_txt = _trade_markers(pf_short, "short")

    # ── 5. WFO ────────────────────────────────────────────────────────────────
    wfo_path = REPORTS_DIR / f"wfo_{slug}.csv"
    wfo_df   = pd.read_csv(wfo_path) if wfo_path.exists() else None

    # ── 6. Build figure ───────────────────────────────────────────────────────
    n_rows       = 4 if wfo_df is not None else 3
    row_heights  = [0.38, 0.22, 0.28, 0.12] if n_rows == 4 else [0.40, 0.25, 0.35]
    subplot_titles = (
        "EUR/USD H1 — OOS Price & Trades",
        "Composite Z-Score",
        "Equity Curve (log scale, base=100)",
        "WFO OOS Sharpe by Fold",
    )[:n_rows]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # ── Panel 1: Price ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=oos_close.values,
        mode="lines",
        line=dict(color=C_PRICE, width=0.8),
        name="EUR/USD",
        hovertemplate="%{x}<br>Price: %{y:.5f}<extra></extra>",
    ), row=1, col=1)

    # Position shading (vrect via shapes — max 500 zones for performance)
    long_zones  = _build_position_zones(sig, threshold,  "long")[:500]
    short_zones = _build_position_zones(sig, threshold, "short")[:500]

    for z in long_zones:
        fig.add_vrect(
            x0=z["start"], x1=z["end"],
            fillcolor=C_LONG, opacity=0.08, line_width=0,
            row=1, col=1,
        )
    for z in short_zones:
        fig.add_vrect(
            x0=z["start"], x1=z["end"],
            fillcolor=C_SHORT, opacity=0.08, line_width=0,
            row=1, col=1,
        )

    # Entry markers (cap at 800 for performance)
    if le_ts:
        fig.add_trace(go.Scatter(
            x=le_ts[:800], y=le_px[:800],
            mode="markers",
            marker=dict(symbol="triangle-up", color=C_LONG, size=6, opacity=0.75),
            name=f"Long entry ({n_long})",
            text=le_txt[:800],
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    if se_ts:
        fig.add_trace(go.Scatter(
            x=se_ts[:800], y=se_px[:800],
            mode="markers",
            marker=dict(symbol="triangle-down", color=C_SHORT, size=6, opacity=0.75),
            name=f"Short entry ({n_short})",
            text=se_txt[:800],
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    # ── Panel 2: Z-score ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=oos_z.values,
        mode="lines",
        line=dict(color=C_SIGNAL, width=0.7),
        name="Composite z",
        hovertemplate="%{x}<br>z = %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # Threshold bands
    for sign, label in [(threshold, f"+{threshold}z (long)"), (-threshold, f"−{threshold}z (short)")]:
        fig.add_hline(
            y=sign, line_dash="dash", line_color=C_THRESH, line_width=1,
            annotation_text=label, annotation_font_color=C_THRESH,
            annotation_font_size=10,
            row=2, col=1,
        )
    fig.add_hline(y=0, line_dash="dot", line_color=C_TEXT, line_width=0.5,
                  opacity=0.3, row=2, col=1)

    # Fill above/below thresholds
    z_vals = oos_z.values
    dates_list = list(dates)
    fig.add_trace(go.Scatter(
        x=dates_list + dates_list[::-1],
        y=np.where(z_vals > threshold,  z_vals, threshold).tolist()
          + [threshold] * len(dates_list),
        fill="toself", fillcolor=C_LONG, opacity=0.15,
        line=dict(width=0), showlegend=False,
        hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=dates_list + dates_list[::-1],
        y=np.where(z_vals < -threshold, z_vals, -threshold).tolist()
          + [-threshold] * len(dates_list),
        fill="toself", fillcolor=C_SHORT, opacity=0.15,
        line=dict(width=0), showlegend=False,
        hoverinfo="skip",
    ), row=2, col=1)

    # ── Panel 3: Equity (log scale) ───────────────────────────────────────────
    for eq, color, label, fin in [
        (eq_long,  C_LONG,  f"Long  ({fin_long:.0f}x)", fin_long),
        (eq_short, C_SHORT, f"Short ({fin_short:.0f}x)", fin_short),
        (eq_comb,  C_COMB,  f"Combined ({fin_comb:.0f}x)", fin_comb),
    ]:
        fig.add_trace(go.Scatter(
            x=dates,
            y=eq.values,
            mode="lines",
            line=dict(color=color, width=1.5 if color == C_COMB else 1.0),
            name=label,
            hovertemplate="%{x}<br>Value: %{y:.1f}x initial<extra></extra>",
        ), row=3, col=1)

    fig.update_yaxes(type="log", tickformat=".0f", row=3, col=1,
                     title_text="Portfolio value (log, start=100)")

    # ── Panel 4: WFO bars ─────────────────────────────────────────────────────
    if wfo_df is not None:
        oos_sharpes = wfo_df["oos_sharpe"].values
        fold_labels = [f"Fold {i+1}" for i in range(len(oos_sharpes))]
        bar_colors  = [C_LONG if s >= 0 else C_SHORT for s in oos_sharpes]

        fig.add_trace(go.Bar(
            x=fold_labels,
            y=oos_sharpes,
            marker_color=bar_colors,
            marker_line_width=0,
            opacity=0.85,
            name="OOS Sharpe",
            hovertemplate="Fold %{x}<br>OOS Sharpe: %{y:.2f}<extra></extra>",
        ), row=4, col=1)

        fig.add_hline(y=0, line_dash="dot", line_color=C_TEXT, line_width=0.5,
                      opacity=0.4, row=4, col=1)
        fig.add_hline(y=1, line_dash="dash", line_color=C_THRESH, line_width=0.8,
                      opacity=0.6, row=4, col=1)

        pos_pct = (oos_sharpes > 0).mean() * 100
        fig.add_annotation(
            text=f"{len(oos_sharpes)} folds | {pos_pct:.0f}% positive | mean Sharpe: {oos_sharpes.mean():.2f}",
            xref="paper", yref="paper",
            x=0.99, y=0.005, xanchor="right", yanchor="bottom",
            showarrow=False, font=dict(size=11, color=C_TEXT),
            row=4, col=1,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    oos_start = dates[0].strftime("%Y-%m")
    oos_end   = dates[-1].strftime("%Y-%m")

    fig.update_layout(
        title=dict(
            text=(
                f"IC MTF — {instrument.replace('_', '/')}   "
                f"OOS: {oos_start} → {oos_end}   "
                f"Threshold: ±{threshold}z   "
                f"Sharpe: {sh_long:.2f}L / {sh_short:.2f}S / {sh_comb:.2f}C"
            ),
            font=dict(size=14, color=C_TEXT),
            x=0.5,
        ),
        height=1300,
        paper_bgcolor=C_BG,
        plot_bgcolor=C_PAPER,
        font=dict(color=C_TEXT, size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        hovermode="x unified",
        margin=dict(l=60, r=30, t=80, b=40),
    )

    # Axes styling
    axis_style = dict(
        gridcolor=C_GRID,
        gridwidth=0.5,
        zeroline=False,
        showgrid=True,
        tickfont=dict(size=10, color=C_TEXT),
        title_font=dict(color=C_TEXT),
    )
    for i in range(1, n_rows + 1):
        fig.update_xaxes(**axis_style, row=i, col=1)
        fig.update_yaxes(**axis_style, row=i, col=1)

    # Subtitle colours
    for ann in fig.layout.annotations:
        ann.font.color = C_TEXT
        ann.font.size  = 11

    out_path = REPORTS_DIR / f"ic_mtf_{slug}_analysis.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"\nInteractive chart saved: {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="IC MTF interactive analysis chart")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--threshold",  type=float, default=0.75)
    args = parser.parse_args()
    build_chart(args.instrument, args.threshold)


if __name__ == "__main__":
    main()
