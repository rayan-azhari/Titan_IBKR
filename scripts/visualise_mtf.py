"""visualise_mtf.py — Comprehensive 6-panel MTF strategy visualization report.

Produces a single interactive Plotly HTML report containing:
  Panel 1 — H4 price chart with trade entry/exit markers (colour-coded by direction + win/loss)
  Panel 2 — Equity curve with IS/OOS region shading and swap-adjusted overlay
  Panel 3 — Drawdown % (filled area, top-3 worst valleys annotated)
  Panel 4 — Rolling 12-month Sharpe (bar chart, green ≥1, amber 0-1, red <0)
  Panel 5 — P&L distribution histograms (long | short, side-by-side)
  Panel 6 — Monthly return calendar heatmap (year × month, RdYlGn)

Usage:
    uv run python scripts/visualise_mtf.py --pair EUR_USD
    uv run python scripts/visualise_mtf.py --pair GBP_USD --scenario stop
    uv run python scripts/visualise_mtf.py --pair EUR_USD --oos-only
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.")
    sys.exit(1)

from titan.models.spread import build_spread_series  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Constants (mirrors run_portfolio.py)
# ─────────────────────────────────────────────────────────────────────

INIT_CASH = 100_000.0
RISK_PER_TRADE = 0.01
ATR_PERIOD = 14
MAX_LEVERAGE = 5.0
H4_BARS_PER_YEAR = 2190
IS_FRAC = 0.70  # 70 % in-sample

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ─────────────────────────────────────────────────────────────────────
# Data + signal helpers (replicated from run_portfolio.py)
# ─────────────────────────────────────────────────────────────────────


def load_data(pair: str, gran: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{pair}_{gran}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def load_config(pair_lower: str) -> dict:
    for name in [f"mtf_{pair_lower}.toml", "mtf.toml"]:
        p = PROJECT_ROOT / "config" / name
        if p.exists():
            with open(p, "rb") as f:
                return tomllib.load(f)
    raise FileNotFoundError(f"No MTF config found for {pair_lower}")


def load_swap_pct(pair: str) -> float:
    p = PROJECT_ROOT / "config" / "spread.toml"
    try:
        with open(p, "rb") as f:
            return float(tomllib.load(f).get("swap", {}).get(pair, 0.015))
    except FileNotFoundError:
        return 0.015


def compute_ma(close: pd.Series, period: int, ma_type: str) -> pd.Series:
    if ma_type == "WMA":
        w = np.arange(1, period + 1, dtype=float)
        return close.rolling(period).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)
    if ma_type == "EMA":
        return close.ewm(span=period, adjust=False).mean()
    return close.rolling(period).mean()  # SMA default


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))


def compute_tf_signal(
    close: pd.Series, fast_ma: int, slow_ma: int, rsi_period: int, ma_type: str
) -> pd.Series:
    fast = compute_ma(close, fast_ma, ma_type)
    slow = compute_ma(close, slow_ma, ma_type)
    rsi = compute_rsi(close, rsi_period)
    ma_sig = pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index)
    rsi_sig = pd.Series(np.where(rsi > 50, 0.5, -0.5), index=close.index)
    return ma_sig + rsi_sig


def compute_confluence(pair: str, cfg: dict) -> tuple[pd.Series, pd.DataFrame]:
    weights = cfg.get("weights", {})
    ma_type = cfg.get("ma_type", "SMA")
    primary_df = load_data(pair, "H4")
    if primary_df is None:
        raise ValueError("H4 data missing — run download_data_mtf.py first")
    primary_index = primary_df.index
    signals_sum = pd.Series(0.0, index=primary_index)
    total_weight = 0.0
    for tf in ["H1", "H4", "D", "W"]:
        w = weights.get(tf, 0.0)
        if w == 0:
            continue
        tf_cfg = cfg.get(tf, {})
        df = load_data(pair, tf)
        if df is None:
            continue
        sig = compute_tf_signal(
            df["close"],
            tf_cfg.get("fast_ma", 20),
            tf_cfg.get("slow_ma", 50),
            tf_cfg.get("rsi_period", 14),
            ma_type,
        )
        signals_sum += sig.reindex(primary_index, method="ffill") * w
        total_weight += w
    if 0 < total_weight < 1.0:
        signals_sum /= total_weight
    return signals_sum, primary_df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat([abs(h - lo), abs(h - c.shift()), abs(lo - c.shift())], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def run_backtest(pair: str, cfg: dict, use_stop: bool) -> tuple:
    """Run VBT portfolio, return (pf, h4_df, swap_pct)."""
    threshold = cfg.get("confirmation_threshold", 0.10)
    stop_mult = float(cfg.get("atr_stop_mult", 2.0))
    swap_pct = load_swap_pct(pair)

    confluence, h4 = compute_confluence(pair, cfg)
    close, high, low = h4["close"], h4["high"], h4["low"]
    atr = compute_atr(h4, ATR_PERIOD)
    conf_sh = confluence.shift(1).fillna(0.0)

    risk_amt = INIT_CASH * RISK_PER_TRADE
    raw_units = risk_amt / (stop_mult * atr)
    max_units = (INIT_CASH * MAX_LEVERAGE) / close
    size = np.minimum(raw_units, max_units).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    avg_spread = float(build_spread_series(h4, pair).mean())

    kwargs: dict = dict(
        close=close,
        high=high,
        low=low,
        entries=conf_sh >= threshold,
        short_entries=conf_sh <= -threshold,
        exits=conf_sh < 0,
        short_exits=conf_sh > 0,
        size=size,
        size_type="amount",
        init_cash=INIT_CASH,
        fees=avg_spread,
        freq="4h",
    )
    if use_stop:
        sl = (stop_mult * atr / close).replace(0, np.nan).fillna(0.0)
        kwargs["sl_stop"] = sl
        kwargs["sl_trail"] = True

    return vbt.Portfolio.from_signals(**kwargs), h4, swap_pct


# ─────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    """Return the first matching column from df, or None."""
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def rolling_sharpe_series(equity: pd.Series, months: int = 12) -> pd.Series:
    monthly = equity.resample("ME").last().pct_change().dropna()
    rs = monthly.rolling(months).apply(
        lambda x: (x.mean() / x.std() * np.sqrt(12)) if x.std() > 0 else 0.0,
        raw=True,
    )
    return rs.reindex(equity.index, method="ffill")


def monthly_heatmap_data(equity: pd.Series) -> tuple[list[str], list[str], list[list]]:
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    df = monthly.to_frame("ret")
    df["year"] = df.index.year.astype(str)
    df["month"] = df.index.month
    years = sorted(df["year"].unique())
    z: list[list] = []
    for yr in years:
        row: list = []
        for mo in range(1, 13):
            s = df[(df["year"] == yr) & (df["month"] == mo)]["ret"]
            row.append(float(s.iloc[0]) if len(s) else None)
        z.append(row)
    return years, MONTH_NAMES, z


def _vrect_is(fig: go.Figure, x0, x1) -> None:
    """Shade the IS period on all 4 time-series panels (rows 1–4)."""
    for row in range(1, 5):
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="rgba(120,120,120,0.07)",
            line_width=0,
            row=row,
            col=1,
        )
    # OOS boundary line on panel 1 (no annotation_text — Plotly Timestamp bug)
    fig.add_vline(
        x=x1,
        line_dash="dash",
        line_color="rgba(200,200,200,0.35)",
        line_width=1,
        row=1,
        col=1,
    )


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MTF 6-panel trade visualization report.")
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument(
        "--scenario",
        choices=["signal", "stop"],
        default="signal",
        help="signal = signal-only exit (default); stop = trailing ATR stop",
    )
    parser.add_argument(
        "--oos-only",
        action="store_true",
        help="Restrict time-series panels 1–4 to OOS window only",
    )
    args = parser.parse_args()

    pair = args.pair.upper()
    pair_lower = pair.lower().replace("_", "")
    use_stop = args.scenario == "stop"

    print(f"MTF Visualise: {pair}  scenario={args.scenario}")
    cfg = load_config(pair_lower)
    print("  Running backtest...")
    pf, h4, swap_pct = run_backtest(pair, cfg, use_stop=use_stop)

    # ── IS/OOS boundary ──────────────────────────────────────────────
    n_is = int(len(h4) * IS_FRAC)
    is_end_ts = h4.index[n_is - 1]
    oos_start_ts = h4.index[n_is]

    # ── Full-history series ──────────────────────────────────────────
    equity_full = pf.value()
    dd_full = pf.drawdown() * 100  # negative %, e.g. -5.0 for 5 % DD
    swap_cum = (pf.asset_value().abs() * (swap_pct / H4_BARS_PER_YEAR)).cumsum()
    equity_adj_full = equity_full - swap_cum
    trades_df = pf.trades.records_readable.copy()

    # ── Optionally restrict to OOS ───────────────────────────────────
    if args.oos_only:
        mask = h4.index >= oos_start_ts
        h4 = h4[mask]
        equity = equity_full[equity_full.index >= oos_start_ts]
        equity_adj = equity_adj_full[equity_adj_full.index >= oos_start_ts]
        dd = dd_full[dd_full.index >= oos_start_ts]
    else:
        equity = equity_full
        equity_adj = equity_adj_full
        dd = dd_full

    # ── Summary stats ────────────────────────────────────────────────
    sharpe = float(pf.sharpe_ratio())
    maxdd = float(pf.max_drawdown())
    total_ret = float(pf.total_return())
    n_trades = int(pf.trades.count())
    wr = float(pf.trades.win_rate())
    swap_total = float(swap_cum.iloc[-1])

    # ── Trade marker data ────────────────────────────────────────────
    e_ts = _col(trades_df, "Entry Time", "Entry Timestamp")
    x_ts = _col(trades_df, "Exit Time", "Exit Timestamp")
    e_px = _col(trades_df, "Entry Price", "entry_price")
    x_px = _col(trades_df, "Exit Price", "exit_price")
    pnl = _col(trades_df, "PnL", "pnl")
    direction = _col(trades_df, "Direction", "Side", "direction")
    has_trades = e_ts is not None and e_px is not None and pnl is not None and direction is not None

    if has_trades and args.oos_only:
        tmask = e_ts >= oos_start_ts
        trades_df = trades_df[tmask]
        e_ts = _col(trades_df, "Entry Time", "Entry Timestamp")
        x_ts = _col(trades_df, "Exit Time", "Exit Timestamp")
        e_px = _col(trades_df, "Entry Price", "entry_price")
        x_px = _col(trades_df, "Exit Price", "exit_price")
        pnl = _col(trades_df, "PnL", "pnl")
        direction = _col(trades_df, "Direction", "Side", "direction")

    # ── Build figure ─────────────────────────────────────────────────
    print("  Building chart...")
    fig = make_subplots(
        rows=6,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        row_heights=[0.28, 0.18, 0.11, 0.11, 0.16, 0.16],
        shared_xaxes=False,
        vertical_spacing=0.04,
        subplot_titles=[
            (f"H4 Price — {pair.replace('_', '/')}  (▲ long, ▽ short | green=win, red=loss)"),
            (
                f"Equity Curve  |  Sharpe={sharpe:.3f}  "
                f"MaxDD={maxdd:.2%}  Return={total_ret:.2%}  "
                f"Trades={n_trades}  WR={wr:.1%}"
            ),
            "Drawdown (%)",
            "Rolling 12-Month Sharpe",
            "Long Trades — P&L",
            "Short Trades — P&L",
            "Monthly Returns (%)",
        ],
    )

    # Link time-series panels 1–4 to share x-axis (rows share xaxis1)
    fig.update_layout(
        xaxis2={"matches": "x"},
        xaxis3={"matches": "x"},
        xaxis4={"matches": "x"},
    )

    # IS/OOS shading
    if not args.oos_only:
        _vrect_is(fig, h4.index[0], is_end_ts)

    # ── Panel 1: H4 price + trade markers ────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=h4.index,
            y=h4["close"],
            name="H4 Close",
            line={"color": "rgba(180,180,180,0.4)", "width": 1},
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if has_trades:
        is_long = direction.astype(str).str.lower().isin(["long", "1", "1.0"])
        is_win = pnl > 0
        groups = [
            (is_long & is_win, "Long Win", "triangle-up", "#00cc66", 9),
            (is_long & ~is_win, "Long Loss", "triangle-up", "#ff3333", 9),
            (~is_long & is_win, "Short Win", "triangle-down", "#00ccff", 9),
            (~is_long & ~is_win, "Short Loss", "triangle-down", "#ff8800", 9),
        ]
        for mask, gname, symbol, color, sz in groups:
            sub = trades_df[mask]
            if len(sub) == 0:
                continue
            ge_ts = _col(sub, "Entry Time", "Entry Timestamp")
            ge_px = _col(sub, "Entry Price", "entry_price")
            gpnl = _col(sub, "PnL", "pnl")
            if ge_ts is None or ge_px is None:
                continue
            htxt = [f"{gname}<br>PnL: ${p:+.2f}" for p in (gpnl if gpnl is not None else [])]
            fig.add_trace(
                go.Scatter(
                    x=ge_ts,
                    y=ge_px,
                    mode="markers",
                    name=gname,
                    marker={
                        "symbol": symbol,
                        "color": color,
                        "size": sz,
                        "line": {"width": 1, "color": "rgba(255,255,255,0.3)"},
                    },
                    hovertext=htxt if htxt else None,
                    hoverinfo="text+x" if htxt else "skip",
                ),
                row=1,
                col=1,
            )

        if x_ts is not None and x_px is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_ts,
                    y=x_px,
                    mode="markers",
                    name="Exit",
                    marker={"symbol": "x", "color": "rgba(200,200,200,0.3)", "size": 5},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    # ── Panel 2: Equity curve ─────────────────────────────────────────
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity, name="Equity", line={"color": "#4da6ff", "width": 2}),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity_adj.index,
            y=equity_adj,
            name="Equity (swap-adj)",
            line={"color": "#99ccff", "dash": "dot", "width": 1.5},
        ),
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=equity.index[-1],
        y=float(equity.iloc[-1]),
        text=f"${equity.iloc[-1]:,.0f}",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-25,
        font={"color": "#4da6ff", "size": 11},
        row=2,
        col=1,
    )
    fig.add_hline(y=INIT_CASH, line_dash="dot", line_color="rgba(200,200,200,0.25)", row=2, col=1)

    # ── Panel 3: Drawdown ─────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd,
            name="Drawdown %",
            fill="tozeroy",
            line={"color": "#ff4444", "width": 1},
            fillcolor="rgba(255,68,68,0.12)",
        ),
        row=3,
        col=1,
    )
    # Annotate top-3 deepest points
    worst3 = dd.nsmallest(3)
    for ts, val in worst3.items():
        fig.add_annotation(
            x=ts,
            y=float(val),
            text=f"{val:.1f}%",
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-20,
            font={"color": "#ff8888", "size": 10},
            row=3,
            col=1,
        )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(200,200,200,0.2)", row=3, col=1)

    # ── Panel 4: Rolling 12-month Sharpe ─────────────────────────────
    rs = rolling_sharpe_series(pf.value())
    rs_plot = rs[equity.index].dropna()
    bar_colors = ["#00cc66" if v >= 1.0 else "#ff6666" if v < 0 else "#ffaa00" for v in rs_plot]
    fig.add_trace(
        go.Bar(
            x=rs_plot.index,
            y=rs_plot,
            name="Rolling Sharpe 12m",
            marker_color=bar_colors,
            showlegend=False,
        ),
        row=4,
        col=1,
    )
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="rgba(0,200,100,0.45)",
        annotation_text="1.0",
        row=4,
        col=1,
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(200,200,200,0.2)", row=4, col=1)

    # ── Panel 5: P&L histograms ───────────────────────────────────────
    if has_trades:
        all_pnl = _col(pf.trades.records_readable, "PnL", "pnl")
        all_dir = _col(pf.trades.records_readable, "Direction", "Side", "direction")
        if all_pnl is not None and all_dir is not None:
            all_long = all_dir.astype(str).str.lower().isin(["long", "1", "1.0"])
            for pnl_data, gname, color, r, c in [
                (all_pnl[all_long], "Long P&L", "#00cc66", 5, 1),
                (all_pnl[~all_long], "Short P&L", "#ff4444", 5, 2),
            ]:
                pnl_clean = pnl_data.dropna()
                if len(pnl_clean) == 0:
                    continue
                fig.add_trace(
                    go.Histogram(
                        x=pnl_clean, name=gname, marker_color=color, opacity=0.7, nbinsx=40
                    ),
                    row=r,
                    col=c,
                )
                mu = float(pnl_clean.mean())
                med = float(pnl_clean.median())
                fig.add_vline(
                    x=mu,
                    line_dash="dash",
                    line_color="white",
                    annotation_text=f"μ=${mu:.0f}",
                    row=r,
                    col=c,
                )
                fig.add_vline(
                    x=med,
                    line_dash="dot",
                    line_color="yellow",
                    annotation_text=f"med=${med:.0f}",
                    row=r,
                    col=c,
                )

    # ── Panel 6: Monthly return heatmap ──────────────────────────────
    years, months, z_mat = monthly_heatmap_data(pf.value())
    text_mat = [[f"{v:.1f}%" if v is not None else "" for v in row] for row in z_mat]
    fig.add_trace(
        go.Heatmap(
            x=months,
            y=[str(y) for y in years],
            z=z_mat,
            colorscale="RdYlGn",
            zmid=0,
            text=text_mat,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar={"title": "%", "len": 0.14, "y": 0.03},
            name="Monthly Return",
        ),
        row=6,
        col=1,
    )

    # ── Layout ───────────────────────────────────────────────────────
    scenario_label = (
        "Signal-Only Exit"
        if not use_stop
        else f"Trailing Stop ({cfg.get('atr_stop_mult', 2.0)}x ATR)"
    )
    window_label = "OOS Only" if args.oos_only else "Full History (IS + OOS)"
    title = (
        f"MTF Strategy Report — {pair.replace('_', '/')}  |  "
        f"{scenario_label}  |  {window_label}<br>"
        f"<sup>Sharpe={sharpe:.3f}  MaxDD={maxdd:.2%}  "
        f"Return={total_ret:.2%}  Trades={n_trades}  WR={wr:.1%}  "
        f"Swap≈${swap_total:,.0f}</sup>"
    )
    fig.update_layout(
        title=title,
        height=1600,
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        legend={"orientation": "h", "y": -0.03, "x": 0},
        margin={"t": 90, "b": 60},
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
    fig.update_yaxes(title_text="DD (%)", row=3, col=1)
    fig.update_yaxes(title_text="Sharpe", row=4, col=1)
    fig.update_xaxes(title_text="P&L ($)", row=5, col=1)
    fig.update_xaxes(title_text="P&L ($)", row=5, col=2)

    # ── Save ─────────────────────────────────────────────────────────
    slug = "stop" if use_stop else "signal"
    out_path = REPORTS_DIR / f"mtf_visualise_{pair_lower}_{slug}.html"
    fig.write_html(str(out_path))
    print(f"  Report saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
