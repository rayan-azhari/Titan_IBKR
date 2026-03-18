"""run_multi_portfolio.py -- Multi-Asset ETF Trend Portfolio.

Loads locked configs for multiple instruments, builds signals independently,
combines into an equal-weight portfolio, and compares vs equal-weight buy-and-hold.

Signal-instrument separation is handled transparently: if a config contains
`signal_instrument`, signals are computed on that instrument and P&L on the
execution instrument (e.g. TQQQ with QQQ signals).

Usage:
    uv run python research/etf_trend/run_multi_portfolio.py --instruments TQQQ IWB
    uv run python research/etf_trend/run_multi_portfolio.py --instruments TQQQ IWB QQQ SPY
"""

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEES = 0.001
SLIPPAGE = 0.0005
INIT_CASH = 10_000.0


# ── Data and config loading ─────────────────────────────────────────────────


def load_config(instrument: str) -> dict:
    config_path = PROJECT_ROOT / "config" / f"etf_trend_{instrument.lower()}.toml"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found. Run pipeline first.")
        sys.exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_df(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_D.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df.sort_index().dropna(subset=["close"])


def load_instrument(config: dict) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Return (sig_close, exec_close, sig_df) from config.

    When config contains `signal_instrument`, signals are built on that
    instrument and P&L computed on the execution instrument.
    """
    sig_str = config.get("signal_instrument", "")
    sig_sym = sig_str.split(".")[0] if sig_str else ""
    exec_sym = config["instrument"].split(".")[0]

    exec_df = load_df(exec_sym)
    if sig_sym and sig_sym.upper() != exec_sym.upper():
        sig_df = load_df(sig_sym)
        common = sig_df.index.intersection(exec_df.index)
        sig_df = sig_df.loc[common]
        exec_df = exec_df.loc[common]
        sig_close = sig_df["close"]
        exec_close = exec_df["close"]
    else:
        sig_df = exec_df
        sig_close = exec_df["close"]
        exec_close = sig_close

    return sig_close, exec_close, sig_df


# ── Signal computation ──────────────────────────────────────────────────────


def compute_strategy_returns(
    sig_close: pd.Series,
    exec_close: pd.Series,
    sig_df: pd.DataFrame,
    config: dict,
) -> pd.Series:
    """Return daily strategy return series (includes fees/slippage at entries/exits).

    Signals are computed on sig_close (e.g. QQQ).
    P&L is computed on exec_close (e.g. TQQQ or same instrument).
    """
    from research.etf_trend.run_stage3_exits import (  # noqa: E402
        apply_sma_confirmation,
        build_entry_signal,
        compute_decel_composite,
        compute_ma,
    )

    ma_type = config["ma_type"]
    slow_ma_p = config["slow_ma"]
    decel_signals = config.get("decel_signals", [])
    decel_params = {
        k: config[k]
        for k in ("d_pct_smooth", "rv_window", "macd_fast")
        if k in config
    }
    entry_mode = config.get("entry_mode", "decel_positive")
    fast_reentry_ma_p = config.get("fast_reentry_ma")
    exit_mode = config["exit_mode"]
    exit_decel_thresh = config["exit_decel_thresh"]
    exit_confirm_days = int(config.get("exit_confirm_days", 1))
    decel_confirm_days = int(config.get("decel_confirm_days", 1))

    slow_ma = compute_ma(sig_close, slow_ma_p, ma_type)
    decel = compute_decel_composite(sig_close, sig_df, slow_ma, decel_signals, decel_params)
    fast_ma = compute_ma(sig_close, int(fast_reentry_ma_p), ma_type) if fast_reentry_ma_p else None
    entry_signal = build_entry_signal(sig_close, slow_ma, decel, entry_mode, fast_ma)

    below_slow = apply_sma_confirmation(~(sig_close > slow_ma), exit_confirm_days)
    if exit_mode == "A":
        exit_signal = below_slow
    elif exit_mode == "C":
        exit_signal = apply_sma_confirmation(decel < exit_decel_thresh, decel_confirm_days)
    else:  # D: confirmed SMA break OR sustained decel collapse
        decel_exit = apply_sma_confirmation(decel < exit_decel_thresh, decel_confirm_days)
        exit_signal = below_slow | decel_exit

    entries = entry_signal.shift(1).fillna(False)
    exits = exit_signal.shift(1).fillna(False)

    # Stateful position: 1 = long, 0 = flat (same logic as VBT's from_signals)
    in_pos = False
    position: list[float] = []
    for i in range(len(entries)):
        if not in_pos and entries.iloc[i]:
            in_pos = True
        elif in_pos and exits.iloc[i]:
            in_pos = False
        position.append(float(in_pos))
    pos = pd.Series(position, index=entries.index)

    entering = (pos == 1) & (pos.shift(1).fillna(0) == 0)
    exiting = (pos == 0) & (pos.shift(1).fillna(0) == 1)

    daily_rets = exec_close.pct_change().fillna(0)
    strat_rets = daily_rets * pos
    strat_rets -= entering.astype(float) * (FEES + SLIPPAGE)
    strat_rets -= exiting.astype(float) * (FEES + SLIPPAGE)
    return strat_rets


# ── Stats ───────────────────────────────────────────────────────────────────


def equity_curve(rets: pd.Series) -> pd.Series:
    return INIT_CASH * (1 + rets).cumprod()


def compute_stats(rets: pd.Series) -> dict:
    eq = equity_curve(rets)
    total_ret = float(eq.iloc[-1] / INIT_CASH - 1)
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max
    max_dd = float(dd.min())
    std = rets.std()
    sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0
    n_bars = len(rets)
    ann_ret = (1 + total_ret) ** (365 / n_bars) - 1
    calmar = ann_ret / abs(max_dd) if max_dd < -1e-9 else 0.0
    return {
        "total_return": round(total_ret, 3),
        "sharpe": round(sharpe, 3),
        "calmar": round(calmar, 3),
        "max_drawdown": round(max_dd, 3),
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF Trend Multi-Asset Portfolio.")
    parser.add_argument(
        "--instruments", nargs="+", default=["TQQQ", "IWB"],
        help="Instruments to combine (must have locked configs). Default: TQQQ IWB",
    )
    args = parser.parse_args()
    instruments = [i.upper() for i in args.instruments]
    n = len(instruments)
    weight = 1.0 / n

    print("=" * 60)
    print("  ETF Trend -- Multi-Asset Equal-Weight Portfolio")
    print("=" * 60)
    print(f"  Instruments:  {instruments}")
    print(f"  Allocation:   {weight:.0%} each (equal-weight)")

    # ── Build per-instrument strategy returns (full history) ─────────────
    all_sig_close: dict[str, pd.Series] = {}
    all_exec_close: dict[str, pd.Series] = {}
    all_strat_rets: dict[str, pd.Series] = {}

    for inst in instruments:
        config = load_config(inst)
        sig_str = config.get("signal_instrument", "")
        sig_label = sig_str.split(".")[0] if sig_str else inst
        label = f"sig={sig_label}" if sig_label != inst else "self"
        print(f"\n  [{inst}] Loading data and computing signals ({label}) ...")

        sig_close, exec_close, sig_df = load_instrument(config)
        all_sig_close[inst] = sig_close
        all_exec_close[inst] = exec_close
        strat_rets = compute_strategy_returns(sig_close, exec_close, sig_df, config)
        all_strat_rets[inst] = strat_rets

    # ── Determine portfolio start: latest OOS start across all instruments ─
    oos_starts = []
    for inst in instruments:
        sig_close = all_sig_close[inst]
        split = int(len(sig_close) * 0.70)
        oos_starts.append(sig_close.index[split])
    portfolio_start = max(oos_starts)

    print(f"\n  Portfolio period: {portfolio_start.date()} to "
          f"{min(s.index[-1] for s in all_strat_rets.values()).date()}")

    # ── Slice to portfolio period and build common index ──────────────────
    common_strat: dict[str, pd.Series] = {}
    common_bah: dict[str, pd.Series] = {}
    for inst in instruments:
        sr = all_strat_rets[inst]
        ec = all_exec_close[inst]
        sr_oos = sr[sr.index >= portfolio_start]
        ec_oos = ec[ec.index >= portfolio_start]
        common_strat[inst] = sr_oos
        common_bah[inst] = ec_oos.pct_change().fillna(0)

    # Intersect dates across all instruments
    common_idx = common_strat[instruments[0]].index
    for inst in instruments[1:]:
        common_idx = common_idx.intersection(common_strat[inst].index)

    # ── Portfolio daily returns (equal-weight combination) ─────────────────
    port_strat_rets = sum(  # type: ignore[assignment]
        common_strat[inst].reindex(common_idx).fillna(0) * weight
        for inst in instruments
    )
    port_bah_rets = sum(  # type: ignore[assignment]
        common_bah[inst].reindex(common_idx).fillna(0) * weight
        for inst in instruments
    )

    # ── Per-instrument stats in common period ─────────────────────────────
    print(f"\n  -- Per-instrument OOS stats "
          f"({portfolio_start.date()} to {common_idx[-1].date()}) --")
    rows = []
    for inst in instruments:
        s_strat = compute_stats(common_strat[inst].reindex(common_idx).fillna(0))
        s_bah = compute_stats(common_bah[inst].reindex(common_idx).fillna(0))
        print(f"  [{inst}] Strat  Ret={s_strat['total_return']:.1%}  "
              f"Sharpe={s_strat['sharpe']:.3f}  MaxDD={s_strat['max_drawdown']:.1%}  "
              f"Calmar={s_strat['calmar']:.3f}")
        print(f"  [{inst}] B&H    Ret={s_bah['total_return']:.1%}  "
              f"Sharpe={s_bah['sharpe']:.3f}  MaxDD={s_bah['max_drawdown']:.1%}  "
              f"Calmar={s_bah['calmar']:.3f}")
        rows.append({**{"instrument": inst, "type": "strategy"}, **s_strat})
        rows.append({**{"instrument": inst, "type": "bah"}, **s_bah})

    # ── Portfolio summary ─────────────────────────────────────────────────
    port_stats = compute_stats(port_strat_rets)
    bah_stats = compute_stats(port_bah_rets)
    rows.append({**{"instrument": "Portfolio", "type": "strategy"}, **port_stats})
    rows.append({**{"instrument": "Portfolio", "type": "bah"}, **bah_stats})

    print("\n  -- Portfolio vs Equal-Weight B&H ---------------")
    print(f"  Portfolio: Ret={port_stats['total_return']:.1%}  "
          f"Sharpe={port_stats['sharpe']:.3f}  "
          f"MaxDD={port_stats['max_drawdown']:.1%}  "
          f"Calmar={port_stats['calmar']:.3f}")
    print(f"  B&H {'+'.join(instruments)}: Ret={bah_stats['total_return']:.1%}  "
          f"Sharpe={bah_stats['sharpe']:.3f}  "
          f"MaxDD={bah_stats['max_drawdown']:.1%}  "
          f"Calmar={bah_stats['calmar']:.3f}")

    better_sharpe = port_stats["sharpe"] >= bah_stats["sharpe"]
    better_dd = abs(port_stats["max_drawdown"]) <= abs(bah_stats["max_drawdown"])
    better_calmar = port_stats["calmar"] >= bah_stats["calmar"]
    n_better = sum([better_sharpe, better_dd, better_calmar])
    print(f"\n  vs B&H: Sharpe {'BETTER' if better_sharpe else 'WORSE'}  "
          f"MaxDD {'BETTER' if better_dd else 'WORSE'}  "
          f"Calmar {'BETTER' if better_calmar else 'WORSE'}  "
          f"-- {n_better}/3 metrics")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = REPORTS_DIR / f"etf_trend_multi_portfolio_{'_'.join(instruments)}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path.relative_to(PROJECT_ROOT)}")

    # ── Interactive Plotly chart ──────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            subplot_titles=["Equity (normalised to 1.0)", "Portfolio Drawdown"],
        )

        COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # Per-instrument strategy equity curves
        for i, inst in enumerate(instruments):
            eq = equity_curve(common_strat[inst].reindex(common_idx).fillna(0)) / INIT_CASH
            fig.add_trace(
                go.Scatter(x=eq.index, y=eq.values, name=f"{inst} Strat",
                           mode="lines", line=dict(color=COLORS[i % len(COLORS)], dash="dot")),
                row=1, col=1,
            )

        # Portfolio strategy equity
        port_eq = equity_curve(port_strat_rets) / INIT_CASH
        fig.add_trace(
            go.Scatter(x=port_eq.index, y=port_eq.values,
                       name=f"Portfolio ({'+'.join(instruments)})",
                       mode="lines", line=dict(width=3, color="black")),
            row=1, col=1,
        )

        # B&H equity
        bah_eq = equity_curve(port_bah_rets) / INIT_CASH
        fig.add_trace(
            go.Scatter(x=bah_eq.index, y=bah_eq.values,
                       name=f"B&H ({'+'.join(instruments)})",
                       mode="lines", line=dict(width=2, dash="dash", color="gray")),
            row=1, col=1,
        )

        # Portfolio drawdown
        port_dd = (port_eq - port_eq.cummax()) / port_eq.cummax()
        fig.add_trace(
            go.Scatter(x=port_dd.index, y=port_dd.values, name="Portfolio DD",
                       mode="lines", fill="tozeroy",
                       line=dict(color="crimson", width=1)),
            row=2, col=1,
        )

        # Stats annotation
        annotation_text = (
            f"Portfolio: Ret={port_stats['total_return']:.1%}  "
            f"Sharpe={port_stats['sharpe']:.2f}  "
            f"MaxDD={port_stats['max_drawdown']:.1%}  "
            f"Calmar={port_stats['calmar']:.2f}<br>"
            f"B&H {'+'.join(instruments)}: Ret={bah_stats['total_return']:.1%}  "
            f"Sharpe={bah_stats['sharpe']:.2f}  "
            f"MaxDD={bah_stats['max_drawdown']:.1%}  "
            f"Calmar={bah_stats['calmar']:.2f}"
        )
        fig.add_annotation(
            text=annotation_text, xref="paper", yref="paper",
            x=0.0, y=-0.12, showarrow=False, align="left",
            font=dict(size=11), xanchor="left",
        )

        fig.update_layout(
            title=(
                f"Multi-Asset ETF Trend Portfolio — {' + '.join(instruments)} "
                f"(OOS from {portfolio_start.date()})"
            ),
            legend_title="Strategy",
            height=750,
            margin=dict(b=100),
        )
        report_path = REPORTS_DIR / f"etf_trend_multi_portfolio_{'_'.join(instruments)}.html"
        fig.write_html(str(report_path))
        print(f"  Chart saved: {report_path.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"  [Chart skipped: {e}]")

    print("\n  [PASS] Multi-asset portfolio complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
