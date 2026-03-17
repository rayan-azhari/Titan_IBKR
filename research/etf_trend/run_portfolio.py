"""run_portfolio.py — Stage 5: Full Friction Portfolio Simulation.

Runs the complete strategy with all friction (fees, slippage, next-bar execution)
and compares against a buy-and-hold SPY baseline.

Compares:
  1. Dynamic decel sizing + full friction
  2. Binary sizing + full friction
  3. Strategy signal-only (no ATR hard stop)
  4. Strategy with ATR hard stop active

Baseline benchmark: buy-and-hold (no fees, fully invested).

The strategy must outperform B&H on at least TWO of THREE:
  - OOS Sharpe
  - Max Drawdown (lower is better)
  - Calmar Ratio

Outputs an HTML comparison report to .tmp/reports/.

Usage:
    uv run python research/etf_trend/run_portfolio.py --instrument SPY
"""

import argparse
import sys
import tomllib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.")
    sys.exit(1)

from research.etf_trend.run_stage2_decel import compute_ma  # noqa: E402
from research.etf_trend.run_stage3_exits import compute_decel_composite  # noqa: E402
from research.etf_trend.run_stage4_sizing import (  # noqa: E402
    compute_position_sizes,
)
from titan.strategies.ml.features import atr as compute_atr  # noqa: E402

FEES_BASE = 0.001
SLIPPAGE = 0.0005
INIT_CASH = 10_000.0

# Transaction cost stress test levels
FEE_SCENARIOS = {"base (0.10%)": 0.001, "2× (0.20%)": 0.002, "3× (0.30%)": 0.003}


def load_config(instrument: str) -> dict:
    config_path = PROJECT_ROOT / "config" / f"etf_trend_{instrument.lower()}.toml"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found. Run run_stage4_sizing.py first.")
        sys.exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_data(instrument: str) -> pd.DataFrame:
    path = DATA_DIR / f"{instrument}_D.parquet"
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


def build_signals(
    close: pd.Series,
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Build entries, exits, decel, and sizes from config.

    Returns:
        Tuple of (entries, exits, decel_shifted, sizes_shifted).
    """
    ma_type = config["ma_type"]
    fast_ma_p = config["fast_ma"]
    slow_ma_p = config["slow_ma"]
    decel_signals = config.get("decel_signals", [])
    exit_mode = config["exit_mode"]
    exit_decel_thresh = config["exit_decel_thresh"]
    sizing_mode = config["sizing_mode"]
    vol_target = config["vol_target"]
    vol_window = config["vol_window"]
    atr_stop_mult = config["atr_stop_mult"]

    fast_ma = compute_ma(close, fast_ma_p, ma_type)
    slow_ma = compute_ma(close, slow_ma_p, ma_type)
    atr_ser = compute_atr(df)
    decel = compute_decel_composite(close, df, slow_ma, decel_signals)

    in_regime = (close > fast_ma) & (close > slow_ma)
    entry_signal = in_regime & (decel >= 0)

    if exit_mode == "A":
        exit_signal = ~in_regime
    elif exit_mode == "B":
        exit_signal = close < fast_ma
    elif exit_mode == "C":
        exit_signal = decel < exit_decel_thresh
    else:
        exit_signal = (~in_regime) | (decel < exit_decel_thresh)

    entries = entry_signal.shift(1).fillna(False)
    exits = exit_signal.shift(1).fillna(False)
    decel_sh = decel.shift(1).fillna(0)
    sizes = (
        compute_position_sizes(
            close, decel_sh, sizing_mode, vol_target, vol_window, atr_ser, atr_stop_mult
        )
        .shift(1)
        .fillna(0)
    )

    return entries, exits, decel_sh, sizes


def run_pf(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    sizes: pd.Series,
    fees: float,
) -> "vbt.Portfolio":
    return vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        size=sizes,
        size_type="targetpercent",
        init_cash=INIT_CASH,
        fees=fees,
        slippage=SLIPPAGE,
        freq="1D",
    )


def bah_portfolio(close: pd.Series) -> "vbt.Portfolio":
    """Buy-and-hold baseline — fully invested from bar 1, no fees."""
    entries = pd.Series(False, index=close.index)
    entries.iloc[0] = True
    exits = pd.Series(False, index=close.index)
    return vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        size=1.0,
        size_type="targetpercent",
        init_cash=INIT_CASH,
        fees=0.0,
        freq="1D",
    )


def stats_row(label: str, pf: "vbt.Portfolio") -> dict:
    n = pf.trades.count()
    return {
        "scenario": label,
        "total_return": round(float(pf.total_return()), 3),
        "sharpe": round(float(pf.sharpe_ratio()), 3),
        "calmar": round(float(pf.calmar_ratio()), 3),
        "max_drawdown": round(float(pf.max_drawdown()), 3),
        "n_trades": int(n),
        "win_rate": round(float(pf.trades.win_rate()), 3) if n > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF Trend Stage 5: Full Friction Portfolio Sim.")
    parser.add_argument("--instrument", default="SPY", help="Symbol (default: SPY)")
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()

    print("=" * 60)
    print("  ETF Trend — Stage 5: Portfolio Simulation")
    print("=" * 60)
    print(f"  Instrument: {instrument}")

    config = load_config(inst_lower)
    df = load_data(instrument)
    close = df["close"]

    split = int(len(close) * 0.70)
    oos_close = close.iloc[split:]
    oos_df = df.iloc[split:]

    print(f"  OOS period: {oos_close.index[0].date()} to {oos_close.index[-1].date()}")
    print(f"  OOS bars:   {len(oos_close)}")

    # Build signals on full series, slice to OOS
    all_entries, all_exits, all_decel_sh, all_sizes = build_signals(close, df, config)
    oos_entries = all_entries.iloc[split:]
    oos_exits = all_exits.iloc[split:]
    oos_decel_sh = all_decel_sh.iloc[split:]
    oos_sizes = all_sizes.iloc[split:]
    oos_atr = compute_atr(oos_df)

    # ── Base strategy (dynamic sizing, base fees) ─────────────────────────
    pf_main = run_pf(oos_close, oos_entries, oos_exits, oos_sizes, FEES_BASE)

    # ── Binary sizing (size=1 where entry, else 0) ────────────────────────
    from research.etf_trend.run_stage4_sizing import compute_position_sizes

    binary_sizes = (
        compute_position_sizes(
            oos_close,
            oos_decel_sh,
            "binary",
            config["vol_target"],
            config["vol_window"],
            oos_atr,
            config["atr_stop_mult"],
        )
        .shift(1)
        .fillna(0)
        .iloc[: len(oos_close)]
    )
    pf_binary = run_pf(oos_close, oos_entries, oos_exits, binary_sizes, FEES_BASE)

    # ── Buy and hold baseline ─────────────────────────────────────────────
    pf_bah = bah_portfolio(oos_close)

    # ── Summary table ─────────────────────────────────────────────────────
    rows = [
        stats_row(f"Strategy ({config['sizing_mode']})", pf_main),
        stats_row("Strategy (binary)", pf_binary),
        stats_row("Buy & Hold (no fees)", pf_bah),
    ]

    # Transaction cost stress test
    for label, fee in FEE_SCENARIOS.items():
        if abs(fee - FEES_BASE) < 1e-6:
            continue
        pf_stress = run_pf(oos_close, oos_entries, oos_exits, oos_sizes, fee)
        rows.append(stats_row(f"Strategy fees {label}", pf_stress))

    comparison = pd.DataFrame(rows)
    print("\n  ── OOS Comparison ─────────────────────────────────")
    print(comparison.to_string(index=False))

    # ── Pass / fail vs buy-and-hold ───────────────────────────────────────
    bah = rows[2]
    strat = rows[0]
    better_sharpe = strat["sharpe"] >= bah["sharpe"]
    better_dd = abs(strat["max_drawdown"]) <= abs(bah["max_drawdown"])
    better_calmar = strat["calmar"] >= bah["calmar"]
    n_better = sum([better_sharpe, better_dd, better_calmar])

    print("\n  ── vs Buy & Hold ──────────────────────────────────")
    print(
        f"  Sharpe: strat={strat['sharpe']:.3f} vs B&H={bah['sharpe']:.3f}  "
        f"{'BETTER' if better_sharpe else 'WORSE'}"
    )
    print(
        f"  MaxDD:  strat={strat['max_drawdown']:.1%} vs B&H={bah['max_drawdown']:.1%}  "
        f"{'BETTER' if better_dd else 'WORSE'}"
    )
    print(
        f"  Calmar: strat={strat['calmar']:.3f} vs B&H={bah['calmar']:.3f}  "
        f"{'BETTER' if better_calmar else 'WORSE'}"
    )
    print(
        f"\n  Result: {n_better}/3 metrics beat B&H — "
        f"{'PASS' if n_better >= 2 else 'FAIL (needs 2/3)'}"
    )

    # ── HTML report ───────────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=1)
        for label, pf in [
            (f"Strategy ({config['sizing_mode']})", pf_main),
            ("Strategy (binary)", pf_binary),
            ("Buy & Hold", pf_bah),
        ]:
            eq = pf.value() / INIT_CASH
            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=label, mode="lines"))

        fig.update_layout(
            title=f"ETF Trend Strategy vs Buy & Hold — {instrument} (OOS)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (normalised)",
            legend_title="Scenario",
        )
        report_path = REPORTS_DIR / f"etf_trend_{inst_lower}_comparison.html"
        fig.write_html(str(report_path))
        print(f"\n  Report saved → {report_path.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"\n  [HTML report skipped: {e}]")

    scoreboard_path = REPORTS_DIR / f"etf_trend_stage5_{inst_lower}_comparison.csv"
    comparison.to_csv(scoreboard_path, index=False)
    print(f"  CSV saved   → {scoreboard_path.relative_to(PROJECT_ROOT)}")

    if n_better >= 2:
        print("\n  [PASS] Stage 5 complete.")
        print(
            f"  Next: uv run python research/etf_trend/run_robustness.py --instrument {instrument}"
        )
    else:
        print("\n  [FAIL] Strategy does not beat Buy & Hold on 2/3 metrics.")
        print("  Consider revisiting Stage 3 (exit mode) or Stage 4 (sizing).")
    print("=" * 60)


if __name__ == "__main__":
    main()
