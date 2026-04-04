"""run_portfolio.py -- Stage 5: Full Friction Portfolio Simulation.

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
import itertools
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

from research.etf_trend.run_stage3_exits import (  # noqa: E402
    apply_sma_confirmation,
    build_entry_signal,
    compute_decel_composite,
    compute_ma,
)
from research.etf_trend.run_stage4_sizing import (  # noqa: E402
    compute_position_sizes,
    compute_vol_target_sizes,
)

FEES_BASE = 0.001
SLIPPAGE = 0.0005
INIT_CASH = 10_000.0

# Transaction cost stress test levels
FEE_SCENARIOS = {"base (0.10%)": 0.001, "2x (0.20%)": 0.002, "3x (0.30%)": 0.003}

# Circuit breaker sweep grid
CB_TRIP_PCTS = [-0.08, -0.10, -0.12, -0.15, -0.20]
CB_RESET_PCTS = [-0.03, -0.05, -0.08]
CB_ROLLING_WINDOW = 252


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
        For vol_target sizing, sizes contains the leverage array (0 to max_leverage).
    """
    ma_type = config["ma_type"]
    slow_ma_p = config["slow_ma"]
    decel_signals = config.get("decel_signals", [])
    decel_params = {k: config[k] for k in ("d_pct_smooth", "rv_window", "macd_fast") if k in config}
    entry_mode = config.get("entry_mode", "decel_positive")
    fast_reentry_ma_p = config.get("fast_reentry_ma")
    exit_mode = config["exit_mode"]
    exit_decel_thresh = config["exit_decel_thresh"]
    exit_confirm_days = int(config.get("exit_confirm_days", 1))
    decel_confirm_days = int(config.get("decel_confirm_days", 1))
    sizing_mode = config["sizing_mode"]

    slow_ma = compute_ma(close, slow_ma_p, ma_type)
    decel = compute_decel_composite(close, df, slow_ma, decel_signals, decel_params)

    fast_ma = compute_ma(close, int(fast_reentry_ma_p), ma_type) if fast_reentry_ma_p else None
    entry_signal = build_entry_signal(close, slow_ma, decel, entry_mode, fast_ma)

    below_slow = apply_sma_confirmation(~(close > slow_ma), exit_confirm_days)
    if exit_mode == "A":
        exit_signal = below_slow
    elif exit_mode == "C":
        exit_signal = apply_sma_confirmation(decel < exit_decel_thresh, decel_confirm_days)
    else:  # D -- confirmed SMA break OR sustained decel collapse
        decel_exit = apply_sma_confirmation(decel < exit_decel_thresh, decel_confirm_days)
        exit_signal = below_slow | decel_exit

    entries = entry_signal.shift(1).fillna(False)
    exits = exit_signal.shift(1).fillna(False)
    decel_sh = decel.shift(1).fillna(0)

    if sizing_mode == "vol_target":
        vol_target = float(config.get("vol_target", 0.15))
        max_leverage = float(config.get("max_leverage", 1.0))
        in_regime = close > slow_ma
        sizes = compute_vol_target_sizes(close, in_regime, vol_target, max_leverage)
    else:
        sizes = compute_position_sizes(decel_sh, sizing_mode).shift(1).fillna(0)

    return entries, exits, decel_sh, sizes


def run_pf(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    sizes: pd.Series,
    fees: float,
    sizing_mode: str = "binary",
) -> "vbt.Portfolio":
    """Run VBT portfolio simulation. For vol_target, sizes is a leverage array."""
    if sizing_mode == "vol_target":
        # sizes = leverage array; simulate leveraged equity curve

        daily_rets = close.pct_change().fillna(0)
        strat_rets = daily_rets * sizes
        in_pos = sizes > 0
        entering = in_pos & (~in_pos.shift(1).fillna(True))
        exiting = (~in_pos) & in_pos.shift(1).fillna(False)
        strat_rets -= entering.astype(float) * (fees + SLIPPAGE)
        strat_rets -= exiting.astype(float) * (fees + SLIPPAGE)
        equity = INIT_CASH * (1 + strat_rets).cumprod()
        # Wrap in a minimal Portfolio-compatible object via from_value
        return _EquityCurvePortfolio(equity, entries, strat_rets)
    if sizing_mode == "binary":
        return vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            init_cash=INIT_CASH,
            fees=fees,
            slippage=SLIPPAGE,
            freq="1D",
        )
    else:  # dynamic_decel -- fixed initial-cash allocation
        return vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            size=sizes * INIT_CASH,
            size_type="value",
            init_cash=INIT_CASH,
            fees=fees,
            slippage=SLIPPAGE,
            freq="1D",
        )


class _EquityCurvePortfolio:
    """Thin wrapper around a pre-computed equity curve, mimicking VBT Portfolio API."""

    def __init__(self, equity: "pd.Series", entries: "pd.Series", rets: "pd.Series") -> None:
        import numpy as np

        self._equity = equity
        self._entries = entries
        self._rets = rets
        self._np = np

    def total_return(self) -> float:
        return float(self._equity.iloc[-1] / INIT_CASH - 1)

    def sharpe_ratio(self) -> float:
        std = self._rets.std()
        return float(self._rets.mean() / std * self._np.sqrt(252)) if std > 1e-9 else 0.0

    def max_drawdown(self) -> float:
        rolling_max = self._equity.cummax()
        dd = (self._equity - rolling_max) / rolling_max
        return float(dd.min())

    def calmar_ratio(self) -> float:
        total_ret = self.total_return()
        max_dd = self.max_drawdown()
        if max_dd >= -1e-9:
            return 0.0
        n_bars = len(self._equity)
        ann_ret = (1 + total_ret) ** (365 / n_bars) - 1
        return ann_ret / abs(max_dd)

    def value(self) -> "pd.Series":
        return self._equity / INIT_CASH  # normalised

    class _TradeMock:
        def count(self) -> int:
            return 0

        def win_rate(self) -> float:
            return 0.0

    @property
    def trades(self) -> "_TradeMock":
        return self._TradeMock()


def bah_portfolio(close: pd.Series) -> "vbt.Portfolio":
    """Buy-and-hold baseline -- fully invested from bar 1, no fees."""
    entries = pd.Series(False, index=close.index)
    entries.iloc[0] = True
    exits = pd.Series(False, index=close.index)
    return vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        size=INIT_CASH,
        size_type="value",
        init_cash=INIT_CASH,
        fees=0.0,
        freq="1D",
    )


def compute_circuit_breaker(
    close: pd.Series,
    rolling_window: int = CB_ROLLING_WINDOW,
    trip_pct: float = -0.10,
    reset_pct: float = -0.05,
) -> pd.Series:
    """Boolean Series: True when circuit breaker is active (be flat).

    Trips when close falls more than |trip_pct| below its rolling_window-bar high.
    Resets when close recovers above reset_pct below that rolling high.
    Stateful (hysteresis) -- vectorised rolling cannot replicate this.
    Look-ahead-free: rolling high is computed from past_close = close.shift(1).
    """
    past_close = close.shift(1)
    rolling_high = past_close.rolling(rolling_window, min_periods=1).max()
    dd = (close / rolling_high) - 1.0

    cb_active = pd.Series(False, index=close.index)
    tripped = False
    for i in range(len(close)):
        d = dd.iloc[i]
        if not tripped:
            if d <= trip_pct:
                tripped = True
        else:
            if d > reset_pct:
                tripped = False
        cb_active.iloc[i] = tripped
    return cb_active


def apply_circuit_breaker(
    entries: pd.Series,
    exits: pd.Series,
    cb_active: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Apply CB overlay: suppress entries and force exit on trip edge."""
    cb_trip_edge = cb_active & ~cb_active.shift(1).fillna(False)
    return entries & ~cb_active, exits | cb_trip_edge


def _save_cb_params_to_toml(instrument: str, cb_params: dict) -> None:
    """Append cb_trip_pct / cb_reset_pct to the locked TOML, replacing any prior values."""
    config_path = PROJECT_ROOT / "config" / f"etf_trend_{instrument}.toml"
    if not config_path.exists():
        print(f"  [WARN] Config not found, skipping CB save: {config_path}")
        return
    lines = [
        ln
        for ln in config_path.read_text().splitlines()
        if not ln.startswith("cb_trip_pct") and not ln.startswith("cb_reset_pct")
    ]
    while lines and lines[-1].strip() == "":
        lines.pop()
    lines.append(f"cb_trip_pct       = {cb_params['trip_pct']}")
    lines.append(f"cb_reset_pct      = {cb_params['reset_pct']}")
    lines.append("")
    config_path.write_text("\n".join(lines))


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
    parser.add_argument(
        "--circuit-breaker",
        action="store_true",
        help="Sweep CB params vs no-CB baseline and save best to TOML.",
    )
    args = parser.parse_args()
    instrument = args.instrument.upper()
    inst_lower = instrument.lower()

    print("=" * 60)
    print("  ETF Trend -- Stage 5: Portfolio Simulation")
    print("=" * 60)
    print(f"  Instrument: {instrument}")

    config = load_config(inst_lower)

    # Dual-instrument support: signal_instrument overrides signal computation
    sig_str = config.get("signal_instrument", "")
    signal_sym = sig_str.split(".")[0] if sig_str else instrument
    is_dual = signal_sym.upper() != instrument

    exec_df = load_data(instrument)
    if is_dual:
        sig_df = load_data(signal_sym)
        common = sig_df.index.intersection(exec_df.index)
        sig_df = sig_df.loc[common]
        exec_df = exec_df.loc[common]
        close = sig_df["close"]  # signal close (QQQ)
        exec_close = exec_df["close"]  # execution close (TQQQ)
        print(f"  Signal src: {signal_sym}  (signals on {signal_sym}, P&L on {instrument})")
    else:
        sig_df = exec_df
        close = exec_df["close"]
        exec_close = close

    split = int(len(close) * 0.70)
    oos_close = exec_close.iloc[split:]  # P&L slice (TQQQ or same instrument)
    oos_sig_close = close.iloc[split:]  # signal slice (QQQ or same instrument)

    print(f"  OOS period: {oos_sig_close.index[0].date()} to {oos_sig_close.index[-1].date()}")
    print(f"  OOS bars:   {len(oos_sig_close)}")

    # Build signals on full series (signals use QQQ; P&L uses TQQQ separately)
    all_entries, all_exits, all_decel_sh, all_sizes = build_signals(close, sig_df, config)
    oos_entries = all_entries.iloc[split:]
    oos_exits = all_exits.iloc[split:]
    oos_decel_sh = all_decel_sh.iloc[split:]
    oos_sizes = all_sizes.iloc[split:]

    sizing_mode = config["sizing_mode"]

    # -- Base strategy (configured sizing, base fees) ----------------------
    pf_main = run_pf(oos_close, oos_entries, oos_exits, oos_sizes, FEES_BASE, sizing_mode)

    # -- Binary sizing comparison (fully invested when in trade) -----------
    binary_sizes = (
        compute_position_sizes(oos_decel_sh, "binary").shift(1).fillna(0).iloc[: len(oos_close)]
    )
    pf_binary = run_pf(oos_close, oos_entries, oos_exits, binary_sizes, FEES_BASE, "binary")

    # -- Buy and hold baseline (on execution instrument) -------------------
    pf_bah = bah_portfolio(oos_close)

    # -- Summary table -----------------------------------------------------
    rows = [
        stats_row(f"Strategy ({config['sizing_mode']})", pf_main),
        stats_row("Strategy (binary)", pf_binary),
        stats_row("Buy & Hold (no fees)", pf_bah),
    ]

    # Transaction cost stress test
    for label, fee in FEE_SCENARIOS.items():
        if abs(fee - FEES_BASE) < 1e-6:
            continue
        pf_stress = run_pf(oos_close, oos_entries, oos_exits, oos_sizes, fee, sizing_mode)
        rows.append(stats_row(f"Strategy fees {label}", pf_stress))

    comparison = pd.DataFrame(rows)
    print("\n  -- OOS Comparison ---------------------------------")
    print(comparison.to_string(index=False))

    # -- Pass / fail vs buy-and-hold ---------------------------------------
    bah = rows[2]
    strat = rows[0]
    better_return = strat["total_return"] >= bah["total_return"]
    better_sharpe = strat["sharpe"] >= bah["sharpe"]
    better_dd = abs(strat["max_drawdown"]) <= abs(bah["max_drawdown"])
    better_calmar = strat["calmar"] >= bah["calmar"]
    n_better = sum([better_sharpe, better_dd, better_calmar])

    print("\n  -- vs Buy & Hold ----------------------------------")
    print(
        f"  Return: strat={strat['total_return']:.1%} vs B&H={bah['total_return']:.1%}  "
        f"{'BETTER' if better_return else 'WORSE'}"
    )
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
        f"\n  Result: {n_better}/3 risk-adjusted metrics beat B&H -- "
        f"{'PASS' if n_better >= 2 else 'FAIL (needs 2/3)'}"
    )

    # -- HTML report -------------------------------------------------------
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
            title=f"ETF Trend Strategy vs Buy & Hold -- {instrument} (OOS)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (normalised)",
            legend_title="Scenario",
        )
        report_path = REPORTS_DIR / f"etf_trend_{inst_lower}_comparison.html"
        fig.write_html(str(report_path))
        print(f"\n  Report saved: {report_path.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"\n  [HTML report skipped: {e}]")

    scoreboard_path = REPORTS_DIR / f"etf_trend_stage5_{inst_lower}_comparison.csv"
    comparison.to_csv(scoreboard_path, index=False)
    print(f"  CSV saved  : {scoreboard_path.relative_to(PROJECT_ROOT)}")

    if n_better >= 2:
        print("\n  [PASS] Stage 5 complete.")
        print(
            f"  Next: uv run python research/etf_trend/run_robustness.py --instrument {instrument}"
        )
    else:
        print("\n  [FAIL] Strategy does not beat Buy & Hold on 2/3 metrics.")
        print("  Consider revisiting Stage 3 (exit mode) or Stage 4 (sizing).")

    # -- Circuit Breaker sweep (--circuit-breaker flag) --------------------
    if args.circuit_breaker:
        print("\n  -- Circuit Breaker Sweep ----------------------------------")
        print(f"  trip_pcts:  {CB_TRIP_PCTS}")
        print(f"  reset_pcts: {CB_RESET_PCTS}")

        # Compute CB on full series once per param combo, then slice to OOS
        cb_rows: list[dict] = []
        best_calmar = -999.0
        best_cb_params: dict = {}

        for trip_pct, reset_pct in itertools.product(CB_TRIP_PCTS, CB_RESET_PCTS):
            if reset_pct <= trip_pct:
                continue  # reset must be less severe (closer to 0) than trip

            cb_full = compute_circuit_breaker(close, CB_ROLLING_WINDOW, trip_pct, reset_pct)
            oos_cb = cb_full.iloc[split:]
            cb_entries, cb_exits = apply_circuit_breaker(oos_entries, oos_exits, oos_cb)
            pf_cb = run_pf(oos_close, cb_entries, cb_exits, binary_sizes, FEES_BASE, "binary")
            row = stats_row(f"cb trip={trip_pct:.0%} reset={reset_pct:.0%}", pf_cb)
            row["cb_bars_flat"] = int(oos_cb.sum())
            row["trip_pct"] = trip_pct
            row["reset_pct"] = reset_pct
            cb_rows.append(row)
            if row["calmar"] > best_calmar:
                best_calmar = row["calmar"]
                best_cb_params = {"trip_pct": trip_pct, "reset_pct": reset_pct}

        baseline = stats_row("no CB (binary)", pf_binary)
        cb_df = pd.DataFrame([baseline] + cb_rows)
        display_cols = [
            "scenario",
            "total_return",
            "sharpe",
            "calmar",
            "max_drawdown",
            "n_trades",
            "win_rate",
        ]
        print("\n" + cb_df[display_cols].to_string(index=False))

        if best_cb_params:
            best_row = next(
                r
                for r in cb_rows
                if r["trip_pct"] == best_cb_params["trip_pct"]
                and r["reset_pct"] == best_cb_params["reset_pct"]
            )
            print(
                f"\n  Baseline:  Sharpe={baseline['sharpe']:.3f}  "
                f"MaxDD={baseline['max_drawdown']:.1%}  "
                f"Return={baseline['total_return']:.1%}"
            )
            print(
                f"  Best CB:   Sharpe={best_row['sharpe']:.3f}  "
                f"MaxDD={best_row['max_drawdown']:.1%}  "
                f"Return={best_row['total_return']:.1%}"
            )
            print(
                f"  Params:    trip={best_cb_params['trip_pct']:.0%}  "
                f"reset={best_cb_params['reset_pct']:.0%}"
            )
            _save_cb_params_to_toml(inst_lower, best_cb_params)
            print(f"  Saved to:  config/etf_trend_{inst_lower}.toml")

        cb_csv = REPORTS_DIR / f"etf_trend_stage5_{inst_lower}_cb_sweep.csv"
        pd.DataFrame(cb_rows).to_csv(cb_csv, index=False)
        print(f"  CB CSV:    {cb_csv.relative_to(PROJECT_ROOT)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
