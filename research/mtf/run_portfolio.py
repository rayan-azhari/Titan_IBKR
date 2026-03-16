"""run_mtf_portfolio.py — Realistic Portfolio Simulation.

Simulates the MTF Confluence strategy with proper risk management:
- Unified Long/Short Portfolio (reversals handled automatically)
- Volatility-Adjusted Sizing (1% Equity Risk per trade)
- Trailing Stop Loss (atr_stop_mult from config, default 2.0 * ATR)
- Leverage Cap (5.0x)
- Overnight Swap Cost (annual % drag from config/spread.toml)

Usage:
    uv run python research/mtf/run_portfolio.py                      # EUR/USD
    uv run python research/mtf/run_portfolio.py --pair GBP_USD
    uv run python research/mtf/run_portfolio.py --pair GBP_USD --config config/mtf_gbpusd.toml
"""

import argparse
import sys
import tomllib
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
    import vectorbt as vbt  # noqa: E402
except ImportError:
    print("ERROR: vectorbt not installed.")
    sys.exit(1)

from titan.models.spread import build_spread_series  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

INIT_CASH = 100_000.0  # $100k account
RISK_PER_TRADE = 0.01  # 1% equity risk per trade
ATR_PERIOD = 14  # For volatility sizing
MAX_LEVERAGE = 5.0  # Hard cap on position size
H4_BARS_PER_YEAR = 2190  # 6 bars/day * 365


# ─────────────────────────────────────────────────────────────────────
# Helpers
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


def load_mtf_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_swap_annual_pct(pair: str) -> float:
    """Return annual swap drag fraction for pair (from config/spread.toml)."""
    spread_cfg = PROJECT_ROOT / "config" / "spread.toml"
    try:
        with open(spread_cfg, "rb") as f:
            cfg = tomllib.load(f)
        return float(cfg.get("swap", {}).get(pair, 0.015))
    except FileNotFoundError:
        return 0.015  # 1.5% annual fallback


def compute_swap_drag(pf, swap_annual_pct: float) -> float:
    """Estimate total overnight swap cost from per-bar position exposure.

    Method: for each H4 bar, charge swap_annual_pct / H4_BARS_PER_YEAR
    of the absolute position value in $. Direction-agnostic.
    Returns total drag in $ terms.
    """
    pos_value = pf.asset_value().abs()
    swap_per_bar = swap_annual_pct / H4_BARS_PER_YEAR
    return float((pos_value * swap_per_bar).sum())


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_tf_signal(close: pd.Series, fast_ma: int, slow_ma: int, rsi_period: int) -> pd.Series:
    fast = close.rolling(fast_ma).mean()
    slow = close.rolling(slow_ma).mean()
    rsi = compute_rsi(close, rsi_period)
    ma_sig = pd.Series(np.where(fast > slow, 0.5, -0.5), index=close.index)
    rsi_sig = pd.Series(np.where(rsi > 50, 0.5, -0.5), index=close.index)
    return ma_sig + rsi_sig


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift())
    tr2 = abs(low - close.shift())
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_confluence(pair: str, mtf_config: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate weighted confluence score."""
    weights = mtf_config.get("weights", {})
    primary_df = load_data(pair, "H4")
    if primary_df is None:
        raise ValueError("H4 data missing")

    primary_index = primary_df.index
    signals_sum = pd.Series(0.0, index=primary_index)
    total_weight = 0.0

    print("  Timeframes:")
    for tf in ["H1", "H4", "D", "W"]:
        w = weights.get(tf, 0.0)
        if w == 0:
            continue

        cfg = mtf_config.get(tf, {})
        df = load_data(pair, tf)
        if df is None:
            continue

        sig = compute_tf_signal(
            df["close"],
            cfg.get("fast_ma", 20),
            cfg.get("slow_ma", 50),
            cfg.get("rsi_period", 14),
        )
        resampled = sig.reindex(primary_index, method="ffill")
        signals_sum += resampled * w
        total_weight += w
        print(f"    {tf}: w={w:.2f} (loaded {len(df)} bars)")

    if total_weight > 0 and total_weight < 1.0:
        signals_sum /= total_weight

    return signals_sum, primary_df


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MTF Portfolio Simulation (risk-managed).")
    parser.add_argument("--pair", default="EUR_USD", help="Instrument (default: EUR_USD)")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to TOML config (default: auto-detect from pair)",
    )
    args = parser.parse_args()
    pair = args.pair.upper()
    pair_lower = pair.lower().replace("_", "")

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
    else:
        pair_cfg = PROJECT_ROOT / "config" / f"mtf_{pair_lower}.toml"
        config_path = pair_cfg if pair_cfg.exists() else PROJECT_ROOT / "config" / "mtf.toml"

    print("=" * 60)
    print("  MTF Portfolio Simulation (Risk-Managed)")
    print("=" * 60)
    print(f"  Pair:           {pair}")
    print(f"  Initial Equity: ${INIT_CASH:,.0f}")
    print(f"  Risk Per Trade: {RISK_PER_TRADE:.1%}")
    print(f"  Using params from: {config_path.name}")

    # 1. Load Config & Confluence
    cfg = load_mtf_config(config_path)
    threshold = cfg.get("confirmation_threshold", 0.10)
    stop_atr_mult = float(cfg.get("atr_stop_mult", 2.0))
    swap_annual_pct = load_swap_annual_pct(pair)
    print(f"  ATR Stop Mult:  {stop_atr_mult}x")
    print(f"  Swap Annual:    {swap_annual_pct:.1%}")
    confluence, primary_df = compute_confluence(pair, cfg)

    close = primary_df["close"]
    high = primary_df["high"]
    low = primary_df["low"]

    # 2. Compute ATR for sizing
    atr = compute_atr(primary_df, ATR_PERIOD)

    # 3. Generate Signals
    # Shift by 1: signal fires at close of bar i, order fills at bar i+1
    conf_sh = confluence.shift(1).fillna(0.0)
    entries_long = conf_sh >= threshold
    entries_short = conf_sh <= -threshold

    # Exits: Neutrality or Reversal will be handled by Portfolio logic
    # But explicitly, we exit if confluence crosses zero against us
    # VBT handles reversals if we provide both entries and short_entries

    # 4. Position Sizing Logic (Fixed Risk Amount based on Initial Equity)
    # Stop Distance = atr_stop_mult * ATR
    # Risk Amount = INIT_CASH * RISK_PER_TRADE ($1,000)
    # Units = Risk Amount / Stop Distance

    risk_amt = INIT_CASH * RISK_PER_TRADE
    stop_dist = stop_atr_mult * atr

    # Calculate Stop Loss % (needed for sl_stop argument)
    stop_loss_pct = stop_dist / close
    stop_loss_pct = stop_loss_pct.replace(0, np.nan).fillna(0.0)

    # Raw unit size
    raw_units = risk_amt / stop_dist

    # Cap leverage (Max Units = (Equity * Leverage) / Price)
    # Approximation using Init Cash since vectorized
    max_units = (INIT_CASH * MAX_LEVERAGE) / close

    target_units = np.minimum(raw_units, max_units)
    target_units = target_units.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 5. Build Portfolio

    spread_series = build_spread_series(primary_df, pair)
    avg_spread = float(spread_series.mean())
    print(f"  Avg Spread Cost: {avg_spread:.5f}")

    # Run 1: With Trailing Stop
    print(f"\nRunning Scenario 1: Trailing Stop ({stop_atr_mult}x ATR)...")
    pf_stop = vbt.Portfolio.from_signals(
        close=close,
        high=high,
        low=low,
        entries=entries_long,
        short_entries=entries_short,
        exits=(conf_sh < 0),
        short_exits=(conf_sh > 0),
        size=target_units,
        size_type="amount",
        init_cash=INIT_CASH,
        fees=avg_spread,
        freq="4h",
        sl_stop=stop_loss_pct,
        sl_trail=True,
    )

    # Run 2: Signal Only (No Stop)
    print("Running Scenario 2: Signal Only (No Stop)...")
    pf_signal = vbt.Portfolio.from_signals(
        close=close,
        high=high,
        low=low,
        entries=entries_long,
        short_entries=entries_short,
        exits=(conf_sh < 0),
        short_exits=(conf_sh > 0),
        size=target_units,
        size_type="amount",
        init_cash=INIT_CASH,
        fees=avg_spread,
        freq="4h",
        # No sl_stop
    )

    # 6. Analyze & Compare
    swap_stop = compute_swap_drag(pf_stop, swap_annual_pct)
    swap_signal = compute_swap_drag(pf_signal, swap_annual_pct)

    print("\n" + "=" * 80)
    print(f"  COMPARISON: Trailing Stop ({stop_atr_mult}x ATR) vs Signal Only")
    print(f"  (Swap cost modelled at {swap_annual_pct:.1%}/yr on open position value)")
    print("=" * 80)

    headers = ["Metric", "With Stop", "Signal Only", "Diff"]
    row_fmt = "{:<20} {:<18} {:<18} {:<15}"
    print(row_fmt.format(*headers))
    print("-" * 71)

    def get_metrics(pf):
        return {
            "Return": pf.total_return(),
            "Sharpe": pf.sharpe_ratio(),
            "Max DD": pf.max_drawdown(),
            "Win Rate": pf.trades.win_rate(),
            "Trades": pf.trades.count(),
            "Final Eq": pf.value().iloc[-1],
        }

    m1 = get_metrics(pf_stop)
    m2 = get_metrics(pf_signal)

    # Swap-adjusted final equity
    m1_adj_eq = m1["Final Eq"] - swap_stop
    m2_adj_eq = m2["Final Eq"] - swap_signal
    m1_adj_ret = (m1_adj_eq - INIT_CASH) / INIT_CASH
    m2_adj_ret = (m2_adj_eq - INIT_CASH) / INIT_CASH

    def fmt(val, is_pct=True):
        return f"{val:.2%}" if is_pct else f"{val:.2f}"

    print(
        row_fmt.format(
            "Total Return",
            fmt(m1["Return"]),
            fmt(m2["Return"]),
            f"{m2['Return'] - m1['Return']:+.2%}",
        )
    )
    print(
        row_fmt.format(
            "Sharpe",
            fmt(m1["Sharpe"], False),
            fmt(m2["Sharpe"], False),
            f"{m2['Sharpe'] - m1['Sharpe']:+.2f}",
        )
    )
    print(
        row_fmt.format(
            "Max Drawdown",
            fmt(m1["Max DD"]),
            fmt(m2["Max DD"]),
            f"{m2['Max DD'] - m1['Max DD']:+.2%}",
        )
    )
    print(
        row_fmt.format(
            "Win Rate",
            fmt(m1["Win Rate"]),
            fmt(m2["Win Rate"]),
            f"{m2['Win Rate'] - m1['Win Rate']:+.2%}",
        )
    )
    print(
        row_fmt.format(
            "Trades", str(m1["Trades"]), str(m2["Trades"]), f"{m2['Trades'] - m1['Trades']}"
        )
    )
    print(
        row_fmt.format(
            "Final Equity",
            f"${m1['Final Eq']:,.0f}",
            f"${m2['Final Eq']:,.0f}",
            f"${m2['Final Eq'] - m1['Final Eq']:,.0f}",
        )
    )
    print("-" * 71)
    print(
        row_fmt.format(
            "Swap Cost ($)",
            f"-${swap_stop:,.0f}",
            f"-${swap_signal:,.0f}",
            f"${swap_signal - swap_stop:,.0f}",
        )
    )
    print(
        row_fmt.format(
            "Adj Return",
            fmt(m1_adj_ret),
            fmt(m2_adj_ret),
            f"{m2_adj_ret - m1_adj_ret:+.2%}",
        )
    )
    print(
        row_fmt.format(
            "Adj Final Equity",
            f"${m1_adj_eq:,.0f}",
            f"${m2_adj_eq:,.0f}",
            f"${m2_adj_eq - m1_adj_eq:,.0f}",
        )
    )

    # 7. Save Reports
    pf_stop.trades.records_readable.to_csv(
        REPORTS_DIR / f"trades_{pair_lower}_with_stop.csv", index=False
    )
    pf_signal.trades.records_readable.to_csv(
        REPORTS_DIR / f"trades_{pair_lower}_signal_only.csv", index=False
    )

    # Plot equity curves (raw and swap-adjusted)
    stop_val = pf_stop.value()
    signal_val = pf_signal.value()
    # Cumulative swap drag per bar (proportional to position value over time)
    stop_swap_cum = (pf_stop.asset_value().abs() * (swap_annual_pct / H4_BARS_PER_YEAR)).cumsum()
    signal_swap_cum = (
        pf_signal.asset_value().abs() * (swap_annual_pct / H4_BARS_PER_YEAR)
    ).cumsum()

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=stop_val.index, y=stop_val, name=f"Stop {stop_atr_mult}x ATR"))
    fig.add_trace(go.Scatter(x=signal_val.index, y=signal_val, name="Signal Only"))
    fig.add_trace(
        go.Scatter(
            x=signal_val.index,
            y=signal_val - signal_swap_cum,
            name="Signal Only (swap-adj)",
            line={"dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stop_val.index,
            y=stop_val - stop_swap_cum,
            name=f"Stop {stop_atr_mult}x (swap-adj)",
            line={"dash": "dot"},
        )
    )
    fig.update_layout(title=f"Equity Curve Comparison — {pair}", yaxis_title="Equity ($)")

    html_path = REPORTS_DIR / f"mtf_{pair_lower}_comparison.html"
    fig.write_html(str(html_path))
    print(f"\nSaved comparison report to {html_path}")

    # Generate comprehensive 6-panel visualisation (non-blocking)
    import subprocess

    sys.stdout.flush()
    vis_script = PROJECT_ROOT / "scripts" / "visualise_mtf.py"
    if vis_script.exists():
        subprocess.run(
            [sys.executable, str(vis_script), "--pair", pair, "--scenario", "signal"],
            check=False,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
