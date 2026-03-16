"""run_portfolio_combined.py — Two-pair combined portfolio simulation.

Tests the diversification benefit of running EUR/USD + USD/CHF simultaneously
with a 50/50 capital split ($50k each, $100k total).

Uses signal-only exit — validated as optimal for trend-following.
Key output: return correlation between the two pairs and whether the combined
equity curve has a lower MaxDD than either individual.

Usage:
    uv run python research/mtf/run_portfolio_combined.py
"""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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

import tomllib

from titan.models.spread import build_spread_series  # noqa: E402

INIT_CASH = 100_000.0
HALF_CASH = INIT_CASH / 2  # $50k per pair
RISK_PER_TRADE = 0.01
ATR_PERIOD = 14
MAX_LEVERAGE = 5.0
H4_BARS_PER_YEAR = 2190


def _resolve_config(pair: str) -> Path:
    """Return config path for pair, falling back to generic mtf.toml."""
    pair_lower = pair.lower().replace("_", "")
    pair_cfg = PROJECT_ROOT / "config" / f"mtf_{pair_lower}.toml"
    return pair_cfg if pair_cfg.exists() else PROJECT_ROOT / "config" / "mtf.toml"


# ─────────────────────────────────────────────────────────────────────
# Helpers (self-contained — no dependency on run_portfolio.py)
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


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_swap_pct(pair: str) -> float:
    spread_cfg = PROJECT_ROOT / "config" / "spread.toml"
    try:
        with open(spread_cfg, "rb") as f:
            cfg = tomllib.load(f)
        return float(cfg.get("swap", {}).get(pair, 0.015))
    except FileNotFoundError:
        return 0.015


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))


def compute_tf_signal(close: pd.Series, fast_ma: int, slow_ma: int, rsi_period: int) -> pd.Series:
    ma_sig = pd.Series(
        np.where(close.rolling(fast_ma).mean() > close.rolling(slow_ma).mean(), 0.5, -0.5),
        index=close.index,
    )
    rsi_sig = pd.Series(
        np.where(compute_rsi(close, rsi_period) > 50, 0.5, -0.5),
        index=close.index,
    )
    return ma_sig + rsi_sig


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            abs(df["high"] - df["low"]),
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift()),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def compute_confluence(pair: str, cfg: dict) -> tuple[pd.Series, pd.DataFrame]:
    weights = cfg.get("weights", {})
    primary_df = load_data(pair, "H4")
    if primary_df is None:
        raise ValueError(f"H4 data missing for {pair}")

    idx = primary_df.index
    signals = pd.Series(0.0, index=idx)
    total_w = 0.0

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
        )
        signals += sig.reindex(idx, method="ffill") * w
        total_w += w

    if 0 < total_w < 1.0:
        signals /= total_w
    return signals, primary_df


def run_pair(pair: str, config_path: Path, risk_base: str = "half") -> dict:
    """Run signal-only portfolio for one pair on HALF_CASH. Returns metrics dict.

    risk_base controls position sizing:
      'half'  — risk 1% of $50k per trade ($500) — default, conservative
      'full'  — risk 1% of $100k per trade ($1000) — 2x leverage on same capital
    """
    cfg = load_config(config_path)
    threshold = cfg.get("confirmation_threshold", 0.10)
    stop_atr_mult = float(cfg.get("atr_stop_mult", 2.0))
    swap_pct = load_swap_pct(pair)

    confluence, primary_df = compute_confluence(pair, cfg)
    close = primary_df["close"]
    high = primary_df["high"]
    low = primary_df["low"]

    atr = compute_atr(primary_df, ATR_PERIOD)
    conf_sh = confluence.shift(1).fillna(0.0)

    risk_capital = INIT_CASH if risk_base == "full" else HALF_CASH
    risk_amt = risk_capital * RISK_PER_TRADE
    stop_dist = stop_atr_mult * atr
    raw_units = risk_amt / stop_dist
    max_units = (HALF_CASH * MAX_LEVERAGE) / close
    size = np.minimum(raw_units, max_units).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    spread_series = build_spread_series(primary_df, pair)
    avg_spread = float(spread_series.mean())
    # IBKR: 0.20 bps per order, min $2. Proportional model covers typical leg sizes.
    total_fees = avg_spread + 0.00020

    pf = vbt.Portfolio.from_signals(
        close=close,
        high=high,
        low=low,
        entries=conf_sh >= threshold,
        short_entries=conf_sh <= -threshold,
        exits=conf_sh < 0,
        short_exits=conf_sh > 0,
        size=size,
        size_type="amount",
        init_cash=HALF_CASH,
        fees=total_fees,
        freq="4h",
    )

    val = pf.value()
    swap_drag = float((pf.asset_value().abs() * (swap_pct / H4_BARS_PER_YEAR)).sum())

    return {
        "pf": pf,
        "val": val,
        "sharpe": pf.sharpe_ratio(),
        "maxdd": pf.max_drawdown(),
        "total_return": pf.total_return(),
        "adj_return": (val.iloc[-1] - swap_drag - HALF_CASH) / HALF_CASH,
        "trades": pf.trades.count(),
        "final_eq": val.iloc[-1],
        "swap_drag": swap_drag,
        "swap_pct": swap_pct,
    }


def sharpe_from_equity(val: pd.Series) -> float:
    ret = val.pct_change().dropna()
    if ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(H4_BARS_PER_YEAR))


def maxdd_from_equity(val: pd.Series) -> float:
    return float(((val - val.cummax()) / val.cummax()).min())


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def _simulate(pair1: str, pair2: str, risk_base: str) -> dict:
    """Run both pairs and return combined metrics dict."""
    pairs = [(pair1, _resolve_config(pair1)), (pair2, _resolve_config(pair2))]
    results: dict[str, dict] = {}
    for pair, cfg_path in pairs:
        results[pair] = run_pair(pair, cfg_path, risk_base=risk_base)

    r1 = results[pair1]
    r2 = results[pair2]

    aligned = (
        pd.concat(
            [r1["val"].rename("p1"), r2["val"].rename("p2")],
            axis=1,
            join="outer",
        )
        .ffill()
        .dropna()
    )
    val1 = aligned["p1"]
    val2 = aligned["p2"]
    val_combined = val1 + val2

    corr = float(val1.pct_change().dropna().corr(val2.pct_change().dropna()))
    total_swap = r1["swap_drag"] + r2["swap_drag"]

    return {
        "r1": r1,
        "r2": r2,
        "val_combined": val_combined,
        "val1": val1,
        "val2": val2,
        "corr": corr,
        "total_swap": total_swap,
        "combined_return": (val_combined.iloc[-1] - INIT_CASH) / INIT_CASH,
        "combined_adj_return": (val_combined.iloc[-1] - total_swap - INIT_CASH) / INIT_CASH,
        "combined_sharpe": sharpe_from_equity(val_combined),
        "combined_maxdd": maxdd_from_equity(val_combined),
        "combined_final_eq": val_combined.iloc[-1],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MTF two-pair combined portfolio simulation.")
    parser.add_argument("--pair1", default="EUR_USD", help="First pair (default: EUR_USD)")
    parser.add_argument("--pair2", default="USD_CHF", help="Second pair (default: USD_CHF)")
    args = parser.parse_args()

    pair1 = args.pair1.upper()
    pair2 = args.pair2.upper()
    label = f"{pair1.replace('_', '/')} + {pair2.replace('_', '/')}"
    slug = f"{pair1.lower()}_{pair2.lower()}"

    print("=" * 80)
    print(f"  MTF COMBINED PORTFOLIO: {label}")
    print(f"  Capital: ${INIT_CASH:,.0f} total  |  50/50 split (${HALF_CASH:,.0f} each)")
    print("  Signal-only exit  |  Risk: 1% per trade")
    print("=" * 80)

    # ── Run both risk modes ───────────────────────────────────────────
    print(f"\n  [A] Conservative: risk 1% of ${HALF_CASH:,.0f} per pair = $500/trade each")
    sim_half = _simulate(pair1, pair2, risk_base="half")

    print(f"\n  [B] Full risk:    risk 1% of ${INIT_CASH:,.0f} per pair = $1,000/trade each")
    sim_full = _simulate(pair1, pair2, risk_base="full")

    # ── Correlation (same for both — sizing doesn't affect signal timing) ──
    corr = sim_half["corr"]  # identical in both modes

    # ── Print comparison table ────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  RESULTS  |  Correlation of H4 returns: {corr:+.4f}")
    print("=" * 80)

    w = 20
    print(
        f"\n  {'Metric':<26} {'[A] Conservative':>{w}} {'[B] Full Risk (2% total)':>{w}}"
        f"  {'Diff':>10}"
    )
    sep = "  " + "-" * (26 + w * 2 + 14)
    print(sep)

    def row(lbl: str, va: str, vb: str, diff: str = "") -> None:
        print(f"  {lbl:<26} {va:>{w}} {vb:>{w}}  {diff:>10}")

    a, b = sim_half, sim_full

    row(
        "Combined Total Return",
        f"{a['combined_return']:.2%}",
        f"{b['combined_return']:.2%}",
        f"{b['combined_return'] - a['combined_return']:+.2%}",
    )
    row(
        "Combined Adj Return (swap)",
        f"{a['combined_adj_return']:.2%}",
        f"{b['combined_adj_return']:.2%}",
        f"{b['combined_adj_return'] - a['combined_adj_return']:+.2%}",
    )
    row(
        "Combined Sharpe",
        f"{a['combined_sharpe']:.3f}",
        f"{b['combined_sharpe']:.3f}",
        f"{b['combined_sharpe'] - a['combined_sharpe']:+.3f}",
    )
    row(
        "Combined Max Drawdown",
        f"{a['combined_maxdd']:.2%}",
        f"{b['combined_maxdd']:.2%}",
        f"{b['combined_maxdd'] - a['combined_maxdd']:+.2%}",
    )
    row(
        "Combined Final Equity",
        f"${a['combined_final_eq']:,.0f}",
        f"${b['combined_final_eq']:,.0f}",
        f"${b['combined_final_eq'] - a['combined_final_eq']:+,.0f}",
    )
    print(sep)
    row(
        "Total Swap Cost",
        f"-${a['total_swap']:,.0f}",
        f"-${b['total_swap']:,.0f}",
        f"${b['total_swap'] - a['total_swap']:+,.0f}",
    )

    print(f"\n  {pair1.replace('_', '/')} individual:")
    row(
        "  Sharpe",
        f"{a['r1']['sharpe']:.3f}",
        f"{b['r1']['sharpe']:.3f}",
        f"{b['r1']['sharpe'] - a['r1']['sharpe']:+.3f}",
    )
    row(
        "  Max Drawdown",
        f"{a['r1']['maxdd']:.2%}",
        f"{b['r1']['maxdd']:.2%}",
        f"{b['r1']['maxdd'] - a['r1']['maxdd']:+.2%}",
    )
    print(f"\n  {pair2.replace('_', '/')} individual:")
    row(
        "  Sharpe",
        f"{a['r2']['sharpe']:.3f}",
        f"{b['r2']['sharpe']:.3f}",
        f"{b['r2']['sharpe'] - a['r2']['sharpe']:+.3f}",
    )
    row(
        "  Max Drawdown",
        f"{a['r2']['maxdd']:.2%}",
        f"{b['r2']['maxdd']:.2%}",
        f"{b['r2']['maxdd'] - a['r2']['maxdd']:+.2%}",
    )

    print("""
  Summary:
    Sharpe is scale-invariant — doubling position size leaves it unchanged.
    Returns roughly double (2x leverage on same capital base).
    MaxDD roughly doubles — you are taking on 2% total portfolio risk per bar.
    Swap costs roughly double (larger average position value).
    Correlation is unchanged — signal timing is independent of size.
    """)

    # ── Plot: both modes + individuals ───────────────────────────────
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=a["val_combined"].index,
            y=a["val_combined"],
            name="[A] Combined — conservative (1% of $50k)",
            line={"color": "steelblue", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=b["val_combined"].index,
            y=b["val_combined"],
            name="[B] Combined — full risk (1% of $100k)",
            line={"color": "seagreen", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=a["val1"].index,
            y=a["val1"] * 2,
            name=f"{pair1.replace('_', '/')} only (scaled, conservative)",
            line={"color": "royalblue", "dash": "dot", "width": 1.2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=a["val2"].index,
            y=a["val2"] * 2,
            name=f"{pair2.replace('_', '/')} only (scaled, conservative)",
            line={"color": "tomato", "dash": "dot", "width": 1.2},
        )
    )
    fig.update_layout(
        title=(f"{label}  |  Conservative vs Full-Risk  |  Corr={corr:+.3f}"),
        yaxis_title="Equity ($)",
        xaxis_title="Date",
        legend={"orientation": "h", "y": -0.2},
        hovermode="x unified",
    )
    html_path = REPORTS_DIR / f"mtf_combined_{slug}.html"
    fig.write_html(str(html_path))
    print(f"  Report saved: {html_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
