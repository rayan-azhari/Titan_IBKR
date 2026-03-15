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

PAIRS = [
    ("EUR_USD", PROJECT_ROOT / "config" / "mtf_eurusd.toml"),
    ("USD_CHF", PROJECT_ROOT / "config" / "mtf_usdchf.toml"),
]


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


def run_pair(pair: str, config_path: Path) -> dict:
    """Run signal-only portfolio for one pair on HALF_CASH. Returns metrics dict."""
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

    risk_amt = HALF_CASH * RISK_PER_TRADE
    stop_dist = stop_atr_mult * atr
    raw_units = risk_amt / stop_dist
    max_units = (HALF_CASH * MAX_LEVERAGE) / close
    size = np.minimum(raw_units, max_units).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    spread_series = build_spread_series(primary_df, pair)
    avg_spread = float(spread_series.mean())

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
        fees=avg_spread,
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


def main() -> None:
    print("=" * 72)
    print("  MTF COMBINED PORTFOLIO: EUR/USD + USD/CHF")
    print(f"  Total capital: ${INIT_CASH:,.0f}  |  Split: 50/50 (${HALF_CASH:,.0f} each)")
    print("  Exit: signal-only  |  Risk: 1% equity per trade per pair")
    print("=" * 72)

    results: dict[str, dict] = {}
    for pair, cfg_path in PAIRS:
        print(f"\n  Computing {pair}...")
        results[pair] = run_pair(pair, cfg_path)
        r = results[pair]
        print(
            f"    Sharpe={r['sharpe']:.3f}  MaxDD={r['maxdd']:.2%}"
            f"  Return={r['total_return']:.2%}  Trades={r['trades']}"
        )

    eur = results["EUR_USD"]
    chf = results["USD_CHF"]

    # ── Align via outer join + ffill (H4 timestamps differ per pair) ──
    aligned = (
        pd.concat(
            [eur["val"].rename("eur"), chf["val"].rename("chf")],
            axis=1,
            join="outer",
        )
        .ffill()
        .dropna()
    )
    val_eur = aligned["eur"]
    val_chf = aligned["chf"]
    val_combined = val_eur + val_chf

    # ── Correlation of bar-level returns ─────────────────────────────
    ret_eur = val_eur.pct_change().dropna()
    ret_chf = val_chf.pct_change().dropna()
    corr = float(ret_eur.corr(ret_chf))

    # ── Combined metrics ──────────────────────────────────────────────
    total_swap = eur["swap_drag"] + chf["swap_drag"]
    combined_return = (val_combined.iloc[-1] - INIT_CASH) / INIT_CASH
    combined_adj_return = (val_combined.iloc[-1] - total_swap - INIT_CASH) / INIT_CASH
    combined_sharpe = sharpe_from_equity(val_combined)
    combined_maxdd = maxdd_from_equity(val_combined)

    # ── Individual metrics scaled to $100k for fair comparison ────────
    # Sharpe is scale-invariant; return/DD are already correct.
    # Final equity is on $50k base — scale x2 for $100k comparison.
    def scaled_final(r: dict) -> float:
        return r["final_eq"] * 2

    # ── Print table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  DIVERSIFICATION ANALYSIS")
    print("=" * 80)
    print(f"\n  H4-bar return correlation:  {corr:+.4f}")
    if corr < -0.15:
        verdict = "GENUINE negative correlation — expect meaningful DD reduction."
    elif corr < 0.05:
        verdict = "Near-zero — modest smoothing, limited DD benefit."
    else:
        verdict = "Positive correlation — pairs move together. Minimal diversification."
    print(f"  Interpretation: {verdict}\n")

    w = 18
    h = f"  {'Metric':<24} {'EUR/USD':>{w}} {'USD/CHF':>{w}} {'Combined':>{w}}"
    sep = "  " + "-" * (24 + w * 3 + 6)
    note = "(each $50k)   (each $50k)  ($100k total)"
    print(f"  {'':24} {note}")
    print(h)
    print(sep)

    def row(label: str, v1: str, v2: str, vc: str) -> None:
        print(f"  {label:<24} {v1:>{w}} {v2:>{w}} {vc:>{w}}")

    row(
        "Total Return",
        f"{eur['total_return']:.2%}",
        f"{chf['total_return']:.2%}",
        f"{combined_return:.2%}",
    )
    row(
        "Adj Return (swap)",
        f"{eur['adj_return']:.2%}",
        f"{chf['adj_return']:.2%}",
        f"{combined_adj_return:.2%}",
    )
    row(
        "Sharpe",
        f"{eur['sharpe']:.3f}",
        f"{chf['sharpe']:.3f}",
        f"{combined_sharpe:.3f}",
    )
    row(
        "Max Drawdown",
        f"{eur['maxdd']:.2%}",
        f"{chf['maxdd']:.2%}",
        f"{combined_maxdd:.2%}",
    )
    row(
        "Trades",
        str(eur["trades"]),
        str(chf["trades"]),
        str(eur["trades"] + chf["trades"]),
    )
    row(
        "Final Equity",
        f"${eur['final_eq']:,.0f}",
        f"${chf['final_eq']:,.0f}",
        f"${val_combined.iloc[-1]:,.0f}",
    )
    print(sep)
    row(
        "Swap Cost ($)",
        f"-${eur['swap_drag']:,.0f}",
        f"-${chf['swap_drag']:,.0f}",
        f"-${total_swap:,.0f}",
    )

    # ── Diversification verdict ───────────────────────────────────────
    best_ind_dd = max(eur["maxdd"], chf["maxdd"])  # max = least negative = smaller DD
    dd_delta = abs(combined_maxdd) - abs(best_ind_dd)
    sharpe_delta = combined_sharpe - max(eur["sharpe"], chf["sharpe"])

    print("\n  -- Diversification verdict ------------------------------------------")
    print(f"  Correlation:          {corr:+.4f}")
    print(
        f"  Combined MaxDD:       {combined_maxdd:.2%}  "
        f"vs best individual: {best_ind_dd:.2%}  "
        f"({'better' if dd_delta < 0 else 'worse'} by {abs(dd_delta):.2%})"
    )
    print(
        f"  Combined Sharpe:      {combined_sharpe:.3f}  "
        f"vs best individual: {max(eur['sharpe'], chf['sharpe']):.3f}  "
        f"({'better' if sharpe_delta > 0 else 'worse'} by {abs(sharpe_delta):.3f})"
    )

    if corr < -0.15 and dd_delta < 0:
        conclusion = "REAL — negative correlation materially reduces drawdown."
    elif corr < 0.05 and dd_delta < 0:
        conclusion = "MODEST — near-zero correlation provides some smoothing."
    else:
        conclusion = "ILLUSORY — pairs too correlated to provide meaningful benefit."
    print(f"\n  Conclusion: {conclusion}")
    print("  " + "-" * 68)

    # ── Plot ──────────────────────────────────────────────────────────
    fig = go.Figure()
    # Scale individuals to $100k for visual comparison
    fig.add_trace(
        go.Scatter(
            x=val_eur.index,
            y=val_eur * 2,
            name="EUR/USD (scaled to $100k)",
            line={"color": "royalblue", "dash": "dot", "width": 1.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=val_chf.index,
            y=val_chf * 2,
            name="USD/CHF (scaled to $100k)",
            line={"color": "tomato", "dash": "dot", "width": 1.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=val_combined.index,
            y=val_combined,
            name="Combined 50/50",
            line={"color": "seagreen", "width": 2.5},
        )
    )
    fig.update_layout(
        title=(
            f"EUR/USD + USD/CHF Combined Portfolio  |  "
            f"Corr={corr:+.3f}  |  "
            f"Combined Sharpe={combined_sharpe:.3f}  MaxDD={combined_maxdd:.2%}"
        ),
        yaxis_title="Equity ($)",
        xaxis_title="Date",
        legend={"orientation": "h", "y": -0.15},
        hovermode="x unified",
    )

    html_path = REPORTS_DIR / "mtf_combined_eurusd_usdchf.html"
    fig.write_html(str(html_path))
    print(f"\n  Report saved: {html_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
