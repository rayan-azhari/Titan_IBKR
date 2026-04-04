"""run_stage1_signals.py — VWAP deviation distribution explorer.

Stage 1 goal: understand the raw signal before adding the regime filter.
  - Compute anchored VWAP and price deviation series.
  - Compare two entry level methods: empirical percentiles vs ATR multiples.
  - Sweep percentile_window to find stable threshold levels.
  - Report trade frequency per tier (we want >= 5 trades/week at Tier 1).
  - Run a simplified single-tier backtest (90th pct entry, mean exit) to
    confirm there is a raw edge before investing in the HMM.

Outputs:
  .tmp/reports/eurusd_mr_deviation_dist.html    ← deviation histogram
  .tmp/reports/eurusd_mr_stage1_summary.csv     ← metrics per param combo
  .tmp/mr_state_eurusd.json                     ← best params persisted

Usage:
    uv run python research/mean_reversion/run_stage1_signals.py
"""

from __future__ import annotations

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
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.  Run: uv sync")
    sys.exit(1)

from research.mean_reversion import signals as sig
from research.mean_reversion import state_manager as sm
from titan.models.spread import build_spread_series

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "config" / "eurusd_mr.toml"
with open(CONFIG_PATH, "rb") as f:
    CFG = tomllib.load(f)

PAIR = CFG["base"]["instrument"]  # "EUR_USD"
TF = CFG["base"]["timeframe"]  # "M5"
ANCHOR_SESSIONS = CFG["vwap"]["anchor_sessions"]
TIERS_PCT = CFG["signal"]["tiers_pct"]
INVALIDATION_PCT = CFG["signal"]["invalidation_pct"]

# Sweep grid
PCT_WINDOWS = [500, 1000, 2000, 3000, 5000]  # bars for rolling percentile
METHODS = ["percentile", "atr"]
ATR_MULTS = CFG["atr"]["mults"]
ATR_PERIOD = CFG["atr"]["period"]

# Simplified single-tier backtest: entry at tier1 pct, exit at mean (deviation -> 0)
TIER1_PCT = TIERS_PCT[0]  # 0.90

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_m5(pair: str) -> pd.DataFrame:
    path = DATA_DIR / f"{pair}_{TF}.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.  Run: uv run python scripts/download_fx_m5.py")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    print(
        f"  Loaded {pair}_{TF}: {len(df):,} bars  [{df.index[0].date()} -> {df.index[-1].date()}]"
    )
    return df


# ---------------------------------------------------------------------------
# Single-tier VectorBT backtest helper
# ---------------------------------------------------------------------------


def run_single_tier_backtest(
    close: pd.Series,
    short_entry: pd.Series,
    long_entry: pd.Series,
    short_exit: pd.Series,
    long_exit: pd.Series,
    spread: pd.Series,
    label: str,
) -> dict:
    pf = vbt.Portfolio.from_signals(
        close,
        entries=long_entry,
        exits=long_exit,
        short_entries=short_entry,
        short_exits=short_exit,
        init_cash=10_000,
        fees=spread.values,
        freq="5min",
    )
    n_trades = pf.trades.count()
    weeks = (close.index[-1] - close.index[0]).days / 7
    return {
        "label": label,
        "sharpe": round(float(pf.sharpe_ratio()), 3),
        "total_return": round(float(pf.total_return()), 4),
        "max_dd": round(float(pf.max_drawdown()), 4),
        "n_trades": int(n_trades),
        "trades_per_week": round(n_trades / max(weeks, 1), 2),
        "win_rate": round(float(pf.trades.win_rate()), 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("Stage 1 — Signal Exploration: Anchored VWAP Deviation")
    print("=" * 60)

    df = load_m5(PAIR)
    close = df["close"]
    spread = build_spread_series(df, PAIR)

    # ── Anchored VWAP and deviation ──────────────────────────────────────
    print("\n[1/4] Computing anchored VWAP and deviation series...")
    vwap = sig.compute_anchored_vwap(df, anchor_sessions=ANCHOR_SESSIONS)
    deviation = sig.compute_deviation(df, vwap)
    print(f"  Deviation  mean={deviation.mean():.6f}  std={deviation.std():.6f}")
    print(
        f"  Deviation  90th pct={deviation.quantile(0.90):.5f}  "
        f"99th pct={deviation.quantile(0.99):.5f}  "
        f"99.9th pct={deviation.quantile(0.999):.5f}"
    )

    # ── Percentile stability sweep ────────────────────────────────────────
    print("\n[2/4] Percentile stability sweep across windows...")
    stability_rows = []
    for w in PCT_WINDOWS:
        lvls = sig.percentile_levels(deviation, w, pcts=TIERS_PCT)
        inv = sig.invalidation_level(deviation, w, pct=INVALIDATION_PCT)
        for pct_col in lvls.columns:
            s = lvls[pct_col].dropna()
            if len(s) == 0:
                continue
            stability_rows.append(
                {
                    "window": w,
                    "percentile": pct_col,
                    "mean_level": round(s.mean(), 6),
                    "std_level": round(s.std(), 6),
                    "cv": round(s.std() / s.mean(), 4) if s.mean() != 0 else np.nan,
                }
            )
        s_inv = inv.dropna()
        stability_rows.append(
            {
                "window": w,
                "percentile": INVALIDATION_PCT,
                "mean_level": round(s_inv.mean(), 6),
                "std_level": round(s_inv.std(), 6),
                "cv": round(s_inv.std() / s_inv.mean(), 4) if s_inv.mean() != 0 else np.nan,
            }
        )
    stab_df = pd.DataFrame(stability_rows)
    print(stab_df.to_string(index=False))

    # ── Single-tier backtests: percentile method ──────────────────────────
    print("\n[3/4] Single-tier backtests (Tier 1 only, entry=pct exit=mean)...")
    results = []
    for w in PCT_WINDOWS:
        lvls = sig.percentile_levels(deviation, w, pcts=[TIER1_PCT])
        tier1_lvl = lvls[TIER1_PCT]
        inv = sig.invalidation_level(deviation, w)

        short_entry = deviation > tier1_lvl
        short_exit = (deviation < 0) | (deviation.abs() > inv.abs())
        long_entry = deviation < -tier1_lvl
        long_exit = (deviation > 0) | (deviation.abs() > inv.abs())

        row = run_single_tier_backtest(
            close,
            short_entry,
            long_entry,
            short_exit,
            long_exit,
            spread,
            label=f"pct_w{w}",
        )
        row["method"] = "percentile"
        row["window"] = w
        results.append(row)
        print(
            f"  pct_w{w:5d}  Sharpe={row['sharpe']:+.2f}  "
            f"n_trades={row['n_trades']:5d}  "
            f"trades/wk={row['trades_per_week']:.1f}  "
            f"WR={row['win_rate']:.2%}"
        )

    # ── Single-tier backtest: ATR method ──────────────────────────────────
    from titan.strategies.ml.features import atr as compute_atr

    atr_series = compute_atr(df, ATR_PERIOD).shift(1)
    tier1_atr_lvl = vwap + ATR_MULTS[0] * atr_series
    inv_atr = vwap + ATR_MULTS[-1] * atr_series * 2  # rough invalidation

    short_entry_atr = close > tier1_atr_lvl
    short_exit_atr = (close < vwap) | (close > inv_atr)
    long_entry_atr = close < (vwap - ATR_MULTS[0] * atr_series)
    long_exit_atr = (close > vwap) | (close < (vwap - ATR_MULTS[-1] * atr_series * 2))

    atr_row = run_single_tier_backtest(
        close,
        short_entry_atr,
        long_entry_atr,
        short_exit_atr,
        long_exit_atr,
        spread,
        label="atr_tier1",
    )
    atr_row["method"] = "atr"
    atr_row["window"] = ATR_PERIOD
    results.append(atr_row)
    print(
        f"  atr_tier1  Sharpe={atr_row['sharpe']:+.2f}  "
        f"n_trades={atr_row['n_trades']:5d}  "
        f"trades/wk={atr_row['trades_per_week']:.1f}  "
        f"WR={atr_row['win_rate']:.2%}"
    )

    # ── Save outputs ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "eurusd_mr_stage1_summary.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Saved results -> {out_csv}")

    # Persist best percentile window
    best_pct_row = (
        results_df[results_df["method"] == "percentile"]
        .sort_values("sharpe", ascending=False)
        .iloc[0]
    )
    sm.save_stage1(
        method="percentile",
        vwap_window=0,  # VWAP uses session anchor, no fixed window
        best_pct_window=int(best_pct_row["window"]),
        best_pct=TIER1_PCT,
        sharpe=float(best_pct_row["sharpe"]),
    )
    print(
        f"  Best percentile window: {int(best_pct_row['window'])}  "
        f"Sharpe={float(best_pct_row['sharpe']):.3f}"
    )

    # ── Deviation distribution plot ───────────────────────────────────────
    print("\n[4/4] Plotting deviation distribution...")
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=["VWAP Deviation Distribution", "Sharpe by Window"]
    )

    fig.add_trace(
        go.Histogram(
            x=deviation.dropna().values,
            nbinsx=200,
            name="Deviation",
            marker_color="steelblue",
            opacity=0.75,
        ),
        row=1,
        col=1,
    )

    # Mark percentile levels on histogram
    best_w = int(best_pct_row["window"])
    best_lvls = sig.percentile_levels(deviation, best_w, TIERS_PCT)
    colors = ["#f39c12", "#e74c3c", "#8e44ad", "#c0392b", "#2c3e50"]
    for pct_col, color in zip(best_lvls.columns, colors):
        lvl = float(best_lvls[pct_col].dropna().mean())
        for sign, side in [(1, "short"), (-1, "long")]:
            fig.add_vline(
                x=sign * lvl,
                line_dash="dash",
                line_color=color,
                opacity=0.7,
                annotation_text=f"{side} {int(pct_col * 100)}th pct",
                row=1,
                col=1,
            )

    # Sharpe by window (percentile method only)
    pct_results = results_df[results_df["method"] == "percentile"]
    fig.add_trace(
        go.Bar(
            x=pct_results["window"].astype(str),
            y=pct_results["sharpe"],
            name="Sharpe",
            marker_color="teal",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(title="Stage 1 — VWAP Deviation Analysis", height=700)
    out_html = REPORTS_DIR / "eurusd_mr_deviation_dist.html"
    fig.write_html(str(out_html))
    print(f"  Saved chart  -> {out_html}")

    print("\n[Stage 1 complete]  Run Stage 2 next:")
    print("  uv run python research/mean_reversion/run_stage2_regime.py")


if __name__ == "__main__":
    main()
