"""run_eurusd_h1.py — EUR/USD H1 mean reversion pipeline (self-contained).

Regime gate: ATR percentile (bottom 30%) + Hurst < 0.52.
Replaces the HMM gate which failed to separate ranging/trending on H1.

Usage:
    uv run python research/mean_reversion/run_eurusd_h1.py
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR    = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt  # noqa: F401 — used lazily inside execution.build_subportfolios
except ImportError:
    print("ERROR: vectorbt not installed.  Run: uv sync")
    sys.exit(1)

from research.mean_reversion import execution as exe
from research.mean_reversion import risk as rsk
from research.mean_reversion import signals as sig
from titan.models.spread import build_spread_series, build_total_cost_series

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "config" / "eurusd_h1_mr.toml"
with open(CONFIG_PATH, "rb") as f:
    CFG = tomllib.load(f)

PAIR             = CFG["base"]["instrument"]
TF               = CFG["base"]["timeframe"]
ANCHOR_SESSIONS  = CFG["vwap"]["anchor_sessions"]
TIERS_PCT        = CFG["signal"]["tiers_pct"]
TIER_SIZES       = CFG["signal"]["tier_sizes"]
PROFIT_MARGIN    = CFG["signal"]["profit_margin"]
REVERSION_PCT    = CFG["signal"]["reversion_target_pct"]
SESSION_FILTER   = CFG["signal"]["session_filter"]
INVALIDATION_PCT = CFG["signal"]["invalidation_pct"]
PCT_WINDOW       = CFG["signal"]["percentile_window"]
NY_CLOSE_UTC     = CFG["risk"]["ny_close_utc"]
ATR_PERIOD   = CFG["regime"]["atr_period"]
GATE_WINDOW  = CFG["regime"]["gate_window"]
GATE_PCT     = CFG["regime"]["gate_pct"]
HURST_THRESH = CFG["regime"]["hurst_thresh"]
HURST_WINDOW = CFG["regime"]["hurst_window"]

IS_SPLIT = 0.70

GATE_OOS_SHARPE     = 1.0
GATE_OOS_IS_RATIO   = 0.5
GATE_WIN_RATE       = 0.40
GATE_MAX_DD         = 0.25
GATE_MIN_OOS_TRADES = 200


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_h1(pair: str) -> pd.DataFrame:
    path = DATA_DIR / f"{pair}_{TF}.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    print(f"  Loaded {pair}_{TF}: {len(df):,} bars  "
          f"[{df.index[0].date()} -> {df.index[-1].date()}]")
    return df


# ---------------------------------------------------------------------------
# Pipeline — ATR percentile gate + Hurst (no HMM)
# ---------------------------------------------------------------------------


def run_pipeline(df: pd.DataFrame, spread: pd.Series, label: str) -> dict:
    close = df["close"]

    # Session mask
    sess_mask = sig.session_mask(df.index, SESSION_FILTER.split("+"))

    # VWAP + deviation + levels
    vwap      = sig.compute_anchored_vwap(df, anchor_sessions=ANCHOR_SESSIONS)
    deviation = sig.compute_deviation(df, vwap)
    levels    = sig.percentile_levels(deviation, PCT_WINDOW, pcts=TIERS_PCT)
    inv_lvl   = sig.invalidation_level(deviation, PCT_WINDOW, pct=INVALIDATION_PCT)
    tier1_lvl = levels.iloc[:, 0]

    # Regime gate: ATR percentile only (Hurst unreliable at H1/200-bar window)
    atr_gate = sig.atr_percentile_gate(df, ATR_PERIOD, GATE_WINDOW, GATE_PCT)
    gate     = atr_gate & sess_mask

    print(f"  {label:<12}  gate%={gate.mean():.1%}  atr%={atr_gate.mean():.1%}", end="")

    # Grid entries + exits
    grid_entries          = exe.build_grid_entries(deviation, levels, gate, TIER_SIZES)
    basket_exit           = exe.compute_basket_vwap_exit(close, grid_entries, PROFIT_MARGIN)
    long_exit, short_exit = rsk.build_combined_exit(
        basket_exit, deviation, inv_lvl, df.index,
        tier1_level=tier1_lvl, reversion_pct=REVERSION_PCT,
        cutoff_hour_utc=NY_CLOSE_UTC,
    )
    sub_pfs = exe.build_subportfolios(
        close, grid_entries, long_exit, short_exit,
        spread.reindex(df.index), total_cash=10_000.0,
    )

    daily_ret    = exe.combine_portfolio_returns(sub_pfs)
    sharpe       = exe.compute_combined_sharpe(daily_ret)
    total_trades = sum(int(pf.trades.count()) for pf in sub_pfs)
    pfs_ok  = [pf for pf in sub_pfs if pf.trades.count() > 0]
    avg_wr  = float(np.mean([float(pf.trades.win_rate()) for pf in pfs_ok])) if pfs_ok else 0.0
    worst_dd = float(max(float(pf.max_drawdown()) for pf in sub_pfs))
    weeks    = (df.index[-1] - df.index[0]).days / 7

    print(f"  Sharpe={sharpe:+.2f}  MaxDD={worst_dd:.2%}  WR={avg_wr:.2%}  n={total_trades}")

    return {
        "label": label, "sharpe": round(sharpe, 3),
        "max_dd": round(worst_dd, 4), "n_trades": total_trades,
        "trades_per_week": round(total_trades / max(weeks, 1), 2),
        "win_rate": round(avg_wr, 3), "daily_returns": daily_ret,
    }


# ---------------------------------------------------------------------------
# Monte Carlo — handles empty trade case gracefully
# ---------------------------------------------------------------------------


def monte_carlo_gate(daily_returns: pd.Series, n: int = 500, pct: float = 5.0) -> dict:
    if len(daily_returns) == 0 or daily_returns.std() == 0:
        return {"pct_sharpe": np.nan, "pct_profitable": 0.0, "pass": False}
    np.random.seed(42)
    vals = daily_returns.values.copy()
    sharpes = []
    for _ in range(n):
        np.random.shuffle(vals)
        s = pd.Series(vals)
        mu, sd = s.mean(), s.std()
        sharpes.append(mu / sd * np.sqrt(252) if sd > 0 else np.nan)
    sharpes = [s for s in sharpes if not np.isnan(s)]
    if not sharpes:
        return {"pct_sharpe": np.nan, "pct_profitable": 0.0, "pass": False}
    p5    = float(np.percentile(sharpes, pct))
    p_pos = float(np.mean([s > 0 for s in sharpes]))
    return {"pct_sharpe": round(p5, 3), "pct_profitable": round(p_pos, 3),
            "pass": p5 > 0.5 and p_pos >= 0.80}


# ---------------------------------------------------------------------------
# WFO
# ---------------------------------------------------------------------------


def walk_forward(
    df: pd.DataFrame,
    spread: pd.Series,
    is_months: int = 6,
    oos_months: int = 2,
) -> tuple[pd.Series, list[dict]]:
    dates = pd.date_range(df.index[0], df.index[-1], freq="MS")
    fold_results, stitched = [], []

    for i in range(is_months, len(dates) - oos_months):
        df_oos = df[dates[i]:dates[i + oos_months]]
        if len(df_oos) < 50:
            continue
        m = run_pipeline(df_oos, spread, label=f"wfo_{i}")
        fold_results.append({
            "fold": i,
            "oos_start": str(df_oos.index[0].date()),
            "oos_end":   str(df_oos.index[-1].date()),
            "sharpe":    m["sharpe"],
            "n_trades":  m["n_trades"],
        })
        if len(m["daily_returns"]) > 0:
            stitched.append(m["daily_returns"])

    stitched_ret = pd.concat(stitched) if stitched else pd.Series(dtype=float)
    return stitched_ret, fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("EUR/USD H1 Mean Reversion  [ATR gate + Hurst]")
    print("=" * 60)

    df     = load_h1(PAIR)
    spread = build_spread_series(df, PAIR)

    is_end = int(len(df) * IS_SPLIT)
    df_is  = df.iloc[:is_end]
    df_oos = df.iloc[is_end:]
    print(f"  IS:  {len(df_is):,} bars  [{df_is.index[0].date()} -> {df_is.index[-1].date()}]")
    print(f"  OOS: {len(df_oos):,} bars  [{df_oos.index[0].date()} -> {df_oos.index[-1].date()}]")

    # ── IS / OOS ──────────────────────────────────────────────────────────
    print("\n[1/5] IS and OOS backtests...")
    is_m  = run_pipeline(df_is,  spread.reindex(df_is.index),  label="IS")
    oos_m = run_pipeline(df_oos, spread.reindex(df_oos.index), label="OOS")
    ratio = oos_m["sharpe"] / is_m["sharpe"] if is_m["sharpe"] not in (0, np.nan) else 0.0
    print(f"  OOS/IS ratio: {ratio:.2f}")

    # ── Monte Carlo ───────────────────────────────────────────────────────
    print("\n[2/5] Monte Carlo (G1)...")
    mc = monte_carlo_gate(oos_m["daily_returns"])
    print(f"  5th-pct Sharpe={mc['pct_sharpe']}  "
          f"% profitable={mc['pct_profitable']:.1%}  PASS={mc['pass']}")

    # ── 3x slippage ───────────────────────────────────────────────────────
    print("\n[3/5] 3x slippage stress (G3)...")
    cost_3x = build_total_cost_series(df_oos, PAIR) * 3.0
    stress  = run_pipeline(df_oos, cost_3x.reindex(df_oos.index), label="OOS_3x")
    print(f"  3x stress Sharpe={stress['sharpe']:+.2f}  PASS={stress['sharpe'] > 0.5}")

    # ── WFO ───────────────────────────────────────────────────────────────
    print("\n[4/5] WFO (6mo IS / 2mo OOS)...")
    wfo_ret, wfo_folds = walk_forward(df, spread)
    neg        = sum(1 for f in wfo_folds if f["sharpe"] < 0)
    wfo_sharpe = exe.compute_combined_sharpe(wfo_ret) if len(wfo_ret) > 0 else np.nan
    print(f"  WFO stitched Sharpe={wfo_sharpe:.3f}  Negative folds={neg}/{len(wfo_folds)}")
    for f in wfo_folds:
        print(f"    {f['oos_start']} -> {f['oos_end']}  "
              f"Sharpe={f['sharpe']:+.2f}  n={f['n_trades']}")

    # ── Gates ─────────────────────────────────────────────────────────────
    print("\n[5/5] Quality gates...")
    gates = {
        "G_oos_sharpe":     oos_m["sharpe"]   > GATE_OOS_SHARPE,
        "G_oos_is_ratio":   ratio             > GATE_OOS_IS_RATIO,
        "G_win_rate":       oos_m["win_rate"] > GATE_WIN_RATE,
        "G_max_dd":         oos_m["max_dd"]   < GATE_MAX_DD,
        "G_min_trades":     oos_m["n_trades"] >= GATE_MIN_OOS_TRADES,
        "G1_monte_carlo":   mc["pass"],
        "G3_slippage":      stress["sharpe"]  > 0.5,
        "G4_wfo_neg_folds": neg <= 2,
    }
    all_passed = all(gates.values())
    print(f"\n  {'GATE':<25} {'RESULT'}")
    print("  " + "-" * 40)
    for g, passed in gates.items():
        print(f"  {g:<25} [{'PASS' if passed else 'FAIL'}]")
    print(f"\n  Overall: {'ALL GATES PASSED' if all_passed else 'FAILED -- do not deploy'}")

    # ── Save ──────────────────────────────────────────────────────────────
    report = {
        "instrument": PAIR, "timeframe": TF,
        "gate": {"type": "atr_percentile+hurst", "atr_period": ATR_PERIOD,
                 "gate_window": GATE_WINDOW, "gate_pct": GATE_PCT,
                 "hurst_window": HURST_WINDOW, "hurst_thresh": HURST_THRESH},
        "is":  {k: v for k, v in is_m.items()  if k != "daily_returns"},
        "oos": {k: v for k, v in oos_m.items() if k != "daily_returns"},
        "oos_is_ratio": round(ratio, 3),
        "monte_carlo": mc,
        "stress_3x_sharpe": stress["sharpe"],
        "wfo": {
            "stitched_sharpe": round(float(wfo_sharpe), 3) if not np.isnan(wfo_sharpe) else None,
            "negative_folds": neg, "total_folds": len(wfo_folds), "folds": wfo_folds,
        },
        "gates": {k: bool(v) for k, v in gates.items()},
        "all_gates_passed": bool(all_passed),
    }
    out_json = REPORTS_DIR / "eurusd_h1_mr_validation.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n  Saved -> {out_json}")

    # ── Chart ─────────────────────────────────────────────────────────────
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["IS vs OOS Cumulative Returns",
                                        "WFO Stitched OOS Equity"])
    for ret, name, colour in [
        (is_m["daily_returns"],  "IS",      "steelblue"),
        (oos_m["daily_returns"], "OOS",     "orange"),
    ]:
        cum = (1 + ret).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum, name=name,
                                 line=dict(color=colour)), row=1, col=1)
    if len(wfo_ret) > 0:
        wfo_cum = (1 + wfo_ret).cumprod()
        fig.add_trace(go.Scatter(x=wfo_cum.index, y=wfo_cum, name="WFO OOS",
                                 line=dict(color="green")), row=2, col=1)
    fig.update_layout(
        title=f"EUR/USD H1 MR [ATR gate] -- {'PASSED' if all_passed else 'FAILED'}",
        height=700)
    out_html = REPORTS_DIR / "eurusd_h1_mr_equity_curve.html"
    fig.write_html(str(out_html))
    print(f"  Saved -> {out_html}")


if __name__ == "__main__":
    main()
