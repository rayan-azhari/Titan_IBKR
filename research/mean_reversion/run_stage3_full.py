"""run_stage3_full.py — Full integrated backtest with IS/OOS validation.

Stage 3 goal: wire all components together and validate the complete pipeline.

Full pipeline per bar:
  VWAP deviation -> percentile levels -> regime gate ->
    -> grid entries (exponential ladder) ->
    -> basket VWAP tracking ->
    -> combined exit (TP | hard invalidation | NY close)

Validation protocol (per directives/Backtesting & Validation.md):
  1. IS/OOS split: 70/30
  2. Session filter variants: none / london / london+ny
  3. Full friction: session-aware spread + slippage
  4. Quality gates (all must pass):
     - OOS Sharpe > 1.0
     - OOS/IS ratio > 0.5
     - Win Rate > 40%
     - Max DD < 25%
     - Min 200 OOS trades
  5. WFO: rolling 6-month IS / 2-month OOS windows, stitched equity curve
  6. G1: Monte Carlo trade shuffle (5th-pct Sharpe > 0.5)
  7. G3: 3× slippage stress test (OOS Sharpe > 0.5)

Outputs:
  .tmp/reports/eurusd_mr_validation.json  ← PASS/FAIL per gate
  .tmp/reports/eurusd_mr_equity_curve.html

Usage:
    uv run python research/mean_reversion/run_stage3_full.py

Prerequisite:
    uv run python research/mean_reversion/run_stage2_regime.py
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

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt  # noqa: F401 — used lazily inside execution.build_subportfolios
except ImportError:
    print("ERROR: vectorbt not installed.  Run: uv sync")
    sys.exit(1)

from research.mean_reversion import execution as exe
from research.mean_reversion import regime as reg
from research.mean_reversion import risk as rsk
from research.mean_reversion import signals as sig
from research.mean_reversion import state_manager as sm
from research.mean_reversion.run_stage2_regime import compute_hurst_h1_ffill
from titan.models.spread import build_spread_series, build_total_cost_series

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "config" / "eurusd_mr.toml"
with open(CONFIG_PATH, "rb") as f:
    CFG = tomllib.load(f)

PAIR = CFG["base"]["instrument"]
TF = CFG["base"]["timeframe"]
ANCHOR_SESSIONS = CFG["vwap"]["anchor_sessions"]
TIERS_PCT = CFG["signal"]["tiers_pct"]
TIER_SIZES = CFG["signal"]["tier_sizes"]
PROFIT_MARGIN = CFG["signal"]["profit_margin"]
REVERSION_PCT = CFG["signal"]["reversion_target_pct"]
SESSION_FILTER = CFG["signal"]["session_filter"]
INVALIDATION_PCT = CFG["signal"]["invalidation_pct"]
NY_CLOSE_UTC = CFG["risk"]["ny_close_utc"]

IS_SPLIT = 0.70
MC_N_SHUFFLES = 500
MC_PERCENTILE = 5.0

# Quality gate thresholds
GATE_OOS_SHARPE = 1.0
GATE_OOS_IS_RATIO = 0.5
GATE_WIN_RATE = 0.40
GATE_MAX_DD = 0.25
GATE_MIN_OOS_TRADES = 200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_m5(pair: str) -> pd.DataFrame:
    path = DATA_DIR / f"{pair}_{TF}.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    print(f"  Loaded {pair}_{TF}: {len(df):,} bars")
    return df


# ---------------------------------------------------------------------------
# Full pipeline runner (returns metrics dict)
# ---------------------------------------------------------------------------


def run_full_pipeline(
    df: pd.DataFrame,
    pct_window: int,
    p_thresh: float,
    hurst_thresh: float,
    hmm_model_path: str,
    tier_sizes: list[int],
    profit_margin: float,
    spread: pd.Series,
    label: str = "full",
    session_filter: str = "london_core",
    reversion_pct: float = 0.5,
) -> dict:
    """Run the complete pipeline on a DataFrame slice and return metrics.

    Args:
        df: OHLCV DataFrame slice (IS or OOS).
        pct_window: Rolling window for percentile levels.
        p_thresh: HMM gate threshold.
        hurst_thresh: Hurst gate threshold.
        hmm_model_path: Path to saved HMM model.
        tier_sizes: Lot sizes per tier.
        profit_margin: Basket TP margin in price units (secondary TP).
        spread: Session-aware spread Series (aligned to df.index).
        label: Human-readable label for reporting.
        session_filter: "none", "london_core", "london", "london+ny".
        reversion_pct: Partial-reversion TP fraction (0.5 = 50% reversion).

    Returns:
        Dict of performance metrics.
    """
    close = df["close"]

    # Session mask (restrict entries to active sessions)
    if session_filter == "none":
        sess_mask = pd.Series(True, index=df.index)
    else:
        parts = session_filter.split("+")
        sess_mask = sig.session_mask(df.index, parts)

    # VWAP and deviation
    vwap = sig.compute_anchored_vwap(df, anchor_sessions=ANCHOR_SESSIONS)
    deviation = sig.compute_deviation(df, vwap)

    # Percentile levels
    levels = sig.percentile_levels(deviation, pct_window, pcts=TIERS_PCT)
    inv_lvl = sig.invalidation_level(deviation, pct_window, pct=INVALIDATION_PCT)
    tier1_lvl = levels.iloc[:, 0]  # first (lowest) tier level for reversion TP

    # Regime filter
    obs = reg.build_observations(close)
    model = reg.load_hmm(hmm_model_path)
    ranging_i = reg.ranging_state_index(model)
    post_arr = reg.rolling_regime_posterior(model, obs, ranging_i, min_bars=100)
    post = pd.Series(post_arr, index=df.index)
    hurst = compute_hurst_h1_ffill(df)
    gate = reg.regime_gate(post, hurst, p_thresh=p_thresh, hurst_thresh=hurst_thresh)
    gate = gate & sess_mask

    # Grid entries
    grid_entries = exe.build_grid_entries(deviation, levels, gate, tier_sizes)

    # Basket VWAP exit (secondary TP: close vs average entry price)
    basket_exit = exe.compute_basket_vwap_exit(close, grid_entries, profit_margin, direction="both")

    # Direction-aware combined exits (primary TP: partial reversion)
    long_exit, short_exit = rsk.build_combined_exit(
        basket_exit,
        deviation,
        inv_lvl,
        df.index,
        tier1_level=tier1_lvl,
        reversion_pct=reversion_pct,
        cutoff_hour_utc=NY_CLOSE_UTC,
    )

    # Sub-portfolios
    sub_pfs = exe.build_subportfolios(
        close,
        grid_entries,
        long_exit,
        short_exit,
        spread.reindex(df.index),
        total_cash=10_000.0,
    )

    daily_ret = exe.combine_portfolio_returns(sub_pfs)
    sharpe = exe.compute_combined_sharpe(daily_ret)

    # Aggregate metrics from sub-portfolios
    total_trades = sum(int(pf.trades.count()) for pf in sub_pfs)
    avg_win_rate = float(
        np.mean([float(pf.trades.win_rate()) for pf in sub_pfs if pf.trades.count() > 0])
    )
    worst_dd = float(max(float(pf.max_drawdown()) for pf in sub_pfs))
    weeks = (df.index[-1] - df.index[0]).days / 7

    return {
        "label": label,
        "session_filter": session_filter,
        "sharpe": round(sharpe, 3),
        "max_dd": round(worst_dd, 4),
        "n_trades": total_trades,
        "trades_per_week": round(total_trades / max(weeks, 1), 2),
        "win_rate": round(avg_win_rate, 3),
        "daily_returns": daily_ret,
    }


# ---------------------------------------------------------------------------
# Monte Carlo shuffle (Gate G1)
# ---------------------------------------------------------------------------


def monte_carlo_gate(daily_returns: pd.Series, n_shuffles: int, pct: float) -> dict:
    """Shuffle trade order n_shuffles times, compute Sharpe distribution."""
    sharpes = []
    vals = daily_returns.values.copy()
    for _ in range(n_shuffles):
        np.random.shuffle(vals)
        s = pd.Series(vals)
        mu = s.mean()
        sd = s.std()
        sharpes.append(mu / sd * np.sqrt(252) if sd > 0 else np.nan)
    sharpes = [s for s in sharpes if not np.isnan(s)]
    pct_sharpe = float(np.percentile(sharpes, pct))
    pct_profitable = float(np.mean([s > 0 for s in sharpes]))
    return {
        "pct_sharpe": round(pct_sharpe, 3),
        "pct_profitable": round(pct_profitable, 3),
        "pass": pct_sharpe > 0.5 and pct_profitable >= 0.80,
    }


# ---------------------------------------------------------------------------
# Walk-Forward Optimisation (rolling 6mo IS / 2mo OOS)
# ---------------------------------------------------------------------------


def walk_forward(
    df: pd.DataFrame,
    pct_window: int,
    p_thresh: float,
    hurst_thresh: float,
    hmm_model_path: str,
    tier_sizes: list[int],
    profit_margin: float,
    spread: pd.Series,
    is_months: int = 6,
    oos_months: int = 2,
    session_filter: str = "london_core",
    reversion_pct: float = 0.5,
) -> tuple[pd.Series, list[dict]]:
    """Rolling WFO: stitch OOS equity curves, record per-fold Sharpe.

    Returns:
        stitched_returns: OOS daily returns concatenated across folds.
        fold_results: List of per-fold metric dicts.
    """
    freq = "MS"  # month start
    dates = pd.date_range(df.index[0], df.index[-1], freq=freq)
    fold_results = []
    stitched = []

    for i in range(is_months, len(dates) - oos_months):
        is_start = dates[i - is_months]
        is_end = dates[i]
        oos_end = dates[i + oos_months]

        df_is = df[is_start:is_end]
        df_oos = df[is_end:oos_end]

        if len(df_is) < 500 or len(df_oos) < 50:
            continue

        metrics_oos = run_full_pipeline(
            df_oos,
            pct_window,
            p_thresh,
            hurst_thresh,
            hmm_model_path,
            tier_sizes,
            profit_margin,
            spread=spread.reindex(df_oos.index),
            label=f"wfo_oos_{i}",
            session_filter=session_filter,
            reversion_pct=reversion_pct,
        )
        fold_results.append(
            {
                "fold": i,
                "oos_start": str(df_oos.index[0].date()),
                "oos_end": str(df_oos.index[-1].date()),
                "sharpe": metrics_oos["sharpe"],
                "n_trades": metrics_oos["n_trades"],
            }
        )
        if len(metrics_oos["daily_returns"]) > 0:
            stitched.append(metrics_oos["daily_returns"])

    stitched_returns = pd.concat(stitched) if stitched else pd.Series(dtype=float)
    return stitched_returns, fold_results


# ---------------------------------------------------------------------------
# Quality gate evaluator
# ---------------------------------------------------------------------------


def evaluate_gates(
    is_metrics: dict,
    oos_metrics: dict,
    mc_result: dict,
    stress_oos_sharpe: float,
    wfo_fold_results: list[dict],
) -> dict[str, bool]:
    ratio = oos_metrics["sharpe"] / is_metrics["sharpe"] if is_metrics["sharpe"] != 0 else 0.0
    neg_folds = sum(1 for f in wfo_fold_results if f["sharpe"] < 0)

    gates = {
        "G_oos_sharpe": oos_metrics["sharpe"] > GATE_OOS_SHARPE,
        "G_oos_is_ratio": ratio > GATE_OOS_IS_RATIO,
        "G_win_rate": oos_metrics["win_rate"] > GATE_WIN_RATE,
        "G_max_dd": oos_metrics["max_dd"] < GATE_MAX_DD,
        "G_min_trades": oos_metrics["n_trades"] >= GATE_MIN_OOS_TRADES,
        "G1_monte_carlo": mc_result["pass"],
        "G3_slippage": stress_oos_sharpe > 0.5,
        "G4_wfo_neg_folds": neg_folds <= 2,
    }
    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("Stage 3 — Full Integrated Backtest & Validation")
    print("=" * 60)

    # Load previous stage states
    s1 = sm.get_stage1()
    s2 = sm.get_stage2()
    if s1 is None or s2 is None:
        print("WARNING: Stage 1/2 state missing.  Using config defaults.")
        pct_window = CFG["signal"]["percentile_window"]
        p_thresh = CFG["regime"]["p_ranging_thresh"]
        hurst_thresh = CFG["regime"]["hurst_thresh"]
        hmm_model_path = str(MODELS_DIR / "eurusd_mr_hmm.joblib")
    else:
        pct_window = s1["best_pct_window"]
        p_thresh = s2["p_thresh"]
        hurst_thresh = s2["hurst_thresh"]
        hmm_model_path = s2["hmm_model_path"]
        print(f"  pct_window={pct_window}  p_thresh={p_thresh}  hurst_thresh={hurst_thresh}")

    if not Path(hmm_model_path).exists():
        print(f"ERROR: HMM model not found at {hmm_model_path}")
        print("       Run Stage 2 first.")
        sys.exit(1)

    df = load_m5(PAIR)
    spread = build_spread_series(df, PAIR)

    is_end = int(len(df) * IS_SPLIT)
    df_is = df.iloc[:is_end]
    df_oos = df.iloc[is_end:]
    print(f"\n  IS:  {len(df_is):,} bars  OOS: {len(df_oos):,} bars")

    # ── IS and OOS backtests (london+ny session filter) ───────────────────
    print("\n[1/6] IS backtest...")
    is_metrics = run_full_pipeline(
        df_is,
        pct_window,
        p_thresh,
        hurst_thresh,
        hmm_model_path,
        TIER_SIZES,
        PROFIT_MARGIN,
        spread=spread.reindex(df_is.index),
        label="IS",
        session_filter=SESSION_FILTER,
        reversion_pct=REVERSION_PCT,
    )
    print(
        f"  IS  Sharpe={is_metrics['sharpe']:+.2f}  MaxDD={is_metrics['max_dd']:.2%}  "
        f"WR={is_metrics['win_rate']:.2%}  n={is_metrics['n_trades']}"
    )

    print("\n[2/6] OOS backtest...")
    oos_metrics = run_full_pipeline(
        df_oos,
        pct_window,
        p_thresh,
        hurst_thresh,
        hmm_model_path,
        TIER_SIZES,
        PROFIT_MARGIN,
        spread=spread.reindex(df_oos.index),
        label="OOS",
        session_filter=SESSION_FILTER,
        reversion_pct=REVERSION_PCT,
    )
    oos_is_ratio = (
        oos_metrics["sharpe"] / is_metrics["sharpe"] if is_metrics["sharpe"] != 0 else 0.0
    )
    print(
        f"  OOS Sharpe={oos_metrics['sharpe']:+.2f}  MaxDD={oos_metrics['max_dd']:.2%}  "
        f"WR={oos_metrics['win_rate']:.2%}  n={oos_metrics['n_trades']}"
    )
    print(f"  OOS/IS ratio: {oos_is_ratio:.2f}")

    # ── Monte Carlo (G1) ──────────────────────────────────────────────────
    print("\n[3/6] Monte Carlo trade shuffle (G1)...")
    np.random.seed(42)
    mc_result = monte_carlo_gate(
        oos_metrics["daily_returns"], n_shuffles=MC_N_SHUFFLES, pct=MC_PERCENTILE
    )
    print(
        f"  5th-pct Sharpe={mc_result['pct_sharpe']:.3f}  "
        f"% profitable={mc_result['pct_profitable']:.1%}  "
        f"PASS={mc_result['pass']}"
    )

    # ── 3× Slippage stress (G3) ───────────────────────────────────────────
    print("\n[4/6] 3× slippage stress test (G3)...")
    total_cost_3x = build_total_cost_series(df_oos, PAIR) * 3.0
    stress_metrics = run_full_pipeline(
        df_oos,
        pct_window,
        p_thresh,
        hurst_thresh,
        hmm_model_path,
        TIER_SIZES,
        PROFIT_MARGIN,
        spread=total_cost_3x.reindex(df_oos.index),
        label="OOS_3xSlippage",
        session_filter=SESSION_FILTER,
        reversion_pct=REVERSION_PCT,
    )
    print(
        f"  3× stress Sharpe={stress_metrics['sharpe']:+.2f}  PASS={stress_metrics['sharpe'] > 0.5}"
    )

    # ── Walk-Forward Optimisation ─────────────────────────────────────────
    print("\n[5/6] Walk-Forward Optimisation (6mo IS / 2mo OOS)...")
    wfo_returns, wfo_folds = walk_forward(
        df,
        pct_window,
        p_thresh,
        hurst_thresh,
        hmm_model_path,
        TIER_SIZES,
        PROFIT_MARGIN,
        spread,
        is_months=6,
        oos_months=2,
        session_filter=SESSION_FILTER,
        reversion_pct=REVERSION_PCT,
    )
    neg_folds = sum(1 for f in wfo_folds if f["sharpe"] < 0)
    wfo_sharpe = exe.compute_combined_sharpe(wfo_returns) if len(wfo_returns) > 0 else np.nan
    print(f"  WFO stitched Sharpe={wfo_sharpe:.3f}  Negative folds={neg_folds}/{len(wfo_folds)}")
    for f in wfo_folds:
        print(
            f"    Fold {f['fold']}: {f['oos_start']} -> {f['oos_end']}  "
            f"Sharpe={f['sharpe']:+.2f}  n={f['n_trades']}"
        )

    # ── Quality gates ─────────────────────────────────────────────────────
    print("\n[6/6] Quality gate evaluation...")
    gates = evaluate_gates(is_metrics, oos_metrics, mc_result, stress_metrics["sharpe"], wfo_folds)
    all_passed = all(gates.values())

    print(f"\n  {'GATE':<25} {'RESULT'}")
    print("  " + "-" * 40)
    for g, passed in gates.items():
        icon = "PASS" if passed else "FAIL"
        print(f"  {g:<25} [{icon}]")
    print(f"\n  Overall: {'ALL GATES PASSED' if all_passed else 'FAILED — do not deploy'}")

    # ── Persist results ───────────────────────────────────────────────────
    sm.save_stage3(
        is_sharpe=is_metrics["sharpe"],
        oos_sharpe=oos_metrics["sharpe"],
        oos_is_ratio=round(oos_is_ratio, 3),
        win_rate=oos_metrics["win_rate"],
        max_dd=oos_metrics["max_dd"],
        n_oos_trades=oos_metrics["n_trades"],
        gates_passed=all_passed,
        gate_results={k: bool(v) for k, v in gates.items()},
    )

    validation_report = {
        "is": {k: v for k, v in is_metrics.items() if k != "daily_returns"},
        "oos": {k: v for k, v in oos_metrics.items() if k != "daily_returns"},
        "oos_is_ratio": round(oos_is_ratio, 3),
        "monte_carlo": mc_result,
        "stress_3x_sharpe": stress_metrics["sharpe"],
        "wfo": {
            "stitched_sharpe": round(float(wfo_sharpe), 3) if not np.isnan(wfo_sharpe) else None,
            "negative_folds": neg_folds,
            "total_folds": len(wfo_folds),
            "folds": wfo_folds,
        },
        "gates": {k: bool(v) for k, v in gates.items()},
        "all_gates_passed": bool(all_passed),
    }
    out_json = REPORTS_DIR / "eurusd_mr_validation.json"
    with open(out_json, "w") as f:
        json.dump(validation_report, f, indent=4)
    print(f"\n  Saved validation report -> {out_json}")

    # ── Equity curve chart ────────────────────────────────────────────────
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        subplot_titles=["IS vs OOS Daily Returns (cumulative)", "WFO Stitched OOS Equity Curve"],
    )

    is_cum = (1 + is_metrics["daily_returns"]).cumprod()
    oos_cum = (1 + oos_metrics["daily_returns"]).cumprod()
    fig.add_trace(
        go.Scatter(x=is_cum.index, y=is_cum, name="IS", line=dict(color="steelblue")), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=oos_cum.index, y=oos_cum, name="OOS", line=dict(color="orange")), row=1, col=1
    )

    if len(wfo_returns) > 0:
        wfo_cum = (1 + wfo_returns).cumprod()
        fig.add_trace(
            go.Scatter(x=wfo_cum.index, y=wfo_cum, name="WFO OOS", line=dict(color="green")),
            row=2,
            col=1,
        )

    title_suffix = "ALL GATES PASSED" if all_passed else "GATES FAILED"
    fig.update_layout(title=f"Stage 3 Validation — {title_suffix}", height=700)
    out_html = REPORTS_DIR / "eurusd_mr_equity_curve.html"
    fig.write_html(str(out_html))
    print(f"  Saved equity curve -> {out_html}")

    if all_passed:
        print("\n[Stage 3 PASSED]  Next steps:")
        print("  1. Update config/eurusd_mr.toml with optimised values (already in state file)")
        print(
            "  2. Optional pairs module: uv run python research/mean_reversion/run_stage4_pairs.py"
        )
        print("  3. When ready for live: implement titan/strategies/mean_reversion/strategy.py")
    else:
        print("\n[Stage 3 FAILED]  Review gate failures above before proceeding.")
        print("  Consider adjusting thresholds in config/eurusd_mr.toml and re-running Stage 1.")


if __name__ == "__main__":
    main()
