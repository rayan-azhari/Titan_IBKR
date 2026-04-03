"""run_stage4_pairs.py — EUR/USD vs GBP/USD cointegrated pairs backtest.

Stage 4 (optional): Test a pairs trade between EUR/USD and GBP/USD using
the same percentile-entry + regime-gate infrastructure.

The pairs structure hedges out directional USD risk — profit depends only
on the spread between the two pairs normalising, not on USD direction.

This script assumes GBP_USD_M5.parquet exists.  If not, download first:
  uv run python scripts/download_fx_m5.py --pair GBP_USD --chunks 6

Key steps:
  1. Load EUR/USD and GBP/USD M5 data (align to common index).
  2. Engle-Granger cointegration test on IS data.
  3. Estimate hedge ratio β on IS data.
  4. Compute spread = EURUSD - β × GBPUSD.
  5. Apply percentile entry levels and regime gate to the spread.
  6. Backtest and report IS/OOS metrics.
  7. WFO with per-fold β re-estimation.

Outputs:
  .tmp/reports/eurusd_gbpusd_pairs_validation.json
  .tmp/reports/eurusd_gbpusd_pairs_equity.html

Usage:
    uv run python research/mean_reversion/run_stage4_pairs.py
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR    = PROJECT_ROOT / "data"
MODELS_DIR  = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from research.mean_reversion import pairs_trading as pt
from research.mean_reversion import regime as reg
from research.mean_reversion import state_manager as sm
from research.mean_reversion.run_stage2_regime import compute_hurst_h1_ffill
from titan.models.spread import build_spread_series

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "config" / "eurusd_mr.toml"
with open(CONFIG_PATH, "rb") as f:
    CFG = tomllib.load(f)

TIERS_PCT       = CFG["signal"]["tiers_pct"]
ANCHOR_SESSIONS = CFG["vwap"]["anchor_sessions"]
IS_SPLIT        = 0.70
PCT_WINDOW      = CFG["signal"]["percentile_window"]

PAIR_A = "EUR_USD"
PAIR_B = "GBP_USD"
TF     = "M5"


# ---------------------------------------------------------------------------
# Data loading and alignment
# ---------------------------------------------------------------------------


def load_m5(pair: str) -> pd.DataFrame:
    path = DATA_DIR / f"{pair}_{TF}.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        print(f"  Download: uv run python scripts/download_fx_m5.py --pair {pair} --chunks 6")
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def align_pairs(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two DataFrames to their common index (inner join on timestamps)."""
    common = df_a.index.intersection(df_b.index)
    return df_a.loc[common], df_b.loc[common]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("Stage 4 (Optional) — EUR/USD vs GBP/USD Pairs Backtest")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    df_eur = load_m5(PAIR_A)
    df_gbp = load_m5(PAIR_B)
    if df_eur is None or df_gbp is None:
        sys.exit(1)

    df_eur, df_gbp = align_pairs(df_eur, df_gbp)
    print(f"\n  Common bars: {len(df_eur):,}  "
          f"[{df_eur.index[0].date()} -> {df_eur.index[-1].date()}]")

    close_eur = df_eur["close"]
    close_gbp = df_gbp["close"]

    # ── IS/OOS split ──────────────────────────────────────────────────────
    is_end = int(len(df_eur) * IS_SPLIT)
    close_eur_is = close_eur.iloc[:is_end]
    close_gbp_is = close_gbp.iloc[:is_end]

    # ── Cointegration test on IS ──────────────────────────────────────────
    print("\n[1/5] Engle-Granger cointegration test (IS data)...")
    coint_result = pt.test_cointegration(close_eur_is, close_gbp_is)
    print(f"  t_stat={coint_result['t_stat']:.4f}  p_value={coint_result['p_value']:.4f}")
    print(f"  Cointegrated: {coint_result['is_cointegrated']}")
    if not coint_result["is_cointegrated"]:
        print("  WARNING: Pairs are not cointegrated on IS data.")
        print("           Results will be unreliable.  Proceeding for research only.")

    # ── Hedge ratio and spread ─────────────────────────────────────────────
    print("\n[2/5] Estimating hedge ratio on IS data...")
    beta = pt.estimate_hedge_ratio(close_eur_is, close_gbp_is)
    print(f"  β = {beta:.4f}  (spread = EUR/USD - {beta:.4f} × GBP/USD)")

    spread_full = pt.compute_spread(close_eur, close_gbp, beta)
    spread_is   = spread_full.iloc[:is_end]
    spread_oos  = spread_full.iloc[is_end:]
    print(f"  Spread IS  mean={spread_is.mean():.6f}  std={spread_is.std():.6f}")
    print(f"  Spread OOS mean={spread_oos.mean():.6f}  std={spread_oos.std():.6f}")

    # ── Load regime filter (from Stage 2) ─────────────────────────────────
    s2 = sm.get_stage2()
    if s2 is None:
        print("  WARNING: Stage 2 state missing — using config defaults for gate.")
        p_thresh     = CFG["regime"]["p_ranging_thresh"]
        hurst_thresh = CFG["regime"]["hurst_thresh"]
        hmm_path     = str(MODELS_DIR / "eurusd_mr_hmm.joblib")
    else:
        p_thresh     = s2["p_thresh"]
        hurst_thresh = s2["hurst_thresh"]
        hmm_path     = s2["hmm_model_path"]

    if not Path(hmm_path).exists():
        print(f"  WARNING: HMM model not found at {hmm_path}")
        print("           Regime gate disabled — running without filter.")
        gate_full = pd.Series(True, index=df_eur.index)
    else:
        print("\n[3/5] Applying regime filter to spread series...")
        model    = reg.load_hmm(hmm_path)
        obs      = reg.build_observations(spread_full)   # HMM on spread, not price
        ranging_i = reg.ranging_state_index(model)
        post_arr = reg.rolling_regime_posterior(model, obs, ranging_i, min_bars=100)
        post     = pd.Series(post_arr, index=df_eur.index)
        hurst    = compute_hurst_h1_ffill(df_eur)  # Use EUR/USD H for regime detection
        gate_full = reg.regime_gate(post, hurst, p_thresh=p_thresh, hurst_thresh=hurst_thresh)

    # ── Percentile levels on spread ───────────────────────────────────────
    print("\n[4/5] Computing spread percentile levels...")
    levels_full = pt.spread_percentile_levels(spread_full, PCT_WINDOW, pcts=TIERS_PCT)

    # Combined cost: two spread legs
    spread_cost_eur = build_spread_series(df_eur, PAIR_A)
    spread_cost_gbp = build_spread_series(df_gbp, PAIR_B)
    combined_cost   = spread_cost_eur + spread_cost_gbp  # two legs to enter/exit

    # ── IS/OOS backtests ──────────────────────────────────────────────────
    print("\n[5/5] Running IS and OOS backtests...")
    for split_name, sl in [("IS", slice(None, is_end)), ("OOS", slice(is_end, None))]:
        s_spread   = spread_full.iloc[sl]
        s_levels   = levels_full.iloc[sl]
        s_gate     = gate_full.iloc[sl]
        s_cost     = combined_cost.iloc[sl]
        tier1_pct  = TIERS_PCT[0]

        short_e, short_ex, long_e, long_ex = pt.build_pairs_signals(
            s_spread, s_levels, s_gate, tier1_pct
        )
        metrics = pt.run_pairs_backtest(
            s_spread, short_e, short_ex, long_e, long_ex, s_cost, label=split_name
        )
        print(f"  {split_name}:  Sharpe={metrics['sharpe']:+.2f}  "
              f"MaxDD={metrics['max_dd']:.2%}  WR={metrics['win_rate']:.2%}  "
              f"n_trades={metrics['n_trades']}")

    # ── WFO with per-fold β re-estimation ────────────────────────────────
    print("\n  WFO with per-fold β re-estimation (6mo IS / 2mo OOS)...")
    fold_results = []
    dates = pd.date_range(df_eur.index[0], df_eur.index[-1], freq="MS")
    for i in range(6, len(dates) - 2):
        is_start = dates[i - 6]
        is_end_dt = dates[i]
        oos_end_dt = dates[i + 2]

        eur_is_fold  = close_eur[is_start:is_end_dt]
        gbp_is_fold  = close_gbp[is_start:is_end_dt]
        eur_oos_fold = close_eur[is_end_dt:oos_end_dt]
        gbp_oos_fold = close_gbp[is_end_dt:oos_end_dt]

        if len(eur_is_fold) < 200 or len(eur_oos_fold) < 20:
            continue

        # Re-estimate β on this IS window
        beta_fold    = pt.estimate_hedge_ratio(eur_is_fold, gbp_is_fold)
        spread_oos_f = pt.compute_spread(eur_oos_fold, gbp_oos_fold, beta_fold)
        levels_oos_f = pt.spread_percentile_levels(
            pt.compute_spread(eur_is_fold, gbp_is_fold, beta_fold),
            min(PCT_WINDOW, len(eur_is_fold) - 10),
            pcts=TIERS_PCT,
        ).iloc[[-1]].reindex(spread_oos_f.index, method="ffill")

        gate_oos_f = gate_full.reindex(spread_oos_f.index, method="ffill").fillna(False)
        cost_oos_f = combined_cost.reindex(spread_oos_f.index, method="ffill")

        short_e, short_ex, long_e, long_ex = pt.build_pairs_signals(
            spread_oos_f, levels_oos_f, gate_oos_f, TIERS_PCT[0]
        )
        m = pt.run_pairs_backtest(
            spread_oos_f, short_e, short_ex, long_e, long_ex, cost_oos_f,
            label=f"wfo_{i}"
        )
        fold_results.append({
            "fold": i,
            "beta": round(beta_fold, 4),
            "oos_start": str(eur_oos_fold.index[0].date()),
            "oos_end": str(eur_oos_fold.index[-1].date()),
            "sharpe": m["sharpe"],
            "n_trades": m["n_trades"],
        })
        print(f"    Fold {i}: β={beta_fold:.4f}  Sharpe={m['sharpe']:+.2f}  n={m['n_trades']}")

    # ── Save results ──────────────────────────────────────────────────────
    result = {
        "cointegration": coint_result,
        "is_beta": round(beta, 4),
        "wfo_folds": fold_results,
        "note": "β re-estimated per WFO fold to track drift.",
    }
    out_json = REPORTS_DIR / "eurusd_gbpusd_pairs_validation.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\n  Saved pairs report -> {out_json}")
    print("\n[Stage 4 complete]")


if __name__ == "__main__":
    main()
