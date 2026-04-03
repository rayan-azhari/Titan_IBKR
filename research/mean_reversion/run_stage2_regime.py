"""run_stage2_regime.py — HMM + Hurst exponent calibration.

Stage 2 goals:
  - Train 3-state Gaussian HMM on IS data only (70% split).
  - Auto-label states: lowest-vol = Ranging, highest-vol = High-Vol.
  - Compute rolling Hurst exponent on H1 prices (resampled for speed),
    forward-fill to M5.
  - Measure HMM posterior P(ranging) correlation with Hurst < 0.5.
  - Sweep gate thresholds: p_thresh × hurst_thresh.
  - Compare Sharpe: no regime filter vs best regime filter.
  - Save fitted HMM to models/eurusd_mr_hmm.joblib.

Outputs:
  .tmp/reports/eurusd_mr_hmm_states.html         ← price + state overlay
  .tmp/reports/eurusd_mr_regime_sensitivity.csv  ← gate sweep results
  models/eurusd_mr_hmm.joblib                    ← saved model
  .tmp/mr_state_eurusd.json                      ← best gate params

Usage:
    uv run python research/mean_reversion/run_stage2_regime.py

Prerequisite:
    uv run python research/mean_reversion/run_stage1_signals.py
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR    = PROJECT_ROOT / "data"
MODELS_DIR  = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: vectorbt not installed.  Run: uv sync")
    sys.exit(1)

from research.mean_reversion import regime as reg
from research.mean_reversion import signals as sig
from research.mean_reversion import state_manager as sm
from titan.models.spread import build_spread_series

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "config" / "eurusd_mr.toml"
with open(CONFIG_PATH, "rb") as f:
    CFG = tomllib.load(f)

PAIR = CFG["base"]["instrument"]
TF   = CFG["base"]["timeframe"]
ANCHOR_SESSIONS = CFG["vwap"]["anchor_sessions"]
TIERS_PCT = CFG["signal"]["tiers_pct"]
SESSION_FILTER = CFG["signal"]["session_filter"]
REVERSION_PCT  = CFG["signal"]["reversion_target_pct"]
HMM_STATES = CFG["regime"]["hmm_states"]
HMM_MIN_BARS = CFG["regime"]["hmm_min_bars"]
HURST_WINDOW = CFG["regime"]["hurst_window"]
SG_WINDOW = CFG["regime"]["sg_window"]
SG_POLY = CFG["regime"]["sg_poly"]

IS_SPLIT = 0.70

# Gate sensitivity sweep
P_THRESHOLDS  = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]
H_THRESHOLDS  = [0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]

HMM_MODEL_PATH = str(MODELS_DIR / "eurusd_mr_hmm.joblib")


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
    print(f"  Loaded {pair}_{TF}: {len(df):,} bars  [{df.index[0].date()} -> {df.index[-1].date()}]")
    return df


# ---------------------------------------------------------------------------
# Hurst on H1 (speed optimisation) forward-filled to M5
# ---------------------------------------------------------------------------


def compute_hurst_h1_ffill(df_m5: pd.DataFrame) -> pd.Series:
    """Compute Hurst on H1 OHLCV (resampled) and forward-fill back to M5.

    R/S on 500 × 100k M5 bars is O(n²) — too slow for research sweeps.
    Resampling to H1 (78 bars/day -> 13 bars/day) cuts computation by ~6×.
    """
    close_h1 = df_m5["close"].resample("1h").last().dropna()
    print(f"  Resampled to H1: {len(close_h1):,} bars for Hurst computation...")
    hurst_h1 = reg.rolling_hurst(close_h1, window=HURST_WINDOW)
    # Forward-fill H1 values to M5 index
    hurst_m5 = hurst_h1.reindex(df_m5.index, method="ffill")
    print(f"  Hurst range: [{hurst_m5.min():.3f}, {hurst_m5.max():.3f}]  "
          f"mean={hurst_m5.mean():.3f}")
    return hurst_m5


# ---------------------------------------------------------------------------
# Simplified single-tier backtest with regime gate
# ---------------------------------------------------------------------------


def run_gated_backtest(
    close: pd.Series,
    deviation: pd.Series,
    tier1_lvl: pd.Series,
    inv_lvl: pd.Series,
    gate: pd.Series,
    spread: pd.Series,
    label: str,
    reversion_pct: float = 0.5,
) -> dict:
    invalidation = deviation.abs() > inv_lvl.abs()
    # Direction-aware exits: short exits when deviation falls back to 50% of
    # entry level; long exits when deviation rises back to -50% of entry level.
    short_entry = (deviation > tier1_lvl) & gate
    short_exit  = (deviation < tier1_lvl.abs() * reversion_pct) | invalidation
    long_entry  = (deviation < -tier1_lvl) & gate
    long_exit   = (deviation > -tier1_lvl.abs() * reversion_pct) | invalidation

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
    print("Stage 2 — Regime Filter: HMM + Hurst Calibration")
    print("=" * 60)

    # Load stage 1 state
    s1 = sm.get_stage1()
    if s1 is None:
        print("WARNING: Stage 1 state not found.  Using config defaults.")
        best_pct_window = CFG["signal"]["percentile_window"]
    else:
        best_pct_window = s1["best_pct_window"]
        print(f"  Stage 1 best percentile window: {best_pct_window}")

    df = load_m5(PAIR)
    close = df["close"]
    spread = build_spread_series(df, PAIR)

    # ── IS/OOS split ──────────────────────────────────────────────────────
    is_end = int(len(df) * IS_SPLIT)
    df_is = df.iloc[:is_end]
    close_is = df_is["close"]
    print(f"\n  IS: {len(df_is):,} bars  [{df_is.index[0].date()} -> {df_is.index[-1].date()}]")
    print(f"  OOS: {len(df) - is_end:,} bars  [{df.iloc[is_end].name.date()} -> {df.index[-1].date()}]")

    # ── VWAP, deviation, percentile levels ───────────────────────────────
    print("\n[1/5] Computing anchored VWAP and levels...")
    vwap = sig.compute_anchored_vwap(df, anchor_sessions=ANCHOR_SESSIONS)
    deviation = sig.compute_deviation(df, vwap)
    tier1_lvl = sig.percentile_levels(deviation, best_pct_window, [TIERS_PCT[0]])[TIERS_PCT[0]]
    inv_lvl   = sig.invalidation_level(deviation, best_pct_window)

    # ── Train HMM on IS only ──────────────────────────────────────────────
    print("\n[2/5] Training HMM on IS data...")
    obs_full = reg.build_observations(close, sg_window=SG_WINDOW, sg_poly=SG_POLY)
    obs_is   = obs_full[:is_end]

    model = reg.train_hmm(obs_is, n_states=HMM_STATES)
    state_labels = reg.label_states(model)
    ranging_idx  = reg.ranging_state_index(model)
    print(f"  State labels: {state_labels}")
    print(f"  Ranging state index: {ranging_idx}")
    print("  State means (ret, vol):")
    for i, m in enumerate(model.means_):
        print(f"    State {i} [{state_labels[i]}]: ret={m[0]:.6f}  vol={m[1]:.6f}")

    reg.save_hmm(model, HMM_MODEL_PATH)

    # ── Rolling posterior on full dataset ─────────────────────────────────
    print("\n[3/5] Computing rolling HMM posterior (may take a few minutes)...")
    post_arr = reg.rolling_regime_posterior(model, obs_full, ranging_idx, min_bars=HMM_MIN_BARS)
    hmm_posterior = pd.Series(post_arr, index=df.index, name="p_ranging")
    pct_active = (hmm_posterior >= 0.65).mean()
    print(f"  P(ranging) mean={hmm_posterior.mean():.3f}  "
          f"% bars >= 0.65: {pct_active:.1%}")

    # ── Hurst exponent ────────────────────────────────────────────────────
    print("\n[4/5] Computing rolling Hurst exponent (H1 resampled)...")
    hurst = compute_hurst_h1_ffill(df)
    pct_mr = (hurst < 0.50).mean()
    print(f"  % bars with H < 0.50 (mean-reverting): {pct_mr:.1%}")

    # ── Gate sensitivity sweep ────────────────────────────────────────────
    print("\n[5/5] Gate sensitivity sweep (p_thresh × hurst_thresh)...")
    # Session mask for entries (same as Stage 3)
    sess_parts = SESSION_FILTER.split("+")
    sess_mask = sig.session_mask(df.index, sess_parts) if SESSION_FILTER != "none" \
        else pd.Series(True, index=df.index)

    # No-filter baseline (session mask still applied, no HMM/Hurst gate)
    baseline = run_gated_backtest(
        close, deviation, tier1_lvl, inv_lvl, sess_mask, spread, "no_filter",
        reversion_pct=REVERSION_PCT,
    )
    print(f"  Baseline (session only): Sharpe={baseline['sharpe']:+.2f}  "
          f"n_trades={baseline['n_trades']}")

    best_p = CFG["regime"]["p_ranging_thresh"]
    best_h = CFG["regime"]["hurst_thresh"]

    def _sweep_one(p: float, h: float) -> dict:
        gate = reg.regime_gate(hmm_posterior, hurst, p_thresh=p, hurst_thresh=h)
        gate = gate & sess_mask
        row = run_gated_backtest(
            close, deviation, tier1_lvl, inv_lvl, gate, spread,
            label=f"p{p:.2f}_h{h:.2f}", reversion_pct=REVERSION_PCT,
        )
        row["p_thresh"] = p
        row["hurst_thresh"] = h
        row["gate_pct"] = round(float(gate.mean()), 3)
        row["sharpe_lift"] = round(row["sharpe"] - baseline["sharpe"], 3)
        return row

    combos = [(p, h) for p in P_THRESHOLDS for h in H_THRESHOLDS]
    sweep_rows = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_sweep_one)(p, h) for p, h in combos
    )

    best_sharpe = -np.inf
    for row in sweep_rows:
        if row["n_trades"] >= 50 and row["sharpe"] > best_sharpe:
            best_sharpe = row["sharpe"]
            best_p = row["p_thresh"]
            best_h = row["hurst_thresh"]

    sweep_df = pd.DataFrame(sweep_rows).sort_values("sharpe", ascending=False)
    out_csv = REPORTS_DIR / "eurusd_mr_regime_sensitivity.csv"
    sweep_df.to_csv(out_csv, index=False)
    print("\n  Top 5 gate configs:")
    print(sweep_df.head(5)[["label", "p_thresh", "hurst_thresh", "sharpe",
                              "sharpe_lift", "n_trades", "gate_pct"]].to_string(index=False))
    print(f"\n  Best gate: p_thresh={best_p}  hurst_thresh={best_h}  Sharpe={best_sharpe:.3f}")

    # ── Persist state ─────────────────────────────────────────────────────
    sm.save_stage2(
        ranging_state_idx=int(ranging_idx),
        p_thresh=float(best_p),
        hurst_thresh=float(best_h),
        sharpe_lift=round(best_sharpe - baseline["sharpe"], 3),
        hmm_model_path=HMM_MODEL_PATH,
    )

    # ── HMM state overlay chart ───────────────────────────────────────────
    print("\n  Generating HMM state chart...")
    # Use Viterbi on IS for a clean state sequence (research visualisation only)
    states_is = model.predict(obs_is)
    state_series_is = pd.Series(states_is, index=df_is.index)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["EUR/USD Close", "HMM State (IS)", "Hurst Exponent (H1->M5)"],
                        row_heights=[0.5, 0.25, 0.25])

    fig.add_trace(go.Scatter(x=df_is.index, y=close_is, name="Close", line=dict(width=1)),
                  row=1, col=1)

    state_colors = {0: "#3498db", 1: "#e74c3c", 2: "#f39c12"}
    for s_idx, s_name in state_labels.items():
        mask = state_series_is == s_idx
        fig.add_trace(go.Scatter(
            x=df_is.index[mask], y=[s_idx] * mask.sum(),
            mode="markers", name=s_name,
            marker=dict(color=state_colors.get(s_idx, "grey"), size=2),
        ), row=2, col=1)

    hurst_is = hurst.iloc[:is_end]
    fig.add_trace(go.Scatter(x=hurst_is.index, y=hurst_is, name="Hurst",
                             line=dict(color="purple", width=1)), row=3, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)

    fig.update_layout(title="Stage 2 — HMM Regime States (IS Period)", height=800)
    out_html = REPORTS_DIR / "eurusd_mr_hmm_states.html"
    fig.write_html(str(out_html))
    print(f"  Saved chart  -> {out_html}")

    print("\n[Stage 2 complete]  Run Stage 3 next:")
    print("  uv run python research/mean_reversion/run_stage3_full.py")


if __name__ == "__main__":
    main()
