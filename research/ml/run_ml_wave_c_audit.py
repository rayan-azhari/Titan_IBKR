"""ml Wave C -- signal-layer V3.6 audit (L58 protocol).

Reproduces the live ml strategy's signal layer on EUR/USD H1 historical
data and runs the standard V3.6 5-axis decision matrix:
    1. CI_lo bootstrap (bootstrap_sharpe_ci)
    2. DSR deflation (deflated_sharpe with N=1 trial since cell is frozen)
    3. MC block bootstrap (run_block_mc)
    4. Sanctuary divergence (sanctuary_divergence_test)
    5. Noise robustness (run_noise_robustness)

L19 causality already verified clean (`research/exploration/
diagnose_ml_l19_causality.py` 2026-05-17): build_features() and
build_tbm_labels._tbm_kernel are both causal.

Signal definition (L58 simplest form):
    position[t] = primary_signal[t] * meta_classifier_decision[t]
    where meta_classifier_decision[t] = 1 if predict_proba > threshold else 0

The primary_signal column is computed inside build_features() and is
the H1 multi-timeframe confluence sign (+1/-1). The meta-model is the
"trade or skip" gate on top.

Run::

    PYTHONIOENCODING=utf-8 uv run python research/ml/run_ml_wave_c_audit.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from titan.research.framework import (  # noqa: E402
    DecisionInputs,
    NoiseConfig,
    StrategyClass,
    decide,
    defaults_for,
    deflated_sharpe,
    run_block_mc,
    run_noise_robustness,
    sanctuary_divergence_test,
)
from titan.research.framework.mc import DEFAULT_MC_WORKERS  # noqa: E402
from titan.research.metrics import BARS_PER_YEAR, bootstrap_sharpe_ci, sharpe  # noqa: E402
from titan.strategies.ml.features import build_features  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "ml_wave_c_audit"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "meta_model_EURUSD_H1_20260517_154132.joblib"
DATA_FILE = DATA_DIR / "EUR_USD_H1.parquet"

PERIODS_PER_YEAR = BARS_PER_YEAR["H1"]
SANCTUARY_MONTHS = 12

COST_BPS = 0.5  # H1 FX: 0.5bp/turnover


def _load_h1() -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().dropna(subset=["close"])
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _compute_signals(df: pd.DataFrame, model_artefact: dict) -> pd.Series:
    """Run the meta-classifier and convert proba to a (-1, 0, +1) signal.

    The classifier predicts P(TP-first), so:
        proba > threshold        -> +1 (long)
        proba < (1 - threshold)  -> -1 (short)
        otherwise                ->  0 (no position)

    Then .shift(1) for V3.6 causality (position effective at close[t]
    earns the return from t -> t+1).
    """
    model = model_artefact["model"]
    threshold = float(model_artefact["threshold"])
    feature_names = list(model_artefact["feature_names"])
    # For audit context (H4/D), reuse what's available -- we replicate the
    # retrain script's context if the data exists, else fall back to empty.
    ctx: dict[str, pd.DataFrame] = {}
    for tf in ("H4", "D"):
        fp = PROJECT_ROOT / "data" / f"EUR_USD_{tf}.parquet"
        if fp.exists():
            cf = pd.read_parquet(fp)
            if "timestamp" in cf.columns:
                cf["timestamp"] = pd.to_datetime(cf["timestamp"]).dt.tz_localize(None)
                cf = cf.set_index("timestamp")
            else:
                cf.index = pd.to_datetime(cf.index).tz_localize(None)
            ctx[tf] = cf.sort_index().dropna(subset=["close"])
    feats = build_features(df, context_data=ctx, cfg=None)
    missing = [c for c in feature_names if c not in feats.columns]
    if missing:
        raise RuntimeError(f"build_features() missing required columns: {missing}")
    X = feats[feature_names].copy()
    clean_mask = X.notna().all(axis=1)
    X_clean = X.loc[clean_mask]
    if X_clean.empty:
        raise RuntimeError("All feature rows are NaN; cannot run model.")
    proba = pd.Series(0.5, index=X.index, dtype=float)
    proba.loc[X_clean.index] = model.predict_proba(X_clean)[:, 1]
    signal = pd.Series(0.0, index=X.index, dtype=float)
    signal.loc[proba > threshold] = 1.0
    signal.loc[proba < (1.0 - threshold)] = -1.0
    # Causal shift.
    signal = signal.shift(1).reindex(df.index).fillna(0.0)
    return signal.rename("ml_signal")


def _strategy_returns(df: pd.DataFrame, signal: pd.Series) -> pd.Series:
    """Per-bar net returns from the (-1, 0, +1) signal applied with V3.6
    causality (signal.shift(1) already done above; we just compute the
    earn-the-next-bar return * signal).
    """
    log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    # Position at close[t-1] earns return from t-1 -> t. Signal is already
    # shifted in _compute_signals, so:
    gross = signal * log_ret
    # Costs: 2 * COST_BPS per position change.
    dpos = signal.diff().abs().fillna(0.0)
    cost = dpos * (2 * COST_BPS / 10_000.0)
    net = gross - cost
    return net.rename("ret")


@dataclass
class AuditResult:
    n_bars_visible: int
    n_bars_sanctuary: int
    n_active: int
    fraction_active: float
    headline_sharpe: float
    ci_lo: float
    ci_hi: float
    dsr_prob: float
    mc_p_maxdd_gt_threshold: float
    mc_threshold_pct: float
    sanctuary_sharpe: float
    sanctuary_percentile: float
    noise_base: float
    noise_passes_mean: bool
    noise_passes_worst: bool
    noise_axis: str
    verdict: str
    rationale: str


def main() -> None:
    print("=" * 80)
    print("ml Wave C -- signal-layer V3.6 audit on EUR/USD H1")
    print("=" * 80)

    print(f"\nLoading data: {DATA_FILE}")
    df = _load_h1()
    print(f"  bars: {len(df)} ({df.index[0]} -> {df.index[-1]})")

    print(f"\nLoading model: {MODEL_PATH.name}")
    art = joblib.load(MODEL_PATH)
    train_auc = float(art.get("train_auc", art.get("avg_auc", float("nan"))))
    sanc_auc = float(art.get("sanctuary_auc", float("nan")))
    print(
        f"  trained_at={art['trained_at']}, threshold={art['threshold']}, "
        f"train_auc={train_auc:.4f}, sanctuary_auc={sanc_auc:.4f}"
    )

    print("\n[1] Computing signal series (model predictions × primary signal, "
          "1-bar shift for causality)...")
    signal = _compute_signals(df, art)
    rets = _strategy_returns(df, signal)
    n_active = int((signal != 0).sum())
    frac_active = float(n_active / max(len(signal), 1))
    print(f"  active bars: {n_active}/{len(signal)} ({frac_active:.2%})")

    print(f"\n[2] Slicing sanctuary ({SANCTUARY_MONTHS} months)...")
    cutoff = df.index[-1] - pd.DateOffset(months=SANCTUARY_MONTHS)
    visible_mask = rets.index <= cutoff
    sanctuary_mask = rets.index > cutoff
    rets_visible = rets[visible_mask]
    rets_sanctuary = rets[sanctuary_mask]
    print(
        f"  visible:   {len(rets_visible)} bars "
        f"({rets_visible.index[0]} -> {rets_visible.index[-1]})"
    )
    print(
        f"  sanctuary: {len(rets_sanctuary)} bars "
        f"({rets_sanctuary.index[0]} -> {rets_sanctuary.index[-1]})"
    )

    print("\n[3] V3.6 5-axis decision matrix on visible window...")
    headline_sr = float(sharpe(rets_visible, periods_per_year=PERIODS_PER_YEAR))
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        rets_visible, periods_per_year=PERIODS_PER_YEAR,
        n_resamples=1000, seed=42,
    )
    # DSR with N=1 (no parameter sweep -- single frozen cell).
    dsr = deflated_sharpe(
        headline_sr, sr_var_across_trials=0.0, returns=rets_visible, n_trials=1,
    )
    cls = StrategyClass.INTRADAY_MICROSTRUCTURE
    d = defaults_for(cls)

    primary_close = df["close"][visible_mask]

    def _strategy_fn(synth_df: pd.DataFrame) -> pd.Series:
        # MC: synthesise close path, recompute signal from features, return
        # the bootstrap-path's strategy returns. The model is INSIDE
        # _compute_signals; on synthetic data the model's predictions
        # reflect the synthetic feature distribution.
        synth_full = pd.DataFrame({
            "open": synth_df["close"],
            "high": synth_df["close"],
            "low": synth_df["close"],
            "close": synth_df["close"],
            "volume": 1.0,
        })
        synth_signal = _compute_signals(synth_full, art)
        return _strategy_returns(synth_full, synth_signal)

    mc = run_block_mc(
        primary_close=primary_close, cfg=d.mc,
        strategy_fn=_strategy_fn, periods_per_year=PERIODS_PER_YEAR,
        seed=42, extra_series=None, n_workers=DEFAULT_MC_WORKERS,
    )
    sanc_sh = (
        float(sharpe(rets_sanctuary, periods_per_year=PERIODS_PER_YEAR))
        if len(rets_sanctuary) >= 50 else 0.0
    )
    div = sanctuary_divergence_test(
        historical_returns=rets_visible,
        sanctuary_returns=rets_sanctuary,
        periods_per_year=PERIODS_PER_YEAR,
    )
    closes_v = primary_close.rename("close").to_frame()

    def _noise_fn(closes_df_local: pd.DataFrame) -> pd.Series:
        synth_full = pd.DataFrame({
            "open": closes_df_local["close"],
            "high": closes_df_local["close"],
            "low": closes_df_local["close"],
            "close": closes_df_local["close"],
            "volume": 1.0,
        })
        sig = _compute_signals(synth_full, art)
        return _strategy_returns(synth_full, sig)

    noise = run_noise_robustness(
        closes_v, _noise_fn, periods_per_year=PERIODS_PER_YEAR,
        cfg=NoiseConfig(noise_levels=(0.1, 0.3, 0.5), n_trials=10, max_degradation=0.30),
    )
    inputs = DecisionInputs(
        ci_lo=ci_lo, dsr_prob=dsr.dsr_prob,
        p_maxdd_gt_threshold=mc.p_maxdd_gt_threshold,
        pass_threshold_prob=mc.pass_threshold_prob,
        sanctuary_sharpe=sanc_sh,
        noise_passes_mean=noise.passes,
        noise_passes_worst=noise.worst_case_passes,
    )
    decision = decide(inputs)

    result = AuditResult(
        n_bars_visible=len(rets_visible),
        n_bars_sanctuary=len(rets_sanctuary),
        n_active=n_active, fraction_active=frac_active,
        headline_sharpe=round(headline_sr, 4),
        ci_lo=round(ci_lo, 4), ci_hi=round(ci_hi, 4),
        dsr_prob=round(dsr.dsr_prob, 4),
        mc_p_maxdd_gt_threshold=round(mc.p_maxdd_gt_threshold, 4),
        mc_threshold_pct=round(mc.threshold_pct, 4),
        sanctuary_sharpe=round(sanc_sh, 4),
        sanctuary_percentile=round(div.percentile, 4) if np.isfinite(div.percentile) else float("nan"),
        noise_base=round(noise.base_sharpe, 4),
        noise_passes_mean=noise.passes,
        noise_passes_worst=noise.worst_case_passes,
        noise_axis=decision.noise_axis,
        verdict=decision.verdict.value,
        rationale=decision.rationale,
    )

    print(f"\n  Sharpe (ann.) = {result.headline_sharpe:+.4f}")
    print(f"  CI95          = [{result.ci_lo:+.3f}, {result.ci_hi:+.3f}]")
    print(f"  DSR           = {result.dsr_prob:.4f}")
    print(f"  MC P(>{result.mc_threshold_pct*100:.0f}%) = {result.mc_p_maxdd_gt_threshold:.4f}")
    print(f"  Sanc Sharpe   = {result.sanctuary_sharpe:+.4f}")
    print(f"  Noise base    = {result.noise_base:+.4f} "
          f"(mean_pass={result.noise_passes_mean}, worst_pass={result.noise_passes_worst}, "
          f"axis={result.noise_axis})")
    print(f"  Verdict       = {result.verdict}")
    print(f"  Rationale     = {result.rationale}")

    # Write result log.
    rpath = REPORTS_DIR / "result_log.md"
    with rpath.open("w", encoding="utf-8") as fh:
        fh.write("# ml Wave C Audit Result Log -- signal layer (L58 protocol)\n\n")
        fh.write("**Run date:** 2026-05-17\n")
        fh.write(f"**Model:** `{MODEL_PATH.name}` (trained {art['trained_at']})\n")
        fh.write(f"**Data:** `{DATA_FILE.name}` -- {len(df)} H1 bars "
                 f"({df.index[0]} -> {df.index[-1]})\n")
        fh.write(f"**Visible / Sanctuary:** {result.n_bars_visible} / "
                 f"{result.n_bars_sanctuary}\n")
        fh.write(f"**Active bars:** {result.n_active} ({result.fraction_active:.2%})\n\n")
        fh.write("## §4.1 5-axis matrix\n\n")
        fh.write("| Metric | Value |\n|---|---:|\n")
        fh.write(f"| Sharpe (ann.) | {result.headline_sharpe:+.4f} |\n")
        fh.write(f"| CI95 lo | {result.ci_lo:+.4f} |\n")
        fh.write(f"| CI95 hi | {result.ci_hi:+.4f} |\n")
        fh.write(f"| DSR prob | {result.dsr_prob:.4f} |\n")
        fh.write(f"| MC P(>{result.mc_threshold_pct*100:.0f}%) | {result.mc_p_maxdd_gt_threshold:.4f} |\n")
        fh.write(f"| Sanctuary Sharpe | {result.sanctuary_sharpe:+.4f} |\n")
        fh.write(f"| Noise base | {result.noise_base:+.4f} |\n")
        fh.write(f"| Noise mean_pass | {result.noise_passes_mean} |\n")
        fh.write(f"| Noise worst_pass | {result.noise_passes_worst} |\n")
        fh.write(f"| Noise axis | {result.noise_axis} |\n\n")
        fh.write(f"**Verdict:** **{result.verdict}**\n\n")
        fh.write(f"**Rationale:** {result.rationale}\n\n")
        fh.write("## §4.2 L19 causality status\n\n")
        fh.write(
            "Confirmed clean by `research/exploration/diagnose_ml_l19_causality.py` "
            "(2026-05-17): `build_features()` is causal (max abs-diff = 0 after "
            "future-bar corruption), `build_tbm_labels._tbm_kernel` is causal "
            "(0/1912 past labels changed after corrupting last 50 close bars). "
            "Signal pipeline `_compute_signals()` adds an explicit `.shift(1)` "
            "before earning the next-bar return (V3.6 causality contract).\n"
        )
    print(f"\nResult log: {rpath}")


if __name__ == "__main__":
    main()
