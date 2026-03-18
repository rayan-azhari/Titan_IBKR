"""
run_ic_regime.py -- Regime-Conditioned IC Analysis

Tests whether the MTF confluence IC improves when conditioned on HMM regime.
Hypothesis: MTF signal has real IC (0.064 at h=5 overall), but ICIR is low (0.12)
because the signal only works in trending regimes (State 1 = high vol).

If IC(State 1) >> IC(State 0), it confirms:
  - Use HMM state as a gate in the MTF strategy
  - Only trade when regime_prob_1 > threshold

Usage:
    python research/ic_analysis/run_ic_regime.py --instrument EUR_USD --timeframe H4
    python research/ic_analysis/run_ic_regime.py --instrument EUR_USD --timeframe H4 --horizons 1,5,10,20
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.ic_analysis.run_ic import (
    compute_forward_returns,
    compute_mtf_confluence,
    load_close,
)
from research.ml_regime.feature_engine import FeatureEngine
from research.ml_regime.regime_detection import RegimeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = [1, 5, 10, 20, 60]
VOL_SPAN = 30
FRAC_D = 0.4
N_HMM_STATES = 2
HMM_MIN_SEQ = 120


def spearman_ic(signal: pd.Series, fwd: pd.Series) -> float:
    """Spearman IC between signal and forward return. Returns NaN if < 30 obs."""
    df = pd.concat([signal, fwd], axis=1).dropna()
    if len(df) < 30:
        return np.nan
    rho, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return float(rho)


def run_regime_ic(
    instrument: str,
    timeframe: str,
    horizons: list[int] | None = None,
) -> None:
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    # ── 1. Load price data ─────────────────────────────────────────────────────
    close = load_close(instrument, timeframe)

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    logger.info("Computing features...")
    engine = FeatureEngine(vol_span=VOL_SPAN, frac_d=FRAC_D)
    feat_df = engine.batch(close).dropna()

    # ── 3. HMM regime detection ────────────────────────────────────────────────
    logger.info("Fitting HMM on full series (no IS/OOS split -- IC research only)...")
    detector = RegimeDetector(n_states=N_HMM_STATES, min_seq_len=HMM_MIN_SEQ)
    detector.fit(feat_df.values)
    for s, desc in detector.state_descriptions.items():
        logger.info(f"  {desc}")

    regimes = detector.predict_sequence(feat_df.values)
    regime_series = pd.Series(regimes, index=feat_df.index, name="regime")
    regime_dist = np.bincount(regimes, minlength=N_HMM_STATES) / len(regimes)
    logger.info(f"  Regime dist: State0={regime_dist[0]:.1%} State1={regime_dist[1]:.1%}")

    # ── 4. MTF confluence signal ───────────────────────────────────────────────
    logger.info("Computing MTF confluence...")
    mtf_sig = compute_mtf_confluence(instrument, timeframe)
    if mtf_sig is None:
        logger.error("MTF confluence could not be computed. Check config/mtf.toml and data files.")
        return

    # ── 5. Forward returns ─────────────────────────────────────────────────────
    fwd_returns = compute_forward_returns(close, horizons)

    # ── 6. Align everything to a common index ─────────────────────────────────
    aligned = pd.concat(
        [mtf_sig.rename("mtf_confluence"), regime_series, fwd_returns],
        axis=1,
    ).dropna(subset=["mtf_confluence", "regime"])

    # ── 7. Compute IC: unconditional + per-regime ──────────────────────────────
    results = {}
    for h in horizons:
        fwd_col = f"fwd_{h}"
        if fwd_col not in aligned.columns:
            continue
        fwd = aligned[fwd_col]
        signal = aligned["mtf_confluence"]

        results[h] = {
            "unconditional": spearman_ic(signal, fwd),
        }
        for state in range(N_HMM_STATES):
            mask = aligned["regime"] == state
            results[h][f"state_{state}"] = spearman_ic(signal[mask], fwd[mask])

    # ── 8. Count observations per regime ──────────────────────────────────────
    n_state0 = int((aligned["regime"] == 0).sum())
    n_state1 = int((aligned["regime"] == 1).sum())

    # ── 9. Print results ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  REGIME-CONDITIONED IC -- MTF Confluence on {instrument} {timeframe}")
    print(f"  State 0 (low vol):  {n_state0} bars ({n_state0/len(aligned):.1%}) -- "
          f"{detector.state_descriptions.get(0, '')}")
    print(f"  State 1 (high vol): {n_state1} bars ({n_state1/len(aligned):.1%}) -- "
          f"{detector.state_descriptions.get(1, '')}")
    print("=" * 72)

    header = f"  {'Horizon':>8}  {'Unconditional':>14}  {'State0 (low vol)':>16}  {'State1 (high vol)':>17}  Lift"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for h in horizons:
        if h not in results:
            continue
        r = results[h]
        unc = r["unconditional"]
        s0 = r.get("state_0", np.nan)
        s1 = r.get("state_1", np.nan)
        lift = (s1 - unc) if not np.isnan(s1) and not np.isnan(unc) else np.nan

        def fmt(v: float) -> str:
            return f"{v:>+.4f}" if not np.isnan(v) else "     NaN"

        lift_str = f"{lift:>+.4f}" if not np.isnan(lift) else "     NaN"
        flag = " ***" if not np.isnan(lift) and lift > 0.01 else ""
        print(f"  {'h='+str(h):>8}  {fmt(unc):>14}  {fmt(s0):>16}  {fmt(s1):>17}  {lift_str}{flag}")

    print("=" * 72)

    # ── 10. Interpretation ─────────────────────────────────────────────────────
    best_h = max(results, key=lambda h: abs(results[h].get("unconditional", 0)))
    best = results[best_h]
    unc_best = best.get("unconditional", np.nan)
    s1_best = best.get("state_1", np.nan)
    s0_best = best.get("state_0", np.nan)

    print()
    if not np.isnan(s1_best) and not np.isnan(unc_best) and s1_best > unc_best + 0.01:
        print(f"  [RESULT] MTF IC is STRONGER in State 1 (high vol) at h={best_h}:")
        print(f"           Unconditional: {unc_best:+.4f}  ->  State 1: {s1_best:+.4f}")
        print("           Hypothesis CONFIRMED: use HMM State 1 as a trade gate.")
    elif not np.isnan(s0_best) and not np.isnan(unc_best) and s0_best > unc_best + 0.01:
        print(f"  [RESULT] MTF IC is STRONGER in State 0 (low vol) at h={best_h}.")
        print(f"           Unconditional: {unc_best:+.4f}  ->  State 0: {s0_best:+.4f}")
        print("           Hypothesis REJECTED: regime gate should use State 0.")
    else:
        print(f"  [RESULT] IC does not improve materially in either regime at h={best_h}.")
        print(f"           Unconditional: {unc_best:+.4f} | State0: {s0_best:+.4f} | State1: {s1_best:+.4f}")
        print("           The MTF signal is regime-independent -- no HMM gate needed.")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-Conditioned IC Analysis")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated forward horizons",
    )
    args = parser.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    run_regime_ic(instrument=args.instrument, timeframe=args.timeframe, horizons=horizons)


if __name__ == "__main__":
    main()
