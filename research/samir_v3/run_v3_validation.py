"""Samir-Stack V3 Layer 1+2 validation harness.

Runs the VIX-HMM-gated strategy and four baselines head-to-head on the
same SPY underlying, 2003-2026. Reports WFO-stitched OOS metrics with
bootstrap-Sharpe CI, sanctuary holdout, and a forward-vol discrimination
diagnostic for the HMM itself.

Baselines:
  B1. SPY buy-and-hold (TR)
  B2. Always-on MES at L=L_target (no gate)
  B3. VIX z-score threshold gate (simple, deploy when VIX z-score < +1)
  B4. V2 6-indicator equal-weight regime score gate (current production
      research-side gate, not the live class — see ``research/samir_stack/
      regime_score.py``)

V3 strategy:
  V3. VIX-HMM filtering > 0.5 → deploy MES at L_target, else cash

Per the v3 design directive §10, V3 layer 1+2 becomes a deployment
candidate only if it beats v2's Calmar CI lo AND clears the four
baselines with statistical-significance margin.

Usage::

    uv run python -m research.samir_v3.run_v3_validation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import load_panel  # noqa: E402
from research.samir_stack.engines import FuturesEngine  # noqa: E402
from research.samir_stack.indicators import build_indicator_panel  # noqa: E402
from research.samir_stack.regime_score import regime_score_equal  # noqa: E402
from research.samir_stack.synthetic_3x import funding_series  # noqa: E402
from research.samir_v3.strategy_v3 import V3Config, run_v3_strategy  # noqa: E402
from research.samir_v3.vix_hmm import (  # noqa: E402
    vix_hmm_forward_vol_discrimination,
    vix_hmm_regime_score,
)
from titan.research.metrics import (  # noqa: E402
    BARS_PER_YEAR,
    annualize_vol,
    bootstrap_sharpe_ci,
    rolling_zscore,
    sharpe,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_v3"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── WFO helpers (same convention as Phase 5) ─────────────────────────────


def _wfo_stitch_oos(
    rets: pd.Series, *, is_days: int = 504, oos_days: int = 252, step: int = 252
) -> np.ndarray:
    rets = rets.dropna()
    arr = rets.to_numpy()
    n = len(arr)
    if n < is_days + oos_days:
        return np.array([])
    chunks = []
    s = is_days
    while s + oos_days <= n:
        chunks.append(arr[s : s + oos_days])
        s += step
    return np.concatenate(chunks) if chunks else np.array([])


def _bootstrap_calmar_ci_lo(rets: np.ndarray, *, n_resamples: int = 2000, seed: int = 42) -> float:
    if len(rets) < 252:
        return 0.0
    rng = np.random.default_rng(seed)
    n_years = len(rets) / 252.0
    calmars = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, len(rets), size=len(rets))
        sample = rets[idx]
        eq = np.cumprod(1.0 + sample)
        cagr = float(eq[-1] ** (1.0 / n_years) - 1.0)
        dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
        calmars[i] = cagr / abs(dd) if dd < -1e-9 else 0.0
    return float(np.quantile(calmars, 0.025))


def _summary(label: str, rets: pd.Series, *, sanctuary_days: int = 252) -> dict:
    rets_full = rets.dropna()
    if len(rets_full) < 504 + 252 + 252:
        return {"variant": label, "error": f"too few bars ({len(rets_full)})"}
    pre = rets_full.iloc[:-sanctuary_days]
    san = rets_full.iloc[-sanctuary_days:]

    stitched = _wfo_stitch_oos(pre)
    if len(stitched) == 0:
        return {"variant": label, "error": "insufficient WFO"}

    sh = sharpe(stitched, periods_per_year=BARS_PER_YEAR["D"])
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        stitched, periods_per_year=BARS_PER_YEAR["D"], n_resamples=2000, seed=42
    )
    n_y = len(stitched) / 252.0
    eq = np.cumprod(1.0 + stitched)
    cagr = float(eq[-1] ** (1.0 / n_y) - 1.0)
    dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
    calmar = cagr / abs(dd) if dd < -1e-9 else 0.0
    calmar_ci_lo = _bootstrap_calmar_ci_lo(stitched)
    vol = annualize_vol(float(stitched.std(ddof=1)), periods_per_year=BARS_PER_YEAR["D"])

    san_arr = san.to_numpy()
    san_sh = sharpe(san_arr, periods_per_year=BARS_PER_YEAR["D"]) if len(san_arr) > 1 else 0.0
    san_eq = np.cumprod(1.0 + san_arr)
    san_dd = float(((san_eq - np.maximum.accumulate(san_eq)) / np.maximum.accumulate(san_eq)).min())

    rets_2022 = rets_full.loc["2022-01-01":"2022-12-31"]
    cum_2022 = float((1.0 + rets_2022).prod() - 1.0) if len(rets_2022) > 0 else float("nan")

    return {
        "variant": label,
        "oos_years": round(n_y, 2),
        "sharpe": round(sh, 3),
        "ci95_lo": round(ci_lo, 3),
        "ci95_hi": round(ci_hi, 3),
        "cagr": round(cagr, 4),
        "vol": round(vol, 4),
        "max_dd": round(dd, 4),
        "calmar": round(calmar, 3),
        "calmar_ci_lo": round(calmar_ci_lo, 3),
        "sanctuary_sharpe": round(san_sh, 3),
        "sanctuary_max_dd": round(san_dd, 4),
        "cum_2022": round(cum_2022, 4),
    }


# ── Baseline strategies ──────────────────────────────────────────────────


def _baseline_spy_buyhold(spy: pd.Series) -> pd.Series:
    """B1. Pure SPY TR returns, no leverage, no gate."""
    return spy.pct_change()


def _baseline_always_on_mes(spy: pd.Series, L: float) -> pd.Series:
    """B2. Constant-L MES futures, no regime gate."""
    return FuturesEngine().daily_returns(spy, leverage=L)


def _baseline_vix_zscore_gate(
    spy: pd.Series, vix: pd.Series, L: float, *, z_threshold: float = 1.0
) -> pd.Series:
    """B3. Deploy MES at L when VIX 252-day z-score < +1, else cash.

    Simple one-line baseline — does the HMM beat this?
    """
    common = spy.index.intersection(vix.index)
    spy_c = spy.reindex(common)
    vix_c = vix.reindex(common)

    z = rolling_zscore(vix_c, window=252, min_periods=63)
    # Deploy when VIX z-score is BELOW threshold (low VIX = benign)
    deployed = (z < z_threshold).shift(1).fillna(False).astype(int)

    ret_lev = FuturesEngine().daily_returns(spy_c, leverage=L).reindex(common).fillna(0.0)
    fund = funding_series(common).reindex(common).ffill().fillna(0.04)
    ret_cash = (fund / 252.0).astype(float)
    return (deployed * ret_lev + (1 - deployed) * ret_cash).astype(float)


def _baseline_v2_6indicator_gate(data: dict, L: float, *, threshold: float = 0.5) -> pd.Series:
    """B4. V2 6-indicator equal-weight gate. Deploy MES at L when score > τ."""
    spy = data["spy"]
    panel = build_indicator_panel(
        spy,
        vix_close=data["vix"],
        hyg_close=data.get("hyg"),
        ief_close=data.get("ief"),
        tlt_close=data.get("tlt"),
    )
    score = regime_score_equal(panel)
    common = spy.index.intersection(score.index)
    spy_c = spy.reindex(common)
    score_c = score.reindex(common)

    deployed = (score_c.shift(1) > threshold).fillna(False).astype(int)
    ret_lev = FuturesEngine().daily_returns(spy_c, leverage=L).reindex(common).fillna(0.0)
    fund = funding_series(common).reindex(common).ffill().fillna(0.04)
    ret_cash = (fund / 252.0).astype(float)
    return (deployed * ret_lev + (1 - deployed) * ret_cash).astype(float)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    print("Samir-Stack V3 Layer 1+2 validation")
    print("=" * 100)

    data = load_panel(start="2003-04-01", end="2026-04-02")
    spy = data["spy"]
    vix = data["vix"]
    common = spy.index.intersection(vix.index)
    spy = spy.reindex(common)
    vix = vix.reindex(common)
    print(f"Window: {common.min().date()} → {common.max().date()} ({len(common)} bars)")
    print()

    # ── Layer 1: compute the VIX-HMM regime score (causal) ──────────────
    print("Computing VIX-HMM regime score (causal rolling 2-state HMM)...", flush=True)
    score = vix_hmm_regime_score(vix)
    print(
        f"  Non-NaN bars: {score.notna().sum()}, "
        f"frac > 0.5: {(score > 0.5).mean():.3f}, "
        f"median: {score.median():.3f}"
    )

    diag = vix_hmm_forward_vol_discrimination(vix, spy, score, threshold=0.5)
    print()
    print("Forward 21-day SPY vol discrimination (threshold=0.5):")
    print(f"  Benign (score>0.5)  → fwd-vol mean: {diag['fwd_vol_benign_mean']:.4f}")
    print(f"  Hostile (score≤0.5) → fwd-vol mean: {diag['fwd_vol_hostile_mean']:.4f}")
    print(f"  Hostile/Benign ratio: {diag['ratio_hostile_over_benign']:.3f}")
    print("  (V2 indicator panel ratios were 1.93-2.43; we want >1.5 for layer-1 pass)")
    print()

    rows: list[dict] = []

    # ── B1: SPY buy-and-hold ─────────────────────────────────────────────
    print("Running B1: SPY buy-and-hold...", flush=True)
    rows.append(_summary("B1: SPY buy-hold", _baseline_spy_buyhold(spy)))

    # ── B2: Always-on MES (no gate) at L=2 and L=3 ──────────────────────
    for L in (2.0, 3.0):
        print(f"Running B2: always-on MES L={L:.0f}...", flush=True)
        rows.append(_summary(f"B2: always-on MES L={int(L)}", _baseline_always_on_mes(spy, L)))

    # ── B3: VIX z-score threshold gate at L=2 and L=3 ───────────────────
    for L in (2.0, 3.0):
        print(f"Running B3: VIX z-score gate L={L:.0f}...", flush=True)
        rows.append(
            _summary(
                f"B3: VIX z-score gate L={int(L)}",
                _baseline_vix_zscore_gate(spy, vix, L),
            )
        )

    # ── B4: V2 6-indicator equal-weight gate at L=2 and L=3 ─────────────
    for L in (2.0, 3.0):
        print(f"Running B4: V2 6-indicator gate L={L:.0f}...", flush=True)
        rows.append(
            _summary(
                f"B4: V2 6-indicator gate L={int(L)}",
                _baseline_v2_6indicator_gate(data, L),
            )
        )

    # ── V3: VIX-HMM gate at L=2, L=3, L=4 ───────────────────────────────
    for L in (2.0, 3.0, 4.0):
        print(f"Running V3: VIX-HMM gate L={L:.0f}...", flush=True)
        cfg = V3Config(score_threshold=0.5, L_target=L)
        df = run_v3_strategy(spy, score, cfg)
        rows.append(_summary(f"V3: VIX-HMM gate L={int(L)}", df["ret_strategy"]))

    # ── Reports ──────────────────────────────────────────────────────────
    summary = pd.DataFrame(rows).set_index("variant")
    if "error" in summary.columns:
        err_mask = summary["error"].notna()
        if err_mask.any():
            print(f"\nWarning: {int(err_mask.sum())} variant(s) errored:")
            print(summary[err_mask][["error"]].to_string())
            summary = summary[~err_mask].drop(columns=["error"])
        else:
            summary = summary.drop(columns=["error"])

    print()
    print("=" * 100)
    print("HEAD-TO-HEAD RESULTS")
    print("=" * 100)
    print(summary.to_string())
    summary.to_csv(REPORTS_DIR / "layer1_head_to_head.csv")

    # ── Verdict gates ────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("V3 LAYER 1 VERDICT GATES")
    print("=" * 100)
    if diag["ratio_hostile_over_benign"] > 1.5:
        print(
            f"  [PASS] HMM forward-vol discrimination > 1.5x ({diag['ratio_hostile_over_benign']:.2f}x)"
        )
    else:
        print(
            f"  [FAIL] HMM forward-vol discrimination ≤ 1.5x ({diag['ratio_hostile_over_benign']:.2f}x)"
        )

    # Compare V3 (L=2) to V2 baseline at L=2 — must clear v2's Calmar CI lo
    v3_l2 = summary.loc["V3: VIX-HMM gate L=2"] if "V3: VIX-HMM gate L=2" in summary.index else None
    v2_l2 = (
        summary.loc["B4: V2 6-indicator gate L=2"]
        if "B4: V2 6-indicator gate L=2" in summary.index
        else None
    )
    if v3_l2 is not None and v2_l2 is not None:
        v3_clo = v3_l2["calmar_ci_lo"]
        v2_clo = v2_l2["calmar_ci_lo"]
        if v3_clo > v2_clo:
            print(f"  [PASS] V3 Calmar CI lo {v3_clo:.3f} > V2 {v2_clo:.3f}")
        else:
            print(
                f"  [INFO] V3 Calmar CI lo {v3_clo:.3f} ≤ V2 {v2_clo:.3f} — V3 layer 1 does not clear v2"
            )

    if v3_l2 is not None:
        if v3_l2["ci95_lo"] > 0:
            print(f"  [PASS] V3 Sharpe CI95 lo > 0 ({v3_l2['ci95_lo']:.3f})")
        else:
            print(f"  [FAIL] V3 Sharpe CI95 lo ≤ 0 ({v3_l2['ci95_lo']:.3f})")
        if v3_l2["sanctuary_sharpe"] >= 0:
            print(f"  [PASS] V3 sanctuary Sharpe ≥ 0 ({v3_l2['sanctuary_sharpe']:.3f})")
        else:
            print(f"  [FAIL] V3 sanctuary Sharpe < 0 ({v3_l2['sanctuary_sharpe']:.3f})")

    print(f"\nSaved: {REPORTS_DIR / 'layer1_head_to_head.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
