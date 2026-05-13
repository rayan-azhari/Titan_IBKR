"""WFO head-to-head for V3 layered defenses (C, B, A) on the realized history.

Fast (~30s) — single causal HMM fit on the actual VIX series, then runs
all 4 V3 variants at L=2 and L=3 through the standard anchored WFO
harness with bootstrap Sharpe CI + bootstrap Calmar CI + sanctuary.

For the MC tail-risk evaluation see ``run_v3_layered_mc.py`` (slow).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import load_panel  # noqa: E402
from research.samir_v3.run_v3_layered_mc import _build_variants  # noqa: E402
from research.samir_v3.strategy_v3 import run_v3_strategy  # noqa: E402
from research.samir_v3.vix_hmm import vix_hmm_regime_score  # noqa: E402
from titan.research.metrics import (  # noqa: E402
    BARS_PER_YEAR,
    annualize_vol,
    bootstrap_sharpe_ci,
    sharpe,
)

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_v3"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


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


def main() -> int:
    print("V3 Layered Defense WFO Head-to-Head")
    print("=" * 100)

    data = load_panel(start="2003-04-01", end="2026-04-02")
    common = data["spy"].index.intersection(data["vix"].index)
    spy = data["spy"].reindex(common)
    vix = data["vix"].reindex(common)
    print(f"Window: {common.min().date()} → {common.max().date()} ({len(common)} bars)")

    print("Computing VIX-HMM regime score (causal rolling)...", flush=True)
    score = vix_hmm_regime_score(vix)
    print()

    rows = []
    for L_target in (2.0, 3.0):
        for name, cfg in _build_variants(L_target).items():
            df = run_v3_strategy(spy, score, cfg)
            label = f"{name}_L{int(L_target)}"
            rows.append(_summary(label, df["ret_strategy"]))

    summary = pd.DataFrame(rows).set_index("variant")
    if "error" in summary.columns:
        summary = summary.drop(columns=["error"])
    print(summary.to_string())
    summary.to_csv(REPORTS_DIR / "layered_defense_wfo.csv")
    print(f"\nSaved: {REPORTS_DIR / 'layered_defense_wfo.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
