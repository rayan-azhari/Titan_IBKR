"""Underlying-resampled Monte Carlo for V3 VIX-HMM gate at L=2 and L=3.

Bootstraps (SPY, VIX) daily returns jointly via shared block indices to
preserve cross-asset correlation, cumprod to rebuild synthetic price
paths, then re-runs the causal VIX-HMM and the V3 strategy on each path.

Output: tail-risk distribution for the deployment RoR gate.

Per the v3 design directive §10, V3 must show MC P(MaxDD>50%) < 1% to
clear the RoR gate. Same threshold as Phase 5's deployment criterion.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import load_panel  # noqa: E402
from research.samir_v3.strategy_v3 import V3Config, run_v3_strategy  # noqa: E402
from research.samir_v3.vix_hmm import vix_hmm_regime_score  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_v3"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _stationary_bootstrap_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        bl = max(1, int(rng.geometric(p)))
        for j in range(bl):
            if i >= n:
                break
            idx[i] = (start + j) % n
            i += 1
    return idx


def _resample_path(
    spy: pd.Series, vix: pd.Series, L_target: float, seed: int, mean_block: int = 21
) -> dict:
    rng = np.random.default_rng(seed)
    spy_r = spy.pct_change().fillna(0.0).to_numpy()
    vix_r = vix.pct_change().fillna(0.0).to_numpy()
    n = len(spy_r)
    idx = _stationary_bootstrap_indices(n, mean_block, rng)

    spy_synth = pd.Series(
        float(spy.iloc[0]) * np.cumprod(1.0 + spy_r[idx]), index=spy.index, name=spy.name
    )
    vix_synth = pd.Series(
        float(vix.iloc[0]) * np.cumprod(1.0 + vix_r[idx]), index=vix.index, name=vix.name
    ).clip(lower=5.0)  # VIX can't go below ~5 in practice; clip after bootstrap

    score = vix_hmm_regime_score(vix_synth)
    cfg = V3Config(score_threshold=0.5, L_target=L_target)
    df = run_v3_strategy(spy_synth, score, cfg)

    rets = df["ret_strategy"].dropna().to_numpy()
    n_y = len(rets) / 252.0
    eq = np.cumprod(1.0 + rets)
    cagr = float(eq[-1] ** (1.0 / n_y) - 1.0) if n_y > 0 else 0.0
    dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
    return {"cagr": cagr, "max_dd": dd, "calmar": cagr / abs(dd) if dd < -1e-9 else 0.0}


def main() -> int:
    data = load_panel(start="2003-04-01", end="2026-04-02")
    common = data["spy"].index.intersection(data["vix"].index)
    spy = data["spy"].reindex(common)
    vix = data["vix"].reindex(common)
    print(f"V3 underlying-resampled MC. Window: {common.min().date()} → {common.max().date()}")

    n_paths = 100  # smaller than Phase 5's 500 due to HMM refit cost per path
    print(f"Running {n_paths} paths per leverage level (this is the slow step — ~25-30 min total)")
    print()

    all_rows = []
    for L_target in (2.0, 3.0):
        print(f"==> L_target = {L_target}", flush=True)
        t0 = time.perf_counter()
        cagrs = np.empty(n_paths)
        maxdds = np.empty(n_paths)
        calmars = np.empty(n_paths)
        rng = np.random.default_rng(42)
        seeds = rng.integers(0, 2**31, size=n_paths)
        for i in range(n_paths):
            m = _resample_path(spy, vix, L_target, int(seeds[i]))
            cagrs[i] = m["cagr"]
            maxdds[i] = m["max_dd"]
            calmars[i] = m["calmar"]
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  path {i + 1}/{n_paths}  elapsed {elapsed:.0f}s",
                    flush=True,
                )

        row = {
            "L_target": L_target,
            "n_paths": n_paths,
            "mc_cagr_median": float(np.median(cagrs)),
            "mc_cagr_p05": float(np.quantile(cagrs, 0.05)),
            "mc_maxdd_median": float(np.median(maxdds)),
            "mc_maxdd_p05": float(np.quantile(maxdds, 0.05)),
            "mc_calmar_median": float(np.median(calmars)),
            "mc_calmar_p025": float(np.quantile(calmars, 0.025)),
            "mc_p_dd_gt_25": float((maxdds < -0.25).mean()),
            "mc_p_dd_gt_35": float((maxdds < -0.35).mean()),
            "mc_p_dd_gt_50": float((maxdds < -0.50).mean()),
            "mc_p_cagr_neg": float((cagrs < 0).mean()),
        }
        all_rows.append(row)
        print()
        for k, v in row.items():
            print(f"  {k}: {v}")
        print()

    out = pd.DataFrame(all_rows)
    out.to_csv(REPORTS_DIR / "layer1_mc_tail_risk.csv", index=False)
    print(f"Saved: {REPORTS_DIR / 'layer1_mc_tail_risk.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
