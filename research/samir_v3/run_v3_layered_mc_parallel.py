"""Parallelised underlying-resampled MC for V3 layered defenses.

Same hypothesis as ``run_v3_layered_mc.py`` but uses
``concurrent.futures.ProcessPoolExecutor`` to run paths across all
available CPU cores. Each MC path is independent (embarrassingly
parallel) so the speedup is near-linear in core count.

Expected runtime on 8 cores: ~5 min for 100 paths × 2 leverages × 4
variants (vs ~35-40 min single-threaded).

Notes on Windows multiprocessing:
- Uses spawn (not fork). Each worker re-imports the module. The worker
  function must be importable at module level (it is — _mc_worker
  is defined here).
- Input series are pickled to the worker. SPY/VIX are ~5800 floats so
  pickle overhead is negligible relative to the 10s per-path compute.
- ``if __name__ == "__main__":`` guard is mandatory on Windows.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.samir_stack.data_loader import load_panel  # noqa: E402
from research.samir_v3.run_v3_layered_mc import (  # noqa: E402
    _build_variants,
    _stationary_bootstrap_indices,
)
from research.samir_v3.strategy_v3 import run_v3_strategy  # noqa: E402
from research.samir_v3.vix_hmm import vix_hmm_regime_score  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / ".tmp" / "reports" / "samir_v3"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _mc_worker(
    spy_values: np.ndarray,
    vix_values: np.ndarray,
    spy_index: pd.DatetimeIndex,
    vix_index: pd.DatetimeIndex,
    L_target: float,
    seed: int,
    mean_block: int = 21,
) -> tuple[int, float, dict[str, dict]]:
    """One MC path. Worker function for the ProcessPoolExecutor.

    Returns (seed, L_target, dict[variant_name -> metrics]).
    Receives arrays + indices rather than Series to keep the pickle
    payload minimal.
    """
    import suppress_warnings  # type: ignore  # noqa: F401 — best-effort

    pass


def _path_metrics(
    spy_values: np.ndarray,
    vix_values: np.ndarray,
    spy_index: pd.DatetimeIndex,
    vix_index: pd.DatetimeIndex,
    L_target: float,
    seed: int,
    mean_block: int,
) -> dict[str, dict]:
    """Run one MC path → all 4 variants → metrics."""
    import warnings

    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    spy_r = np.diff(spy_values) / spy_values[:-1]
    spy_r = np.concatenate([[0.0], spy_r])
    vix_r = np.diff(vix_values) / vix_values[:-1]
    vix_r = np.concatenate([[0.0], vix_r])
    n = len(spy_r)
    idx = _stationary_bootstrap_indices(n, mean_block, rng)

    spy_synth = pd.Series(
        float(spy_values[0]) * np.cumprod(1.0 + spy_r[idx]),
        index=spy_index,
        name="SPY",
    )
    vix_synth_arr = float(vix_values[0]) * np.cumprod(1.0 + vix_r[idx])
    vix_synth_arr = np.clip(vix_synth_arr, 5.0, None)
    vix_synth = pd.Series(vix_synth_arr, index=vix_index, name="VIX")

    score = vix_hmm_regime_score(vix_synth)

    out = {}
    for name, cfg in _build_variants(L_target).items():
        df = run_v3_strategy(spy_synth, score, cfg)
        rets = df["ret_strategy"].dropna().to_numpy()
        n_y = len(rets) / 252.0
        eq = np.cumprod(1.0 + rets)
        cagr = float(eq[-1] ** (1.0 / n_y) - 1.0) if n_y > 0 else 0.0
        dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
        out[name] = {
            "cagr": cagr,
            "max_dd": dd,
            "calmar": cagr / abs(dd) if dd < -1e-9 else 0.0,
            "frac_deployed": float(df["deployed"].mean()),
            "frac_in_kill_state": float((df["dd_state"] == "killed").mean())
            if "dd_state" in df.columns
            else 0.0,
        }
    return out


def main() -> int:
    data = load_panel(start="2003-04-01", end="2026-04-02")
    common = data["spy"].index.intersection(data["vix"].index)
    spy = data["spy"].reindex(common)
    vix = data["vix"].reindex(common)
    n_cores = max(1, (os.cpu_count() or 4) - 1)
    print(f"V3 layered MC (parallel). Window: {common.min().date()} → {common.max().date()}")
    print(f"Workers: {n_cores} (cpu_count - 1)")

    n_paths = 100
    print(f"Running {n_paths} paths × 2 leverages × 4 variants...")
    print()

    spy_values = spy.to_numpy()
    vix_values = vix.to_numpy()
    spy_index = spy.index
    vix_index = vix.index

    all_results = []

    for L_target in (2.0, 3.0):
        print(f"==> L_target = {L_target}", flush=True)
        t0 = time.perf_counter()
        per_variant: dict[str, dict[str, list]] = {
            name: {
                "cagr": [],
                "max_dd": [],
                "calmar": [],
                "frac_deployed": [],
                "frac_in_kill_state": [],
            }
            for name in _build_variants(L_target)
        }

        rng = np.random.default_rng(42)
        seeds = rng.integers(0, 2**31, size=n_paths)

        completed = 0
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            futures = {
                pool.submit(
                    _path_metrics,
                    spy_values,
                    vix_values,
                    spy_index,
                    vix_index,
                    L_target,
                    int(seed),
                    21,
                ): seed
                for seed in seeds
            }
            for fut in as_completed(futures):
                res = fut.result()
                for name, m in res.items():
                    for k, v in m.items():
                        per_variant[name][k].append(v)
                completed += 1
                if completed % 10 == 0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"  {completed}/{n_paths} paths done  elapsed {elapsed:.0f}s",
                        flush=True,
                    )

        print()
        print(f"=== L_target={L_target} results ===")
        for name, m in per_variant.items():
            cagrs = np.asarray(m["cagr"])
            maxdds = np.asarray(m["max_dd"])
            calmars = np.asarray(m["calmar"])
            row = {
                "L_target": L_target,
                "variant": name,
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
                "avg_frac_deployed": float(np.mean(m["frac_deployed"])),
                "avg_frac_in_kill": float(np.mean(m["frac_in_kill_state"])),
            }
            all_results.append(row)
            print(
                f"  {name:20s}  CAGR(med)={row['mc_cagr_median'] * 100:6.2f}%  "
                f"MaxDD(med)={row['mc_maxdd_median'] * 100:7.2f}%  "
                f"P(DD>25)={row['mc_p_dd_gt_25'] * 100:5.1f}%  "
                f"P(DD>35)={row['mc_p_dd_gt_35'] * 100:5.1f}%  "
                f"P(DD>50)={row['mc_p_dd_gt_50'] * 100:5.1f}%  "
                f"deploy={row['avg_frac_deployed'] * 100:.1f}%"
            )
        print()

    out = pd.DataFrame(all_results)
    out.to_csv(REPORTS_DIR / "layered_defense_mc.csv", index=False)
    print(f"Saved: {REPORTS_DIR / 'layered_defense_mc.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
