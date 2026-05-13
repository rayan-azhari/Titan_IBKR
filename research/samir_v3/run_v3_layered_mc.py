"""Underlying-resampled MC for V3 with layered defenses (C, B, A).

Tests four V3 variants on the same set of bootstrap paths to measure the
incremental risk-reduction from each defense layer. Because the HMM
refit is the dominant per-path cost, running 4 strategy configurations
on the same path costs negligibly more than 1.

Variants tested at each leverage level:
  V0  Layer 1+2 only (HMM gate, no defenses)
  VC  + Layer C (DD circuit breaker at 15% kill)
  VCB + Layer C + Layer B (momentum confluence)
  VCBA + Layer C + Layer B + Layer A (asymmetric hysteresis)

Reports tail-risk distribution (median MaxDD, P(MaxDD>X%)) per variant
per leverage. The hypothesis from the design directive §11.7:
  V0 at L=2: P(MaxDD>50%) ~ 69% (measured)
  VC: P(MaxDD>50%) ~ 5%   (mechanical cap on tail)
  VCB: ~ 1-2%             (faster gate via momentum)
  VCBA: ~ 0.5%            (asymmetric speeds exit)

If VC alone clears < 1%, V3 is deployable as pure equity at L=2 with
just the DD failsafe — no bonds required.
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


def _build_variants(L_target: float) -> dict[str, V3Config]:
    """Pre-registered 4 variants at the given leverage."""
    base_kwargs = dict(L_target=L_target)
    return {
        "V0_layer12": V3Config(**base_kwargs),
        "VC_dd_breaker": V3Config(
            **base_kwargs,
            dd_kill=0.15,
            dd_re_entry_score=0.70,
            dd_re_entry_bars=5,
        ),
        "VCB_momentum": V3Config(
            **base_kwargs,
            dd_kill=0.15,
            dd_re_entry_score=0.70,
            dd_re_entry_bars=5,
            use_momentum_confluence=True,
            mom_12_1_threshold=0.0,
            dd_velocity_threshold=-0.05,
            require_above_200d_sma=True,
        ),
        "VCBA_asymmetric": V3Config(
            **base_kwargs,
            dd_kill=0.15,
            dd_re_entry_score=0.70,
            dd_re_entry_bars=5,
            use_momentum_confluence=True,
            mom_12_1_threshold=0.0,
            dd_velocity_threshold=-0.05,
            require_above_200d_sma=True,
            score_enter_threshold=0.7,
            score_exit_threshold=0.5,
        ),
    }


def _resample_path_all_variants(
    spy: pd.Series, vix: pd.Series, L_target: float, seed: int, mean_block: int = 21
) -> dict[str, dict]:
    """One synthetic path → run ALL 4 variants → return their metrics."""
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
    ).clip(lower=5.0)

    # The expensive step — done ONCE per path, shared across variants
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
    print(f"V3 layered-defense MC. Window: {common.min().date()} → {common.max().date()}")

    n_paths = 100
    print(f"Running {n_paths} paths × 2 leverages × 4 variants (~40-45 min total)")
    print()

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
        for i in range(n_paths):
            res = _resample_path_all_variants(spy, vix, L_target, int(seeds[i]))
            for name, m in res.items():
                for k, v in m.items():
                    per_variant[name][k].append(v)
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  path {i + 1}/{n_paths}  elapsed {elapsed:.0f}s", flush=True)

        # Aggregate per variant
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
