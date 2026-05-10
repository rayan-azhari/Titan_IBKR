"""Block-bootstrap Monte Carlo for RoR distribution.

For RoR-first design we need the distribution of outcomes, not a single
realisation. Standard IID bootstrap destroys serial correlation, which
matters because regime persistence is a key feature of equity returns
(vol clusters). We use a stationary bootstrap (Politis & Romano 1994)
with mean block length = 21 days, equivalent to one trading month.

Reports:
    - CAGR distribution (median, 5%, 25%, 75%, 95%)
    - MaxDD distribution
    - Calmar distribution
    - Tail probabilities: P(MaxDD > 25%), P(MaxDD > 50%), P(CAGR < 0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stationary_bootstrap_indices(
    n_obs: int, mean_block_len: int = 21, *, seed: int | None = None
) -> np.ndarray:
    """Generate one stationary-bootstrap resample of indices.

    Uses geometric block-length distribution with mean = mean_block_len.
    Each block starts at a uniformly-random observation index. Indices
    are recycled circularly (so the right-edge of the data is never
    over-represented).
    """
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block_len
    indices = np.empty(n_obs, dtype=np.int64)
    i = 0
    while i < n_obs:
        start = rng.integers(0, n_obs)
        # Geometric block length (always at least 1)
        block_len = max(1, int(rng.geometric(p)))
        for j in range(block_len):
            if i >= n_obs:
                break
            indices[i] = (start + j) % n_obs
            i += 1
    return indices


def _path_metrics(rets: np.ndarray) -> dict:
    """Compute CAGR / MaxDD / Calmar on a single path (np array)."""
    n_years = len(rets) / 252.0
    eq = np.cumprod(1.0 + rets)
    cagr = float(eq[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd_series = (eq - peak) / peak
    maxdd = float(dd_series.min())
    calmar = cagr / abs(maxdd) if maxdd < -1e-9 else 0.0
    return {"cagr": cagr, "max_dd": maxdd, "calmar": calmar}


def monte_carlo_ror(
    returns: pd.Series,
    *,
    n_paths: int = 10_000,
    mean_block_len: int = 21,
    seed: int = 42,
) -> dict:
    """Run a block-bootstrap RoR distribution on a daily return series.

    Returns a dict with:
        n_paths, n_bars
        cagr_median, cagr_p05, cagr_p25, cagr_p75, cagr_p95
        maxdd_median, maxdd_p05, maxdd_p95  (note: p05 = worst 5% of paths)
        prob_maxdd_gt_25pct, prob_maxdd_gt_50pct, prob_cagr_negative
        prob_calmar_positive
    """
    rets = returns.dropna().to_numpy()
    n_obs = len(rets)
    if n_obs < 252:
        raise ValueError(f"Need at least 252 bars, got {n_obs}")

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_paths)

    cagrs = np.empty(n_paths)
    maxdds = np.empty(n_paths)
    calmars = np.empty(n_paths)

    for i in range(n_paths):
        idx = stationary_bootstrap_indices(n_obs, mean_block_len, seed=int(seeds[i]))
        path_rets = rets[idx]
        m = _path_metrics(path_rets)
        cagrs[i] = m["cagr"]
        maxdds[i] = m["max_dd"]
        calmars[i] = m["calmar"]

    return {
        "n_paths": n_paths,
        "n_bars": n_obs,
        "n_years": round(n_obs / 252.0, 2),
        "mean_block_len": mean_block_len,
        # CAGR distribution
        "cagr_median": round(float(np.median(cagrs)), 4),
        "cagr_p05": round(float(np.quantile(cagrs, 0.05)), 4),
        "cagr_p25": round(float(np.quantile(cagrs, 0.25)), 4),
        "cagr_p75": round(float(np.quantile(cagrs, 0.75)), 4),
        "cagr_p95": round(float(np.quantile(cagrs, 0.95)), 4),
        # MaxDD distribution (note: more negative = worse, so p05 is worst)
        "maxdd_median": round(float(np.median(maxdds)), 4),
        "maxdd_p05": round(float(np.quantile(maxdds, 0.05)), 4),
        "maxdd_p25": round(float(np.quantile(maxdds, 0.25)), 4),
        "maxdd_p95": round(float(np.quantile(maxdds, 0.95)), 4),
        # Tail probabilities — RoR-first metrics
        "prob_maxdd_gt_25pct": round(float((maxdds < -0.25).mean()), 4),
        "prob_maxdd_gt_35pct": round(float((maxdds < -0.35).mean()), 4),
        "prob_maxdd_gt_50pct": round(float((maxdds < -0.50).mean()), 4),
        "prob_cagr_negative": round(float((cagrs < 0).mean()), 4),
        "prob_cagr_lt_3pct": round(float((cagrs < 0.03).mean()), 4),
        # Calmar distribution
        "calmar_median": round(float(np.median(calmars)), 3),
        "calmar_p05": round(float(np.quantile(calmars, 0.05)), 3),
    }
