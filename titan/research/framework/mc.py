"""Standardised block-bootstrap Monte Carlo.

Specified in directives/Methodology Audit & Unified Framework 2026-05-14.md
§2.4. Fixes gaps F1 (block-size sensitivity), F2 (broken threshold),
F3 (method ambiguity), F4 (correlation preservation), F6 (no class
defaults).

Three bootstrap methods:

    "block"        — Resample (block-size) chunks from a single return
                     series. Preserves serial autocorrelation up to
                     block boundaries.
    "shared_block" — Two or more series sampled at the SAME block
                     indices. Preserves cross-asset correlation.
    "stationary"   — Politis & Romano (1994) stationary block bootstrap
                     with random block lengths. Not yet implemented;
                     placeholder for future addition.

All methods rebuild synthetic price paths via cumulative product of
resampled log returns, then re-run the strategy on each path. The
strategy's MaxDD distribution + Sharpe distribution are reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from titan.research.framework.typology import McConfig
from titan.research.metrics import max_drawdown, sharpe


@dataclass(frozen=True)
class McResult:
    """Aggregate stats from a Monte Carlo run."""

    n_paths_completed: int
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    median_maxdd: float
    p_maxdd_gt_threshold: float
    threshold_pct: float
    pass_threshold_prob: float
    passes: bool
    method: str
    block_size: int


def _resample_indices(
    n_bars: int, block_size: int, rng: np.random.Generator,
) -> np.ndarray:
    """Return an array of n_bars resampled indices, drawn as overlapping
    blocks of `block_size`. Each block start is uniformly random in
    [0, n_bars - block_size]. Trailing partial block is truncated.
    """
    if n_bars <= block_size:
        return np.arange(n_bars)
    n_blocks = (n_bars + block_size - 1) // block_size
    starts = rng.integers(0, n_bars - block_size, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
    return idx[:n_bars]


def _rebuild_path(
    log_returns: np.ndarray, indices: np.ndarray, initial_price: float,
) -> np.ndarray:
    """cumprod of resampled log returns → synthetic price series."""
    resampled = log_returns[indices]
    return initial_price * np.exp(np.cumsum(resampled))


def run_block_mc(
    primary_close: pd.Series,
    cfg: McConfig,
    strategy_fn: Callable[[pd.DataFrame], pd.Series],
    *,
    periods_per_year: int,
    seed: int = 42,
    extra_series: dict[str, pd.Series] | None = None,
) -> McResult:
    """Block bootstrap of the primary close (and optional extras at SHARED
    indices for cross-asset strategies).

    Parameters
    ----------
    primary_close:
        The underlying price series the strategy trades.
    cfg:
        McConfig from typology.defaults_for(strategy_class).mc.
    strategy_fn:
        Callable that takes a DataFrame with at least 'close' column
        (and any 'extra_series' keys re-named lowercase) and returns
        a per-bar return Series.
    periods_per_year:
        Annualisation factor for Sharpe.
    seed:
        RNG seed for reproducibility.
    extra_series:
        Optional dict of additional series (e.g. {'bond_close': ...,
        'high': ..., 'low': ...}) that share the primary's bar
        timestamps. For "shared_block" method these are resampled at
        the SAME indices as the primary to preserve correlation.

    Returns
    -------
    McResult
    """
    if cfg.bootstrap_method == "stationary":
        raise NotImplementedError(
            "stationary bootstrap not yet implemented. "
            "Use 'block' or 'shared_block' until added in a follow-up."
        )

    primary_close = primary_close.dropna()
    if len(primary_close) < cfg.block_size_bars * 2:
        return McResult(
            n_paths_completed=0,
            median_sharpe=0.0, p5_sharpe=0.0, p95_sharpe=0.0,
            median_maxdd=0.0, p_maxdd_gt_threshold=1.0,
            threshold_pct=cfg.max_dd_threshold_pct,
            pass_threshold_prob=cfg.max_dd_pass_prob,
            passes=False,
            method=cfg.bootstrap_method, block_size=cfg.block_size_bars,
        )

    log_returns_primary = np.log(primary_close).diff().dropna().to_numpy()
    n_bars = len(log_returns_primary)
    initial_primary = float(primary_close.iloc[0])

    extras_log_rets: dict[str, np.ndarray] = {}
    extras_initial: dict[str, float] = {}
    if extra_series:
        for name, s in extra_series.items():
            s = s.reindex(primary_close.index).dropna()
            # Truncate primary to the common index for the MC run.
            common = primary_close.index.intersection(s.index)
            if len(common) < cfg.block_size_bars * 2:
                continue
            extras_log_rets[name] = np.log(s.reindex(common)).diff().dropna().to_numpy()
            extras_initial[name] = float(s.iloc[0])
        if extras_log_rets:
            # Re-align primary to the common index
            common_idx = primary_close.index
            for name, _ in extras_log_rets.items():
                common_idx = common_idx.intersection(extra_series[name].index)
            primary_close = primary_close.reindex(common_idx).dropna()
            log_returns_primary = np.log(primary_close).diff().dropna().to_numpy()
            n_bars = len(log_returns_primary)

    rng = np.random.default_rng(seed)
    sharpes: list[float] = []
    maxdds: list[float] = []

    for _path in range(cfg.n_paths):
        if cfg.bootstrap_method == "shared_block":
            indices = _resample_indices(n_bars, cfg.block_size_bars, rng)
            synth_primary = _rebuild_path(log_returns_primary, indices, initial_primary)
            df = pd.DataFrame({"close": synth_primary}, index=primary_close.index[: len(synth_primary)])
            for name, lr in extras_log_rets.items():
                synth = _rebuild_path(lr, indices[: len(lr)], extras_initial[name])
                df[name] = pd.Series(synth, index=df.index[: len(synth)])
        else:  # plain "block"
            indices = _resample_indices(n_bars, cfg.block_size_bars, rng)
            synth_primary = _rebuild_path(log_returns_primary, indices, initial_primary)
            df = pd.DataFrame({"close": synth_primary}, index=primary_close.index[: len(synth_primary)])
            for name, lr in extras_log_rets.items():
                # Each extra resampled independently (block bootstrap doesn't
                # preserve cross-asset correlation)
                ext_indices = _resample_indices(len(lr), cfg.block_size_bars, rng)
                synth = _rebuild_path(lr, ext_indices, extras_initial[name])
                df[name] = pd.Series(synth, index=df.index[: len(synth)])

        try:
            strat_ret = strategy_fn(df)
        except Exception:
            continue
        if len(strat_ret) < 20:
            continue
        sh = sharpe(strat_ret, periods_per_year=periods_per_year)
        mdd = max_drawdown(strat_ret)
        sharpes.append(sh)
        maxdds.append(mdd)

    if not sharpes:
        return McResult(
            n_paths_completed=0,
            median_sharpe=0.0, p5_sharpe=0.0, p95_sharpe=0.0,
            median_maxdd=0.0, p_maxdd_gt_threshold=1.0,
            threshold_pct=cfg.max_dd_threshold_pct,
            pass_threshold_prob=cfg.max_dd_pass_prob,
            passes=False,
            method=cfg.bootstrap_method, block_size=cfg.block_size_bars,
        )

    sh_arr = np.asarray(sharpes)
    mdd_arr = np.asarray(maxdds)
    p_mdd_gt = float((mdd_arr < -cfg.max_dd_threshold_pct).mean())
    return McResult(
        n_paths_completed=len(sharpes),
        median_sharpe=float(round(np.median(sh_arr), 4)),
        p5_sharpe=float(round(np.quantile(sh_arr, 0.05), 4)),
        p95_sharpe=float(round(np.quantile(sh_arr, 0.95), 4)),
        median_maxdd=float(round(np.median(mdd_arr), 4)),
        p_maxdd_gt_threshold=round(p_mdd_gt, 4),
        threshold_pct=cfg.max_dd_threshold_pct,
        pass_threshold_prob=cfg.max_dd_pass_prob,
        passes=p_mdd_gt <= cfg.max_dd_pass_prob,
        method=cfg.bootstrap_method,
        block_size=cfg.block_size_bars,
    )
