"""Standardised walk-forward fold construction.

Specified in directives/Methodology Audit & Unified Framework 2026-05-14.md
§2.3. Fixes gaps C1 (inconsistent WFO designs) and C6 (no standard config).

Two fold modes per strategy class:

    "expanding"  — anchored at the start, IS expands each fold, non-overlapping OOS
    "rolling"    — fixed-length IS slides forward, optional OOS overlap

All folds operate on the VISIBLE portion (post-sanctuary slice). Sanctuary
discipline is enforced via the SanctuarySlice already applied upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from titan.research.framework.typology import WfoConfig


@dataclass(frozen=True)
class Fold:
    """One WFO fold. Index positions into the visible DataFrame."""

    fold_id: int
    is_start: int
    is_end_excl: int  # exclusive
    oos_start: int  # == is_end_excl
    oos_end_excl: int  # exclusive
    # Wall-clock boundaries for audit logs.
    is_start_ts: pd.Timestamp
    is_end_ts: pd.Timestamp
    oos_start_ts: pd.Timestamp
    oos_end_ts: pd.Timestamp

    @property
    def n_is_bars(self) -> int:
        return self.is_end_excl - self.is_start

    @property
    def n_oos_bars(self) -> int:
        return self.oos_end_excl - self.oos_start


def build_folds(
    visible_index: pd.DatetimeIndex,
    cfg: WfoConfig,
    *,
    bars_per_year: float,
) -> list[Fold]:
    """Construct WFO folds per the per-class config.

    Parameters
    ----------
    visible_index:
        The post-sanctuary DatetimeIndex of the data.
    cfg:
        WfoConfig from typology.defaults_for(strategy_class).wfo.
    bars_per_year:
        Conversion from `cfg.is_min_years` / `cfg.oos_years` to bar
        counts. For D-frequency strategies use 252; for H1 use 252*24;
        etc.

    Returns:
    -------
    folds : list[Fold]
        Possibly empty if the visible window is too short for the
        requested IS_min + OOS configuration.
    """
    n = len(visible_index)
    if n == 0:
        return []
    is_min_bars = int(cfg.is_min_years * bars_per_year)
    oos_bars = int(cfg.oos_years * bars_per_year)
    if is_min_bars <= 0 or oos_bars <= 0:
        return []
    if n < is_min_bars + oos_bars:
        return []

    folds: list[Fold] = []
    if cfg.is_mode == "expanding":
        # Anchored at index 0; IS expands; non-overlapping OOS.
        # For fold k: IS = [0, is_min_bars + k * oos_bars), OOS = next oos_bars.
        for k in range(cfg.fold_count):
            is_end = is_min_bars + k * oos_bars
            oos_end = is_end + oos_bars
            if oos_end > n:
                break
            folds.append(
                Fold(
                    fold_id=k,
                    is_start=0,
                    is_end_excl=is_end,
                    oos_start=is_end,
                    oos_end_excl=oos_end,
                    is_start_ts=visible_index[0],
                    is_end_ts=visible_index[is_end - 1],
                    oos_start_ts=visible_index[is_end],
                    oos_end_ts=visible_index[oos_end - 1],
                )
            )
    elif cfg.is_mode == "rolling":
        # Fixed-length IS that slides forward. Stride = oos_bars; overlap
        # of OOS windows is allowed if cfg.stride_overlap_allowed.
        # For fold k: IS = [stride * k, stride * k + is_min_bars), OOS = next oos_bars.
        stride = oos_bars if not cfg.stride_overlap_allowed else max(1, oos_bars // 2)
        for k in range(cfg.fold_count):
            is_start = stride * k
            is_end = is_start + is_min_bars
            oos_end = is_end + oos_bars
            if oos_end > n:
                break
            folds.append(
                Fold(
                    fold_id=k,
                    is_start=is_start,
                    is_end_excl=is_end,
                    oos_start=is_end,
                    oos_end_excl=oos_end,
                    is_start_ts=visible_index[is_start],
                    is_end_ts=visible_index[is_end - 1],
                    oos_start_ts=visible_index[is_end],
                    oos_end_ts=visible_index[oos_end - 1],
                )
            )
    else:
        raise ValueError(f"Unknown WFO mode: {cfg.is_mode!r}")
    return folds


def iter_folds(
    visible: pd.DataFrame,
    cfg: WfoConfig,
    *,
    bars_per_year: float,
) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    """Convenience generator: yields ``(fold, is_df, oos_df)`` tuples."""
    folds = build_folds(visible.index, cfg, bars_per_year=bars_per_year)
    for f in folds:
        yield (
            f,
            visible.iloc[f.is_start : f.is_end_excl],
            visible.iloc[f.oos_start : f.oos_end_excl],
        )
