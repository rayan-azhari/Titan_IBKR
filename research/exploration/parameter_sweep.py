"""Pardo-style parameter sweep + plateau detection.

**EXPLORATORY ONLY** — outputs from this module are NEVER promotion-eligible.

V3.6 hybrid workflow (see L52):

1. **Sweep** a parameter grid on IS-only data → per-cell Sharpe surface
   (Pardo: stable region, not peak).
2. **Plateau detect** → candidate cell whose neighbourhood is uniformly
   high *and* stable (low spread).
3. **Pre-reg commit** the detected canonical + 4 explicit neighbours from
   the same plateau region. Freeze in git.
4. **Full V3.6 audit** on OOS data (sanctuary held out). The sweep never
   sees OOS.
5. **5-axis decision matrix** is the deployment gate.

Why this hybrid:
- Pre-reg with arbitrary canonical is brittle (L43 knife-edge plateau);
  picking the centre of a flat region first avoids that.
- Pre-reg discipline still enforced — the sweep is a PRIOR; the audit on
  unseen data is the gate.
- Cheap: a typical EWMAC-grade sweep over 5x5 cells takes <30s on IS.

Causality (L04 / A1): caller must pass IS-only data. The module raises if
``closes_df`` has fewer rows than ``min_is_bars``. There is no OOS slice.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from titan.research.metrics import sharpe

# Inputs: closes (DataFrame), free-form param kwargs → per-bar returns (Series).
StrategyFn = Callable[..., pd.Series]


@dataclass(frozen=True)
class SweepResult:
    """Result of a parameter sweep.

    All cells are evaluated on the SAME IS-only data slice. ``sharpes`` is
    indexed in row-major order of the parameter grid (the same order as
    ``itertools.product`` would yield).

    Attributes:
        param_names:
            Names of the swept parameters (e.g., ``["fast_hl", "slow_hl"]``).
        param_grid:
            For each name, the list of values swept.
        cells:
            List of param-dicts, one per cell, in row-major order.
        sharpes:
            Per-cell annualised Sharpe.
        vols:
            Per-cell annualised vol.
        n_obs:
            Per-cell number of non-NaN return bars (sanity check).
        meta:
            Free-form audit metadata (data source, IS window dates,
            periods_per_year, generator notes).
    """

    param_names: tuple[str, ...]
    param_grid: dict[str, list[Any]]
    cells: list[dict[str, Any]]
    sharpes: np.ndarray
    vols: np.ndarray
    n_obs: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def to_dataframe(self) -> pd.DataFrame:
        """Long-format DataFrame: one row per cell, columns = params + metrics."""
        rows = []
        for i, cell in enumerate(self.cells):
            row = dict(cell)
            row["sharpe"] = float(self.sharpes[i])
            row["vol"] = float(self.vols[i])
            row["n_obs"] = int(self.n_obs[i])
            rows.append(row)
        return pd.DataFrame(rows)

    def to_surface(self) -> pd.DataFrame:
        """2-D Sharpe surface — only valid when exactly 2 params are swept.

        Rows = first param values; columns = second. NaN where the cell is
        missing (e.g., dominated combinations skipped by ``allow_cell``).
        """
        if len(self.param_names) != 2:
            raise ValueError(
                f"to_surface() requires exactly 2 swept params; got {len(self.param_names)}"
            )
        p1, p2 = self.param_names
        v1 = self.param_grid[p1]
        v2 = self.param_grid[p2]
        surf = pd.DataFrame(
            np.full((len(v1), len(v2)), np.nan, dtype=float),
            index=pd.Index(v1, name=p1),
            columns=pd.Index(v2, name=p2),
        )
        for cell, sr in zip(self.cells, self.sharpes, strict=True):
            surf.loc[cell[p1], cell[p2]] = float(sr)
        return surf


def _row_major_cells(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product, row-major, returning param-dicts."""
    names = list(param_grid.keys())
    values = [param_grid[n] for n in names]
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*values)]


def run_parameter_sweep(
    closes_df: pd.DataFrame,
    *,
    strategy_fn: StrategyFn,
    param_grid: dict[str, list[Any]],
    periods_per_year: int,
    min_is_bars: int = 252,
    allow_cell: Callable[[dict[str, Any]], bool] | None = None,
    meta: dict[str, Any] | None = None,
) -> SweepResult:
    """Sweep a parameter grid on IS-only data and report per-cell Sharpe.

    Parameters:
        closes_df:
            IS-only price DataFrame. Caller is responsible for slicing
            (this module enforces no OOS leakage by refusing to peek
            beyond the supplied frame).
        strategy_fn:
            Callable ``strategy_fn(closes_df, **params) -> pd.Series`` that
            returns per-bar returns. Must be deterministic given the same
            ``closes_df`` and params.
        param_grid:
            Mapping from param name to list of values to sweep. Order of
            keys defines the row-major iteration order.
        periods_per_year:
            Annualisation factor for Sharpe / vol (see BARS_PER_YEAR).
        min_is_bars:
            Reject if ``closes_df`` has fewer rows. Default 252 (1y daily).
        allow_cell:
            Optional predicate to skip dominated combinations (e.g.,
            ``lambda c: c["fast_hl"] < c["slow_hl"]``). Skipped cells
            receive Sharpe=NaN.
        meta:
            Audit metadata to attach to the result.

    Returns:
        ``SweepResult`` containing Sharpe / vol / n_obs per cell plus the
        full cell list and metadata.

    Raises:
        ValueError: if ``closes_df`` has too few bars or ``param_grid`` is
            empty.
    """
    if closes_df.shape[0] < min_is_bars:
        raise ValueError(
            f"closes_df has {closes_df.shape[0]} rows; need >= {min_is_bars} for IS sweep"
        )
    if not param_grid:
        raise ValueError("param_grid is empty")

    cells = _row_major_cells(param_grid)
    n = len(cells)
    sharpes = np.full(n, np.nan, dtype=float)
    vols = np.full(n, np.nan, dtype=float)
    n_obs = np.zeros(n, dtype=int)

    sqrt_ppy = float(np.sqrt(periods_per_year))

    for i, cell in enumerate(cells):
        if allow_cell is not None and not allow_cell(cell):
            continue
        rets = strategy_fn(closes_df, **cell)
        if not isinstance(rets, pd.Series):
            raise TypeError(
                f"strategy_fn must return pd.Series; cell={cell} returned {type(rets).__name__}"
            )
        clean = rets.dropna()
        n_obs[i] = int(clean.shape[0])
        if n_obs[i] < 20:
            continue
        sharpes[i] = sharpe(clean, periods_per_year=periods_per_year)
        sd = float(clean.std(ddof=1))
        vols[i] = sd * sqrt_ppy if np.isfinite(sd) else np.nan

    out_meta: dict[str, Any] = {
        "n_bars": int(closes_df.shape[0]),
        "first_bar": str(closes_df.index[0]) if not closes_df.empty else None,
        "last_bar": str(closes_df.index[-1]) if not closes_df.empty else None,
        "periods_per_year": periods_per_year,
        "exploratory_only": True,
        "promotion_eligible": False,
    }
    if meta:
        out_meta.update(meta)

    return SweepResult(
        param_names=tuple(param_grid.keys()),
        param_grid={k: list(v) for k, v in param_grid.items()},
        cells=cells,
        sharpes=sharpes,
        vols=vols,
        n_obs=n_obs,
        meta=out_meta,
    )


# ── Plateau detection ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlateauCandidate:
    """A candidate plateau cell + its neighbourhood.

    A "plateau" is a region where the cell AND all immediate grid neighbours
    have high Sharpe AND low cross-neighbour spread. Picking the centre as
    the canonical avoids the L43 knife-edge failure mode (canonical sits on
    a peak; nearby cells are negative).

    Attributes:
        center:
            Param-dict at the plateau centre. This is the candidate canonical.
        neighbour_cells:
            Param-dicts of the immediate grid neighbours (in any axis).
        center_sharpe:
            Sharpe at ``center``.
        neighbour_sharpes:
            Sharpes at each neighbour, in the order listed above.
        mean_neighbourhood_sharpe:
            ``mean([center_sharpe] + neighbour_sharpes)``.
        spread_pct:
            ``(max - min) / |mean|`` over the neighbourhood. The L27
            plateau gate threshold is 30%.
        rank:
            1-based rank among all candidates considered (1 = best by
            ``mean_neighbourhood_sharpe`` after passing the spread gate).
    """

    center: dict[str, Any]
    neighbour_cells: list[dict[str, Any]]
    center_sharpe: float
    neighbour_sharpes: list[float]
    mean_neighbourhood_sharpe: float
    spread_pct: float
    rank: int


def detect_plateau(
    result: SweepResult,
    *,
    spread_pct_max: float = 0.30,
    min_neighbours: int = 2,
    top_k: int = 5,
) -> list[PlateauCandidate]:
    """Find plateau candidates in a sweep result.

    A cell is a plateau candidate iff:
        - it has >= ``min_neighbours`` non-NaN grid neighbours,
        - the spread ``(max - min) / |mean|`` over (cell + neighbours) is
          <= ``spread_pct_max`` (the L27 gate),
        - the neighbourhood mean Sharpe is positive (no point in a stable
          loser).

    Returns the top-``top_k`` candidates ranked by mean neighbourhood
    Sharpe (highest first). Empty list if none satisfy.

    Notes:
        - "Neighbour" = differs in exactly one param by exactly one grid
          step (axis-aligned, not diagonals). This matches the V3.1
          plateau-neighbour convention of "±1 step on one knob".
        - For 1-D grids each cell has up to 2 neighbours; ``min_neighbours``
          defaults to 2 so edge cells can still qualify with one valid
          neighbour iff they're not literally at the grid boundary on both
          sides — set to 1 to allow edge cells, but they're more fragile.
    """
    if result.n_cells == 0:
        return []

    names = list(result.param_names)
    # Index params for fast neighbour lookup.
    grid_values: dict[str, list[Any]] = {n: list(result.param_grid[n]) for n in names}
    value_to_idx: dict[str, dict[Any, int]] = {
        n: {v: i for i, v in enumerate(vals)} for n, vals in grid_values.items()
    }
    cell_to_idx: dict[tuple[Any, ...], int] = {
        tuple(cell[n] for n in names): i for i, cell in enumerate(result.cells)
    }

    def neighbours_of(cell: dict[str, Any]) -> list[int]:
        out: list[int] = []
        for name in names:
            i = value_to_idx[name][cell[name]]
            vals = grid_values[name]
            for di in (-1, +1):
                j = i + di
                if 0 <= j < len(vals):
                    nb_key = tuple(vals[j] if n == name else cell[n] for n in names)
                    idx = cell_to_idx.get(nb_key)
                    if idx is not None:
                        out.append(idx)
        return out

    candidates: list[PlateauCandidate] = []
    for i, cell in enumerate(result.cells):
        sc = result.sharpes[i]
        if not np.isfinite(sc):
            continue
        nb_idxs = neighbours_of(cell)
        nb_sharpes = [float(result.sharpes[j]) for j in nb_idxs if np.isfinite(result.sharpes[j])]
        if len(nb_sharpes) < min_neighbours:
            continue
        hood = [float(sc), *nb_sharpes]
        mean_h = float(np.mean(hood))
        if mean_h <= 0:
            continue
        spread = (max(hood) - min(hood)) / abs(mean_h)
        if spread > spread_pct_max:
            continue
        nb_cells = [result.cells[j] for j in nb_idxs if np.isfinite(result.sharpes[j])]
        candidates.append(
            PlateauCandidate(
                center=cell,
                neighbour_cells=nb_cells,
                center_sharpe=float(sc),
                neighbour_sharpes=nb_sharpes,
                mean_neighbourhood_sharpe=mean_h,
                spread_pct=float(spread),
                rank=0,  # filled below
            )
        )

    candidates.sort(key=lambda c: c.mean_neighbourhood_sharpe, reverse=True)
    return [
        PlateauCandidate(
            center=c.center,
            neighbour_cells=c.neighbour_cells,
            center_sharpe=c.center_sharpe,
            neighbour_sharpes=c.neighbour_sharpes,
            mean_neighbourhood_sharpe=c.mean_neighbourhood_sharpe,
            spread_pct=c.spread_pct,
            rank=k + 1,
        )
        for k, c in enumerate(candidates[:top_k])
    ]


def format_plateau_report(
    result: SweepResult,
    candidates: list[PlateauCandidate],
    *,
    audit_label: str = "EXPLORATORY SWEEP",
) -> str:
    """Markdown-friendly report — paste into the pre-reg directive draft.

    The report header always carries the EXPLORATORY / NOT FOR DEPLOYMENT
    banner so the document can't accidentally be cited as audit evidence.
    """
    lines = [
        f"# {audit_label} — Plateau detection",
        "",
        "**EXPLORATORY ONLY. Outputs below are a PRIOR for the next pre-reg directive, NOT a deployment gate.**",
        "",
        f"- Bars: {result.meta.get('n_bars')}  "
        f"({result.meta.get('first_bar')} → {result.meta.get('last_bar')})",
        f"- Cells evaluated: {result.n_cells}",
        f"- Params swept: {', '.join(result.param_names)}",
        f"- periods_per_year: {result.meta.get('periods_per_year')}",
        "",
    ]
    if not candidates:
        lines.append("**No plateau candidates** passed the spread + positivity gate.")
        lines.append("")
        lines.append(
            "Action: widen the grid, relax `spread_pct_max`, or reconsider the strategy hypothesis."
        )
        return "\n".join(lines)
    lines.append("## Top plateau candidates")
    lines.append("")
    for c in candidates:
        center_str = ", ".join(f"{k}={v}" for k, v in c.center.items())
        lines.append(f"### #{c.rank} — center: {center_str}")
        lines.append("")
        lines.append(f"- center Sharpe: **{c.center_sharpe:.4f}**")
        lines.append(
            f"- neighbourhood mean Sharpe: **{c.mean_neighbourhood_sharpe:.4f}**  "
            f"(n_neighbours={len(c.neighbour_sharpes)})"
        )
        lines.append(f"- spread (max-min / |mean|): **{c.spread_pct * 100:.2f}%**  (L27 gate: 30%)")
        lines.append("- neighbours:")
        for nb_cell, nb_sr in zip(c.neighbour_cells, c.neighbour_sharpes, strict=True):
            nb_str = ", ".join(f"{k}={v}" for k, v in nb_cell.items())
            lines.append(f"  - {nb_str} → Sharpe {nb_sr:.4f}")
        lines.append("")
    lines.append("## Next step (V3.6 hybrid workflow)")
    lines.append("")
    lines.append(
        "Promote the #1 center as the candidate canonical in a fresh pre-reg "
        "directive. Pin 4 of the listed neighbours as the plateau cells. "
        "Then run the full V3.6 audit harness on OOS data and apply the "
        "5-axis decision matrix. The audit, not this sweep, is the gate."
    )
    return "\n".join(lines)
