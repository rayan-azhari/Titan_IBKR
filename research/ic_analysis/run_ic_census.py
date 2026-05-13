"""IC Signal Census — multi-instrument multi-timeframe multi-horizon scan.

CLI orchestrator implementing the gates pre-registered in
``directives/IC Signal Census 2026-05-13.md``. Reads the universe from
``config/ic_census_universe.toml``. Reuses ``run_ic`` for the Spearman IC
math; reuses ``ic_census_lib`` for the new primitives (causality test,
anchored aggregation, DSR, plateau, MTF agreement, sanctuary slicing).

Output: ``.tmp/reports/ic_census/ic_census_{stamp}.parquet`` with the §6
schema, plus a ranked console leaderboard.

Usage::

    # Full Phase A census
    python research/ic_analysis/run_ic_census.py

    # Methodology dry-run on a single signal × instrument × TF
    python research/ic_analysis/run_ic_census.py \\
        --instruments AUD_JPY --timeframes H1 --signals vwap_overshoot

    # All signals on AUD_JPY across all TFs (MTF-agreement check)
    python research/ic_analysis/run_ic_census.py --instruments AUD_JPY

Look-ahead invariants:

* Every cell drops the trailing 12-month sanctuary window BEFORE computing
  any statistic (gate §3.6).
* Forward returns use ``close.shift(-h)`` -- intentional target, only via
  ``run_ic.compute_forward_returns``.
* Cross-TF aggregations route through ``anchored_aggregate`` and are
  wrapped by ``assert_causal``. ``ffill`` of lower-TF onto higher-TF is
  banned; see ``directives/IC Signal Census 2026-05-13.md`` §4.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.ic_analysis.ic_census_lib import (  # noqa: E402
    CellResult,
    anchored_aggregate,
    assert_causal,
    deflated_t_pvalue,
    fold_ic_signs,
    mtf_agreement,
    plateau_stable,
    signal_factories,
    slice_sanctuary,
)
from research.ic_analysis.run_ic import (  # noqa: E402
    compute_forward_returns,
    load_ohlcv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Universe loading ────────────────────────────────────────────────────────


def load_universe(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


# ── IC + NW t-stat for the full sample ──────────────────────────────────────


def ic_with_nw_tstat(
    signal: pd.Series,
    fwd_return: pd.Series,
    horizon: int,
) -> tuple[float, float, int]:
    """Spearman IC plus Newey-West t-stat of the rank-rank regression.

    Returns ``(ic_spearman, t_stat_nw, n_obs)``. NaN pairs are dropped
    first. ``maxlags = horizon - 1`` to absorb the overlap autocorrelation
    induced by ``h``-bar forward returns. Falls back to ``(NaN, NaN, 0)``
    when fewer than 30 aligned observations.
    """
    df = pd.concat([signal, fwd_return], axis=1).dropna()
    if len(df) < 30:
        return float("nan"), float("nan"), len(df)
    s_rank = df.iloc[:, 0].rank().to_numpy()
    f_rank = df.iloc[:, 1].rank().to_numpy()
    # Standardise ranks to centre Pearson at 0 (Spearman ≡ Pearson on ranks).
    rho, _ = stats.spearmanr(s_rank, f_rank)
    # OLS rank ~ rank with HAC SE.
    X = sm.add_constant(s_rank)
    try:
        ols = sm.OLS(f_rank, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": max(horizon - 1, 1)}
        )
        t_stat = float(ols.tvalues[1])
    except Exception as exc:  # pragma: no cover - statsmodels rarely fails
        logger.warning(f"HAC OLS failed (h={horizon}, n={len(df)}): {exc}")
        t_stat = float("nan")
    return round(float(rho), 5), round(t_stat, 4), len(df)


# ── Per-cell evaluation ─────────────────────────────────────────────────────


def _anchor_external_close(
    ext_ohlcv: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    sanctuary_start: pd.Timestamp,
    *,
    label: str,
    causal_trials: int = 3,
) -> pd.Series:
    """Apply the Anchored MTF Rule to an external close series.

    1. Drop sanctuary bars from the external (keep math symmetric with target).
    2. ``anchored_aggregate(higher_tf=False)`` -- ``.shift(1)`` + reindex+ffill
       onto ``target_index``. At target time T, the aligned series sees only
       external values at times strictly less than T.
    3. ``assert_causal`` smoke test on the alignment function (3 random
       corruption trials). Raises AssertionError if the wrapping leaks.
    """
    ext_close = ext_ohlcv["close"][ext_ohlcv.index < sanctuary_start]
    if len(ext_close) < 50:
        raise ValueError(f"{label}: only {len(ext_close)} pre-sanctuary bars")

    def _agg(s: pd.Series, di: pd.DatetimeIndex) -> pd.Series:
        return anchored_aggregate(s, di, higher_tf=False)

    # Causality check on the alignment function -- mandatory wrap per §4.1 of
    # directives/IC Signal Census 2026-05-13.md. The test must run over the
    # INDEX OVERLAP between the external src and the target dst; otherwise
    # the corruption point can land before dst_index starts and there is no
    # past baseline to compare against (degenerate-test trap, not a real
    # causality violation). Bound both src and dst to their overlap before
    # invoking assert_causal.
    overlap_start = max(ext_close.index[0], target_index[0])
    overlap_end = min(ext_close.index[-1], target_index[-1])
    src_test = ext_close[(ext_close.index >= overlap_start) & (ext_close.index <= overlap_end)]
    dst_test_full = target_index[
        (target_index >= overlap_start) & (target_index <= overlap_end)
    ]
    test_dst = dst_test_full[: min(len(dst_test_full), 300)]
    if len(src_test) >= 50 and len(test_dst) >= 10:
        assert_causal(_agg, src_test, test_dst, n_trials=causal_trials)
    return _agg(ext_close, target_index)


def evaluate_cell(
    *,
    signal_name: str,
    signal_fn,
    needs: list[str],
    externals: list[str],
    params: dict[str, Any],
    ohlcv: pd.DataFrame,
    instrument: str,
    timeframe: str,
    horizons: list[int],
    sanctuary_months: int,
    n_folds: int,
    fold_quorum: int,
    ohlcv_cache: dict[tuple[str, str], pd.DataFrame],
) -> list[dict[str, Any]]:
    """Evaluate one (signal, params) cell across all horizons. Returns one
    dict per horizon ready to enter the output parquet.

    Cross-asset signals: when ``externals`` is non-empty, the function
    loads each external instrument at the SAME timeframe as the target,
    applies ``_anchor_external_close`` (Anchored MTF Rule + assert_causal),
    and passes each as a kwarg named with the lowercase ticker.

    Self-correlation guard: if any external == ``instrument``, the cell
    is skipped (signal would be trivially auto-correlated).
    """
    # Self-correlation guard for cross-asset signals.
    if externals and instrument in externals:
        logger.debug(f"  Skip {signal_name} on {instrument}: external == target")
        return []

    visible, sanc_start, sanc_end = slice_sanctuary(ohlcv, months=sanctuary_months)
    if len(visible) < 200:
        logger.warning(
            f"  Skip {signal_name} {params}: only {len(visible)} bars after sanctuary"
        )
        return []
    # Build kwargs for the signal function from the OHLCV columns it needs.
    inputs = {col: visible[col] for col in needs}

    # Load + anchor externals (cross-asset signals). Each goes through
    # anchored_aggregate(higher_tf=False) + assert_causal smoke test.
    ext_kwargs: dict[str, pd.Series] = {}
    for ext_inst in externals:
        ext_key = (ext_inst, timeframe)
        if ext_key not in ohlcv_cache:
            try:
                from research.ic_analysis.run_ic import load_ohlcv
                ohlcv_cache[ext_key] = load_ohlcv(ext_inst, timeframe)
            except FileNotFoundError:
                logger.warning(
                    f"  Skip {signal_name} on {instrument}: external "
                    f"{ext_inst}_{timeframe} parquet missing"
                )
                return []
        ext_ohlcv = ohlcv_cache[ext_key]
        if ext_ohlcv.empty:
            return []
        try:
            aligned = _anchor_external_close(
                ext_ohlcv, visible.index, sanc_start, label=f"{ext_inst}@{timeframe}"
            )
        except (ValueError, AssertionError) as exc:
            logger.error(
                f"  Skip {signal_name} on {instrument}: external "
                f"{ext_inst} anchoring failed: {exc}"
            )
            return []
        ext_kwargs[ext_inst.lower()] = aligned

    # Pass parameters as kwargs; the close column is the first positional.
    try:
        if externals:
            signal = signal_fn(inputs["close"], **ext_kwargs, **params)
        elif "close" in needs and len(needs) == 1:
            signal = signal_fn(inputs["close"], **params)
        else:
            # Functions like intraday_range_atr take (close, period, high, low).
            signal = signal_fn(inputs["close"], **params, **{k: v for k, v in inputs.items() if k != "close"})
    except TypeError as exc:
        logger.error(f"  Signal {signal_name} {params}: {exc}")
        return []

    fwd = compute_forward_returns(visible["close"], horizons, vol_adjust=True)
    rows: list[dict[str, Any]] = []
    for h in horizons:
        fwd_col = f"fwd_{h}"
        ic, t_nw, n_obs = ic_with_nw_tstat(signal, fwd[fwd_col], horizon=h)
        fold_ic, sign_stable, modal = fold_ic_signs(
            signal, fwd[fwd_col], n_folds=n_folds, quorum=fold_quorum
        )
        rows.append({
            "signal": signal_name,
            "params": json.dumps(params, sort_keys=True),
            "horizon": h,
            "n_bars": int(n_obs),
            "ic_spearman": ic,
            "t_stat_nw": t_nw,
            "raw_p_value": _ic_two_tailed_p(t_nw, n_obs),
            "fold_ic": fold_ic,
            "fold_stable": bool(sign_stable),
            "fold_quorum": int(modal),
            "sanctuary_start": sanc_start,
            "sanctuary_end": sanc_end,
        })
    return rows


def _ic_two_tailed_p(t_stat: float, n: int) -> float:
    if not np.isfinite(t_stat) or n < 30:
        return float("nan")
    df = max(n - 2, 1)
    return float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df)))


# ── Plateau + DSR + MTF + BH gates (per signal-class triple) ───────────────


def apply_plateau_and_dsr(
    rows: list[dict[str, Any]],
    *,
    headline_t_floor: float,
    neighbour_t_floor: float,
    ic_range_max: float,
    cell_order: list[str],
    n_trials: int,
) -> list[dict[str, Any]]:
    """Group rows by (signal, instrument, TF, horizon) -- one group has 3
    cells (the 3 grid points). Apply plateau stability; attach DSR p-value
    based on ``n_trials`` (the total cell count across the whole sweep).

    ``cell_order`` is the pre-registered parameter-cell order from the
    universe TOML -- we MUST honour it for the neighbour check (V3.1: the
    grid is committed before the scan).
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    keys = ["signal", "instrument", "timeframe", "horizon"]
    out: list[dict[str, Any]] = []
    for key, group in df.groupby(keys, sort=False):
        # Order group cells by the pre-registered grid order, NOT by |t|.
        group = group.set_index("params").reindex(cell_order).reset_index()
        cells = [
            CellResult(
                params=json.loads(p),
                ic=float(r["ic_spearman"]) if pd.notna(r["ic_spearman"]) else float("nan"),
                t_stat=float(r["t_stat_nw"]) if pd.notna(r["t_stat_nw"]) else float("nan"),
                n_obs=int(r["n_bars"]) if pd.notna(r["n_bars"]) else 0,
            )
            for p, r in zip(group["params"], group.to_dict("records"), strict=True)
        ]
        passes, headline, reason = plateau_stable(
            cells,
            headline_t_floor=headline_t_floor,
            neighbour_t_floor=neighbour_t_floor,
            ic_range_max=ic_range_max,
        )
        if headline is None:
            continue
        dsr_p = deflated_t_pvalue(headline.t_stat, n_trials)
        # Find the row corresponding to the headline cell.
        headline_params_json = json.dumps(headline.params, sort_keys=True)
        match = group[group["params"] == headline_params_json].iloc[0].to_dict()
        match.update({
            "instrument": key[1],
            "timeframe": key[2],
            "plateau_stable": bool(passes),
            "plateau_reason": reason,
            "dsr_pvalue": round(dsr_p, 4),
            "dsr_pass": bool(dsr_p >= 0.95) and bool(abs(headline.t_stat) >= headline_t_floor),
        })
        out.append(match)
    return out


def apply_mtf_agreement(
    headline_rows: list[dict[str, Any]],
    *,
    quorum: int,
    t_floor: float,
) -> list[dict[str, Any]]:
    """For each (signal, instrument, horizon-bucket) compute MTF agreement
    across timeframes. Horizon bucketing is by ordinal index into the
    per-TF horizon list (h_idx 0/1/2 = short/mid/long), since the absolute
    bar counts differ per TF (D 1/5/21 vs H1 1/8/40)."""
    df = pd.DataFrame(headline_rows)
    if df.empty:
        return headline_rows
    # Compute horizon ordinal index within each (instrument, TF).
    df["h_idx"] = df.groupby(["instrument", "timeframe"])["horizon"].rank(method="dense").astype(int) - 1
    for (signal, instrument, h_idx), group in df.groupby(["signal", "instrument", "h_idx"]):
        per_tf = {
            r["timeframe"]: CellResult(
                params=json.loads(r["params"]),
                ic=r["ic_spearman"],
                t_stat=r["t_stat_nw"],
                n_obs=r["n_bars"],
            )
            for _, r in group.iterrows()
        }
        agree, n_pass = mtf_agreement(per_tf, quorum=quorum, t_floor=t_floor)
        df.loc[group.index, "mtf_agree"] = bool(agree)
        df.loc[group.index, "mtf_n_pass"] = int(n_pass)
    return df.to_dict("records")


def apply_bh_fdr(rows: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    """Benjamini-Hochberg FDR across the full pool of t-stats."""
    if not rows:
        return rows
    from statsmodels.stats.multitest import multipletests

    pvals = np.array([r.get("raw_p_value", float("nan")) for r in rows])
    nan_mask = np.isnan(pvals)
    clean = np.where(nan_mask, 1.0, pvals)
    reject, adj, _, _ = multipletests(clean, alpha=alpha, method="fdr_bh")
    for i, r in enumerate(rows):
        r["bh_pvalue_adj"] = float("nan") if nan_mask[i] else round(float(adj[i]), 6)
        r["bh_significant"] = (not nan_mask[i]) and bool(reject[i])
    return rows


def assign_tier(r: dict[str, Any]) -> str:
    if not r.get("dsr_pass") or not r.get("bh_significant") or not r.get("fold_stable"):
        return "unconfirmed"
    if not r.get("plateau_stable"):
        return "unconfirmed"
    if r.get("mtf_agree"):
        return "TIER_A"
    return "TIER_B"


# ── Cell enumeration from universe.toml ────────────────────────────────────


def enumerate_cells(
    universe: dict[str, Any],
    instruments_filter: set[str] | None,
    timeframes_filter: set[str] | None,
    signals_filter: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Walk the universe and emit (instrument, tf, signal, params) cells.

    Returns ``(cells, signal_cell_order)`` where ``signal_cell_order`` maps
    each signal name to the canonical ordered list of param JSONs (used
    by the plateau gate to know which cells are "neighbours").
    """
    tf_horizons = {tf: spec["horizons"] for tf, spec in universe["timeframes"].items()}
    cells: list[dict[str, Any]] = []
    cell_order: dict[str, list[str]] = {}

    phase_a_instruments = universe.get("phase_a", {}).get("instruments", {})
    leveraged = universe.get("phase_a", {}).get("leveraged_etfs", {})

    sig_registry = signal_factories()

    # Iterate signal-class branches in the universe TOML.
    signals_section = universe.get("signals", {})
    for cls_name, cls_branch in signals_section.items():
        if not isinstance(cls_branch, dict):
            continue
        for sig_name, sig_spec in cls_branch.items():
            if not isinstance(sig_spec, dict) or "cells" not in sig_spec:
                continue
            if signals_filter and sig_name not in signals_filter:
                continue
            if sig_name not in sig_registry:
                logger.warning(f"Signal {sig_name} declared in TOML but not in registry; skipping")
                continue
            # Capture canonical cell order (pre-registered) for the plateau gate.
            cell_order[sig_name] = [
                json.dumps(c, sort_keys=True) for c in sig_spec["cells"]
            ]
            applies_to_tfs = set(sig_spec.get("applies_to_tfs", tf_horizons.keys()))
            for instrument, tfs in phase_a_instruments.items():
                if instruments_filter and instrument not in instruments_filter:
                    continue
                for tf in tfs:
                    if timeframes_filter and tf not in timeframes_filter:
                        continue
                    if tf not in applies_to_tfs:
                        continue
                    for params in sig_spec["cells"]:
                        cells.append({
                            "signal": sig_name,
                            "signal_class": cls_name,
                            "instrument": instrument,
                            "timeframe": tf,
                            "params": params,
                            "horizons": tf_horizons[tf],
                            "needs": sig_registry[sig_name]["needs"],
                            "externals": sig_registry[sig_name].get("externals", []),
                            "leveraged_etf": instrument in leveraged,
                        })
    return cells, cell_order


# ── Main ───────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> Path:
    universe = load_universe(Path(args.universe))
    meta = universe["meta"]

    instruments_filter = set(args.instruments.split(",")) if args.instruments else None
    timeframes_filter = set(args.timeframes.split(",")) if args.timeframes else None
    signals_filter = set(args.signals.split(",")) if args.signals else None

    cells, cell_order = enumerate_cells(
        universe, instruments_filter, timeframes_filter, signals_filter
    )
    n_cells_meta = sum(len(c["horizons"]) for c in cells)
    logger.info(
        f"Cell enumeration: {len(cells)} (signal, instrument, TF, params) cells "
        f"→ {n_cells_meta} (cell × horizon) rows"
    )
    logger.info(
        f"Pre-registered gates: |t_NW| ≥ {meta['dsr_t_floor_phase_a']}, "
        f"neighbour |t| ≥ {meta['plateau_neighbour_t_floor']}, "
        f"|IC| range < {meta['plateau_ic_range_max']:.0%}, "
        f"MTF quorum {meta['mtf_quorum']}/{len(universe['timeframes'])}, "
        f"fold quorum {meta['fold_sign_quorum']}/{meta['fold_count']}, "
        f"sanctuary {meta['sanctuary_months']}mo"
    )
    logger.info(
        f"DSR null-expected max |t| at N={n_cells_meta}: "
        f"≈ {math.sqrt(2 * math.log(max(n_cells_meta, 2))):.2f} "
        f"(hard floor {meta['dsr_t_floor_phase_a']})"
    )

    sig_registry = signal_factories()
    all_rows: list[dict[str, Any]] = []
    # Cache loaded parquets per (instrument, tf) to avoid re-reading.
    cache: dict[tuple[str, str], pd.DataFrame] = {}

    for cell in cells:
        key = (cell["instrument"], cell["timeframe"])
        if key not in cache:
            try:
                cache[key] = load_ohlcv(cell["instrument"], cell["timeframe"])
            except FileNotFoundError:
                logger.warning(f"  Missing data: {cell['instrument']}_{cell['timeframe']} -- skipping")
                cache[key] = pd.DataFrame()
        ohlcv = cache[key]
        if ohlcv.empty:
            continue

        rows = evaluate_cell(
            signal_name=cell["signal"],
            signal_fn=sig_registry[cell["signal"]]["fn"],
            needs=cell["needs"],
            externals=cell["externals"],
            params=cell["params"],
            ohlcv=ohlcv,
            instrument=cell["instrument"],
            timeframe=cell["timeframe"],
            horizons=cell["horizons"],
            sanctuary_months=meta["sanctuary_months"],
            n_folds=meta["fold_count"],
            fold_quorum=meta["fold_sign_quorum"],
            ohlcv_cache=cache,
        )
        for r in rows:
            r["instrument"] = cell["instrument"]
            r["timeframe"] = cell["timeframe"]
            r["signal_class"] = cell["signal_class"]
            r["leveraged_etf"] = cell["leveraged_etf"]
            r["phase"] = "A"
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No rows produced -- check filters and data availability.")
        return Path()

    logger.info(f"Produced {len(all_rows)} raw rows; applying gates...")

    # BH-FDR across the full pool of raw p-values.
    all_rows = apply_bh_fdr(all_rows, alpha=meta["bh_alpha"])

    # Plateau + DSR per (signal, instrument, TF, horizon).
    headline_rows: list[dict[str, Any]] = []
    n_trials = len(all_rows)
    for sig_name, sig_cell_order in cell_order.items():
        sig_rows = [r for r in all_rows if r["signal"] == sig_name]
        headline_rows.extend(apply_plateau_and_dsr(
            sig_rows,
            headline_t_floor=meta["dsr_t_floor_phase_a"],
            neighbour_t_floor=meta["plateau_neighbour_t_floor"],
            ic_range_max=meta["plateau_ic_range_max"],
            cell_order=sig_cell_order,
            n_trials=n_trials,
        ))

    # MTF agreement.
    headline_rows = apply_mtf_agreement(
        headline_rows, quorum=meta["mtf_quorum"], t_floor=meta["dsr_t_floor_phase_a"]
    )

    # Tier assignment.
    for r in headline_rows:
        r["tier"] = assign_tier(r)

    # Write parquet.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / ".tmp" / "reports" / "ic_census"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ic_census_{stamp}.parquet"
    raw_path = out_dir / f"ic_census_{stamp}_raw.parquet"

    df_headline = pd.DataFrame(headline_rows)
    df_raw = pd.DataFrame(all_rows)
    # Serialise list-valued fold_ic column for parquet compatibility.
    if "fold_ic" in df_headline.columns:
        df_headline["fold_ic"] = df_headline["fold_ic"].apply(json.dumps)
    if "fold_ic" in df_raw.columns:
        df_raw["fold_ic"] = df_raw["fold_ic"].apply(json.dumps)
    df_headline.to_parquet(out_path, index=False)
    df_raw.to_parquet(raw_path, index=False)
    logger.info(f"Wrote headline:  {out_path}  ({len(df_headline)} rows)")
    logger.info(f"Wrote raw:       {raw_path}   ({len(df_raw)} rows)")

    # Console leaderboard.
    _print_leaderboard(df_headline)
    return out_path


def _print_leaderboard(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df["abs_t"] = df["t_stat_nw"].abs()
    df = df.sort_values("abs_t", ascending=False)
    cols = [
        "tier", "signal", "instrument", "timeframe", "horizon",
        "ic_spearman", "t_stat_nw", "dsr_pvalue", "bh_significant",
        "plateau_stable", "fold_stable", "mtf_agree", "n_bars",
    ]
    print()
    print("=" * 100)
    print("  IC CENSUS LEADERBOARD (top by |t_NW|)")
    print("=" * 100)
    with pd.option_context("display.max_rows", 50, "display.width", 200):
        print(df[cols].head(30).to_string(index=False))
    print()
    counts = df["tier"].value_counts().to_dict()
    print(f"  Tier counts: {counts}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="IC Signal Census (pre-registered)")
    parser.add_argument(
        "--universe", default=str(ROOT / "config" / "ic_census_universe.toml"),
        help="Path to ic_census_universe.toml",
    )
    parser.add_argument("--instruments", default="", help="CSV of instruments (default: all)")
    parser.add_argument("--timeframes", default="", help="CSV of timeframes (default: all)")
    parser.add_argument("--signals", default="", help="CSV of signal names (default: all)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
