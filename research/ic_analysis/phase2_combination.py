"""Phase 2 -- Signal Combination IC/ICIR Analysis.

Tests whether composite signals built from the 52-signal library produce higher
IC/ICIR than any individual signal at a given horizon. Six methods are tested:

    1a. Correlation clustering  -- identify redundant signal pairs (diagnostic)
    1b. Partial IC              -- incremental alpha beyond the dominant signal
    2.  Equal-weight (top-N)    -- average top-N signals by |IC|, sign-normalised
    3.  ICIR-weighted (top-N)   -- same, weighted by |ICIR_i| / sum(|ICIR|)
    4.  Group-diversified       -- best signal per group (A-G), equal-weighted
    5.  PCA (PC1)               -- first principal component of USABLE signals
    6.  AND-gating              -- signal_A * sign(signal_B) for complementary pairs

Usage:
    uv run python research/ic_analysis/phase2_combination.py
    uv run python research/ic_analysis/phase2_combination.py \\
        --instrument EUR_USD --timeframe H4 --horizon 20

Sign normalisation:
    Before any averaging, each signal is multiplied by sign(IC_i) so that all
    components point in the same direction. Mean-reversion signals (negative IC)
    are negated. The composite IC is then reported with its natural sign.

Look-ahead safety:
    All signal computations are causal (inherited from phase1_sweep.py).
    sign(IC) orientation uses in-sample statistics as a label, not a feature.
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import research.ic_analysis.phase1_sweep as _sweep  # noqa: E402
from research.ic_analysis.phase1_sweep import (  # noqa: E402
    _get_annual_bars,
    _load_ohlcv,
    build_all_signals,
)
from research.ic_analysis.run_ic import (  # noqa: E402
    compute_forward_returns,
    compute_ic_table,
    compute_icir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

IC_USABLE_THRESHOLD = 0.03
CLUSTER_CORR_THRESHOLD = 0.70
DEFAULT_TOP_N = [3, 5, 10]
ICIR_WINDOW = 1638  # default H1 annual bars; overridden dynamically in run_combination


# ── Verdict ───────────────────────────────────────────────────────────────


def _verdict_local(ic: float, icir: float) -> str:
    abs_ic = abs(ic) if not np.isnan(ic) else 0.0
    abs_ir = abs(icir) if not np.isnan(icir) else 0.0
    if abs_ic >= 0.05 and abs_ir >= 0.5:
        return "STRONG"
    elif abs_ic >= 0.05:
        return "USABLE"
    elif abs_ic >= 0.03:
        return "WEAK"
    return "NOISE"


# ── Core scoring helper ───────────────────────────────────────────────────


def _score_composite(
    composite: pd.Series,
    fwd_returns: pd.DataFrame,
    horizon: int,
    window: int = ICIR_WINDOW,
) -> tuple[float, float, str]:
    """Compute IC and ICIR for a composite signal at a single horizon."""
    fwd_col = f"fwd_{horizon}"
    if fwd_col not in fwd_returns.columns:
        return np.nan, np.nan, "NOISE"

    name = composite.name or "composite"
    sig_df = composite.to_frame(name=name)
    ic_df = compute_ic_table(sig_df, fwd_returns[[fwd_col]])
    ic = float(ic_df.iloc[0, 0]) if not ic_df.empty else np.nan

    icir_s = compute_icir(
        sig_df,
        fwd_returns[[fwd_col]],
        horizons=[horizon],
        window=window,
    )
    icir = float(icir_s.iloc[0]) if not icir_s.empty else np.nan

    return ic, icir, _verdict_local(ic, icir)


# ── Signal selection ──────────────────────────────────────────────────────


def _select_usable(
    ic_df: pd.DataFrame,
    all_signals: pd.DataFrame,
    horizon: int,
    threshold: float = IC_USABLE_THRESHOLD,
) -> pd.DataFrame:
    """Return signal columns where |IC| >= threshold at the given horizon."""
    fwd_col = f"fwd_{horizon}"
    if fwd_col not in ic_df.columns:
        logger.warning("Horizon fwd_%d not in IC table", horizon)
        return all_signals.iloc[:, :5]

    mask = ic_df[fwd_col].abs() >= threshold
    selected = ic_df.index[mask].tolist()
    if not selected:
        logger.warning(
            "No signals pass |IC| >= %.3f at h=%d -- using top 5 by |IC|",
            threshold,
            horizon,
        )
        selected = ic_df[fwd_col].abs().nlargest(5).index.tolist()

    valid = [s for s in selected if s in all_signals.columns]
    return all_signals[valid]


def _sign_orient(
    signals: pd.DataFrame,
    ic_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Multiply each signal by sign(IC) so all point in the same direction.

    M6 FIX: When IC is NaN (insufficient data or computation failure), the
    sign defaults to +1 (arbitrary positive orientation). Log a warning so the
    user knows which signals carry no measurable IC and are included by default.
    Ideally these signals should already be below the USABLE threshold and
    excluded before combination, but we warn defensively.
    """
    fwd_col = f"fwd_{horizon}"
    ic_at_horizon = ic_df.loc[signals.columns, fwd_col]
    nan_signals = ic_at_horizon[ic_at_horizon.isna()].index.tolist()
    if nan_signals:
        logger.warning(
            "sign_orient: IC is NaN for %d signal(s) at horizon %d. "
            "Dropping these signals from the composite to prevent orientation to noise: %s",
            len(nan_signals),
            horizon,
            nan_signals,
        )
        signals = signals.drop(columns=nan_signals)
        ic_at_horizon = ic_at_horizon.dropna()
        
    signs = np.sign(ic_at_horizon)
    return signals.multiply(signs, axis=1)


# ── Method 1a: Correlation clustering ────────────────────────────────────


def method_correlation_analysis(
    all_signals: pd.DataFrame,
    threshold: float = CLUSTER_CORR_THRESHOLD,
) -> dict:
    """Compute pairwise Pearson correlation and cluster redundant signals.

    Uses union-find to group connected components (avoids transitive
    false-positives from naive pairwise thresholding).

    Returns dict with keys: corr_matrix, clusters, independent_signals.
    """
    clean = all_signals.dropna()
    if clean.empty:
        return {"corr_matrix": pd.DataFrame(), "clusters": [], "independent_signals": []}

    corr = clean.corr(method="spearman")
    signals = list(corr.index)

    # Union-Find
    parent = {s: s for s in signals}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, a in enumerate(signals):
        for b in signals[i + 1 :]:
            if abs(corr.loc[a, b]) > threshold:
                union(a, b)

    groups: dict[str, list[str]] = defaultdict(list)
    for s in signals:
        groups[find(s)].append(s)

    clusters = sorted(
        [g for g in groups.values() if len(g) > 1],
        key=len,
        reverse=True,
    )
    independent = sorted([g[0] for g in groups.values() if len(g) == 1])

    return {
        "corr_matrix": corr,
        "clusters": clusters,
        "independent_signals": independent,
    }


# ── Method 1b: Partial IC ────────────────────────────────────────────────


def method_partial_ic(
    all_signals: pd.DataFrame,
    ic_df: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """For each USABLE signal, compute IC on residuals after regressing out dominant signals.

    The dominant signal is the one with highest |IC| at this horizon.
    Partial IC > 0.02 means the signal adds incremental alpha beyond the dominant signal.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression

    fwd_col = f"fwd_{horizon}"
    dominant = ic_df[fwd_col].abs().idxmax()  # highest |IC| signal

    rows = []
    for sig in all_signals.columns:
        if sig == dominant:
            rows.append(
                {
                    "signal": sig,
                    "partial_ic": ic_df.loc[sig, fwd_col],
                    "is_dominant": True,
                }
            )
            continue

        both = pd.concat([all_signals[[dominant, sig]], fwd_returns[[fwd_col]]], axis=1).dropna()
        if len(both) < 50:
            rows.append({"signal": sig, "partial_ic": np.nan, "is_dominant": False})
            continue

        # Regress out the dominant signal from this signal.
        # L3 FIX: Check condition number before regression. When two signals are
        # nearly perfectly collinear, residuals are near-zero → Spearman returns
        # NaN, which is indistinguishable from a data-quality NaN. Log the root
        # cause explicitly ("collinear" vs "insufficient data") so downstream
        # consumers can distinguish the two failure modes.
        X = both[[dominant]].values
        y = both[sig].values
        cond_num = float(np.linalg.cond(X))
        if cond_num > 1e8:
            logger.warning(
                "partial_ic: near-perfect collinearity between '%s' and dominant "
                "signal '%s' (condition number=%.2e). Partial IC will be assigned NaN.",
                sig,
                dominant,
                cond_num,
            )
        lr = LinearRegression().fit(X, y)
        residual = y - lr.predict(X)

        rho, _ = spearmanr(residual, both[fwd_col].values)
        rows.append(
            {
                "signal": sig,
                "partial_ic": float(rho) if not np.isnan(rho) else np.nan,
                "is_dominant": False,
                "cond_num": round(cond_num, 2),
            }
        )

    return pd.DataFrame(rows)


# ── Method 2: Equal-weight composite ─────────────────────────────────────


def method_equal_weight(
    usable: pd.DataFrame,
    ic_df: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizon: int,
    top_n: int | None = None,
    window: int = ICIR_WINDOW,
) -> dict:
    """Average top-N USABLE signals (by |IC|), sign-normalised."""
    fwd_col = f"fwd_{horizon}"
    ranked = (
        ic_df.loc[
            [c for c in usable.columns if c in ic_df.index],
            fwd_col,
        ]
        .abs()
        .nlargest(top_n or len(usable.columns))
    )

    selected = usable[[c for c in ranked.index if c in usable.columns]]
    oriented = _sign_orient(selected, ic_df, horizon)
    composite = oriented.mean(axis=1).rename(f"ew_top{top_n or 'all'}")

    ic, icir, verdict = _score_composite(composite, fwd_returns, horizon, window=window)
    return {
        "name": f"Equal-weight (top {top_n or len(selected)})",
        "composite": composite,
        "ic": ic,
        "icir": icir,
        "verdict": verdict,
        "n_signals": len(selected.columns),
        "signals_used": list(selected.columns),
    }


# ── Method 3: ICIR-weighted composite ───────────────────────────────────


def method_icir_weighted(
    usable: pd.DataFrame,
    ic_df: pd.DataFrame,
    icir_s: pd.Series,
    fwd_returns: pd.DataFrame,
    horizon: int,
    top_n: int | None = None,
    window: int = ICIR_WINDOW,
) -> dict:
    """Weight each signal by |ICIR_i| / sum(|ICIR|), sign-normalised first."""
    fwd_col = f"fwd_{horizon}"
    ranked = (
        ic_df.loc[
            [c for c in usable.columns if c in ic_df.index],
            fwd_col,
        ]
        .abs()
        .nlargest(top_n or len(usable.columns))
    )

    selected_names = [c for c in ranked.index if c in usable.columns]
    selected = usable[selected_names]
    oriented = _sign_orient(selected, ic_df, horizon)

    # ICIR weights
    icir_vals = icir_s.reindex(selected_names).abs().fillna(0.0)
    total = icir_vals.sum()
    if total == 0.0:
        weights = pd.Series(1.0 / len(selected_names), index=selected_names)
    else:
        weights = icir_vals / total

    composite = oriented.multiply(weights, axis=1).sum(axis=1)
    composite.name = f"icir_w_top{top_n or len(selected_names)}"

    ic, icir, verdict = _score_composite(composite, fwd_returns, horizon, window=window)
    return {
        "name": f"ICIR-weighted (top {top_n or len(selected_names)})",
        "composite": composite,
        "ic": ic,
        "icir": icir,
        "verdict": verdict,
        "n_signals": len(selected_names),
        "signals_used": selected_names,
    }


# ── Method 4: Group-diversified composite ───────────────────────────────


def method_group_diversified(
    all_signals: pd.DataFrame,
    ic_df: pd.DataFrame,
    group_map: dict[str, str],
    fwd_returns: pd.DataFrame,
    horizon: int,
    window: int = ICIR_WINDOW,
) -> dict:
    """Take best |IC| signal from each group, average them equally (sign-normalised)."""
    fwd_col = f"fwd_{horizon}"
    unique_groups = sorted(set(group_map.values()))
    best_per_group: dict[str, str] = {}

    for grp in unique_groups:
        candidates = [
            s
            for s, g in group_map.items()
            if g == grp and s in all_signals.columns and s in ic_df.index
        ]
        if not candidates:
            continue
        best = ic_df.loc[candidates, fwd_col].abs().idxmax()
        best_per_group[grp] = best

    selected_names = list(best_per_group.values())
    selected = all_signals[[s for s in selected_names if s in all_signals.columns]]
    oriented = _sign_orient(selected, ic_df, horizon)
    composite = oriented.mean(axis=1).rename("group_diversified")

    ic, icir, verdict = _score_composite(composite, fwd_returns, horizon, window=window)
    return {
        "name": f"Group-diversified ({len(selected.columns)} groups)",
        "composite": composite,
        "ic": ic,
        "icir": icir,
        "verdict": verdict,
        "n_signals": len(selected.columns),
        "signals_used": selected_names,
    }


# ── Method 5: PCA first component ───────────────────────────────────────


def method_pca(
    usable: pd.DataFrame,
    ic_df: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizon: int,
    window: int = ICIR_WINDOW,
) -> dict:
    """First principal component of USABLE signals after StandardScaler.

    PC1 is sign-oriented so its IC at the target horizon is >= 0.
    Logs the explained variance ratio (low = signals are diverse; high = redundant).
    """
    clean = usable.dropna()
    if len(clean) < window or clean.shape[1] < 2:
        logger.warning("PCA: insufficient data (%d bars, %d signals)", len(clean), clean.shape[1])
        return {
            "name": "PCA (PC1)",
            "composite": pd.Series(dtype=float),
            "ic": np.nan,
            "icir": np.nan,
            "verdict": "NOISE",
            "n_signals": 0,
            "signals_used": [],
        }

    scaler = StandardScaler()
    X = scaler.fit_transform(clean.values)

    pca = PCA(n_components=1, random_state=42)
    pc1_values = pca.fit_transform(X).ravel()
    ev = float(pca.explained_variance_ratio_[0])
    logger.info(
        "PCA PC1 explains %.1f%% of variance across %d USABLE signals",
        ev * 100,
        clean.shape[1],
    )

    pc1 = pd.Series(pc1_values, index=clean.index, name="pca_pc1")

    # Sign-orient: negate if IC is negative
    ic_tmp, _, _ = _score_composite(pc1, fwd_returns, horizon, window=window)
    if not np.isnan(ic_tmp) and ic_tmp < 0:
        pc1 = -pc1

    ic, icir, verdict = _score_composite(pc1, fwd_returns, horizon, window=window)
    return {
        "name": f"PCA (PC1, {ev * 100:.0f}% var, {clean.shape[1]} sigs)",
        "composite": pc1,
        "ic": ic,
        "icir": icir,
        "verdict": verdict,
        "n_signals": clean.shape[1],
        "signals_used": list(usable.columns),
    }


# ── Method 6: AND-gating ─────────────────────────────────────────────────


def method_gating(
    all_signals: pd.DataFrame,
    ic_df: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizon: int,
    pairs: list[tuple[str, str]] | None = None,
    window: int = ICIR_WINDOW,
) -> list[dict]:
    """Gate signal_A by sign(signal_B): composite = oriented_A * sign(oriented_B).

    Default pairs chosen for low cross-correlation and complementary information types.
    """
    if pairs is None:
        pairs = [
            ("accel_stoch_k", "ma_spread_5_20"),  # momentum accel gated by trend
            ("bb_zscore_20", "adx_14"),  # mean-reversion gated by trend strength
            ("roc_10", "realized_vol_20"),  # momentum gated by vol regime
        ]

    fwd_col = f"fwd_{horizon}"
    results = []

    for sig_a, sig_b in pairs:
        if sig_a not in all_signals.columns or sig_b not in all_signals.columns:
            logger.warning("Gate pair (%s, %s): missing signal(s) -- skipping", sig_a, sig_b)
            continue

        # Orient A so that high values predict positive returns
        ic_a = float(ic_df.loc[sig_a, fwd_col]) if sig_a in ic_df.index else 0.0
        sign_a = np.sign(ic_a) if ic_a != 0.0 else 1.0
        oriented_a = all_signals[sig_a] * sign_a

        # Orient B so that positive values predict positive returns
        ic_b = float(ic_df.loc[sig_b, fwd_col]) if sig_b in ic_df.index else 0.0
        sign_b = np.sign(ic_b) if ic_b != 0.0 else 1.0
        oriented_b = all_signals[sig_b] * sign_b

        # Gate: active (same direction) when B is positive; attenuated when B is negative
        gate = np.sign(oriented_b).replace(0, 1.0)
        composite = (oriented_a * gate).rename(f"gate_{sig_a[:8]}_{sig_b[:8]}")

        ic, icir, verdict = _score_composite(composite, fwd_returns, horizon, window=window)
        results.append(
            {
                "name": f"Gate: {sig_a} x sign({sig_b})",
                "composite": composite,
                "ic": ic,
                "icir": icir,
                "verdict": verdict,
                "n_signals": 2,
                "signals_used": [sig_a, sig_b],
            }
        )

    return results


# ── Display ───────────────────────────────────────────────────────────────


def _print_report(
    instrument: str,
    timeframe: str,
    horizon: int,
    n_bars: int,
    n_usable: int,
    individual_top10: pd.DataFrame,
    combo_results: list[dict],
    best_ind_ic: float,
    corr_analysis: dict,
    partial_ic_df: pd.DataFrame | None = None,
) -> None:
    W = 74
    print("\n" + "=" * W)
    print(f"  SIGNAL COMBINATION ANALYSIS -- {instrument} {timeframe}  (horizon={horizon} bars)")
    print(
        f"  Bars: {n_bars:,}  |  USABLE signals: {n_usable}"
        f"  |  Threshold: |IC| >= {IC_USABLE_THRESHOLD:.3f}"
    )
    print("=" * W)

    # Individual top-10
    print()
    print("  INDIVIDUAL SIGNALS (top 10 for comparison):")
    print("  " + "-" * (W - 2))
    print(f"  {'Rank':>4}  {'Signal':<24}  {'IC':>8}  {'ICIR':>7}  Verdict")
    print("  " + "-" * (W - 2))
    for i, row in individual_top10.iterrows():
        ic_s = f"{row['ic']:>+8.4f}" if not np.isnan(row["ic"]) else "     NaN"
        ir_s = f"{row['icir']:>+7.3f}" if not np.isnan(row["icir"]) else "    NaN"
        print(f"  {i + 1:>4}  {row['signal']:<24}  {ic_s}  {ir_s}  {row['verdict']}")

    # Partial IC results (Method 1b)
    if partial_ic_df is not None and not partial_ic_df.empty:
        print()
        print("  PARTIAL IC (incremental alpha vs dominant signal):")
        print("  " + "-" * (W - 2))
        incremental = partial_ic_df[
            (~partial_ic_df["is_dominant"]) & (partial_ic_df["partial_ic"].abs() > 0.02)
        ].sort_values("partial_ic", key=lambda s: s.abs(), ascending=False)
        dominant_row = partial_ic_df[partial_ic_df["is_dominant"]]
        if not dominant_row.empty:
            dom_sig = dominant_row.iloc[0]["signal"]
            dom_pic = dominant_row.iloc[0]["partial_ic"]
            print(f"  Dominant signal: {dom_sig}  (IC={dom_pic:+.4f})")
        if incremental.empty:
            print("  No signals with Partial IC > 0.02 found.")
        else:
            print(f"  {'Signal':<28}  {'Partial IC':>10}  Note")
            for _, row in incremental.iterrows():
                pic = row["partial_ic"]
                pic_s = f"{pic:>+10.4f}" if not np.isnan(pic) else "       NaN"
                note = "adds incremental alpha" if abs(pic) > 0.02 else ""
                print(f"  {row['signal']:<28}  {pic_s}  {note}")

    # Combination results
    print()
    print("  COMBINATION METHOD RESULTS:")
    print("  " + "-" * (W - 2))
    print(f"  {'Method':<38}  {'IC':>8}  {'ICIR':>7}  {'Verdict':<8}  vs_best_ind")
    print("  " + "-" * (W - 2))
    for r in combo_results:
        if r["ic"] is None or np.isnan(r["ic"]):
            ic_s, ir_s = "     NaN", "    NaN"
            delta_s = "    NaN"
            v = "NOISE"
        else:
            ic_s = f"{r['ic']:>+8.4f}"
            ir_s = f"{r['icir']:>+7.3f}" if not np.isnan(r["icir"]) else "    NaN"
            delta = abs(r["ic"]) - abs(best_ind_ic)
            delta_s = f"{delta:>+8.4f}"
            v = r["verdict"]
        print(f"  {r['name']:<38}  {ic_s}  {ir_s}  {v:<8}  {delta_s}")

    # Correlation clusters
    print()
    print("  CORRELATION CLUSTERS (|corr| > {:.2f}):".format(CLUSTER_CORR_THRESHOLD))
    print("  " + "-" * (W - 2))
    if corr_analysis["clusters"]:
        for i, cl in enumerate(corr_analysis["clusters"], 1):
            print(f"  Cluster {i} ({len(cl)} signals): {', '.join(cl)}")
    else:
        print("  No redundant pairs found.")

    ind = corr_analysis.get("independent_signals", [])
    if ind:
        print(f"\n  Independent signals ({len(ind)}): {', '.join(ind[:10])}", end="")
        print(f" ...+{len(ind) - 10} more" if len(ind) > 10 else "")

    # Best composite
    valid = [r for r in combo_results if not np.isnan(r.get("ic") or np.nan)]
    if valid:
        best = max(valid, key=lambda r: abs(r["ic"]))
        delta = abs(best["ic"]) - abs(best_ind_ic)
        print()
        print("=" * W)
        print(f"  BEST COMPOSITE: {best['name']}")
        print(f"  IC={best['ic']:+.4f}  ICIR={best['icir']:+.3f}  Verdict={best['verdict']}")
        print(f"  Improvement over best individual: {delta:+.4f}")
    print("=" * W)


# ── Orchestrator ──────────────────────────────────────────────────────────


def run_combination(
    instrument: str,
    timeframe: str,
    horizon: int,
    top_n_list: list[int] | None = None,
    regime_df: pd.DataFrame | None = None,  # reserved for future use
) -> pd.DataFrame:
    if top_n_list is None:
        top_n_list = DEFAULT_TOP_N

    # 1. Load data
    df = _load_ohlcv(instrument, timeframe)
    close = df["close"]

    # 2. Build all 52 signals (also populates _sweep._SIGNAL_GROUP)
    logger.info("Computing 52 signals...")
    window_1y = _get_annual_bars(timeframe)
    all_signals = build_all_signals(df, window_1y)
    group_map = dict(_sweep._SIGNAL_GROUP)

    # 3. Forward returns at single horizon
    fwd_returns = compute_forward_returns(close, [horizon])

    # 4. Drop warmup rows
    valid = all_signals.notna().any(axis=1) & fwd_returns.notna().any(axis=1)
    all_signals = all_signals[valid]
    fwd_returns = fwd_returns[valid]
    n_bars = len(all_signals)
    logger.info("Analysis window: %d bars", n_bars)

    # 5. IC and ICIR for all 52 signals at the target horizon
    logger.info("Computing IC at h=%d...", horizon)
    ic_df = compute_ic_table(all_signals, fwd_returns)
    logger.info("Computing ICIR (window=%d)...", window_1y)
    icir_s = compute_icir(all_signals, fwd_returns, horizons=[horizon], window=window_1y)

    # 6. Select USABLE signals
    usable = _select_usable(ic_df, all_signals, horizon)
    n_usable = len(usable.columns)
    logger.info("USABLE signals (|IC| >= %.3f): %d", IC_USABLE_THRESHOLD, n_usable)

    # 7. Individual top-10 for display
    fwd_col = f"fwd_{horizon}"
    ind_rows = []
    for sig in ic_df.index:
        ic_val = ic_df.loc[sig, fwd_col]
        ir_val = icir_s.get(sig, np.nan)
        ind_rows.append(
            {
                "signal": sig,
                "ic": ic_val,
                "icir": ir_val,
                "verdict": _verdict_local(ic_val, ir_val),
            }
        )
    ind_df = (
        pd.DataFrame(ind_rows)
        .assign(abs_ic=lambda x: x["ic"].abs())
        .sort_values("abs_ic", ascending=False)
        .reset_index(drop=True)
        .head(10)
    )
    best_ind_ic = float(ind_df["ic"].abs().max())

    # 8. Method 1a: correlation clustering
    logger.info("Method 1a: correlation clustering...")
    corr_analysis = method_correlation_analysis(all_signals)

    # 8b. Method 1b: partial IC
    logger.info("Method 1b: partial IC test...")
    partial_ic_df = method_partial_ic(all_signals, ic_df, fwd_returns, horizon)
    incremental_count = (
        int(((partial_ic_df["partial_ic"].abs() > 0.02) & (~partial_ic_df["is_dominant"])).sum())
        if not partial_ic_df.empty
        else 0
    )
    logger.info(
        "Partial IC: %d signal(s) with |partial IC| > 0.02 (add incremental alpha)",
        incremental_count,
    )

    # 9. Collect all combination results
    combo_results: list[dict] = []

    # Method 2: equal-weight
    logger.info("Method 2: equal-weight composites...")
    for n in top_n_list:
        actual_n = min(n, n_usable)
        r = method_equal_weight(
            usable, ic_df, fwd_returns, horizon, top_n=actual_n, window=window_1y
        )
        combo_results.append(r)

    # Method 3: ICIR-weighted
    logger.info("Method 3: ICIR-weighted composites...")
    for n in top_n_list[:2]:
        actual_n = min(n, n_usable)
        r = method_icir_weighted(
            usable, ic_df, icir_s, fwd_returns, horizon, top_n=actual_n, window=window_1y
        )
        combo_results.append(r)

    # Method 4: group-diversified
    logger.info("Method 4: group-diversified composite...")
    r = method_group_diversified(
        all_signals, ic_df, group_map, fwd_returns, horizon, window=window_1y
    )
    combo_results.append(r)

    # Method 5: PCA
    logger.info("Method 5: PCA composite...")
    r = method_pca(usable, ic_df, fwd_returns, horizon, window=window_1y)
    combo_results.append(r)

    # Method 6: gating
    logger.info("Method 6: AND-gating pairs...")
    gate_results = method_gating(all_signals, ic_df, fwd_returns, horizon, window=window_1y)
    combo_results.extend(gate_results)

    # 10. Print report
    _print_report(
        instrument,
        timeframe,
        horizon,
        n_bars,
        n_usable,
        ind_df,
        combo_results,
        best_ind_ic,
        corr_analysis,
        partial_ic_df=partial_ic_df,
    )

    # 11. Build partial IC lookup for CSV enrichment
    partial_ic_lookup: dict[str, float] = {}
    if not partial_ic_df.empty:
        partial_ic_lookup = dict(zip(partial_ic_df["signal"], partial_ic_df["partial_ic"]))

    # 12. Save CSV
    report_dir = ROOT / ".tmp" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in combo_results:
        delta = abs(r["ic"]) - abs(best_ind_ic) if not np.isnan(r.get("ic") or np.nan) else np.nan
        # partial_ic_vs_dominant: mean partial IC of signals used in this method
        used = r.get("signals_used", [])
        if used and partial_ic_lookup:
            pics = [partial_ic_lookup.get(s, np.nan) for s in used]
            pics_valid = [v for v in pics if not np.isnan(v)]
            mean_pic = float(np.mean(pics_valid)) if pics_valid else np.nan
        else:
            mean_pic = np.nan

        rows.append(
            {
                "method": r["name"],
                "ic": r["ic"],
                "icir": r["icir"],
                "verdict": r["verdict"],
                "n_signals": r["n_signals"],
                "vs_best_ind": delta,
                "partial_ic_vs_dominant": mean_pic,
                "signals_used": ",".join(used),
            }
        )
    result_df = pd.DataFrame(rows)
    out_path = report_dir / f"phase2_{instrument}_{timeframe}_h{horizon}.csv"
    result_df.to_csv(out_path, index=False)
    logger.info("Saved: %s", out_path)

    return result_df


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Signal Combination IC/ICIR Analysis")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Forward return horizon in bars (single value for combination testing)",
    )
    parser.add_argument(
        "--top_n",
        default="3,5,10",
        help="Comma-separated N values for equal-weight/ICIR-weighted tests",
    )
    args = parser.parse_args()
    top_n_list = [int(n) for n in args.top_n.split(",")]
    run_combination(
        instrument=args.instrument,
        timeframe=args.timeframe,
        horizon=args.horizon,
        top_n_list=top_n_list,
    )


if __name__ == "__main__":
    main()
