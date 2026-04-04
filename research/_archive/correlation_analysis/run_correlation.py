"""Cross-Asset Correlation Analysis.

Computes pairwise Pearson (+ Spearman) correlation of daily log returns,
clusters assets by similarity, classifies pairs, and outputs figures + CSVs.

Usage
-----
  # Default curated cross-asset set
  uv run python research/correlation_analysis/run_correlation.py

  # Pin to a common window (apples-to-apples across FX + equities since 2016)
  uv run python research/correlation_analysis/run_correlation.py --start 2016-01-01

  # All 526 daily symbols
  uv run python research/correlation_analysis/run_correlation.py --all

  # Custom list
  uv run python research/correlation_analysis/run_correlation.py --symbols SPY_D QQQ_D GLD_D EUR_USD_D

  # Stricter threshold + longer history requirement
  uv run python research/correlation_analysis/run_correlation.py --threshold 0.4 --min-bars 504

Output
------
  .tmp/correlation_heatmap.png      -- clustered correlation matrix
  .tmp/correlation_dendrogram.png   -- hierarchical clustering tree
  .tmp/correlation_matrix.csv       -- full N x N matrix
  .tmp/correlation_pairs.csv        -- all pairs ranked by |r|
"""

from __future__ import annotations

import argparse
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / ".tmp"
TMP.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Default curated symbol list
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS = [
    # US equities
    "SPY_D",
    "QQQ_D",
    "UNH_D",
    "AMAT_D",
    "TXN_D",
    "INTC_D",
    "CAT_D",
    "WMT_D",
    "TMO_D",
    # Indices
    "^FTSE_D",
    "^GDAXI_D",
    # FX
    "EUR_USD_D",
    "GBP_USD_D",
    "USD_JPY_D",
    "AUD_USD_D",
    "USD_CHF_D",
    "AUD_JPY_D",
    # Commodities
    "GLD_D",
]


# Short display labels (strip _D suffix for readability)
def _label(sym: str) -> str:
    return sym.replace("_D", "").replace("^", "")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_returns(
    symbols: list[str],
    start: str | None,
    end: str | None,
    min_bars: int,
) -> pd.DataFrame:
    raw: dict[str, pd.Series] = {}
    skipped: list[str] = []

    for sym in symbols:
        path = ROOT / "data" / f"{sym}.parquet"
        if not path.exists():
            print(f"  [SKIP] {sym} — file not found")
            skipped.append(sym)
            continue
        try:
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                    df.index = pd.to_datetime(df.index, utc=True)
                else:
                    skipped.append(sym)
                    continue
            df.columns = [c.lower() for c in df.columns]
            close = df["close"].dropna().sort_index()
            # Strip timezone first
            if hasattr(close.index, "tz") and close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            # For daily data different sources use different times of day (05:00 vs 22:00 UTC).
            # Detect daily vs intraday: if avg bars per calendar date <= 2, treat as daily
            # and normalise to midnight so cross-source inner-join works.
            dates_normalized = close.index.normalize()
            avg_bars_per_day = len(close) / dates_normalized.nunique()
            if avg_bars_per_day <= 2:
                close.index = dates_normalized
                close = close.groupby(level=0).last()  # deduplicate same-day rows
            ret = np.log(close / close.shift(1)).dropna()
            # Apply date window before recording
            if start:
                ret = ret[ret.index >= pd.Timestamp(start)]
            if end:
                ret = ret[ret.index <= pd.Timestamp(end)]
            raw[sym] = ret
        except Exception as e:
            print(f"  [SKIP] {sym} — {e}")
            skipped.append(sym)

    if not raw:
        raise ValueError("No data loaded.")

    # Inner join on dates
    aligned = pd.DataFrame(raw).dropna()

    # Coverage report
    common_start = aligned.index[0].date()
    common_end = aligned.index[-1].date()
    n_common = len(aligned)
    print(f"\nCommon window: {common_start} to {common_end}  ({n_common} bars)")
    for sym, s in raw.items():
        full_n = len(s)
        lost = full_n - n_common
        if lost > 5:
            print(
                f"  {sym}: {full_n} bars total, {lost} dropped to align "
                f"(individual start: {s.index[0].date()})"
            )

    # Drop symbols with too few bars after alignment
    survivors = [c for c in aligned.columns if aligned[c].count() >= min_bars]
    dropped = [c for c in aligned.columns if c not in survivors]
    if dropped:
        print(f"\n  [EXCL] Fewer than {min_bars} bars after alignment: {dropped}")
    aligned = aligned[survivors].dropna()

    if skipped:
        print(f"\n  Skipped entirely: {skipped}")

    return aligned


# ---------------------------------------------------------------------------
# Correlation + clustering
# ---------------------------------------------------------------------------


def compute_correlations(aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson = aligned.corr(method="pearson")
    spearman = aligned.corr(method="spearman")
    return pearson, spearman


def cluster_order(corr: pd.DataFrame) -> list[int]:
    """Return row/col order from Ward hierarchical clustering."""
    n = len(corr)
    if n < 3:
        return list(range(n))
    # Distance = 1 - correlation (clipped to [0,2] to avoid neg due to float noise)
    dist_sq = np.clip(1.0 - corr.values, 0, 2)
    np.fill_diagonal(dist_sq, 0.0)
    dist_condensed = squareform(dist_sq, checks=False)
    Z = linkage(dist_condensed, method="ward")
    return list(leaves_list(Z))


def classify_pairs(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    threshold: float,
    aligned: pd.DataFrame,
) -> pd.DataFrame:
    symbols = list(pearson.columns)
    rows = []
    n_obs = len(aligned)
    for a, b in combinations(symbols, 2):
        r_p = pearson.loc[a, b]
        r_s = spearman.loc[a, b]
        if r_p > threshold:
            cls = "positive"
        elif r_p < -threshold:
            cls = "negative"
        else:
            cls = "uncorrelated"
        rows.append(
            {
                "symbol_a": a,
                "symbol_b": b,
                "pearson_r": round(r_p, 4),
                "spearman_r": round(r_s, 4),
                "class": cls,
                "n_obs": n_obs,
            }
        )
    df = pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)
    return df


def identify_clusters(
    corr: pd.DataFrame, order: list[int], threshold: float
) -> dict[int, list[str]]:
    """Label clusters by cutting the dendrogram at distance = 1 - threshold."""
    from scipy.cluster.hierarchy import fcluster

    symbols = list(corr.columns)
    n = len(symbols)
    if n < 3:
        return {1: symbols}
    dist_sq = np.clip(1.0 - corr.values, 0, 2)
    np.fill_diagonal(dist_sq, 0.0)
    dist_condensed = squareform(dist_sq, checks=False)
    Z = linkage(dist_condensed, method="ward")
    cut = 1.0 - threshold
    labels = fcluster(Z, t=cut, criterion="distance")
    clusters: dict[int, list[str]] = {}
    for sym, lbl in zip(symbols, labels):
        clusters.setdefault(int(lbl), []).append(sym)
    return clusters


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_heatmap(corr_clustered: pd.DataFrame, out_path: Path, title: str) -> None:
    n = len(corr_clustered)
    fig_size = max(10, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    labels = [_label(c) for c in corr_clustered.columns]
    mat = corr_clustered.values

    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8 if n > 20 else 10)
    ax.set_yticklabels(labels, fontsize=8 if n > 20 else 10)

    # Annotate cells only for small matrices
    if n <= 20:
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dendrogram(corr: pd.DataFrame, out_path: Path, title: str, threshold: float) -> None:
    n = len(corr)
    if n < 3:
        return
    labels = [_label(c) for c in corr.columns]
    dist_sq = np.clip(1.0 - corr.values, 0, 2)
    np.fill_diagonal(dist_sq, 0.0)
    dist_condensed = squareform(dist_sq, checks=False)
    Z = linkage(dist_condensed, method="ward")

    fig_w = max(12, n * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    cut = 1.0 - threshold
    dendrogram(
        Z,
        labels=labels,
        ax=ax,
        color_threshold=cut,
        leaf_rotation=45,
        leaf_font_size=9 if n > 20 else 10,
    )
    ax.axhline(
        cut,
        color="red",
        linewidth=1.0,
        linestyle="--",
        label=f"Cut at 1-r={cut:.1f} (r={threshold})",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Ward linkage distance")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary(
    pairs: pd.DataFrame,
    clusters: dict[int, list[str]],
    threshold: float,
    n_obs: int,
) -> None:
    pos = pairs[pairs["class"] == "positive"].head(10)
    neg = pairs[pairs["class"] == "negative"].sort_values("pearson_r").head(10)
    unc = pairs[pairs["class"] == "uncorrelated"]

    def fmt_pair(row: pd.Series) -> str:
        a, b = _label(row["symbol_a"]), _label(row["symbol_b"])
        return (
            f"  {a:<14} <-> {b:<14}  r={row['pearson_r']:+.4f}  (spearman={row['spearman_r']:+.4f})"
        )

    print(f"\n{'=' * 65}")
    print(f"CORRELATION SUMMARY  (n_obs={n_obs}, threshold=+/-{threshold})")
    print(f"{'=' * 65}")

    print(
        f"\nTop POSITIVE pairs  (r > +{threshold})  [{len(pairs[pairs['class'] == 'positive'])} total]"
    )
    for _, row in pos.iterrows():
        print(fmt_pair(row))

    print(
        f"\nTop NEGATIVE pairs  (r < -{threshold})  [{len(pairs[pairs['class'] == 'negative'])} total]"
    )
    for _, row in neg.iterrows():
        print(fmt_pair(row))

    print(f"\nUncorrelated pairs  (|r| <= {threshold}): {len(unc)}")
    if len(unc) <= 10:
        for _, row in unc.sort_values("pearson_r", key=abs, ascending=False).iterrows():
            print(fmt_pair(row))

    print(f"\nCluster groups ({len(clusters)} clusters):")
    for cid, members in sorted(clusters.items()):
        labels = ", ".join(_label(m) for m in members)
        print(f"  Cluster {cid} ({len(members)} assets): {labels}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Parquet stems (without .parquet). Default: curated cross-asset set.",
    )
    parser.add_argument("--all", action="store_true", help="Use all *_D.parquet files in data/")
    parser.add_argument("--start", default=None, help="Start date e.g. 2016-01-01")
    parser.add_argument("--end", default=None, help="End date e.g. 2024-12-31")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Abs correlation threshold for positive/negative/uncorrelated (default 0.3)",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=252,
        dest="min_bars",
        help="Minimum bars in common window to include a symbol (default 252)",
    )
    parser.add_argument("--tag", default="", help="Optional output filename tag")
    args = parser.parse_args()

    if args.all:
        symbols = [p.stem for p in sorted((ROOT / "data").glob("*_D.parquet"))]
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = DEFAULT_SYMBOLS

    tag = f"_{args.tag}" if args.tag else ""
    print(f"Loading {len(symbols)} symbols...")
    aligned = load_returns(symbols, args.start, args.end, args.min_bars)
    n = len(aligned.columns)
    n_obs = len(aligned)
    date_range = f"{aligned.index[0].date()} to {aligned.index[-1].date()}"
    print(f"\n{n} symbols x {n_obs} bars in common window")

    print("Computing correlations...")
    pearson, spearman = compute_correlations(aligned)

    print("Clustering...")
    order = cluster_order(pearson)
    pearson_cl = pearson.iloc[order, order]
    clusters = identify_clusters(pearson, order, args.threshold)

    print("Classifying pairs...")
    pairs = classify_pairs(pearson, spearman, args.threshold, aligned)

    # Save CSVs
    matrix_path = TMP / f"correlation_matrix{tag}.csv"
    pairs_path = TMP / f"correlation_pairs{tag}.csv"
    pearson.to_csv(matrix_path)
    pairs.to_csv(pairs_path, index=False)
    print(f"\nMatrix -> {matrix_path}")
    print(f"Pairs  -> {pairs_path}")

    # Save figures
    hm_title = f"Cross-Asset Correlation Matrix ({date_range}, n={n_obs})"
    hm_path = TMP / f"correlation_heatmap{tag}.png"
    plot_heatmap(pearson_cl, hm_path, hm_title)
    print(f"Heatmap -> {hm_path}")

    dg_title = f"Hierarchical Clustering Dendrogram ({date_range})"
    dg_path = TMP / f"correlation_dendrogram{tag}.png"
    plot_dendrogram(pearson, dg_path, dg_title, args.threshold)
    print(f"Dendrogram -> {dg_path}")

    # Console summary
    print_summary(pairs, clusters, args.threshold, n_obs)


if __name__ == "__main__":
    main()
