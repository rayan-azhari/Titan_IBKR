"""pca_archetypes.py — Day archetype discovery via PCA + K-means.

Two-pass approach:
  Pass 1: sklearn PCA + K-means (fast, no new deps). Try first.
  Pass 2: tslearn multivariate DTW (if installed and silhouette < 0.25).

All models fit on IS data only. OOS uses frozen IS models.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Pass 1: PCA + K-means (Euclidean)
# ---------------------------------------------------------------------------


class PCAArchetypes:
    """PCA dimensionality reduction + K-means clustering on flattened daily tensors.

    Input shape: (n_days, n_bars, 3) — flattened to (n_days, n_bars*3) before PCA.
    """

    def __init__(
        self,
        n_components: int = 15,
        n_clusters: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.km_: KMeans | None = None
        self.silhouette_: float = float("nan")

    def fit(self, matrices: np.ndarray) -> "PCAArchetypes":
        """Fit scaler, PCA, and K-means on IS daily matrices.

        Args:
            matrices: IS data, shape (n_days, n_bars, 3).

        Returns:
            self
        """
        n_days = matrices.shape[0]
        flat = matrices.reshape(n_days, -1)

        self.scaler_ = StandardScaler()
        flat_sc = self.scaler_.fit_transform(flat)

        # Cap n_components at min(n_days-1, n_features)
        n_comp = min(self.n_components, flat_sc.shape[0] - 1, flat_sc.shape[1])
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
        scores = self.pca_.fit_transform(flat_sc)

        # Clamp n_clusters: need at least 2 samples per cluster
        effective_k = min(self.n_clusters, n_days // 2)
        effective_k = max(effective_k, 2)
        self.km_ = KMeans(
            n_clusters=effective_k,
            random_state=self.random_state,
            n_init=20,
        )
        labels = self.km_.fit_predict(scores)

        if n_days > effective_k:
            self.silhouette_ = float(silhouette_score(scores, labels))

        return self

    def predict(self, matrices: np.ndarray) -> np.ndarray:
        """Classify new daily matrices using IS-frozen models.

        Args:
            matrices: shape (n_days, n_bars, 3) or (n_bars, 3) for a single day.

        Returns:
            Integer cluster labels, shape (n_days,).
        """
        assert self.scaler_ is not None, "Call fit() first."
        if matrices.ndim == 2:
            matrices = matrices[np.newaxis]
        flat = matrices.reshape(len(matrices), -1)
        flat_sc = self.scaler_.transform(flat)
        scores = self.pca_.transform(flat_sc)
        return self.km_.predict(scores)

    def explained_variance(self) -> float:
        """Total explained variance ratio of the PCA components."""
        assert self.pca_ is not None, "Call fit() first."
        return float(self.pca_.explained_variance_ratio_.sum())


def sweep_k(
    matrices: np.ndarray,
    k_values: list[int] | None = None,
    n_components: int = 15,
    random_state: int = 42,
) -> list[dict]:
    """Fit PCAArchetypes for each K and return silhouette scores.

    Args:
        matrices: IS data, shape (n_days, n_bars, 3).
        k_values: List of K values to try. Defaults to [4, 5, 6].
        n_components: PCA dimensionality.
        random_state: Seed.

    Returns:
        List of dicts: [{k, silhouette, explained_variance, model}, ...] sorted by silhouette.
    """
    if k_values is None:
        k_values = [4, 5, 6]

    results = []
    for k in k_values:
        m = PCAArchetypes(n_components=n_components, n_clusters=k, random_state=random_state)
        m.fit(matrices)
        results.append(
            {
                "k": k,
                "silhouette": m.silhouette_,
                "explained_variance": m.explained_variance(),
                "model": m,
            }
        )

    return sorted(results, key=lambda r: r["silhouette"], reverse=True)


# ---------------------------------------------------------------------------
# Pass 2: Multivariate DTW (optional — requires tslearn)
# ---------------------------------------------------------------------------


def fit_dtw_archetypes(
    matrices: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
    metric: str = "softdtw",
) -> tuple[object, float]:
    """Fit multivariate DTW K-means on IS daily matrices.

    Requires: uv add tslearn

    Args:
        matrices: IS data, shape (n_days, n_bars, 3).
        n_clusters: Number of archetypes.
        random_state: Seed.
        metric: "dtw" or "softdtw" (softdtw is faster for large K).

    Returns:
        Tuple of (fitted TimeSeriesKMeans, silhouette_score).

    Raises:
        ImportError if tslearn is not installed.
    """
    try:
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.metrics import cdist_dtw
    except ImportError as exc:
        raise ImportError("tslearn not installed. Run: uv add tslearn") from exc

    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric=metric,
        random_state=random_state,
        n_init=5,
        max_iter=10,
        verbose=False,
    )
    labels = km.fit_predict(matrices)

    # Silhouette via DTW distance matrix (expensive but exact)
    try:
        dist = cdist_dtw(matrices)
        sil = float(silhouette_score(dist, labels, metric="precomputed"))
    except Exception:
        sil = float("nan")

    return km, sil


# ---------------------------------------------------------------------------
# Archetype characterisation helpers
# ---------------------------------------------------------------------------


def describe_archetypes(
    matrices: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> list[dict]:
    """Summarise each cluster by its mean centroid CLV evolution.

    Args:
        matrices: shape (n_days, n_bars, 3) — columns: [TR, CLV, RelVol].
        labels: Cluster assignment per day, shape (n_days,).
        n_clusters: Number of clusters.

    Returns:
        List of dicts, one per cluster, with:
          label, n_days, mean_clv_start, mean_clv_end, mean_tr, mean_relvol,
          suggested_name (heuristic label based on CLV trajectory).
    """
    descriptions = []
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            descriptions.append(
                {
                    "label": k,
                    "n_days": 0,
                    "suggested_name": "Empty",
                    "_clv_end_raw": 0.0,
                    "_clv_start_raw": 0.0,
                }
            )
            continue
        cluster = matrices[mask]  # (n_k, n_bars, 3)
        mean_clv = cluster[:, :, 1].mean(axis=0)  # (n_bars,)
        mean_tr = float(cluster[:, :, 0].mean())
        mean_relvol = float(cluster[:, :, 2].mean())

        clv_start = float(mean_clv[: len(mean_clv) // 4].mean())
        clv_end = float(mean_clv[3 * len(mean_clv) // 4 :].mean())

        descriptions.append(
            {
                "label": k,
                "n_days": int(mask.sum()),
                "mean_clv_start": round(clv_start, 4),
                "mean_clv_end": round(clv_end, 4),
                "mean_tr": round(mean_tr, 6),
                "mean_relvol": round(mean_relvol, 4),
                "_clv_end_raw": clv_end,
                "_clv_start_raw": clv_start,
                "suggested_name": "Flat",  # overwritten in relative pass below
            }
        )

    # Relative labelling: rank by CLV_end across valid (n>=3) clusters.
    valid = [d for d in descriptions if d["n_days"] >= 3]
    if valid:
        ends = [d["_clv_end_raw"] for d in valid]
        max_e, min_e = max(ends), min(ends)
        for d in valid:
            e, s = d["_clv_end_raw"], d["_clv_start_raw"]
            if e == max_e:
                d["suggested_name"] = "TrendUp"
            elif e == min_e:
                d["suggested_name"] = "TrendDown"
            elif s < -abs(e) * 0.5 and e > 0:
                d["suggested_name"] = "VReversal"
            elif s > abs(e) * 0.5 and e < 0:
                d["suggested_name"] = "InvVReversal"
            else:
                d["suggested_name"] = "Flat"

    for d in descriptions:
        d.pop("_clv_end_raw", None)
        d.pop("_clv_start_raw", None)

    return descriptions
