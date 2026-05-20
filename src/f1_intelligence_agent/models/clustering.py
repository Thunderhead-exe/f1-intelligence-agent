"""DBSCAN clustering for lap model matrices."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def _fallback(labels: np.ndarray, eps: float = 0.0, min_samples: int = 1) -> dict[str, Any]:
    return {
        "labels": labels,
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": int(len(set(labels.tolist()) - {-1})),
        "n_noise": int(np.sum(labels == -1)),
        "method": "fallback" if len(labels) < 3 else "dbscan",
    }


def run_dbscan_clustering(
    X: np.ndarray,
    sensitivity: str = "medium",
) -> dict[str, Any]:
    """Run DBSCAN with an adaptive eps derived from nearest-neighbor distances."""

    if X is None or X.size == 0:
        return _fallback(np.array([], dtype=int))

    n_rows = int(X.shape[0])
    if n_rows < 4:
        return _fallback(np.zeros(n_rows, dtype=int), min_samples=max(1, n_rows))

    sensitivity = sensitivity if sensitivity in {"low", "medium", "high"} else "medium"
    min_samples_map = {"low": 5, "medium": 4, "high": 3}
    quantile_map = {"low": 0.90, "medium": 0.82, "high": 0.72}
    min_samples = min(min_samples_map[sensitivity], n_rows)

    try:
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        kth_distances = np.sort(distances[:, -1])
        eps = float(np.quantile(kth_distances, quantile_map[sensitivity]))
        if not np.isfinite(eps) or eps <= 0:
            eps = float(np.nanmedian(kth_distances[kth_distances > 0])) if np.any(kth_distances > 0) else 0.5
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    except Exception:
        labels = np.zeros(n_rows, dtype=int)
        eps = 0.0

    return _fallback(np.asarray(labels, dtype=int), eps=eps, min_samples=min_samples)

