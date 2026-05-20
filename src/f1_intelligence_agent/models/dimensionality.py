"""Dimensionality reduction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def run_pca_projection(X: np.ndarray) -> pd.DataFrame:
    """Return a two-dimensional PCA projection with explained variance metadata."""

    if X is None or X.size == 0:
        return pd.DataFrame(columns=["PC1", "PC2", "ExplainedVariancePC1", "ExplainedVariancePC2"])

    if X.shape[0] == 1:
        return pd.DataFrame(
            {
                "PC1": [0.0],
                "PC2": [0.0],
                "ExplainedVariancePC1": [1.0],
                "ExplainedVariancePC2": [0.0],
            }
        )

    n_components = min(2, X.shape[0], X.shape[1])
    if n_components == 0:
        return pd.DataFrame({"PC1": np.zeros(X.shape[0]), "PC2": np.zeros(X.shape[0])})

    projection = PCA(n_components=n_components, random_state=42).fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42).fit(X)
    pc1 = projection[:, 0]
    pc2 = projection[:, 1] if n_components > 1 else np.zeros(X.shape[0])
    evr = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "PC1": pc1,
            "PC2": pc2,
            "ExplainedVariancePC1": evr[0] if len(evr) > 0 else 0.0,
            "ExplainedVariancePC2": evr[1] if len(evr) > 1 else 0.0,
        }
    )

