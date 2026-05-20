"""Isolation Forest anomaly detection with small-data fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest


def _robust_z_fallback(X: np.ndarray, sensitivity: str) -> dict[str, Any]:
    if X is None or X.size == 0:
        scores = np.array([], dtype=float)
        labels = np.array([], dtype=int)
        is_anomaly = np.array([], dtype=bool)
    else:
        median = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - median), axis=0)
        mad = np.where(mad == 0, 1.0, mad)
        z = 0.6745 * (X - median) / mad
        scores = np.nanmax(np.abs(z), axis=1)
        threshold_map = {"low": 4.5, "medium": 3.5, "high": 2.8}
        threshold = threshold_map.get(sensitivity, 3.5)
        is_anomaly = scores >= threshold
        labels = np.where(is_anomaly, -1, 1)
        return {
            "labels": labels,
            "is_anomaly": is_anomaly,
            "anomaly_scores": scores,
            "threshold": threshold,
            "method": "robust_z_score",
            "contamination": None,
        }
    return {
        "labels": labels,
        "is_anomaly": is_anomaly,
        "anomaly_scores": scores,
        "threshold": 0.0,
        "method": "robust_z_score",
        "contamination": None,
    }


def run_isolation_forest_anomaly_detection(
    X: np.ndarray,
    sensitivity: str = "medium",
    random_state: int = 42,
) -> dict[str, Any]:
    """Run Isolation Forest and return higher scores for more anomalous rows."""

    if X is None or X.size == 0 or X.shape[0] < 8:
        return _robust_z_fallback(X, sensitivity)

    sensitivity = sensitivity if sensitivity in {"low", "medium", "high"} else "medium"
    contamination_map = {"low": 0.03, "medium": 0.07, "high": 0.12}
    contamination = min(contamination_map[sensitivity], max(1 / X.shape[0], 0.49))

    model = IsolationForest(
        n_estimators=250,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    labels = model.fit_predict(X)
    anomaly_scores = -model.decision_function(X)
    threshold = float(np.quantile(anomaly_scores, 1 - contamination))
    is_anomaly = labels == -1
    return {
        "labels": labels.astype(int),
        "is_anomaly": is_anomaly.astype(bool),
        "anomaly_scores": anomaly_scores.astype(float),
        "threshold": threshold,
        "method": "isolation_forest",
        "contamination": contamination,
    }

