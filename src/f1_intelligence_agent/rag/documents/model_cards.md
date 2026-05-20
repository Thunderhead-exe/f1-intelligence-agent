# Model Cards

DBSCAN: Density-Based Spatial Clustering of Applications with Noise. It groups points that are close in feature space and marks sparse points as noise. It is useful for discovering hidden lap groups without choosing the number of clusters. Results are sensitive to scaling, feature choice, and the eps/min_samples settings.

Isolation Forest: An unsupervised anomaly detection model that isolates unusual observations with random trees. It can rank laps that differ from the session feature distribution. It does not explain the cause of an anomaly by itself.

PCA: Principal Component Analysis projects high-dimensional features into lower-dimensional axes. A PCA scatter plot is useful for visual inspection, but principal components are summaries, not direct physical causes.

Robust z-score: A deviation measure based on median and median absolute deviation. It is more resistant to outliers than a standard z-score and can identify which lap features most differ from a baseline.

Unsupervised anomaly detection limitations: These models detect unusual patterns in the available data. They cannot prove driver error, tyre failure, traffic, damage, or strategy decisions. Findings should be treated as analytical hypotheses and validated against telemetry, race-control messages, official notes, onboard footage, or domain context.

