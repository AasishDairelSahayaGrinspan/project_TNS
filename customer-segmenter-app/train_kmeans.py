"""K-Means training with marketing personas.

Reads customers.csv (produced by T1), fits StandardScaler + KMeans(k=3),
maps centroids to Budget/Standard/Premium by sorted centroid income+score,
and emits kmeans.pkl, scaler.pkl, customers_clustered.csv.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
INPUT_CSV = APP_DIR / "customers.csv"
KMEANS_PKL = APP_DIR / "kmeans.pkl"
SCALER_PKL = APP_DIR / "scaler.pkl"
CLUSTERED_CSV = APP_DIR / "customers_clustered.csv"

PERSONAS_SORTED = ["Budget", "Standard", "Premium"]


def main() -> int:
    if not INPUT_CSV.exists():
        print(f"[train] missing input {INPUT_CSV}", file=sys.stderr)
        return 1
    df = pd.read_csv(INPUT_CSV)
    required = ["annual_income_k", "spending_score"]
    if list(df.columns) != required:
        print(f"[train] unexpected columns {list(df.columns)}, expected {required}", file=sys.stderr)
        return 1
    if df.empty:
        print("[train] empty dataset", file=sys.stderr)
        return 1

    X = df[required].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = model.fit_predict(Xs)

    centroids_orig = scaler.inverse_transform(model.cluster_centers_)
    scores = centroids_orig[:, 0] + centroids_orig[:, 1]
    order = sorted(range(3), key=lambda i: scores[i])
    cluster_to_persona = {int(c): PERSONAS_SORTED[r] for r, c in enumerate(order)}
    # Persist mapping on the model for backend contract (joblib load).
    model.cluster_to_persona_ = cluster_to_persona  # type: ignore[attr-defined]

    df["cluster"] = clusters.astype(int)
    df["persona"] = df["cluster"].map(cluster_to_persona)

    joblib.dump(model, KMEANS_PKL)
    joblib.dump(scaler, SCALER_PKL)
    df.to_csv(CLUSTERED_CSV, index=False)

    sizes = df["cluster"].value_counts().sort_index()
    print(f"[train] rows={len(df)} k=3 random_state=42")
    for c in sorted(sizes.index):
        print(f"[train] cluster {c}: n={int(sizes[c])}")
    for c in range(3):
        inc, sco = float(centroids_orig[c, 0]), float(centroids_orig[c, 1])
        print(f"[train] centroid {c}: income={inc:.2f} score={sco:.2f} -> persona={cluster_to_persona[c]}")
    print(f"[train] wrote {KMEANS_PKL.name}, {SCALER_PKL.name}, {CLUSTERED_CSV.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
