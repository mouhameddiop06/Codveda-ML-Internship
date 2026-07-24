"""
Level 2 - Task 3: K-Means Clustering (Customer Segmentation)
Codveda Technologies ML Internship

Segment telecom customers with K-Means; choose K via elbow + silhouette; visualize with PCA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

DATA_PATH = "../../data/churn_train.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.select_dtypes(include="number").copy()
    X_scaled = StandardScaler().fit_transform(X)

    # Elbow + silhouette ------------------------------------------------------
    K_range = range(2, 11)
    inertia, sils = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        sils.append(silhouette_score(X_scaled, labels))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(list(K_range), inertia, marker="o"); ax[0].set_title("Elbow Method")
    ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia")
    ax[1].plot(list(K_range), sils, marker="o", color="darkorange")
    ax[1].set_title("Silhouette Analysis"); ax[1].set_xlabel("K"); ax[1].set_ylabel("Score")
    plt.tight_layout(); plt.savefig("output_elbow_silhouette.png", dpi=110)

    # Final clustering + PCA viz ---------------------------------------------
    k = 4
    clusters = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    coords = PCA(n_components=2).fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=clusters, palette="viridis", s=30)
    plt.title(f"Customer Segments (K={k}, PCA 2D)")
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
    plt.tight_layout(); plt.savefig("output_clusters.png", dpi=110)

    # Segment profiles --------------------------------------------------------
    seg = X.copy(); seg["Cluster"] = clusters
    cols = ["Account length", "Total day minutes", "Total day charge",
            "Customer service calls", "Number vmail messages"]
    print(seg.groupby("Cluster")[cols].mean().round(1))
    print("\nSaved plots.")


if __name__ == "__main__":
    main()
