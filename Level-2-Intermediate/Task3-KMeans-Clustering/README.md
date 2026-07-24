# Level 2 · Task 3 — K-Means Clustering

**Objective:** Group unlabeled data into clusters (customer segmentation).

## Steps covered
- Load a dataset and preprocess it (scaling).
- Apply K-Means and find the optimal number of clusters with the **elbow method** (and silhouette score).
- Visualize clusters using a 2D scatter plot (via PCA).
- Interpret the resulting segments.

## Dataset
`data/churn_train.csv` — telecom customers. We cluster on the **numerical usage/billing
features** (the `Churn` label is ignored, since clustering is unsupervised).

## Run
```bash
python kmeans_clustering.py     # script version
# or open kmeans_clustering.ipynb
```

## Tools
Python, scikit-learn, matplotlib, seaborn.
