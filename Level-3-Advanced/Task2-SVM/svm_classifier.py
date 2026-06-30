"""
Level 3 - Task 2: Support Vector Machine (SVM) for Classification
Codveda Technologies ML Internship

Compare linear vs RBF kernels, visualize the decision boundary, report AUC.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, classification_report)

DATA_PATH = "../../data/iris.csv"
TARGET = "species"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = LabelEncoder().fit_transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Compare kernels ---------------------------------------------------------
    for kernel in ["linear", "rbf"]:
        clf = SVC(kernel=kernel, probability=True, random_state=42)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        proba = clf.predict_proba(X_test_s)
        auc = roc_auc_score(y_test, proba, multi_class="ovr")
        print(f"=== Kernel: {kernel} ===")
        print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.3f}")
        print(f"Recall:    {recall_score(y_test, y_pred, average='macro'):.3f}")
        print(f"AUC (OvR): {auc:.3f}\n")

    print(classification_report(y_test, SVC(kernel='rbf').fit(X_train_s, y_train).predict(X_test_s)))

    # Decision boundary on two features ---------------------------------------
    feats = ["petal_length", "petal_width"]
    X2 = StandardScaler().fit_transform(df[feats])
    clf2 = SVC(kernel="rbf").fit(X2, y)

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    plt.scatter(X2[:, 0], X2[:, 1], c=y, edgecolor="k", cmap="viridis")
    plt.xlabel(feats[0]); plt.ylabel(feats[1])
    plt.title("SVM Decision Boundary (RBF kernel)")
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=120)
    print("Saved decision boundary to decision_boundary.png")


if __name__ == "__main__":
    main()
