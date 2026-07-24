"""
Level 1 - Task 3: K-Nearest Neighbors (KNN) Classifier
Codveda Technologies ML Internship

Classify iris species with KNN; evaluate and compare values of K.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

DATA_PATH = "../../data/iris.csv"
TARGET = "species"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = LabelEncoder().fit_transform(df[TARGET])
    class_names = sorted(df[TARGET].unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train KNN (K=5) ---------------------------------------------------------
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_test_s)
    print(f"Accuracy (K=5): {accuracy_score(y_test, y_pred):.3f}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix --------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
    plt.title("Confusion Matrix (KNN, K=5)")
    plt.tight_layout()
    plt.savefig("output_confusion_matrix.png", dpi=110)

    # Compare K ---------------------------------------------------------------
    k_values = range(1, 21)
    accuracies = []
    for k in k_values:
        m = KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train)
        accuracies.append(accuracy_score(y_test, m.predict(X_test_s)))
    best_k = list(k_values)[int(np.argmax(accuracies))]
    print(f"\nBest K = {best_k} (accuracy = {max(accuracies):.3f})")

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), accuracies, marker="o")
    plt.axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
    plt.xlabel("K (number of neighbors)"); plt.ylabel("Test accuracy")
    plt.title("KNN Accuracy vs K"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output_k_comparison.png", dpi=110)
    print("Saved plots.")


if __name__ == "__main__":
    main()
