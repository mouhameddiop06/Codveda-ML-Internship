"""
Level 2 - Task 2: Decision Trees for Classification
Codveda Technologies ML Internship

Classify Iris species, visualize and prune the tree.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_PATH = "../../data/iris.csv"
TARGET = "species"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pruned tree (max_depth limits growth to avoid overfitting) ---------------
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-score (macro): {f1_score(y_test, y_pred, average='macro'):.3f}\n")
    print(classification_report(y_test, y_pred))

    # Visualize ---------------------------------------------------------------
    plt.figure(figsize=(14, 8))
    plot_tree(model, feature_names=X.columns, class_names=model.classes_,
              filled=True, rounded=True)
    plt.title("Decision Tree - Iris (max_depth=3)")
    plt.tight_layout()
    plt.savefig("decision_tree.png", dpi=120)
    print("Saved tree visualization to decision_tree.png")


if __name__ == "__main__":
    main()
