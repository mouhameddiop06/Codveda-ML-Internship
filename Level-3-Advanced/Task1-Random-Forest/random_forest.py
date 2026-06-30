"""
Level 3 - Task 1: Random Forest Classifier
Codveda Technologies ML Internship

Churn classification with hyperparameter tuning + feature importance.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report

TRAIN_PATH = "../../data/churn_train.csv"
TEST_PATH = "../../data/churn_test.csv"
TARGET = "Churn"


def load_xy(path):
    df = pd.read_csv(path)
    y = (df[TARGET] == True).astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def main():
    X_train, y_train = load_xy(TRAIN_PATH)
    X_test, y_test = load_xy(TEST_PATH)

    cat = X_train.select_dtypes(include=["object"]).columns.tolist()
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
        remainder="passthrough",
    )

    pipe = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(random_state=42)),
    ])

    # Hyperparameter tuning ---------------------------------------------------
    grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [None, 10, 20],
    }
    search = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1:  {search.best_score_:.3f}\n")

    best = search.best_estimator_

    # Cross-validation on train -----------------------------------------------
    cv_scores = cross_val_score(best, X_train, y_train, cv=5, scoring="f1")
    print(f"5-fold CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})\n")

    # Test evaluation ---------------------------------------------------------
    y_pred = best.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Feature importance ------------------------------------------------------
    feat_names = best.named_steps["pre"].get_feature_names_out()
    importances = pd.Series(
        best.named_steps["rf"].feature_importances_, index=feat_names
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    importances.iloc[::-1].plot.barh()
    plt.title("Top 15 Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    print("Saved feature importance plot to feature_importance.png")


if __name__ == "__main__":
    main()
