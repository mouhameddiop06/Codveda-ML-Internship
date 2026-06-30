"""
Level 2 - Task 1: Logistic Regression for Binary Classification
Codveda Technologies ML Internship

Predict customer churn (binary) and evaluate with ROC / odds ratios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             classification_report, roc_curve, roc_auc_score)

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
    num = X_train.select_dtypes(include=["number"]).columns.tolist()

    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])

    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    # Evaluate ----------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.3f}\n")
    print(classification_report(y_test, y_pred))

    # Odds ratios -------------------------------------------------------------
    feat_names = model.named_steps["pre"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    odds = pd.Series(np.exp(coefs), index=feat_names).sort_values(ascending=False)
    print("\nTop 10 features by odds ratio:")
    print(odds.head(10))

    # ROC curve ---------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Churn Logistic Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=120)
    print("\nSaved ROC curve to roc_curve.png")


if __name__ == "__main__":
    main()
