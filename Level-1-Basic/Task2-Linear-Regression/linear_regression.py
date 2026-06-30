"""
Level 1 - Task 2: Simple Linear Regression
Codveda Technologies ML Internship

Predict median house value (MEDV) using linear regression.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "../../data/house_prediction.csv"
COLUMNS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
           "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
TARGET = "MEDV"


def main():
    # 1. Load (whitespace-delimited, no header) -------------------------------
    df = pd.read_csv(DATA_PATH, delim_whitespace=True, header=None, names=COLUMNS)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 2. Split + scale --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 3. Train ----------------------------------------------------------------
    model = LinearRegression()
    model.fit(X_train_s, y_train)

    # 4. Interpret coefficients -----------------------------------------------
    coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
    print("\nModel coefficients (standardized features):")
    print(coefs)
    print(f"Intercept: {model.intercept_:.3f}")

    # 5. Evaluate -------------------------------------------------------------
    y_pred = model.predict(X_test_s)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nMean Squared Error (MSE): {mse:.3f}")
    print(f"R-squared (R2):           {r2:.3f}")


if __name__ == "__main__":
    main()
