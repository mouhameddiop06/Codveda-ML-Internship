"""
Level 1 - Task 1: Data Preprocessing for Machine Learning
Codveda Technologies ML Internship

Demonstrates a complete preprocessing pipeline:
  1. Load raw data
  2. Handle missing values
  3. Encode categorical variables
  4. Scale numerical features
  5. Train / test split
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_PATH = "../../data/churn_train.csv"
TARGET = "Churn"


def main():
    # 1. Load -----------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Separate target from features
    y = df[TARGET].astype(int) if df[TARGET].dtype != "object" else (df[TARGET] == True).astype(int)
    X = df.drop(columns=[TARGET])

    # 2. Identify column types ------------------------------------------------
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    print(f"\nCategorical columns: {categorical_cols}")
    print(f"Numerical columns:   {numerical_cols}")

    # 3. Build preprocessing pipelines ----------------------------------------
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # handle missing data
        ("scaler", StandardScaler()),                    # standardize
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    # 4. Train / test split ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Fit on train, transform both -----------------------------------------
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"\nProcessed training set shape: {X_train_processed.shape}")
    print(f"Processed testing set shape:  {X_test_processed.shape}")
    print("\nPreprocessing complete - data is ready for modeling.")


if __name__ == "__main__":
    main()
