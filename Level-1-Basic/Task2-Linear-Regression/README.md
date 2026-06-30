# Level 1 · Task 2 — Simple Linear Regression

**Objective:** Build a linear regression model to predict a continuous variable (house prices).

## Steps covered
- Load and preprocess the dataset.
- Train a linear regression model with scikit-learn.
- Interpret the model coefficients.
- Evaluate with R-squared and Mean Squared Error (MSE).

## Dataset
`data/house_prediction.csv` — Boston-style housing data (whitespace-separated, no header).
Columns assigned in the script:
`CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV`
Target = **MEDV** (median home value).

## Run
```bash
python linear_regression.py
```

## Tools
Python, pandas, scikit-learn.
