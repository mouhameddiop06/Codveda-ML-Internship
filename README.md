# Codveda Technologies — Machine Learning Internship

This repository contains my project work for the **Machine Learning Internship** at
**Codveda Technologies** (June–July 2026).

The internship is organized into three levels of increasing difficulty. The requirement
is to complete **any two tasks per level**. The tasks selected and completed in this
repository are highlighted below.

## 👤 Intern

- **Name:** Mouhamed Diop
- **Role:** Machine Learning Intern
- **Intern ID:** CV/A1/79278
- **Tools:** Python, pandas, scikit-learn, TensorFlow/Keras, matplotlib, seaborn

## 📁 Repository Structure

```
Codveda-ML-Internship/
├── data/                          # Datasets used across tasks
├── Level-1-Basic/
│   ├── Task1-Data-Preprocessing/
│   └── Task2-Linear-Regression/
├── Level-2-Intermediate/
│   ├── Task1-Logistic-Regression/
│   └── Task2-Decision-Trees/
├── Level-3-Advanced/
│   ├── Task1-Random-Forest/
│   └── Task2-SVM/
├── requirements.txt
└── README.md
```

## ✅ Tasks Completed

| Level | Task | Project | Dataset | Status |
|-------|------|---------|---------|--------|
| 1 — Basic | 1 | Data Preprocessing for ML | churn / house | ⬜ In progress |
| 1 — Basic | 2 | Simple Linear Regression | house_prediction | ⬜ In progress |
| 2 — Intermediate | 1 | Logistic Regression (binary) | churn | ⬜ In progress |
| 2 — Intermediate | 2 | Decision Trees | iris | ⬜ In progress |
| 3 — Advanced | 1 | Random Forest Classifier | churn | ⬜ In progress |
| 3 — Advanced | 2 | Support Vector Machine (SVM) | iris | ⬜ In progress |

> Update the **Status** column to ✅ as each task is completed. Each task folder has its
> own README describing the objectives, approach, and results.

## 📊 Datasets

| File | Description |
|------|-------------|
| `data/iris.csv` | Iris flower dataset (classification, 150 rows) |
| `data/house_prediction.csv` | Boston-style housing dataset (regression) |
| `data/churn_train.csv` / `churn_test.csv` | Telecom customer churn (binary classification) |
| `data/sentiment.csv` | Social-media sentiment dataset (NLP, optional) |

> The large **Stock Prices** dataset (~24 MB) is not committed to keep the repo light.
> Place it in `data/` locally if needed.

## ⚙️ Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Codveda-ML-Internship.git
cd Codveda-ML-Internship

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run a task
python Level-1-Basic/Task2-Linear-Regression/linear_regression.py
```

## 🔗 Links

- **LinkedIn:** project demos posted with #CodvedaJourney #CodvedaExperience #FutureWithCodveda
- **Company:** [Codveda Technologies](https://www.codveda.com) · @Codveda

---
*Completed as part of the Codveda Technologies internship program.*
