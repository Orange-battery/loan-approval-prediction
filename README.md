# Loan Approval Prediction — HDFC Bank

**AI for Engineers (UCS321) — Mini Project**

A supervised machine-learning model that predicts whether a loan application should be **Approved** or **Rejected** from applicant financial details, and interprets which factors drive the decision.

---

## Problem

HDFC Bank processes loan applications manually, which is slow and inconsistent. We build a classification model to predict the approval decision and highlight the factors that matter most.

- **Type of problem:** Classification (Approved / Rejected)
- **Target variable:** `loan_status`

## Dataset

- **Source:** Loan Approval Prediction Dataset (Kaggle, by Archit Sharma)
- **Size:** 4,269 applications, 13 columns, no missing values
- **Key fields:** annual income, loan amount & term, CIBIL score, dependents, education, employment, four asset-value columns

## Workflow

Problem Definition → Data Collection → Data Understanding → Preprocessing → EDA / Visualization → Feature Engineering → Model Development → Evaluation → Interpretation → Conclusion (see `Flow_Diagram.png`).

## Methods

- **Preprocessing:** stripped the dataset's leading-space quirk, encoded categoricals, scaled features (train split only).
- **Feature engineering:** `Loan_Income_Ratio` (repayment burden) and `Total_Assets` (collateral backing).
- **Baseline:** a majority-class DummyClassifier (62% accuracy) as the floor every model must beat.
- **Models compared:** Logistic Regression, Decision Tree, Random Forest.
- **Evaluation:** accuracy, precision, recall, F1, confusion matrix, **ROC-AUC + ROC curves**, and **5-fold cross-validation**.
- **Hyperparameter tuning:** GridSearchCV on the Random Forest.
- **Live prediction tool:** returns Approved/Rejected for a new applicant.

## Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Baseline (Dummy) | 0.62 | — | 0.50 |
| Logistic Regression | 0.91 | 0.93 | 0.97 |
| Decision Tree | 0.99 | 0.995 | 0.99 |
| Random Forest (tuned, best) | 0.995 | 0.996 | 1.00 |

5-fold cross-validation: mean accuracy 0.996 (std 0.003) — the result is stable, not a one-split fluke.

**Main insight:** CIBIL score is by far the strongest predictor. Approval rate jumps from ~11% below a CIBIL of 550 to ~99% above it. `loan_term` and the engineered `Loan_Income_Ratio` are the next most important factors.

**Honest note:** the near-perfect accuracy reflects a clean, synthetic dataset in which CIBIL score almost deterministically decides the outcome — not a guarantee of the same performance on real-world data.

## How to run

1. Open `Loan_Approval_Prediction_Final.ipynb` in Google Colab.
2. `Runtime → Run all`. The notebook loads the dataset directly from a GitHub link — no manual upload needed. (A local copy, `loan_approval_dataset.csv`, is included as backup.)

## Files

- `Loan_Approval_Prediction_Final.ipynb` — full notebook (code + explanations)
- `loan_approval_dataset.csv` — the dataset
- `Flow_Diagram.png` — project workflow diagram
- `README.md` — this file


