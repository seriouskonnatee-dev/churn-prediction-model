# Churn Prediction Model

Binary classification of customer churn on a synthetic subscription-retail dataset, comparing an
interpretable **Logistic Regression** baseline against a **Random Forest**, with EDA, ROC-AUC/confusion
matrix evaluation, and feature importance to explain *why* customers are flagged as at-risk.

**Design doc first:** [`docs/design.md`](docs/design.md) — problem statement, data/feature-flow diagram,
data dictionary, and schema design rationale, written before any modeling code.

**Model card:** [`MODEL_CARD.md`](MODEL_CARD.md) — intended use, evaluation results, and limitations.

## Results (75/25 stratified holdout)

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | ~0.77 | ~0.28 | ~0.72 | ~0.41 |
| Random Forest | ~0.75 | ~0.31 | ~0.57 | ~0.40 |

Contract type (month-to-month vs term contracts) is the single strongest churn driver in both models —
see the notebook's feature importance and coefficient plots.

## Why this exists

Portfolio project demonstrating a churn-modeling workflow end to end: synthetic data generation with a
deliberately imbalanced, realistic churn rate; leakage-safe feature engineering; comparing an
interpretable linear model against a non-linear ensemble; honest evaluation under class imbalance
(ROC-AUC/precision/recall, not just accuracy); and a model card documenting limitations.

## Repo structure

```
docs/design.md            Design doc: problem, ER/flow diagram, data dictionary, schema
MODEL_CARD.md              Intended use, metrics, and limitations
src/
  data_generation.py       Synthetic customers generator (contract, tenure, usage, support tickets...)
  features.py               One-hot encoding + numeric scaling helpers
  models.py                  Logistic Regression + Random Forest training
  evaluation.py              ROC-AUC/precision/recall/F1 + confusion-matrix helpers
scripts/build_notebook.py   One-time script that authored notebooks/churn_prediction.ipynb
notebooks/churn_prediction.ipynb   Executed end-to-end notebook (EDA -> features -> models -> evaluation)
outputs/                    Sample plots + model_comparison.csv committed so results are visible without running anything
```

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/churn_prediction.ipynb
```

Or open `notebooks/churn_prediction.ipynb` directly in Jupyter/VS Code — fully self-contained, generates
its own synthetic data, no external services or credentials required.

## Skills demonstrated

Python, pandas, EDA, scikit-learn (Logistic Regression, Random Forest), classification evaluation
(ROC-AUC, confusion matrix, precision/recall), feature importance, model documentation, design-first
analytics engineering workflow.
