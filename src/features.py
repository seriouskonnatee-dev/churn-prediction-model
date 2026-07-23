"""Feature preparation for the churn model: one-hot encode categoricals, split
features/label, and return a fitted scaler for the numeric columns (needed by
Logistic Regression; Random Forest ignores scaling but shares the same matrix)."""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERIC_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "avg_monthly_usage",
]
BOOLEAN_COLUMNS = ["has_addon_service", "auto_pay_enabled"]
CATEGORICAL_COLUMNS = ["contract_type", "payment_method"]
TARGET_COLUMN = "churned"
ID_COLUMN = "customer_id"


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    for col in BOOLEAN_COLUMNS + [TARGET_COLUMN]:
        df[col] = df[col].astype(int)

    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, prefix=CATEGORICAL_COLUMNS)

    feature_cols = (
        NUMERIC_COLUMNS
        + BOOLEAN_COLUMNS
        + [c for c in encoded.columns if c.startswith(tuple(f"{c}_" for c in CATEGORICAL_COLUMNS))]
    )
    X = encoded[feature_cols]
    y = encoded[TARGET_COLUMN]
    return X, y


def scale_numeric(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on train numeric columns only, apply to both splits."""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERIC_COLUMNS] = scaler.fit_transform(X_train[NUMERIC_COLUMNS])
    X_test_scaled[NUMERIC_COLUMNS] = scaler.transform(X_test[NUMERIC_COLUMNS])
    return X_train_scaled, X_test_scaled, scaler
