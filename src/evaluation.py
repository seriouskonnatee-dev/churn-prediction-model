"""Evaluation helpers: metrics table and confusion-matrix/ROC plotting data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(y_true, y_pred, y_proba) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def get_roc_points(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return fpr, tpr


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
