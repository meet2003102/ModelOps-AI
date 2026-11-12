"""Evaluation metrics for classification and regression."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as M

# metric name -> (sklearn scorer name, higher_is_better)
CLASSIFICATION_SCORERS = {
    "accuracy": ("accuracy", True),
    "f1_macro": ("f1_macro", True),
    "precision_macro": ("precision_macro", True),
    "recall_macro": ("recall_macro", True),
    "roc_auc": ("roc_auc_ovr", True),
}
REGRESSION_SCORERS = {
    "rmse": ("neg_root_mean_squared_error", False),
    "mae": ("neg_mean_absolute_error", False),
    "r2": ("r2", True),
}


def scorers_for(task: str) -> dict:
    return CLASSIFICATION_SCORERS if task == "classification" else REGRESSION_SCORERS


def classification_metrics(y_true, y_pred, y_proba=None, labels=None) -> dict:
    out = {
        "accuracy": M.accuracy_score(y_true, y_pred),
        "precision_macro": M.precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": M.recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": M.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": M.confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        try:
            if n_classes == 2:
                p = y_proba[:, 1]
                out["roc_auc"] = M.roc_auc_score(y_true, p)
                out["pr_auc"] = M.average_precision_score(y_true, p)
                out["brier"] = M.brier_score_loss(y_true, p)
                fpr, tpr, _ = M.roc_curve(y_true, p)
                prec, rec, _ = M.precision_recall_curve(y_true, p)
                idx = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)
                idx2 = np.linspace(0, len(prec) - 1, min(100, len(prec))).astype(int)
                out["roc_curve"] = {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}
                out["pr_curve"] = {"precision": prec[idx2].round(4).tolist(), "recall": rec[idx2].round(4).tolist()}
            else:
                out["roc_auc"] = M.roc_auc_score(y_true, y_proba, multi_class="ovr", labels=labels)
        except ValueError:
            pass  # e.g. a class missing from the test split
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": round(float(np.sqrt(M.mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(M.mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(M.r2_score(y_true, y_pred)), 4),
    }


def calibration(y_true, p_pos, n_bins: int = 10) -> dict:
    """Reliability curve for binary classifiers."""
    from sklearn.calibration import calibration_curve
    frac, mean_pred = calibration_curve(y_true, p_pos, n_bins=n_bins, strategy="quantile")
    return {"mean_predicted": mean_pred.round(4).tolist(), "fraction_positive": frac.round(4).tolist()}
