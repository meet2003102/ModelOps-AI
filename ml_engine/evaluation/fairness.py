"""Group fairness report for classification models.

Definitions follow Fairlearn:
  selection rate            P(y_hat = positive | group)
  demographic parity diff   max - min selection rate across groups
  equal opportunity diff    max - min true-positive rate across groups
  equalized odds diff       max of (TPR diff, FPR diff)
0 means equal treatment; a common rule of thumb flags differences above 0.1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def _rate(mask: np.ndarray, cond: np.ndarray) -> float:
    return float(mask[cond].mean()) if cond.any() else float("nan")


def group_report(y_true, y_pred, sensitive: pd.Series, positive=1, min_group_size: int = 30,
                 threshold: float = 0.1) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    s = pd.Series(sensitive).astype(str).fillna("missing").to_numpy()
    yt, yp = y_true == positive, y_pred == positive
    groups = []
    for g in pd.unique(s):
        m = s == g
        groups.append({
            "group": g, "n": int(m.sum()),
            "selection_rate": round(_rate(yp, m), 4),
            "tpr": round(_rate(yp, m & yt), 4),
            "fpr": round(_rate(yp, m & ~yt), 4),
            "accuracy": round(float(accuracy_score(y_true[m], y_pred[m])), 4),
            "f1": round(float(f1_score(yt[m], yp[m], zero_division=0)), 4),
            "small_group": bool(m.sum() < min_group_size),
        })
    df = pd.DataFrame(groups)
    big = df[~df["small_group"]] if (~df["small_group"]).sum() >= 2 else df

    def spread(col):
        v = big[col].dropna()
        return round(float(v.max() - v.min()), 4) if len(v) >= 2 else 0.0

    dp, eo, fpr_d = spread("selection_rate"), spread("tpr"), spread("fpr")
    return {
        "groups": sorted(groups, key=lambda r: -r["n"]),
        "demographic_parity_difference": dp,
        "equal_opportunity_difference": eo,
        "equalized_odds_difference": max(eo, fpr_d),
        "flagged": bool(max(dp, eo) > threshold),
        "threshold": threshold,
        "note": "Groups with fewer than %d rows are shown but excluded from the differences." % min_group_size,
    }


def fairness_for_model(bundle: dict, test_raw: pd.DataFrame, target: str,
                       sensitive_cols: list[str], positive_label: str | None = None) -> dict:
    """Fairness of a saved model on its own held-out test rows, per sensitive column.
    Works even if the sensitive column was excluded from training."""
    le = bundle["label_encoder"]
    if le is None:
        raise ValueError("Fairness report is for classification models only.")
    classes = list(le.classes_)
    pos = positive_label if positive_label is not None else classes[-1]
    if pos not in classes:
        raise ValueError(f"Positive label '{pos}' not in {classes}.")
    y_true = test_raw[target].astype(str).to_numpy()
    pred = le.inverse_transform(bundle["pipeline"].predict(test_raw[bundle["features"]]).astype(int))
    out = {"positive_label": pos, "attributes": {}}
    for col in sensitive_cols:
        if col not in test_raw:
            raise ValueError(f"Column '{col}' not found in test data.")
        out["attributes"][col] = group_report(y_true, pred, test_raw[col], positive=pos)
    return out
