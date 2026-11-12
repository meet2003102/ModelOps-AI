"""Champion-challenger retraining for saved tabular models.

Deterministic promotion rule (the LLM is never involved):
  1. New labelled data is split by time: the most recent `holdout_share` is held out.
  2. A challenger of the same model type is trained on the older part with AutoML.
  3. Champion and challenger are scored on the SAME holdout rows.
  4. The challenger is promoted only if it beats the champion by at least `min_gain`
     on the model's selection metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml_engine.evaluation import metrics as EM
from ml_engine.registry import model_registry as reg
from ml_engine.training.automl import run_automl


def score(bundle_or_pipe, X: pd.DataFrame, y_raw: pd.Series, task: str, metric: str,
          label_encoder=None) -> float:
    pipe = bundle_or_pipe["pipeline"] if isinstance(bundle_or_pipe, dict) else bundle_or_pipe
    le = bundle_or_pipe["label_encoder"] if isinstance(bundle_or_pipe, dict) else label_encoder
    if task == "classification":
        known = y_raw.astype(str).isin(le.classes_)
        X, y_raw = X[known], y_raw[known]
        y = le.transform(y_raw.astype(str))
        proba = pipe.predict_proba(X) if hasattr(pipe, "predict_proba") else None
        m = EM.classification_metrics(y, pipe.predict(X), proba, labels=list(range(len(le.classes_))))
    else:
        m = EM.regression_metrics(y_raw.astype(float), pipe.predict(X))
    return float(m[metric])


def decide(champion_score: float, challenger_score: float, metric: str, min_gain: float = 0.01) -> dict:
    higher = metric not in ("rmse", "mae")
    gain = (challenger_score - champion_score) if higher else (champion_score - challenger_score)
    return {"metric": metric, "champion": round(champion_score, 4), "challenger": round(challenger_score, 4),
            "gain": round(gain, 4), "min_gain": min_gain, "promote": bool(gain >= min_gain)}


def retrain_and_compare(champion_id: str, new_labelled: pd.DataFrame, holdout_share: float = 0.25,
                        min_gain: float = 0.01, n_trials: int = 15, auto_promote: bool = True,
                        root=None) -> dict:
    champ = reg.load_model(champion_id, root)
    meta = champ["metadata"]
    target, task, metric = meta["features"]["target"], meta["task"], meta["metric"]
    if target not in new_labelled:
        raise ValueError(f"New data needs the true label column '{target}'.")
    n_hold = max(int(len(new_labelled) * holdout_share), 30)
    train_part, hold = new_labelled.iloc[:-n_hold], new_labelled.iloc[-n_hold:]

    res = run_automl(train_part, target, task=task, metric=metric, models=[meta["best_model"]],
                     tune_top_k=1, n_trials=n_trials)
    feats = champ["features"]
    s_champ = score(champ, hold[feats], hold[target], task, metric)
    s_chall = score(res.pipeline, hold[res.plan.features], hold[target], task, metric, res.label_encoder)
    decision = decide(s_champ, s_chall, metric, min_gain)
    challenger_id = reg.save_model(res, meta.get("dataset", ""), root=root)
    if decision["promote"] and auto_promote:
        reg.set_status(challenger_id, "champion", root)
    else:
        reg.set_status(challenger_id, "candidate", root)
    return {"champion_id": champion_id, "challenger_id": challenger_id, "holdout_rows": int(n_hold),
            **decision, "promoted": bool(decision["promote"] and auto_promote)}
