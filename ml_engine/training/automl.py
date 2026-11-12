"""Module 3 - AutoML engine.

1. clean rows -> 2. stratified train/test split -> 3. Pipeline(preprocessor, model)
4. 5-fold CV for every candidate on TRAIN only -> 5. tune the top-k with Optuna
6. pick the best by the user's metric -> 7. refit on all train -> 8. score on test once.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from ml_engine import config
from ml_engine.evaluation import metrics as EM
from ml_engine.ingestion import profiling
from ml_engine.preprocessing.pipeline_builder import (FeaturePlan, build_preprocessor,
                                                      clean_rows, plan_features)
from ml_engine.training import model_zoo
from ml_engine.training.tuner import tune

# the majority baseline never predicts some classes -> harmless precision warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

Progress = Callable[[float, str], None]


@dataclass
class AutoMLResult:
    task: str
    metric: str
    best_model_name: str
    pipeline: Pipeline
    label_encoder: LabelEncoder | None
    plan: FeaturePlan
    leaderboard: list[dict]
    test_metrics: dict
    best_params: dict
    notes: list[str] = field(default_factory=list)
    # kept for SHAP, drift reference and champion-challenger comparison
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_test: np.ndarray | None = None
    test_raw: pd.DataFrame | None = None   # all original columns of the test rows (fairness)
    train_seconds: float = 0.0

    def summary(self) -> dict:
        return {"task": self.task, "metric": self.metric, "best_model": self.best_model_name,
                "best_params": self.best_params, "test_metrics": self.test_metrics,
                "leaderboard": self.leaderboard, "features": self.plan.to_dict(),
                "classes": list(map(str, self.label_encoder.classes_)) if self.label_encoder else None,
                "notes": self.notes, "train_seconds": round(self.train_seconds, 1)}


def _cv(task: str, y, folds: int):
    if task == "classification" and pd.Series(y).value_counts().min() >= folds:
        return StratifiedKFold(folds, shuffle=True, random_state=config.RANDOM_STATE)
    return KFold(folds, shuffle=True, random_state=config.RANDOM_STATE)


def run_automl(df: pd.DataFrame, target: str, task: str | None = None, metric: str | None = None,
               models: list[str] | None = None, tune_top_k: int = 2, n_trials: int = 20,
               exclude: list[str] | None = None, cv_folds: int = config.CV_FOLDS,
               progress: Progress | None = None) -> AutoMLResult:
    t0 = time.time()
    say = progress or (lambda p, m: None)

    df, notes = clean_rows(df, target)
    task = task or profiling.detect_problem_type(df[target])
    metric = metric or config.DEFAULT_METRIC[task]
    scorers = EM.scorers_for(task)
    if metric not in scorers:
        raise ValueError(f"Metric '{metric}' not valid for {task}. Choose from {list(scorers)}.")
    if task == "classification" and metric == "roc_auc" and df[target].nunique() == 2:
        scorers = {**scorers, "roc_auc": ("roc_auc", True)}
    primary = scorers[metric][0]

    plan = plan_features(df, target, exclude=exclude)
    if plan.dropped:
        notes.append("Dropped: " + ", ".join(f"{k} ({v})" for k, v in plan.dropped.items()))
    X = df[plan.features]
    y_raw = df[target]

    le = None
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.astype(float).to_numpy()

    strat = y if task == "classification" and np.bincount(y).min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.TEST_SIZE,
                                              random_state=config.RANDOM_STATE, stratify=strat)
    notes.append(f"Train/test: {len(X_tr)}/{len(X_te)} rows" + (" (stratified)" if strat is not None else ""))
    cv = _cv(task, y_tr, cv_folds)
    say(0.05, "Data split done")

    # ---- 1) cross-validate every candidate with default params
    candidates = model_zoo.get_models(task, include=models)
    scoring = {k: v[0] for k, v in scorers.items()}
    board = []
    for i, (name, (est, space)) in enumerate(candidates.items()):
        linear = any(s in name for s in ("Logistic", "Linear", "Ridge"))
        pipe = Pipeline([("pre", build_preprocessor(plan, scale=linear)), ("model", clone(est))])
        s = time.time()
        try:
            cvr = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, error_score="raise")
        except Exception as e:  # a model failing must not stop the run
            notes.append(f"{name} failed: {e}")
            continue
        row = {"model": name, "fit_seconds": round(time.time() - s, 2), "tuned": False}
        for m, (_, higher) in scorers.items():
            vals = cvr[f"test_{m}"] * (1 if higher else -1)
            row[m] = round(float(vals.mean()), 4)
            row[f"{m}_std"] = round(float(vals.std()), 4)
        row["_pipe"], row["_space"] = pipe, space
        board.append(row)
        say(0.05 + 0.5 * (i + 1) / len(candidates), f"Cross-validated {name}")

    if not board:
        raise RuntimeError("All models failed to train.")
    higher = scorers[metric][1]
    board.sort(key=lambda r: r[metric], reverse=higher)

    # ---- 2) tune the top-k real models (not the baseline)
    best_params_by_model = {}
    to_tune = [r for r in board if not r["model"].startswith("Baseline") and r["_space"]][:tune_top_k]
    for j, row in enumerate(to_tune):
        say(0.55 + 0.35 * j / max(len(to_tune), 1), f"Tuning {row['model']}")
        params, score = tune(row["_pipe"], row["_space"], X_tr, y_tr, cv, primary, n_trials=n_trials)
        tuned_score = score if higher else -score
        if (tuned_score > row[metric]) == higher and tuned_score != row[metric]:
            tuned_pipe = clone(row["_pipe"]).set_params(**params)
            # re-run CV once so every metric and its std describe the tuned model
            cvr = cross_validate(tuned_pipe, X_tr, y_tr, cv=cv, scoring=scoring)
            for m, (_, hib) in scorers.items():
                vals = cvr[f"test_{m}"] * (1 if hib else -1)
                row[m] = round(float(vals.mean()), 4)
                row[f"{m}_std"] = round(float(vals.std()), 4)
            row["tuned"] = True
            best_params_by_model[row["model"]] = {k.replace("model__", ""): v for k, v in params.items()}
            row["_pipe"] = tuned_pipe
    board.sort(key=lambda r: r[metric], reverse=higher)

    # ---- 3) refit the winner on all training data, evaluate once on test
    best = board[0]
    if best["model"].startswith("Baseline") and len(board) > 1:
        notes.append("Warning: no model beat the baseline.")
    say(0.92, f"Refitting {best['model']}")
    pipe = best["_pipe"].fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    if task == "classification":
        proba = pipe.predict_proba(X_te) if hasattr(pipe, "predict_proba") else None
        test = EM.classification_metrics(y_te, pred, proba, labels=list(range(len(le.classes_))))
        if proba is not None and proba.shape[1] == 2:
            test["calibration"] = EM.calibration(y_te, proba[:, 1])
    else:
        test = EM.regression_metrics(y_te, pred)

    leaderboard = [{k: v for k, v in r.items() if not k.startswith("_")} for r in board]
    for i, r in enumerate(leaderboard, 1):
        r["rank"] = i
    say(1.0, "Done")
    return AutoMLResult(task, metric, best["model"], pipe, le, plan, leaderboard, test,
                        best_params_by_model.get(best["model"], {}), notes,
                        X_tr, X_te, y_te, df.loc[X_te.index], time.time() - t0)
