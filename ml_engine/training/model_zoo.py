"""Candidate models and their hyperparameter search spaces.

XGBoost and LightGBM are used when installed and skipped otherwise, so the
platform still runs on a minimal environment.

Search space format (works for Optuna and for random search):
  ("int", low, high) | ("float", low, high, log) | ("cat", [choices])
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

from ml_engine.config import RANDOM_STATE

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:  # pragma: no cover
    HAS_LGBM = False

RS = RANDOM_STATE

_BOOST_SPACE = {
    "n_estimators": ("int", 100, 500),
    "learning_rate": ("float", 0.01, 0.3, True),
    "max_depth": ("int", 3, 10),
    "subsample": ("float", 0.6, 1.0, False),
}


def classification_models() -> dict:
    m = {
        "Baseline (majority)": (DummyClassifier(strategy="most_frequent"), {}),
        "Logistic Regression": (LogisticRegression(max_iter=2000),
                                {"C": ("float", 1e-3, 100, True)}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RS),
                          {"max_depth": ("int", 2, 20), "min_samples_leaf": ("int", 1, 50)}),
        "Random Forest": (RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RS),
                          {"n_estimators": ("int", 100, 500), "max_depth": ("int", 3, 30),
                           "min_samples_leaf": ("int", 1, 20), "max_features": ("cat", ["sqrt", "log2", None])}),
    }
    if HAS_XGB:
        m["XGBoost"] = (XGBClassifier(eval_metric="logloss", random_state=RS, n_jobs=-1), dict(_BOOST_SPACE))
    if HAS_LGBM:
        m["LightGBM"] = (LGBMClassifier(random_state=RS, verbose=-1, n_jobs=-1),
                         {**_BOOST_SPACE, "num_leaves": ("int", 15, 127)})
    return m


def regression_models() -> dict:
    m = {
        "Baseline (mean)": (DummyRegressor(strategy="mean"), {}),
        "Linear Regression": (LinearRegression(), {}),
        "Ridge Regression": (Ridge(), {"alpha": ("float", 1e-3, 100, True)}),
        "Random Forest Regressor": (RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RS),
                                    {"n_estimators": ("int", 100, 500), "max_depth": ("int", 3, 30),
                                     "min_samples_leaf": ("int", 1, 20)}),
        "Gradient Boosting Regressor": (GradientBoostingRegressor(random_state=RS), dict(_BOOST_SPACE)),
    }
    if HAS_XGB:
        m["XGBoost Regressor"] = (XGBRegressor(random_state=RS, n_jobs=-1), dict(_BOOST_SPACE))
    if HAS_LGBM:
        m["LightGBM Regressor"] = (LGBMRegressor(random_state=RS, verbose=-1, n_jobs=-1),
                                   {**_BOOST_SPACE, "num_leaves": ("int", 15, 127)})
    return m


def get_models(task: str, include: list[str] | None = None) -> dict:
    models = classification_models() if task == "classification" else regression_models()
    if include:
        models = {k: v for k, v in models.items() if k in include or k.startswith("Baseline")}
    return models
