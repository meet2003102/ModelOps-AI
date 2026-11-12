"""Leakage-free preprocessing.

The preprocessor is returned UNFITTED and placed inside a full
Pipeline(preprocessor -> model). Cross-validation then refits it on each
training fold, so no statistic from validation/test rows leaks in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_engine.ingestion import profiling as P


class DateFeatures(BaseEstimator, TransformerMixin):
    """Datetime columns -> year, month, day of week."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for c in self.columns or []:
            d = pd.to_datetime(X[c], errors="coerce", format="mixed")
            out[f"{c}_year"] = d.dt.year
            out[f"{c}_month"] = d.dt.month
            out[f"{c}_dayofweek"] = d.dt.dayofweek
        return out.astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_{p}" for c in self.columns or [] for p in ("year", "month", "dayofweek")])


class ToStr(BaseEstimator, TransformerMixin):
    """Casts categorical columns to str (keeps NaN) so mixed types encode safely."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            X[c] = X[c].where(X[c].isna(), X[c].astype(str))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


@dataclass
class FeaturePlan:
    target: str
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    dropped: dict = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical + self.datetime

    def to_dict(self) -> dict:
        return {"target": self.target, "numeric": self.numeric, "categorical": self.categorical,
                "datetime": self.datetime, "dropped": self.dropped}


def plan_features(df: pd.DataFrame, target: str, profile: dict | None = None,
                  drop_missing_above: float = 50.0, exclude: list[str] | None = None) -> FeaturePlan:
    """Decide which columns are used and how. `exclude` = columns the user removes
    (e.g. sensitive attributes such as sex or race)."""
    types = (profile or P.profile_dataset(df))["column_types"]
    plan = FeaturePlan(target=target)
    for c in df.columns:
        if c == target:
            continue
        if exclude and c in exclude:
            plan.dropped[c] = "excluded by user"
            continue
        t = types.get(c) or P.infer_column_type(df[c])
        miss = df[c].isna().mean() * 100
        if t in (P.ID, P.CONSTANT, P.TEXT):
            plan.dropped[c] = f"{t} column"
        elif miss > drop_missing_above:
            plan.dropped[c] = f"{miss:.0f}% missing"
        elif t == P.NUMERIC:
            plan.numeric.append(c)
        elif t == P.DATETIME:
            plan.datetime.append(c)
        else:
            plan.categorical.append(c)
    if not plan.features:
        raise ValueError("No usable feature columns left after cleaning.")
    return plan


def build_preprocessor(plan: FeaturePlan, scale: bool = True, max_categories: int = 20) -> ColumnTransformer:
    num = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("scale", StandardScaler()))
    cat = [("to_str", ToStr()),
           ("impute", SimpleImputer(strategy="most_frequent")),
           ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                    max_categories=max_categories, sparse_output=False))]
    parts = []
    if plan.numeric:
        parts.append(("num", Pipeline(num), plan.numeric))
    if plan.categorical:
        parts.append(("cat", Pipeline(cat), plan.categorical))
    if plan.datetime:
        parts.append(("date", Pipeline([("extract", DateFeatures(plan.datetime)),
                                        ("impute", SimpleImputer(strategy="median"))]), plan.datetime))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def clean_rows(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Row-level cleaning that is safe before the split (no statistics learned)."""
    notes = []
    n0 = len(df)
    df = df.drop_duplicates()
    if len(df) < n0:
        notes.append(f"Removed {n0 - len(df)} duplicate rows.")
    n1 = len(df)
    df = df[df[target].notna()]
    if len(df) < n1:
        notes.append(f"Removed {n1 - len(df)} rows with missing target.")
    return df, notes
