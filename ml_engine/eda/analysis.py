"""Module 2 - automated EDA. Returns JSON-ready data for the frontend charts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml_engine.ingestion import profiling


def _hist(s: pd.Series, bins: int = 30) -> dict:
    s = s.dropna()
    if s.empty:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(s, bins=bins)
    return {"bins": [round(float(e), 4) for e in edges], "counts": counts.tolist()}


def run_eda(df: pd.DataFrame, target: str | None = None, profile: dict | None = None,
            max_categories: int = 20) -> dict:
    profile = profile or profiling.profile_dataset(df)
    types = profile["column_types"]
    num = profile["numeric_columns"]
    cat = profile["categorical_columns"]

    numeric_dist = {c: {**_hist(df[c]),
                        "outliers": profiling.outlier_count_iqr(df[c])} for c in num}
    categorical_dist = {
        c: {str(k): int(v) for k, v in df[c].astype(str).value_counts().head(max_categories).items()}
        for c in cat}

    corr = None
    if len(num) >= 2:
        m = df[num].corr().round(3)
        corr = {"columns": num, "matrix": m.fillna(0).values.tolist()}

    missing = {c: round(float(df[c].isna().mean() * 100), 2) for c in df.columns if df[c].isna().any()}
    outliers = {c: v["outliers"] for c, v in numeric_dist.items() if v["outliers"] > 0}

    out = {
        "numeric_distributions": numeric_dist,
        "categorical_distributions": categorical_dist,
        "correlation": corr,
        "missing_pct": missing,
        "outliers": outliers,
        "column_types": types,
    }
    if target:
        t = profiling.target_summary(df, target)
        if t["problem_type"] == "regression":
            t["distribution"] = _hist(df[target])
        out["target"] = t
    return out
