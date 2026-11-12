"""Automatic exploratory data analysis (Phase 1).

profile_dataset(df) returns a plain dict so it can be shown in the UI,
saved as JSON, and later passed to the LLM-insights module (Phase 7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Column roles the platform understands
NUMERIC, CATEGORICAL, DATETIME, TEXT, ID, CONSTANT, BOOLEAN = (
    "numeric", "categorical", "datetime", "text", "id", "constant", "boolean",
)


def infer_column_type(s: pd.Series) -> str:
    """Guess the role of a column from its values."""
    n = len(s)
    non_null = s.dropna()
    nunique = non_null.nunique()

    if nunique <= 1:
        return CONSTANT
    if pd.api.types.is_bool_dtype(s) or (nunique == 2 and set(map(str, non_null.unique())) <= {"0", "1", "True", "False", "true", "false", "yes", "no", "Yes", "No"}):
        return BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(s):
        return DATETIME

    if pd.api.types.is_numeric_dtype(s):
        # Integer column with all-unique values and a name like "id" -> identifier
        if pd.api.types.is_integer_dtype(s) and nunique / max(len(non_null), 1) > 0.95 and n > 20:
            name = str(s.name).lower()
            if name == "id" or name.endswith("_id") or name.endswith("id") or name.startswith("id"):
                return ID
        return NUMERIC

    # Object / string columns
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        sample = non_null.astype(str).head(200)
        # Try parsing as dates
        if len(sample) > 0:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.9 and not sample.str.fullmatch(r"-?\d+(\.\d+)?").all():
                return DATETIME
        unique_ratio = nunique / max(len(non_null), 1)
        avg_len = sample.str.len().mean() if len(sample) else 0
        if unique_ratio > 0.95 and n > 20:
            return TEXT if avg_len > 30 else ID
        if avg_len > 50:
            return TEXT
        return CATEGORICAL
    return CATEGORICAL


def detect_problem_type(y: pd.Series) -> str:
    """'classification' or 'regression' for a chosen target column."""
    y = y.dropna()
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    nunique = y.nunique()
    if pd.api.types.is_float_dtype(y) and not np.allclose(y, y.round()):
        return "regression"
    # Few distinct integer values -> classes
    if nunique <= 20 and nunique / max(len(y), 1) < 0.05:
        return "classification"
    if nunique <= 2:
        return "classification"
    return "regression"


def outlier_count_iqr(s: pd.Series) -> int:
    s = s.dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def suggest_target(df: pd.DataFrame, col_types: dict) -> str | None:
    """Heuristic guess of the target column (user can override)."""
    keywords = ("target", "label", "class", "churn", "outcome", "y", "price", "survived", "default", "fraud")
    for col in df.columns:
        if str(col).lower() in keywords:
            return col
    usable = [c for c, t in col_types.items() if t in (NUMERIC, CATEGORICAL, BOOLEAN)]
    return usable[-1] if usable else None


def profile_dataset(df: pd.DataFrame) -> dict:
    """Full automatic profile of a dataset."""
    col_types = {c: infer_column_type(df[c]) for c in df.columns}
    missing = df.isna().sum()
    numeric_cols = [c for c, t in col_types.items() if t == NUMERIC]
    categorical_cols = [c for c, t in col_types.items() if t in (CATEGORICAL, BOOLEAN)]

    columns = []
    for c in df.columns:
        s = df[c]
        info = {
            "name": c,
            "dtype": str(s.dtype),
            "inferred_type": col_types[c],
            "missing": int(missing[c]),
            "missing_pct": round(float(missing[c]) / len(df) * 100, 2) if len(df) else 0.0,
            "unique": int(s.nunique(dropna=True)),
        }
        if col_types[c] == NUMERIC:
            info.update(
                mean=float(s.mean()), std=float(s.std()), min=float(s.min()),
                max=float(s.max()), skew=float(s.skew()), outliers=outlier_count_iqr(s),
            )
        elif col_types[c] in (CATEGORICAL, BOOLEAN):
            top = s.value_counts(dropna=True).head(5)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        columns.append(info)

    # Strongest correlations between numeric features
    top_corr = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = corr.where(mask).stack().sort_values(ascending=False)
        top_corr = [
            {"a": a, "b": b, "corr": round(float(v), 3)} for (a, b), v in pairs.head(10).items()
        ]

    warnings = build_warnings(df, columns, top_corr)

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_pct": round(float(missing.sum()) / max(df.size, 1) * 100, 2),
        "column_types": col_types,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "columns": columns,
        "top_correlations": top_corr,
        "suggested_target": suggest_target(df, col_types),
        "warnings": warnings,
    }


def build_warnings(df: pd.DataFrame, columns: list[dict], top_corr: list[dict]) -> list[str]:
    w = []
    dup = int(df.duplicated().sum())
    if dup:
        w.append(f"{dup} duplicate rows found - they will be removed during preprocessing.")
    for c in columns:
        if c["missing_pct"] > 50:
            w.append(f"'{c['name']}' is {c['missing_pct']}% missing - consider dropping it.")
        elif c["missing_pct"] > 0:
            w.append(f"'{c['name']}' has {c['missing_pct']}% missing values - will be imputed.")
        if c["inferred_type"] in (ID, CONSTANT):
            w.append(f"'{c['name']}' looks like a {c['inferred_type']} column - it will be dropped.")
        if c["inferred_type"] == NUMERIC and abs(c.get("skew", 0)) > 2:
            w.append(f"'{c['name']}' is highly skewed (skew={c['skew']:.1f}).")
        if c["inferred_type"] == CATEGORICAL and c["unique"] > 50:
            w.append(f"'{c['name']}' has high cardinality ({c['unique']} categories).")
    for p in top_corr:
        if p["corr"] > 0.95:
            w.append(f"'{p['a']}' and '{p['b']}' are almost identical (corr={p['corr']}).")
    return w


def target_summary(df: pd.DataFrame, target: str) -> dict:
    """Problem type and class balance / distribution of the target."""
    y = df[target]
    ptype = detect_problem_type(y)
    out = {"target": target, "problem_type": ptype, "missing": int(y.isna().sum())}
    if ptype == "classification":
        counts = y.value_counts(dropna=True)
        out["n_classes"] = int(len(counts))
        out["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        out["imbalance_ratio"] = round(float(counts.max() / counts.min()), 2) if len(counts) > 1 else None
    else:
        out.update(mean=float(y.mean()), std=float(y.std()), min=float(y.min()), max=float(y.max()))
    return out
