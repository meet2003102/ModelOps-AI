"""Module 4 - explainable AI.

Explains a fitted Pipeline(pre -> model) with SHAP and maps every one-hot /
date column back to the ORIGINAL feature, so users see "contract", not
"contract_Month-to-month".

Explainer choice:  tree models -> TreeExplainer, linear -> LinearExplainer,
anything else -> model-agnostic Permutation explainer.
If SHAP is not installed, a slower model-agnostic fallback is used
(permutation importance for global, occlusion for local) and the output
says method="fallback".

All text in `explanation_text` is generated from the numbers by a fixed
template - never by an LLM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ml_engine.config import RANDOM_STATE

try:
    import shap
    HAS_SHAP = True
except ImportError:  # pragma: no cover
    HAS_SHAP = False

TREE_HINTS = ("Forest", "Tree", "XGB", "LGBM", "GradientBoosting", "Boost")
LINEAR_HINTS = ("Logistic", "Linear", "Ridge", "Lasso")


# ---------------------------------------------------------------- helpers
def _split(pipeline):
    return pipeline.named_steps["pre"], pipeline.named_steps["model"]


def feature_mapping(pre) -> tuple[list[str], list[str]]:
    """(transformed column names, original column for each)."""
    names, origin = [], []
    for _, trans, cols in pre.transformers_:
        if trans == "drop" or not len(cols):
            continue
        out = list(trans.get_feature_names_out(cols))
        for o in out:
            # longest original column name that prefixes the output name
            match = max((c for c in cols if o == c or o.startswith(f"{c}_")), key=len, default=o)
            names.append(o)
            origin.append(match)
    return names, origin


def _aggregate(values: np.ndarray, origin: list[str]) -> pd.DataFrame:
    """Sum SHAP values of transformed columns that come from the same original feature."""
    df = pd.DataFrame(values, columns=range(values.shape[1]))
    groups = pd.Series(origin)
    return df.T.groupby(groups.values).sum().T


def _model_kind(model) -> str:
    n = type(model).__name__
    if any(h in n for h in TREE_HINTS):
        return "tree"
    if any(h in n for h in LINEAR_HINTS):
        return "linear"
    if n.startswith("Dummy"):
        return "dummy"
    return "other"


def _pick_class(arr: np.ndarray, class_idx: int) -> np.ndarray:
    """SHAP returns (n, f) or (n, f, classes) depending on model/version."""
    return arr[..., class_idx] if arr.ndim == 3 else arr


# ---------------------------------------------------------------- SHAP core
def _shap_values(pipeline, X: pd.DataFrame, background: pd.DataFrame, task: str):
    pre, model = _split(pipeline)
    Xt = pre.transform(X)
    Bt = pre.transform(background)
    kind = _model_kind(model)
    if kind == "tree":
        explainer = shap.TreeExplainer(model)
        exp = explainer(Xt, check_additivity=False)
    elif kind == "linear":
        explainer = shap.LinearExplainer(model, Bt)
        exp = explainer(Xt)
    else:
        if task == "classification":
            f = model.predict_proba
        else:
            f = model.predict
        explainer = shap.Explainer(f, Bt, algorithm="permutation", seed=RANDOM_STATE)
        exp = explainer(Xt)
    return np.asarray(exp.values), np.asarray(exp.base_values), kind


def _class_index(task, n_classes, pred_class=None):
    if task != "classification":
        return 0
    if n_classes == 2:
        return 1          # explain the positive class
    return int(pred_class or 0)


# ---------------------------------------------------------------- public API
def explain_global(pipeline, X: pd.DataFrame, task: str, n_classes: int = 2,
                   max_rows: int = 1000, top_k: int = 15) -> dict:
    """Global importance (mean |SHAP| per original feature) + beeswarm data."""
    X = X.sample(min(max_rows, len(X)), random_state=RANDOM_STATE)
    pre, model = _split(pipeline)
    _, origin = feature_mapping(pre)

    if HAS_SHAP and _model_kind(model) != "dummy":
        bg = X.sample(min(100, len(X)), random_state=RANDOM_STATE)
        vals, _, kind = _shap_values(pipeline, X, bg, task)
        if vals.ndim == 3 and n_classes > 2:
            vals = np.abs(vals).mean(axis=2)            # multiclass: average over classes
        else:
            vals = _pick_class(vals, _class_index(task, n_classes))
        agg = _aggregate(vals, origin)
        importance = agg.abs().mean().sort_values(ascending=False)
        top = list(importance.index[:top_k])
        # beeswarm: SHAP value + standardised feature value for each sample (numeric feats)
        beeswarm = {}
        for f in top:
            fv = X[f] if f in X else None
            if fv is not None and pd.api.types.is_numeric_dtype(fv):
                z = (fv - fv.mean()) / (fv.std() or 1)
                beeswarm[f] = {"shap": agg[f].round(4).tolist()[:300],
                               "feature_value_z": z.round(3).fillna(0).tolist()[:300]}
            else:
                beeswarm[f] = {"shap": agg[f].round(4).tolist()[:300]}
        method = f"shap-{kind}"
    else:
        # fallback: permutation importance on original columns
        y_dummy = pipeline.predict(X)
        r = permutation_importance(pipeline, X, y_dummy, n_repeats=5, random_state=RANDOM_STATE)
        importance = pd.Series(r.importances_mean, index=X.columns).clip(lower=0).sort_values(ascending=False)
        top, beeswarm, method = list(importance.index[:top_k]), {}, "fallback-permutation"

    total = float(importance.sum()) or 1.0
    return {
        "method": method,
        "n_rows": int(len(X)),
        "importance": [{"feature": f, "mean_abs_shap": round(float(importance[f]), 5),
                        "share": round(float(importance[f]) / total, 4)} for f in importance.index[:top_k]],
        "beeswarm": beeswarm,
    }


def explain_local(pipeline, row: pd.DataFrame, background: pd.DataFrame, task: str,
                  classes: list[str] | None = None, top_k: int = 8) -> dict:
    """Why did the model predict this for ONE row?"""
    row = row.iloc[[0]]
    pre, model = _split(pipeline)
    _, origin = feature_mapping(pre)
    n_classes = len(classes) if classes else 2

    if task == "classification":
        proba = pipeline.predict_proba(row)[0]
        pred_idx = int(np.argmax(proba))
        cls_idx = _class_index(task, n_classes, pred_idx)
        prediction = classes[pred_idx] if classes else pred_idx
        explained = classes[cls_idx] if classes else cls_idx
        score = float(proba[cls_idx])
    else:
        prediction = float(pipeline.predict(row)[0])
        explained, score, cls_idx = None, prediction, 0

    if HAS_SHAP and _model_kind(model) != "dummy":
        bg = background.sample(min(100, len(background)), random_state=RANDOM_STATE)
        vals, base, kind = _shap_values(pipeline, row, bg, task)
        vals = _pick_class(vals, cls_idx)
        base = np.ravel(base)
        base_value = float(base[cls_idx] if base.size > 1 else base[0])
        contrib = _aggregate(vals, origin).iloc[0]
        method = f"shap-{kind}"
    else:
        contrib = _occlusion(pipeline, row, background, task, cls_idx)
        base_value, method = None, "fallback-occlusion"

    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    items = [{"feature": f, "value": _fmt(row[f].iloc[0]) if f in row else None,
              "contribution": round(float(c), 5)} for f, c in contrib.head(top_k).items()]
    out = {"method": method, "task": task, "prediction": _fmt(prediction),
           "explained_class": explained, "score": round(score, 4),
           "base_value": None if base_value is None else round(base_value, 4),
           "contributions": items}
    out["explanation_text"] = explanation_text(out)
    return out


def _occlusion(pipeline, row, background, task, cls_idx) -> pd.Series:
    """Fallback local explanation: replace one feature at a time with a typical value."""
    def score(df):
        return pipeline.predict_proba(df)[:, cls_idx] if task == "classification" else pipeline.predict(df)
    base = score(row)[0]
    out = {}
    for c in row.columns:
        typical = background[c].median() if pd.api.types.is_numeric_dtype(background[c]) else background[c].mode().iloc[0]
        r = row.copy()
        r[c] = typical
        out[c] = base - score(r)[0]
    return pd.Series(out)


def _fmt(v):
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v if v is None or isinstance(v, (int, str, bool)) else str(v)


def explanation_text(e: dict, n: int = 2) -> str:
    """Deterministic sentence built only from the computed contributions."""
    biggest = max((abs(c["contribution"]) for c in e["contributions"]), default=0)
    cut = 0.05 * biggest          # ignore features with under 5% of the largest effect
    up = [c["feature"] for c in e["contributions"] if c["contribution"] > cut][:n]
    down = [c["feature"] for c in e["contributions"] if c["contribution"] < -cut][:n]
    if e["task"] == "classification":
        head = (f"The model predicted '{e['prediction']}'. Probability of '{e['explained_class']}' "
                f"is {e['score']:.0%}.")
        up_s = f" {', '.join(up)} pushed it towards '{e['explained_class']}'." if up else ""
        down_s = f" {', '.join(down)} pushed it away." if down else ""
    else:
        head = f"The model predicted {e['prediction']:.4g}."
        up_s = f" {', '.join(up)} increased the prediction." if up else ""
        down_s = f" {', '.join(down)} decreased it." if down else ""
    return head + up_s + down_s


def explanation_stability(pipeline, row: pd.DataFrame, background: pd.DataFrame, task: str,
                          classes=None, noise: float = 0.01, repeats: int = 10, top_k: int = 5) -> float:
    """Share of top-k features that stay in the top-k when numeric inputs get small noise
    (1.0 = perfectly stable). Used in the XAI evaluation."""
    rng = np.random.default_rng(RANDOM_STATE)
    ref = {c["feature"] for c in explain_local(pipeline, row, background, task, classes, top_k)["contributions"][:top_k]}
    scores = []
    for _ in range(repeats):
        r = row.copy()
        for c in r.columns:
            if pd.api.types.is_numeric_dtype(r[c]) and pd.notna(r[c].iloc[0]):
                sd = background[c].std() if c in background else 0
                r[c] = r[c] + rng.normal(0, noise * (sd or 1))
        got = {c["feature"] for c in explain_local(pipeline, r, background, task, classes, top_k)["contributions"][:top_k]}
        scores.append(len(ref & got) / max(len(ref), 1))
    return round(float(np.mean(scores)), 4)
