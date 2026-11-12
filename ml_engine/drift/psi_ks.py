"""Module 5A - data drift: does the INPUT distribution differ from the training reference?

Per feature:
  numeric      PSI (10 quantile bins from the reference) + two-sample KS test
  categorical  PSI on category shares + chi-square test
A feature is flagged when PSI >= 0.2, or when the test is significant (p < alpha)
AND PSI >= 0.1. Requiring both avoids the well-known problem that KS/chi-square
flag tiny, harmless shifts on large samples.

PSI guide: < 0.1 no change, 0.1-0.2 moderate, >= 0.2 major.
Data drift alone does NOT prove the model got worse - that needs labels (concept drift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-4


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna().astype(float), cur.dropna().astype(float)
    if ref.empty or cur.empty:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # almost constant column
        edges = np.unique(np.concatenate([edges, [ref.min() - 1, ref.max() + 1]]))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    ref, cur = ref.astype(str), cur.astype(str)
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _severity(psi: float) -> str:
    return "major" if psi >= 0.2 else "moderate" if psi >= 0.1 else "none"


def _as_text(s: pd.Series) -> pd.Series:
    return s.astype("object").where(s.notna(), "__missing__").astype(str)


def _as_dates(s: pd.Series) -> pd.Series | None:
    """Days since epoch if the column is mostly dates, else None."""
    if pd.api.types.is_numeric_dtype(s):
        return None
    d = pd.to_datetime(s, errors="coerce", format="mixed")
    if d.notna().mean() < 0.9:
        return None
    return (d - pd.Timestamp("1970-01-01")).dt.days.astype(float)


def feature_drift(ref: pd.Series, cur: pd.Series, alpha: float = 0.05) -> dict:
    dr, dc = _as_dates(ref), _as_dates(cur)
    if dr is not None and dc is not None:      # compare dates on a time axis
        ref, cur = dr, dc
    numeric = pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur)
    if not numeric:
        ref, cur = _as_text(ref), _as_text(cur)
    if numeric:
        psi = psi_numeric(ref, cur)
        stat, p = stats.ks_2samp(ref.dropna(), cur.dropna())
        test = "ks"
    else:
        psi = psi_categorical(ref, cur)
        cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
        table = np.array([ref.astype(str).value_counts().reindex(cats, fill_value=0),
                          cur.astype(str).value_counts().reindex(cats, fill_value=0)])
        table = table[:, table.sum(axis=0) > 0]
        stat, p = (stats.chi2_contingency(table)[:2] if table.shape[1] > 1 else (0.0, 1.0))
        test = "chi2"
    drifted = psi >= 0.2 or (p < alpha and psi >= 0.1)
    return {"type": "numeric" if numeric else "categorical", "psi": round(psi, 4),
            "test": test, "statistic": round(float(stat), 4), "p_value": round(float(p), 6),
            "severity": _severity(psi), "drifted": bool(drifted),
            "ref_mean": round(float(ref.mean()), 4) if numeric else None,
            "cur_mean": round(float(cur.mean()), 4) if numeric else None,
            "missing_ref_pct": round(float(ref.isna().mean() * 100), 2),
            "missing_cur_pct": round(float(cur.isna().mean() * 100), 2)}


def data_drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str],
                      importance: dict[str, float] | None = None, alpha: float = 0.05,
                      dataset_share: float = 0.3) -> dict:
    """Drift for every feature, plus a SHAP-weighted 'impact' ranking.

    importance: {feature: share of global SHAP importance} from Step 2. Drift in a feature
    the model barely uses matters less than drift in its top driver; impact = PSI x share.
    """
    missing = [f for f in features if f not in current.columns]
    if missing:
        raise ValueError(f"Current data is missing columns: {missing}")
    rows = {f: feature_drift(reference[f], current[f], alpha) for f in features}
    if importance:
        for f, r in rows.items():
            r["importance_share"] = round(float(importance.get(f, 0.0)), 4)
            r["impact"] = round(r["psi"] * r["importance_share"], 5)
    drifted = [f for f, r in rows.items() if r["drifted"]]
    share = len(drifted) / max(len(features), 1)
    key = "impact" if importance else "psi"
    ranking = sorted(rows, key=lambda f: rows[f][key], reverse=True)
    return {
        "n_reference": int(len(reference)), "n_current": int(len(current)),
        "features": rows, "drifted_features": drifted,
        "share_drifted": round(share, 4),
        "dataset_drift": bool(share >= dataset_share),
        "ranking_by": key, "ranking": ranking,
        "top_important_drifted": [f for f in ranking if rows[f]["drifted"]][:5],
    }
