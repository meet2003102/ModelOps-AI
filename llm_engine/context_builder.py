"""Collect the FACTS the LLM may talk about, as compact JSON.

Only measured outputs of the platform go in: model metadata, test metrics,
leaderboard, SHAP importance, the latest drift / concept-drift / retraining /
fairness reports saved next to the model. Raw data never goes to the LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

from ml_engine import config
from ml_engine.registry import model_registry as reg

REPORT_FILES = {
    "global_explanation": "global_explanation.json",
    "data_drift": "latest_data_drift.json",
    "concept_drift": "latest_concept_drift.json",
    "retraining": "latest_retraining.json",
    "fairness": "fairness.json",
}


def model_dir(model_id: str) -> Path:
    return Path(config.ARTIFACTS_DIR) / "models" / model_id


def save_report(model_id: str, kind: str, data: dict) -> None:
    """Called by the API/scripts whenever a report is produced, so insights can use it."""
    (model_dir(model_id) / REPORT_FILES[kind]).write_text(json.dumps(data, indent=2, default=str))


def _load(model_id: str, kind: str) -> dict | None:
    p = model_dir(model_id) / REPORT_FILES[kind]
    return json.loads(p.read_text()) if p.exists() else None


def _r(v, n=4):
    return round(v, n) if isinstance(v, float) else v


def build_facts(model_id: str) -> dict:
    meta = reg.load_model(model_id)["metadata"]
    numeric_test = {k: _r(v) for k, v in meta["test_metrics"].items() if isinstance(v, (int, float))}
    facts = {
        "scope": {"dataset": meta.get("dataset"), "model_version": model_id,
                  "covers": "only this model version and its latest reports",
                  "not_covered": "earlier time periods, other datasets, business or financial impact"},
        "model": {"id": model_id, "algorithm": meta["best_model"], "task": meta["task"],
                  "dataset": meta.get("dataset"), "status": meta.get("status"),
                  "selection_metric": meta["metric"], "classes": meta.get("classes"),
                  "target": meta["features"]["target"],
                  "n_features_used": len(meta["features"]["numeric"]) + len(meta["features"]["categorical"])
                  + len(meta["features"]["datetime"]),
                  "dropped_columns": meta["features"]["dropped"],
                  "dropped_column_names": list(meta["features"]["dropped"])},
        "test_metrics": numeric_test,
        "leaderboard_cv": [{"rank": r["rank"], "model": r["model"], meta["metric"]: r[meta["metric"]],
                            "std": r.get(f"{meta['metric']}_std"), "tuned": r.get("tuned")}
                           for r in meta["leaderboard"][:5]],
        "data_quality_notes": meta.get("notes", []),
    }
    g = _load(model_id, "global_explanation")
    if g:
        facts["top_features"] = [{"rank": r, "feature": i["feature"], "importance_share": i["share"]}
                                 for r, i in enumerate(g["importance"][:8], 1)]
        facts["feature_ranking_summary"] = {
            "most_important": facts["top_features"][0]["feature"],
            "least_important_listed": facts["top_features"][-1]["feature"],
            "n_listed": len(facts["top_features"])}
    d = _load(model_id, "data_drift")
    if d:
        drifted = sorted((f for f in d["features"] if d["features"][f]["drifted"]),
                         key=lambda f: d["features"][f]["psi"], reverse=True)
        facts["data_drift"] = {
            "dataset_drift": d["dataset_drift"], "share_drifted": d["share_drifted"],
            "share_drifted_percent": f"{d['share_drifted'] * 100:.2f}%",
            "n_drifted": len(drifted), "n_features_checked": len(d["features"]),
            "n_reference": d["n_reference"], "n_current": d["n_current"],
            "largest_psi_feature": drifted[0] if drifted else None,
            "drifted_features_sorted_by_psi": [
                {"rank": r, "feature": f, "psi": d["features"][f]["psi"],
                 "severity": d["features"][f]["severity"],
                 "importance_share": d["features"][f].get("importance_share")}
                for r, f in enumerate(drifted[:8], 1)]}
    c = _load(model_id, "concept_drift")
    if c and c.get("concept_drift") is not None:
        facts["concept_drift"] = {"detected": c["concept_drift"], "detector": c["detector"],
                                  "labelled_rows": c["n"], "error_rate": c["error_rate"],
                                  "first_alarm_at_row": c["alarms"][0] if c["alarms"] else None}
    rt = _load(model_id, "retraining")
    if rt:
        facts["last_retraining"] = {k: rt[k] for k in ("metric", "champion", "challenger", "gain",
                                                       "min_gain", "promoted", "challenger_id") if k in rt}
    f = _load(model_id, "fairness")
    if f:
        facts["fairness"] = {col: {"demographic_parity_difference": r["demographic_parity_difference"],
                                   "equal_opportunity_difference": r["equal_opportunity_difference"],
                                   "flagged": r["flagged"]} for col, r in f["attributes"].items()}
    optional = {"top_features": "feature importance (SHAP)", "data_drift": "data drift check",
                "concept_drift": "concept drift check (needs true labels)",
                "last_retraining": "retraining / champion-challenger", "fairness": "fairness report"}
    facts["scope"]["reports_available"] = [v for k, v in optional.items() if k in facts]
    facts["scope"]["reports_NOT_measured"] = [v for k, v in optional.items() if k not in facts]
    return facts


# question topic -> fact sections (keywords are matched in the lower-cased question)
TOPICS = {
    ("accura", "precision", "recall", "f1", "auc", "brier", "score", "metric", "perform",
     "leaderboard", "second", "candidate", "compare", "best", "selected", "algorithm"): ("test_metrics", "leaderboard_cv"),
    ("feature", "important", "driver", "drive", "why", "shap", "influen"): ("top_features", "feature_ranking_summary"),
    ("drift", "psi", "shift", "distribution", "changed", "change",
     "reference", "current data", "rows"): ("data_drift",),
    ("concept", "error rate", "labelled", "labeled", "detector", "adwin", "worse", "degrad"): ("concept_drift", "data_drift"),
    ("retrain", "challenger", "champion", "promot"): ("last_retraining",),
    ("fair", "bias", "gender", "sex", "race", "group"): ("fairness",),
    ("quality", "missing", "duplicate", "dropped", "clean", "prepar"): ("data_quality_notes",),
}
ALWAYS = ("scope", "model")


def select_facts(facts: dict, question: str) -> dict:
    """Keep only the sections relevant to the question (small models answer better with
    less context). Falls back to all facts when no topic matches."""
    q = question.lower()
    keep = set()
    for words, sections in TOPICS.items():
        if any(w in q for w in words):
            keep.update(sections)
    if not keep:
        return facts
    return {k: v for k, v in facts.items() if k in ALWAYS or k in keep}
