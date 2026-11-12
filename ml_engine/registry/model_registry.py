"""File-based model registry: one folder per model version.

artifacts/models/<model_id>/
    pipeline.joblib     preprocessor + model (+ label encoder)
    metadata.json       metrics, features, params, status
    reference.csv       training sample, used later as the drift reference
The backend keeps an index of these in SQLite.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from ml_engine import config


def _dir(model_id: str, root: Path | None = None) -> Path:
    return (root or config.ARTIFACTS_DIR) / "models" / model_id


def save_model(result, dataset_name: str = "", root: Path | None = None,
               reference_rows: int = 5000) -> str:
    model_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = _dir(model_id, root)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": result.pipeline, "label_encoder": result.label_encoder,
                 "features": result.plan.features}, d / "pipeline.joblib")
    ref = result.X_train.sample(min(reference_rows, len(result.X_train)), random_state=config.RANDOM_STATE)
    ref.to_csv(d / "reference.csv", index=False)
    if result.test_raw is not None:
        result.test_raw.to_csv(d / "test_raw.csv", index=False)
    meta = {"model_id": model_id, "dataset": dataset_name,
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "candidate",
            **result.summary()}
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return model_id


def load_model(model_id: str, root: Path | None = None) -> dict:
    d = _dir(model_id, root)
    if not d.exists():
        raise FileNotFoundError(f"Model '{model_id}' not found.")
    bundle = joblib.load(d / "pipeline.joblib")  # only load files this app wrote
    bundle["metadata"] = json.loads((d / "metadata.json").read_text())
    return bundle


def load_reference(model_id: str, root: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(_dir(model_id, root) / "reference.csv")


def load_test_raw(model_id: str, root: Path | None = None) -> pd.DataFrame:
    """Held-out test rows with ALL original columns (used for fairness)."""
    return pd.read_csv(_dir(model_id, root) / "test_raw.csv")


def list_models(root: Path | None = None) -> list[dict]:
    base = (root or config.ARTIFACTS_DIR) / "models"
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir(), reverse=True):
        f = d / "metadata.json"
        if f.exists():
            m = json.loads(f.read_text())
            out.append({k: m.get(k) for k in ("model_id", "dataset", "created_at", "status",
                                              "task", "best_model", "metric", "test_metrics")})
    return out


def champion_id(dataset: str | None = None, root: Path | None = None) -> str | None:
    """Current champion (optionally for one dataset), else None."""
    for m in list_models(root):
        if m["status"] == "champion" and (dataset is None or m["dataset"] == dataset):
            return m["model_id"]
    return None


def set_status(model_id: str, status: str, root: Path | None = None) -> None:
    """status: candidate | champion | archived. Promoting one champion archives the old one
    for the same dataset."""
    f = _dir(model_id, root) / "metadata.json"
    meta = json.loads(f.read_text())
    if status == "champion":
        for m in list_models(root):
            if m["status"] == "champion" and m["dataset"] == meta["dataset"] and m["model_id"] != model_id:
                set_status(m["model_id"], "archived", root)
    meta["status"] = status
    f.write_text(json.dumps(meta, indent=2, default=str))


def predict(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Predict for new rows; missing feature columns are an error, extra ones are ignored."""
    feats = bundle["features"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[feats]
    pipe, le = bundle["pipeline"], bundle["label_encoder"]
    out = pd.DataFrame(index=df.index)
    pred = pipe.predict(X)
    if le is not None:
        out["prediction"] = le.inverse_transform(pred.astype(int))
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)
            out["confidence"] = proba.max(axis=1).round(4)
            for i, c in enumerate(le.classes_):
                out[f"proba_{c}"] = proba[:, i].round(4)
    else:
        out["prediction"] = pred
    return out
