from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app import db as D
from backend.app import services as S
from backend.app.api.training import job_out
from ml_engine import config as engine_config
from ml_engine.drift.concept_monitor import monitor_labelled_batch
from ml_engine.drift.evidently_report import save_html_report
from ml_engine.drift.psi_ks import data_drift_report
from ml_engine.explainability.shap_explainer import explain_global
from llm_engine.context_builder import save_report
from ml_engine.registry import model_registry as reg

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def _model_dir(model_id: str) -> Path:
    return Path(engine_config.ARTIFACTS_DIR) / "models" / model_id


def _alert(db, model_id, kind, severity, summary, details):
    db.add(D.Alert(model_id=model_id, kind=kind, severity=severity, summary=summary,
                   details=D.jsonable(details)))
    db.commit()


@router.post("/{model_id}/data-drift")
async def data_drift(model_id: str, file: UploadFile | None = File(None), last_n: int = Form(1000),
                     db: Session = Depends(D.get_db)):
    """Compare new data with the training reference. Uses the uploaded CSV, or, if none,
    the inputs of the last `last_n` logged predictions."""
    b = S.get_model(model_id)
    meta, feats = b["metadata"], b["features"]
    ref = reg.load_reference(model_id)
    if file is not None:
        cur = S.read_upload_csv(await file.read(), file.filename or "current.csv")
    else:
        preds = db.scalars(select(D.Prediction).where(D.Prediction.model_id == model_id)
                           .order_by(D.Prediction.id.desc()).limit(last_n)).all()
        if len(preds) < 30:
            raise HTTPException(400, "Upload a CSV or log at least 30 predictions first.")
        cur = pd.DataFrame([p.inputs for p in reversed(preds)])
    imp = explain_global(b["pipeline"], ref, meta["task"], n_classes=len(meta.get("classes") or [0, 0]),
                         max_rows=300)
    try:
        rep = data_drift_report(ref, cur, feats, importance={i["feature"]: i["share"] for i in imp["importance"]})
    except ValueError as e:
        raise HTTPException(400, str(e))
    html = save_html_report(ref, cur[feats], _model_dir(model_id) / "evidently_drift.html")
    rep["evidently_html"] = bool(html)
    save_report(model_id, "data_drift", D.jsonable(rep))
    save_report(model_id, "global_explanation", D.jsonable(imp))
    if rep["drifted_features"]:
        sev = "major" if rep["dataset_drift"] else "moderate"
        _alert(db, model_id, "data_drift", sev,
               f"{len(rep['drifted_features'])} features drifted; most important: "
               f"{', '.join(rep['top_important_drifted'][:3])}", rep)
    return D.jsonable(rep)


@router.get("/{model_id}/data-drift/html")
def data_drift_html(model_id: str):
    path = _model_dir(model_id) / "evidently_drift.html"
    if not path.exists():
        raise HTTPException(404, "No Evidently report yet - run a data-drift check first.")
    return FileResponse(path, media_type="text/html")


@router.get("/{model_id}/concept-drift")
def concept_drift(model_id: str, detector: str = "adwin", db: Session = Depends(D.get_db)):
    """Run the detector over logged predictions that received true labels (feedback)."""
    meta = S.get_model(model_id)["metadata"]
    preds = db.scalars(select(D.Prediction).where(D.Prediction.model_id == model_id,
                                                  D.Prediction.actual.is_not(None))
                       .order_by(D.Prediction.id)).all()
    if len(preds) < 30:
        return {"n": len(preds), "concept_drift": None,
                "message": "Need at least 30 predictions with feedback labels."}
    y_true = [p.actual for p in preds]
    y_pred = [p.prediction for p in preds]
    if meta["task"] == "regression":
        y_true, y_pred = list(map(float, y_true)), list(map(float, y_pred))
    out = monitor_labelled_batch(y_true, y_pred, meta["task"], detector=detector)
    save_report(model_id, "concept_drift", D.jsonable(out))
    if out["concept_drift"]:
        _alert(db, model_id, "concept_drift", "major",
               f"{out['detector']} alarm at labelled prediction #{out['alarms'][0]}", out)
    return D.jsonable(out)


@router.get("/{model_id}/alerts")
def alerts(model_id: str, db: Session = Depends(D.get_db)):
    rows = db.scalars(select(D.Alert).where(D.Alert.model_id == model_id)
                      .order_by(D.Alert.id.desc())).all()
    return [{"id": a.id, "kind": a.kind, "severity": a.severity, "summary": a.summary,
             "created_at": a.created_at.isoformat()} for a in rows]


@router.post("/{model_id}/retrain", status_code=202)
async def retrain(model_id: str, background: BackgroundTasks, file: UploadFile = File(...),
                  min_gain: float = Form(0.01), db: Session = Depends(D.get_db)):
    """Champion-challenger on newly labelled data (CSV must contain the target column)."""
    meta = S.get_model(model_id)["metadata"]
    data = S.read_upload_csv(await file.read(), file.filename or "labelled.csv")
    if meta["features"]["target"] not in data:
        raise HTTPException(400, f"CSV must contain the target column '{meta['features']['target']}'")
    job = D.Job(kind="retraining", params={"model_id": model_id, "rows": len(data), "min_gain": min_gain})
    db.add(job)
    db.commit()
    background.add_task(S.run_retraining_job, job.id, model_id, data, min_gain)
    return job_out(job)
