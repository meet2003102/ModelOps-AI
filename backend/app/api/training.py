from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app import db as D
from backend.app import services as S
from ml_engine.evaluation.metrics import CLASSIFICATION_SCORERS, REGRESSION_SCORERS
from ml_engine.training.model_zoo import get_models

router = APIRouter(tags=["training"])


class TrainRequest(BaseModel):
    dataset_id: str
    target: str
    task: str | None = Field(None, pattern="^(classification|regression)$")
    metric: str | None = None
    models: list[str] | None = None
    exclude: list[str] | None = None
    n_trials: int = Field(20, ge=0, le=200)


def job_out(j: D.Job) -> dict:
    return {"id": j.id, "kind": j.kind, "status": j.status, "progress": j.progress,
            "message": j.message, "params": j.params, "result": j.result, "error": j.error,
            "created_at": j.created_at.isoformat(), "updated_at": j.updated_at.isoformat()}


@router.get("/training/options")
def options():
    """Models and metrics available on this machine (XGBoost/LightGBM only if installed)."""
    return {"classification": {"models": list(get_models("classification")),
                               "metrics": list(CLASSIFICATION_SCORERS)},
            "regression": {"models": list(get_models("regression")),
                           "metrics": list(REGRESSION_SCORERS)}}


@router.post("/training/jobs", status_code=202)
def start_training(req: TrainRequest, background: BackgroundTasks, db: Session = Depends(D.get_db)):
    ds, df = S.load_dataset(req.dataset_id, db)
    if req.target not in df.columns:
        raise HTTPException(400, f"Target column '{req.target}' not in dataset")
    job = D.Job(kind="training", params=req.model_dump())
    db.add(job)
    db.commit()
    background.add_task(S.run_training_job, job.id, req.dataset_id, req.model_dump())
    return job_out(job)


@router.get("/jobs")
def list_jobs(limit: int = 20, db: Session = Depends(D.get_db)):
    rows = db.scalars(select(D.Job).order_by(D.Job.created_at.desc()).limit(limit)).all()
    return [job_out(j) for j in rows]


@router.get("/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(D.get_db)):
    job = db.get(D.Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job_out(job)
