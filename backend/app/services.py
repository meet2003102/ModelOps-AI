"""Glue between the API and ml_engine. No ML logic lives here."""
from __future__ import annotations

import io
import traceback
from functools import lru_cache

import pandas as pd
from fastapi import HTTPException

from backend.app import db as D
from backend.app.config import DATASETS_DIR
from llm_engine.context_builder import save_report
from ml_engine.ingestion.profiling import profile_dataset
from ml_engine.ingestion.validation import ValidationError, load_dataframe, validate_dataframe
from ml_engine.registry import model_registry as reg
from ml_engine.retraining.champion_challenger import retrain_and_compare
from ml_engine.training.automl import run_automl


# ------------------------------------------------------------------ datasets
def store_dataset(content: bytes, filename: str, name: str | None, session) -> D.Dataset:
    try:
        df = load_dataframe(content, filename)
    except ValidationError as e:
        raise HTTPException(400, str(e))
    validation = validate_dataframe(df)
    if not validation["valid"]:
        raise HTTPException(422, {"message": "Dataset failed validation", **validation})
    df = df.dropna(axis=1, how="all")
    ds = D.Dataset(name=name or filename.rsplit(".", 1)[0], filename=filename, path="",
                   n_rows=len(df), n_cols=df.shape[1], validation=D.jsonable(validation),
                   profile=D.jsonable(profile_dataset(df)))
    session.add(ds)
    session.flush()
    path = DATASETS_DIR / f"{ds.id}.csv"
    df.to_csv(path, index=False)
    ds.path = str(path)
    session.commit()
    return ds


def load_dataset(dataset_id: str, session) -> tuple[D.Dataset, pd.DataFrame]:
    ds = session.get(D.Dataset, dataset_id)
    if ds is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return ds, pd.read_csv(ds.path)


def read_upload_csv(content: bytes, filename: str) -> pd.DataFrame:
    try:
        return load_dataframe(content, filename)
    except ValidationError as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------ models
@lru_cache(maxsize=16)
def _cached(model_id: str):
    return reg.load_model(model_id)


def get_model(model_id: str) -> dict:
    try:
        return _cached(model_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{model_id}' not found")


def refresh_model(model_id: str):
    _cached.cache_clear()


# ------------------------------------------------------------------ background jobs
def _update(job_id: str, **fields):
    with D.SessionLocal() as s:
        job = s.get(D.Job, job_id)
        for k, v in fields.items():
            setattr(job, k, v)
        s.commit()


def run_training_job(job_id: str, dataset_id: str, params: dict):
    _update(job_id, status="running", message="Loading data")
    try:
        with D.SessionLocal() as s:
            ds, df = load_dataset(dataset_id, s)
            ds_name = ds.name
        res = run_automl(
            df, params["target"], task=params.get("task"), metric=params.get("metric"),
            models=params.get("models"), n_trials=params.get("n_trials", 20),
            exclude=params.get("exclude"),
            progress=lambda p, m: _update(job_id, progress=round(p, 3), message=m))
        model_id = reg.save_model(res, dataset_name=ds_name)
        if reg.champion_id(ds_name) is None:
            reg.set_status(model_id, "champion")
        refresh_model(model_id)
        _update(job_id, status="done", progress=1.0, message="Done",
                result=D.jsonable({"model_id": model_id, **res.summary()}))
    except Exception as e:  # noqa: BLE001 - report every failure to the user
        _update(job_id, status="failed", error=f"{e}\n{traceback.format_exc(limit=3)}", message=str(e))


def run_retraining_job(job_id: str, model_id: str, data: pd.DataFrame, min_gain: float):
    _update(job_id, status="running", message="Training challenger", progress=0.2)
    try:
        out = retrain_and_compare(model_id, data, min_gain=min_gain)
        save_report(model_id, "retraining", D.jsonable(out))
        refresh_model(model_id)
        _update(job_id, status="done", progress=1.0, message="Done", result=D.jsonable(out))
    except Exception as e:  # noqa: BLE001
        _update(job_id, status="failed", error=f"{e}\n{traceback.format_exc(limit=3)}", message=str(e))


def df_to_records(df: pd.DataFrame) -> list[dict]:
    return D.jsonable(df.astype(object).where(df.notna(), None).to_dict(orient="records"))


def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()
