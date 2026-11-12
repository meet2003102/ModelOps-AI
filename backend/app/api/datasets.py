from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app import db as D
from backend.app import services as S
from ml_engine.eda.analysis import run_eda

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _out(ds: D.Dataset) -> dict:
    return {"id": ds.id, "name": ds.name, "filename": ds.filename, "n_rows": ds.n_rows,
            "n_cols": ds.n_cols, "created_at": ds.created_at.isoformat(),
            "suggested_target": ds.profile.get("suggested_target"),
            "warnings": ds.validation.get("warnings", []) + ds.profile.get("warnings", [])}


@router.post("", status_code=201)
async def upload(file: UploadFile = File(...), name: str | None = Form(None),
                 db: Session = Depends(D.get_db)):
    content = await file.read()
    ds = S.store_dataset(content, file.filename or "upload.csv", name, db)
    return _out(ds)


@router.get("")
def list_datasets(db: Session = Depends(D.get_db)):
    rows = db.scalars(select(D.Dataset).order_by(D.Dataset.created_at.desc())).all()
    return [_out(d) for d in rows]


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(D.get_db)):
    ds = db.get(D.Dataset, dataset_id)
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    return {**_out(ds), "validation": ds.validation}


@router.get("/{dataset_id}/profile")
def profile(dataset_id: str, db: Session = Depends(D.get_db)):
    ds = db.get(D.Dataset, dataset_id)
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    return ds.profile


@router.get("/{dataset_id}/preview")
def preview(dataset_id: str, rows: int = 50, db: Session = Depends(D.get_db)):
    _, df = S.load_dataset(dataset_id, db)
    return {"columns": list(df.columns), "rows": S.df_to_records(df.head(min(rows, 500)))}


@router.get("/{dataset_id}/eda")
def eda(dataset_id: str, target: str | None = None, db: Session = Depends(D.get_db)):
    ds, df = S.load_dataset(dataset_id, db)
    if target and target not in df.columns:
        raise HTTPException(400, f"Unknown target column '{target}'")
    return D.jsonable(run_eda(df, target, ds.profile))
