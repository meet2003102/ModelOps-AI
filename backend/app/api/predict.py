import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app import db as D
from backend.app import services as S
from ml_engine.registry import model_registry as reg

router = APIRouter(tags=["prediction"])


class PredictRequest(BaseModel):
    records: list[dict] = Field(..., min_length=1, max_length=1000)


def _predict_and_log(model_id: str, df: pd.DataFrame, source: str, db: Session) -> pd.DataFrame:
    b = S.get_model(model_id)
    try:
        out = reg.predict(b, df)
    except ValueError as e:
        raise HTTPException(400, str(e))
    feats = b["features"]
    rows = S.df_to_records(df[feats])
    ids = []
    for rec, (_, r) in zip(rows, out.iterrows()):
        p = D.Prediction(model_id=model_id, inputs=rec, prediction=str(r["prediction"]),
                         confidence=float(r["confidence"]) if "confidence" in r else None, source=source)
        db.add(p)
        db.flush()
        ids.append(p.id)
    db.commit()
    out.insert(0, "prediction_id", ids)
    return out


@router.post("/models/{model_id}/predict")
def predict(model_id: str, req: PredictRequest, db: Session = Depends(D.get_db)):
    out = _predict_and_log(model_id, pd.DataFrame(req.records), "single", db)
    return {"model_id": model_id, "predictions": S.df_to_records(out)}


@router.post("/models/{model_id}/predict/batch")
async def predict_batch(model_id: str, file: UploadFile = File(...), as_csv: bool = False,
                        db: Session = Depends(D.get_db)):
    df = S.read_upload_csv(await file.read(), file.filename or "batch.csv")
    out = _predict_and_log(model_id, df, "batch", db)
    if as_csv:
        merged = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
        return Response(S.csv_bytes(merged), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=predictions.csv"})
    return {"model_id": model_id, "n": len(out), "predictions": S.df_to_records(out.head(1000))}


@router.get("/predictions")
def history(model_id: str | None = None, limit: int = 100, db: Session = Depends(D.get_db)):
    q = select(D.Prediction).order_by(D.Prediction.id.desc()).limit(min(limit, 1000))
    if model_id:
        q = q.where(D.Prediction.model_id == model_id)
    return [{"id": p.id, "model_id": p.model_id, "prediction": p.prediction,
             "confidence": p.confidence, "actual": p.actual, "source": p.source,
             "created_at": p.created_at.isoformat(), "inputs": p.inputs}
            for p in db.scalars(q).all()]


class Feedback(BaseModel):
    prediction_id: int
    actual: str


class FeedbackBatch(BaseModel):
    items: list[Feedback] = Field(..., min_length=1)


@router.post("/predictions/feedback")
def feedback(req: FeedbackBatch, db: Session = Depends(D.get_db)):
    """Attach true labels to earlier predictions (needed for concept-drift monitoring)."""
    updated, missing = 0, []
    for f in req.items:
        p = db.get(D.Prediction, f.prediction_id)
        if p is None:
            missing.append(f.prediction_id)
            continue
        p.actual = f.actual
        updated += 1
    db.commit()
    return {"updated": updated, "not_found": missing}
