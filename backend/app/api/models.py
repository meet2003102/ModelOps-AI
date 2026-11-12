from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.app import db as D
from backend.app import services as S
from ml_engine import config as engine_config
from ml_engine.evaluation.fairness import fairness_for_model
from ml_engine.explainability.shap_explainer import explain_global, explain_local
from llm_engine.context_builder import save_report
from ml_engine.registry import model_registry as reg

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(dataset: str | None = None):
    ms = reg.list_models()
    return [m for m in ms if dataset is None or m["dataset"] == dataset]


@router.get("/{model_id}")
def get_model(model_id: str):
    return D.jsonable(S.get_model(model_id)["metadata"])


@router.get("/{model_id}/download")
def download(model_id: str):
    S.get_model(model_id)
    path = Path(engine_config.ARTIFACTS_DIR) / "models" / model_id / "pipeline.joblib"
    return FileResponse(path, filename=f"modelops_{model_id}.joblib",
                        media_type="application/octet-stream")


@router.post("/{model_id}/promote")
def promote(model_id: str):
    S.get_model(model_id)
    reg.set_status(model_id, "champion")
    S.refresh_model(model_id)
    return {"model_id": model_id, "status": "champion"}


@router.get("/{model_id}/explain/global")
def global_explanation(model_id: str, max_rows: int = 500):
    b = S.get_model(model_id)
    meta = b["metadata"]
    X = reg.load_test_raw(model_id)[b["features"]]
    out = D.jsonable(explain_global(b["pipeline"], X, meta["task"],
                                    n_classes=len(meta.get("classes") or [0, 0]), max_rows=max_rows))
    save_report(model_id, "global_explanation", out)
    return out


@router.get("/{model_id}/schema")
def input_schema(model_id: str, examples: int = 3):
    """Feature types, category options and example rows - used to build the prediction form."""
    b = S.get_model(model_id)
    f = b["metadata"]["features"]
    ref = reg.load_reference(model_id)
    cats = {c: sorted(map(str, ref[c].dropna().unique()))[:100] for c in f["categorical"] if c in ref}
    ex = reg.load_test_raw(model_id)[b["features"]].head(max(1, min(examples, 20)))
    return {"features": b["features"], "numeric": f["numeric"], "categorical": f["categorical"],
            "datetime": f["datetime"], "categories": cats, "examples": S.df_to_records(ex),
            "target": f["target"], "classes": b["metadata"].get("classes")}


class LocalRequest(BaseModel):
    record: dict | None = None        # a new input row
    test_row: int | None = None       # or the index of a held-out test row


@router.post("/{model_id}/explain/local")
def local_explanation(model_id: str, req: LocalRequest):
    b = S.get_model(model_id)
    meta = b["metadata"]
    if req.record is not None:
        row = pd.DataFrame([req.record])
        missing = [f for f in b["features"] if f not in row]
        if missing:
            raise HTTPException(400, f"Missing fields: {missing}")
        row = row[b["features"]]
    else:
        test = reg.load_test_raw(model_id)
        i = req.test_row or 0
        if not 0 <= i < len(test):
            raise HTTPException(400, f"test_row must be between 0 and {len(test) - 1}")
        row = test[b["features"]].iloc[[i]]
    return D.jsonable(explain_local(b["pipeline"], row, reg.load_reference(model_id),
                                    meta["task"], meta.get("classes")))


@router.get("/{model_id}/fairness")
def fairness(model_id: str, sensitive: list[str] = Query(...), positive: str | None = None):
    b = S.get_model(model_id)
    meta = b["metadata"]
    try:
        out = D.jsonable(fairness_for_model(b, reg.load_test_raw(model_id),
                                            meta["features"]["target"], sensitive, positive))
        save_report(model_id, "fairness", out)
        return out
    except ValueError as e:
        raise HTTPException(400, str(e))
