"""ModelOps AI - FastAPI backend.

Run from the project root:
    uvicorn backend.app.main:app --reload
Interactive API docs: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.app.api import datasets, insights, models, monitoring, predict, training
from backend.app.config import API_PREFIX, CORS_ORIGINS
from backend.app.db import init_db
from ml_engine.drift.concept_monitor import HAS_RIVER
from ml_engine.explainability.shap_explainer import HAS_SHAP
from ml_engine.training.model_zoo import HAS_LGBM, HAS_XGB
from ml_engine.training.tuner import HAS_OPTUNA


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="ModelOps AI", version="0.1.0", lifespan=lifespan,
              description="AutoML, explainability, drift monitoring and model serving.")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

for r in (datasets.router, training.router, models.router, predict.router, monitoring.router, insights.router):
    app.include_router(r, prefix=API_PREFIX)


@app.get(f"{API_PREFIX}/health")
def health():
    return {"status": "ok", "libraries": {"xgboost": HAS_XGB, "lightgbm": HAS_LGBM, "optuna": HAS_OPTUNA,
                                          "shap": HAS_SHAP, "river": HAS_RIVER}}


@app.get("/", include_in_schema=False)
def root():
    """The API has no page of its own; send visitors to the interactive docs."""
    return RedirectResponse("/docs")
