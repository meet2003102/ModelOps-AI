from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app import db as D
from backend.app import services as S
from llm_engine import client
from llm_engine.insight_generator import ask, report

router = APIRouter(prefix="/insights", tags=["insights"])


@router.get("/status")
def status():
    return client.status()


class ReportRequest(BaseModel):
    model_id: str


class AskRequest(BaseModel):
    model_id: str
    question: str = Field(..., min_length=3, max_length=500)


@router.post("/report")
def make_report(req: ReportRequest):
    S.get_model(req.model_id)            # 404 if unknown
    return D.jsonable(report(req.model_id))


@router.post("/ask")
def ask_question(req: AskRequest):
    S.get_model(req.model_id)
    return D.jsonable(ask(req.model_id, req.question))
