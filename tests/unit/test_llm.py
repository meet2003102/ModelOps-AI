"""Step 6 tests: facts, number checking, template fallback and the LLM path (Ollama faked)."""
import numpy as np
import pandas as pd
import pytest

from llm_engine import client, context_builder, insight_generator as IG
from ml_engine import config
from ml_engine.explainability.shap_explainer import explain_global
from ml_engine.registry import model_registry as reg
from ml_engine.training.automl import run_automl


@pytest.fixture()
def model_id(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"x": x, "z": rng.normal(0, 1, n), "y": np.where(x + rng.normal(0, .5, n) > 0, "Yes", "No")})
    r = run_automl(df, "y", models=["Logistic Regression"], n_trials=2, cv_folds=3)
    mid = reg.save_model(r, "toy")
    context_builder.save_report(mid, "global_explanation", explain_global(r.pipeline, r.X_test, "classification"))
    return mid


def test_facts(model_id):
    f = context_builder.build_facts(model_id)
    assert f["model"]["algorithm"] == "Logistic Regression"
    assert f["top_features"][0]["feature"] == "x"
    assert "data_drift" not in f                       # nothing invented when no report exists


def test_number_check():
    facts = {"test_metrics": {"roc_auc": 0.7857, "f1": 0.7182}}
    ok = IG.check_numbers("ROC-AUC is 0.786 (78.6%), F1 0.7182, top 3 features.", facts)
    assert ok["unsupported_numbers"] == []
    bad = IG.check_numbers("Accuracy is 0.93 and recall 64%.", facts)
    assert set(bad["unsupported_numbers"]) == {"0.93", "64%"}


def test_template_when_llm_offline(model_id, monkeypatch):
    monkeypatch.setattr(client, "status", lambda timeout=2.0: {"available": False, "model": "m", "message": "off"})
    r = IG.report(model_id)
    assert r["source"] == "template" and "Logistic Regression" in r["text"]
    assert r["check"]["unsupported_numbers"] == []     # template only uses facts
    a = IG.ask(model_id, "Which features are most important?")
    assert "x" in a["answer"]


def test_llm_path_and_hallucination_flag(model_id, monkeypatch):
    monkeypatch.setattr(client, "status", lambda timeout=2.0: {"available": True, "model": "qwen2.5:3b", "message": "ready"})
    monkeypatch.setattr(client, "chat", lambda system, user, **k: ("The model reaches an accuracy of 0.991.", 0.5))
    r = IG.report(model_id)
    assert r["source"] == "llm:qwen2.5:3b"
    assert "0.991" in r["check"]["unsupported_numbers"]   # invented number is caught


def test_select_facts_and_scope(model_id):
    f = context_builder.build_facts(model_id)
    assert "scope" in f and f["top_features"][0]["rank"] == 1
    s = context_builder.select_facts(f, "What is the most important feature?")
    assert set(s) == {"scope", "model", "top_features", "feature_ranking_summary"}
    assert context_builder.select_facts(f, "hello there") == f          # no topic -> all facts


def test_focused_answer_checked_against_all_facts(model_id, monkeypatch):
    monkeypatch.setattr(client, "status", lambda timeout=2.0: {"available": True, "model": "m", "message": "ready"})
    seen = {}
    def fake_chat(system, user, **k):
        seen["prompt"] = user
        return ("Not available in the current reports.", 0.1)
    monkeypatch.setattr(client, "chat", fake_chat)
    r = IG.ask(model_id, "What was the accuracy last year?")
    assert r["focused"] and "test_metrics" in seen["prompt"] and "top_features" not in seen["prompt"]
    assert r["check"]["unsupported_numbers"] == []


def test_scope_lists_missing_reports(model_id):
    sc = context_builder.build_facts(model_id)["scope"]
    assert "feature importance (SHAP)" in sc["reports_available"]
    assert "concept drift check (needs true labels)" in sc["reports_NOT_measured"]
