"""End-to-end API test: upload -> EDA -> train -> explain -> predict -> feedback -> drift -> retrain.
Run: python -m pytest tests/api -q
"""
import io

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402

P = "/api/v1"
rng = np.random.default_rng(0)


def churn(n, shift=False):
    contract = rng.choice(["Monthly", "One year", "Two year"], n, p=[.75, .15, .1] if shift else [.5, .3, .2])
    tenure = rng.integers(0, 72, n).astype(float)
    charges = rng.normal(80 if shift else 60, 20, n)
    logit = 1.8 * (contract == "Monthly") - 0.05 * tenure + (2.5 if shift else 0) * (charges > 80)
    return pd.DataFrame({"customer_id": np.arange(n), "tenure": tenure, "charges": charges,
                         "contract": contract, "sex": rng.choice(["M", "F"], n),
                         "churn": np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")})


def as_file(df, name="data.csv"):
    return {"file": (name, io.BytesIO(df.to_csv(index=False).encode()), "text/csv")}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def trained(client):
    r = client.post(f"{P}/datasets", files=as_file(churn(500)), data={"name": "churn"})
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["suggested_target"] == "churn"
    r = client.post(f"{P}/training/jobs", json={"dataset_id": ds["id"], "target": "churn",
                                                 "models": ["Logistic Regression", "Random Forest"],
                                                 "n_trials": 2, "exclude": ["sex"]})
    assert r.status_code == 202, r.text
    job = client.get(f"{P}/jobs/{r.json()['id']}").json()      # background task already ran
    assert job["status"] == "done", job["error"]
    return ds, job["result"]["model_id"]


def test_health(client):
    r = client.get(f"{P}/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_upload_rejects_bad_files(client):
    r = client.post(f"{P}/datasets", files={"file": ("x.exe", io.BytesIO(b"abc"), "application/octet-stream")})
    assert r.status_code == 400
    r = client.post(f"{P}/datasets", files=as_file(churn(10)))
    assert r.status_code == 422                                     # too few rows


def test_dataset_endpoints(client, trained):
    ds, _ = trained
    assert any(d["id"] == ds["id"] for d in client.get(f"{P}/datasets").json())
    assert client.get(f"{P}/datasets/{ds['id']}/profile").json()["column_types"]["customer_id"] == "id"
    eda = client.get(f"{P}/datasets/{ds['id']}/eda", params={"target": "churn"}).json()
    assert eda["target"]["problem_type"] == "classification"
    assert len(client.get(f"{P}/datasets/{ds['id']}/preview", params={"rows": 5}).json()["rows"]) == 5


def test_model_endpoints(client, trained):
    _, mid = trained
    meta = client.get(f"{P}/models/{mid}").json()
    assert meta["status"] == "champion" and meta["leaderboard"]
    assert client.get(f"{P}/models/{mid}/download").status_code == 200
    g = client.get(f"{P}/models/{mid}/explain/global").json()
    assert g["importance"][0]["feature"] in ("contract", "tenure")
    loc = client.post(f"{P}/models/{mid}/explain/local", json={"test_row": 0}).json()
    assert loc["contributions"] and loc["explanation_text"]
    fair = client.get(f"{P}/models/{mid}/fairness", params={"sensitive": "sex", "positive": "Yes"}).json()
    assert "sex" in fair["attributes"]
    assert "classification" in client.get(f"{P}/training/options").json()
    sch = client.get(f"{P}/models/{mid}/schema").json()
    assert "contract" in sch["categories"] and sch["examples"]


def test_predict_feedback_and_monitoring(client, trained):
    _, mid = trained
    rec = churn(1).drop(columns=["churn"]).to_dict(orient="records")[0]
    r = client.post(f"{P}/models/{mid}/predict", json={"records": [rec]})
    assert r.status_code == 200, r.text
    assert r.json()["predictions"][0]["prediction"] in ("Yes", "No")
    r = client.post(f"{P}/models/{mid}/predict", json={"records": [{"tenure": 3}]})
    assert r.status_code == 400                                     # missing columns

    new = churn(300, shift=True)
    r = client.post(f"{P}/models/{mid}/predict/batch", files=as_file(new.drop(columns=["churn"])))
    preds = r.json()["predictions"]
    assert r.json()["n"] == 300
    fb = [{"prediction_id": p["prediction_id"], "actual": a} for p, a in zip(preds, new["churn"])]
    assert client.post(f"{P}/predictions/feedback", json={"items": fb}).json()["updated"] == 300
    assert len(client.get(f"{P}/predictions", params={"model_id": mid, "limit": 10}).json()) == 10

    dd = client.post(f"{P}/monitoring/{mid}/data-drift", files=as_file(new)).json()
    assert "charges" in dd["drifted_features"]
    dd2 = client.post(f"{P}/monitoring/{mid}/data-drift").json()     # from logged predictions
    assert dd2["n_current"] >= 300
    cd = client.get(f"{P}/monitoring/{mid}/concept-drift").json()
    assert cd["n"] == 300
    assert client.get(f"{P}/monitoring/{mid}/alerts").json()


def test_retrain(client, trained):
    _, mid = trained
    r = client.post(f"{P}/monitoring/{mid}/retrain", files=as_file(churn(600, shift=True)))
    assert r.status_code == 202, r.text
    job = client.get(f"{P}/jobs/{r.json()['id']}").json()
    assert job["status"] == "done", job["error"]
    assert {"champion", "challenger", "promote"} <= set(job["result"])


def test_insights_template_mode(client, trained, monkeypatch):
    from llm_engine import client as llm
    monkeypatch.setattr(llm, "status", lambda timeout=2.0: {"available": False, "model": "x", "message": "off"})
    _, mid = trained
    client.get(f"{P}/models/{mid}/explain/global")              # saves the SHAP report
    r = client.post(f"{P}/insights/report", json={"model_id": mid}).json()
    assert r["source"] == "template" and "top_features" in r["facts_used"]
    assert r["check"]["unsupported_numbers"] == []
    a = client.post(f"{P}/insights/ask", json={"model_id": mid, "question": "Which features drifted?"}).json()
    assert a["answer"]
