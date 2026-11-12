"""Unit tests for Step 1 (ingestion, EDA, AutoML, registry). Run: python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

from ml_engine.eda.analysis import run_eda
from ml_engine.ingestion import profiling
from ml_engine.ingestion.validation import ValidationError, validate_dataframe, validate_file
from ml_engine.registry import model_registry as reg
from ml_engine.training.automl import run_automl


def churn_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Monthly", "One year", "Two year"], n)
    tenure = rng.integers(0, 72, n)
    logit = 1.5 * (contract == "Monthly") - 0.05 * tenure
    return pd.DataFrame({
        "customer_id": np.arange(n),
        "tenure": tenure.astype(float),
        "charges": rng.normal(60, 20, n),
        "contract": contract,
        "signup": pd.date_range("2022-01-01", periods=n).astype(str),
        "country": "IN",
        "churn": np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No"),
    })


def test_file_validation():
    validate_file("a.csv", 100)
    with pytest.raises(ValidationError):
        validate_file("a.exe", 100)
    with pytest.raises(ValidationError):
        validate_file("a.csv", 10**9)
    assert not validate_dataframe(pd.DataFrame({"a": [1]}))["valid"]


def test_profile_and_eda():
    df = churn_df()
    p = profiling.profile_dataset(df)
    assert p["column_types"]["customer_id"] == profiling.ID
    assert p["suggested_target"] == "churn"
    eda = run_eda(df, "churn", p)
    assert "tenure" in eda["numeric_distributions"]
    assert eda["target"]["problem_type"] == "classification"


def test_automl_classification(tmp_path):
    r = run_automl(churn_df(), "churn", n_trials=3, cv_folds=3)
    assert r.task == "classification"
    assert r.leaderboard[0]["rank"] == 1
    assert "customer_id" in r.plan.dropped and "country" in r.plan.dropped
    assert r.test_metrics["f1_macro"] > 0.5
    mid = reg.save_model(r, "churn", root=tmp_path)
    bundle = reg.load_model(mid, root=tmp_path)
    out = reg.predict(bundle, churn_df(10, seed=1))
    assert set(out["prediction"]) <= {"Yes", "No"}
    reg.set_status(mid, "champion", root=tmp_path)
    assert reg.list_models(tmp_path)[0]["status"] == "champion"


def test_automl_regression():
    d = load_diabetes(as_frame=True).frame
    r = run_automl(d, "target", n_trials=3, cv_folds=3)
    assert r.task == "regression"
    assert r.test_metrics["r2"] > 0.3
    assert r.leaderboard[0]["model"] != "Baseline (mean)"


def test_no_leakage_pipeline():
    d = load_breast_cancer(as_frame=True).frame
    r = run_automl(d, "target", metric="roc_auc", n_trials=2, cv_folds=3)
    # preprocessing lives inside the pipeline that is refit per CV fold
    assert r.pipeline.steps[0][0] == "pre"
    assert r.test_metrics["roc_auc"] > 0.9
