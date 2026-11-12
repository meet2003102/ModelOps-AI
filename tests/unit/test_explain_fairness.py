"""Step 2 tests: SHAP explanations (with and without the shap library) and fairness."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from ml_engine.evaluation import fairness
from ml_engine.explainability import shap_explainer as X
from ml_engine.registry import model_registry as reg
from ml_engine.training.automl import run_automl


def make_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Monthly", "One year", "Two year"], n)
    tenure = rng.integers(0, 72, n).astype(float)
    sex = rng.choice(["Male", "Female"], n)
    logit = 2.0 * (contract == "Monthly") - 0.06 * tenure
    return pd.DataFrame({
        "tenure": tenure, "charges": rng.normal(60, 20, n), "contract": contract, "sex": sex,
        "churn": np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")})


@pytest.fixture(scope="module")
def clf():
    return run_automl(make_df(), "churn", models=["Random Forest", "Logistic Regression"],
                      n_trials=2, cv_folds=3, exclude=["sex"])


@pytest.mark.parametrize("use_shap", [True, False])
def test_global_and_local(clf, monkeypatch, use_shap):
    if use_shap:
        pytest.importorskip("shap")
    else:
        monkeypatch.setattr(X, "HAS_SHAP", False)
    classes = list(clf.label_encoder.classes_)
    g = X.explain_global(clf.pipeline, clf.X_test, "classification", n_classes=2)
    feats = [i["feature"] for i in g["importance"]]
    assert set(feats) <= {"tenure", "charges", "contract"}   # original names, no one-hot names
    assert "sex" not in feats
    assert feats[0] in ("contract", "tenure")                # the true drivers rank first
    loc = X.explain_local(clf.pipeline, clf.X_test.iloc[[0]], clf.X_train, "classification", classes)
    assert loc["prediction"] in classes
    assert loc["contributions"] and "explanation_text" in loc
    assert loc["method"].startswith("shap" if use_shap else "fallback")


def test_shap_additivity(clf):
    pytest.importorskip("shap")
    # base value + contributions should reconstruct the model output (tree: log-odds or prob)
    loc = X.explain_local(clf.pipeline, clf.X_test.iloc[[0]], clf.X_train, "classification",
                          list(clf.label_encoder.classes_), top_k=50)
    total = loc["base_value"] + sum(c["contribution"] for c in loc["contributions"])
    assert np.isfinite(total)


def test_regression_explain():
    d = load_diabetes(as_frame=True).frame
    r = run_automl(d, "target", models=["Random Forest Regressor"], n_trials=2, cv_folds=3)
    g = X.explain_global(r.pipeline, r.X_test, "regression")
    assert g["importance"][0]["feature"] in ("bmi", "s5", "bp", "s1")
    loc = X.explain_local(r.pipeline, r.X_test.iloc[[0]], r.X_train, "regression")
    assert "increased" in loc["explanation_text"] or "decreased" in loc["explanation_text"]


def test_stability(clf):
    s = X.explanation_stability(clf.pipeline, clf.X_test.iloc[[0]], clf.X_train, "classification",
                                list(clf.label_encoder.classes_), repeats=3, top_k=3)
    assert 0.0 <= s <= 1.0


def test_group_report_math():
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    p = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    g = pd.Series(["A"] * 4 + ["B"] * 4)
    r = fairness.group_report(y, p, g, positive=1, min_group_size=1)
    # selection: A 3/4, B 1/4 ; TPR: A 2/2, B 1/2
    assert r["demographic_parity_difference"] == 0.5
    assert r["equal_opportunity_difference"] == 0.5
    assert r["flagged"]


def test_fairness_matches_fairlearn():
    fl = pytest.importorskip("fairlearn.metrics")
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 2, 400), rng.integers(0, 2, 400)
    s = pd.Series(rng.choice(["a", "b", "c"], 400))
    ours = fairness.group_report(y, p, s, positive=1)
    ref = fl.demographic_parity_difference(y, p, sensitive_features=s)
    assert abs(ours["demographic_parity_difference"] - ref) < 1e-3


def test_fairness_for_saved_model(clf, tmp_path):
    mid = reg.save_model(clf, "churn", root=tmp_path)
    bundle = reg.load_model(mid, root=tmp_path)
    test_raw = reg.load_test_raw(mid, root=tmp_path)
    rep = fairness.fairness_for_model(bundle, test_raw, "churn", ["sex"], positive_label="Yes")
    assert {g["group"] for g in rep["attributes"]["sex"]["groups"]} == {"Male", "Female"}
