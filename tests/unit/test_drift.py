"""Step 3 tests: data drift, concept drift detectors, stream benchmark, champion-challenger."""
import numpy as np
import pandas as pd
import pytest

from ml_engine.drift import concept_monitor as CM
from ml_engine.drift.psi_ks import data_drift_report, psi_categorical, psi_numeric
from ml_engine.drift.stream_experiment import run_stream
from ml_engine.drift.stream_sim import scenario
from ml_engine.registry import model_registry as reg
from ml_engine.retraining.champion_challenger import decide, retrain_and_compare
from ml_engine.training.automl import run_automl

rng = np.random.default_rng(0)


def test_psi_values():
    a = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(a, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert psi_numeric(a, pd.Series(rng.normal(1, 1, 5000))) > 0.2
    c1 = pd.Series(rng.choice(list("ab"), 2000, p=[.5, .5]))
    c2 = pd.Series(rng.choice(list("ab"), 2000, p=[.9, .1]))
    assert psi_categorical(c1, c2) > 0.2


def test_data_drift_report_and_impact():
    ref = pd.DataFrame({"x": rng.normal(0, 1, 3000), "z": rng.normal(0, 1, 3000),
                        "c": rng.choice(list("abc"), 3000)})
    cur = ref.copy()
    cur["x"] = cur["x"] + 1.0               # shifted, important
    cur["z"] = cur["z"] + 1.0               # shifted, unimportant
    rep = data_drift_report(ref, cur, ["x", "z", "c"], importance={"x": 0.8, "z": 0.05, "c": 0.15})
    assert set(rep["drifted_features"]) == {"x", "z"}
    assert rep["ranking"][0] == "x"         # impact ranking puts the important one first
    assert not rep["features"]["c"]["drifted"]


@pytest.mark.parametrize("name", ["page_hinkley", "fixed_window"] + (["adwin"] if CM.HAS_RIVER else []))
def test_detector_finds_error_jump(name):
    errs = np.r_[rng.random(2000) < 0.1, rng.random(2000) < 0.4].astype(float)
    det = CM.make_detector(name)
    alarms = [i for i, e in enumerate(errs) if det.update(e)]
    assert alarms and alarms[0] >= 1900      # no early false alarm, detects the jump


def test_stream_benchmark_adaptive_beats_static():
    df, cfg = scenario("sudden", seed=42)
    static = run_stream(df, cfg, "fixed_window", adapt=False)
    adaptive = run_stream(df, cfg, "fixed_window", adapt=True)
    assert static["detected"] and static["false_alarms"] == 0
    assert adaptive["post_drift_acc"] > static["post_drift_acc"] + 0.02
    assert any(r["promoted"] for r in adaptive["retrains"])


def test_stable_stream_few_false_alarms():
    df, cfg = scenario("stable", seed=7)
    assert run_stream(df, cfg, "adwin")["false_alarms"] <= 2


def test_decide_rule():
    assert decide(0.80, 0.85, "f1_macro")["promote"]
    assert not decide(0.80, 0.805, "f1_macro")["promote"]
    assert decide(10.0, 8.0, "rmse")["promote"]         # lower is better


def test_champion_challenger_on_tabular(tmp_path):
    def make(n, flip):
        x = rng.normal(0, 1, n)
        y = np.where((x > 0) ^ flip, "Yes", "No")
        return pd.DataFrame({"x": x, "noise": rng.normal(0, 1, n), "y": y})
    res = run_automl(make(400, False), "y", models=["Logistic Regression"], n_trials=2, cv_folds=3)
    champ = reg.save_model(res, "toy", root=tmp_path)
    reg.set_status(champ, "champion", tmp_path)
    out = retrain_and_compare(champ, make(400, True), n_trials=2, root=tmp_path)   # concept flipped
    assert out["promote"] and out["promoted"]
    statuses = {m["model_id"]: m["status"] for m in reg.list_models(tmp_path)}
    assert statuses[out["challenger_id"]] == "champion" and statuses[champ] == "archived"
