"""Drift monitoring for a saved model + the controlled stream benchmark.

Examples:
  python experiments/run_drift.py --current datasets/sample/churn_production.csv
  python experiments/run_drift.py --current datasets/sample/churn_production.csv --retrain
  python experiments/run_drift.py --benchmark
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_engine.drift.concept_monitor import monitor_labelled_batch  # noqa: E402
from ml_engine.drift.evidently_report import save_html_report  # noqa: E402
from ml_engine.drift.psi_ks import data_drift_report  # noqa: E402
from ml_engine.drift.stream_experiment import run_benchmark  # noqa: E402
from ml_engine.explainability.shap_explainer import explain_global  # noqa: E402
from ml_engine.registry import model_registry as reg  # noqa: E402
from ml_engine.retraining.champion_challenger import retrain_and_compare  # noqa: E402
from llm_engine.context_builder import save_report  # noqa: E402

pd.set_option("display.width", 200)
ap = argparse.ArgumentParser()
ap.add_argument("--model")
ap.add_argument("--current", help="new production CSV (with true labels for concept drift)")
ap.add_argument("--retrain", action="store_true", help="run champion-challenger retraining")
ap.add_argument("--benchmark", action="store_true", help="run the synthetic stream benchmark")
a = ap.parse_args()

if a.current:
    mid = a.model or reg.champion_id() or reg.list_models()[0]["model_id"]
    bundle = reg.load_model(mid)
    meta = bundle["metadata"]
    target, task = meta["features"]["target"], meta["task"]
    ref = reg.load_reference(mid)
    cur = pd.read_csv(a.current)
    feats = bundle["features"]

    imp = explain_global(bundle["pipeline"], ref, task, n_classes=len(meta.get("classes") or [0, 0]))
    shares = {i["feature"]: i["share"] for i in imp["importance"]}
    rep = data_drift_report(ref, cur, feats, importance=shares)
    print(f"Model {mid} ({meta['best_model']}) - reference {rep['n_reference']} rows, "
          f"current {rep['n_current']} rows\n")
    save_report(mid, "data_drift", rep)
    save_report(mid, "global_explanation", imp)
    print("DATA DRIFT (ranked by impact = PSI x SHAP importance share)")
    tbl = pd.DataFrame(rep["features"]).T.loc[rep["ranking"],
                                              ["type", "psi", "severity", "p_value", "importance_share", "impact", "drifted"]]
    print(tbl.to_string())
    print(f"\nDrifted features: {rep['drifted_features']}  ({rep['share_drifted']:.0%} of features)"
          f"  dataset drift: {rep['dataset_drift']}")
    print(f"Most important drifted features: {rep['top_important_drifted']}")
    html = save_html_report(ref, cur[feats], Path("artifacts/models") / mid / "evidently_drift.html")
    print(f"Evidently HTML report: {html or 'not created (Evidently unavailable)'}")

    out = {"data_drift": rep}
    if target in cur:
        known = cur[cur[target].notna()]
        pred = reg.predict(bundle, known)["prediction"]
        cd = monitor_labelled_batch(known[target].astype(str) if task == "classification" else known[target],
                                    pred.astype(str) if task == "classification" else pred, task)
        print(f"\nCONCEPT DRIFT ({cd['detector']} on the error stream of {cd['n']} labelled rows)")
        print(f"  error rate {cd['error_rate']:.1%}, alarms at rows {cd['alarms']}  ->  "
              f"{'CONCEPT DRIFT' if cd['concept_drift'] else 'no concept drift'}")
        out["concept_drift"] = cd
        save_report(mid, "concept_drift", cd)
        if a.retrain:
            print("\nCHAMPION-CHALLENGER RETRAINING ...")
            r = retrain_and_compare(mid, known)
            print(f"  {r['metric']} on {r['holdout_rows']} most recent rows: champion {r['champion']} vs "
                  f"challenger {r['challenger']} (gain {r['gain']:+.4f}, needed {r['min_gain']})")
            print(f"  -> {'PROMOTED ' + r['challenger_id'] if r['promoted'] else 'kept champion'}")
            out["retraining"] = r
            save_report(mid, "retraining", r)
    Path("artifacts/models", mid, "monitoring.json").write_text(json.dumps(out, indent=2, default=str))

if a.benchmark:
    print("\nSTREAM BENCHMARK (SEA generator, seeds 42, 7, 123) ...")
    df = run_benchmark(seeds=(42, 7, 123))
    Path("experiments/drift").mkdir(parents=True, exist_ok=True)
    df.to_csv("experiments/drift/benchmark_results.csv", index=False)
    summary = (df.groupby(["scenario", "detector", "adaptive"])
               .agg(alarms=("n_alarms", "mean"), false_alarms=("false_alarms", "mean"),
                    detection_rate=("detected", lambda s: s.dropna().astype(float).mean() if s.notna().any() else None),
                    delay=("delay", "mean"), post_drift_acc=("post_drift_acc", "mean"),
                    recovery=("recovery", "mean"), overall_acc=("overall_accuracy", "mean"))
               .round(3))
    print(summary.to_string())
    summary.to_csv("experiments/drift/benchmark_summary.csv")
    print("\nSaved experiments/drift/benchmark_results.csv and benchmark_summary.csv")
