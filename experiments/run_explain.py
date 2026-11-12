"""Explain a saved model and (optionally) report fairness.

Examples:
  python experiments/run_explain.py                      # latest saved model
  python experiments/run_explain.py --model 20261112-103058-1c0940 --row 5
  python experiments/run_explain.py --sensitive sex race   # fairness by group
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_engine.evaluation.fairness import fairness_for_model  # noqa: E402
from ml_engine.explainability.shap_explainer import (explain_global, explain_local,  # noqa: E402
                                                     explanation_stability)
from llm_engine.context_builder import save_report  # noqa: E402
from ml_engine.registry import model_registry as reg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", help="model id (default: latest)")
ap.add_argument("--row", type=int, default=0, help="test row to explain")
ap.add_argument("--sensitive", nargs="*", default=[], help="columns for the fairness report")
ap.add_argument("--positive", help="positive class label for fairness (default: last class)")
a = ap.parse_args()

models = reg.list_models()
if not models:
    sys.exit("No saved models. Run experiments/run_automl.py first.")
mid = a.model or models[0]["model_id"]
bundle = reg.load_model(mid)
meta = bundle["metadata"]
task, classes = meta["task"], meta.get("classes")
X_ref = reg.load_reference(mid)
test_raw = reg.load_test_raw(mid)
X_test = test_raw[bundle["features"]]

print(f"Model {mid}: {meta['best_model']} ({task})\n")
g = explain_global(bundle["pipeline"], X_test, task, n_classes=len(classes or [0, 0]))
save_report(mid, "global_explanation", g)
print(f"Global feature importance  [{g['method']}, {g['n_rows']} rows]")
for i in g["importance"]:
    print(f"  {i['feature']:<22} {i['mean_abs_shap']:>9.4f}   {i['share']:>6.1%}  {'#' * int(i['share'] * 50)}")

loc = explain_local(bundle["pipeline"], X_test.iloc[[a.row]], X_ref, task, classes)
print(f"\nLocal explanation for test row {a.row}  [{loc['method']}]")
for c in loc["contributions"]:
    sign = "+" if c["contribution"] > 0 else "-" if c["contribution"] < 0 else " "
    print(f"  {sign} {c['feature']:<22} = {str(c['value']):<18} {c['contribution']:+.4f}")
print("\n" + loc["explanation_text"])

stab = explanation_stability(bundle["pipeline"], X_test.iloc[[a.row]], X_ref, task, classes, repeats=5)
print(f"\nExplanation stability (top-5 overlap under 1% noise): {stab:.0%}")

if a.sensitive:
    rep = fairness_for_model(bundle, test_raw, meta["features"]["target"], a.sensitive, a.positive)
    print(f"\nFairness report (positive class = '{rep['positive_label']}')")
    for col, r in rep["attributes"].items():
        print(f"\n  {col}: DP diff {r['demographic_parity_difference']}, "
              f"EO diff {r['equal_opportunity_difference']}  {'FLAGGED' if r['flagged'] else 'ok'}")
        for grp in r["groups"]:
            print(f"    {grp['group']:<15} n={grp['n']:<5} sel={grp['selection_rate']:<7} "
                  f"tpr={grp['tpr']:<7} f1={grp['f1']}")
    Path("artifacts/models", mid, "fairness.json").write_text(json.dumps(rep, indent=2))
