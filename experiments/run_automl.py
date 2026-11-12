"""Run AutoML from the command line and save the result.

Examples:
  python experiments/run_automl.py datasets/sample/sample_churn.csv churn
  python experiments/run_automl.py datasets/adult.csv income --metric roc_auc --exclude sex race
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_engine.registry.model_registry import champion_id, save_model, set_status  # noqa: E402
from ml_engine.training.automl import run_automl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("target")
ap.add_argument("--metric")
ap.add_argument("--trials", type=int, default=20)
ap.add_argument("--exclude", nargs="*", default=None)
a = ap.parse_args()

df = pd.read_csv(a.csv)
res = run_automl(df, a.target, metric=a.metric, n_trials=a.trials, exclude=a.exclude,
                 progress=lambda p, m: print(f"[{p:5.0%}] {m}"))
print("\nLeaderboard (CV on training data):")
cols = ["rank", "model", res.metric, f"{res.metric}_std", "tuned", "fit_seconds"]
print(pd.DataFrame(res.leaderboard)[cols].to_string(index=False))
print(f"\nBest: {res.best_model_name}")
print("Test metrics:", json.dumps({k: v for k, v in res.test_metrics.items()
                                   if not isinstance(v, (list, dict))}, indent=2))
print("Notes:", *res.notes, sep="\n  ")
mid = save_model(res, Path(a.csv).stem)
if champion_id(Path(a.csv).stem) is None:      # first model for this dataset becomes champion
    set_status(mid, "champion")
    print("Saved model:", mid, "(champion)")
else:
    print("Saved model:", mid, "(candidate - promote it via retraining or the API)")
