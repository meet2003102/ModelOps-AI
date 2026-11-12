"""Re-score saved runs with the current scorer WITHOUT calling the LLM again.
Use after a scoring fix (e.g. number words like "five"); the LLM answers are unchanged.
  python experiments/llm_eval/rescore.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
sys.argv = [sys.argv[0]]                       # import the scorer without running the eval
src = (OUT / "run_llm_eval.py").read_text(encoding="utf-8").split("\nap = argparse.ArgumentParser()")[0]
ns = {"__file__": str(OUT / "run_llm_eval.py")}
exec(compile(src, "run_llm_eval.py", "exec"), ns)   # defines correct(), NUM_WORDS, ...
correct = ns["correct"]


def parse_expected(v, kind):
    if kind == "refusal" or pd.isna(v):
        return None
    if kind == "number":
        return float(v)
    if kind == "yesno":
        return str(v) == "True"
    s = str(v)
    if s.startswith("["):
        return json.loads(s.replace("'", '"'))
    return s


rows = []
for f in sorted(OUT.glob("results_*.csv")):
    df = pd.read_csv(f)
    old = df["grounded_correct"].mean()
    df["grounded_correct"] = [correct({"kind": k, "expected": parse_expected(e, k)}, str(a))
                              for k, e, a in zip(df["kind"], df["expected"], df["grounded_answer"])]
    df.to_csv(f, index=False)
    tag = f.stem.replace("results_", "")
    sp = OUT / f"summary_{tag}.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        s["grounded_accuracy"] = round(df["grounded_correct"].mean(), 3)
        s["refusal_accuracy"] = round(df[df.kind == "refusal"]["grounded_correct"].mean(), 3)
        s["rescored"] = True
        sp.write_text(json.dumps(s, indent=2))
    rows.append({"tag": tag, "n": len(df), "before_rescore": round(old, 3),
                 "after_rescore": round(df["grounded_correct"].mean(), 3)})
print(pd.DataFrame(rows).to_string(index=False))
