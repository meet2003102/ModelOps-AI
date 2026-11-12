"""LLM evaluation: factual consistency, unsupported numbers, refusals, latency.

1. Builds ~30 questions whose correct answers come from the model's facts
   (saved to questions.json - REVIEW these by hand; that makes it your verified reference set).
2. Includes 3 unanswerable control questions: the right answer is "not available".
3. Asks each question (grounded = with facts). With --ungrounded, also asks without facts
   to show what grounding adds.

Run AFTER the drift step so the facts include drift:
  python experiments/llm_eval/run_llm_eval.py
  python experiments/llm_eval/run_llm_eval.py --ungrounded
  python experiments/llm_eval/run_llm_eval.py --mode full --tag full_facts   # ablation: no fact selection
  python experiments/llm_eval/run_llm_eval.py --questions experiments/llm_eval/heldout_questions.json
Results go to results_<tag>.csv / summary_<tag>.json; all summaries are compared at the end.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_engine import client  # noqa: E402
from llm_engine.context_builder import build_facts  # noqa: E402
from llm_engine.insight_generator import ask, check_numbers  # noqa: E402
from ml_engine.registry import model_registry as reg  # noqa: E402

OUT = Path(__file__).resolve().parent
REFUSAL = re.compile(r"not available|no information|not (in|included in|provided in) the (facts|reports?)|cannot (be )?determine|unknown", re.I)


def build_questions(f: dict) -> list[dict]:
    q = []
    add = lambda text, kind, expected: q.append({"id": len(q) + 1, "question": text, "kind": kind, "expected": expected})
    m = f["model"]
    add("Which algorithm was selected as the final model?", "text", m["algorithm"])
    add("What is the target column the model predicts?", "text", m["target"])
    add("Which metric was used to select the best model?", "text", m["selection_metric"])
    for k, v in f["test_metrics"].items():
        add(f"What is the model's {k} on the held-out test set?", "number", v)
    if len(f["leaderboard_cv"]) > 1:
        add("Which model came second in cross-validation?", "text", f["leaderboard_cv"][1]["model"])
    for i, t in enumerate(f.get("top_features", [])[:3]):
        add(["What is the most important feature?", "What is the second most important feature?",
             "What is the third most important feature?"][i], "text", t["feature"])
    if "data_drift" in f:
        d = f["data_drift"]
        add("Was dataset-level data drift detected? Answer yes or no first.", "yesno", d["dataset_drift"])
        drifted = d["drifted_features_sorted_by_psi"]
        if drifted:
            add("Which drifted feature has the largest PSI?", "text", drifted[0]["feature"])
        for x in drifted[:4]:
            add(f"What is the PSI of the feature {x['feature']}?", "number", x["psi"])
        add("How many rows were in the current data used for the drift check?", "number", d["n_current"])
    if "concept_drift" in f:
        c = f["concept_drift"]
        add("Was concept drift detected? Answer yes or no first.", "yesno", c["detected"])
        add("What error rate was measured on the labelled production data?", "number", c["error_rate"])
        add("Which detector was used for concept drift?", "text", c["detector"])
        add("How many labelled rows were monitored for concept drift?", "number", c["labelled_rows"])
    if "last_retraining" in f:
        r = f["last_retraining"]
        add("What score did the challenger model get in the last retraining?", "number", r["challenger"])
        add("Was the challenger promoted? Answer yes or no first.", "yesno", r["promoted"])
    for text in ("What was the model's accuracy last year?",
                 "How much revenue did the model save the company?",
                 "What is the model's performance on the Adult Census dataset?"):
        add(text, "refusal", None)
    return q


def resolve(facts: dict, path: str):
    """'top_features[0].feature' -> value in facts, or None if the report/field is missing."""
    cur = facts
    for part in path.split("."):
        m = re.fullmatch(r"([\w]+)((?:\[-?\d+\])*)", part)
        if not m or not isinstance(cur, dict) or m.group(1) not in cur:
            return None
        cur = cur[m.group(1)]
        for idx in re.findall(r"\[(-?\d+)\]", m.group(2)):
            if not isinstance(cur, list) or not -len(cur) <= int(idx) < len(cur):
                return None
            cur = cur[int(idx)]
    if isinstance(cur, dict):                 # e.g. dropped_columns -> its keys
        return list(cur)
    return cur


def load_handwritten(path: Path, facts: dict) -> list[dict]:
    """Hand-written questions. 'expected_fact' points into the facts (string or list of paths);
    if that part of the facts does not exist, the question becomes a refusal test."""
    items = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, it in enumerate(items, 1):
        q, kind, exp = it["question"], it.get("kind", "text"), it.get("expected")
        missing = False
        for ref in re.findall(r"\{([^}]+)\}", q):           # {path} placeholders in the question
            v = resolve(facts, ref)
            missing |= v is None
            q = q.replace("{" + ref + "}", str(v))
        if missing:
            continue                                         # cannot even phrase it for this model
        if "expected_compare" in it:                         # [path_a, ">" or "<", path_b] -> yes/no
            pa, op, pb = it["expected_compare"]
            va, vb = resolve(facts, pa), resolve(facts, pb)
            if va is None or vb is None:
                kind, exp = "refusal", None
            else:
                kind, exp = "yesno", (va > vb) if op == ">" else (va < vb)
        elif "expected_fact" in it:
            refs = it["expected_fact"] if isinstance(it["expected_fact"], list) else [it["expected_fact"]]
            vals = [resolve(facts, r) for r in refs]
            if any(v is None for v in vals):
                kind, exp = "refusal", None                  # the report does not exist -> must refuse
            else:
                flat = [x for v in vals for x in (v if isinstance(v, list) else [v])]
                exp = flat if len(flat) > 1 else flat[0]
        out.append({"id": i, "question": q, "kind": kind, "expected": exp})
    return out


NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}


def correct(item: dict, answer: str) -> bool:
    exp, a = item["expected"], answer.lower()
    if item["kind"] == "refusal":
        return bool(REFUSAL.search(answer))
    if item["kind"] == "yesno":
        first = re.findall(r"\b(yes|no)\b", a)
        return bool(first) and (first[0] == "yes") == bool(exp)
    if item["kind"] == "number":
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", answer.replace(",", ""))]
        nums += [float(NUM_WORDS[w]) for w in re.findall(r"[a-z]+", a) if w in NUM_WORDS]   # "five" = 5
        tol = 0.0051 if abs(exp) >= 0.1 else 0.0006
        return any(abs(n - exp) <= max(tol, 0.001 * abs(exp)) or abs(n / 100 - exp) <= tol for n in nums)
    if isinstance(exp, list):
        return all(str(e).lower() in a for e in exp)
    return str(exp).lower() in a


def ungrounded(model_name: str, question: str) -> tuple[str, float]:
    system = "You are an assistant for a machine-learning platform. Answer briefly."
    return client.chat(system, f"About the {model_name} model in our platform: {question}")


ap = argparse.ArgumentParser()
ap.add_argument("--model")
ap.add_argument("--ungrounded", action="store_true", help="also ask without facts (needs Ollama)")
ap.add_argument("--mode", choices=["focused", "full"], default="focused",
                help="focused = only relevant fact sections per question (default)")
ap.add_argument("--tag", help="name for the output files (default: the mode)")
ap.add_argument("--questions", help="JSON file of hand-written questions (e.g. experiments/llm_eval/heldout_questions.json)")
a = ap.parse_args()

mid = a.model or reg.champion_id() or reg.list_models()[0]["model_id"]
facts = build_facts(mid)
if a.questions:
    questions = load_handwritten(Path(a.questions), facts)
    (OUT / "questions_handwritten_resolved.json").write_text(json.dumps(questions, indent=2, default=str))
else:
    questions = build_questions(facts)
    (OUT / "questions.json").write_text(json.dumps(questions, indent=2, default=str))
st = client.status()
print(f"Model {mid}; {len(questions)} questions; LLM: {st['model'] if st['available'] else 'OFFLINE -> template answers'}\n")

rows = []
for it in questions:
    t0 = time.time()
    r = ask(mid, it["question"], facts=facts, focused=(a.mode == "focused"))
    row = {"id": it["id"], "kind": it["kind"], "question": it["question"], "expected": it["expected"],
           "grounded_answer": r["answer"], "grounded_correct": correct(it, r["answer"]),
           "grounded_unsupported": len(r["check"]["unsupported_numbers"]),
           "latency_s": r["latency_s"] or round(time.time() - t0, 2), "source": r["source"]}
    if a.ungrounded and st["available"]:
        ans, secs = ungrounded(facts["model"]["algorithm"], it["question"])
        row.update(ungrounded_answer=ans, ungrounded_correct=correct(it, ans),
                   ungrounded_unsupported=len(check_numbers(ans, facts)["unsupported_numbers"]))
    rows.append(row)
    print(f"[{it['id']:>2}] {'OK ' if row['grounded_correct'] else 'BAD'} {it['question'][:70]}")

df = pd.DataFrame(rows)
tag = a.tag or (f"heldout_{a.mode}" if a.questions else a.mode)
df.to_csv(OUT / f"results_{tag}.csv", index=False)
summary = {"tag": tag, "mode": a.mode, "model_id": mid, "source": df["source"].iloc[0], "n_questions": len(df),
           "grounded_accuracy": round(df["grounded_correct"].mean(), 3),
           "refusal_accuracy": round(df[df.kind == "refusal"]["grounded_correct"].mean(), 3),
           "answers_with_unsupported_numbers": int((df["grounded_unsupported"] > 0).sum()),
           "mean_latency_s": round(df["latency_s"].mean(), 2)}
if "ungrounded_correct" in df:
    summary.update(ungrounded_accuracy=round(df["ungrounded_correct"].mean(), 3),
                   ungrounded_answers_with_unsupported_numbers=int((df["ungrounded_unsupported"] > 0).sum()))
(OUT / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
print("\n" + json.dumps(summary, indent=2))
print(f"\nSaved questions.json, results_{tag}.csv, summary_{tag}.json in {OUT}")

runs = [json.loads(p.read_text()) for p in sorted(OUT.glob("summary_*.json"))]
if len(runs) > 1:
    cols = ["tag", "n_questions", "grounded_accuracy", "refusal_accuracy",
            "answers_with_unsupported_numbers", "mean_latency_s", "ungrounded_accuracy"]
    print("\nCOMPARISON OF RUNS")
    print(pd.DataFrame(runs).reindex(columns=cols).to_string(index=False))
