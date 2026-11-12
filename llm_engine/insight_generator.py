"""Module 6 - LLM-powered insights, grounded in platform facts.

report(model_id)          plain-language model report
ask(model_id, question)   answer one question about the model

If Ollama is not running, a deterministic template report is returned instead
(source = "template"), so the platform never depends on the LLM.
Every LLM answer is checked: numbers it mentions that do not occur in the facts
are listed as `unsupported_numbers` (shown in the UI and used in the LLM evaluation).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from llm_engine import client
from llm_engine.context_builder import build_facts, select_facts

SYSTEM = (Path(__file__).parent / "prompt_templates" / "system.txt").read_text(encoding="utf-8")


# ------------------------------------------------------------------ number check
def _fact_numbers(obj, out=None) -> set[float]:
    out = set() if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            _fact_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _fact_numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(float(obj))
    elif isinstance(obj, str):
        for m in re.findall(r"\d+(?:\.\d+)?", obj):
            out.add(float(m))
    return out


def check_numbers(text: str, facts: dict, small_int_ok: int = 10) -> dict:
    """Which numbers in `text` cannot be traced to the facts (with rounding / % tolerance)."""
    known = _fact_numbers(facts)
    found = re.findall(r"(?<![\w.])(-?\d+(?:\.\d+)?)(\s*%)?", text)
    unsupported = []
    for num, pct in found:
        v = float(num)
        if v.is_integer() and abs(v) <= small_int_ok:
            continue                      # "top 3", "5-fold", list numbering ...
        # "78.6%" or "78.6" may stand for 0.786; a plain 0.991 is only compared as itself
        candidates = [v] + ([v / 100] if pct or abs(v) > 1 else [])

        def close(c, k):
            tol = 0.0051 if abs(k) >= 0.1 else max(0.0006, 0.06 * abs(k))   # rounding to 2-3 decimals
            return abs(c - k) <= max(tol, 0.001 * abs(k))
        ok = any(close(c, k) for c in candidates for k in known)
        if not ok:
            unsupported.append(num + (pct or "").strip())
    return {"numbers_checked": len(found), "unsupported_numbers": sorted(set(unsupported))}


# ------------------------------------------------------------------ template fallback
def template_report(f: dict) -> str:
    m, t = f["model"], f["test_metrics"]
    sel = m["selection_metric"]
    lines = [f"Model {m['id']}: {m['algorithm']} ({m['task']}) for target '{m['target']}', status {m['status']}.",
             "", "Performance (held-out test set):"]
    lines += [f"- {k}: {v}" for k, v in t.items()]
    lb = f["leaderboard_cv"]
    if len(lb) > 1:
        lines += ["", f"It ranked first by cross-validated {sel} ({lb[0][sel]}); "
                      f"runner-up: {lb[1]['model']} ({lb[1][sel]})."]
    if "top_features" in f:
        top = ", ".join(f"{x['feature']} ({x['importance_share']:.0%})" for x in f["top_features"][:5])
        lines += ["", f"Most important features (SHAP): {top}."]
    if "data_drift" in f:
        d = f["data_drift"]
        names = ", ".join(x["feature"] for x in d["drifted_features_sorted_by_psi"][:5]) or "none"
        lines += ["", f"Data drift: {d['share_drifted']:.0%} of features drifted "
                      f"(dataset drift: {'yes' if d['dataset_drift'] else 'no'}). Largest PSI first: {names}.",
                  "Data drift alone does not prove the model got worse."]
    if "concept_drift" in f:
        c = f["concept_drift"]
        lines += ["", f"Concept drift ({c['detector']} on {c['labelled_rows']} labelled rows): "
                      f"{'detected' if c['detected'] else 'not detected'}, error rate {c['error_rate']:.1%}."]
    if "last_retraining" in f:
        r = f["last_retraining"]
        lines += ["", f"Last retraining: champion {r['metric']} {r['champion']} vs challenger {r['challenger']} "
                      f"-> {'challenger promoted' if r['promoted'] else 'champion kept'} (rule: gain >= {r['min_gain']})."]
    if "fairness" in f:
        for col, r in f["fairness"].items():
            lines.append(f"Fairness by {col}: parity diff {r['demographic_parity_difference']}, "
                         f"{'flagged' if r['flagged'] else 'within threshold'}.")
    if f.get("data_quality_notes"):
        lines += ["", "Data preparation notes:"] + [f"- {n}" for n in f["data_quality_notes"]]
    lines += ["", "Suggested checks:"]
    if f.get("data_drift", {}).get("drifted_features_sorted_by_psi"):
        lines.append("- Investigate why the top drifted features changed (new customer segment, pipeline change?).")
    if "concept_drift" not in f:
        lines.append("- Add true outcomes to logged predictions so concept drift can be measured.")
    lines.append("- Review the SHAP explanations of a few wrong predictions.")
    return "\n".join(lines)


def template_answer(f: dict, q: str) -> str:
    ql = q.lower()
    if any(w in ql for w in ("drift", "change", "shift")):
        parts = []
        if "data_drift" in f:
            d = f["data_drift"]
            parts.append("Drifted features (largest PSI first): " + (", ".join(f"{x['feature']} (PSI {x['psi']})" for x in d["drifted_features_sorted_by_psi"]) or "none") + ".")
        if "concept_drift" in f:
            parts.append(f"Concept drift detected: {f['concept_drift']['detected']} (error rate {f['concept_drift']['error_rate']}).")
        return " ".join(parts) or "No drift report is available yet - run a drift check first."
    if any(w in ql for w in ("feature", "important", "why", "driver")):
        if "top_features" in f:
            return "Top features by SHAP importance: " + ", ".join(
                f"{x['feature']} ({x['importance_share']:.0%})" for x in f["top_features"]) + "."
        return "No explanation report is available yet - open the Explainability page first."
    if any(w in ql for w in ("fair", "bias", "gender", "sex", "race")):
        return json.dumps(f.get("fairness", "No fairness report available yet."))
    return "Test metrics: " + ", ".join(f"{k} {v}" for k, v in f["test_metrics"].items()) + "."


# ------------------------------------------------------------------ public API
REPORT_PROMPT = """FACTS:
{facts}

Write a model report with these sections:
1. Summary (2 sentences)
2. Performance - key test metrics and how the model compared with the other candidates
3. What drives predictions - the top features
4. Monitoring - data drift, concept drift and retraining, only if present in the FACTS
5. Recommendations - 2 to 4 concrete checks a data scientist should do next"""

ASK_PROMPT = """FACTS:
{facts}

QUESTION: {question}

Answer in at most 6 sentences, using only the FACTS."""


def _run(facts: dict, prompt: str, fallback: str) -> dict:
    st = client.status()
    if not st["available"]:
        return {"text": fallback, "source": "template", "llm_status": st["message"],
                "check": check_numbers(fallback, facts), "latency_s": None}
    try:
        text, secs = client.chat(SYSTEM, prompt)
    except client.LLMUnavailable as e:
        return {"text": fallback, "source": "template", "llm_status": str(e),
                "check": check_numbers(fallback, facts), "latency_s": None}
    return {"text": text, "source": f"llm:{st['model']}", "llm_status": "ok",
            "check": check_numbers(text, facts), "latency_s": secs}


def report(model_id: str, facts: dict | None = None) -> dict:
    facts = facts or build_facts(model_id)
    out = _run(facts, REPORT_PROMPT.format(facts=json.dumps(facts, indent=1)), template_report(facts))
    return {**out, "facts_used": facts}


def ask(model_id: str, question: str, facts: dict | None = None, focused: bool = True) -> dict:
    """focused=True sends only the fact sections relevant to the question."""
    facts = facts or build_facts(model_id)
    sent = select_facts(facts, question) if focused else facts
    out = _run(sent, ASK_PROMPT.format(facts=json.dumps(sent, indent=1), question=question[:500]),
               template_answer(facts, question))
    out["check"] = check_numbers(out["text"], facts)     # verify against ALL facts
    return {"answer": out.pop("text"), **out, "facts_used": sent, "focused": focused}
