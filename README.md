# ModelOps AI

**An intelligent end-to-end machine learning platform with automated model selection, explainable AI, drift monitoring and LLM-powered insights.**

Upload a CSV → validated profile and EDA → AutoML trains and tunes 5 algorithms without data leakage → SHAP explains the best model → it is served through a REST API with every prediction logged → data drift and concept drift are monitored → a challenger is retrained and promoted only if it measurably beats the champion → a local LLM writes reports grounded in those measured results.

What makes it different from an AutoML tool: **it watches the model after deployment, explains why it degrades, and repairs it under fixed, testable rules.** The LLM describes results; it never chooses or promotes models.

![stack](https://img.shields.io/badge/Python-FastAPI%20·%20scikit--learn%20·%20LightGBM%20·%20SHAP%20·%20River-blue) ![stack](https://img.shields.io/badge/Frontend-React%20·%20Vite%20·%20Tailwind%20·%20Recharts-indigo) ![llm](https://img.shields.io/badge/LLM-Qwen2.5%20via%20Ollama-green)

## Architecture

```
React (8 pages) ──► FastAPI /api/v1 ──► ml_engine (pure Python)          llm_engine
                      │                   ingestion → EDA → preprocessing   facts (JSON) → Qwen2.5
                      │                   AutoML + Optuna → evaluation      number check / template fallback
                      │                   SHAP + fairness → registry
                      │                   drift (PSI/KS, ADWIN) → champion–challenger
                      └── SQLite: datasets, jobs, predictions (+ true labels), alerts
                          artifacts/: model versions (pipeline, reference sample, reports)
```

| Module | Location | What it does |
|---|---|---|
| 1 Ingestion & validation | `ml_engine/ingestion` | type/size/schema checks, column-type inference, duplicates, missing values, target suggestion |
| 2 Automated EDA | `ml_engine/eda` | distributions, correlations, outliers (IQR), target & class imbalance |
| 3 AutoML | `ml_engine/training` | 5 models + baseline per task, 5-fold CV on train only, Optuna tuning of the top 2, single test evaluation |
| 4 Explainable AI | `ml_engine/explainability`, `evaluation/fairness.py` | SHAP global/local (Tree/Linear/Permutation explainer), one-hot mapped back to original features, explanation stability, group fairness |
| 5 Drift & retraining | `ml_engine/drift`, `ml_engine/retraining` | PSI + KS/χ² per feature, SHAP-weighted drift impact, Evidently HTML, ADWIN on the error stream, fixed-window baseline, SEA drift benchmark, champion–challenger promotion rule |
| 6 LLM insights | `llm_engine` | facts builder, Qwen2.5 via Ollama, automatic unsupported-number check, template fallback |
| 7 Serving | `backend` | 31 REST endpoints, background jobs, prediction log + feedback, model registry |

## Quick start (Windows / PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m pytest -q                       # all tests
```

Run the app (two terminals):

```powershell
uvicorn backend.app.main:app --reload     # API  -> http://127.0.0.1:8000/docs
cd frontend; npm install; npm run dev     # UI   -> http://localhost:5173
```

Optional LLM (the app works without it, using template reports). The best tested model is `qwen2.5:7b`:

```powershell
ollama pull qwen2.5:7b                    # or qwen2.5:3b on smaller machines
$env:OLLAMA_MODEL="qwen2.5:7b"            # before starting uvicorn
```

Docker: `docker compose up --build` (UI on http://localhost:8080), add `--profile llm` to start Ollama too.

## Reproduce the experiments

```powershell
python experiments/run_automl.py datasets/sample/sample_churn.csv churn
python experiments/run_explain.py --sensitive gender
python experiments/make_production_data.py
python experiments/run_drift.py --current datasets/sample/churn_production.csv --retrain
python experiments/run_drift.py --benchmark                 # -> experiments/drift/benchmark_summary.csv
python experiments/llm_eval/run_llm_eval.py --ungrounded    # -> experiments/llm_eval/summary.json
```

For the public datasets: Adult Census Income (OpenML id 1590; train with `--exclude sex race` and audit fairness with `--sensitive sex race`) and California Housing (`sklearn.datasets.fetch_california_housing`). Raw datasets are not committed.

## Key design decisions

- **No data leakage:** preprocessing is inside `Pipeline(pre → model)`, refit in every CV fold; the test set is scored once.
- **Data drift ≠ worse model:** data drift (PSI ≥ 0.2, or significant test and PSI ≥ 0.1) only raises an alert; performance claims need labels (concept drift).
- **Drift impact = PSI × SHAP importance share**, so drift in features the model relies on is ranked first.
- **Deterministic promotion:** challenger trained on older new data, both scored on the same most-recent 25 %; promoted only if gain ≥ 0.01.
- **Grounded LLM:** the LLM sees only structured facts; every number it writes is checked against them; it is never claimed to be hallucination-free — it is measured.
- **Simulated shifts are labelled as simulated** (`make_production_data.py`, SEA streams with documented parameters and seeds).

## Results (sample run, fill in your own)

| Experiment | Result |
|---|---|
| AutoML on churn sample | LightGBM best, test ROC-AUC 0.786, F1 0.718 (baseline F1 0.424) |
| SHAP | top drivers: contract, tenure_months, monthly_charges; stability 80 % |
| Drift benchmark (3 seeds) | ADWIN: 0 false alarms on stable streams, 100 % detection; post-drift accuracy 0.875–0.883 with retraining vs 0.831 static |
| Churn production data | concept drift detected; challenger F1 0.571 → 0.671, promoted |
| LLM, grounding effect (24 generated questions, 3B) | 4.2 % without facts → 83.3 % with facts; + scope rules and ranked facts → 91.7 %; + per-question fact selection → 100 % |
| LLM, unseen test set (35 questions, written before the run) | qwen2.5:3b **80.0 %** · qwen2.5:7b **94.3 %**; refusals 73 % → 100 %; 0 invented numbers; 4.44 s vs 11.79 s per answer (GTX 1650, 8 GB RAM) |

## LLM evaluation method and limitations

- Expected answers are read from the platform's facts (`experiments/llm_eval/*.json`), so they cannot contain typos.
- Questions about reports that were never run, other datasets, other time periods or money must be refused.
- Fixes were developed on a development set; the final 35-question set was written before the final run and run once per model.
- The automatic number check catches invented numbers but **not real numbers used in the wrong context**, which is why the question-based evaluation is needed.
- Remaining 7B errors: one self-contradicting yes/no answer, and one caused by a fact-routing bug that was fixed after the test (not re-scored on the same set).
- Small question sets on one model version: the results are indicative, not a general guarantee.

## Project structure

```
ml_engine/      ingestion · eda · preprocessing · training · evaluation · explainability · drift · retraining · registry
llm_engine/     client · context_builder · insight_generator · prompt_templates/
backend/app/    main · config · db · services · api/ (datasets, training, models, predict, monitoring, insights)
frontend/src/   pages/ (Dashboard, Datasets, AutoML, Models, Explain, Predict, Monitor, Assistant) · components/ · api.js
experiments/    run_automl · run_explain · make_production_data · run_drift · llm_eval/
tests/          unit/ (engine, XAI, fairness, drift, LLM) · api/ (end-to-end)
```
