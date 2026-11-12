import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, waitForJob } from "../api.js";
import { Badge, Button, Card, ErrorBox, PageTitle, Select, Table, fmt } from "../components/ui.jsx";

export default function AutoML() {
  const [datasets, setDatasets] = useState([]);
  const [options, setOptions] = useState(null);
  const [dsId, setDsId] = useState("");
  const [columns, setColumns] = useState([]);
  const [target, setTarget] = useState("");
  const [task, setTask] = useState("");
  const [metric, setMetric] = useState("");
  const [chosen, setChosen] = useState([]);
  const [exclude, setExclude] = useState([]);
  const [trials, setTrials] = useState(20);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.datasets(), api.trainingOptions()]).then(([d, o]) => {
      setDatasets(d); setOptions(o); if (d.length) setDsId(d[0].id);
    }).catch(setError);
  }, []);

  useEffect(() => {
    if (!dsId) return;
    api.profile(dsId).then((p) => {
      setColumns(p.columns.map((c) => c.name));
      setTarget(p.suggested_target || "");
      setExclude([]);
    }).catch(setError);
  }, [dsId]);

  // problem type follows the target unless the user overrides it
  useEffect(() => {
    if (!dsId || !target) return;
    api.eda(dsId, target).then((e) => setTask(e.target.problem_type)).catch(() => {});
  }, [dsId, target]);

  useEffect(() => {
    if (!options || !task) return;
    setChosen(options[task].models);
    setMetric(task === "classification" ? "f1_macro" : "rmse");
  }, [task, options]);

  const toggle = (list, setList, v) => setList(list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);

  const start = async () => {
    setError(null);
    try {
      const j = await api.startTraining({ dataset_id: dsId, target, task, metric, models: chosen,
        exclude: exclude.length ? exclude : null, n_trials: Number(trials) });
      setJob(j);
      const done = await waitForJob(j.id, setJob);
      if (done.status === "failed") setError(done.error);
    } catch (e) { setError(e.message); }
  };

  const running = job && (job.status === "queued" || job.status === "running");
  const result = job?.status === "done" ? job.result : null;

  return (
    <>
      <PageTitle title="AutoML Studio" subtitle="Train, tune and compare models with leakage-free cross-validation" />
      <Card title="Configuration" className="mb-6">
        <div className="grid md:grid-cols-4 gap-4">
          <Select label="Dataset" value={dsId} onChange={setDsId} options={datasets.map((d) => ({ value: d.id, label: `${d.name} (${d.n_rows} rows)` }))} />
          <Select label="Target column" value={target} onChange={setTarget} options={columns} placeholder="Choose…" />
          <Select label="Problem type" value={task} onChange={setTask} options={["classification", "regression"]} />
          <Select label="Selection metric" value={metric} onChange={setMetric} options={options && task ? options[task].metrics : []} />
        </div>

        {options && task && (
          <div className="mt-5">
            <div className="text-sm text-slate-600 mb-2">Models (a majority/mean baseline is always included)</div>
            <div className="flex flex-wrap gap-2">
              {options[task].models.filter((m) => !m.startsWith("Baseline")).map((m) => (
                <label key={m} className={`px-3 py-1.5 rounded-full border text-sm cursor-pointer ${chosen.includes(m) ? "bg-indigo-50 border-indigo-400" : "border-slate-300"}`}>
                  <input type="checkbox" className="mr-2" checked={chosen.includes(m)} onChange={() => toggle(chosen, setChosen, m)} />{m}
                </label>
              ))}
            </div>
          </div>
        )}

        <div className="mt-5">
          <div className="text-sm text-slate-600 mb-2">Exclude columns from training (e.g. sensitive attributes such as sex or race)</div>
          <div className="flex flex-wrap gap-2">
            {columns.filter((c) => c !== target).map((c) => (
              <label key={c} className={`px-3 py-1 rounded-full border text-xs cursor-pointer ${exclude.includes(c) ? "bg-rose-50 border-rose-400" : "border-slate-300"}`}>
                <input type="checkbox" className="mr-1" checked={exclude.includes(c)} onChange={() => toggle(exclude, setExclude, c)} />{c}
              </label>
            ))}
          </div>
        </div>

        <div className="flex items-end gap-4 mt-5">
          <label className="text-sm">
            <span className="text-slate-600">Tuning trials (Optuna)</span>
            <input type="number" min={0} max={200} value={trials} onChange={(e) => setTrials(e.target.value)}
              className="mt-1 block w-32 rounded-lg border border-slate-300 px-3 py-2" />
          </label>
          <Button disabled={!dsId || !target || !task || !chosen.length || running} onClick={start}>
            {running ? "Training…" : "Start training"}
          </Button>
        </div>
        <ErrorBox error={error} />
      </Card>

      {job && (
        <Card title="Progress" className="mb-6">
          <div className="flex items-center gap-3 mb-2">
            <Badge color={job.status === "done" ? "green" : job.status === "failed" ? "red" : "indigo"}>{job.status}</Badge>
            <span className="text-sm text-slate-600">{job.message}</span>
          </div>
          <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
            <div className="h-full bg-indigo-500 transition-all" style={{ width: `${Math.round((job.progress || 0) * 100)}%` }} />
          </div>
        </Card>
      )}

      {result && <TrainingResult r={result} />}
    </>
  );
}

export function TrainingResult({ r }) {
  const m = r.metric;
  return (
    <Card title={`Leaderboard — best: ${r.best_model}`}
      actions={<Link to="/models" className="text-sm text-indigo-600">Open in Model Registry →</Link>}>
      <p className="text-sm text-slate-500 mb-3">Mean ± std of 5-fold CV on the training split. Model id <code>{r.model_id}</code>, trained in {r.train_seconds}s.</p>
      <Table rows={r.leaderboard} columns={[
        { key: "rank" }, { key: "model" },
        { key: m, label: `${m} (CV)`, render: (v, row) => `${fmt(v)} ± ${fmt(row[`${m}_std`])}` },
        { key: "tuned", render: (v) => (v ? <Badge color="indigo">tuned</Badge> : "") },
        { key: "fit_seconds", label: "fit s" },
      ]} />
      <h3 className="font-medium mt-5 mb-2 text-sm">Held-out test set (scored once)</h3>
      <div className="flex flex-wrap gap-2">
        {Object.entries(r.test_metrics).filter(([, v]) => typeof v === "number").map(([k, v]) => (
          <Badge key={k} color="slate">{k}: {fmt(v)}</Badge>
        ))}
      </div>
      {r.notes?.length > 0 && <ul className="text-xs text-slate-500 mt-4 list-disc pl-5">{r.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>}
    </Card>
  );
}
