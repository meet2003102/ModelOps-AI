import { useEffect, useState } from "react";
import { Bar, BarChart, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, waitForJob } from "../api.js";
import { Badge, Button, Card, ErrorBox, FileInput, ModelPicker, PageTitle, Spinner, Table, fmt, useModels } from "../components/ui.jsx";

const sevColor = { major: "red", moderate: "amber", none: "green" };

export default function Monitor() {
  const { models, reload } = useModels();
  const [mid, setMid] = useState("");
  const [file, setFile] = useState(null);
  const [drift, setDrift] = useState(null);
  const [concept, setConcept] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [busy, setBusy] = useState(false);
  const [retrainFile, setRetrainFile] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);

  const refresh = (id) => {
    api.alerts(id).then(setAlerts).catch(setError);
    api.conceptDrift(id).then(setConcept).catch(setError);
  };
  useEffect(() => { if (mid) { setDrift(null); setJob(null); refresh(mid); } }, [mid]);

  const checkDrift = async (useLog) => {
    setBusy(true); setError(null);
    try { setDrift(await api.dataDrift(mid, useLog ? null : file)); refresh(mid); } catch (e) { setError(e.message); }
    setBusy(false);
  };

  const retrain = async () => {
    setError(null);
    try {
      const j = await api.retrain(mid, retrainFile, 0.01);
      setJob(j);
      const done = await waitForJob(j.id, setJob);
      if (done.status === "failed") setError(done.error);
      reload();
    } catch (e) { setError(e.message); }
  };

  const rows = drift ? drift.ranking.map((f) => ({ feature: f, ...drift.features[f] })) : [];

  return (
    <>
      <PageTitle title="Drift Monitor" subtitle="Data drift (PSI, KS/χ², Evidently) · concept drift (ADWIN on errors) · champion–challenger retraining" />
      <div className="max-w-xl mb-6"><ModelPicker models={models} value={mid} onChange={setMid} /></div>
      <ErrorBox error={error} />

      <Card title="1 · Data drift — has the input distribution changed?" className="mb-6">
        <div className="flex flex-wrap items-center gap-4">
          <FileInput onFile={setFile} />
          <Button disabled={!file || busy} onClick={() => checkDrift(false)}>Check uploaded file</Button>
          <Button variant="secondary" disabled={busy} onClick={() => checkDrift(true)}>Check logged predictions</Button>
        </div>
        {busy && <Spinner text="Comparing with the training reference…" />}
        {drift && (
          <div className="mt-5">
            <div className="flex flex-wrap gap-2 mb-3 items-center">
              <Badge color={drift.dataset_drift ? "red" : drift.drifted_features.length ? "amber" : "green"}>
                {drift.dataset_drift ? "dataset drift" : drift.drifted_features.length ? "some features drifted" : "no drift"}</Badge>
              <span className="text-sm text-slate-600">{drift.drifted_features.length} of {rows.length} features drifted ({(drift.share_drifted * 100).toFixed(0)}%) · reference {drift.n_reference} rows vs current {drift.n_current}</span>
              {drift.evidently_html && <a className="text-sm text-indigo-600 underline" href={api.driftHtmlUrl(mid)} target="_blank" rel="noreferrer">Open Evidently report</a>}
            </div>
            <p className="text-sm text-slate-500 mb-2">Ranked by <b>impact = PSI × SHAP importance share</b>: drift in features the model relies on matters most.</p>
            <div className="grid lg:grid-cols-2 gap-6">
              <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 28)}>
                <BarChart data={rows} layout="vertical" margin={{ left: 40 }}>
                  <XAxis type="number" fontSize={11} /><YAxis type="category" dataKey="feature" width={130} fontSize={12} />
                  <Tooltip formatter={(v) => fmt(v, 4)} />
                  <Bar dataKey={drift.ranking_by} name={drift.ranking_by} fill="#f59e0b" />
                </BarChart>
              </ResponsiveContainer>
              <Table rows={rows} columns={[
                { key: "feature" }, { key: "psi", label: "PSI" },
                { key: "severity", render: (v) => <Badge color={sevColor[v]}>{v}</Badge> },
                { key: "test" }, { key: "p_value", label: "p", render: (v) => (v < 0.001 ? "<0.001" : fmt(v, 3)) },
                { key: "importance_share", label: "importance" }, { key: "drifted", render: (v) => (v ? "yes" : "") },
              ]} />
            </div>
          </div>
        )}
      </Card>

      <Card title="2 · Concept drift — is the model getting worse? (needs true outcomes)" className="mb-6">
        {!concept ? <Spinner /> : concept.concept_drift === null ? (
          <p className="text-sm text-slate-500">{concept.message} Add outcomes in the Prediction Studio history, or use retraining with a labelled file below.</p>
        ) : (
          <>
            <div className="flex gap-2 items-center mb-3">
              <Badge color={concept.concept_drift ? "red" : "green"}>{concept.concept_drift ? "concept drift detected" : "stable"}</Badge>
              <span className="text-sm text-slate-600">{concept.detector} over {concept.n} labelled predictions · error rate {(concept.error_rate * 100).toFixed(1)}%{concept.alarms.length ? ` · first alarm at #${concept.alarms[0]}` : ""}</span>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={concept.rolling_error.map((v, i) => ({ i, error: v }))}>
                <XAxis dataKey="i" fontSize={11} /><YAxis fontSize={11} domain={[0, "auto"]} /><Tooltip />
                <Line dataKey="error" stroke="#f43f5e" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </>
        )}
      </Card>

      <Card title="3 · Champion–challenger retraining" className="mb-6">
        <p className="text-sm text-slate-500 mb-3">Upload recent data <b>with the true target column</b>. A challenger is trained on the older part and both models are scored on the most recent 25%. It is promoted only if it wins by ≥ 0.01 — a fixed rule, never the LLM.</p>
        <div className="flex items-center gap-4">
          <FileInput onFile={setRetrainFile} />
          <Button disabled={!retrainFile || (job && job.status === "running")} onClick={retrain}>Retrain & compare</Button>
        </div>
        {job && job.status !== "done" && job.status !== "failed" && <Spinner text={job.message || "Training challenger…"} />}
        {job?.status === "done" && (
          <div className="mt-4 grid grid-cols-3 gap-4 text-center">
            <div className="rounded-lg bg-slate-50 p-4"><div className="text-xs text-slate-500">champion {job.result.metric}</div><div className="text-2xl font-semibold">{fmt(job.result.champion)}</div></div>
            <div className="rounded-lg bg-slate-50 p-4"><div className="text-xs text-slate-500">challenger {job.result.metric}</div><div className="text-2xl font-semibold">{fmt(job.result.challenger)}</div></div>
            <div className={`rounded-lg p-4 ${job.result.promoted ? "bg-emerald-50" : "bg-amber-50"}`}>
              <div className="text-xs text-slate-500">decision (gain {fmt(job.result.gain)})</div>
              <div className="text-lg font-semibold">{job.result.promoted ? `Promoted ${job.result.challenger_id}` : "Kept champion"}</div>
            </div>
          </div>
        )}
      </Card>

      <Card title="Alert history">
        {alerts.length ? <Table rows={alerts} columns={[{ key: "created_at", render: (v) => v?.slice(0, 19).replace("T", " ") },
          { key: "kind" }, { key: "severity", render: (v) => <Badge color={sevColor[v] || "slate"}>{v}</Badge> }, { key: "summary" }]} />
          : <p className="text-sm text-slate-500">No alerts for this model.</p>}
      </Card>
    </>
  );
}
