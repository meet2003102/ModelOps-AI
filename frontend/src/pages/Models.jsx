import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api.js";
import { Badge, Button, Card, ErrorBox, PageTitle, Spinner, Table, fmt, useModels } from "../components/ui.jsx";
import { TrainingResult } from "./AutoML.jsx";

export default function Models() {
  const { models, error, reload } = useModels();
  const [selected, setSelected] = useState(null);
  const [compare, setCompare] = useState([]);

  useEffect(() => { if (!selected && models.length) setSelected(models[0].model_id); }, [models, selected]);

  const promote = async (id) => { await api.promote(id); reload(); };
  const toggle = (id) => setCompare(compare.includes(id) ? compare.filter((x) => x !== id) : [...compare, id].slice(-3));

  return (
    <>
      <PageTitle title="Model Registry" subtitle="Every trained version, its status and metrics" />
      <ErrorBox error={error} />
      <Card title="Versions" className="mb-6">
        <Table rows={models} columns={[
          { key: "model_id", label: "", render: (id) => <input type="checkbox" checked={compare.includes(id)} onChange={() => toggle(id)} title="compare" /> },
          { key: "status", render: (v) => <Badge color={v === "champion" ? "green" : v === "archived" ? "slate" : "amber"}>{v}</Badge> },
          { key: "dataset" }, { key: "best_model", label: "model" }, { key: "task" }, { key: "metric" },
          { key: "test_metrics", label: "test score", render: (v, r) => fmt(v?.[r.metric]) },
          { key: "created_at", render: (v) => v?.slice(0, 19).replace("T", " ") },
          { key: "id", label: "actions", render: (_, r) => (
            <div className="flex gap-2">
              <Button variant="secondary" onClick={() => setSelected(r.model_id)}>Details</Button>
              {r.status !== "champion" && <Button variant="secondary" onClick={() => promote(r.model_id)}>Promote</Button>}
              <a href={api.downloadUrl(r.model_id)}><Button variant="secondary">Download</Button></a>
            </div>) },
        ]} />
      </Card>
      {compare.length >= 2 && <Compare ids={compare} />}
      {selected && <ModelDetail id={selected} />}
    </>
  );
}

function Compare({ ids }) {
  const [metas, setMetas] = useState([]);
  useEffect(() => { Promise.all(ids.map(api.model)).then(setMetas); }, [ids]);
  if (!metas.length) return <Spinner />;
  const keys = [...new Set(metas.flatMap((m) => Object.keys(m.test_metrics).filter((k) => typeof m.test_metrics[k] === "number")))];
  const rows = keys.map((k) => ({ metric: k, ...Object.fromEntries(metas.map((m) => [m.model_id, m.test_metrics[k]])) }));
  return (
    <Card title="Version comparison (held-out test metrics)" className="mb-6">
      <Table rows={rows} columns={[{ key: "metric" }, ...metas.map((m) => ({ key: m.model_id, label: `${m.best_model} · ${m.model_id.slice(-6)}` }))]} />
    </Card>
  );
}

function ModelDetail({ id }) {
  const [m, setM] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => { setM(null); api.model(id).then(setM).catch(setError); }, [id]);
  if (error) return <ErrorBox error={error} />;
  if (!m) return <Spinner />;
  const t = m.test_metrics;
  return (
    <div className="space-y-6">
      <TrainingResult r={{ ...m, model_id: m.model_id }} />
      {m.task === "classification" && (
        <div className="grid md:grid-cols-3 gap-6">
          <Card title="Confusion matrix"><Confusion cm={t.confusion_matrix} classes={m.classes} /></Card>
          {t.roc_curve && (
            <Card title={`ROC curve (AUC ${fmt(t.roc_auc, 3)})`}>
              <Curve data={t.roc_curve.fpr.map((x, i) => ({ x, y: t.roc_curve.tpr[i] }))} xLabel="FPR" yLabel="TPR" diagonal />
            </Card>
          )}
          {t.calibration && (
            <Card title={`Calibration (Brier ${fmt(t.brier, 3)})`}>
              <Curve data={t.calibration.mean_predicted.map((x, i) => ({ x, y: t.calibration.fraction_positive[i] }))} xLabel="predicted" yLabel="observed" diagonal />
            </Card>
          )}
        </div>
      )}
      <Card title="Features used">
        <div className="text-sm space-y-1">
          <div><b>Numeric:</b> {m.features.numeric.join(", ") || "–"}</div>
          <div><b>Categorical:</b> {m.features.categorical.join(", ") || "–"}</div>
          <div><b>Date:</b> {m.features.datetime.join(", ") || "–"}</div>
          <div><b>Dropped:</b> {Object.entries(m.features.dropped).map(([k, v]) => `${k} (${v})`).join(", ") || "–"}</div>
          {Object.keys(m.best_params || {}).length > 0 && <div><b>Tuned params:</b> <code className="text-xs">{JSON.stringify(m.best_params)}</code></div>}
        </div>
      </Card>
    </div>
  );
}

function Confusion({ cm, classes }) {
  const max = Math.max(...cm.flat());
  return (
    <table className="text-sm mx-auto">
      <thead><tr><th className="text-xs text-slate-500 p-1">true \ pred</th>{classes.map((c) => <th key={c} className="p-1">{c}</th>)}</tr></thead>
      <tbody>{cm.map((row, i) => (
        <tr key={i}><th className="p-1 text-right">{classes[i]}</th>
          {row.map((v, j) => <td key={j} className="w-16 h-12 text-center font-medium"
            style={{ background: `rgba(99,102,241,${v / max})`, color: v / max > 0.5 ? "white" : "#0f172a" }}>{v}</td>)}
        </tr>))}</tbody>
    </table>
  );
}

function Curve({ data, xLabel, yLabel, diagonal }) {
  const d = diagonal ? data.map((p) => ({ ...p, ref: p.x })) : data;
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={d}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" type="number" domain={[0, 1]} fontSize={11} label={{ value: xLabel, position: "insideBottom", offset: -2, fontSize: 11 }} />
        <YAxis domain={[0, 1]} fontSize={11} label={{ value: yLabel, angle: -90, position: "insideLeft", fontSize: 11 }} />
        <Tooltip formatter={(v) => fmt(v, 3)} />
        <Line dataKey="y" stroke="#6366f1" dot={false} strokeWidth={2} />
        {diagonal && <Line dataKey="ref" stroke="#94a3b8" strokeDasharray="4 4" dot={false} />}
      </LineChart>
    </ResponsiveContainer>
  );
}
