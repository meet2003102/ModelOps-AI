import { useEffect, useState } from "react";
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api.js";
import { Badge, Button, Card, ErrorBox, ModelPicker, PageTitle, Spinner, Table, fmt, useModels } from "../components/ui.jsx";

export default function Explain() {
  const { models } = useModels();
  const [mid, setMid] = useState("");
  const [meta, setMeta] = useState(null);
  const [global, setGlobal] = useState(null);
  const [row, setRow] = useState(0);
  const [local, setLocal] = useState(null);
  const [sensitive, setSensitive] = useState([]);
  const [fair, setFair] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!mid) return;
    setGlobal(null); setLocal(null); setFair(null); setError(null);
    api.model(mid).then(setMeta).catch(setError);
    api.explainGlobal(mid).then(setGlobal).catch(setError);
  }, [mid]);

  const explainRow = () => api.explainLocal(mid, { test_row: Number(row) }).then(setLocal).catch(setError);
  const runFairness = () => api.fairness(mid, sensitive).then(setFair).catch(setError);
  useEffect(() => { if (mid) explainRow(); }, [mid]); // eslint-disable-line

  const candidateSensitive = meta ? [...meta.features.categorical, ...Object.keys(meta.features.dropped)] : [];

  return (
    <>
      <PageTitle title="Explainability" subtitle="SHAP explanations computed from the model — text is generated from the values, not by an LLM" />
      <div className="max-w-xl mb-6"><ModelPicker models={models} value={mid} onChange={setMid} /></div>
      <ErrorBox error={error} />

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        <Card title="Global feature importance" actions={global && <Badge color="indigo">{global.method}</Badge>}>
          {global ? (
            <ResponsiveContainer width="100%" height={Math.max(220, global.importance.length * 30)}>
              <BarChart data={global.importance} layout="vertical" margin={{ left: 40 }}>
                <XAxis type="number" fontSize={11} />
                <YAxis type="category" dataKey="feature" width={130} fontSize={12} />
                <Tooltip formatter={(v, n, p) => [`${fmt(v)} (${(p.payload.share * 100).toFixed(1)}%)`, "mean |SHAP|"]} />
                <Bar dataKey="mean_abs_shap" fill="#6366f1" />
              </BarChart>
            </ResponsiveContainer>
          ) : <Spinner text="Computing SHAP values…" />}
        </Card>

        <Card title="Explain one prediction" actions={
          <div className="flex gap-2 items-center text-sm">test row
            <input type="number" min={0} value={row} onChange={(e) => setRow(e.target.value)} className="w-20 border rounded px-2 py-1" />
            <Button variant="secondary" onClick={explainRow}>Explain</Button>
          </div>}>
          {local ? <LocalView e={local} /> : <Spinner />}
        </Card>
      </div>

      {meta?.task === "classification" && (
        <Card title="Fairness by group">
          <p className="text-sm text-slate-500 mb-3">Pick sensitive attributes. Columns excluded from training can still be audited.</p>
          <div className="flex flex-wrap gap-2 mb-3">
            {candidateSensitive.map((c) => (
              <label key={c} className={`px-3 py-1 rounded-full border text-xs cursor-pointer ${sensitive.includes(c) ? "bg-indigo-50 border-indigo-400" : "border-slate-300"}`}>
                <input type="checkbox" className="mr-1" checked={sensitive.includes(c)}
                  onChange={() => setSensitive(sensitive.includes(c) ? sensitive.filter((x) => x !== c) : [...sensitive, c])} />{c}
              </label>
            ))}
          </div>
          <Button disabled={!sensitive.length} onClick={runFairness}>Run fairness report</Button>
          {fair && Object.entries(fair.attributes).map(([col, r]) => (
            <div key={col} className="mt-5">
              <div className="flex gap-2 items-center mb-2">
                <b className="text-sm">{col}</b>
                <Badge color={r.flagged ? "red" : "green"}>{r.flagged ? "flagged" : "ok"}</Badge>
                <span className="text-xs text-slate-500">demographic parity diff {r.demographic_parity_difference} · equal opportunity diff {r.equal_opportunity_difference} (threshold {r.threshold}, positive = "{fair.positive_label}")</span>
              </div>
              <Table rows={r.groups} columns={[{ key: "group" }, { key: "n" }, { key: "selection_rate", label: "selection rate" },
                { key: "tpr", label: "TPR" }, { key: "fpr", label: "FPR" }, { key: "accuracy" }, { key: "f1" },
                { key: "small_group", label: "small", render: (v) => (v ? "yes" : "") }]} />
            </div>
          ))}
        </Card>
      )}
    </>
  );
}

export function LocalView({ e }) {
  const data = [...e.contributions].reverse();
  return (
    <>
      <div className="flex flex-wrap gap-2 mb-3">
        <Badge color="indigo">prediction: {String(e.prediction)}</Badge>
        {e.task === "classification" && <Badge>P({e.explained_class}) = {(e.score * 100).toFixed(1)}%</Badge>}
        {e.base_value !== null && <Badge>base value {fmt(e.base_value, 3)}</Badge>}
        <Badge>{e.method}</Badge>
      </div>
      <ResponsiveContainer width="100%" height={Math.max(200, data.length * 30)}>
        <BarChart data={data} layout="vertical" margin={{ left: 40 }}>
          <XAxis type="number" fontSize={11} />
          <YAxis type="category" dataKey="feature" width={130} fontSize={12} />
          <ReferenceLine x={0} stroke="#64748b" />
          <Tooltip formatter={(v, n, p) => [fmt(v), `value = ${p.payload.value}`]} />
          <Bar dataKey="contribution">
            {data.map((d, i) => <Cell key={i} fill={d.contribution >= 0 ? "#f43f5e" : "#3b82f6"} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-sm bg-slate-50 rounded-lg p-3 mt-2">{e.explanation_text}</p>
      <p className="text-xs text-slate-400 mt-1">Red pushes towards "{e.explained_class ?? "higher"}", blue pushes away.</p>
    </>
  );
}
