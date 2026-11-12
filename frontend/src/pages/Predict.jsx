import { useEffect, useState } from "react";
import { api } from "../api.js";
import { Badge, Button, Card, ErrorBox, FileInput, ModelPicker, PageTitle, Table, fmt, useModels } from "../components/ui.jsx";
import { LocalView } from "./Explain.jsx";

export default function Predict() {
  const { models } = useModels();
  const [mid, setMid] = useState("");
  const [schema, setSchema] = useState(null);
  const [record, setRecord] = useState({});
  const [result, setResult] = useState(null);
  const [why, setWhy] = useState(null);
  const [batchFile, setBatchFile] = useState(null);
  const [batch, setBatch] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  const loadHistory = (id) => api.predictions(id, 50).then(setHistory).catch(setError);
  useEffect(() => {
    if (!mid) return;
    setResult(null); setWhy(null); setBatch(null);
    api.schema(mid).then((s) => { setSchema(s); setRecord({ ...s.examples[0] }); }).catch(setError);
    loadHistory(mid);
  }, [mid]);

  const predictOne = async () => {
    setError(null);
    try {
      const rec = Object.fromEntries(Object.entries(record).map(([k, v]) =>
        [k, schema.numeric.includes(k) && v !== "" && v !== null ? Number(v) : v]));
      const r = await api.predict(mid, [rec]);
      setResult(r.predictions[0]);
      setWhy(await api.explainLocal(mid, { record: rec }));
      loadHistory(mid);
    } catch (e) { setError(e.message); }
  };

  const runBatch = async () => {
    setError(null);
    try { setBatch(await api.predictBatch(mid, batchFile)); loadHistory(mid); } catch (e) { setError(e.message); }
  };

  const sendFeedback = async (id, actual) => {
    try { await api.feedback([{ prediction_id: id, actual }]); loadHistory(mid); } catch (e) { setError(e.message); }
  };

  return (
    <>
      <PageTitle title="Prediction Studio" subtitle="Single and batch predictions; every prediction is logged for monitoring" />
      <div className="max-w-xl mb-6"><ModelPicker models={models} value={mid} onChange={setMid} /></div>
      <ErrorBox error={error} />

      {schema && (
        <div className="grid lg:grid-cols-2 gap-6 mb-6">
          <Card title="Single prediction" actions={
            <div className="flex gap-2">{schema.examples.map((ex, i) => (
              <Button key={i} variant="secondary" onClick={() => setRecord({ ...ex })}>Example {i + 1}</Button>))}</div>}>
            <div className="grid grid-cols-2 gap-3">
              {schema.features.map((f) => (
                <label key={f} className="text-sm">
                  <span className="text-slate-600">{f}</span>
                  {schema.categories[f] ? (
                    <select className="mt-1 block w-full border border-slate-300 rounded-lg px-2 py-1.5 bg-white"
                      value={record[f] ?? ""} onChange={(e) => setRecord({ ...record, [f]: e.target.value })}>
                      {schema.categories[f].map((o) => <option key={o}>{o}</option>)}
                    </select>
                  ) : (
                    <input className="mt-1 block w-full border border-slate-300 rounded-lg px-2 py-1.5"
                      type={schema.numeric.includes(f) ? "number" : schema.datetime.includes(f) ? "date" : "text"}
                      value={record[f] ?? ""} onChange={(e) => setRecord({ ...record, [f]: e.target.value })} />
                  )}
                </label>
              ))}
            </div>
            <Button className="mt-4" onClick={predictOne}>Predict</Button>
          </Card>

          <Card title="Result & explanation">
            {result ? (
              <>
                <div className="text-3xl font-semibold mb-1">{String(result.prediction)}</div>
                {result.confidence !== undefined && <div className="text-sm text-slate-500 mb-4">confidence {(result.confidence * 100).toFixed(1)}% · prediction #{result.prediction_id}</div>}
                {why && <LocalView e={why} />}
              </>
            ) : <p className="text-sm text-slate-500">Fill the form and click Predict.</p>}
          </Card>
        </div>
      )}

      <Card title="Batch prediction" className="mb-6">
        <div className="flex items-center gap-4">
          <FileInput onFile={setBatchFile} />
          <Button disabled={!batchFile || !mid} onClick={runBatch}>Predict file</Button>
        </div>
        {batch && (
          <div className="mt-4">
            <div className="text-sm text-slate-600 mb-2">{batch.n} rows predicted (showing first 100).</div>
            <Table rows={batch.predictions.slice(0, 100)} columns={Object.keys(batch.predictions[0] || {}).map((k) => ({ key: k }))} maxHeight={320} />
          </div>
        )}
      </Card>

      <Card title="Prediction history (latest 50) — add the true outcome to enable concept-drift monitoring">
        <Table rows={history} columns={[
          { key: "id" }, { key: "created_at", render: (v) => v?.slice(0, 19).replace("T", " ") },
          { key: "source" }, { key: "prediction" }, { key: "confidence", render: (v) => fmt(v, 3) },
          { key: "actual", label: "true outcome", render: (v, r) => v ? <Badge color={v === r.prediction ? "green" : "red"}>{v}</Badge> : (
            schema?.classes ? (
              <select className="border rounded px-1 text-xs" defaultValue="" onChange={(e) => e.target.value && sendFeedback(r.id, e.target.value)}>
                <option value="">set…</option>{schema.classes.map((c) => <option key={c}>{c}</option>)}
              </select>
            ) : <input className="border rounded px-1 text-xs w-24" placeholder="value, Enter"
                onKeyDown={(e) => e.key === "Enter" && e.target.value && sendFeedback(r.id, e.target.value)} />) },
        ]} />
      </Card>
    </>
  );
}
