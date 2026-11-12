import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api.js";
import { Badge, Button, Card, ErrorBox, FileInput, PageTitle, Select, Spinner, Stat, Table, fmt } from "../components/ui.jsx";

export default function Datasets() {
  const [list, setList] = useState([]);
  const [selected, setSelected] = useState(null);
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const reload = () => api.datasets().then((d) => { setList(d); if (!selected && d.length) setSelected(d[0].id); }).catch(setError);
  useEffect(() => { reload(); }, []); // eslint-disable-line

  const upload = async () => {
    setBusy(true); setError(null);
    try {
      const ds = await api.uploadDataset(file);
      setFile(null); await reload(); setSelected(ds.id);
    } catch (e) { setError(e.message); }
    setBusy(false);
  };

  return (
    <>
      <PageTitle title="Dataset Manager" subtitle="Upload a CSV/XLSX; it is validated and profiled automatically" />
      <Card title="Upload dataset" className="mb-6">
        <div className="flex items-center gap-4">
          <FileInput onFile={setFile} />
          <Button disabled={!file || busy} onClick={upload}>{busy ? "Uploading…" : "Upload & validate"}</Button>
        </div>
        <ErrorBox error={error} />
      </Card>

      {list.length > 0 && (
        <Card title="Datasets" className="mb-6">
          <Table rows={list} columns={[
            { key: "name" }, { key: "n_rows", label: "rows" }, { key: "n_cols", label: "columns" },
            { key: "suggested_target", label: "suggested target" },
            { key: "warnings", render: (w) => <Badge color={w.length ? "amber" : "green"}>{w.length}</Badge> },
            { key: "id", label: "", render: (id) => <Button variant="secondary" onClick={() => setSelected(id)}>{id === selected ? "Viewing" : "View"}</Button> },
          ]} />
        </Card>
      )}
      {selected && <DatasetDetail id={selected} />}
    </>
  );
}

function DatasetDetail({ id }) {
  const [ds, setDs] = useState(null);
  const [profile, setProfile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [eda, setEda] = useState(null);
  const [target, setTarget] = useState("");
  const [col, setCol] = useState("");
  const [error, setError] = useState(null);

  useEffect(() => {
    setEda(null);
    Promise.all([api.dataset(id), api.profile(id), api.preview(id, 20)]).then(([d, p, pr]) => {
      setDs(d); setProfile(p); setPreview(pr); setTarget(d.suggested_target || ""); setCol(p.numeric_columns[0] || pr.columns[0]);
    }).catch(setError);
  }, [id]);
  useEffect(() => { if (ds) api.eda(id, target || undefined).then(setEda).catch(setError); }, [id, target, ds]);

  if (error) return <ErrorBox error={error} />;
  if (!profile || !preview) return <Spinner />;

  const dist = eda && (eda.numeric_distributions[col]
    ? eda.numeric_distributions[col].counts.map((c, i) => ({ x: fmt(eda.numeric_distributions[col].bins[i], 2), count: c }))
    : Object.entries(eda.categorical_distributions[col] || {}).map(([x, count]) => ({ x, count })));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <Stat label="Rows" value={profile.n_rows} />
        <Stat label="Columns" value={profile.n_cols} />
        <Stat label="Missing" value={`${profile.total_missing_pct}%`} />
        <Stat label="Duplicates" value={profile.duplicate_rows} />
        <Stat label="Memory" value={`${profile.memory_mb} MB`} />
      </div>

      <Card title={`Validation & data quality (${(ds.warnings || []).length} notes)`}>
        {(ds.warnings || []).length ? (
          <ul className="text-sm space-y-1 list-disc pl-5 text-amber-800">{ds.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
        ) : <p className="text-sm text-emerald-700">No data-quality issues found.</p>}
      </Card>

      <Card title="Columns">
        <Table rows={profile.columns} columns={[
          { key: "name" }, { key: "inferred_type", label: "type", render: (v) => <Badge color={v === "numeric" ? "indigo" : v === "id" || v === "constant" ? "red" : "slate"}>{v}</Badge> },
          { key: "missing_pct", label: "missing %" }, { key: "unique" }, { key: "mean" }, { key: "std" },
          { key: "min" }, { key: "max" }, { key: "outliers" }]} />
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        <Card title="Distribution" actions={<div className="w-56"><Select value={col} onChange={setCol} options={preview.columns} /></div>}>
          {dist ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={dist}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="x" fontSize={11} /><YAxis fontSize={11} /><Tooltip /><Bar dataKey="count" fill="#6366f1" /></BarChart>
            </ResponsiveContainer>
          ) : <Spinner />}
        </Card>

        <Card title="Target" actions={<div className="w-56"><Select value={target} onChange={setTarget} options={preview.columns} placeholder="(none)" /></div>}>
          {eda?.target ? <TargetView t={eda.target} /> : <p className="text-sm text-slate-500">Pick a target column.</p>}
        </Card>
      </div>

      {eda?.correlation && (
        <Card title="Correlation heatmap (numeric features)"><Heatmap corr={eda.correlation} /></Card>
      )}

      <Card title="Preview (first 20 rows)">
        <Table rows={preview.rows} columns={preview.columns.map((c) => ({ key: c }))} maxHeight={360} />
      </Card>
    </div>
  );
}

function TargetView({ t }) {
  if (t.problem_type === "classification") {
    const data = Object.entries(t.class_counts).map(([k, v]) => ({ x: k, count: v }));
    return (
      <>
        <div className="flex gap-2 mb-2"><Badge color="indigo">classification</Badge>
          {t.imbalance_ratio > 3 && <Badge color="amber">imbalanced {t.imbalance_ratio}:1</Badge>}</div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data}><XAxis dataKey="x" /><YAxis /><Tooltip /><Bar dataKey="count" fill="#10b981" /></BarChart>
        </ResponsiveContainer>
      </>
    );
  }
  const data = t.distribution.counts.map((c, i) => ({ x: fmt(t.distribution.bins[i], 2), count: c }));
  return (
    <>
      <Badge color="indigo">regression · mean {fmt(t.mean, 3)}</Badge>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data}><XAxis dataKey="x" fontSize={11} /><YAxis /><Tooltip /><Bar dataKey="count" fill="#10b981" /></BarChart>
      </ResponsiveContainer>
    </>
  );
}

function Heatmap({ corr }) {
  const color = (v) => {
    const a = Math.min(Math.abs(v), 1);
    return v >= 0 ? `rgba(99,102,241,${a})` : `rgba(244,63,94,${a})`;
  };
  return (
    <div className="overflow-auto">
      <table className="text-xs">
        <thead><tr><th />{corr.columns.map((c) => <th key={c} className="px-1 py-1 font-medium text-slate-600 rotate-0">{c}</th>)}</tr></thead>
        <tbody>
          {corr.matrix.map((row, i) => (
            <tr key={i}>
              <td className="pr-2 font-medium text-slate-600 whitespace-nowrap">{corr.columns[i]}</td>
              {row.map((v, j) => (
                <td key={j} title={`${corr.columns[i]} × ${corr.columns[j]}: ${v}`}
                  className="w-12 h-9 text-center" style={{ background: color(v), color: Math.abs(v) > 0.5 ? "white" : "#0f172a" }}>
                  {v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
