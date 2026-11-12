import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Badge, Card, ErrorBox, PageTitle, Stat, Table, fmt } from "../components/ui.jsx";

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.health(), api.datasets(), api.models(), api.jobs()])
      .then(async ([h, d, m, j]) => {
        setHealth(h); setDatasets(d); setModels(m); setJobs(j);
        const champs = m.filter((x) => x.status === "champion");
        const lists = await Promise.all(champs.map((c) => api.alerts(c.model_id).then((a) => a.map((x) => ({ ...x, model_id: c.model_id })))));
        setAlerts(lists.flat().slice(0, 8));
      })
      .catch(setError);
  }, []);

  const champions = models.filter((m) => m.status === "champion");
  return (
    <>
      <PageTitle title="Dashboard" subtitle="Overview of datasets, models and monitoring" />
      <ErrorBox error={error && `Cannot reach the API. Is "uvicorn backend.app.main:app" running? (${error})`} />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <Stat label="Datasets" value={datasets.length} />
        <Stat label="Models" value={models.length} />
        <Stat label="Champions" value={champions.length} />
        <Stat label="Open alerts" value={alerts.length} />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <Card title="Champion models">
          {champions.length ? (
            <Table columns={[
              { key: "dataset" }, { key: "best_model", label: "model" }, { key: "metric" },
              { key: "test_metrics", label: "test score", render: (v, r) => fmt(v?.[r.metric]) },
              { key: "model_id", label: "id" }]} rows={champions} />
          ) : <p className="text-sm text-slate-500">No models yet. <Link className="text-indigo-600" to="/automl">Train one →</Link></p>}
        </Card>

        <Card title="Recent drift alerts">
          {alerts.length ? (
            <ul className="space-y-2 text-sm">
              {alerts.map((a) => (
                <li key={`${a.model_id}-${a.id}`} className="flex gap-2 items-start">
                  <Badge color={a.severity === "major" ? "red" : "amber"}>{a.kind.replace("_", " ")}</Badge>
                  <span>{a.summary}</span>
                </li>
              ))}
            </ul>
          ) : <p className="text-sm text-slate-500">No alerts.</p>}
        </Card>

        <Card title="Recent jobs">
          <Table columns={[{ key: "kind" }, { key: "status", render: (v) => <Badge color={v === "done" ? "green" : v === "failed" ? "red" : "indigo"}>{v}</Badge> },
            { key: "message" }, { key: "created_at", render: (v) => v?.slice(0, 19).replace("T", " ") }]} rows={jobs.slice(0, 6)} />
        </Card>

        <Card title="Libraries on the server">
          <div className="flex flex-wrap gap-2">
            {health && Object.entries(health.libraries).map(([k, v]) => (
              <Badge key={k} color={v ? "green" : "red"}>{k}: {v ? "ok" : "missing"}</Badge>
            ))}
          </div>
        </Card>
      </div>
    </>
  );
}
