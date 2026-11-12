// Thin client for the FastAPI backend. All ML logic stays on the server.
const BASE = "/api/v1";

async function request(path, options = {}) {
  const res = await fetch(BASE + path, options);
  const type = res.headers.get("content-type") || "";
  const body = type.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    let msg = typeof body === "string" ? body : body.detail;
    if (typeof msg === "object") msg = msg.message ? `${msg.message}: ${(msg.errors || []).join("; ")}` : JSON.stringify(msg);
    throw new Error(msg || `Request failed (${res.status})`);
  }
  return body;
}

const json = (method, data) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data),
});

const form = (fields) => {
  const fd = new FormData();
  Object.entries(fields).forEach(([k, v]) => v !== undefined && v !== null && fd.append(k, v));
  return { method: "POST", body: fd };
};

export const api = {
  health: () => request("/health"),
  // datasets
  datasets: () => request("/datasets"),
  dataset: (id) => request(`/datasets/${id}`),
  uploadDataset: (file, name) => request("/datasets", form({ file, name })),
  profile: (id) => request(`/datasets/${id}/profile`),
  preview: (id, rows = 20) => request(`/datasets/${id}/preview?rows=${rows}`),
  eda: (id, target) => request(`/datasets/${id}/eda${target ? `?target=${encodeURIComponent(target)}` : ""}`),
  // training
  trainingOptions: () => request("/training/options"),
  startTraining: (body) => request("/training/jobs", json("POST", body)),
  job: (id) => request(`/jobs/${id}`),
  jobs: () => request("/jobs"),
  // models
  models: () => request("/models"),
  model: (id) => request(`/models/${id}`),
  schema: (id) => request(`/models/${id}/schema`),
  promote: (id) => request(`/models/${id}/promote`, { method: "POST" }),
  downloadUrl: (id) => `${BASE}/models/${id}/download`,
  explainGlobal: (id) => request(`/models/${id}/explain/global`),
  explainLocal: (id, body) => request(`/models/${id}/explain/local`, json("POST", body)),
  fairness: (id, cols, positive) =>
    request(`/models/${id}/fairness?${cols.map((c) => `sensitive=${encodeURIComponent(c)}`).join("&")}` +
      (positive ? `&positive=${encodeURIComponent(positive)}` : "")),
  // prediction
  predict: (id, records) => request(`/models/${id}/predict`, json("POST", { records })),
  predictBatch: (id, file) => request(`/models/${id}/predict/batch`, form({ file })),
  batchCsvUrl: (id) => `${BASE}/models/${id}/predict/batch?as_csv=true`,
  predictions: (id, limit = 50) => request(`/predictions?model_id=${id}&limit=${limit}`),
  feedback: (items) => request("/predictions/feedback", json("POST", { items })),
  // monitoring
  dataDrift: (id, file) => request(`/monitoring/${id}/data-drift`, form({ file })),
  driftHtmlUrl: (id) => `${BASE}/monitoring/${id}/data-drift/html`,
  conceptDrift: (id) => request(`/monitoring/${id}/concept-drift`),
  alerts: (id) => request(`/monitoring/${id}/alerts`),
  retrain: (id, file, minGain) => request(`/monitoring/${id}/retrain`, form({ file, min_gain: minGain })),
  // insights (Step 6)
  insightsStatus: () => request("/insights/status"),
  insightReport: (body) => request("/insights/report", json("POST", body)),
  ask: (body) => request("/insights/ask", json("POST", body)),
};

// Poll a background job until it finishes. onUpdate gets every status.
export async function waitForJob(id, onUpdate, intervalMs = 1000) {
  for (;;) {
    const j = await api.job(id);
    onUpdate?.(j);
    if (j.status === "done" || j.status === "failed") return j;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
