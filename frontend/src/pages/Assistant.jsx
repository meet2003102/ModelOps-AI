import { useEffect, useState } from "react";
import { api } from "../api.js";
import { Badge, Button, Card, ErrorBox, ModelPicker, PageTitle, Spinner, useModels } from "../components/ui.jsx";

// Talks to /insights (Step 6). The page works even when the LLM is offline:
// the backend then returns a template-based report built from the same facts.
export default function Assistant() {
  const { models } = useModels();
  const [mid, setMid] = useState("");
  const [status, setStatus] = useState(null);
  const [report, setReport] = useState(null);
  const [question, setQuestion] = useState("");
  const [chat, setChat] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => { api.insightsStatus().then(setStatus).catch(() => setStatus({ available: false, message: "Insights API not available yet" })); }, []);

  const makeReport = async () => {
    setBusy(true); setError(null);
    try { setReport(await api.insightReport({ model_id: mid })); } catch (e) { setError(e.message); }
    setBusy(false);
  };
  const ask = async () => {
    if (!question.trim()) return;
    const q = question; setQuestion(""); setBusy(true); setError(null);
    try {
      const r = await api.ask({ model_id: mid, question: q });
      setChat([...chat, { q, a: r.answer, source: r.source, facts: r.facts_used }]);
    } catch (e) { setError(e.message); }
    setBusy(false);
  };

  return (
    <>
      <PageTitle title="AI Assistant" subtitle="Plain-language summaries grounded only in the platform's measured results" />
      <div className="flex items-end gap-4 mb-6">
        <div className="w-full max-w-xl"><ModelPicker models={models} value={mid} onChange={setMid} /></div>
        {status && <Badge color={status.available ? "green" : "amber"}>{status.available ? `LLM: ${status.model}` : "LLM offline · template mode"}</Badge>}
      </div>
      <ErrorBox error={error} />
      <div className="grid lg:grid-cols-2 gap-6">
        <Card title="Model report" actions={<Button disabled={!mid || busy} onClick={makeReport}>Generate report</Button>}>
          {busy && !report && <Spinner text="Writing report…" />}
          {report ? (
            <>
              <div className="flex gap-2 mb-3"><Badge>{report.source}</Badge>{report.latency_s && <Badge>{report.latency_s}s</Badge>}
                {report.check && <Badge color={report.check.unsupported_numbers?.length ? "amber" : "green"}>
                  {report.check.unsupported_numbers?.length ? `${report.check.unsupported_numbers.length} unverified numbers` : "all numbers verified"}</Badge>}</div>
              <div className="text-sm whitespace-pre-wrap leading-relaxed">{report.text}</div>
            </>
          ) : !busy && <p className="text-sm text-slate-500">Summarises performance, important features, drift and data quality. Decisions (best model, retraining) stay rule-based.</p>}
        </Card>
        <Card title="Ask about this model">
          <div className="space-y-4 mb-4 max-h-[420px] overflow-auto">
            {chat.map((c, i) => (
              <div key={i}>
                <div className="text-sm font-medium">Q: {c.q}</div>
                <div className="text-sm bg-slate-50 rounded-lg p-3 mt-1 whitespace-pre-wrap">{c.a}</div>
                <div className="text-xs text-slate-400 mt-1">{c.source}</div>
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <input className="flex-1 border border-slate-300 rounded-lg px-3 py-2 text-sm" placeholder="e.g. Why did accuracy drop? Which features drifted?"
              value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask()} />
            <Button disabled={!mid || busy} onClick={ask}>Ask</Button>
          </div>
        </Card>
      </div>
    </>
  );
}
