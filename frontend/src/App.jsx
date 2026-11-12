import { NavLink, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import Datasets from "./pages/Datasets.jsx";
import AutoML from "./pages/AutoML.jsx";
import Models from "./pages/Models.jsx";
import Explain from "./pages/Explain.jsx";
import Predict from "./pages/Predict.jsx";
import Monitor from "./pages/Monitor.jsx";
import Assistant from "./pages/Assistant.jsx";

const NAV = [
  ["/", "Dashboard"],
  ["/datasets", "Dataset Manager"],
  ["/automl", "AutoML Studio"],
  ["/models", "Model Registry"],
  ["/explain", "Explainability"],
  ["/predict", "Prediction Studio"],
  ["/monitor", "Drift Monitor"],
  ["/assistant", "AI Assistant"],
];

export default function App() {
  return (
    <div className="flex min-h-screen">
      <aside className="w-60 shrink-0 bg-slate-900 text-slate-200 p-4 flex flex-col">
        <div className="text-lg font-bold text-white mb-1">ModelOps AI</div>
        <div className="text-xs text-slate-400 mb-6">AutoML · XAI · Drift · LLM</div>
        <nav className="flex flex-col gap-1">
          {NAV.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === "/"}
              className={({ isActive }) =>
                `px-3 py-2 rounded-lg text-sm ${isActive ? "bg-indigo-600 text-white" : "hover:bg-slate-800"}`}>
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto text-xs text-slate-500">API: /api/v1 · <a className="underline" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">docs</a></div>
      </aside>
      <main className="flex-1 p-8 overflow-x-hidden">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/automl" element={<AutoML />} />
          <Route path="/models" element={<Models />} />
          <Route path="/explain" element={<Explain />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="/assistant" element={<Assistant />} />
        </Routes>
      </main>
    </div>
  );
}
