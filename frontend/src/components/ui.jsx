// Small shared UI pieces (Tailwind).
import { useEffect, useState } from "react";
import { api } from "../api.js";

export const Card = ({ title, actions, children, className = "" }) => (
  <section className={`bg-white rounded-xl border border-slate-200 p-5 ${className}`}>
    {(title || actions) && (
      <div className="flex items-center justify-between mb-4">
        {title && <h2 className="font-semibold text-slate-800">{title}</h2>}
        {actions}
      </div>
    )}
    {children}
  </section>
);

export const Stat = ({ label, value, hint }) => (
  <div className="bg-white rounded-xl border border-slate-200 p-4">
    <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
    <div className="text-2xl font-semibold mt-1">{value ?? "–"}</div>
    {hint && <div className="text-xs text-slate-500 mt-1">{hint}</div>}
  </div>
);

export const Button = ({ children, variant = "primary", className = "", ...p }) => {
  const styles = {
    primary: "bg-indigo-600 text-white hover:bg-indigo-700 disabled:bg-indigo-300",
    secondary: "bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 disabled:text-slate-400",
    danger: "bg-rose-600 text-white hover:bg-rose-700",
  };
  return (
    <button className={`px-4 py-2 rounded-lg text-sm font-medium transition ${styles[variant]} ${className}`} {...p}>
      {children}
    </button>
  );
};

export const Badge = ({ children, color = "slate" }) => {
  const c = {
    slate: "bg-slate-100 text-slate-700", green: "bg-emerald-100 text-emerald-700",
    amber: "bg-amber-100 text-amber-800", red: "bg-rose-100 text-rose-700", indigo: "bg-indigo-100 text-indigo-700",
  }[color];
  return <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${c}`}>{children}</span>;
};

export const ErrorBox = ({ error }) =>
  error ? <div className="bg-rose-50 border border-rose-200 text-rose-700 text-sm rounded-lg p-3 my-3">{String(error)}</div> : null;

export const Spinner = ({ text = "Loading…" }) => (
  <div className="flex items-center gap-2 text-slate-500 text-sm py-4">
    <div className="h-4 w-4 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
    {text}
  </div>
);

export const Select = ({ label, value, onChange, options, placeholder }) => (
  <label className="block text-sm">
    {label && <span className="text-slate-600">{label}</span>}
    <select className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2 bg-white"
            value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
      {placeholder && <option value="">{placeholder}</option>}
      {options.map((o) => (typeof o === "string"
        ? <option key={o} value={o}>{o}</option>
        : <option key={o.value} value={o.value}>{o.label}</option>))}
    </select>
  </label>
);

export const Table = ({ columns, rows, maxHeight = 420 }) => (
  <div className="overflow-auto border border-slate-200 rounded-lg" style={{ maxHeight }}>
    <table className="min-w-full text-sm">
      <thead className="bg-slate-50 sticky top-0">
        <tr>{columns.map((c) => <th key={c.key} className="text-left font-medium text-slate-600 px-3 py-2 whitespace-nowrap">{c.label ?? c.key}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className="border-t border-slate-100 hover:bg-slate-50">
            {columns.map((c) => (
              <td key={c.key} className="px-3 py-2 whitespace-nowrap">
                {c.render ? c.render(r[c.key], r) : fmt(r[c.key])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

export function fmt(v, digits = 4) {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(digits).replace(/\.?0+$/, "");
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

// Model picker shared by several pages; defaults to the champion.
export function useModels() {
  const [models, setModels] = useState([]);
  const [error, setError] = useState(null);
  const reload = () => api.models().then(setModels).catch(setError);
  useEffect(() => { reload(); }, []);
  return { models, error, reload };
}

export function ModelPicker({ models, value, onChange }) {
  useEffect(() => {
    if (!value && models.length) onChange((models.find((m) => m.status === "champion") || models[0]).model_id);
  }, [models, value, onChange]);
  return (
    <Select label="Model" value={value} onChange={onChange}
      options={models.map((m) => ({ value: m.model_id,
        label: `${m.best_model} · ${m.dataset} · ${m.model_id}${m.status === "champion" ? "  ★ champion" : ""}` }))} />
  );
}

export const FileInput = ({ onFile, accept = ".csv,.xlsx" }) => (
  <input type="file" accept={accept} onChange={(e) => onFile(e.target.files?.[0] || null)}
    className="block text-sm file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100" />
);

export const PageTitle = ({ title, subtitle }) => (
  <div className="mb-6">
    <h1 className="text-2xl font-semibold">{title}</h1>
    {subtitle && <p className="text-slate-500 text-sm mt-1">{subtitle}</p>}
  </div>
);
