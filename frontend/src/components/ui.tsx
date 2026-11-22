import { ReactNode } from "react";
import { ApiError } from "../api";

export const fmt = (v: number | null | undefined, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));
export const hhmm = (iso: string) => iso.slice(11, 16);
export const dayhh = (iso: string) => `${iso.slice(5, 10)} ${iso.slice(11, 13)}h`;
export const SOURCE_COLORS: Record<string, string> = { geothermal: "#e4572e", hydro: "#3a86ff", wind: "#2ec4b6", solar: "#f2b84b", thermal: "#8d99ae", import: "#b388eb" };
export const CHART = { grid: "#1c2a42", axis: "#8696b3", tooltip: { background: "#0e1626", border: "1px solid #1c2a42", borderRadius: 8, fontSize: 12 } };

export function Card({ title, subtitle, right, children, className = "" }: { title?: string; subtitle?: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-line bg-panel p-4 ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>{title && <h2 className="text-sm font-semibold text-slate-100">{title}</h2>}{subtitle && <p className="text-xs text-mute">{subtitle}</p>}</div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, unit, tone, hint }: { label: string; value: string; unit?: string; tone?: "good" | "warn" | "bad"; hint?: string }) {
  const c = tone === "bad" ? "text-danger" : tone === "warn" ? "text-amber" : tone === "good" ? "text-good" : "text-slate-50";
  return (
    <div className="rounded-xl border border-line bg-panel p-4" title={hint}>
      <div className="text-xs uppercase tracking-wide text-mute">{label}</div>
      <div className={`mt-1 font-mono text-2xl font-semibold ${c}`}>{value}<span className="ml-1 text-sm font-normal text-mute">{unit}</span></div>
      {hint && <div className="mt-1 text-xs text-mute">{hint}</div>}
    </div>
  );
}

const ORIGIN: Record<string, string> = {
  synthetic: "bg-slate-700/60 text-slate-200", forecast: "bg-teal/20 text-teal", "synthetic+model": "bg-amber/20 text-amber",
  "forecast+synthetic": "bg-teal/20 text-teal", historical: "bg-blue-500/20 text-blue-300", live: "bg-good/20 text-good",
};
export function Provenance({ origin, model, extra }: { origin?: string; model?: string; extra?: string }) {
  if (!origin) return null;
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs" aria-label="data provenance">
      <span className={`rounded-full px-2 py-0.5 font-medium ${ORIGIN[origin] ?? "bg-slate-700 text-slate-200"}`}>data: {origin}</span>
      {model && <span className="rounded-full bg-panel2 px-2 py-0.5 text-mute">model {model}</span>}
      {extra && <span className="text-mute">{extra}</span>}
    </div>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return <div role="status" className="animate-pulse rounded-xl border border-line bg-panel p-6 text-sm text-mute">{label}…</div>;
}

export function ErrorBox({ error, onRetry }: { error: ApiError | Error; onRetry?: () => void }) {
  const e = error as ApiError;
  return (
    <div role="alert" className="rounded-xl border border-danger/50 bg-danger/10 p-4 text-sm">
      <div className="font-semibold text-danger">{e.status === 403 ? "Not permitted" : e.status === 401 ? "Not signed in" : "Something went wrong"}</div>
      <div className="mt-1 text-slate-300">{e.message}{e.requestId && <span className="ml-2 font-mono text-xs text-mute">ref {e.requestId}</span>}</div>
      {onRetry && <button className="mt-3 rounded-md border border-line px-3 py-1 text-xs hover:bg-panel2" onClick={onRetry}>Retry</button>}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded-xl border border-dashed border-line p-6 text-center text-sm text-mute">{children}</div>;
}

export function Async<T>({ state, children }: { state: { data: T | null; error: ApiError | null; loading: boolean; reload: () => void }; children: (d: T) => ReactNode }) {
  if (state.error) return <ErrorBox error={state.error} onRetry={state.reload} />;
  if (state.loading && !state.data) return <Loading />;
  if (!state.data) return <Empty>No data.</Empty>;
  return <>{children(state.data)}</>;
}

export function Table({ cols, rows, empty = "No rows." }: { cols: { key: string; label: string; render?: (r: any) => ReactNode; align?: "right" }[]; rows: any[]; empty?: string }) {
  if (!rows.length) return <Empty>{empty}</Empty>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead><tr className="border-b border-line text-mute">{cols.map(c => <th key={c.key} scope="col" className={`px-2 py-2 font-medium ${c.align === "right" ? "text-right" : ""}`}>{c.label}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i} className="border-b border-line/50 hover:bg-panel2">{cols.map(c => <td key={c.key} className={`px-2 py-1.5 ${c.align === "right" ? "text-right font-mono" : ""}`}>{c.render ? c.render(r) : r[c.key]}</td>)}</tr>
        ))}</tbody>
      </table>
    </div>
  );
}

export function Pill({ children, tone = "mute" }: { children: ReactNode; tone?: "mute" | "good" | "warn" | "bad" }) {
  const c = { mute: "bg-slate-700/60 text-slate-300", good: "bg-good/20 text-good", warn: "bg-amber/20 text-amber", bad: "bg-danger/20 text-danger" }[tone];
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${c}`}>{children}</span>;
}

export function Limitations({ items }: { items?: string[] }) {
  if (!items?.length) return null;
  return <details className="mt-3 text-xs text-mute"><summary className="cursor-pointer">Limitations</summary><ul className="mt-1 list-disc space-y-1 pl-5">{items.map((l, i) => <li key={i}>{l}</li>)}</ul></details>;
}

export function Select<T extends string | number>({ label, value, onChange, options }: { label: string; value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <label className="flex items-center gap-2 text-xs text-mute">{label}
      <select className="rounded-md border border-line bg-panel2 px-2 py-1 text-slate-200" value={value} onChange={e => onChange((typeof value === "number" ? Number(e.target.value) : e.target.value) as T)}>
        {options.map(o => <option key={String(o.value)} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}
