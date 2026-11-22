import { FormEvent, useEffect, useState } from "react";
import { api, getKey, setKey } from "./api";
import { Card, ErrorBox } from "./components/ui";
import { Climate, Generation, Grid, Losses, Maintenance, Renewable } from "./pages/Analytics";
import { AICommand, ReportsAudit } from "./pages/Command";
import FaultHub from "./pages/FaultHub";
import DigitalTwin from "./pages/DigitalTwin";
import { CurtailmentStorage, DemandStructure, Fleet, KenyaProblems, ReserveRisk, StressTests, TheftPlanner } from "./pages/Insights";
import QueryStudio from "./pages/QueryStudio";
import Overview from "./pages/Overview";
import Storage from "./pages/Storage";

export const PAGES: { id: string; label: string; group: string; el: () => JSX.Element }[] = [
  { id: "overview", label: "National Overview", group: "Operations", el: Overview },
  { id: "generation", label: "Generation Intelligence", group: "Operations", el: Generation },
  { id: "maintenance", label: "Predictive Maintenance", group: "Operations", el: Maintenance },
  { id: "grid", label: "Grid Intelligence", group: "Operations", el: Grid },
  { id: "renewable", label: "Renewable Intelligence", group: "Operations", el: Renewable },
  { id: "storage", label: "Storage Optimization", group: "Operations", el: Storage },
  { id: "losses", label: "Energy Loss Detection", group: "Operations", el: Losses },
  { id: "climate", label: "Climate Intelligence", group: "Operations", el: Climate },
  { id: "twin", label: "National Digital Twin", group: "Operations", el: DigitalTwin },
  { id: "faults", label: "Fault Hub (Live Map)", group: "Operations", el: FaultHub },

  { id: "kenya", label: "Kenya Problems & AI Analysis", group: "Analysis", el: KenyaProblems },
  { id: "reserve", label: "Reserve Risk (Monte Carlo)", group: "Analysis", el: ReserveRisk },
  { id: "curtailment", label: "Curtailment to Storage", group: "Analysis", el: CurtailmentStorage },
  { id: "stress", label: "Drought & Wind Stress Tests", group: "Analysis", el: StressTests },
  { id: "fleet", label: "Power Station Fleet", group: "Analysis", el: Fleet },
  { id: "theft", label: "Theft Inspection Planner", group: "Analysis", el: TheftPlanner },
  { id: "demandstruct", label: "Demand Structure (STL)", group: "Analysis", el: DemandStructure },
  { id: "reports", label: "Reports and Audit", group: "Governance", el: ReportsAudit },

  { id: "command", label: "AI Command Centre & Copilot", group: "AI and Query", el: AICommand },
  { id: "query", label: "Query Studio (Ask Anything)", group: "AI and Query", el: QueryStudio },
];

export function Login({ onDone }: { onDone: () => void }) {
  const [key, setK] = useState(""); const [err, setErr] = useState<Error | null>(null); const [busy, setBusy] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault(); setBusy(true); setErr(null); setKey(key.trim());
    try { await api("/overview"); onDone(); } catch (x) { setKey(""); setErr(x as Error); } finally { setBusy(false); }
  };
  return (
    <div className="flex min-h-full items-center justify-center p-6">
      <form onSubmit={submit} className="w-full max-w-md space-y-4">
        <div><div className="text-xs font-semibold uppercase tracking-widest text-amber">POWERIQ Kenya</div><h1 className="mt-1 text-2xl font-semibold text-slate-50">Predict the problem before it becomes a blackout.</h1></div>
        <Card>
          <label className="block text-xs text-mute" htmlFor="key">API key</label>
          <input id="key" type="password" autoComplete="off" value={key} onChange={e => setK(e.target.value)} className="mt-1 w-full rounded-lg border border-line bg-panel2 px-3 py-2 text-sm" />
          <button disabled={busy || !key.trim()} className="mt-3 w-full rounded-lg bg-amber py-2 text-sm font-semibold text-ink disabled:opacity-50">{busy ? "Checking…" : "Enter command centre"}</button>
          <p className="mt-3 text-xs text-mute">Keys are issued by the operator (POWERIQ_API_KEYS) and kept only in this tab's session storage. Local demo mode accepts <code className="font-mono">demo-viewer-key</code>, <code className="font-mono">demo-analyst-key</code> and <code className="font-mono">demo-approver-key</code>.</p>
        </Card>
        {err && <ErrorBox error={err} />}
        <p className="text-xs text-mute">Decision-support prototype on synthetic data. Not connected to any grid control system.</p>
      </form>
    </div>
  );
}

export default function App() {
  const [authed, setAuthed] = useState(!!getKey());
  const [page, setPage] = useState(() => PAGES.find(p => p.id === location.hash.slice(1))?.id ?? "overview");
  const [nav, setNav] = useState(false);
  useEffect(() => { location.hash = page; }, [page]);
  if (!authed) return <Login onDone={() => setAuthed(true)} />;
  const Cur = PAGES.find(p => p.id === page)!.el;
  return (
    <div className="flex h-full">
      <aside className={`${nav ? "fixed inset-y-0 z-20 flex" : "hidden"} w-64 shrink-0 flex-col border-r border-line bg-panel p-4 lg:static lg:flex`} aria-label="Primary">
        <div className="mb-6"><div className="text-lg font-bold tracking-tight text-slate-50">POWER<span className="text-amber">IQ</span> <span className="text-xs font-normal text-mute">Kenya</span></div><div className="text-xs text-mute">Energy intelligence command centre</div></div>
        <nav className="flex-1 space-y-1 overflow-y-auto">
          {PAGES
            .sort((a, b) => a.group.localeCompare(b.group))
            .map(p => (
            <button key={p.id} onClick={() => { setPage(p.id); setNav(false); }} aria-current={p.id === page ? "page" : undefined}
              className={`block w-full rounded-lg px-3 py-2 text-left text-sm ${p.id === page ? "bg-amber/15 text-amber" : "text-slate-300 hover:bg-panel2"}`}>{p.label}</button>))}
        </nav>
        <button onClick={() => { setKey(""); setAuthed(false); }} className="mt-4 rounded-lg border border-line px-3 py-2 text-xs text-mute hover:bg-panel2">Sign out</button>
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto p-4 lg:p-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <button className="rounded-md border border-line px-2 py-1 text-sm lg:hidden" onClick={() => setNav(n => !n)} aria-label="Toggle navigation">Menu</button>
            <div>
              <div className="text-xs font-medium uppercase tracking-wider text-mute">Command centre</div>
              <h1 className="mt-0.5 text-xl font-semibold text-slate-50">{PAGES.find(p => p.id === page)!.label}</h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <label className="hidden items-center gap-2 text-xs text-mute md:flex" htmlFor="page-navigation">Workspace
              <select id="page-navigation" value={page} onChange={e => setPage(e.target.value)} className="min-w-56 rounded-lg border border-line bg-panel2 px-3 py-2 text-xs text-slate-100 shadow-sm">
                <optgroup label="Operations">
                  {PAGES.filter(p => p.group === "Operations").map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
                </optgroup>
                <optgroup label="Analysis">
                  {PAGES.filter(p => p.group === "Analysis").map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
                </optgroup>
                <optgroup label="Governance">
                  {PAGES.filter(p => p.group === "Governance").map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
                </optgroup>
                <optgroup label="AI and Query">
                  {PAGES.filter(p => p.group === "AI and Query").map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
                </optgroup>
              </select>
            </label>
            <div className="rounded-full border border-amber/40 bg-amber/10 px-3 py-1 text-xs text-amber">Synthetic data · advisory only</div>
          </div>
        </div>
        <Cur />
      </main>
    </div>
  );
}
