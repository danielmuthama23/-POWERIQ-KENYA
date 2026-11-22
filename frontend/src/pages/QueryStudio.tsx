import { useEffect, useState } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, useApi } from "../api";
import { Card, CHART, Empty, ErrorBox, fmt, Pill, Provenance, Table } from "../components/ui";

const PALETTE = ["#f2b84b", "#2ec4b6", "#3a86ff", "#e4572e", "#b388eb", "#8d99ae", "#7bd389", "#ff7eb6"];
const EXAMPLES = ["Why could the reserve margin fall this week and what should we do?", "Compare curtailment by plant and non-technical loss by region", "What is the lowest reserve margin in the next 72 hours?", "Total installed capacity by source", "Which power stations have the highest failure risk?",
  "Total non-technical loss by region", "Show the supply mix by hour", "Average demand by hour of day", "Which lines are most loaded?", "How much energy did Lake Turkana wind curtail?"];
const STORE = "poweriq_pinned_tiles";

function loadPins(): any[] { try { return JSON.parse(sessionStorage.getItem(STORE) ?? "[]"); } catch { return []; } }
function savePins(p: any[]) { try { sessionStorage.setItem(STORE, JSON.stringify(p)); } catch { /* storage unavailable */ } }

function toCsv(r: any): string {
  const q = (v: any) => (v === null || v === undefined ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  return [r.columns.map(q).join(","), ...r.rows.map((row: any[]) => row.map(q).join(","))].join("\n");
}

function Download({ r }: { r: any }) {
  const href = `data:text/csv;charset=utf-8,${encodeURIComponent(toCsv(r))}`;
  return <a className="rounded-md border border-line px-2 py-1 text-xs hover:bg-panel2" href={href} download={`${(r.title || "result").replace(/[^a-z0-9]+/gi, "_").slice(0, 40)}.csv`}>CSV</a>;
}

export function ResultChart({ r }: { r: any }) {
  const c = r.chart;
  if (!c || c.type === "table" || !c.data?.length) return null;
  const common = <><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey={c.x} stroke={CHART.axis} fontSize={10} minTickGap={24} /><YAxis stroke={CHART.axis} fontSize={10} /><Tooltip contentStyle={CHART.tooltip} /><Legend /></>;
  return (
    <div className="h-64" aria-label={`${c.type} chart: ${r.title}`}>
      <ResponsiveContainer>
        {c.type === "pie" ? (
          <PieChart><Tooltip contentStyle={CHART.tooltip} /><Legend />
            <Pie data={c.data} dataKey={c.series[0]} nameKey={c.x} outerRadius={90} label>{c.data.map((_: any, i: number) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}</Pie></PieChart>
        ) : c.type === "line" ? (
          <LineChart data={c.data}>{common}{c.series.map((s: string, i: number) => <Line key={s} dataKey={s} stroke={PALETTE[i % PALETTE.length]} dot={false} />)}</LineChart>
        ) : c.type === "area" ? (
          <AreaChart data={c.data}>{common}{c.series.map((s: string, i: number) => <Area key={s} dataKey={s} stackId="1" stroke={PALETTE[i % PALETTE.length]} fill={PALETTE[i % PALETTE.length] + "55"} />)}</AreaChart>
        ) : (
          <BarChart data={c.data}>{common}{c.series.map((s: string, i: number) => <Bar key={s} dataKey={s} fill={PALETTE[i % PALETTE.length]} />)}</BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

export function ResultCard({ r, onPin, onUnpin }: { r: any; onPin?: (r: any) => void; onUnpin?: () => void }) {
  const [showSpec, setShowSpec] = useState(false);
  const cols = r.columns.map((c: string, i: number) => ({ key: String(i), label: c, align: typeof r.rows[0]?.[i] === "number" ? ("right" as const) : undefined,
    render: (row: any) => { const v = row.__v[i]; return typeof v === "number" ? fmt(v, 3) : String(v ?? "n/a"); } }));
  const rows = r.rows.slice(0, 50).map((v: any[]) => ({ __v: v }));
  return (
    <Card title={r.title} subtitle={`${r.dataset} · ${r.row_count} rows${r.truncated ? " (truncated)" : ""}`}
      right={<div className="flex gap-2"><Download r={r} />{onPin && <button onClick={() => onPin(r)} className="rounded-md border border-line px-2 py-1 text-xs hover:bg-panel2">Pin</button>}
        {onUnpin && <button onClick={onUnpin} className="rounded-md border border-line px-2 py-1 text-xs hover:bg-panel2">Unpin</button>}</div>}>
      <ResultChart r={r} />
      <details className="mt-2"><summary className="cursor-pointer text-xs text-mute">Data table</summary><div className="mt-2 max-h-72 overflow-y-auto"><Table cols={cols} rows={rows} /></div></details>
      <button onClick={() => setShowSpec(s => !s)} className="mt-2 text-xs text-mute underline">{showSpec ? "Hide" : "Show"} query</button>
      {showSpec && <pre className="mt-1 overflow-x-auto rounded-lg bg-panel2 p-2 font-mono text-xs text-slate-300">{JSON.stringify(r.spec, null, 2)}</pre>}
    </Card>
  );
}

export default function QueryStudio() {
  const [q, setQ] = useState("");
  const [res, setRes] = useState<any>(null); const [busy, setBusy] = useState(false); const [err, setErr] = useState<Error | null>(null);
  const [hist, setHist] = useState<string[]>([]);
  const [pins, setPins] = useState<any[]>(loadPins);
  const [goal, setGoal] = useState(""); const [dash, setDash] = useState<any>(null); const [dbusy, setDbusy] = useState(false);
  const cat = useApi<any>("/query/catalog");
  const st = useApi<any>("/query/status");
  const [probe, setProbe] = useState<any>(null); const [probing, setProbing] = useState(false);
  const testKey = async () => {
    setProbing(true); setProbe(null);
    try { setProbe(await api("/query/test-openai", { method: "POST" })); } catch (e) { setProbe({ ok: false, error: (e as Error).message }); } finally { setProbing(false); }
  };
  useEffect(() => savePins(pins), [pins]);
  const ask = async (text: string) => {
    if (text.trim().length < 3) return;
    setBusy(true); setErr(null); setQ(text);
    try { const r = await api("/query/ask", { method: "POST", body: { question: text, history: hist.slice(-4).map(h => ({ question: h })) } }); setRes(r); setHist(h => [...h, text]); }
    catch (e) { setErr(e as Error); } finally { setBusy(false); }
  };
  const build = async () => {
    setDbusy(true); setErr(null);
    try { setDash(await api("/query/dashboard", { method: "POST", body: { goal: goal.trim() || null } })); } catch (e) { setErr(e as Error); } finally { setDbusy(false); }
  };
  const pin = (r: any) => setPins(p => (p.some(x => x.title === r.title) ? p : [...p, r]));
  return (
    <div className="space-y-4">
      <Card title="Ask anything about the system" subtitle="Plans a read-only query over the catalog, runs it, and returns a table, chart and explanation. With OPENAI_API_KEY set, OpenAI plans the queries and writes the narrative; otherwise a local keyword planner is used.">
        {st.data && <div className="mb-3 flex flex-wrap items-center gap-2 rounded-lg border border-line bg-panel2 px-3 py-2 text-xs">
          <Pill tone={st.data.openai_configured ? "good" : "warn"}>{st.data.openai_configured ? `OpenAI key configured (${st.data.model})` : "OpenAI key not configured"}</Pill>
          <span className="text-mute">{st.data.openai_configured ? "Questions are planned and explained by OpenAI; every query is still validated and run locally." : st.data.how_to_enable + " Until then a local planner answers."}</span>
          <button onClick={testKey} disabled={probing} className="ml-auto rounded-md border border-line px-2 py-1 hover:bg-panel">{probing ? "Testing…" : "Test connection (analyst)"}</button>
          {probe && <span className={probe.ok ? "text-good" : "text-danger"}>{probe.ok ? "OpenAI replied: " + probe.reply : "Not connected: " + (probe.error ?? "")}</span>}
        </div>}
        <form onSubmit={e => { e.preventDefault(); ask(q); }} className="flex gap-2">
          <input aria-label="Question" value={q} onChange={e => setQ(e.target.value)} maxLength={500} placeholder="Ask anything about Kenya's power system, e.g. why could reserve fall and what should we do?" className="min-w-0 flex-1 rounded-lg border border-line bg-panel2 px-3 py-2 text-sm" />
          <button disabled={busy || q.trim().length < 3} className="rounded-lg bg-amber px-4 py-2 text-sm font-semibold text-ink disabled:opacity-50">{busy ? "Querying…" : "Ask"}</button>
        </form>
        <div className="mt-2 flex flex-wrap gap-2">{EXAMPLES.map(x => <button key={x} onClick={() => ask(x)} className="rounded-full border border-line px-3 py-1 text-xs text-slate-300 hover:bg-panel2">{x}</button>)}</div>
      </Card>
      {err && <ErrorBox error={err} />}
      {res && <>
        <Card title="Answer" right={<div className="flex items-center gap-2"><Pill tone={res.mode === "openai" ? "good" : "mute"}>{res.mode}</Pill><Pill tone={res.kind === "out_of_scope" ? "warn" : "mute"}>{String(res.kind).replace("_", " ")}</Pill></div>}>
          <Provenance origin={res.data_origin} extra={res.llm_status} />
          <pre className="mt-2 whitespace-pre-wrap font-sans text-sm text-slate-200">{res.narrative}</pre>
          {res.analysis?.some((a: any) => a.notes.length) && <div className="mt-3 rounded-lg border border-line bg-panel2 p-3"><div className="mb-1 text-xs font-semibold text-slate-200">Computed analysis</div>
            <ul className="list-disc space-y-1 pl-5 text-xs text-slate-300">{res.analysis.flatMap((a: any) => a.notes.map((n: string) => <li key={a.title + n}>{n}</li>))}</ul></div>}
          {res.findings?.length > 0 && <div className="mt-3 space-y-2">{res.findings.map((f: any) => <div key={f.id} className="rounded-lg border border-line bg-panel2 p-3 text-xs"><span className="font-semibold text-slate-200">{f.area}</span> <Pill tone={f.severity === "high" ? "bad" : f.severity === "medium" ? "warn" : "mute"}>{f.severity}</Pill><p className="mt-1 text-slate-300">{f.finding}</p><p className="mt-1 text-mute">Suggested action: {f.action}</p></div>)}</div>}
          {res.errors.map((e: string) => <p key={e} className="mt-1 text-xs text-danger">{e}</p>)}
          {res.follow_ups.length > 0 && <div className="mt-3 flex flex-wrap gap-2">{res.follow_ups.map((f: string) => <button key={f} onClick={() => ask(f)} className="rounded-full border border-line px-3 py-1 text-xs text-mute hover:bg-panel2">{f}</button>)}</div>}
          <p className="mt-2 text-xs text-mute">{res.notice}</p>
        </Card>
        <div className="grid gap-4 xl:grid-cols-2">{res.results.map((r: any, i: number) => <ResultCard key={i} r={r} onPin={pin} />)}</div>
      </>}
      <Card title="Dashboard builder" subtitle="Power BI-style tiles from the same catalog. Describe a goal, or leave blank for the default national dashboard."
        right={<div className="flex gap-2"><input aria-label="Dashboard goal" value={goal} onChange={e => setGoal(e.target.value)} placeholder="e.g. reserve and curtailment risks" className="w-64 rounded-md border border-line bg-panel2 px-2 py-1 text-xs" />
          <button onClick={build} disabled={dbusy} className="rounded-md bg-amber px-3 py-1 text-xs font-semibold text-ink disabled:opacity-50">{dbusy ? "Building…" : "Build dashboard"}</button></div>}>
        {!dash && <Empty>Build a dashboard to see KPI-style charts. Pin any tile to keep it below for this browser tab.</Empty>}
        {dash && <><Provenance origin={dash.data_origin} extra={`${dash.mode} · ${dash.llm_status}`} /><p className="mb-3 text-xs text-mute">{dash.notice}</p>
          <div className="grid gap-4 xl:grid-cols-2">{dash.tiles.map((r: any, i: number) => <ResultCard key={i} r={r} onPin={pin} />)}</div></>}
      </Card>
      <Card title={`Pinned tiles (${pins.length})`} subtitle="Kept in this browser tab only (session storage)">
        {!pins.length ? <Empty>Nothing pinned yet.</Empty> : <div className="grid gap-4 xl:grid-cols-2">{pins.map((r, i) => <ResultCard key={r.title + i} r={r} onUnpin={() => setPins(p => p.filter((_, j) => j !== i))} />)}</div>}
      </Card>
      {cat.data && <Card title="Data catalog" subtitle="Everything the planner can query">
        <Table cols={[{ key: "name", label: "Dataset" }, { key: "rows", label: "Rows", align: "right" }, { key: "description", label: "Description" }, { key: "cols", label: "Columns", render: (r: any) => r.columns.map((c: any) => c.name).join(", ") }]} rows={cat.data.datasets} /></Card>}
    </div>
  );
}
