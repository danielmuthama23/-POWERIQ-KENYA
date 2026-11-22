import { FormEvent, useRef, useState } from "react";
import { api, useApi } from "../api";
import { Async, Card, ErrorBox, Limitations, Pill, Provenance, Table } from "../components/ui";

const EXAMPLES = [
  "What is the predicted electricity demand in Kenya over the next 24 hours?", "Which generation assets have the highest maintenance risk?",
  "What is causing the simulated generation-demand gap?", "Which renewable resources are being curtailed?", "What happens if geothermal output decreases by 15%?",
  "Which battery schedule minimizes peak demand?", "Which distribution regions show unusual energy losses?", "Generate a maintenance priority report", "Compare alternative energy storage scenarios", "Give me an overall status of the grid",
];

type Turn = { q: string; a?: any; err?: Error };

function Section({ title, items, tone }: { title: string; items: string[]; tone: string }) {
  if (!items?.length) return null;
  return <div className="mt-2"><div className={`text-xs font-semibold uppercase tracking-wide ${tone}`}>{title}</div><ul className="mt-1 list-disc space-y-1 pl-5 text-xs text-slate-300">{items.map((x, i) => <li key={i}>{x}</li>)}</ul></div>;
}

export function Answer({ a }: { a: any }) {
  return (
    <div className="rounded-xl border border-line bg-panel2 p-4 text-sm">
      <div className="whitespace-pre-wrap text-slate-100">{a.mode === "openai" ? a.answer : a.answer.split("\n\n")[0]}</div>
      {a.mode !== "openai" && (<>
        <Section title="Measured facts" items={a.measured_facts} tone="text-good" /><Section title="Model predictions" items={a.model_predictions} tone="text-teal" />
        <Section title="Assumptions" items={a.assumptions} tone="text-amber" /><Section title="Recommendations (advisory, need human approval)" items={a.recommendations} tone="text-blue-300" />
      </>)}
      {a.citations?.length > 0 && <div className="mt-3 text-xs text-mute">Sources: {a.citations.map((c: any) => `${c.citation} ${c.title}`).join(" · ")}</div>}
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-mute">
        <Pill tone={a.mode === "openai" ? "good" : "mute"}>{a.mode === "openai" ? "OpenAI narrative" : "local template narrative"}</Pill>
        <span>{a.timestamp}</span><span>model {a.model_version}</span><span>uncertainty: {a.uncertainty}</span>
        {a.recommendation_records?.length > 0 && <span>filed for approval: {a.recommendation_records.map((r: any) => r.id).join(", ")}</span>}
      </div>
      {a.llm_status && a.mode !== "openai" && <div className="mt-1 text-xs text-mute">LLM: {a.llm_status}</div>}
      <Limitations items={a.limitations} />
    </div>
  );
}

export function Copilot() {
  const [turns, setTurns] = useState<Turn[]>([]); const [input, setInput] = useState(""); const [busy, setBusy] = useState(false);
  const cid = useRef<string | undefined>(undefined);
  const ask = async (q: string) => {
    if (!q.trim() || busy) return;
    setBusy(true); setInput("");
    const t: Turn = { q }; setTurns(x => [...x, t]);
    try { const a = await api("/copilot/chat", { method: "POST", body: { message: q, conversation_id: cid.current } }); cid.current = a.conversation_id; setTurns(x => x.map(y => y === t ? { ...y, a } : y)); }
    catch (e) { setTurns(x => x.map(y => y === t ? { ...y, err: e as Error } : y)); } finally { setBusy(false); }
  };
  const submit = (e: FormEvent) => { e.preventDefault(); ask(input); };
  return (
    <Card title="AI Copilot" subtitle="Grounded in agent outputs and cited knowledge. Never issues commands to infrastructure.">
      <div className="mb-3 flex flex-wrap gap-2">{EXAMPLES.map(x => <button key={x} onClick={() => ask(x)} disabled={busy} className="rounded-full border border-line px-3 py-1 text-xs text-slate-300 hover:bg-panel2 disabled:opacity-50">{x}</button>)}</div>
      <div className="max-h-[560px] space-y-3 overflow-y-auto pr-1" aria-live="polite">
        {turns.length === 0 && <div className="rounded-xl border border-dashed border-line p-6 text-center text-sm text-mute">Ask a question or pick an example above. Every answer separates measured facts, model predictions, assumptions and recommendations.</div>}
        {turns.map((t, i) => (<div key={i}><div className="mb-1 text-right"><span className="inline-block rounded-xl bg-amber/15 px-3 py-1.5 text-sm text-amber">{t.q}</span></div>{t.err ? <ErrorBox error={t.err} /> : t.a ? <Answer a={t.a} /> : <div role="status" className="text-xs text-mute">Agents working…</div>}</div>))}
      </div>
      <form onSubmit={submit} className="mt-3 flex gap-2"><input aria-label="Question" value={input} onChange={e => setInput(e.target.value)} placeholder="Ask about demand, risk, losses, curtailment, storage or a what-if…" className="flex-1 rounded-lg border border-line bg-panel2 px-3 py-2 text-sm" /><button disabled={busy || input.trim().length < 3} className="rounded-lg bg-amber px-4 py-2 text-sm font-semibold text-ink disabled:opacity-50">Ask</button></form>
    </Card>
  );
}

export function AgentsPanel() {
  const ag = useApi<any>("/agents");
  const [res, setRes] = useState<any>(null); const [busy, setBusy] = useState(false); const [err, setErr] = useState<Error | null>(null);
  const run = async () => { setBusy(true); setErr(null); try { setRes(await api("/agents/chief/assess", { method: "POST", body: { question: "overall assessment" } })); ag.reload(); } catch (e) { setErr(e as Error); } finally { setBusy(false); } };
  const tone = (l: string) => l === "high" ? "bad" : l === "elevated" ? "warn" : l === "moderate" ? "mute" : "good";
  return (
    <div className="space-y-4">
      <Card title="Chief Energy Intelligence Agent" subtitle="Runs all seven specialists, resolves conflicts, lists missing information" right={<button onClick={run} disabled={busy} className="rounded-md bg-amber px-3 py-1 text-xs font-semibold text-ink disabled:opacity-50">{busy ? "Assessing…" : "Run assessment"}</button>}>
        {err && <ErrorBox error={err} />}
        {!res && !err && <div className="text-sm text-mute">Run an assessment to see the consolidated picture.</div>}
        {res && <div className="space-y-3 text-sm">
          <div>Overall risk level: <Pill tone={tone(res.overall_risk_level) as any}>{res.overall_risk_level}</Pill> <span className="text-xs text-mute">{res.agents_run.length} agents · {res.duration_ms} ms · {res.timestamp}</span></div>
          {res.findings.map((f: any, i: number) => <p key={i} className="text-xs text-slate-300">{f.statement}</p>)}
          {res.conflicts_resolved.length > 0 && <div><div className="text-xs font-semibold text-amber">Conflicts resolved</div><ul className="list-disc pl-5 text-xs">{res.conflicts_resolved.map((c: any, i: number) => <li key={i}>{c.resolution}</li>)}</ul></div>}
          {res.coupled_scenario && <p className="text-xs text-slate-300">Coupled check: loss of {res.coupled_scenario.unit} changes unserved energy by {res.coupled_scenario.scenario.unserved_mwh_delta.toFixed(1)} MWh; min reserve {res.coupled_scenario.scenario.min_reserve_pct.toFixed(1)}%.</p>}
          <div><div className="text-xs font-semibold text-blue-300">Recommendations (pending human approval)</div><ul className="list-disc pl-5 text-xs">{res.recommendations.map((r: string, i: number) => <li key={i}>{r}</li>)}</ul></div>
          <div><div className="text-xs font-semibold text-mute">Missing information</div><ul className="list-disc pl-5 text-xs text-mute">{res.missing_information.map((m: string, i: number) => <li key={i}>{m}</li>)}</ul></div>
        </div>}
      </Card>
      <Card title="Agent roster" subtitle={ag.data?.policy}>
        <Async state={ag}>{d => <Table cols={[{ key: "name", label: "Agent" }, { key: "responsibility", label: "Responsibility" }, { key: "tools", label: "Tools (read-only)", render: r => <span className="text-mute">{r.tools.join(", ")}</span> }, { key: "runs", label: "Runs", align: "right" }, { key: "errors", label: "Errors", align: "right" }, { key: "last_duration_ms", label: "Last ms", align: "right" }]} rows={d.agents} />}</Async>
      </Card>
    </div>
  );
}

export function AICommand() {
  return <div className="grid gap-4 xl:grid-cols-2"><Copilot /><AgentsPanel /></div>;
}

export function ReportsAudit() {
  const recs = useApi<any>("/recommendations"); const ev = useApi<any>("/audit/events?limit=60"); const integ = useApi<any>("/integrations/status");
  const [verify, setVerify] = useState<any>(null); const [err, setErr] = useState<Error | null>(null);
  const decide = async (id: string, decision: "approved" | "rejected") => { try { await api(`/recommendations/${id}/decision`, { method: "POST", body: { decision } }); recs.reload(); ev.reload(); setErr(null); } catch (e) { setErr(e as Error); } };
  return (
    <div className="space-y-4">
      {err && <ErrorBox error={err} />}
      <Card title="Recommendations" subtitle="Approving records a human decision only. POWERIQ cannot execute operational actions.">
        <Async state={recs}>{d => <Table cols={[{ key: "id", label: "ID" }, { key: "source", label: "Source" }, { key: "text", label: "Recommendation" }, { key: "status", label: "Status", render: r => <Pill tone={r.status === "approved" ? "good" : r.status === "rejected" ? "bad" : "warn"}>{r.status}</Pill> },
          { key: "act", label: "", render: r => r.status === "pending" ? <span className="flex gap-1"><button onClick={() => decide(r.id, "approved")} className="rounded border border-good/50 px-2 py-0.5 text-good hover:bg-good/10">Approve</button><button onClick={() => decide(r.id, "rejected")} className="rounded border border-danger/50 px-2 py-0.5 text-danger hover:bg-danger/10">Reject</button></span> : <span className="text-mute">{r.decided_by}</span> }]} rows={d.recommendations} empty="No recommendations yet. Ask the copilot or run an assessment." />}</Async>
        <p className="mt-2 text-xs text-mute">Approve/Reject requires the approver role; other roles will see a permission message.</p>
      </Card>
      <Card title="Audit trail" subtitle="Hash-chained, tamper-evident (analyst role)" right={<button onClick={async () => { try { setVerify(await api("/audit/verify")); } catch (e) { setErr(e as Error); } }} className="rounded-md border border-line px-3 py-1 text-xs hover:bg-panel2">Verify chain</button>}>
        {verify && <div className="mb-2 text-xs">{verify.valid ? <Pill tone="good">chain valid · {verify.events_checked} events</Pill> : <Pill tone="bad">chain broken at #{verify.broken_at_seq}</Pill>} <span className="text-mute">{verify.note}</span></div>}
        <Async state={ev}>{d => <div className="max-h-96 overflow-y-auto"><Table cols={[{ key: "seq", label: "#", align: "right" }, { key: "ts", label: "Time" }, { key: "event_type", label: "Event" }, { key: "actor", label: "Actor" }, { key: "role", label: "Role" }, { key: "payload", label: "Detail", render: r => <span className="font-mono text-mute">{JSON.stringify(r.payload).slice(0, 110)}</span> }]} rows={d.events} /></div>}</Async>
      </Card>
      <Card title="Integration status" subtitle="Nothing is claimed as integrated until a connection has been tested">
        <Async state={integ}>{d => <Table cols={[{ key: "name", label: "Integration" }, { key: "status", label: "Status", render: r => <Pill tone={r.status.includes("not") || r.status === "never_connected" ? "warn" : "good"}>{r.status}</Pill> }, { key: "note", label: "Note", render: r => <span className="text-mute">{r.note}</span> }]} rows={d.integrations} />}</Async>
      </Card>
    </div>
  );
}
