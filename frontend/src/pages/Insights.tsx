import { useState } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, useApi } from "../api";
import { Async, Card, CHART, ErrorBox, fmt, Limitations, Pill, Provenance, SOURCE_COLORS, Stat, Table } from "../components/ui";

const SEV: Record<string, "bad" | "warn" | "mute"> = { high: "bad", medium: "warn", low: "mute" };
const ST: Record<string, "good" | "warn" | "mute"> = { implemented: "good", partial: "warn" };

export function KenyaProblems() {
  const sc = useApi<any>("/insights/scorecard");
  const [q, setQ] = useState("");
  const [an, setAn] = useState<any>(null); const [busy, setBusy] = useState(false); const [err, setErr] = useState<Error | null>(null);
  const [sent, setSent] = useState<Record<string, string>>({});
  const run = async () => {
    setBusy(true); setErr(null);
    try { setAn(await api(`/insights/analysis${q.trim() ? `?question=${encodeURIComponent(q.trim())}` : ""}`)); } catch (e) { setErr(e as Error); } finally { setBusy(false); }
  };
  const propose = async (id: string) => {
    try { const r = await api("/insights/propose", { method: "POST", body: { finding_id: id } }); setSent(s => ({ ...s, [id]: r.recommendation_id })); }
    catch (e) { setErr(e as Error); }
  };
  return (
    <div className="space-y-4">
      <Async state={sc}>{d => (<>
        <Provenance origin={d.data_origin} extra="briefing figures are context only; synthetic values are not calibrated to them" />
        <Card title="Kenya's power-sector problems vs this platform" subtitle={d.brief.source}>
          <Table cols={[{ key: "problem", label: "Problem" }, { key: "brief", label: "Reported in briefing" }, { key: "synthetic", label: "In this synthetic model" }, { key: "model", label: "Handled by" }]} rows={d.rows} />
          <p className="mt-2 text-xs text-mute">{d.note}</p>
        </Card>
        <Card title="Briefing modules → this build" subtitle="Status is what exists in the code today">
          <Table cols={[{ key: "module", label: "Module" }, { key: "status", label: "Status", render: r => <Pill tone={ST[r.status]}>{r.status}</Pill> }, { key: "where", label: "Where" }]} rows={d.modules} />
          <details className="mt-3 text-xs text-mute"><summary className="cursor-pointer">Not implemented (listed in the briefing's stack)</summary><ul className="mt-1 list-disc pl-5">{d.not_implemented.map((x: string) => <li key={x}>{x}</li>)}</ul></details>
        </Card>
      </>)}</Async>
      <Card title="AI analysis" subtitle="Rule-based findings from model outputs; narrative by the LLM if configured, otherwise a local template"
        right={<div className="flex gap-2"><input aria-label="Analysis question" value={q} onChange={e => setQ(e.target.value)} placeholder="Optional question" className="w-56 rounded-md border border-line bg-panel2 px-2 py-1 text-xs" />
          <button onClick={run} disabled={busy} className="rounded-md bg-amber px-3 py-1 text-xs font-semibold text-ink disabled:opacity-50">{busy ? "Analysing…" : "Run analysis"}</button></div>}>
        {err && <ErrorBox error={err} />}
        {!an && !err && <p className="text-sm text-mute">Run the analysis to rank findings across reserve risk, curtailment, drought exposure, losses and maintenance.</p>}
        {an && <div className="space-y-3">
          <Provenance origin={an.data_origin} extra={`narrative: ${an.mode} · ${an.llm_status}`} />
          {an.findings.map((f: any) => (
            <div key={f.id} className="rounded-lg border border-line bg-panel2 p-3 text-sm">
              <div className="flex items-center justify-between gap-2"><div className="font-semibold text-slate-100">{f.area} <Pill tone={SEV[f.severity]}>{f.severity}</Pill></div>
                {sent[f.id] ? <span className="font-mono text-xs text-good">queued {sent[f.id]}</span> : <button onClick={() => propose(f.id)} className="rounded-md border border-line px-2 py-1 text-xs hover:bg-panel">Send to approvals</button>}</div>
              <p className="mt-1 text-slate-300">{f.finding}</p><p className="mt-1 text-xs text-mute">Suggested action: {f.action}</p>
            </div>))}
          <pre className="whitespace-pre-wrap rounded-lg border border-line bg-panel2 p-3 font-sans text-xs text-slate-300">{an.narrative}</pre>
          <p className="text-xs text-mute">{an.notice} “Send to approvals” needs the analyst role.</p>
        </div>}
      </Card>
    </div>
  );
}

export function ReserveRisk() {
  const r = useApi<any>("/insights/reserve-risk");
  return <Async state={r}>{d => (
    <div className="space-y-4">
      <Provenance origin={d.data_origin} extra={d.method} />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="P(any shortfall) 72 h" value={fmt(d.p_any_shortfall_72h * 100, 1)} unit="%" tone={d.p_any_shortfall_72h > 0.25 ? "bad" : d.p_any_shortfall_72h > 0.05 ? "warn" : "good"} hint={`${fmt(d.n_sims)} simulations`} />
        <Stat label="Expected unserved" value={fmt(d.expected_unserved_mwh_72h, 1)} unit="MWh" />
        <Stat label="P95 unserved" value={fmt(d.p95_unserved_mwh_72h, 1)} unit="MWh" />
        <Stat label="Min expected margin" value={fmt(d.min_margin_pct, 1)} unit="%" hint={`briefing reports ${d.briefing_reserve_margin_pct}% for Kenya (not comparable)`} />
      </div>
      <Card title="Required vs expected supply (MW) and loss-of-load probability" subtitle="Supply is available capability after sampled outages, not dispatched output. Right axis = LOLP per hour">
        <div className="h-72"><ResponsiveContainer><ComposedChart data={d.hourly}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="h" stroke={CHART.axis} fontSize={10} /><YAxis yAxisId="l" stroke={CHART.axis} fontSize={10} /><YAxis yAxisId="r" orientation="right" domain={[0, 1]} stroke={CHART.axis} fontSize={10} /><Tooltip contentStyle={CHART.tooltip} /><Legend />
          <Line yAxisId="l" dataKey="required_mw" name="Required MW" stroke="#f2b84b" dot={false} /><Line yAxisId="l" dataKey="expected_supply_mw" name="Expected supply MW" stroke="#2ec4b6" dot={false} /><Line yAxisId="r" dataKey="lolp" name="LOLP" stroke="#e4572e" dot={false} /></ComposedChart></ResponsiveContainer></div>
      </Card>
      <Card title="Stations contributing most to shortfall risk" subtitle="Expected unserved energy removed if the station were fully available">
        <Table cols={[{ key: "plant", label: "Power station" }, { key: "source", label: "Source" }, { key: "outage_prob_72h", label: "72 h outage prob", align: "right" }, { key: "eens_reduction_mwh", label: "EENS cut MWh", align: "right" }]} rows={d.top_risk_contributors} />
        <Limitations items={d.assumptions} />
      </Card>
    </div>)}</Async>;
}

export function CurtailmentStorage() {
  const r = useApi<any>("/insights/curtailment-storage");
  return <Async state={r}>{d => (
    <div className="space-y-4">
      <Provenance origin={d.data_origin} extra={`last ${d.window_days} days`} />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Curtailed 30 d" value={fmt(d.curtailed_mwh_30d / 1000, 2)} unit="GWh" />
        {Object.entries(d.by_source_mwh).map(([k, v]) => <Stat key={k} label={`${k} curtailed`} value={fmt((v as number) / 1000, 2)} unit="GWh" />)}
        <Stat label="Briefing (annual)" value={fmt(d.briefing_annual_curtailment_gwh.total, 1)} unit="GWh" hint={`geothermal ${d.briefing_annual_curtailment_gwh.geothermal}, wind ${d.briefing_annual_curtailment_gwh.wind}`} />
      </div>
      <Card title="Curtailment by hour of day (MWh, 30 d)">
        <div className="h-56"><ResponsiveContainer><BarChart data={Object.entries(d.by_hour_of_day_mwh).map(([h, v]) => ({ h: `${h}h`, v }))}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="h" stroke={CHART.axis} fontSize={10} /><YAxis stroke={CHART.axis} fontSize={10} /><Tooltip contentStyle={CHART.tooltip} /><Bar dataKey="v" name="MWh" fill="#2ec4b6" /></BarChart></ResponsiveContainer></div>
      </Card>
      <Card title="What would a battery change over the next 72 h?" subtitle={d.method}>
        <Table cols={[{ key: "s", label: "Battery", render: r => `${r.power_mw} MW / ${r.energy_mwh} MWh` }, { key: "curtailment_recovered_mwh", label: "Curtailment recovered", align: "right" }, { key: "thermal_avoided_mwh", label: "Thermal avoided MWh", align: "right" }, { key: "unserved_avoided_mwh", label: "Unserved avoided", align: "right" }, { key: "corridor_limited_mwh", label: "Corridor-limited MWh", align: "right" }]} rows={d.battery_options} />
        <p className="mt-2 text-xs text-slate-300">{d.insight}</p><p className="mt-1 text-xs text-mute">{d.caveat}</p>
      </Card>
      <Card title="Most curtailed stations"><Table cols={[{ key: "plant", label: "Station" }, { key: "reason", label: "Reason" }, { key: "mwh", label: "MWh", align: "right", render: r => fmt(r.mwh, 0) }, { key: "hours", label: "Hours", align: "right" }]} rows={d.by_plant} /></Card>
    </div>)}</Async>;
}

export function StressTests() {
  const r = useApi<any>("/insights/climate-stress");
  return <Async state={r}>{d => (
    <div className="space-y-4">
      <Provenance origin={d.data_origin} extra={`${d.horizon_hours} h horizon`} />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Water factor now" value={fmt(d.water_factor.current, 2)} hint="1.0 = normal hydrology" />
        <Stat label="Trend / 30 d" value={fmt(d.water_factor.trend_per_30d, 3)} tone={d.water_factor.trend_per_30d < 0 ? "warn" : "good"} />
        <Stat label="60 d min" value={fmt(d.water_factor.min_60d, 2)} /><Stat label="60 d max" value={fmt(d.water_factor.max_60d, 2)} />
      </div>
      <Card title="Drought, wind-lull and cloud stress tests" subtitle="Forecast potential is scaled and the whole system is re-dispatched">
        <div className="h-64"><ResponsiveContainer><BarChart data={d.scenarios}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="scenario" stroke={CHART.axis} fontSize={9} interval={0} angle={-15} height={60} /><YAxis stroke={CHART.axis} fontSize={10} /><Tooltip contentStyle={CHART.tooltip} /><Legend />
          <Bar dataKey="thermal_mwh" name="Thermal MWh" fill={SOURCE_COLORS.thermal} /><Bar dataKey="unserved_mwh" name="Unserved MWh" fill="#e4572e" /></BarChart></ResponsiveContainer></div>
        <Table cols={[{ key: "scenario", label: "Scenario" }, { key: "extra_thermal_mwh", label: "Extra thermal MWh", align: "right" }, { key: "extra_unserved_mwh", label: "Extra unserved MWh", align: "right" }, { key: "hours_unserved", label: "Hours short", align: "right" }, { key: "min_reserve_pct", label: "Min reserve %", align: "right" }]} rows={d.scenarios} />
        <p className="mt-2 text-xs text-mute">{d.note}</p>
      </Card>
    </div>)}</Async>;
}

export function Fleet() {
  const r = useApi<any>("/insights/plant-fleet");
  return <Async state={r}>{d => (
    <div className="space-y-4">
      <Provenance origin={d.data_origin} extra={d.note} />
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">{d.clusters.map((c: any) => (
        <Card key={c.cluster} title={`Group ${c.cluster + 1}: ${c.label}`} subtitle={`${c.n_plants} stations · ${fmt(c.capacity_mw)} MW`}><p className="text-xs text-slate-300">{c.plants.join(", ")}</p></Card>))}</div>
      <Card title="Power station profiles" subtitle={d.method}>
        <Table cols={[{ key: "name", label: "Station" }, { key: "source", label: "Source" }, { key: "capacity_mw", label: "MW", align: "right" }, { key: "capacity_factor_pct", label: "CF %", align: "right" }, { key: "availability_pct", label: "Avail %", align: "right" }, { key: "curtailed_mwh_30d", label: "Curtailed MWh", align: "right" }, { key: "max_unit_risk", label: "Max risk", align: "right" }, { key: "flagged_units", label: "Flagged", align: "right" }, { key: "share_of_72h_supply_pct", label: "72 h share %", align: "right" }, { key: "cluster", label: "Group", align: "right", render: (r: any) => r.cluster + 1 }]} rows={d.plants} />
      </Card>
    </div>)}</Async>;
}

export function TheftPlanner() {
  const r = useApi<any>("/insights/theft-planner");
  return <Async state={r}>{d => (
    <div className="space-y-4">
      <Provenance origin={d.data_origin} extra={d.method} />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Flagged feeders" value={fmt(d.flagged_feeders)} tone="warn" /><Stat label="Estimated NTL" value={fmt(d.total_ntl_mwh)} unit="MWh" />
        {d.pareto.slice(0, 2).map((p: any) => <Stat key={p.feeders_inspected} label={`Inspect top ${p.feeders_inspected}`} value={fmt(p.share_of_estimated_ntl_pct, 1)} unit="% of NTL" />)}
      </div>
      <Card title="Regions vs allowable loss" subtitle={`Allowable ${d.allowable_loss_pct}% (value quoted in the briefing)`}>
        <Table cols={[{ key: "region", label: "Region" }, { key: "loss_pct", label: "Loss %", align: "right" }, { key: "ntl_pct", label: "NTL %", align: "right" }, { key: "tech_pct", label: "Technical %", align: "right" }, { key: "mwh_above_allowable", label: "MWh above allowable", align: "right" }]} rows={d.regions_vs_allowable} />
      </Card>
      <Card title="Inspection priority list"><Table cols={[{ key: "rank", label: "#" }, { key: "feeder_id", label: "Feeder" }, { key: "region", label: "Region" }, { key: "ntl_est_pct", label: "NTL %", align: "right" }, { key: "ntl_mwh_est", label: "NTL MWh", align: "right" }, { key: "customers", label: "Customers", align: "right" }, { key: "mwh_per_100_customers", label: "MWh/100 cust.", align: "right" }]} rows={d.inspection_list} />
        <p className="mt-2 text-xs text-mute">{d.caveat}</p></Card>
    </div>)}</Async>;
}

export function DemandStructure() {
  const r = useApi<any>("/insights/demand-structure");
  return <Async state={r}>{d => (
    <div className="space-y-4">
      <Provenance origin={d.data_origin} extra={`${d.method} · ${d.window_days} days`} />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Load factor" value={fmt(d.load_factor_pct, 1)} unit="%" /><Stat label="Peak / valley" value={fmt(d.peak_to_valley, 2)} unit="×" hint={`${fmt(d.peak_mw)} / ${fmt(d.min_mw)} MW`} />
        <Stat label="Evening share 18-22h" value={fmt(d.evening_peak_share_pct, 1)} unit="%" /><Stat label="Seasonal strength" value={fmt(d.seasonal_strength, 2)} hint={`trend strength ${fmt(d.trend_strength, 2)}`} />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Average daily shape (MW)"><div className="h-60"><ResponsiveContainer><AreaChart data={d.hourly_profile_mw.map((v: number, h: number) => ({ h: `${h}h`, v }))}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="h" stroke={CHART.axis} fontSize={10} /><YAxis stroke={CHART.axis} fontSize={10} /><Tooltip contentStyle={CHART.tooltip} /><Area dataKey="v" name="MW" stroke="#f2b84b" fill="#f2b84b33" /></AreaChart></ResponsiveContainer></div></Card>
        <Card title="STL trend vs daily mean (MW)" subtitle={`56-day trend change ${fmt(d.trend_change_pct_56d, 2)}%`}><div className="h-60"><ResponsiveContainer><ComposedChart data={d.trend_daily}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="date" stroke={CHART.axis} fontSize={10} tickFormatter={(x: string) => x.slice(5)} /><YAxis stroke={CHART.axis} fontSize={10} domain={["auto", "auto"]} /><Tooltip contentStyle={CHART.tooltip} /><Legend /><Line dataKey="mean_mw" name="Daily mean" stroke="#8696b3" dot={false} /><Line dataKey="trend_mw" name="Trend" stroke="#2ec4b6" dot={false} /></ComposedChart></ResponsiveContainer></div></Card>
      </div>
    </div>)}</Async>;
}
