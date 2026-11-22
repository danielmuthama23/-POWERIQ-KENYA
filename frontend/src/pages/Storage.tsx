import { useState } from "react";
import { Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Card, CHART, Empty, ErrorBox, fmt, Pill, Provenance, Stat, Table } from "../components/ui";

export default function Storage() {
  const [power, setPower] = useState(150); const [energy, setEnergy] = useState(600);
  const [res, setRes] = useState<any>(null); const [cmp, setCmp] = useState<any>(null);
  const [busy, setBusy] = useState(false); const [err, setErr] = useState<Error | null>(null);
  const run = async () => {
    setBusy(true); setErr(null);
    try {
      const [r, c] = await Promise.all([
        api("/storage/optimize", { method: "POST", body: { power_mw: power, energy_mwh: energy } }),
        api("/storage/compare", { method: "POST", body: { options: [{ power_mw: 50, energy_mwh: 200 }, { power_mw: 100, energy_mwh: 400 }, { power_mw: 150, energy_mwh: 600 }, { power_mw: 250, energy_mwh: 1000 }] } })]);
      setRes(r); setCmp(c);
    } catch (e) { setErr(e as Error); } finally { setBusy(false); }
  };
  const rows = res ? res.time.map((t: string, i: number) => ({ t: t.slice(11, 13) + "h", before: res.net_load_before_mw[i], after: res.net_load_after_mw[i], ch: res.charge_mw[i], dis: -res.discharge_mw[i], soc: res.soc_mwh[i], curt: res.curtailed_mw_available[i] })) : [];
  return (
    <div className="space-y-4">
      <Card title="Battery storage optimisation" subtitle="Linear programme (OR-Tools GLOP) on the 24 h forecast residual load. Requires analyst role."
        right={<div className="flex flex-wrap items-center gap-2 text-xs text-mute">
          <label>Power MW <input aria-label="Power MW" type="number" min={1} value={power} onChange={e => setPower(Number(e.target.value))} className="ml-1 w-20 rounded-md border border-line bg-panel2 px-2 py-1 text-slate-200" /></label>
          <label>Energy MWh <input aria-label="Energy MWh" type="number" min={1} value={energy} onChange={e => setEnergy(Number(e.target.value))} className="ml-1 w-24 rounded-md border border-line bg-panel2 px-2 py-1 text-slate-200" /></label>
          <button onClick={run} disabled={busy} className="rounded-md bg-amber px-3 py-1 font-semibold text-ink disabled:opacity-50">{busy ? "Solving…" : "Optimise"}</button></div>}>
        {err && <ErrorBox error={err} />}
        {!res && !err && <Empty>Set a battery size and optimise to see the charge/discharge schedule.</Empty>}
        {res && <>
          <Provenance origin={res.data_origin} model={res.model_version} extra={`solver status ${res.status} · schedule independently validated: ${res.schedule_valid ? "yes" : "NO"}`} />
          <div className="my-3 grid grid-cols-2 gap-3 lg:grid-cols-4"><Stat label="Peak before" value={fmt(res.peak_before_mw)} unit="MW" /><Stat label="Peak after" value={fmt(res.peak_after_mw)} unit="MW" tone="good" /><Stat label="Energy discharged" value={fmt(res.energy_discharged_mwh)} unit="MWh" hint={`${fmt(res.equivalent_full_cycles, 2)} cycles`} /><Stat label="Degradation cost" value={fmt(res.degradation_cost_usd)} unit="USD" hint="throughput proxy" /></div>
          <div className="grid gap-4 xl:grid-cols-2">
            <div className="h-64"><ResponsiveContainer><ComposedChart data={rows}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="t" stroke={CHART.axis} fontSize={10} /><YAxis stroke={CHART.axis} fontSize={10} width={52} domain={["auto", "auto"]} /><Tooltip contentStyle={CHART.tooltip} /><Legend /><Line dataKey="before" name="Residual load before" stroke="#8696b3" dot={false} /><Line dataKey="after" name="After battery" stroke="#f2b84b" dot={false} strokeWidth={2} /></ComposedChart></ResponsiveContainer></div>
            <div className="h-64"><ResponsiveContainer><ComposedChart data={rows}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="t" stroke={CHART.axis} fontSize={10} /><YAxis yAxisId="a" stroke={CHART.axis} fontSize={10} width={44} /><YAxis yAxisId="b" orientation="right" stroke={CHART.axis} fontSize={10} width={44} /><Tooltip contentStyle={CHART.tooltip} /><Legend /><Bar yAxisId="a" dataKey="ch" name="Charge MW" fill="#2ec4b6" /><Bar yAxisId="a" dataKey="dis" name="Discharge MW" fill="#e4572e" /><Line yAxisId="b" dataKey="soc" name="State of charge MWh" stroke="#b388eb" dot={false} /></ComposedChart></ResponsiveContainer></div>
          </div>
          <p className="mt-2 text-xs text-mute">{res.load_definition}. Assumptions: round-trip efficiency {fmt(res.assumptions.round_trip_efficiency * 100, 0)}%, SOC reserve {fmt(res.assumptions.reserve_frac * 100, 0)}%, price signal is a {res.assumptions.price_signal}.</p>
        </>}
      </Card>
      {cmp && <Card title="Sizing comparison"><Table cols={[{ key: "p", label: "Size", render: r => `${r.power_mw} MW / ${r.energy_mwh} MWh` }, { key: "peak_reduction_mw", label: "Peak cut MW", align: "right", render: r => fmt(r.peak_reduction_mw, 0) }, { key: "peak_after_mw", label: "Peak after MW", align: "right", render: r => fmt(r.peak_after_mw, 0) }, { key: "energy_discharged_mwh", label: "Discharged MWh", align: "right", render: r => fmt(r.energy_discharged_mwh, 0) }, { key: "cycles", label: "Cycles", align: "right", render: r => fmt(r.cycles, 2) }, { key: "ok", label: "Valid", render: r => r.schedule_valid ? <Pill tone="good">valid</Pill> : <Pill tone="bad">invalid</Pill> }]} rows={cmp.options} /></Card>}
    </div>
  );
}
