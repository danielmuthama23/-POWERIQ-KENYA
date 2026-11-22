import { useEffect, useMemo, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, Area } from "recharts";
import { api, useApi } from "../api";
import PowerMap, { Assets, CellsFC, SEQ_STOPS, UNITS } from "../components/PowerMap";
import { Async, Card, CHART, Empty, ErrorBox, fmt, Limitations, Pill, Provenance, Select, Table, dayhh } from "../components/ui";

const SOURCE_NOTE: Record<string, string> = {
  demand_mw: "Forecast demand in each hex cell (regional ML forecast × static within-region share).",
  generation_mw: "Dispatched output of plants located in the cell.",
  net_flow_mw: "Local generation minus demand incl. losses. Teal = exporter, red = importer.",
  import_dependence_pct: "Share of the cell's demand that must arrive over the grid.",
  loss_pct: "Technical + non-technical loss, latest month of feeder accounting.",
  ntl_pct: "Estimated non-technical (commercial) loss.",
  tech_pct: "Estimated technical loss (I²R, transformers).",
  maintenance_risk: "Highest 14-day failure probability among units in the cell.",
  renewable_mw: "Geothermal + hydro + wind + solar output in the cell.",
  curtailed_mwh_30d: "Wind/solar energy curtailed in the last 30 days.",
  line_loading_pct: "Highest DC-model loading among lines touching the cell's bus.",
  composite_risk: "Blend of loss, maintenance risk, import dependence, line loading and flagged feeders.",
};

function MapPanel() {
  const [metric, setMetric] = useState("demand_mw");
  const [hour, setHour] = useState(19);
  const [playing, setPlaying] = useState(false);
  const [sqrtScale, setSqrt] = useState(true);
  const [pitch3d, setPitch] = useState(true);
  const [layers, setLayers] = useState({ lines: true, plants: true, subs: true });
  const [cells, setCells] = useState<CellsFC | null>(null);
  const [assets, setAssets] = useState<Assets | null>(null);
  const [err, setErr] = useState<Error | null>(null);
  const timeline = useApi<any>("/map/timeline");
  const seq = useRef(0);

  useEffect(() => {
    const my = ++seq.current;
    const t = setTimeout(() => {
      Promise.all([api<CellsFC>(`/map/cells?hour=${hour}`), api<Assets>(`/map/assets?hour=${hour}`)])
        .then(([c, a]) => { if (my === seq.current) { setCells(c); setAssets(a); setErr(null); } })
        .catch(e => my === seq.current && setErr(e));
    }, 120);
    return () => clearTimeout(t);
  }, [hour]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setHour(h => (h + 1) % 72), 1100);
    return () => clearInterval(id);
  }, [playing]);

  const metrics = cells?.meta.metrics ?? {};
  const mx = cells?.meta.max[metric] ?? 0;
  const top = useMemo(() => cells ? [...cells.features].sort((a, b) => Math.abs(b.properties[metric]) - Math.abs(a.properties[metric])).slice(0, 6).map(f => f.properties) : [], [cells, metric]);
  const tl = timeline.data ? timeline.data.time.map((t: string, i: number) => ({ t: dayhh(t), i, demand: timeline.data.demand_mw[i], unserved: timeline.data.unserved_mw[i], line: timeline.data.max_line_loading_pct[i] })) : [];

  if (err) return <ErrorBox error={err} />;
  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_320px]">
      <div className="relative h-[640px]">
        <PowerMap cells={cells} assets={assets} metric={metric} sqrtScale={sqrtScale} pitch3d={pitch3d} layers={layers} />
        <div className="pointer-events-none absolute left-3 top-3 rounded-lg bg-ink/80 px-3 py-2 text-xs backdrop-blur">
          <div className="font-semibold text-slate-100">{metrics[metric] ?? "Loading"}</div>
          <div className="text-mute">{cells ? dayhh(cells.meta.time) + " EAT (forecast)" : ""}</div>
        </div>
        <div className="pointer-events-none absolute bottom-3 left-3 rounded-lg bg-ink/80 px-3 py-2 text-xs backdrop-blur">
          <div className="mb-1 text-mute">{metric === "net_flow_mw" ? "importer → exporter" : "low → high"} ({UNITS[metric]})</div>
          <div className="h-2 w-52 rounded" style={{ background: metric === "net_flow_mw" ? "linear-gradient(90deg,#d1495b,#26334d,#2ec4b6)" : `linear-gradient(90deg,${SEQ_STOPS.join(",")})` }} />
          <div className="mt-1 flex justify-between font-mono text-mute"><span>{metric === "net_flow_mw" ? fmt(-mx, 0) : "0"}</span><span>{fmt(mx, mx < 10 ? 1 : 0)}</span></div>
          <div className="mt-1 text-mute">Height = value{sqrtScale ? " (√ scale)" : ""}. Columns: plants. Lines: loading.</div>
        </div>
      </div>

      <div className="space-y-4">
        <Card title="Metric" subtitle="Colour and height of each hex">
          <div className="grid gap-1" role="radiogroup" aria-label="Map metric">
            {Object.entries(metrics).map(([k, label]) => (
              <button key={k} role="radio" aria-checked={metric === k} onClick={() => setMetric(k)}
                className={`rounded-md border px-2 py-1.5 text-left text-xs ${metric === k ? "border-amber bg-amber/10 text-amber" : "border-line text-slate-300 hover:bg-panel2"}`}>{label}</button>
            ))}
          </div>
          <p className="mt-2 text-xs text-mute">{SOURCE_NOTE[metric]}</p>
        </Card>
        <Card title="Time (next 72 h)" subtitle={cells ? dayhh(cells.meta.time) : ""}>
          <input aria-label="Forecast hour" type="range" min={0} max={71} value={hour} onChange={e => setHour(Number(e.target.value))} className="w-full accent-amber" />
          <div className="mt-2 flex items-center gap-2">
            <button onClick={() => setPlaying(p => !p)} className="rounded-md border border-line px-3 py-1 text-xs hover:bg-panel2">{playing ? "Pause" : "Play"}</button>
            <button onClick={() => setHour(19)} className="rounded-md border border-line px-3 py-1 text-xs hover:bg-panel2">Evening peak</button>
            <span className="font-mono text-xs text-mute">+{hour + 1} h</span>
          </div>
        </Card>
        <Card title="View">
          <div className="space-y-1 text-xs">
            {([["sqrt", "Square-root height scale", sqrtScale, setSqrt], ["pitch", "3D tilt", pitch3d, setPitch]] as const).map(([k, l, v, f]) => (
              <label key={k} className="flex items-center gap-2"><input type="checkbox" checked={v} onChange={e => f(e.target.checked)} />{l}</label>
            ))}
            {(["lines", "plants", "subs"] as const).map(k => (
              <label key={k} className="flex items-center gap-2"><input type="checkbox" checked={layers[k]} onChange={e => setLayers({ ...layers, [k]: e.target.checked })} />{{ lines: "Transmission lines", plants: "Power plants", subs: "Substations" }[k]}</label>
            ))}
          </div>
        </Card>
        <Card title="Hotspots" subtitle={metrics[metric]}>
          <Table cols={[{ key: "cell_id", label: "Cell" }, { key: "region", label: "Region", render: r => r.region.replace("_", " ") },
            { key: "v", label: UNITS[metric], align: "right", render: r => fmt(r[metric], Math.abs(r[metric]) < 10 ? 1 : 0) }]} rows={top} />
        </Card>
      </div>

      <Card className="xl:col-span-2" title="72 h national demand, unserved energy and peak line loading" subtitle="Click the chart to jump to an hour"
        right={cells && <Provenance origin={cells.meta.data_origin} model={cells.meta.model_version} />}>
        <div className="h-52" aria-label="national timeline chart">
          <ResponsiveContainer>
            <ComposedChart data={tl} onClick={(e: any) => e?.activePayload && setHour(e.activePayload[0].payload.i)}>
              <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" />
              <XAxis dataKey="t" stroke={CHART.axis} fontSize={10} interval={5} />
              <YAxis yAxisId="a" stroke={CHART.axis} fontSize={10} unit=" MW" width={64} />
              <YAxis yAxisId="b" orientation="right" stroke={CHART.axis} fontSize={10} unit="%" width={44} />
              <Tooltip contentStyle={CHART.tooltip} />
              <Legend />
              <Area yAxisId="a" type="monotone" dataKey="demand" name="Demand (MW)" stroke="#f2b84b" fill="#f2b84b22" />
              <Line yAxisId="a" type="monotone" dataKey="unserved" name="Unserved (MW)" stroke="#e4572e" dot={false} />
              <Line yAxisId="b" type="monotone" dataKey="line" name="Max line loading (%)" stroke="#2ec4b6" dot={false} strokeDasharray="4 3" />
              {tl[hour] && <ReferenceLine yAxisId="a" x={tl[hour].t} stroke="#fff" strokeDasharray="2 2" />}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <Limitations items={[cells?.meta.geometry_note ?? "", cells ? "Composite index: " + cells.meta.composite_formula : ""].filter(Boolean)} />
      </Card>
    </div>
  );
}

function UsagePanel() {
  const u = useApi<any>("/usage");
  const [region, setRegion] = useState("NAIROBI");
  return (
    <Async state={u}>{d => {
      const prof = (d.hourly_profile_mw[region] ?? []).map((v: number, h: number) => ({ h: `${h}h`, mw: v }));
      return (
        <div className="grid gap-4 xl:grid-cols-3">
          <Card className="xl:col-span-2" title="Regional usage analysis" subtitle="Last 30 days of history plus next-24 h outlook" right={<Provenance origin={d.data_origin} />}>
            <Table cols={[
              { key: "name", label: "Region" },
              { key: "energy_gwh_30d", label: "Energy GWh", align: "right", render: r => fmt(r.energy_gwh_30d, 1) },
              { key: "share_of_national_pct", label: "Share %", align: "right", render: r => fmt(r.share_of_national_pct, 1) },
              { key: "peak_mw", label: "Peak MW", align: "right", render: r => fmt(r.peak_mw, 0) },
              { key: "peak_hour", label: "Peak hr", align: "right" },
              { key: "load_factor_pct", label: "Load factor %", align: "right", render: r => fmt(r.load_factor_pct, 0) },
              { key: "growth_vs_prev_30d_pct", label: "Growth %", align: "right", render: r => fmt(r.growth_vs_prev_30d_pct, 1) },
              { key: "loss_pct", label: "Loss %", align: "right", render: r => <span className={r.loss_pct > 14 ? "text-danger" : ""}>{fmt(r.loss_pct, 1)}</span> },
              { key: "import_dependence_pct", label: "Import dep. %", align: "right", render: r => fmt(r.import_dependence_pct, 0) },
            ]} rows={d.regions} />
          </Card>
          <Card title="Daily load shape" right={<Select label="Region" value={region} onChange={setRegion} options={d.regions.map((r: any) => ({ value: r.region, label: r.name }))} />}>
            <div className="h-56"><ResponsiveContainer><BarChart data={prof}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="h" stroke={CHART.axis} fontSize={10} interval={2} /><YAxis stroke={CHART.axis} fontSize={10} width={44} /><Tooltip contentStyle={CHART.tooltip} /><Bar dataKey="mw" name="Mean MW" fill="#f2b84b" /></BarChart></ResponsiveContainer></div>
          </Card>
        </div>
      );
    }}</Async>
  );
}

const PRESET_FALLBACK = "Geothermal output -15%";

function ScenarioPanel() {
  const presets = useApi<any>("/scenarios/presets");
  const [sel, setSel] = useState(PRESET_FALLBACK);
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<Error | null>(null);
  const run = async () => {
    const p = presets.data?.presets.find((x: any) => x.name === sel); if (!p) return;
    setBusy(true); setErr(null);
    try { setRes(await api("/scenarios/run", { method: "POST", body: p })); } catch (e) { setErr(e as Error); } finally { setBusy(false); }
  };
  const ser = res ? res.series.time.map((t: string, i: number) => ({ t: t.slice(11, 13) + "h", base: res.series.baseline_unserved_mw[i], scen: res.series.unserved_mw[i], reserve: res.series.reserve_margin_pct[i], thermal: res.series.thermal_mw[i] })) : [];
  return (
    <Card title="Scenario simulation" subtitle="Advisory what-if on the forecast baseline (requires analyst role)"
      right={<div className="flex items-center gap-2"><Select label="Preset" value={sel} onChange={setSel} options={(presets.data?.presets ?? [{ name: PRESET_FALLBACK }]).map((p: any) => ({ value: p.name, label: p.name }))} />
        <button disabled={busy || !presets.data} onClick={run} className="rounded-md bg-amber px-3 py-1 text-xs font-semibold text-ink disabled:opacity-50">{busy ? "Running…" : "Run"}</button></div>}>
      {err && <ErrorBox error={err} />}
      {!res && !err && <Empty>Choose a preset and run it to compare against the baseline forecast.</Empty>}
      {res && (
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <p className="mb-2 text-xs text-mute">{res.applied_changes.join(" · ")}</p>
            <Table cols={[{ key: "k", label: "24 h metric" }, { key: "b", label: "Baseline", align: "right" }, { key: "s", label: "Scenario", align: "right" }]} rows={[
              { k: "Unserved energy (MWh)", b: fmt(res.baseline.unserved_energy_mwh), s: fmt(res.scenario.unserved_energy_mwh) },
              { k: "Min reserve margin (%)", b: fmt(res.baseline.min_reserve_margin_pct, 1), s: fmt(res.scenario.min_reserve_margin_pct, 1) },
              { k: "Thermal energy (MWh)", b: fmt(res.baseline.thermal_energy_mwh), s: fmt(res.scenario.thermal_energy_mwh) },
              { k: "Hydro energy (MWh)", b: fmt(res.baseline.hydro_energy_mwh), s: fmt(res.scenario.hydro_energy_mwh) },
              { k: "Curtailed (MWh)", b: fmt(res.baseline.curtailed_energy_mwh), s: fmt(res.scenario.curtailed_energy_mwh) }]} />
            <div className="mt-3 space-y-1">{res.warnings.length ? res.warnings.map((w: string, i: number) => <div key={i}><Pill tone="warn">warning</Pill> <span className="text-xs">{w}</span></div>) : <Pill tone="good">no adequacy or line warnings</Pill>}</div>
            {res.recommendation_records?.length > 0 && <p className="mt-2 text-xs text-mute">Recommendation filed as pending approval: {res.recommendation_records[0].id}</p>}
          </div>
          <div className="h-60"><ResponsiveContainer><ComposedChart data={ser}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="t" stroke={CHART.axis} fontSize={10} /><YAxis yAxisId="a" stroke={CHART.axis} fontSize={10} width={44} /><YAxis yAxisId="b" orientation="right" stroke={CHART.axis} fontSize={10} unit="%" width={44} /><Tooltip contentStyle={CHART.tooltip} /><Legend />
            <Line yAxisId="a" dataKey="base" name="Baseline unserved MW" stroke="#8696b3" dot={false} /><Line yAxisId="a" dataKey="scen" name="Scenario unserved MW" stroke="#e4572e" dot={false} /><Line yAxisId="b" dataKey="reserve" name="Reserve margin %" stroke="#2ec4b6" dot={false} strokeDasharray="4 3" /></ComposedChart></ResponsiveContainer></div>
        </div>
      )}
      {res && <p className="mt-3 text-xs text-mute">{res.label} {res.note}</p>}
    </Card>
  );
}

export default function DigitalTwin() {
  return (
    <div className="space-y-4">
      <MapPanel />
      <UsagePanel />
      <ScenarioPanel />
    </div>
  );
}
