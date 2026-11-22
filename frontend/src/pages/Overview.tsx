import { useState } from "react";
import { Area, AreaChart, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useApi } from "../api";
import { Async, Card, CHART, dayhh, fmt, Limitations, Provenance, Select, SOURCE_COLORS, Stat } from "../components/ui";

export default function Overview() {
  const ov = useApi<any>("/overview");
  const dem = useApi<any>("/demand");
  const gen = useApi<any>("/generation");
  const [horizon, setHorizon] = useState<24 | 72>(24);
  const [demandSeries, setDemandSeries] = useState("both");
  const [generationMix, setGenerationMix] = useState("all");

  const renewableSources = ["geothermal", "hydro", "wind", "solar"];
  const conventionalSources = ["thermal", "import"];
  const visibleSources = generationMix === "renewable"
    ? renewableSources
    : generationMix === "conventional"
      ? conventionalSources
      : Object.keys(SOURCE_COLORS);

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-3 rounded-xl border border-line bg-panel px-4 py-3 md:flex-row md:items-center md:justify-between">
        <div>
          <div className="text-sm font-semibold text-slate-100">National operating view</div>
          <div className="mt-0.5 text-xs text-mute">Demand, generation and system risk</div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Select
            label="Forecast window"
            value={horizon}
            onChange={setHorizon}
            options={[{ value: 24, label: "Next 24 hours" }, { value: 72, label: "Next 72 hours" }]}
          />
          <span className="inline-flex items-center gap-2 rounded-full border border-good/30 bg-good/10 px-3 py-1.5 text-xs font-medium text-good">
            <span className="h-1.5 w-1.5 rounded-full bg-good" />System monitoring
          </span>
        </div>
      </div>

      <Async state={ov}>{o => (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <Provenance origin={o.data_origin} extra={`as of ${o.as_of} · ${o.provenance.timezone}`} />
            <span className="text-xs text-mute">Window: next {horizon} hours</span>
          </div>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Demand now" value={fmt(o.current.demand_mw)} unit="MW" hint="latest observed hour" />
            <Async state={dem}>{d => {
              const peak = d.national_mw.slice(0, horizon).reduce((max: number, v: number, i: number, rows: number[]) => v > rows[max] ? i : max, 0);
              const peakMw = d.national_mw[peak];
              return <Stat label={`Predicted ${horizon} h peak`} value={fmt(peakMw)} unit="MW" hint={`${d.time[peak].slice(5, 10)} · ${d.time[peak].slice(11, 16)} · 80% ${fmt(d.lo80_mw[peak])}–${fmt(d.hi80_mw[peak])}`} />;
            }}</Async>
            <Async state={gen}>{g => {
              const reserves: number[] = g.dispatch_next_72h.reserve_margin_pct.slice(0, horizon);
              const minimum = reserves.length ? Math.min(...reserves) : null;
              return <Stat label={`Min reserve margin ${horizon} h`} value={fmt(minimum, 1)} unit="%" tone={minimum !== null && minimum < 5 ? "bad" : minimum !== null && minimum < 10 ? "warn" : "good"} />;
            }}</Async>
            <Stat label="Renewable share now" value={fmt(o.current.renewable_share_pct, 0)} unit="%" tone="good" hint="geothermal + hydro + wind + solar" />
            <Stat label="Units flagged high-risk" value={fmt(o.assets.units_flagged_high_risk)} tone="warn" hint="14-day fault model" />
            <Stat label="Telemetry anomalies 72 h" value={fmt(o.assets.telemetry_anomalies_72h)} />
            <Stat label="Feeders flagged for losses" value={fmt(o.assets.feeders_flagged_losses)} />
            <Stat label="Lines >80% (24 h)" value={fmt(o.assets.lines_over_80pct_next_24h)} tone={o.assets.lines_over_80pct_next_24h ? "warn" : "good"} hint="DC model, illustrative" />
          </div>
        </div>
      )}</Async>

      <div className="grid items-stretch gap-4 2xl:grid-cols-2">
        <Card
          title="National demand"
          subtitle={`Observed history and forecast · next ${horizon} hours`}
          right={<Select label="Series" value={demandSeries} onChange={setDemandSeries} options={[{ value: "both", label: "Observed & forecast" }, { value: "observed", label: "Observed only" }, { value: "forecast", label: "Forecast only" }]} />}
        >
          <Async state={dem}>{d => {
            const historyRows = d.history.time.map((t: string, i: number) => ({ t: dayhh(t), hist: d.history.demand_mw[i] })).slice(-168);
            const forecastRows = d.time.slice(0, horizon).map((t: string, i: number) => ({ t: dayhh(t), fc: d.national_mw[i], band: [d.lo80_mw[i], d.hi80_mw[i]] }));
            const rows = (demandSeries === "forecast" ? [] : historyRows).concat(demandSeries === "observed" ? [] : forecastRows);
            return (
              <div className="h-72">
                <ResponsiveContainer><ComposedChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="t" stroke={CHART.axis} fontSize={10} minTickGap={32} />
                  <YAxis stroke={CHART.axis} fontSize={10} width={58} unit=" MW" />
                  <Tooltip contentStyle={CHART.tooltip} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {(demandSeries === "both" || demandSeries === "forecast") && <Area dataKey="band" name="80% interval" stroke="none" fill="#2ec4b633" />}
                  {(demandSeries === "both" || demandSeries === "observed") && <Line dataKey="hist" name="Observed (synthetic)" stroke="#f2b84b" strokeWidth={2} dot={false} />}
                  {(demandSeries === "both" || demandSeries === "forecast") && <Line dataKey="fc" name="Forecast" stroke="#2ec4b6" strokeWidth={2} dot={false} />}
                </ComposedChart></ResponsiveContainer>
              </div>
            );
          }}</Async>
        </Card>

        <Card
          title="Generation mix"
          subtitle={`Dispatch view · next ${horizon} hours`}
          right={<Select label="Sources" value={generationMix} onChange={setGenerationMix} options={[{ value: "all", label: "All sources" }, { value: "renewable", label: "Renewable mix" }, { value: "conventional", label: "Thermal & imports" }]} />}
        >
          <Async state={gen}>{g => {
            const historyRows = g.history.time.map((t: string, i: number) => ({ t: dayhh(t), geothermal: g.history.geothermal[i], hydro: g.history.hydro[i], wind: g.history.wind[i], solar: g.history.solar[i], thermal: g.history.thermal[i], import: g.history.import[i] })).slice(-168);
            const dispatchRows = g.time.slice(0, horizon).map((t: string, i: number) => ({ t: dayhh(t), geothermal: g.dispatch_next_72h.geo[i], hydro: g.dispatch_next_72h.hydro[i], wind: g.dispatch_next_72h.wind[i], solar: g.dispatch_next_72h.solar[i], thermal: g.dispatch_next_72h.thermal[i], import: g.dispatch_next_72h.imp[i] }));
            const rows = historyRows.concat(dispatchRows);
            return (
              <div className="h-72">
                <ResponsiveContainer><AreaChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="t" stroke={CHART.axis} fontSize={10} minTickGap={32} />
                  <YAxis stroke={CHART.axis} fontSize={10} width={58} unit=" MW" />
                  <Tooltip contentStyle={CHART.tooltip} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {visibleSources.map(k => <Area key={k} type="monotone" stackId="1" dataKey={k} name={k[0].toUpperCase() + k.slice(1)} stroke={SOURCE_COLORS[k]} fill={SOURCE_COLORS[k]} fillOpacity={0.72} />)}
                </AreaChart></ResponsiveContainer>
              </div>
            );
          }}</Async>
        </Card>
      </div>

      <Async state={ov}>{o => <Card title="Model status" subtitle="Every figure on this dashboard comes from the backend API"><div className="text-xs text-mute">Versions: {Object.entries(o.model_versions).map(([k, v]) => `${k} ${v}`).join(" · ")}. Holdout 24 h demand MAPE {fmt(o.demand_forecast_holdout_mape_24h_pct, 2)}%.</div><Limitations items={[o.provenance.caveat, "All history is synthetic; no live-connected data."]} /></Card>}</Async>
    </div>
  );
}
