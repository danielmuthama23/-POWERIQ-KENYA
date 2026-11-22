import { Area, Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useApi } from "../api";
import { Async, Card, CHART, dayhh, Empty, fmt, Limitations, Pill, Provenance, SOURCE_COLORS, Stat, Table } from "../components/ui";

function MetricsTable({ metrics }: { metrics: Record<string, any> }) {
  const rows = Object.values(metrics).map((m: any) => ({ ...m, mae: m.model.mae, base: Math.min(m.baseline_persistence.mae, m.baseline_same_hour_last_available_day.mae), rmse: m.model.rmse, mape: m.model.mape_pct }));
  return <Table cols={[{ key: "source", label: "Source" }, { key: "horizon_h", label: "Horizon h", align: "right" },
    { key: "mae", label: "MAE MW", align: "right", render: r => fmt(r.mae, 1) }, { key: "rmse", label: "RMSE MW", align: "right", render: r => fmt(r.rmse, 1) },
    { key: "mape", label: "MAPE %", align: "right", render: r => fmt(r.mape, 1) }, { key: "base", label: "Best baseline MAE", align: "right", render: r => fmt(r.base, 1) },
    { key: "beats", label: "Beats baseline (MAE)", render: r => r.beats_best_baseline_mae ? <Pill tone="good">yes</Pill> : <Pill tone="warn">no{r.beats_best_baseline_rmse ? " (wins RMSE)" : ""}</Pill> }]} rows={rows} />;
}

export function Generation() {
  const g = useApi<any>("/generation");
  return (
    <Async state={g}>{d => {
      const rows = d.time.map((t: string, i: number) => ({ t: dayhh(t), ...Object.fromEntries(Object.keys(d.mean_mw).map(k => [k, d.mean_mw[k][i]])) }));
      return (
        <div className="space-y-4">
          <Provenance origin={d.data_origin} model={d.model_version} />
          <Card title="72 h generation forecast by source (MW)"><div className="h-72"><ResponsiveContainer><ComposedChart data={rows}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="t" stroke={CHART.axis} fontSize={10} interval={5} /><YAxis stroke={CHART.axis} fontSize={10} width={52} /><Tooltip contentStyle={CHART.tooltip} /><Legend />
            {Object.keys(SOURCE_COLORS).map(k => <Line key={k} dataKey={k} stroke={SOURCE_COLORS[k]} dot={false} strokeWidth={2} />)}</ComposedChart></ResponsiveContainer></div></Card>
          <div className="grid gap-4 xl:grid-cols-3">
            <Card title="Installed capacity (approximate, synthetic)"><Table cols={[{ key: "k", label: "Source" }, { key: "v", label: "MW", align: "right" }]} rows={Object.entries(d.installed_capacity_mw).map(([k, v]) => ({ k, v: fmt(v as number) }))} /></Card>
            <Card className="xl:col-span-2" title="Held-out accuracy vs baselines" subtitle="Last 28 days; model vs persistence and same-hour-yesterday"><div className="max-h-72 overflow-y-auto"><MetricsTable metrics={d.metrics} /></div></Card>
          </div>
          <Limitations items={[d.note]} />
        </div>
      );
    }}</Async>
  );
}

export function Maintenance() {
  const m = useApi<any>("/maintenance");
  return (
    <Async state={m}>{d => {
      const mm = d.metrics.model, bb = d.metrics.baseline_vibration_threshold_p90;
      const imp = Object.entries(d.feature_importance).slice(0, 8).map(([k, v]) => ({ k, v: v as number }));
      return (
        <div className="space-y-4">
          <Provenance origin={d.data_origin} model={d.model_version} extra={`label: ${d.label_definition}`} />
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="Precision" value={fmt(mm.precision, 2)} hint="share of flagged that faulted" /><Stat label="Recall" value={fmt(mm.recall, 2)} hint="share of faults caught" />
            <Stat label="F1 vs baseline" value={fmt(mm.f1, 2)} hint={`vibration-threshold F1 ${fmt(bb.f1, 2)}`} /><Stat label="ROC AUC" value={fmt(mm.roc_auc, 2)} hint={`confusion: TP ${mm.confusion_matrix.tp} FP ${mm.confusion_matrix.fp} FN ${mm.confusion_matrix.fn} TN ${mm.confusion_matrix.tn}`} />
          </div>
          <div className="grid gap-4 xl:grid-cols-3">
            <Card className="xl:col-span-2" title="Units ranked by 14-day failure risk"><div className="max-h-[520px] overflow-y-auto"><Table cols={[
              { key: "unit_id", label: "Unit" }, { key: "plant_name", label: "Plant" }, { key: "source", label: "Source" },
              { key: "risk_probability", label: "Risk", align: "right", render: r => <span className="inline-flex items-center gap-2"><span className="inline-block h-1.5 rounded bg-danger" style={{ width: 60 * r.risk_probability }} />{fmt(r.risk_probability * 100, 0)}%</span> },
              { key: "flagged", label: "", render: r => r.flagged ? <Pill tone="bad">flagged</Pill> : null },
              { key: "drivers", label: "Drivers", render: r => <span className="text-mute">{r.drivers.slice(0, 2).join("; ")}</span> }]} rows={d.units} /></div></Card>
            <Card title="What the model uses" subtitle="XGBoost feature importance"><div className="h-72"><ResponsiveContainer><BarChart data={imp} layout="vertical" margin={{ left: 40 }}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis type="number" stroke={CHART.axis} fontSize={10} /><YAxis type="category" dataKey="k" stroke={CHART.axis} fontSize={10} width={110} /><Tooltip contentStyle={CHART.tooltip} /><Bar dataKey="v" fill="#f2b84b" /></BarChart></ResponsiveContainer></div></Card>
          </div>
          <Limitations items={d.limitations} />
        </div>
      );
    }}</Async>
  );
}

export function Grid() {
  const g = useApi<any>("/grid?hour=19"); const n1 = useApi<any>("/grid/n1");
  return (
    <div className="space-y-4">
      <Async state={g}>{d => (
        <>
          <Provenance origin={d.data_origin} extra="DC approximation, illustrative network" />
          <Card title="Line loading at the evening peak (hour +20) and 72 h maximum" subtitle={d.label}>
            <div className="h-80"><ResponsiveContainer><BarChart data={d.lines.map((l: any) => ({ id: l.line_id, now: l.loading_pct, max: l.max_loading_pct_72h }))}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="id" stroke={CHART.axis} fontSize={9} angle={-35} textAnchor="end" height={80} interval={0} /><YAxis stroke={CHART.axis} fontSize={10} unit="%" /><Tooltip contentStyle={CHART.tooltip} /><Legend /><Bar dataKey="now" name="At peak hour" fill="#2ec4b6" /><Bar dataKey="max" name="72 h max" fill="#f2b84b" /></BarChart></ResponsiveContainer></div>
          </Card>
          <Card title="Bottlenecks"><Table cols={[{ key: "line_id", label: "Line" }, { key: "m", label: "Max loading %", align: "right", render: r => fmt(r.max_loading_pct, 0) }]} rows={d.bottlenecks} empty="No line exceeds 70% of its assumed capacity in the next 72 h." /></Card>
        </>
      )}</Async>
      <Async state={n1}>{d => (
        <Card title="N-1 contingency screening" subtitle={`${d.contingencies_with_violations} of ${d.n_contingencies} single-line outages cause a violation at ${d.time.slice(5, 16)}`}>
          <Table cols={[{ key: "outage", label: "Outage" }, { key: "type", label: "Effect", render: r => r.islanding ? <Pill tone="warn">islands a bus (radial spur)</Pill> : <Pill tone="bad">overload</Pill> },
            { key: "o", label: "Overloaded line", render: r => r.overloads.map((x: any) => `${x.line_id} ${fmt(x.loading_pct, 0)}%`).join(", ") }]} rows={d.violations} empty="No N-1 violations." />
          <Limitations items={[d.label]} />
        </Card>
      )}</Async>
    </div>
  );
}

export function Renewable() {
  const c = useApi<any>("/curtailment"); const g = useApi<any>("/generation");
  return (
    <div className="space-y-4">
      <Async state={c}>{d => (
        <>
          <Provenance origin={d.data_origin} extra={`last ${d.window_days} days`} />
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-3"><Stat label="Curtailed (30 d)" value={fmt(d.total_curtailed_mwh)} unit="MWh" tone="warn" /><Stat label="Forecast curtailment next 24 h" value={fmt(d.forecast_next_24h_mwh)} unit="MWh" /><Stat label="Top resource" value={d.by_plant[0]?.plant ?? "none"} /></div>
          <div className="grid gap-4 xl:grid-cols-2">
            <Card title="Curtailment by resource and cause"><Table cols={[{ key: "plant", label: "Resource" }, { key: "reason", label: "Cause" }, { key: "mwh", label: "MWh", align: "right", render: r => fmt(r.mwh) }, { key: "hours", label: "Hours", align: "right" }, { key: "max_mw", label: "Peak MW", align: "right", render: r => fmt(r.max_mw) }]} rows={[...d.by_plant].sort((a: any, b: any) => b.mwh - a.mwh)} empty="No curtailment in the window." /></Card>
            <Card title="When curtailment happens (MWh by hour of day)"><div className="h-60"><ResponsiveContainer><BarChart data={Object.entries(d.by_hour_of_day_mwh).map(([h, v]) => ({ h: `${h}h`, v }))}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="h" stroke={CHART.axis} fontSize={10} /><YAxis stroke={CHART.axis} fontSize={10} /><Tooltip contentStyle={CHART.tooltip} /><Bar dataKey="v" name="MWh" fill="#2ec4b6" /></BarChart></ResponsiveContainer></div></Card>
          </div>
          <p className="text-xs text-mute">{d.opportunity}</p>
        </>
      )}</Async>
      <Async state={g}>{d => <Card title="Wind and solar forecast (MW, 80% band)"><div className="h-60"><ResponsiveContainer><ComposedChart data={d.time.map((t: string, i: number) => ({ t: dayhh(t), wind: d.mean_mw.wind[i], windBand: [d.lo80_mw.wind[i], d.hi80_mw.wind[i]], solar: d.mean_mw.solar[i] }))}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="t" stroke={CHART.axis} fontSize={10} interval={5} /><YAxis stroke={CHART.axis} fontSize={10} width={44} /><Tooltip contentStyle={CHART.tooltip} /><Legend /><Area dataKey="windBand" name="Wind 80%" stroke="none" fill="#2ec4b633" /><Line dataKey="wind" stroke="#2ec4b6" dot={false} /><Line dataKey="solar" stroke="#f2b84b" dot={false} /></ComposedChart></ResponsiveContainer></div></Card>}</Async>
    </div>
  );
}

export function Losses() {
  const l = useApi<any>("/losses");
  return (
    <Async state={l}>{d => {
      const imf = d.metrics.isolation_forest, hy = d.metrics.hybrid_iforest_and_ntl_ge_4pct, bs = d.metrics.baseline_ntl_est_ge_5pct;
      return (
        <div className="space-y-4">
          <Provenance origin={d.data_origin} model={d.model_version} />
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4"><Stat label="Total loss" value={fmt(d.totals.loss_pct, 1)} unit="%" tone="warn" /><Stat label="Est. non-technical" value={fmt(d.totals.ntl_mwh)} unit="MWh/mo" /><Stat label="Est. technical" value={fmt(d.totals.tech_mwh)} unit="MWh/mo" /><Stat label="Held-out F1 (hybrid)" value={fmt(hy.f1, 2)} hint={`IForest alone ${fmt(imf.f1, 2)} · 5% rule ${fmt(bs.f1, 2)}`} /></div>
          <div className="grid gap-4 xl:grid-cols-2">
            <Card title="Loss by region (%)"><div className="h-72"><ResponsiveContainer><BarChart data={d.regions}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="region" stroke={CHART.axis} fontSize={9} angle={-25} textAnchor="end" height={60} interval={0} /><YAxis stroke={CHART.axis} fontSize={10} unit="%" /><Tooltip contentStyle={CHART.tooltip} /><Legend /><Bar dataKey="tech_pct" name="Technical" stackId="a" fill="#3a86ff" /><Bar dataKey="ntl_pct" name="Non-technical" stackId="a" fill="#e4572e" /></BarChart></ResponsiveContainer></div></Card>
            <Card title="Flagged feeders (largest est. NTL)"><div className="max-h-72 overflow-y-auto"><Table cols={[{ key: "feeder_id", label: "Feeder" }, { key: "region", label: "Region" }, { key: "ntl_est_pct", label: "NTL %", align: "right", render: r => fmt(r.ntl_est_pct * 100, 1) }, { key: "ntl_mwh_est", label: "MWh/mo", align: "right", render: r => fmt(r.ntl_mwh_est, 0) }]} rows={d.flagged_feeders} /></div></Card>
          </div>
          <Limitations items={[d.caveat, "Flags justify a field audit, not enforcement."]} />
        </div>
      );
    }}</Async>
  );
}

export function Climate() {
  const c = useApi<any>("/climate");
  return (
    <Async state={c}>{d => (
      <div className="space-y-4">
        <Provenance origin={d.data_origin} extra="Open-Meteo / NASA POWER not enabled" />
        <Card title="Reservoir water factor (60 days)" subtitle="Rainfall-derived index driving hydro availability"><div className="h-56"><ResponsiveContainer><LineChart data={d.water_factor_dates.map((t: string, i: number) => ({ t, wf: d.water_factor_history[i] }))}><CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" /><XAxis dataKey="t" stroke={CHART.axis} fontSize={10} /><YAxis domain={[0.4, 1]} stroke={CHART.axis} fontSize={10} /><Tooltip contentStyle={CHART.tooltip} /><Line dataKey="wf" stroke="#3a86ff" dot={false} strokeWidth={2} /></LineChart></ResponsiveContainer></div></Card>
        <Card title="Next 24 h mean weather by region (synthetic forecast)"><Table cols={[{ key: "region", label: "Region" }, { key: "temp_c", label: "Temp °C", align: "right" }, { key: "ghi_wm2", label: "Irradiance W/m²", align: "right", render: r => fmt(r.ghi_wm2, 0) }, { key: "wind_ms", label: "Wind m/s", align: "right" }, { key: "cloud", label: "Cloud", align: "right" }, { key: "rain_mm", label: "Rain mm/h", align: "right" }]} rows={d.next_24h_mean_synthetic_forecast} /></Card>
        <Limitations items={[d.note]} />
      </div>
    )}</Async>
  );
}
