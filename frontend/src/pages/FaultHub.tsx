import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, useApi } from "../api";
import LocationMap from "../components/LocationMap";
import FaultMap, { FaultAlert, FaultAssets, SEV_COLOR } from "../components/FaultMap";
import { Card, ErrorBox, Empty, fmt, Pill, Stat } from "../components/ui";

const TYPE_LABEL: Record<string, string> = { plant: "Power station", substation: "Substation", line: "Power line", other: "Other" };
const STATUS_TONE: Record<string, "bad" | "warn" | "good" | "mute"> = { active: "bad", reported: "bad", detected: "bad", unverified: "warn", acknowledged: "warn", predicted: "mute", resolved: "good" };

function useDeviceLocation(setPin: (p: [number, number] | null) => void) {
  const watchId = useRef<number | null>(null);
  const [locating, setLocating] = useState(false);
  const [locationError, setLocationError] = useState<string | null>(null);
  const [accuracy, setAccuracy] = useState<number | null>(null);

  const stop = useCallback(() => {
    if (watchId.current !== null && navigator.geolocation) navigator.geolocation.clearWatch(watchId.current);
    watchId.current = null;
  }, []);

  const capture = useCallback(() => {
    if (!("geolocation" in navigator)) {
      setLocationError("GPS is not available in this browser.");
      return;
    }
    setLocating(true);
    setLocationError(null);
    stop();
    watchId.current = navigator.geolocation.watchPosition(
      position => {
        const lat = +position.coords.latitude.toFixed(6);
        const lon = +position.coords.longitude.toFixed(6);
        if (lat < -5.2 || lat > 5.6 || lon < 33.5 || lon > 42.5) {
          setLocationError("Your GPS position is outside Kenya. Move to the reporting area or select a point on the map.");
          setLocating(false);
          stop();
          return;
        }
        setPin([lat, lon]);
        setAccuracy(position.coords.accuracy);
        setLocationError(null);
        setLocating(false);
        stop();
      },
      error => {
        setLocating(false);
        setLocationError(error.code === 1 ? "Location permission was denied. Enable GPS permission or pick a point on the map." : "Could not determine the device location. Try again or pick a point on the map.");
        stop();
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 10000 }
    );
  }, [setPin, stop]);

  useEffect(() => () => stop(), [stop]);
  return { capture, stop, locating, locationError, accuracy };
}

function Report({ onDone, pin, picking, setPicking, setPin }: { onDone: () => void; pin: [number, number] | null; picking: boolean; setPicking: (b: boolean) => void; setPin: (p: [number, number] | null) => void }) {
  const [type, setType] = useState("other"), [sev, setSev] = useState("high"), [desc, setDesc] = useState(""), [asset, setAsset] = useState("");
  const [msg, setMsg] = useState<string | null>(null), [err, setErr] = useState<Error | null>(null), [busy, setBusy] = useState(false);
  const location = useDeviceLocation(setPin);
  useEffect(() => { location.capture(); }, [location.capture]);
  const submit = async (e: FormEvent) => {
    e.preventDefault(); setBusy(true); setErr(null); setMsg(null);
    try {
      const r = await api("/faults/report", { method: "POST", body: { asset_type: type, asset_id: asset.trim() || null, lat: pin?.[0] ?? null, lon: pin?.[1] ?? null, severity: sev, description: desc } });
      setMsg(`Logged ${r.id} near ${r.nearest_asset.name} (${r.region.replace("_", " ")}).`); setDesc(""); setPin(null); onDone();
    } catch (x) { setErr(x as Error); } finally { setBusy(false); }
  };
  const inp = "w-full rounded-md border border-line bg-panel2 px-2 py-1.5 text-xs";
  return (
    <Card title="Report a fault" subtitle="Real-time input from the field or SCADA operators (analyst role)">
      <form onSubmit={submit} className="space-y-2">
        <div className="grid grid-cols-2 gap-2">
          <label className="text-xs text-mute">Asset type<select className={inp} value={type} onChange={e => setType(e.target.value)}>{Object.entries(TYPE_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select></label>
          <label className="text-xs text-mute">Severity<select className={inp} value={sev} onChange={e => setSev(e.target.value)}>{["critical", "high", "medium", "low"].map(s => <option key={s}>{s}</option>)}</select></label>
        </div>
        <label className="block text-xs text-mute">Asset id (optional, e.g. SUSWA-EMBAKASI)<input className={inp} value={asset} onChange={e => setAsset(e.target.value)} /></label>
        <div className="flex items-center gap-2 text-xs">
          <button type="button" onClick={() => setPicking(!picking)} className={`rounded-md border px-2 py-1 ${picking ? "border-amber text-amber" : "border-line text-slate-300"}`}>{picking ? "Click the map…" : "Pick on map"}</button>
          <span className="font-mono text-mute">{pin ? `${pin[0]}, ${pin[1]}` : asset ? "location from asset" : "no location"}</span>
        </div>
        <label className="block text-xs text-mute">What is wrong?<textarea className={inp} rows={2} maxLength={300} value={desc} onChange={e => setDesc(e.target.value)} /></label>
        <button disabled={busy || desc.trim().length < 3 || (!pin && !asset.trim())} className="w-full rounded-md bg-amber py-1.5 text-xs font-semibold text-ink disabled:opacity-50">{busy ? "Sending…" : "Log fault"}</button>
        {msg && <p className="text-xs text-good" role="status">{msg}</p>}
        {err && <ErrorBox error={err} />}
        <div className="flex items-center gap-2 text-xs">
          <button type="button" onClick={location.capture} className={`rounded-md border px-2 py-1 ${location.locating ? "border-amber text-amber" : "border-line text-slate-300"}`}>{location.locating ? "Locating…" : "Use my location"}</button>
          {location.locationError && <span className="text-red-500">{location.locationError}</span>}
        </div>
        {location.accuracy !== null && <p className="text-xs text-mute">Accuracy: ±{location.accuracy.toFixed(0)} m</p>}
      </form>
    </Card>
  );
}

const KINDS: Record<string, string> = { smoke: "Smoke", fire: "Fire", sparking: "Sparking", missing_component: "Missing equipment", damaged: "Damaged equipment", pole_down: "Fallen pole or line", vandalism: "Vandalism or theft", sagging_line: "Sagging line", tree_on_line: "Tree on line", other: "Something else" };

function Community({ onDone, pin, picking, setPicking, setPin }: { onDone: () => void; pin: [number, number] | null; picking: boolean; setPicking: (b: boolean) => void; setPin: (p: [number, number] | null) => void }) {
  const [open, setOpen] = useState(false), [kind, setKind] = useState("smoke"), [desc, setDesc] = useState("");
  const [msg, setMsg] = useState<string | null>(null), [err, setErr] = useState<Error | null>(null), [busy, setBusy] = useState(false);
  const { capture: captureLoc, locating, locationError } = useDeviceLocation(setPin);
  const submit = async (e: FormEvent) => {
    e.preventDefault(); if (!pin) return; setBusy(true); setErr(null); setMsg(null);
    try { const r = await api("/faults/community", { method: "POST", body: { kind, lat: pin[0], lon: pin[1], description: desc } }); setMsg(`${r.message} (${r.id}: ${r.headline})`); setDesc(""); setPin(null); onDone(); }
    catch (x) { setErr(x as Error); } finally { setBusy(false); }
  };
  const inp = "w-full rounded-md border border-line bg-panel2 px-2 py-1.5 text-xs";
  return (
    <Card title="See something? Report it" subtitle="Community concerns: smoke, missing equipment, damaged lines. Checked by an operator before any crew is sent.">
      {!open ? <button onClick={() => setOpen(true)} className="w-full rounded-md border border-amber py-1.5 text-xs font-semibold text-amber">Report a concern</button> : (
        <form onSubmit={submit} className="space-y-2">
          <label className="block text-xs text-mute">What do you see?<select className={inp} value={kind} onChange={e => setKind(e.target.value)}>{Object.entries(KINDS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select></label>
          <div className="flex items-center gap-2 text-xs">
            <button type="button" onClick={() => setPicking(!picking)} className={`rounded-md border px-2 py-1 ${picking ? "border-amber text-amber" : "border-line text-slate-300"}`}>{picking ? "Click the map…" : "Mark the place on the map"}</button>
            <span className="font-mono text-mute">{pin ? `${pin[0]}, ${pin[1]}` : "no place yet"}</span>
          </div>
          <label className="block text-xs text-mute">Details (optional)<textarea className={inp} rows={2} maxLength={300} value={desc} onChange={e => setDesc(e.target.value)} /></label>
          <button disabled={busy || !pin} className="w-full rounded-md bg-amber py-1.5 text-xs font-semibold text-ink disabled:opacity-50">{busy ? "Sending…" : "Send report"}</button>
          <p className="text-xs text-mute">If anyone is in danger, call the emergency services first.</p>
          {msg && <p className="text-xs text-good" role="status">{msg}</p>}
          {err && <ErrorBox error={err} />}
          <div className="flex items-center gap-2 text-xs">
            <button type="button" onClick={captureLoc} className={`rounded-md border px-2 py-1 ${locating ? "border-amber text-amber" : "border-line text-slate-300"}`}>{locating ? "Locating…" : "Use my location"}</button>
            {locationError && <span className="text-red-500">{locationError}</span>}
          </div>
        </form>)}
    </Card>
  );
}

export default function FaultHub() {
  const [live, setLive] = useState(true), [offset, setOffset] = useState(0), [sevF, setSevF] = useState("all"), [typeF, setTypeF] = useState("all");
  const [pred, setPred] = useState(true), [sel, setSel] = useState<string | null>(null), [picking, setPicking] = useState(false), [pin, setPin] = useState<[number, number] | null>(null);
  const [data, setData] = useState<any>(null), [err, setErr] = useState<Error | null>(null), [tick, setTick] = useState(0), [note, setNote] = useState<string | null>(null), [updated, setUpdated] = useState(""), [acknowledging, setAcknowledging] = useState(false);
  const assets = useApi<FaultAssets>("/faults/assets");

  const load = useCallback(() => {
    api(`/faults/live?offset_h=${offset}&include_predicted=${pred}`).then(d => { setData(d); setErr(null
  }, [offset, pred]);
  useEffect(() => { load(); }, [load, tick]);
  useEffect(() => { if (!live) return; const id = setInterval(() => setTick(t => t + 1), 5000); return () => clearInterval(id); }, [live]);

  const alerts: FaultAlert[] = useMemo(() => (data?.alerts ?? []).filter((a: FaultAlert) => (sevF === "all" || a.severity === sevF) && (typeF === "all" || a.asset_type === typeF)), [data, sevF, typeF]);
  const cur = alerts.find(a => a.id === sel) ?? null;
  const S = data?.summary;

  const act = async (path: string, label: string) => {
    try { const r = await api(path, { method: "POST" }); setNote(label + (r.recommendation_id ? ` (${r.recommendation_id})` : "")); setTick(t => t + 1); } catch (e) { setErr(e as Error); }
  };

  const acknowledge = async () => {
    if (!cur || acknowledging) return;
    setAcknowledging(true);
    try {
      await api(`/faults/ack?alert_id=${encodeURIComponent(cur.id)}`, { method: "POST", body: {} });
      setNote("Acknowledged and recorded in the audit log");
      setTick(t => t + 1);
    } catch (e) { setErr(e as Error); }
    finally { setAcknowledging(false); }
  };

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-amber/40 bg-amber/10 px-3 py-2 text-xs text-slate-200" role="note">{data?.notice ?? "Loading fault feed…"}</div>
      {err && <ErrorBox error={err} onRetry={load} />}
      {(data?.banners ?? []).map((b: any, i: number) => <div key={i} className="rounded-lg border border-danger/50 bg-danger/10 px-3 py-2 text-sm">{b.text}</div>)}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Active now" value={fmt(S?.active_now)} tone={S?.active_now ? "bad" : "good"} hint={S ? `${S.detected ?? 0} detected by model · ${S.unverified ?? 0} to verify` : undefined} />
        <Stat label="Critical / high" value={`${S?.by_severity.critical ?? 0} / ${S?.by_severity.high ?? 0}`} tone={S?.by_severity.critical ? "bad" : "warn"} />
        <Stat label="MW at risk (active)" value={fmt(S?.mw_at_risk_active)} unit="MW" />
        <Stat label="Predicted" value={fmt(S?.predicted)} hint="model forecasts" />
        <Stat label="Stations / subs / lines" value={`${S?.by_type.plant ?? 0} / ${S?.by_type.substation ?? 0} / ${S?.by_type.line ?? 0}`} />
      </div>

      <div className="flex flex-wrap items-center gap-3 text-xs">
        <button onClick={() => setLive(l => !l)} className={`rounded-md border px-2.5 py-1 ${live ? "border-good text-good" : "border-line text-slate-300"}`} aria-pressed={live}>
          <span className={`mr-1.5 inline-block h-2 w-2 rounded-full ${live ? "animate-pulse bg-good" : "bg-mute"}`} />{live ? "Live: refreshing every 5 s" : "Paused"}
        </button>
        <span className="text-mute">Updated {updated || "…"} · feed clock {data?.as_of?.slice(0, 16) ?? ""}</span>
        <label className="flex items-center gap-2 text-mute">Replay
          <input type="range" min={-72} max={0} value={offset} onChange={e => setOffset(+e.target.value)} aria-label="Replay hours back" />
          <span className="font-mono">{offset === 0 ? "now" : `${offset} h`}</span>
        </label>
        <select aria-label="Severity filter" className="rounded-md border border-line bg-panel2 px-2 py-1" value={sevF} onChange={e => setSevF(e.target.value)}>{["all", "critical", "high", "medium", "low"].map(s => <option key={s}>{s}</option>)}</select>
        <select aria-label="Asset filter" className="rounded-md border border-line bg-panel2 px-2 py-1" value={typeF} onChange={e => setTypeF(e.target.value)}>{["all", "plant", "substation", "line", "other"].map(s => <option key={s}>{s}</option>)}</select>
        <label className="flex items-center gap-1 text-mute"><input type="checkbox" checked={pred} onChange={e => setPred(e.target.checked)} />Show predicted</label>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_380px]">
        <div className="space-y-2">
          <div className="relative h-[560px]">
            <FaultMap assets={assets.data} alerts={alerts} selected={sel} onSelect={setSel} showAssets picking={picking} pin={pin} onPick={(la, lo) => { setPin([la, lo]); setPicking(false); }} />
            <div className="pointer-events-none absolute bottom-3 left-3 rounded-lg bg-ink/85 px-3 py-2 text-xs">
              {Object.entries(SEV_COLOR).map(([k, c]) => <span key={k} className="mr-3 inline-flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: c }} />{k}</span>)}
              <div className="mt-1 text-mute">Filled pulsing = active or reported. Hollow ring = predicted. Thick line = faulted line.</div>
            </div>
          </div>
        </div>
        <div className="space-y-4">
          {cur && (
            <Card title={`#${cur.rank} ${cur.name}`} subtitle={`${TYPE_LABEL[cur.asset_type]} · ${cur.region.replace("_", " ")} · ${cur.lat.toFixed(3)}, ${cur.lon.toFixed(3)}`}
              right={<Pill tone={STATUS_TONE[cur.status]}>{cur.status}</Pill>}>
              <div className="space-y-2 text-xs">
                <div className="font-semibold text-slate-100">{cur.title}</div>
                <p className="text-slate-300">{cur.detail}</p>
                <p><span className="text-mute">Do next: </span>{cur.recommended_action}</p>
                <p className="text-mute">Respond within {cur.respond_within_h} h (illustrative target) · {fmt(cur.mw_at_risk, 0)} MW at risk · nearest {cur.nearest_asset.type} {cur.nearest_asset.name} ({cur.nearest_asset.km} km)</p>
                <p className="text-mute">Source: {cur.source}{cur.confidence !== undefined ? ` · confidence ${(cur.confidence * 100).toFixed(0)}%` : ""}{cur.corroborating_reports ? ` · ${cur.corroborating_reports} similar report(s) nearby` : ""}</p>
                {cur.ack && <p className="text-warn">Acknowledged by {cur.ack.by}{cur.ack.note ? `: ${cur.ack.note}` : ""}</p>}
                <div className="flex flex-wrap gap-2 pt-1">
                  {cur.status !== "predicted" && cur.status !== "resolved" && cur.status !== "unverified" && <button className="rounded-md border border-line px-2 py-1 hover:bg-panel2" onClick={() => act(`/faults/ack?alert_id=${encodeURIComponent(cur.id)}`, "Acknowledged")}>Acknowledge</button>}
                  <button className="rounded-md bg-amber px-2 py-1 font-semibold text-ink" onClick={() => act(`/faults/dispatch?alert_id=${encodeURIComponent(cur.id)}&offset_h=${offset}`, "Request crew dispatch")}>Request crew dispatch</button>
                  {cur.community_id && cur.status === "unverified" && <>
                    <button className="rounded-md border border-good px-2 py-1 text-good" onClick={() => act(`/faults/community/${cur.community_id}/review?action=verify`, "Verified")}>Verify</button>
                    <button className="rounded-md border border-line px-2 py-1" onClick={() => { act(`/faults/community/${cur.community_id}/review?action=dismiss`, "Dismissed"); setSel(null); }}>Dismiss</button></>}
                  {cur.detection_id && <>
                    {cur.status === "unverified" && <button className="rounded-md border border-good px-2 py-1 text-good" onClick={() => act(`/faults/detections/${cur.detection_id}/review?action=confirm`, "Confirmed")}>Confirm detection</button>}
                    <button className="rounded-md border border-line px-2 py-1" onClick={() => { act(`/faults/detections/${cur.detection_id}/review?action=dismiss`, "Dismissed as false alarm"); setSel(null); }}>False alarm</button></>}
                  {(cur as any).report_id && cur.status !== "resolved" && <button className="rounded-md border border-good px-2 py-1 text-good" onClick={() => act(`/faults/reports/${(cur as any).report_id}/resolve`, "Marked resolved")}>Mark resolved</button>}
                </div>
                {note && <p className="text-good" role="status">{note}. POWERIQ does not dispatch crews itself; a human approver decides.</p>}
              </div>
            </Card>
          )}
          <Community onDone={() => setTick(t => t + 1)} pin={pin} picking={picking} setPicking={setPicking} setPin={setPin} />
          <Report onDone={() => setTick(t => t + 1)} pin={pin} picking={picking} setPicking={setPicking} setPin={setPin} />
        </div>
      </div>

      <Card title="Real location view" subtitle="Switch between the live street map and satellite imagery to identify the exact area of a fault">
        <LocationMap assets={assets.data} alerts={alerts} selected={sel} onSelect={setSel} picking={picking} pin={pin} onPick={(la, lo) => { setPin([la, lo]); setPicking(false); }} />
      </Card>

      <Card title="Fix queue: most urgent first" subtitle="Priority = severity × MW at risk × how certain and imminent the fault is">
        {alerts.length === 0 ? <Empty>No faults match the current filters.</Empty> : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-mute"><tr><th className="py-1 pr-2">#</th><th className="pr-2">Severity</th><th className="pr-2">Asset</th><th className="pr-2">Where</th><th className="pr-2">Issue</th><th className="pr-2 text-right">MW</th><th className="pr-2">Fix within</th><th>Status</th></tr></thead>
              <tbody>{alerts.map(a => (
                <tr key={a.id} onClick={() => setSel(a.id)} className={`cursor-pointer border-t border-line hover:bg-panel2 ${a.id === sel ? "bg-panel2" : ""}`}>
                  <td className="py-1.5 pr-2 font-mono">{a.rank}</td>
                  <td className="pr-2"><span className="inline-block h-2 w-2 rounded-full" style={{ background: SEV_COLOR[a.severity] }} /> {a.severity}</td>
                  <td className="pr-2"><div className="text-slate-100">{a.name}</div><div className="text-mute">{TYPE_LABEL[a.asset_type]}</div></td>
                  <td className="pr-2 font-mono text-mute">{a.region.replace("_", " ")}<br />{a.lat.toFixed(2)}, {a.lon.toFixed(2)}</td>
                  <td className="pr-2">{a.title}</td><td className="pr-2 text-right font-mono">{fmt(a.mw_at_risk, 0)}</td>
                  <td className="pr-2">{a.respond_within_h} h</td><td><Pill tone={STATUS_TONE[a.status]}>{a.status}</Pill></td>
                </tr>))}</tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
