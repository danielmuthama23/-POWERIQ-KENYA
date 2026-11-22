import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { FaultAlert, FaultAssets, SEV_COLOR } from "./FaultMap";

// Real-world base maps. Tiles load from the providers below in the viewer's browser; attribution is required by both.
const BASES = {
  street: { label: "Street map (live)", url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", attr: "&copy; OpenStreetMap contributors", max: 19 },
  satellite: { label: "Satellite", url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr: "Imagery &copy; Esri, Maxar, Earthstar Geographics", max: 19 },
} as const;
const LABELS_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}";
type Base = keyof typeof BASES;

type Props = { assets: FaultAssets | null; alerts: FaultAlert[]; selected: string | null; onSelect: (id: string | null) => void; picking?: boolean; onPick?: (lat: number, lon: number) => void; pin?: [number, number] | null };

export default function LocationMap({ assets, alerts, selected, onSelect, picking, onPick, pin }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const tiles = useRef<L.TileLayer | null>(null);
  const labels = useRef<L.TileLayer | null>(null);
  const fg = useRef<L.LayerGroup | null>(null);
  const pinLayer = useRef<L.CircleMarker | null>(null);
  const cb = useRef({ onSelect, picking, onPick }); cb.current = { onSelect, picking, onPick };
  const [base, setBase] = useState<Base>("street");
  const [tileErr, setTileErr] = useState(0);
  const [coord, setCoord] = useState<string>("");

  useEffect(() => {
    if (!box.current) return;
    const m = L.map(box.current, { center: [-0.5, 37.9], zoom: 6, zoomControl: true, attributionControl: true, preferCanvas: true });
    map.current = m; fg.current = L.layerGroup().addTo(m);
    m.on("click", (e: L.LeafletMouseEvent) => { if (cb.current.picking) cb.current.onPick?.(+e.latlng.lat.toFixed(4), +e.latlng.lng.toFixed(4)); });
    m.on("mousemove", (e: L.LeafletMouseEvent) => setCoord(`${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)}`));
    const ro = new ResizeObserver(() => m.invalidateSize()); ro.observe(box.current);
    return () => { ro.disconnect(); m.remove(); map.current = null; };
  }, []);

  useEffect(() => {
    const m = map.current; if (!m) return;
    tiles.current?.remove(); labels.current?.remove(); setTileErr(0);
    const b = BASES[base];
    tiles.current = L.tileLayer(b.url, { attribution: b.attr, maxZoom: b.max, crossOrigin: true }).on("tileerror", () => setTileErr(n => n + 1)).addTo(m);
    tiles.current.bringToBack();
    if (base === "satellite") labels.current = L.tileLayer(LABELS_URL, { maxZoom: 19, opacity: 0.9, attribution: "Labels &copy; Esri" }).addTo(m);
  }, [base]);

  useEffect(() => {
    const g = fg.current; if (!g) return;
    g.clearLayers();
    if (assets) {
      const lineCol = base === "satellite" ? "#9fc3ff" : "#4b6a99";
      assets.lines.forEach(l => L.polyline(l.coords.map((c: number[]) => [c[1], c[0]] as [number, number]), { color: lineCol, weight: 2, opacity: 0.8 }).bindTooltip(l.name + " line").addTo(g));
      assets.substations.forEach(s => L.circleMarker([s.lat, s.lon], { radius: 4, color: "#0b1322", weight: 1, fillColor: "#e8eefc", fillOpacity: 1 }).bindTooltip(s.name + " substation").addTo(g));
      assets.plants.forEach(p => L.circleMarker([p.lat, p.lon], { radius: 5, color: "#0b1322", weight: 1, fillColor: "#2ec4b6", fillOpacity: 1 }).bindTooltip(p.name + " (" + p.source + ")").addTo(g));
    }
    alerts.forEach(a => {
      const c = SEV_COLOR[a.severity], live = a.status !== "predicted" && a.status !== "resolved" && a.status !== "unverified", sel = a.id === selected;
      if (a.geometry) L.polyline(a.geometry.map(q => [q[1], q[0]] as [number, number]), { color: c, weight: sel ? 7 : 4, opacity: 0.95, dashArray: live ? undefined : "8 6" }).on("click", () => cb.current.onSelect(a.id)).addTo(g);
      L.circleMarker([a.lat, a.lon], { radius: sel ? 13 : 9, color: sel ? "#ffffff" : c, weight: sel ? 4 : 3, fillColor: c, fillOpacity: live ? 0.9 : 0.15 })
        .bindTooltip(`#${a.rank} ${a.name}: ${a.title}`).on("click", () => cb.current.onSelect(a.id)).addTo(g);
    });
  }, [assets, alerts, selected, base]);

  useEffect(() => {
    const m = map.current; if (!m) return;
    pinLayer.current?.remove(); pinLayer.current = null;
    if (pin) pinLayer.current = L.circleMarker(pin, { radius: 8, color: "#f2b84b", weight: 4, fillColor: "#ffffff", fillOpacity: 1 }).addTo(m);
  }, [pin]);

  const a = alerts.find(x => x.id === selected);
  useEffect(() => { const m = map.current; if (m && a) m.setView([a.lat, a.lon], Math.max(m.getZoom(), 14), { animate: true }); }, [selected]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (map.current) map.current.getContainer().style.cursor = picking ? "crosshair" : ""; }, [picking]);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <div role="radiogroup" aria-label="Base map" className="inline-flex overflow-hidden rounded-md border border-line">
          {(Object.keys(BASES) as Base[]).map(k => (
            <button key={k} role="radio" aria-checked={base === k} onClick={() => setBase(k)} className={`px-3 py-1.5 ${base === k ? "bg-amber/15 text-amber" : "text-slate-300 hover:bg-panel2"}`}>{BASES[k].label}</button>))}
        </div>
        <button className="rounded-md border border-line px-2 py-1.5 text-slate-300 hover:bg-panel2" onClick={() => map.current?.setView([-0.5, 37.9], 6)}>Whole Kenya</button>
        {a && <button className="rounded-md border border-line px-2 py-1.5 text-slate-300 hover:bg-panel2" onClick={() => map.current?.setView([a.lat, a.lon], 16)}>Zoom to selected fault</button>}
        <span className="ml-auto font-mono text-mute">{coord || "move over the map for coordinates"}</span>
      </div>
      <div className="relative h-[480px]">
        <div ref={box} style={{ background: "#0b1322" }} className="h-full w-full overflow-hidden rounded-xl border border-line" role="application" aria-label="Real-location map with street and satellite views" />
        {tileErr > 6 && <div role="alert" className="absolute left-3 top-3 z-[1000] max-w-xs rounded-lg bg-ink/90 px-3 py-2 text-xs text-amber">Base-map tiles could not load (offline, or the tile server is blocked on this network). The grid assets and faults are still shown; try the other base map.</div>}
      </div>
      <p className="text-xs text-mute">Zoom in to street level to identify the exact site. Click a marker to select the fault; use "Pick on map" in the report forms, then click here to mark the spot on real imagery. Network layout is illustrative (synthetic data).</p>
    </div>
  );
}
