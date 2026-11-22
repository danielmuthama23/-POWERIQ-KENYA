import { useEffect, useRef, useState } from "react";
import maplibregl, { Map as MLMap } from "maplibre-gl";

export const SEV_COLOR: Record<string, string> = { critical: "#ff3b30", high: "#ff8a1f", medium: "#f2b84b", low: "#5b9dff" };
const EMPTY: any = { type: "FeatureCollection", features: [] };
const BLANK: maplibregl.StyleSpecification = { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#070b14" } }] };

export type FaultAlert = {
  id: string; rank: number; priority: number; severity: string; status: string; kind: string; asset_type: string; asset_id: string; name: string;
  lat: number; lon: number; geometry: number[][] | null; title: string; detail: string; recommended_action: string; source: string; mw_at_risk: number;
  region: string; nearest_asset: { type: string; name: string; km: number }; respond_within_h: number; started?: string | null; hours_until?: number | null; ack?: { by: string; ts: string; note: string } | null; community_id?: string; detection_id?: string; confidence?: number; corroborating_reports?: number; report_id?: string;
};
export type FaultAssets = { plants: any[]; substations: any[]; lines: any[]; outline: number[][]; cells: { id: string; poly: number[][] }[] };

const pt = (lon: number, lat: number, props: any) => ({ type: "Feature", properties: props, geometry: { type: "Point", coordinates: [lon, lat] } });

export function alertsToFC(alerts: FaultAlert[], selected: string | null) {
  return {
    type: "FeatureCollection",
    features: alerts.map(a => pt(a.lon, a.lat, {
      id: a.id, sev: a.severity, color: SEV_COLOR[a.severity], status: a.status, live: a.status === "predicted" || a.status === "resolved" || a.status === "unverified" ? 0 : 1,
      sel: a.id === selected ? 1 : 0, rank: a.rank, r: 7 + Math.min(9, Math.sqrt(a.mw_at_risk) / 2),
    })),
  };
}
export function alertLinesFC(alerts: FaultAlert[]) {
  return { type: "FeatureCollection", features: alerts.filter(a => a.geometry).map(a => ({ type: "Feature", properties: { id: a.id, color: SEV_COLOR[a.severity], predicted: a.status === "predicted" ? 1 : 0 }, geometry: { type: "LineString", coordinates: a.geometry } })) };
}

type Props = { assets: FaultAssets | null; alerts: FaultAlert[]; selected: string | null; onSelect: (id: string | null) => void; showAssets: boolean; picking?: boolean; onPick?: (lat: number, lon: number) => void; pin?: [number, number] | null };

export default function FaultMap({ assets, alerts, selected, onSelect, showAssets, picking, onPick, pin }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const [ready, setReady] = useState(false);
  const [webglError, setWebglError] = useState<string | null>(null);
  const cb = useRef({ onSelect, picking, onPick }); cb.current = { onSelect, picking, onPick };

  useEffect(() => {
    if (!box.current) return;
    let map: MLMap;
    try { map = new maplibregl.Map({ container: box.current, style: BLANK, center: [37.9, 0.2], zoom: 5.1, attributionControl: false }); }
    catch (e) { setWebglError((e as Error).message || "WebGL is not available in this browser."); return; }
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    let raf = 0;
    map.on("load", () => {
      for (const s of ["outline", "cells", "lines", "plants", "subs", "alines", "alerts", "pin"]) map.addSource(s, { type: "geojson", data: EMPTY });
      map.addLayer({ id: "cells", type: "fill", source: "cells", paint: { "fill-color": "#13233a", "fill-opacity": 0.55, "fill-outline-color": "#1c2a42" } });
      map.addLayer({ id: "outline", type: "line", source: "outline", paint: { "line-color": "#f2b84b", "line-width": 1.2, "line-opacity": 0.6 } });
      map.addLayer({ id: "lines", type: "line", source: "lines", paint: { "line-color": "#4b6a99", "line-width": 1.4, "line-opacity": 0.8 } });
      map.addLayer({ id: "subs", type: "circle", source: "subs", paint: { "circle-radius": 3.5, "circle-color": "#e8eefc", "circle-stroke-color": "#0b1322", "circle-stroke-width": 1 } });
      map.addLayer({ id: "plants", type: "circle", source: "plants", paint: { "circle-radius": 5, "circle-color": "#2ec4b6", "circle-stroke-color": "#0b1322", "circle-stroke-width": 1 } });
      map.addLayer({ id: "alines-glow", type: "line", source: "alines", paint: { "line-color": ["get", "color"], "line-width": 9, "line-opacity": 0.22, "line-blur": 3 } });
      map.addLayer({ id: "alines", type: "line", source: "alines", paint: { "line-color": ["get", "color"], "line-width": 3.5, "line-dasharray": [1, 0] } });
      map.addLayer({ id: "halo", type: "circle", source: "alerts", filter: ["==", ["get", "live"], 1], paint: { "circle-radius": 18, "circle-color": ["get", "color"], "circle-opacity": 0.3, "circle-blur": 0.6 } });
      map.addLayer({ id: "alerts", type: "circle", source: "alerts", paint: {
        "circle-radius": ["get", "r"], "circle-color": ["case", ["==", ["get", "live"], 1], ["get", "color"], "rgba(0,0,0,0)"],
        "circle-stroke-color": ["get", "color"], "circle-stroke-width": ["case", ["==", ["get", "sel"], 1], 4, 2.5] } });
      map.addLayer({ id: "pin", type: "circle", source: "pin", paint: { "circle-radius": 7, "circle-color": "#fff", "circle-stroke-color": "#f2b84b", "circle-stroke-width": 3 } });
      const t0 = performance.now();
      const pulse = (t: number) => {
        const k = ((t - t0) % 1600) / 1600;
        if (map.getLayer("halo")) { map.setPaintProperty("halo", "circle-radius", 12 + 26 * k); map.setPaintProperty("halo", "circle-opacity", 0.38 * (1 - k)); }
        raf = requestAnimationFrame(pulse);
      };
      raf = requestAnimationFrame(pulse);
      map.on("click", e => {
        if (cb.current.picking) { cb.current.onPick?.(+e.lngLat.lat.toFixed(4), +e.lngLat.lng.toFixed(4)); return; }
        const f = map.queryRenderedFeatures([[e.point.x - 10, e.point.y - 10], [e.point.x + 10, e.point.y + 10]], { layers: ["alerts"] });
        cb.current.onSelect(f.length ? String(f[0].properties!.id) : null);
      });
      map.on("mousemove", e => { map.getCanvas().style.cursor = cb.current.picking ? "crosshair" : map.queryRenderedFeatures([[e.point.x - 8, e.point.y - 8], [e.point.x + 8, e.point.y + 8]], { layers: ["alerts"] }).length ? "pointer" : ""; });
      setReady(true);
    });
    return () => { cancelAnimationFrame(raf); map.remove(); mapRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapRef.current; if (!map || !ready || !assets) return;
    const src = (n: string, d: any) => (map.getSource(n) as maplibregl.GeoJSONSource).setData(d);
    src("outline", { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: assets.outline.concat([assets.outline[0]]) } });
    src("cells", { type: "FeatureCollection", features: assets.cells.map(c => ({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [c.poly] } })) });
    src("lines", { type: "FeatureCollection", features: assets.lines.map(l => ({ type: "Feature", properties: { id: l.id }, geometry: { type: "LineString", coordinates: l.coords } })) });
    src("plants", { type: "FeatureCollection", features: assets.plants.map(p => pt(p.lon, p.lat, p)) });
    src("subs", { type: "FeatureCollection", features: assets.substations.map(p => pt(p.lon, p.lat, p)) });
  }, [assets, ready]);

  useEffect(() => {
    const map = mapRef.current; if (!map || !ready) return;
    (map.getSource("alerts") as maplibregl.GeoJSONSource).setData(alertsToFC(alerts, selected) as any);
    (map.getSource("alines") as maplibregl.GeoJSONSource).setData(alertLinesFC(alerts) as any);
  }, [alerts, selected, ready]);

  useEffect(() => {
    const map = mapRef.current; if (!map || !ready) return;
    for (const id of ["cells", "lines", "subs", "plants"]) map.setLayoutProperty(id, "visibility", showAssets ? "visible" : "none");
  }, [showAssets, ready]);

  useEffect(() => {
    const map = mapRef.current; if (!map || !ready) return;
    (map.getSource("pin") as maplibregl.GeoJSONSource).setData(pin ? (pt(pin[1], pin[0], {}) as any) : EMPTY);
  }, [pin, ready]);

  useEffect(() => {
    const map = mapRef.current; if (!map || !ready || !selected) return;
    const a = alerts.find(x => x.id === selected);
    if (a) map.easeTo({ center: [a.lon, a.lat], zoom: Math.max(map.getZoom(), 6.4), duration: 600 });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, ready]);

  if (webglError) return <div role="alert" className="flex h-full items-center justify-center rounded-xl border border-danger/50 bg-danger/10 p-6 text-center text-sm">The map needs WebGL, which is not available here ({webglError}). The priority list still works.</div>;
  return <div ref={box} className="h-full w-full overflow-hidden rounded-xl border border-line" role="img" aria-label="Map of Kenya showing faulty power stations, substations and lines" />;
}
