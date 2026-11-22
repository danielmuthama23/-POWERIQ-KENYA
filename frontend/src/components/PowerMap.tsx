import { useEffect, useRef, useState } from "react";
import maplibregl, { Map as MLMap, Popup } from "maplibre-gl";
import { SOURCE_COLORS } from "./ui";

export type MapMeta = { hour: number; time: string; metrics: Record<string, string>; max: Record<string, number>; data_origin?: string; model_version?: string; geometry_note?: string; composite_formula?: string };
export type CellsFC = { type: "FeatureCollection"; features: any[]; meta: MapMeta };
export type Assets = { plants: any[]; substations: any[]; lines: any; outline: any; hour: number };

const BLANK_STYLE: maplibregl.StyleSpecification = {
  version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#070b14" } }],
};
const EMPTY: any = { type: "FeatureCollection", features: [] };

export const UNITS: Record<string, string> = {
  demand_mw: "MW", generation_mw: "MW", net_flow_mw: "MW", import_dependence_pct: "%", loss_pct: "%", ntl_pct: "%", tech_pct: "%",
  maintenance_risk: "%", renewable_mw: "MW", curtailed_mwh_30d: "MWh", line_loading_pct: "%", composite_risk: "/100",
};
export const SEQ_STOPS = ["#10243b", "#1d5d7a", "#6bb36b", "#f2b84b", "#e4572e"];

export function colorExpression(metric: string, max: number): any {
  const m = Math.max(max, 1e-9);
  if (metric === "net_flow_mw") return ["interpolate", ["linear"], ["get", metric], -m, "#d1495b", 0, "#26334d", m, "#2ec4b6"];
  const n = ["min", 1, ["/", ["abs", ["get", metric]], m]];
  return ["interpolate", ["linear"], n, 0, SEQ_STOPS[0], 0.2, SEQ_STOPS[1], 0.45, SEQ_STOPS[2], 0.7, SEQ_STOPS[3], 1, SEQ_STOPS[4]];
}

export function heightExpression(metric: string, max: number, maxHeight: number, sqrtScale: boolean): any {
  const n = ["min", 1, ["/", ["abs", ["get", metric]], Math.max(max, 1e-9)]];
  return ["+", 900, ["*", maxHeight, sqrtScale ? ["sqrt", n] : n]];
}

function squareAround(lon: number, lat: number, half: number) {
  return [[[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half], [lon - half, lat + half], [lon - half, lat - half]]];
}

export function plantsToFC(plants: any[]) {
  return {
    type: "FeatureCollection",
    features: plants.map(p => ({
      type: "Feature", properties: { ...p, color: SOURCE_COLORS[p.source] ?? "#fff", h: 20000 + 150000 * Math.sqrt(Math.max(p.dispatch_mw, 0) / 320) },
      geometry: { type: "Polygon", coordinates: squareAround(p.lon, p.lat, 0.06) },
    })),
  };
}

export function substationsToFC(subs: any[]) {
  return { type: "FeatureCollection", features: subs.map(s => ({ type: "Feature", properties: s, geometry: { type: "Polygon", coordinates: squareAround(s.lon, s.lat, 0.035) } })) };
}

export function popupHtml(p: any, meta: MapMeta, metric: string) {
  const rows = Object.entries(meta.metrics).map(([k, label]) => {
    const v = p[k]; const hl = k === metric;
    return `<tr style="${hl ? "color:#f2b84b;font-weight:600" : ""}"><td style="padding-right:12px;color:${hl ? "#f2b84b" : "#8696b3"}">${label}</td><td style="text-align:right;font-family:ui-monospace,monospace">${typeof v === "number" ? v.toLocaleString("en-US", { maximumFractionDigits: 1 }) : v}</td></tr>`;
  }).join("");
  return `<div style="font-weight:600;margin-bottom:4px">${p.cell_id} · ${String(p.region).replace("_", " ")} <span style="color:#8696b3;font-weight:400">bus ${p.bus_id}</span></div><table>${rows}</table>`;
}

type Props = {
  cells: CellsFC | null; assets: Assets | null; metric: string; sqrtScale: boolean; pitch3d: boolean;
  layers: { lines: boolean; plants: boolean; subs: boolean }; onReady?: () => void;
};

export default function PowerMap({ cells, assets, metric, sqrtScale, pitch3d, layers, onReady }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const popup = useRef<Popup | null>(null);
  const latest = useRef<{ cells: CellsFC | null; metric: string }>({ cells, metric });
  const [ready, setReady] = useState(false);
  const [webglError, setWebglError] = useState<string | null>(null);
  latest.current = { cells, metric };

  useEffect(() => {
    if (!box.current) return;
    let map: MLMap;
    try {
      const w = box.current.clientWidth || 800, h = box.current.clientHeight || 600;
      const zoom = Math.min(Math.log2((0.98 * w) / (8.9 * (512 / 360))), Math.log2((1.25 * h) / (9.6 * (512 / 360)))) - 0.12;
      map = new maplibregl.Map({ container: box.current, style: BLANK_STYLE, center: [37.95, -0.5], zoom, pitch: 58, bearing: -14, attributionControl: false, maxPitch: 75 });
    } catch (e) {
      setWebglError((e as Error).message || "WebGL is not available in this browser.");
      return;
    }
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    popup.current = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 12, maxWidth: "320px" });
    map.on("load", () => {
      map.addSource("outline", { type: "geojson", data: EMPTY });
      map.addSource("cells", { type: "geojson", data: EMPTY });
      map.addSource("lines", { type: "geojson", data: EMPTY });
      map.addSource("plants", { type: "geojson", data: EMPTY });
      map.addSource("subs", { type: "geojson", data: EMPTY });
      map.addLayer({ id: "outline", type: "line", source: "outline", paint: { "line-color": "#f2b84b", "line-width": 1.2, "line-opacity": 0.6 } });
      map.addLayer({ id: "cells-3d", type: "fill-extrusion", source: "cells", paint: { "fill-extrusion-color": "#1d5d7a", "fill-extrusion-height": 900, "fill-extrusion-opacity": 0.92, "fill-extrusion-vertical-gradient": true } });
      map.addLayer({ id: "lines", type: "line", source: "lines", layout: { "line-cap": "round" }, paint: {
        "line-color": ["interpolate", ["linear"], ["get", "loading_pct"], 0, "#3b82f6", 60, "#2ec4b6", 80, "#f2b84b", 100, "#ff3b30"],
        "line-width": ["interpolate", ["linear"], ["get", "capacity_mw"], 80, 1.5, 1000, 5], "line-opacity": 0.95 } });
      map.addLayer({ id: "subs-3d", type: "fill-extrusion", source: "subs", paint: { "fill-extrusion-color": "#ffffff", "fill-extrusion-height": 26000, "fill-extrusion-opacity": 0.9 } });
      map.addLayer({ id: "plants-3d", type: "fill-extrusion", source: "plants", paint: { "fill-extrusion-color": ["get", "color"], "fill-extrusion-height": ["get", "h"], "fill-extrusion-opacity": 0.95 } });
      map.on("mousemove", "cells-3d", e => {
        const f = e.features?.[0]; const c = latest.current.cells;
        if (!f || !c) return;
        map.getCanvas().style.cursor = "pointer";
        popup.current!.setLngLat(e.lngLat).setHTML(popupHtml(f.properties, c.meta, latest.current.metric)).addTo(map);
      });
      map.on("mouseleave", "cells-3d", () => { map.getCanvas().style.cursor = ""; popup.current?.remove(); });
      map.on("mousemove", "plants-3d", e => {
        const p = e.features?.[0]?.properties; if (!p) return;
        popup.current!.setLngLat(e.lngLat).setHTML(`<b>${p.name}</b><br/>${p.source} · ${p.capacity_mw} MW installed<br/>dispatch ${p.dispatch_mw} MW<br/>max unit risk ${(Number(p.max_unit_risk) * 100).toFixed(0)}%`).addTo(map);
      });
      map.on("mouseleave", "plants-3d", () => popup.current?.remove());
      setReady(true);
      onReady?.();
    });
    return () => { popup.current?.remove(); map.remove(); mapRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !cells) return;
    (map.getSource("cells") as maplibregl.GeoJSONSource).setData(cells as any);
    const mx = cells.meta.max[metric] ?? 1;
    map.setPaintProperty("cells-3d", "fill-extrusion-color", colorExpression(metric, mx));
    map.setPaintProperty("cells-3d", "fill-extrusion-height", heightExpression(metric, mx, 115000, sqrtScale));
  }, [cells, metric, sqrtScale, ready]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !assets) return;
    (map.getSource("outline") as maplibregl.GeoJSONSource).setData(assets.outline);
    (map.getSource("lines") as maplibregl.GeoJSONSource).setData(assets.lines);
    (map.getSource("plants") as maplibregl.GeoJSONSource).setData(plantsToFC(assets.plants) as any);
    (map.getSource("subs") as maplibregl.GeoJSONSource).setData(substationsToFC(assets.substations) as any);
  }, [assets, ready]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const vis = (id: string, on: boolean) => map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
    vis("lines", layers.lines); vis("plants-3d", layers.plants); vis("subs-3d", layers.subs);
  }, [layers, ready]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    map.easeTo({ pitch: pitch3d ? 58 : 0, bearing: pitch3d ? -14 : 0, duration: 700 });
  }, [pitch3d, ready]);

  if (webglError) {
    return <div role="alert" className="flex h-full items-center justify-center rounded-xl border border-danger/50 bg-danger/10 p-6 text-center text-sm">
      The 3D map needs WebGL, which is unavailable here ({webglError}). The tables and charts on this page still show the same data.</div>;
  }
  return <div ref={box} className="h-full w-full overflow-hidden rounded-xl border border-line" role="img" aria-label="3D extruded map of Kenya electricity metrics by area" />;
}
