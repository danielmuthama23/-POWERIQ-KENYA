"""Geographic helpers for the POWERIQ digital twin.

IMPORTANT: The Kenya outline below is a hand-simplified approximation (about 40 vertices)
used for illustrative visualisation only. It is NOT an official boundary dataset.
Regions are Voronoi-style groupings around representative centres, not administrative
county or utility-region boundaries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# (lon, lat) approximate outline, clockwise from the Lake Victoria shore.
KENYA_OUTLINE: list[tuple[float, float]] = [
    (34.00, -1.00), (35.20, -1.80), (36.50, -2.30), (37.65, -3.05), (37.75, -3.55),
    (39.20, -4.68), (39.45, -4.35), (39.80, -3.60), (40.10, -2.90), (40.60, -2.40),
    (41.10, -1.95), (41.55, -1.68), (41.00, -0.90), (40.95, 0.00), (41.00, 1.50),
    (41.00, 3.20), (41.90, 3.95), (41.00, 3.95), (40.10, 4.05), (39.50, 3.55),
    (39.05, 3.50), (38.10, 3.60), (37.50, 3.95), (36.90, 4.35), (36.00, 4.45),
    (35.30, 4.80), (34.60, 4.40), (34.20, 4.10), (34.40, 3.50), (34.00, 3.00),
    (34.10, 2.20), (34.40, 1.35), (34.15, 0.95), (33.95, 0.15), (34.05, -0.30),
    (34.00, -1.00),
]


def point_in_polygon(lon: float, lat: float, poly: list[tuple[float, float]] = KENYA_OUTLINE) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    lon: float
    lat: float
    base_peak_mw: float       # synthetic regional peak input demand
    residential: float        # 0..1 weight of evening residential peak
    commercial: float         # 0..1 weight of daytime commercial/industrial load
    temp_base_c: float
    wind_base_ms: float
    cloud_base: float


REGIONS: list[Region] = [
    Region("NAIROBI", "Nairobi Metro", 36.85, -1.30, 700, 0.9, 0.9, 19.0, 3.2, 0.45),
    Region("CENTRAL", "Central", 37.05, -0.45, 300, 0.8, 0.5, 18.0, 3.0, 0.50),
    Region("COAST", "Coast", 39.50, -3.20, 300, 0.7, 0.8, 27.5, 4.2, 0.40),
    Region("SOUTH_RIFT", "South Rift", 36.05, -0.95, 210, 0.7, 0.6, 19.5, 4.0, 0.45),
    Region("NORTH_RIFT", "North Rift", 35.40, 1.00, 140, 0.8, 0.5, 20.5, 3.4, 0.50),
    Region("WESTERN", "Western", 34.60, 0.50, 105, 0.9, 0.4, 22.5, 2.6, 0.60),
    Region("NYANZA", "Nyanza", 34.80, -0.70, 130, 0.9, 0.5, 23.0, 3.0, 0.55),
    Region("EASTERN", "Eastern", 37.90, -0.80, 135, 0.8, 0.4, 22.0, 3.3, 0.40),
    Region("NORTHERN", "Northern", 37.80, 2.80, 25, 0.9, 0.3, 28.5, 6.0, 0.25),
    Region("NORTH_EASTERN", "North Eastern", 40.00, 1.00, 28, 0.9, 0.3, 30.0, 4.5, 0.25),
]
REGION_IDS = [r.id for r in REGIONS]

# Demand centres used to distribute regional demand across hex cells (illustrative).
DEMAND_CENTRES: list[tuple[str, float, float, float, float]] = [
    # name, lon, lat, weight, sigma_deg
    ("Nairobi", 36.82, -1.29, 10.0, 0.22), ("Thika", 37.08, -1.04, 2.5, 0.15),
    ("Mombasa", 39.67, -4.04, 7.0, 0.22), ("Malindi", 40.12, -3.22, 1.0, 0.15),
    ("Kisumu", 34.76, -0.10, 3.0, 0.18), ("Kisii", 34.77, -0.68, 1.5, 0.14),
    ("Nakuru", 36.07, -0.30, 3.5, 0.20), ("Eldoret", 35.27, 0.52, 3.0, 0.18),
    ("Kakamega", 34.75, 0.28, 1.5, 0.15), ("Nyeri", 36.95, -0.42, 2.0, 0.15),
    ("Meru", 37.65, 0.05, 1.8, 0.15), ("Machakos", 37.26, -1.52, 2.2, 0.18),
    ("Embu", 37.45, -0.53, 1.2, 0.12), ("Garissa", 39.65, -0.45, 1.0, 0.15),
    ("Kitale", 35.00, 1.02, 1.2, 0.14), ("Narok", 35.87, -1.08, 0.8, 0.15),
    ("Lodwar", 35.60, 3.12, 0.5, 0.15), ("Marsabit", 37.99, 2.33, 0.4, 0.15),
    ("Mandera", 41.86, 3.94, 0.3, 0.10), ("Wajir", 40.06, 1.75, 0.3, 0.12),
]

CELL_RADIUS_DEG = 0.27  # hex circumradius in degrees (~30 km)


def _hex_polygon(cx: float, cy: float, r: float) -> list[list[float]]:
    pts = []
    for k in range(6):
        ang = math.radians(60 * k + 30)
        pts.append([round(cx + r * math.cos(ang), 5), round(cy + r * math.sin(ang), 5)])
    pts.append(pts[0])
    return pts


def build_cells() -> list[dict]:
    """Pointy-top hex grid clipped to the approximate Kenya outline."""
    r = CELL_RADIUS_DEG
    dx = math.sqrt(3) * r
    dy = 1.5 * r
    cells: list[dict] = []
    row = 0
    lat = -4.7
    while lat < 5.0:
        lon = 33.9 + (dx / 2 if row % 2 else 0.0)
        while lon < 42.0:
            if point_in_polygon(lon, lat):
                cells.append({"lon": round(lon, 4), "lat": round(lat, 4)})
            lon += dx
        lat += dy
        row += 1
    for i, c in enumerate(cells):
        c["cell_id"] = f"C{i:03d}"
        c["polygon"] = _hex_polygon(c["lon"], c["lat"], r)
        d = [haversine_km(c["lon"], c["lat"], rg.lon, rg.lat) for rg in REGIONS]
        c["region"] = REGIONS[int(np.argmin(d))].id
        w = 0.05  # small rural baseline
        for _, clon, clat, cw, sig in DEMAND_CENTRES:
            dd = math.hypot(c["lon"] - clon, c["lat"] - clat)
            w += cw * math.exp(-0.5 * (dd / sig) ** 2)
        c["weight"] = w
    # normalise weights within region
    totals: dict[str, float] = {}
    for c in cells:
        totals[c["region"]] = totals.get(c["region"], 0.0) + c["weight"]
    for c in cells:
        c["share"] = c["weight"] / totals[c["region"]]
    return cells


def nearest_index(lon: float, lat: float, points: list[tuple[float, float]]) -> int:
    d = [haversine_km(lon, lat, p[0], p[1]) for p in points]
    return int(np.argmin(d))
