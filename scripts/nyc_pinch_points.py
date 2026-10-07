"""Measure real sidewalk pinch points in New York City from public records.

Data (NYC Open Data, free to reuse under the NYC Open Data Terms of Use):
* NYC Planimetric Database: Sidewalk (``52n9-sdep``), curb-to-property-line sidewalk polygons;
* 2015 Street Tree Census (``uvpi-gqnh``), street-tree locations.

For each sidewalk polygon in the study areas, the walkable area is rasterised at ``--res`` metres,
standard tree pits (1.2 m across x 3.0 m along the nearest curb edge) are removed at every live tree,
and the Euclidean distance transform gives the local clear width (twice the distance to the nearest
obstacle) along the ridge of the walkway. Per polygon we report the typical width (median ridge
width), the pinch width (5th percentile) and the pinch length (extent of the ridge where the width is
within 0.25 m of the pinch width), and whether a tree pit causes the pinch.

Usage:
    python scripts/nyc_pinch_points.py --out data/nyc/pinch_points.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage
from shapely import contains_xy
from shapely.geometry import Polygon, shape
from shapely.ops import transform

# name: (north lat, west lon, south lat, east lon)
AREAS = {
    "chinatown_les": (40.7220, -74.0000, 40.7110, -73.9850),
    "midtown": (40.7600, -73.9900, 40.7500, -73.9750),
    "williamsburg": (40.7160, -73.9650, 40.7050, -73.9500),
    "jackson_heights": (40.7560, -73.8950, 40.7450, -73.8780),
}
BASE = "https://data.cityofnewyork.us/resource"
PIT_ACROSS, PIT_ALONG = 1.2, 3.0


def fetch(dataset: str, where: str, cache: Path) -> list[dict]:
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"{BASE}/{dataset}.json?" + urllib.parse.urlencode({"$where": where, "$limit": 100000})
    with urllib.request.urlopen(url, timeout=300) as r:
        rows = json.loads(r.read())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


def projector(lat0: float, lon0: float):
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    return lambda lon, lat, z=None: ((np.asarray(lon) - lon0) * kx, (np.asarray(lat) - lat0) * ky)


def tree_pit(poly: Polygon, x: float, y: float) -> Polygon | None:
    """Rectangle aligned with the polygon edge nearest to the tree."""
    ring = np.asarray(poly.exterior.coords)
    seg = ring[1:] - ring[:-1]
    rel = np.array([x, y]) - ring[:-1]
    t = np.clip((rel * seg).sum(1) / np.maximum((seg ** 2).sum(1), 1e-9), 0, 1)
    d = np.linalg.norm(rel - t[:, None] * seg, axis=1)
    k = int(d.argmin())
    if d[k] > 3.0:
        return None
    u = seg[k] / max(np.linalg.norm(seg[k]), 1e-9)
    v = np.array([-u[1], u[0]])
    c = np.array([x, y])
    pts = [c + a * u * PIT_ALONG / 2 + b * v * PIT_ACROSS / 2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return Polygon(pts)


def measure(poly: Polygon, pits: list[Polygon], res: float) -> dict | None:
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx + res, res)
    ys = np.arange(miny, maxy + res, res)
    if len(xs) * len(ys) > 8e6:
        return None
    gx, gy = np.meshgrid(xs, ys)
    free = contains_xy(poly, gx, gy)
    blocked = np.zeros_like(free)
    for p in pits:
        blocked |= contains_xy(p, gx, gy)
    walk = free & ~blocked
    if walk.sum() < 50:
        return None
    edt = ndimage.distance_transform_edt(walk) * res
    ridge = walk & (edt >= ndimage.maximum_filter(edt, size=3)) & (edt > 0.3)
    if ridge.sum() < 20:
        return None
    width = 2 * edt[ridge]
    wide = float(np.median(width))
    narrow = float(np.percentile(width, 5))
    # Pinch length: the contiguous narrow stretch around the narrowest ridge point (not every narrow
    # spot of the block, which would merge separate tree pits).
    narrow_mask = np.zeros_like(walk)
    narrow_mask[ridge] = width <= narrow + 0.25
    labels, _ = ndimage.label(ndimage.binary_dilation(narrow_mask, iterations=2))
    ridge_idx = np.argwhere(ridge)
    seed = ridge_idx[np.argmin(np.abs(width - narrow))]
    comp = (labels == labels[tuple(seed)]) & narrow_mask
    pts = np.stack([gx[comp], gy[comp]], 1)
    if len(pts) > 1:
        centred = pts - pts.mean(0)
        axis = np.linalg.svd(centred, full_matrices=False)[2][0]
        proj = centred @ axis
        length = float(proj.max() - proj.min()) + res
    else:
        length = res
    # Does a tree pit cause the pinch? Compare with the pinch width without pits.
    edt0 = ndimage.distance_transform_edt(free) * res
    ridge0 = free & (edt0 >= ndimage.maximum_filter(edt0, size=3)) & (edt0 > 0.3)
    narrow0 = float(np.percentile(2 * edt0[ridge0], 5)) if ridge0.sum() >= 20 else narrow
    return {"wide_m": round(wide, 2), "pinch_m": round(narrow, 2), "pinch_len_m": round(length, 2),
            "pinch_no_trees_m": round(narrow0, 2), "tree_caused": bool(narrow < narrow0 - 0.2),
            "area_m2": round(float(walk.sum()) * res * res, 1), "trees": len(pits)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--areas", nargs="+", default=list(AREAS))
    parser.add_argument("--res", type=float, default=0.1)
    parser.add_argument("--cache", type=Path, default=Path("data/nyc/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/nyc/pinch_points.csv"))
    args = parser.parse_args()
    rows_out = []
    for area in args.areas:
        n, w, s, e = AREAS[area]
        sidewalks = fetch("52n9-sdep", f"within_box(the_geom, {n}, {w}, {s}, {e})", args.cache / f"{area}_sidewalk.json")
        trees = fetch("uvpi-gqnh", f"latitude between {s} and {n} and longitude between {w} and {e} and status = 'Alive'",
                      args.cache / f"{area}_trees.json")
        proj = projector((n + s) / 2, (w + e) / 2)
        tx, ty = proj([float(t["longitude"]) for t in trees], [float(t["latitude"]) for t in trees])
        kept = 0
        for row in sidewalks:
            geom = transform(proj, shape(row["the_geom"]))
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                if poly.area < 30:
                    continue
                near = contains_xy(poly.buffer(1.0), tx, ty)
                pits = [p for p in (tree_pit(poly, x, y) for x, y in zip(tx[near], ty[near])) if p is not None]
                m = measure(poly, pits, args.res)
                if m is None:
                    continue
                m.update({"area": area, "source_id": row.get("source_id", "")})
                rows_out.append(m)
                kept += 1
        print(f"{area:16s} {len(sidewalks):5d} sidewalk rows, {len(trees):5d} trees -> {kept} measured polygons", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    widths = np.array([r["pinch_m"] for r in rows_out])
    print(f"{len(rows_out)} polygons; pinch width quantiles (m) 5/25/50/75/95%: "
          f"{np.percentile(widths, [5, 25, 50, 75, 95]).round(2).tolist()}; "
          f"single-file (<1.3 m): {100 * (widths < 1.3).mean():.0f}%; tree-caused: "
          f"{100 * np.mean([r['tree_caused'] for r in rows_out]):.0f}%")


if __name__ == "__main__":
    main()
