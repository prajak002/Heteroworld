"""Build a real Manhattan block scene around one measured sidewalk pinch point (for the 3D viewer).

Picks a measured single-file pinch in the Chinatown/Lower East Side study area, locates its narrowest
point and walking direction, and exports the surroundings within ``--radius`` metres in a local frame
(origin at the pinch, +x along the sidewalk, +y towards the building line):

* building footprints with roof heights (NYC Open Data ``5zhs-2jue``, ``height_roof`` in feet);
* sidewalk polygons (``52n9-sdep``) and street trees (``uvpi-gqnh``), from the pinch-point cache.

It also writes the pinch's own measured geometry so episodes can be simulated for exactly this place
with ``run_benchmark.py/record_scenes.py --sidewalk WIDE PINCH LENGTH``.

Usage:
    python scripts/nyc_scene.py --out docs/nyc_viewer/scene.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from shapely import contains_xy
from shapely.geometry import Point, shape
from shapely.ops import transform

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nyc_pinch_points import AREAS, fetch, projector, tree_pit  # noqa: E402

FT = 0.3048


def pinch_location(poly, pits, res=0.1):
    """Narrowest clear-path point, its clear width, and the walking direction (unit vector)."""
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx + res, res)
    ys = np.arange(miny, maxy + res, res)
    gx, gy = np.meshgrid(xs, ys)
    walk = contains_xy(poly, gx, gy)
    for p in pits:
        walk &= ~contains_xy(p, gx, gy)
    edt = ndimage.distance_transform_edt(walk) * res
    ridge = walk & (edt >= ndimage.maximum_filter(edt, size=3)) & (edt > 0.3)
    width = 2 * edt[ridge]
    pts = np.stack([gx[ridge], gy[ridge]], 1)
    k = int(np.argmin(np.abs(width - np.percentile(width, 5))))
    centre = pts[k]
    near = pts[np.linalg.norm(pts - centre, axis=1) < 8.0]
    axis = np.linalg.svd(near - near.mean(0), full_matrices=False)[2][0]
    return centre, float(width[k]), axis / np.linalg.norm(axis)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="chinatown_les")
    parser.add_argument("--radius", type=float, default=110.0)
    parser.add_argument("--csv", type=Path, default=Path("data/nyc/pinch_points.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/nyc/raw"))
    parser.add_argument("--out", type=Path, default=Path("docs/nyc_viewer/scene.json"))
    args = parser.parse_args()
    import csv
    rows = [r for r in csv.DictReader(open(args.csv)) if r["area"] == args.area]
    # A typical single-file, tree-caused pinch on a block sidewalk of ordinary width.
    cands = [r for r in rows if 0.8 <= float(r["pinch_m"]) <= 1.2 and 2.6 <= float(r["wide_m"]) <= 3.6
             and r["tree_caused"] == "True" and 2.0 <= float(r["pinch_len_m"]) <= 4.0]
    cands.sort(key=lambda r: abs(float(r["pinch_m"]) - 1.0) + 0.2 * abs(float(r["wide_m"]) - 3.0))
    chosen = cands[0]
    n, w, s, e = AREAS[args.area]
    proj = projector((n + s) / 2, (w + e) / 2)
    sidewalks = fetch("52n9-sdep", "", args.cache / f"{args.area}_sidewalk.json")
    trees = fetch("uvpi-gqnh", "", args.cache / f"{args.area}_trees.json")
    tx, ty = proj([float(t["longitude"]) for t in trees], [float(t["latitude"]) for t in trees])
    polys = []
    for row in sidewalks:
        geom = transform(proj, shape(row["the_geom"]))
        for poly in (list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]):
            polys.append((row.get("source_id", ""), poly))
    target = [p for sid, p in polys if sid == chosen["source_id"] and abs(p.area - float(chosen["area_m2"])) / p.area < 0.3]
    poly = max(target or [p for sid, p in polys if sid == chosen["source_id"]], key=lambda p: p.area)
    near = contains_xy(poly.buffer(1.0), tx, ty)
    pits = [p for p in (tree_pit(poly, x, y) for x, y in zip(tx[near], ty[near])) if p is not None]
    centre, pinch_w, axis = pinch_location(poly, pits)
    # Local frame: +x along the sidewalk, +y towards the nearest buildings, origin on the sidewalk's
    # centreline at the pinch (so the simulated walkway, which has the measured full width, lines up
    # with the real sidewalk outside the pinch).
    normal = np.array([-axis[1], axis[0]])
    probe = np.array([centre + t * normal for t in np.arange(-6, 6.01, 0.02)])
    inside = contains_xy(poly, probe[:, 0], probe[:, 1])
    ts = np.arange(-6, 6.01, 0.02)[inside]
    offset = 0.5 * (ts.min() + ts.max()) if len(ts) else 0.0
    origin = centre + offset * normal
    _pending_origin = origin
    lat0, lon0 = (n + s) / 2, (w + e) / 2
    kx = 111320.0 * math.cos(math.radians(lat0))
    clat = lat0 + origin[1] / 110540.0
    clon = lon0 + origin[0] / kx
    buildings = fetch("5zhs-2jue", f"within_circle(the_geom, {clat}, {clon}, {args.radius + 40})",
                      args.cache / f"scene_{args.area}_{chosen['source_id']}_buildings.json")
    bgeoms = [transform(proj, shape(b["the_geom"])) for b in buildings]
    nearest = min(bgeoms, key=lambda g: Point(origin).distance(g))
    if np.dot(np.array(nearest.centroid.coords[0]) - origin, normal) < 0:
        normal = -normal
    to_local = lambda x, y: (float(np.dot([x - origin[0], y - origin[1]], axis)),
                             float(np.dot([x - origin[0], y - origin[1]], normal)))
    out_b = []
    for b in buildings:
        geom = transform(proj, shape(b["the_geom"]))
        h = float(b.get("height_roof") or 30.0) * FT
        for p in (list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]):
            if Point(origin).distance(p) > args.radius:
                continue
            out_b.append({"h": round(h, 1), "pts": [list(map(lambda v: round(v, 2), to_local(x, y))) for x, y in p.exterior.coords]})
    out_s = []
    for _, p in polys:
        if Point(origin).distance(p) > args.radius:
            continue
        out_s.append([list(map(lambda v: round(v, 2), to_local(x, y))) for x, y in p.exterior.coords])
    out_t = [list(map(lambda v: round(v, 2), to_local(x, y))) for x, y in zip(tx, ty)
             if math.hypot(x - origin[0], y - origin[1]) < args.radius]
    out_p = [[list(map(lambda v: round(v, 2), to_local(x, y))) for x, y in p.exterior.coords] for p in pits
             if Point(origin).distance(p) < args.radius]
    scene = {"source": "NYC Open Data: Building Footprints (5zhs-2jue), Planimetric Sidewalk (52n9-sdep), 2015 Street Tree Census (uvpi-gqnh)",
             "area": args.area, "sidewalk_source_id": chosen["source_id"], "lat": round(clat, 6), "lon": round(clon, 6),
             # Local frame in east/north metres: world_en = origin + x * axis_en + y * normal_en.
             "axis_en": [round(float(axis[0]), 6), round(float(axis[1]), 6)],
             "normal_en": [round(float(normal[0]), 6), round(float(normal[1]), 6)],
             "pinch_clear_path_y": round(float(-offset if np.dot(normal, normal) else 0), 2),
             "pinch": {"wide": float(chosen["wide_m"]), "pinch": float(chosen["pinch_m"]), "length": float(chosen["pinch_len_m"])},
             "buildings": out_b, "sidewalks": out_s, "trees": out_t, "pits": out_p}
    args.out.write_text(json.dumps(scene, separators=(",", ":")))
    print(f"pinch at {clat:.6f}, {clon:.6f}: wide {chosen['wide_m']} m, pinch {chosen['pinch_m']} m, length {chosen['pinch_len_m']} m")
    print(f"{len(out_b)} buildings, {len(out_s)} sidewalk polygons, {len(out_t)} trees, {len(out_p)} pits -> {args.out} "
          f"({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
