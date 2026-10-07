"""Data for the WebGPU street viewer (docs/hw_viewer/index.html).

Writes docs/hw_viewer/data/scene.json (the real Two Bridges block around the measured pinch, the recorded A/B episodes,
road edges estimated from the sidewalk polygons) and docs/hw_viewer/data/results.json (final campaign statistics from
the analyze.py stats.json files and the ETH/UCY forecast study). Copies the baked SOMA / G1 actors from twin/assets.

Usage: .venv/bin/python docs/hw_viewer/build_data.py
"""
from __future__ import annotations

import json
import shutil
import statistics as st
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, Polygon

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data"
METHODS = ["hw_robust", "union_select", "mixture_robust", "cv_robust", "oracle_robust"]
SETTINGS = [
    ("v2/main/h4", "Corridor, 4 pedestrians", "synthetic"),
    ("v2/dense/h6a100", "Dense: 6 assertive pedestrians", "synthetic"),
    ("v2/faulty/f2", "2 faulty robots", "synthetic"),
    ("v2/nyc/nyc1", "NYC type 1 (40% of pinches)", "synthetic"),
    ("v2/nyc/nyc2", "NYC type 2 (32%)", "synthetic"),
    ("v2/nyc/nyc3", "NYC type 3 (25%)", "synthetic"),
    ("v2/nyc/nyc4", "NYC type 4 (3%, 12 m pinch)", "synthetic"),
    ("v2_learned/main/h4", "Learned, 4 pedestrians", "learned"),
    ("v2_learned/faulty/f2", "Learned, 2 faulty robots", "learned"),
]


def r2(v):
    return round(float(v), 2)


def scene():
    S = json.loads((ROOT / "docs/nyc_viewer/scene.json").read_text())
    R = json.loads((ROOT / "twin/data/replay.json").read_text())
    # the sidewalk under the pinch and the one across the street, to place the kerbs and the carriageway
    polys = [Polygon(p).buffer(0) for p in S["sidewalks"]]
    here = max((p for p in polys if p.contains(Point(0, 0)) or p.distance(Point(0, 0)) < 0.5), key=lambda p: p.area)
    probe = LineString([(0, -60), (0, 60)])
    cut = here.intersection(probe)
    ys = [c[1] for g in getattr(cut, "geoms", [cut]) for c in g.coords]
    near_kerb = min(ys)
    far = []
    for p in polys:
        if p is here:
            continue
        c = p.intersection(probe)
        if c.is_empty:
            continue
        cy = [q[1] for g in getattr(c, "geoms", [c]) for q in g.coords]
        if max(cy) < near_kerb - 4:
            far.append(max(cy))
    far_kerb = max(far) if far else near_kerb - 10.5
    eps = {}
    for m, ep in R["episodes"].items():
        eps[m] = dict(method=m, outcome=ep["outcome"], is_human=ep["is_human"],
                      frames=[dict(p=[[r2(a), r2(b)] for a, b in f["p"]], h=[r2(a) for a in f["h"]], d=f["d"]) for f in ep["frames"]],
                      plans=[dict(t=pl["t"], xy=[[[r2(a), r2(b)] for a, b in row] for row in pl["xy"]],
                                  forecast=[[[[r2(a), r2(b)] for a, b in row] for row in atom] for atom in pl["forecast"]])
                             for pl in ep["plans"]])
    return dict(
        source=S["source"], lat=S["lat"], lon=S["lon"], pinch=S["pinch"],
        buildings=[dict(h=b["h"], pts=[[r2(x), r2(y)] for x, y in b["pts"]]) for b in S["buildings"]],
        sidewalks=[[[r2(x), r2(y)] for x, y in s] for s in S["sidewalks"]],
        trees=[[r2(x), r2(y)] for x, y in S["trees"]],
        pits=[[[r2(x), r2(y)] for x, y in p] for p in S["pits"]],
        kerb_near=r2(near_kerb), kerb_far=r2(far_kerb),
        dt=R["dt"], agent_radius=R["agent_radius"], corridor_half_width=R["corridor_half_width"], walls=R["walls"],
        stats=R["stats"], seed=R["seed"], episodes=eps,
    )


def results():
    out = []
    for path, label, kind in SETTINGS:
        d = json.loads((ROOT / "results" / path / "stats.json").read_text())
        row = dict(id=path, label=label, kind=kind, methods={}, tests={})
        for m in METHODS:
            if m not in d["methods"]:
                continue
            md = d["methods"][m]
            row["methods"][m] = dict(n=md["episodes"], **{k: dict(mean=r2(md[k]["mean"]), ci=[r2(c) for c in md[k]["ci95"]])
                                                          for k in ("completed", "collision_robot_human", "collision", "deadlock")})
        for metric in ("completed", "collision_robot_human", "collision"):
            row["tests"][metric] = {m: dict(diff=r2(t["diff_pct"]), ci=[r2(c) for c in t["diff_ci95"]], p=float(f"{t['p_holm']:.3g}"))
                                    for m, t in d["tests"].get(metric, {}).items() if m in METHODS}
        out.append(row)
    E = json.loads((ROOT / "results/ethucy/kaggle_v2_run2/ethucy_results.json").read_text())["results"]
    models = ["barycenter", "rssm", "mixture", "cv", "jepa", "diffusion"]
    eth = {m: dict(energy=r2(st.mean(E[s][m]["energy_all"] for s in E)), energy_sd=r2(st.stdev(E[s][m]["energy_all"] for s in E)),
                   ade=r2(st.mean(E[s][m]["ADEmean_target"] for s in E))) for m in models}
    wins = sum(E[s]["barycenter"]["energy_all"] < E[s]["mixture"]["energy_all"] for s in E)
    return dict(settings=out, ethucy=dict(models=eth, runs=len(E), bary_beats_mixture=wins, wilcoxon_p=1e-4))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scene.json").write_text(json.dumps(scene(), separators=(",", ":")))
    (OUT / "results.json").write_text(json.dumps(results(), separators=(",", ":")))
    # binaries travel as base64 inside JSON (the artifact host serves only standard web types)
    import base64
    for f in ("soma.json", "g1.json"):
        shutil.copy(ROOT / "twin/assets" / f, OUT / f)
    for f in ("soma", "g1"):
        (OUT / f"{f}.bin.json").write_text(json.dumps({"b64": base64.b64encode((ROOT / "twin/assets" / f"{f}.bin").read_bytes()).decode()}))
        (OUT / f"{f}.bin").unlink(missing_ok=True)
    tree = OUT / "tree"
    if (tree / "scene.bin").exists():
        g = json.loads((tree / "scene.gltf").read_text())
        g["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode((tree / "scene.bin").read_bytes()).decode()
        (tree / "scene.gltf").write_text(json.dumps(g))
        (tree / "scene.bin").unlink()
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:14s} {f.stat().st_size / 1e6:6.2f} MB")
    sc = json.loads((OUT / "scene.json").read_text())
    print("kerbs", sc["kerb_near"], sc["kerb_far"])
