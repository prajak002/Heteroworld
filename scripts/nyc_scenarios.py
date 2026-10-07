"""Summarise measured NYC sidewalk pinch points into representative benchmark geometries.

Keeps polygons whose pinch is single-file for two agents (``0.7 m <= pinch < 1.3 m``), then clusters
(typical width, pinch width, pinch length) with k-means (standardised features, fixed seed) and reports
each cluster centre with its share of the real pinch points. The centres become ``--sidewalk W w L``
settings for run_benchmark.py.

Usage:
    python scripts/nyc_scenarios.py data/nyc/pinch_points.csv --k 4 --out data/nyc/scenarios.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        label = ((x[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        new = np.stack([x[label == j].mean(0) if (label == j).any() else centres[j] for j in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    return centres, label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-length", type=float, default=12.0, help="cap on pinch length (m)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    total = len(rows)
    feats = np.array([[float(r["wide_m"]), float(r["pinch_m"]), min(float(r["pinch_len_m"]), args.max_length)]
                      for r in rows])
    keep = (feats[:, 1] >= 0.7) & (feats[:, 1] < 1.3) & (feats[:, 0] > feats[:, 1] + 0.3)
    x = feats[keep]
    mu, sd = x.mean(0), x.std(0) + 1e-9
    centres, label = kmeans((x - mu) / sd, args.k)
    centres = centres * sd + mu
    order = np.argsort(-np.bincount(label, minlength=args.k))
    scenarios = []
    for rank, j in enumerate(order):
        members = x[label == j]
        tree_share = float(np.mean([rows[i]["tree_caused"] == "True" for i in np.nonzero(keep)[0][label == j]]))
        scenarios.append({"name": f"nyc{rank + 1}", "wide": round(float(np.median(members[:, 0])), 2),
                          "pinch": round(float(np.median(members[:, 1])), 2),
                          "length": round(float(np.median(members[:, 2])), 2),
                          "share_of_single_file": round(len(members) / len(x), 3), "tree_caused": round(tree_share, 2),
                          "n": int(len(members))})
    by_area = {}
    for r, k in zip(rows, keep):
        by_area.setdefault(r["area"], [0, 0])
        by_area[r["area"]][0] += 1
        by_area[r["area"]][1] += int(k)
    report = {"polygons": total, "single_file": int(keep.sum()),
              "single_file_share_by_area": {a: round(v[1] / v[0], 3) for a, v in by_area.items()},
              "scenarios": scenarios}
    args.out.write_text(json.dumps(report, indent=1))
    print(f"{total} sidewalk polygons, {keep.sum()} ({100 * keep.mean():.0f}%) with a single-file pinch (0.7-1.3 m)")
    for a, v in by_area.items():
        print(f"  {a:16s} {v[1]}/{v[0]} ({100 * v[1] / v[0]:.0f}%)")
    for s in scenarios:
        print(f"  {s['name']}: wide {s['wide']} m, pinch {s['pinch']} m, length {s['length']} m "
              f"({100 * s['share_of_single_file']:.0f}% of single-file pinches, {100 * s['tree_caused']:.0f}% tree-caused)")


if __name__ == "__main__":
    main()
