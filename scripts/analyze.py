"""Multi-seed statistics for one experiment: per-seed rates, across-seed mean +- std, pooled 95% CIs and
paired significance tests of a reference method against every other method.

Layout: ``<root>/seed<k>/<method>.json`` (one file per method and run seed, as written by run_benchmark.py).
Episodes are paired by their scenario seed, so every test compares methods on identical scenarios.

Tests (binary metrics): exact two-sided McNemar test on the discordant pairs, Holm-corrected over the
compared methods; we also report a paired bootstrap 95% CI of the rate difference (reference - other).

Usage:
    python scripts/analyze.py results/v2_main --ref hw_robust --out results/v2_main/stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hetero_world.benchmark import wilson_interval  # noqa: E402

BINARY = ("collision_robot_human", "collision", "deadlock", "completed", "collision_robot_robot", "collision_wall")
CONTINUOUS = ("makespan_s", "min_separation", "bandwidth_kBps")


def load(root: Path) -> dict[str, dict[int, list[dict]]]:
    """``{method: {seed: episodes}}``."""
    out: dict[str, dict[int, list[dict]]] = {}
    for seed_dir in sorted(root.glob("seed*")):
        seed = int(seed_dir.name[4:])
        for f in sorted(seed_dir.glob("*.json")):
            payload = json.loads(f.read_text())
            if "episodes" not in payload:
                continue
            # Shards of one method (``<method>_part<k>.json``) are pooled; episodes are unique by scenario seed.
            pool = out.setdefault(payload["method"], {}).setdefault(seed, {})
            for episode in payload["episodes"]:
                pool[episode["seed"]] = episode
    return {m: {s: [eps[k] for k in sorted(eps)] for s, eps in seeds.items()} for m, seeds in out.items()}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def paired(ref: dict[int, list[dict]], other: dict[int, list[dict]], key: str, boot: int = 2000):
    """Paired episodes of two methods over the seeds both have; returns arrays and stats."""
    xs, ys = [], []
    for seed in sorted(set(ref) & set(other)):
        a = {e["seed"]: e for e in ref[seed]}
        b = {e["seed"]: e for e in other[seed]}
        for s in sorted(set(a) & set(b)):
            xs.append(float(a[s][key]))
            ys.append(float(b[s][key]))
    n = len(xs)
    if n == 0:
        return None
    b_cnt = sum(1 for x, y in zip(xs, ys) if x > 0.5 and y <= 0.5)
    c_cnt = sum(1 for x, y in zip(xs, ys) if x <= 0.5 and y > 0.5)
    diff = [x - y for x, y in zip(xs, ys)]
    rng = random.Random(0)
    means = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(boot))
    return {"n": n, "ref_pct": 100 * sum(xs) / n, "other_pct": 100 * sum(ys) / n,
            "diff_pct": 100 * sum(diff) / n,
            "diff_ci95": [100 * means[int(0.025 * boot)], 100 * means[int(0.975 * boot) - 1]],
            "ref_only": b_cnt, "other_only": c_cnt, "p_mcnemar": mcnemar_exact(b_cnt, c_cnt)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--ref", default="hw_robust")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    data = load(args.root)
    if not data:
        raise SystemExit(f"no results under {args.root}/seed*/")
    report: dict = {"methods": {}, "tests": {}}
    for method, seeds in sorted(data.items()):
        entry = {"seeds": sorted(seeds), "episodes": sum(len(v) for v in seeds.values())}
        for key in BINARY:
            per_seed = [100 * sum(e[key] for e in eps) / len(eps) for eps in seeds.values()]
            hits = int(sum(e[key] for eps in seeds.values() for e in eps))
            total = entry["episodes"]
            lo, hi = wilson_interval(hits, total)
            mean = sum(per_seed) / len(per_seed)
            sd = math.sqrt(sum((v - mean) ** 2 for v in per_seed) / (len(per_seed) - 1)) if len(per_seed) > 1 else 0.0
            entry[key] = {"per_seed": per_seed, "mean": mean, "sd_seeds": sd,
                          "pooled_pct": 100 * hits / total, "ci95": [max(0.0, 100 * lo), min(100.0, 100 * hi)]}
        for key in CONTINUOUS:
            vals = [e[key] for eps in seeds.values() for e in eps if e.get(key) is not None and math.isfinite(e[key])]
            entry[key] = {"mean": sum(vals) / len(vals) if vals else None, "n": len(vals)}
        report["methods"][method] = entry
    if args.ref in data:
        for key in BINARY:
            raw = {m: paired(data[args.ref], data[m], key) for m in data if m != args.ref}
            raw = {m: r for m, r in raw.items() if r is not None}
            adj = holm({m: r["p_mcnemar"] for m, r in raw.items()})
            for m, r in raw.items():
                r["p_holm"] = adj[m]
            report["tests"][key] = raw
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text)
    # Console table.
    print(f"{'method':16s} seeds  eps  " + "  ".join(f"{k[:14]:>22s}" for k in BINARY[:4]))
    for method, e in sorted(report["methods"].items()):
        cells = [f"{e[k]['mean']:5.1f}±{e[k]['sd_seeds']:4.1f} [{e[k]['ci95'][0]:4.1f},{e[k]['ci95'][1]:5.1f}]" for k in BINARY[:4]]
        print(f"{method:16s} {len(e['seeds']):5d} {e['episodes']:4d}  " + "  ".join(f"{c:>22s}" for c in cells))
    for key in BINARY[:4]:
        tests = report["tests"].get(key, {})
        if tests:
            print(f"\n{args.ref} vs others on {key}: diff (ref-other) [95% CI], p_holm")
            for m, r in sorted(tests.items()):
                print(f"  {m:16s} {r['diff_pct']:+6.1f} [{r['diff_ci95'][0]:+6.1f},{r['diff_ci95'][1]:+6.1f}]  "
                      f"p={r['p_holm']:.3g}  (n={r['n']}, discordant {r['ref_only']}/{r['other_only']})")


if __name__ == "__main__":
    main()
