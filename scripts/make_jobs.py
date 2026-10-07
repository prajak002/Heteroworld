"""Write job files for the benchmark-v2 campaign (one line per 25-episode shard) for scripts/job_queue.py.

Every experiment is a set of settings x methods x run seeds. Outputs go to
``results/v2/<experiment>/<setting>/seed<k>/<method>_part<first>.json`` so that
``scripts/analyze.py results/v2/<experiment>/<setting>`` works on each setting directly.

Usage:
    python scripts/make_jobs.py main sweep_humans --seeds 0 1 2 --episodes 100 > jobs/campaign.txt
"""

from __future__ import annotations

import argparse
import sys

PY = ".venv/bin/python scripts/run_benchmark.py"
ALL = ("noncoop_mpc", "independent", "union_select", "hetero_world", "cv_robust", "mixture_robust", "hw_robust",
       "oracle_robust")
CORE = ("union_select", "cv_robust", "mixture_robust", "hw_robust", "oracle_robust")

# name -> list of (setting label, extra CLI flags, methods)
EXPERIMENTS: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {
    "main": [("h4", "--humans 4", ALL)],
    "main_learned": [("h4", "--humans 4 --learned", ALL[1:])],
    "sweep_humans": [(f"h{h}", f"--humans {h}", CORE) for h in (2, 6, 8)],
    "sweep_assertive": [(f"a{int(100 * a)}", f"--humans 4 --assertive-prob {a}", CORE) for a in (0.0, 1.0)],
    "dense": [("h6a100", "--humans 6 --assertive-prob 1.0", CORE)],
    # Real NYC sidewalk pinch points (data/nyc/scenarios.json, from NYC Open Data planimetrics + tree census).
    "nyc": [("nyc1", "--humans 4 --sidewalk 2.67 1.17 2.83", CORE), ("nyc2", "--humans 4 --sidewalk 2.95 0.81 3.09", CORE), ("nyc3", "--humans 4 --sidewalk 3.8 1.0 3.1", CORE), ("nyc4", "--humans 4 --sidewalk 2.13 1.04 12.0", CORE)],
    "faulty": [(f"f{k}", f"--humans 4 --faulty-agents {k}", CORE) for k in (1, 2, 3)],
    "packet_loss": [(f"loss{int(100 * p)}", f"--humans 4 --packet-loss {p}", CORE[:4]) for p in (0.1, 0.3, 0.5)],
    "fleet_size": [(f"m{m}", f"--agents {m} --humans {m * 2 // 5}", CORE) for m in (6, 14)],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", choices=sorted(EXPERIMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--shard", type=int, default=25)
    parser.add_argument("--methods", nargs="*", default=None, help="restrict to these methods")
    args = parser.parse_args()
    lines = []
    for exp in args.experiments:
        for label, flags, methods in EXPERIMENTS[exp]:
            for method in methods:
                if args.methods and method not in args.methods:
                    continue
                for seed in args.seeds:
                    out_dir = f"results/v2/{exp}/{label}/seed{seed}"
                    for first in range(0, args.episodes, args.shard):
                        n = min(args.shard, args.episodes - first)
                        out = f"{out_dir}/{method}_part{first}.json"
                        lines.append(f"{out} {PY} --methods {method} {flags} --seed {seed} --episodes {n} "
                                     f"--first-episode {first} --chunk 5 --threads 1 --tag _part{first} --out {out_dir}")
    sys.stdout.write("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs", file=sys.stderr)


if __name__ == "__main__":
    main()
