"""Merge partial result files of one method (split with --first-episode) into a single summary file.

Usage:
    python scripts/merge_results.py results/main/procrustes_part0.json results/main/procrustes_part1.json \
        --out results/main/procrustes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hetero_world.benchmark import summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(p.read_text()) for p in args.parts]
    if len({p["method"] for p in payloads}) != 1:
        raise ValueError("all parts must belong to the same method")
    episodes = sorted((e for p in payloads for e in p["episodes"]), key=lambda e: e["seed"])
    if len({e["seed"] for e in episodes}) != len(episodes):
        raise ValueError("parts overlap in episode seeds")
    merged = dict(payloads[0])
    merged["settings"] = dict(payloads[0]["settings"], episodes=len(episodes), first_episode=0)
    merged["wall_clock_s"] = sum(p["wall_clock_s"] for p in payloads)
    merged["episodes"] = episodes
    merged["summary"] = summarize(episodes)
    args.out.write_text(json.dumps(merged, indent=1))
    for part in args.parts:
        part.rename(part.with_suffix(".part"))
    print(f"merged {len(episodes)} episodes into {args.out}")


if __name__ == "__main__":
    main()
