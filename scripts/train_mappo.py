"""Train the MAPPO baseline on the M-agent bottleneck corridor and save a checkpoint.

Usage:
    python scripts/train_mappo.py --iterations 400 --agents 10 --out checkpoints/mappo_M10.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hetero_world.config import CorridorConfig  # noqa: E402
from hetero_world.mappo import MAPPOConfig, train_mappo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/mappo_M10.pt"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = CorridorConfig(num_agents=args.agents)
    pcfg = MAPPOConfig(iterations=args.iterations, num_envs=args.envs)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save(agent, log, iteration: int) -> None:
        torch.save({"state_dict": agent.state_dict(), "config": asdict(pcfg), "num_agents": args.agents,
                    "iterations_completed": iteration}, args.out)
        args.out.with_suffix(".log.json").write_text(json.dumps(log, indent=2))
        print(f"saved {args.out} at iteration {iteration}", flush=True)

    agent, log = train_mappo(cfg, pcfg, seed=args.seed, checkpoint=save)
    save(agent, log, args.iterations)


if __name__ == "__main__":
    main()
