"""Run the Monte Carlo benchmark for one or more methods and save per-episode records and summaries.

Usage:
    python scripts/run_benchmark.py --methods hetero_world,noncoop_mpc --episodes 100 --out results/main
    python scripts/run_benchmark.py --methods hetero_world --packet-loss 0.2 --episodes 50 --out results/loss
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hetero_world.benchmark import ALL_METHODS, EVAL_SEED_BASE, SEED_STRIDE, make_controller, run_episodes, summarize  # noqa: E402
from hetero_world.config import ConsensusConfig, CorridorConfig, PredictionConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--sinkhorn-iters", type=int, default=50)
    parser.add_argument("--outer-iters", type=int, default=2)
    parser.add_argument("--packet-loss", type=float, default=0.0)
    parser.add_argument("--humans", type=int, default=0,
                        help="number of agents that are uncontrolled, non-communicating pedestrians")
    parser.add_argument("--faulty-agents", type=int, default=0,
                        help="number of agents whose world model hallucinates over-fast motion")
    parser.add_argument("--learned-scope", choices=("all", "humans"), default="all",
                        help="with --learned: forecast every agent (all) or only pedestrians (humans; robot candidates stay synthetic)")
    parser.add_argument("--learned", action="store_true",
                        help="use the trained world models (checkpoints/wm_*.pt) instead of the synthetic ones")
    parser.add_argument("--seed", type=int, default=0,
                        help="run seed: fresh scenarios (episode seeds offset by 100000*seed), controller randomness, "
                             "synthetic-codec weights and, with --learned, the world models in checkpoints/seed<k>/")
    parser.add_argument("--assertive-prob", type=float, default=None,
                        help="probability that a pedestrian enters the gap without waiting (benchmark v2)")
    parser.add_argument("--sidewalk", type=float, nargs=3, default=None, metavar=("WIDE", "PINCH", "LENGTH"),
                        help="real sidewalk geometry in metres (typical width, pinch width, pinch length)")
    parser.add_argument("--belief-every", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mappo-checkpoint", default="checkpoints/mappo_M10.pt")
    parser.add_argument("--first-episode", type=int, default=0,
                        help="index of the first episode (to split one method across processes)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", type=Path, default=Path("results/main"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    cfg = CorridorConfig(num_agents=args.agents, num_humans=args.humans)
    if args.sidewalk is not None:
        cfg = cfg.with_sidewalk(*args.sidewalk)
    if args.assertive_prob is not None:
        cfg = dataclasses.replace(cfg, assertive_prob=args.assertive_prob)
    pcfg = PredictionConfig()
    ccfg = ConsensusConfig(epsilon=args.epsilon, max_iters=args.sinkhorn_iters, outer_iters=args.outer_iters)
    args.out.mkdir(parents=True, exist_ok=True)
    for method in args.methods.split(","):
        start = time.time()
        controller = make_controller(method, cfg, pcfg, ccfg, packet_loss=args.packet_loss,
                                     mappo_checkpoint=args.mappo_checkpoint, belief_every=args.belief_every,
                                     faulty_agents=args.faulty_agents, learned=args.learned, learned_scope=args.learned_scope,
                                     seed=args.seed)
        records = run_episodes(controller, cfg, args.episodes, args.chunk, seed_base=EVAL_SEED_BASE + SEED_STRIDE * args.seed + args.first_episode,
                               progress=lambda m: print(m, flush=True))
        summary = summarize(records)
        payload = {
            "method": method,
            "settings": {"agents": args.agents, "episodes": args.episodes, "epsilon": args.epsilon,
                         "sinkhorn_iters": args.sinkhorn_iters, "outer_iters": args.outer_iters,
                         "packet_loss": args.packet_loss, "faulty_agents": args.faulty_agents, "humans": args.humans, "learned": args.learned, "learned_scope": args.learned_scope, "seed": args.seed,
                         "assertive_prob": cfg.assertive_prob, "sidewalk": args.sidewalk, "courtesy": cfg.courtesy, "num_rollouts": pcfg.num_rollouts,
                         "first_episode": args.first_episode,
                         "horizon": pcfg.horizon, "plan_dt": pcfg.plan_dt,
                         "replan_hz": pcfg.replan_hz(cfg.dt)},
            "wall_clock_s": round(time.time() - start, 1),
            "summary": summary,
            "episodes": records,
        }
        name = f"{method}{args.tag}.json"
        (args.out / name).write_text(json.dumps(payload, indent=1))
        brief = {k: (round(v["mean"], 3) if v.get("mean") is not None else None) for k, v in summary.items()}
        print(f"[done] {method} in {payload['wall_clock_s']}s: {brief}", flush=True)


if __name__ == "__main__":
    main()
