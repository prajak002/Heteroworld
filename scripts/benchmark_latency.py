"""Measure per-agent consensus latency, its scaling with M, and Sinkhorn convergence versus (epsilon, iterations).

Usage:
    python scripts/benchmark_latency.py --out results/latency.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hetero_world.benchmark import EVAL_SEED_BASE  # noqa: E402
from hetero_world.config import ConsensusConfig, CorridorConfig, PredictionConfig  # noqa: E402
from hetero_world.controllers import PlanConsensusController  # noqa: E402
from hetero_world.env import BottleneckCorridorEnv  # noqa: E402
from hetero_world.latency import barycenter_closure, consensus_closure, pairwise_closure  # noqa: E402
from hetero_world.sinkhorn import SinkhornBarycenterConsensus, measure_latency_ms  # noqa: E402
from hetero_world.world_models import scene_from_env  # noqa: E402

METHODS = ("hetero_world", "union_select", "independent", "feature_sharing", "procrustes")


def scene_predictions(controller: PlanConsensusController, cfg: CorridorConfig, seed: int, warmup_steps: int):
    """Rollouts and state of one scene after ``warmup_steps`` of constant-speed driving."""
    env = BottleneckCorridorEnv(cfg, 1)
    env.reset([seed])
    for _ in range(warmup_steps):
        env.step(env.v_des.clone(), torch.zeros_like(env.v_des))
    gen = torch.Generator().manual_seed(seed)
    preds = controller.fleet.predict(scene_from_env(env), env.agent_type, gen)[0]
    return preds, env.agent_type[0], env.progress_coordinate[0], env.direction[0], env.active[0]


def stats(values: list[float]) -> dict:
    """Mean, standard deviation and median."""
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values), "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--scaling-scenes", type=int, default=5)
    parser.add_argument("--agents", default="2,4,8,16,24,32,40,50")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/latency.json"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    pcfg = PredictionConfig()
    ccfg = ConsensusConfig(outer_iters=2)
    result: dict = {"hardware": {"machine": platform.machine(), "processor": platform.processor(),
                                 "platform": platform.platform(), "torch": torch.__version__,
                                 "threads": args.threads, "device": "cpu"},
                    "consensus_config": ccfg.__dict__}

    # 1. Per-agent latency at M = 10 on the benchmark scenes.
    cfg = CorridorConfig(num_agents=10)
    controller = PlanConsensusController("hetero_world", cfg, pcfg, ccfg)
    per_method: dict[str, list[float]] = {m: [] for m in METHODS}
    per_method["pairwise_sinkhorn"] = []
    for k in range(args.episodes):
        preds, types, u, d, act = scene_predictions(controller, cfg, EVAL_SEED_BASE + k, warmup_steps=20)
        for method in METHODS:
            fn = consensus_closure(method, controller, preds, types, u, d, act)
            per_method[method].append(statistics.median(measure_latency_ms(fn, device, warmup=2, repeats=5)))
        fn = pairwise_closure(preds, ccfg)
        per_method["pairwise_sinkhorn"].append(statistics.median(measure_latency_ms(fn, device, warmup=1, repeats=3)))
        if (k + 1) % 20 == 0:
            print(f"latency scenes {k + 1}/{args.episodes}", flush=True)
    result["per_agent_latency_ms_M10"] = {m: stats(v) for m, v in per_method.items()}

    # 2. Scaling with the number of agents.
    scaling = []
    for num_agents in [int(x) for x in args.agents.split(",")]:
        cfg_m = CorridorConfig(num_agents=num_agents)
        ctrl_m = PlanConsensusController("hetero_world", cfg_m, pcfg, ccfg, belief_every=0)
        rows: dict[str, list[float]] = {}
        for k in range(args.scaling_scenes):
            preds, types, u, d, act = scene_predictions(ctrl_m, cfg_m, EVAL_SEED_BASE + k, warmup_steps=0)
            joint = preds.reshape(num_agents, pcfg.num_rollouts, -1)
            fixed = preds[:, :, :, 0, :].reshape(num_agents, pcfg.num_rollouts, -1)
            timers = {
                "hetero_world_full": consensus_closure("hetero_world", ctrl_m, preds, types, u, d, act),
                "barycenter_joint": barycenter_closure(joint, ccfg),
                "barycenter_fixed_dim": barycenter_closure(fixed, ccfg),
                "pairwise_joint": pairwise_closure(preds, ccfg),
                "pairwise_fixed_dim": pairwise_closure(preds[:, :, :, :1, :], ccfg),
                "union_select": consensus_closure("union_select", ctrl_m, preds, types, u, d, act),
                "procrustes": consensus_closure("procrustes", ctrl_m, preds, types, u, d, act),
            }
            for name, fn in timers.items():
                repeats = 3 if name.startswith("pairwise") else 5
                rows.setdefault(name, []).append(statistics.median(measure_latency_ms(fn, device, warmup=1, repeats=repeats)))
        scaling.append({"agents": num_agents, **{name: stats(v) for name, v in rows.items()}})
        print(f"scaling M={num_agents}: " + ", ".join(f"{n}={statistics.fmean(v):.1f}ms" for n, v in rows.items()), flush=True)
    result["scaling"] = scaling

    # 3. Sinkhorn convergence versus epsilon and iteration count (M = 10).
    convergence = []
    scenes = [scene_predictions(controller, cfg, EVAL_SEED_BASE + k, warmup_steps=20)[0] for k in range(10)]
    for eps in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        for iters in (10, 25, 50, 100, 200, 500):
            module = SinkhornBarycenterConsensus(eps, iters, ccfg.outer_iters)
            margs, w2s, times = [], [], []
            for preds in scenes:
                flat = preds.reshape(preds.shape[0], preds.shape[1], -1)
                out = module(flat)
                margs.append(float(out.marginal_error.mean()))
                w2s.append(float(out.w2_per_point.mean()))
                times.append(statistics.median(measure_latency_ms(lambda: module(flat), device, warmup=1, repeats=3)))
            convergence.append({"epsilon": eps, "iters": iters, "marginal_error": stats(margs),
                                "w2_per_point_m": stats(w2s), "latency_ms": stats(times)})
        print(f"convergence eps={eps} done", flush=True)
    result["convergence"] = convergence

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
