"""Offline forecast quality of the pedestrian reference used for scoring (no planner in the loop).

Held-out benchmark-v2 scenes (pedestrians only, scenario seeds of run seed ``--seed``) are sampled
every ``--every`` steps. In each scene a random subset of ``--robots`` agents plays the broadcasting
robots; the remaining agents are the pedestrians whose futures must be forecast. Compared references
(each a set of joint futures, evaluated on the pedestrian rows of active agents):

* ``single``    the rollouts of one broadcasting robot (its own world model only);
* ``mixture``   ``N`` futures drawn uniformly from the pooled broadcasts;
* ``barycenter`` the support of the entropic Wasserstein barycenter of the broadcasts (ours);
* ``cv``        constant-velocity extrapolation (one deterministic future).

Metrics (metres, lower is better): energy score (a proper scoring rule for multivariate forecasts),
ADE/FDE of the forecast mean, and minADE over the forecast set. Results are averaged over scenes and
reported with a 95% bootstrap CI over episodes.

Usage:
    python scripts/eval_forecasts.py --episodes 200 --seed 0 [--learned] --out results/forecast/seed0.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hetero_world.benchmark import EVAL_SEED_BASE, SEED_STRIDE  # noqa: E402
from hetero_world.config import ConsensusConfig, CorridorConfig, PredictionConfig  # noqa: E402
from hetero_world.env import BottleneckCorridorEnv  # noqa: E402
from hetero_world.learned_models import LearnedWorldModelFleet, observe  # noqa: E402
from hetero_world.sinkhorn import SinkhornBarycenterConsensus  # noqa: E402
from hetero_world.world_models import HeterogeneousWorldModelFleet, scene_from_env  # noqa: E402

REFERENCES = ("single", "mixture", "barycenter", "cv")


def energy_score(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """``samples (S, n, D)``, ``truth (S, D)`` -> ``(S,)`` energy score ``E|X-y| - 0.5 E|X-X'|``."""
    first = (samples - truth.unsqueeze(1)).norm(dim=-1).mean(-1)
    if samples.shape[1] == 1:
        return first
    second = torch.cdist(samples, samples).mean(dim=(-1, -2))
    return first - 0.5 * second


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--robots", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learned", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    base = CorridorConfig()
    cfg = base.with_humans(base.num_agents)
    pcfg, ccfg = PredictionConfig(), ConsensusConfig()
    num_agents, horizon, n = cfg.num_agents, pcfg.horizon, pcfg.num_rollouts
    synthetic = HeterogeneousWorldModelFleet(cfg, pcfg, seed=args.seed)
    learned = None
    if args.learned:
        ckpt = "checkpoints" if args.seed == 0 else f"checkpoints/seed{args.seed}"
        learned = LearnedWorldModelFleet(n, ckpt)
    consensus = SinkhornBarycenterConsensus(ccfg.epsilon, ccfg.max_iters, ccfg.outer_iters)
    gen = torch.Generator().manual_seed(1000 + args.seed)
    pick = random.Random(args.seed)
    stride = pcfg.plan_dt / cfg.dt
    offsets = stride * torch.arange(1, horizon + 1, dtype=torch.float)
    span = int(offsets[-1].ceil())

    per_episode: dict[str, dict[int, list[torch.Tensor]]] = {}
    for start in range(0, args.episodes, args.chunk):
        seeds = [EVAL_SEED_BASE + SEED_STRIDE * args.seed + k
                 for k in range(start, min(start + args.chunk, args.episodes))]
        env = BottleneckCorridorEnv(cfg, len(seeds))
        env.reset(seeds)
        zeros = torch.zeros(len(seeds), num_agents)
        frames, snaps = [], []
        t = 0
        while not bool(env.over.all()):
            frames.append(env.pos.clone())
            if t % args.every == 0:
                snaps.append((t, scene_from_env(env), observe(env), env.agent_type.clone(), env.active.clone(),
                              env.speed.clone(), env.theta.clone()))
            env.step(zeros, zeros)
            t += 1
        frames.append(env.pos.clone())
        pos = torch.stack(frames, dim=1)
        direction = env.direction.view(len(seeds), 1, num_agents)
        goal = torch.stack([direction * cfg.goal_distance, direction * cfg.lane_offset], dim=-1).expand_as(pos)
        pos = torch.where((pos[..., 0].abs() > cfg.half_length + 10.0).unsqueeze(-1), goal, pos)
        num_t = pos.shape[1]
        for t, scene, obs, agent_type, active, speed, theta in snaps:
            if t + span >= num_t:
                continue
            lo = (t + offsets).floor().long()
            w = ((t + offsets) - lo).view(1, -1, 1, 1)
            truth = (1 - w) * pos[:, lo] + w * pos[:, (lo + 1).clamp(max=num_t - 1)]     # (B, H, M, 2)
            if learned is not None:
                preds = learned.predict(obs, scene.pos, agent_type, gen)
            else:
                preds = synthetic.predict(scene, agent_type, gen)                         # (B, M, N, H, M, 2)
            for b in range(len(seeds)):
                if not bool(active[b].any()):
                    continue
                robots = sorted(pick.sample(range(num_agents), args.robots))
                peds = [j for j in range(num_agents) if j not in robots and bool(active[b, j])]
                if not peds:
                    continue
                own = preds[b, robots]                                                   # (R, N, H, M, 2)
                flat = own.reshape(1, len(robots), n, -1)
                bary = consensus(flat, torch.ones(1, len(robots))).support.view(n, horizon, num_agents, 2)
                pooled = own.reshape(-1, horizon, num_agents, 2)
                mix = pooled[torch.randint(0, pooled.shape[0], (n,), generator=gen)]
                tau = (pcfg.plan_dt * torch.arange(1, horizon + 1)).view(-1, 1, 1)
                vel = speed[b].unsqueeze(-1) * torch.stack([torch.cos(theta[b]), torch.sin(theta[b])], dim=-1)
                cv = (scene.pos[b] + tau * vel).unsqueeze(0)
                forecasts = {"single": own[0], "mixture": mix, "barycenter": bary, "cv": cv}
                y = truth[b][:, peds]                                                    # (H, P, 2)
                for name, f in forecasts.items():
                    fp = f[:, :, peds]                                                   # (n, H, P, 2)
                    err = (fp - y.unsqueeze(0)).norm(dim=-1)                             # (n, H, P)
                    mean = fp.mean(0)
                    row = torch.tensor([
                        float(energy_score(fp.flatten(1).unsqueeze(0), y.flatten().unsqueeze(0))[0]),
                        float((mean - y).norm(dim=-1).mean()),
                        float((mean[-1] - y[-1]).norm(dim=-1).mean()),
                        float(err.mean(dim=(1, 2)).min()),
                    ])
                    per_episode.setdefault(name, {}).setdefault(seeds[b], []).append(row)
        print(f"episodes {start + len(seeds)}/{args.episodes}", flush=True)

    metrics = ("energy_score", "ADE_mean", "FDE_mean", "minADE")
    rng = random.Random(0)
    report = {"settings": vars(args) | {"out": str(args.out)}, "references": {}}
    for name in REFERENCES:
        eps = {s: torch.stack(rows).mean(0) for s, rows in per_episode[name].items()}
        keys = sorted(eps)
        stack = torch.stack([eps[k] for k in keys])
        boots = torch.stack([stack[[rng.randrange(len(keys)) for _ in keys]].mean(0) for _ in range(1000)])
        lo, hi = boots.quantile(0.025, dim=0), boots.quantile(0.975, dim=0)
        report["references"][name] = {m: {"mean": float(stack[:, i].mean()), "ci95": [float(lo[i]), float(hi[i])]}
                                      for i, m in enumerate(metrics)}
        report["references"][name]["episodes"] = len(keys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"{'reference':12s}" + "".join(f"{m:>22s}" for m in metrics))
    for name, r in report["references"].items():
        print(f"{name:12s}" + "".join(f"{r[m]['mean']:8.3f} [{r[m]['ci95'][0]:.3f},{r[m]['ci95'][1]:.3f}]" for m in metrics))


if __name__ == "__main__":
    main()
