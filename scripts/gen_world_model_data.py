"""Generate training data for the learned world models from closed-loop simulator episodes.

Every agent is a pedestrian (``num_humans = num_agents``), so episodes simulate quickly and contain
real interactions: queueing, braking for agents ahead and a random passing order through the gap.
Every ``--every`` simulator steps we store the observed scene and the true joint future over the
plan horizon (``H`` steps of ``plan_dt`` seconds, linearly interpolated between simulator steps).

Output ``.pt`` file with
    obs     (S, M, 7)   x, y, vx, vy, direction, v_des, done
    future  (S, H, M, 2) future positions (parked agents keep their current position)

Usage:
    python scripts/gen_world_model_data.py --episodes 2000 --out data/wm_train.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hetero_world.config import CorridorConfig, PredictionConfig  # noqa: E402
from hetero_world.env import BottleneckCorridorEnv  # noqa: E402
from hetero_world.learned_models import observe as observation  # noqa: E402

TRAIN_SEED_BASE = 90_000_000  # disjoint from the evaluation seeds (20_250_000 + k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=TRAIN_SEED_BASE)
    parser.add_argument("--out", type=Path, default=Path("data/wm_train.pt"))
    args = parser.parse_args()
    base = CorridorConfig()
    cfg = base.with_humans(base.num_agents)
    pcfg = PredictionConfig()
    stride = pcfg.plan_dt / cfg.dt
    offsets = stride * torch.arange(1, pcfg.horizon + 1, dtype=torch.float)
    span = int(offsets[-1].ceil())
    obs_all, fut_all = [], []
    t0 = time.time()
    for start in range(0, args.episodes, args.chunk):
        seeds = list(range(args.seed_base + start, args.seed_base + min(start + args.chunk, args.episodes)))
        env = BottleneckCorridorEnv(cfg, len(seeds), terminate_on_deadlock=True)
        env.reset(seeds)
        frames, obs, alive = [], [], []
        zeros = torch.zeros(len(seeds), cfg.num_agents)
        while not bool(env.over.all()):
            frames.append(env.pos.clone())
            obs.append(observation(env))
            alive.append(~env.over.clone())
            env.step(zeros, zeros)
        frames.append(env.pos.clone())
        pos = torch.stack(frames, dim=1)                       # (B, T+1, M, 2)
        # Agents that reached their goal are parked far away by the simulator; pin them to the goal.
        direction = env.direction.view(len(seeds), 1, cfg.num_agents)
        goal = torch.stack([direction * cfg.goal_distance, direction * cfg.lane_offset], dim=-1).expand_as(pos)
        parked = pos[..., 0].abs() > cfg.half_length + 10.0
        pos = torch.where(parked.unsqueeze(-1), goal, pos)
        num_t = pos.shape[1]
        for t in range(0, len(obs), args.every):
            if t + span >= num_t:
                break
            keep = alive[t]
            lo = (t + offsets).floor().long()
            w = ((t + offsets) - lo).view(1, -1, 1, 1)
            fut = (1 - w) * pos[:, lo] + w * pos[:, (lo + 1).clamp(max=num_t - 1)]
            obs_all.append(obs[t][keep])
            fut_all.append(fut[keep])
        print(f"episodes {start + len(seeds)}/{args.episodes}: {sum(o.shape[0] for o in obs_all)} samples "
              f"({time.time() - t0:.0f}s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"obs": torch.cat(obs_all), "future": torch.cat(fut_all), "plan_dt": pcfg.plan_dt,
                "horizon": pcfg.horizon, "num_agents": cfg.num_agents}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
