"""Record single episodes for scene visualisation.

For every (method, seed) pair, run one closed-loop episode and save per-step agent positions,
headings, finish flags and collision flags, plus the joint plan selected by observer 0 at every
replan. The output JSON drives ``docs/scenes.html``.

Usage:
    python scripts/record_scenes.py --seeds 20250000,20250009 \
        --methods hetero_world,union_select,independent,feature_sharing,noncoop_mpc --out docs/scenes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hetero_world.benchmark import make_controller  # noqa: E402
from hetero_world.config import ConsensusConfig, CorridorConfig, PredictionConfig  # noqa: E402
from hetero_world.env import BottleneckCorridorEnv  # noqa: E402
from hetero_world.geometry import free_half_width  # noqa: E402
from hetero_world.world_models import WORLD_MODEL_NAMES  # noqa: E402


def r2(t: torch.Tensor) -> list:
    """Round a tensor to centimetres and convert it to nested lists."""
    def rnd(v):
        return [rnd(x) for x in v] if isinstance(v, list) else round(v, 2)
    return rnd(t.tolist())


def record(method: str, seed: int, cfg: CorridorConfig, pcfg: PredictionConfig, ccfg: ConsensusConfig) -> dict:
    controller = make_controller(method, cfg, pcfg, ccfg)
    env = BottleneckCorridorEnv(cfg, 1)
    env.reset([seed])
    controller.reset(env, seed=seed * 31 + 5)
    frames, plans = [], []
    last_plan = None
    with torch.no_grad():
        while not bool(env.over.all()):
            controller.record_step(env)
            v_cmd, omega = controller.act(env)
            plan = getattr(controller, "plan", None)
            if plan is not None and plan is not last_plan:
                entry = {"t": len(frames), "xy": r2(plan[0, 0].permute(1, 0, 2))}
                # The pedestrian forecast the plan was scored against: the reference futures of a robust
                # method (12 atoms), or the selected candidate's own pedestrian rows (union / hetero_world).
                ref = getattr(controller, "last_ref_humans", None)
                if ref is not None and method in ("hw_robust", "mixture_robust", "cv_robust", "oracle_robust"):
                    atoms = ref[0, 0]
                    pick = torch.linspace(0, atoms.shape[0] - 1, min(12, atoms.shape[0])).long()
                    entry["forecast"] = r2(atoms[pick].permute(0, 2, 1, 3))      # (S, M, H, 2)
                else:
                    entry["forecast"] = [entry["xy"]]                            # (1, M, H, 2)
                plans.append(entry)
                last_plan = plan
            env.step(v_cmd, omega)
            frames.append({"p": r2(env.pos[0]), "h": r2(env.theta[0]), "d": env.done[0].int().tolist(),
                           "c": int(env.collided[0])})
    metrics = {k: float(v[0]) for k, v in env.episode_metrics().items()}
    outcome = ("collision" if metrics["collision"] else "deadlock" if metrics["deadlock"]
               else "completed" if metrics["completed"] else "timeout")
    return {"method": method, "seed": seed, "outcome": outcome, "metrics": metrics, "frames": frames,
            "plans": plans, "agent_type": env.agent_type[0].tolist(), "direction": env.direction[0].tolist(),
            "is_human": env.is_human[0].int().tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="20250000,20250009")
    parser.add_argument("--methods", default="hetero_world,union_select,independent,feature_sharing,noncoop_mpc")
    parser.add_argument("--humans", type=int, default=0)
    parser.add_argument("--sidewalk", type=float, nargs=3, default=None, metavar=("WIDE", "PINCH", "LENGTH"))
    parser.add_argument("--label", default="", help="scenario label stored in the output")
    parser.add_argument("--out", default="docs/scenes.json")
    args = parser.parse_args()
    cfg, pcfg, ccfg = CorridorConfig(num_humans=args.humans), PredictionConfig(), ConsensusConfig()
    if args.sidewalk is not None:
        cfg = cfg.with_sidewalk(*args.sidewalk)
    xs = torch.linspace(-cfg.half_length, cfg.half_length, 241)
    out = {"label": args.label, "sidewalk": args.sidewalk, "corridor_half_width": cfg.corridor_half_width,
           "gap_half_width": cfg.gap_half_width, "gap_half_length": cfg.gap_half_length, "taper_length": cfg.taper_length,
           "dt": cfg.dt, "agent_radius": cfg.agent_radius, "half_length": cfg.half_length,
           "goal_distance": cfg.goal_distance, "world_models": list(WORLD_MODEL_NAMES),
           "walls": {"x": r2(xs), "w": r2(free_half_width(xs, cfg))}, "episodes": []}
    for seed in [int(s) for s in args.seeds.split(",")]:
        for method in args.methods.split(","):
            ep = record(method, seed, cfg, pcfg, ccfg)
            print(f"{method} seed {seed}: {ep['outcome']} in {len(ep['frames'])} steps", flush=True)
            out["episodes"].append(ep)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
