"""Real-pedestrian forecast study on ETH/UCY (leave-one-scene-out).

For every held-out scene and run seed, the three world-model families (diffusion, JEPA, RSSM) are
trained on the other four scenes. On the held-out scene each family draws ``K`` joint futures; we then
compare the pedestrian forecasts a robot team could score against:

* ``<family>``    one model's own samples;
* ``mixture``     ``K`` samples drawn uniformly from the pooled ``3K``;
* ``barycenter``  the ``K``-atom entropic Wasserstein barycenter of the three sample sets (ours);
* ``cv``          constant velocity (one deterministic future).

Metrics (metres, lower is better), on the target pedestrian (agent 0, the standard protocol) and on all
real agents of the scene: minADE/minFDE over the K samples, ADE/FDE of the sample mean, and the energy
score of the joint forecast.

Usage:
    python scripts/ethucy_experiment.py --seeds 0 1 2 --steps 8000 --out results/ethucy/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hetero_world.learned_models import build  # noqa: E402
from hetero_world.sinkhorn import SinkhornBarycenterConsensus  # noqa: E402

SCENES = ("eth", "hotel", "univ", "zara1", "zara2")
FAMILIES = ("diffusion", "jepa", "rssm")
FAMILY_SEED = {"diffusion": 11, "jepa": 22, "rssm": 33}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(family: str, obs: torch.Tensor, fut: torch.Tensor, steps: int, seed: int, device: torch.device,
          batch: int = 256, lr: float = 3e-4):
    torch.manual_seed(FAMILY_SEED[family] + 1000 * seed)
    model = build(family, obs.shape[1], fut.shape[1]).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    obs, fut = obs.to(device), fut.to(device)
    for step in range(steps):
        idx = torch.randint(0, obs.shape[0], (batch,), device=device)
        o, f = obs[idx], fut[idx]
        # Rotation augmentation about the target (scenes are already target-centred).
        ang = torch.rand(batch, device=device) * 2 * torch.pi
        c, s = torch.cos(ang).view(-1, 1, 1), torch.sin(ang).view(-1, 1, 1)
        rot = lambda p: torch.stack([c[..., 0] * p[..., 0] - s[..., 0] * p[..., 1],
                                     s[..., 0] * p[..., 0] + c[..., 0] * p[..., 1]], dim=-1)
        o = o.clone()
        o[..., 0:2] = rot(o[..., 0:2])
        o[..., 2:4] = rot(o[..., 2:4])
        f = torch.stack([c * f[..., 0] - s * f[..., 1], s * f[..., 0] + c * f[..., 1]], dim=-1)
        loss = model.loss(o, f)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr * 0.5 * (1 + torch.cos(torch.tensor(step / steps * torch.pi))).item()
        opt.step()
        if hasattr(model, "update_target"):
            model.update_target()
    return model.eval()


def energy_score(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    first = (samples - truth.unsqueeze(1)).norm(dim=-1).mean(-1)
    if samples.shape[1] == 1:
        return first
    return first - 0.5 * torch.cdist(samples, samples).mean(dim=(-1, -2))


def metrics(pred: torch.Tensor, fut: torch.Tensor, real: torch.Tensor) -> dict[str, torch.Tensor]:
    """``pred (S, K, H, M, 2)``, ``fut (S, H, M, 2)``, ``real (S, M)`` -> per-sample metric tensors."""
    err = (pred - fut.unsqueeze(1)).norm(dim=-1)                     # (S, K, H, M)
    w = real.float().unsqueeze(1)                                    # (S, 1, M)
    ade_all = (err.mean(2) * w).sum(-1) / w.sum(-1)                  # (S, K)
    fde_all = (err[:, :, -1] * w).sum(-1) / w.sum(-1)
    mean = pred.mean(1)
    merr = (mean - fut).norm(dim=-1)                                 # (S, H, M)
    flat_p = (pred * real.view(real.shape[0], 1, 1, -1, 1)).flatten(2)
    flat_y = (fut * real.view(real.shape[0], 1, -1, 1)).flatten(1)
    return {
        "minADE_target": err[:, :, :, 0].mean(2).min(1).values,
        "minFDE_target": err[:, :, -1, 0].min(1).values,
        "ADEmean_target": merr[:, :, 0].mean(1),
        "FDEmean_target": merr[:, -1, 0],
        "minADE_all": ade_all.min(1).values,
        "minFDE_all": fde_all.min(1).values,
        "energy_all": energy_score(flat_p, flat_y),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/ethucy"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = pick_device(args.device)
    data = {s: torch.load(args.data / f"{s}.pt") for s in SCENES}
    consensus = SinkhornBarycenterConsensus(epsilon=0.01, max_iters=50, outer_iters=2)
    report: dict = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    "results": {}}
    for seed in args.seeds:
        for test in args.scenes:
            t0 = time.time()
            train_obs = torch.cat([data[s]["obs"] for s in SCENES if s != test])
            train_fut = torch.cat([data[s]["future"] for s in SCENES if s != test])
            models = {f: train(f, train_obs, train_fut, args.steps, seed, device) for f in FAMILIES}
            obs, fut, real = data[test]["obs"], data[test]["future"], data[test]["real"]
            gen = torch.Generator().manual_seed(seed)
            sums: dict[str, dict[str, float]] = {}
            for i in range(0, obs.shape[0], 256):
                o = obs[i:i + 256].to(device)
                f, r = fut[i:i + 256], real[i:i + 256]
                with torch.no_grad():
                    samples = {fam: m.sample(o, args.k, gen).cpu() for fam, m in models.items()}
                stack = torch.stack([samples[fam] for fam in FAMILIES], dim=1)        # (S, 3, K, H, M, 2)
                flat = stack.flatten(3)
                bary = consensus(flat, torch.ones(flat.shape[0], len(FAMILIES))).support.view_as(stack[:, 0])
                pooled = stack.flatten(1, 2)
                pick = torch.randint(0, pooled.shape[1], (pooled.shape[0], args.k), generator=gen)
                mix = pooled[torch.arange(pooled.shape[0]).unsqueeze(1), pick]
                tau = (0.4 * torch.arange(1, fut.shape[1] + 1)).view(1, 1, -1, 1, 1)
                oc = o.cpu()
                cv = oc[:, None, None, :, 0:2] + tau * oc[:, None, None, :, 2:4]
                refs = dict(samples) | {"mixture": mix, "barycenter": bary, "cv": cv}
                for name, pred in refs.items():
                    for key, val in metrics(pred, f, r).items():
                        sums.setdefault(name, {}).setdefault(key, 0.0)
                        sums[name][key] += float(val.sum())
            n = obs.shape[0]
            res = {name: {k: v / n for k, v in m.items()} for name, m in sums.items()}
            report["results"][f"{test}/seed{seed}"] = res
            print(f"seed {seed} {test:6s} ({n} samples, {time.time() - t0:.0f}s)", flush=True)
            for name, m in res.items():
                print(f"   {name:10s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
