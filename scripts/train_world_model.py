"""Train one learned world-model family on simulator trajectories and report forecasting metrics.

Each family is trained on its own random 70% subset of the training episodes with its own seed,
so no two families share weights, latents or training data.

Usage:
    python scripts/train_world_model.py --family diffusion --data data/wm_train.pt --minutes 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hetero_world.learned_models import FAMILIES, build  # noqa: E402

FAMILY_SEED = {"diffusion": 11, "jepa": 22, "rssm": 33}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, obs, fut, device, n: int = 20, batch: int = 512) -> dict:
    """minADE/minFDE over ``n`` samples and mean ADE of all samples, on active agents only (metres)."""
    model.eval()
    gen = torch.Generator().manual_seed(0)
    tot = {"minADE": 0.0, "minFDE": 0.0, "meanADE": 0.0, "cv_ADE": 0.0, "cv_FDE": 0.0}
    count = 0
    for i in range(0, obs.shape[0], batch):
        o, f = obs[i:i + batch].to(device), fut[i:i + batch].to(device)
        s = model.sample(o, n, gen)                                   # (S, n, H, M, 2)
        act = (o[..., 6] < 0.5).float()                               # (S, M)
        err = (s - f.unsqueeze(1)).norm(dim=-1)                       # (S, n, H, M)
        denom = act.sum(-1).clamp_min(1)
        ade = (err.mean(2) * act.unsqueeze(1)).sum(-1) / denom.unsqueeze(1)
        fde = (err[:, :, -1] * act.unsqueeze(1)).sum(-1) / denom.unsqueeze(1)
        tot["minADE"] += ade.min(1).values.sum().item()
        tot["minFDE"] += fde.min(1).values.sum().item()
        tot["meanADE"] += ade.mean(1).sum().item()
        tau = 0.25 * torch.arange(1, f.shape[1] + 1, device=device).view(1, -1, 1, 1)
        cv = o[:, None, :, 0:2] + o[:, None, :, 2:4] * tau
        cerr = (cv - f).norm(dim=-1)
        tot["cv_ADE"] += ((cerr.mean(1) * act).sum(-1) / denom).sum().item()
        tot["cv_FDE"] += ((cerr[:, -1] * act).sum(-1) / denom).sum().item()
        count += o.shape[0]
    model.train()
    return {k: round(v / count, 4) for k, v in tot.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--data", type=Path, default=Path("data/wm_train.pt"))
    parser.add_argument("--minutes", type=float, default=30.0, help="wall-clock training budget")
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--subset", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0, help="training seed (initialisation and data subset)")
    parser.add_argument("--out", type=Path, default=None, help="default: checkpoints/ for seed 0, else checkpoints/seed<k>/")
    args = parser.parse_args()

    device = pick_device(args.device)
    if args.out is None:
        args.out = Path("checkpoints") if args.seed == 0 else Path(f"checkpoints/seed{args.seed}")
    family_seed = FAMILY_SEED[args.family] + 1000 * args.seed
    torch.manual_seed(family_seed)
    data = torch.load(args.data)
    obs, fut = data["obs"].float(), data["future"].float()
    num = obs.shape[0]
    gen = torch.Generator().manual_seed(1234)
    perm = torch.randperm(num, generator=gen)
    num_val = min(5000, num // 10)
    val_idx, pool = perm[:num_val], perm[num_val:]
    own = pool[torch.randperm(pool.numel(), generator=torch.Generator().manual_seed(family_seed))]
    own = own[: int(args.subset * own.numel())]
    tr_obs, tr_fut = obs[own].to(device), fut[own].to(device)
    va_obs, va_fut = obs[val_idx], fut[val_idx]

    model = build(args.family, data["num_agents"], data["horizon"]).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    params = sum(p.numel() for p in model.parameters())
    print(f"{args.family}: {params / 1e6:.2f}M params, {own.numel()} training samples, device {device}", flush=True)
    t0, step, log = time.time(), 0, []
    sched_total = args.max_steps
    while step < args.max_steps and (time.time() - t0) < 60 * args.minutes:
        idx = torch.randint(0, tr_obs.shape[0], (args.batch,), device=device)
        loss = model.loss(tr_obs[idx], tr_fut[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            # Cosine decay over whichever budget binds first (steps or wall clock).
            frac = min(1.0, max(step / args.max_steps, (time.time() - t0) / (60 * args.minutes)))
            g["lr"] = args.lr * 0.5 * (1 + torch.cos(torch.tensor(frac * torch.pi))).item()
        opt.step()
        if hasattr(model, "update_target"):
            model.update_target()
        step += 1
        if step % 1000 == 0:
            print(f"step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            log.append({"step": step, "loss": loss.item(), "time_s": round(time.time() - t0, 1)})
    metrics = evaluate(model, va_obs, va_fut, device)
    print(f"{args.family} validation: {metrics}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"family": args.family, "state_dict": model.cpu().state_dict(), "num_agents": data["num_agents"],
                "horizon": data["horizon"], "steps": step, "train_samples": int(own.numel()),
                "params": params, "val": metrics}, args.out / f"wm_{args.family}.pt")
    (args.out / f"wm_{args.family}.json").write_text(json.dumps(
        {"family": args.family, "params": params, "steps": step, "train_samples": int(own.numel()),
         "minutes": round((time.time() - t0) / 60, 1), "device": str(device), "val": metrics, "log": log}, indent=1))


if __name__ == "__main__":
    main()
