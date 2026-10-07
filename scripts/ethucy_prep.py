"""Preprocess ETH/UCY (BIWI ETH, Hotel; UCY Zara1, Zara2, Univ excerpt) into joint-future samples.

Standard protocol: 2.5 Hz (0.4 s), 8 observed steps, 12 predicted steps (4.8 s). One sample per
pedestrian and window in which it is present for all 20 frames. The scene is centred on that target
pedestrian at the last observed frame and rotated so its observed heading points along +x. Up to
``M - 1`` neighbours that are present for the whole window are added, nearest first; empty slots
are padding (``done = 1``).

Observation features per agent (same layout as the corridor models): x, y, vx, vy, 1, speed, done.
Output per scene: ``data/ethucy/<scene>.pt`` with obs (S, M, 7), future (S, 12, M, 2),
history (S, 8, M, 2) and real (S, M) masks.

Sources: Pellegrini et al., ICCV 2009 (BIWI); Lerner et al., Eurographics 2007 (UCY). Research use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

SCENES = {"eth": "biwi_eth.txt", "hotel": "biwi_hotel.txt", "zara1": "crowds_zara01.txt",
          "zara2": "crowds_zara02.txt", "univ": "uni_examples.txt"}
OBS, PRED, DT = 8, 12, 0.4


def build(path: Path, num_agents: int) -> dict[str, torch.Tensor]:
    raw = np.loadtxt(path)
    frames = np.unique(raw[:, 0])
    step = np.median(np.diff(frames))
    index = {f: i for i, f in enumerate(frames)}
    peds = np.unique(raw[:, 1]).astype(int)
    track = np.full((len(frames), peds.max() + 1, 2), np.nan)
    for f, p, x, y in raw:
        track[index[f], int(p)] = (x, y)
    obs_all, fut_all, hist_all, real_all = [], [], [], []
    win = OBS + PRED
    for t0 in range(0, len(frames) - win + 1):
        if frames[t0 + win - 1] - frames[t0] != step * (win - 1):
            continue                                   # skip windows across frame gaps
        seg = track[t0:t0 + win]                       # (20, P, 2)
        full = ~np.isnan(seg).any(axis=(0, 2))
        ids = np.nonzero(full)[0]
        for target in ids:
            now = seg[OBS - 1, target]
            vel = (seg[OBS - 1, target] - seg[OBS - 2, target]) / DT
            heading = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel) > 0.05 else 0.0
            c, s = np.cos(-heading), np.sin(-heading)
            rot = np.array([[c, -s], [s, c]])
            others = [j for j in ids if j != target]
            others.sort(key=lambda j: np.linalg.norm(seg[OBS - 1, j] - now))
            chosen = [target] + others[:num_agents - 1]
            local = (seg[:, chosen] - now) @ rot.T      # (20, k, 2)
            k = len(chosen)
            hist = np.zeros((OBS, num_agents, 2))
            fut = np.zeros((PRED, num_agents, 2))
            obs = np.zeros((num_agents, 7))
            obs[:, 6] = 1.0
            hist[:, :k] = local[:OBS]
            fut[:, :k] = local[OBS:]
            v = (local[OBS - 1] - local[OBS - 2]) / DT
            obs[:k, 0:2] = local[OBS - 1]
            obs[:k, 2:4] = v
            obs[:k, 4] = 1.0
            obs[:k, 5] = np.linalg.norm((local[OBS - 1] - local[0]) / ((OBS - 1) * DT), axis=-1)
            obs[:k, 6] = 0.0
            real = np.zeros(num_agents, dtype=bool)
            real[:k] = True
            obs_all.append(obs)
            fut_all.append(fut)
            hist_all.append(hist)
            real_all.append(real)
    return {"obs": torch.tensor(np.stack(obs_all), dtype=torch.float32),
            "future": torch.tensor(np.stack(fut_all), dtype=torch.float32),
            "history": torch.tensor(np.stack(hist_all), dtype=torch.float32),
            "real": torch.tensor(np.stack(real_all)), "dt": DT, "horizon": PRED, "num_agents": num_agents}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/ethucy/eth_ucy"))
    parser.add_argument("--out", type=Path, default=Path("data/ethucy"))
    parser.add_argument("--agents", type=int, default=10)
    args = parser.parse_args()
    for scene, name in SCENES.items():
        data = build(args.raw / name, args.agents)
        torch.save(data, args.out / f"{scene}.pt")
        n = data["obs"].shape[0]
        print(f"{scene:6s} {n:6d} samples, mean agents {float(data['real'].float().sum(1).mean()):.1f}")


if __name__ == "__main__":
    main()
