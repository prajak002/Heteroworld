"""PyTorch dataset of heterogeneous joint-trajectory rollouts in the bottleneck corridor.

Each item is one bottleneck scene. Every agent ``i`` of the scene runs its private world model and
emits ``N`` rollouts of the joint future. The rollouts of agent ``i`` define the empirical measure

    mu_i = (1/N) sum_n delta_{x_i^(n)},     x_i^(n) in (R^2)^{H x M}.

All measures live on the same physical trajectory space regardless of the latent dimension of the
model that produced them. Items are returned with shape ``(M, N, H, D)`` where ``D = 2 M`` is the
dimension of the joint workspace (x and y of every agent at one time step).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .config import CorridorConfig, PredictionConfig
from .env import BottleneckCorridorEnv
from .world_models import WORLD_MODEL_NAMES, WORLD_MODEL_SPECS, HeterogeneousWorldModelFleet, scene_from_env


@dataclass
class RolloutSample:
    """One scene worth of rollouts.

    Attributes:
        rollouts: ``(M, N, H, 2M)`` joint trajectories predicted by each agent's world model.
        agent_type: ``(M,)`` world-model family index (0 diffusion, 1 jepa, 2 rssm).
        latent_dim: ``(M,)`` latent dimension of each agent's world model.
        initial_pos: ``(M, 2)`` observed positions.
        direction: ``(M,)`` travel direction (+1 or -1).
    """

    rollouts: torch.Tensor
    agent_type: torch.Tensor
    latent_dim: torch.Tensor
    initial_pos: torch.Tensor
    direction: torch.Tensor


class HeteroTrajectoryDataset(Dataset):
    """Deterministic dataset of multi-agent bottleneck rollouts.

    Args:
        num_scenes: number of scenes (dataset length).
        num_agents: agents per scene ``M``.
        num_rollouts: rollouts per agent ``N``.
        horizon: plan steps ``H``.
        warmup_steps: simulator steps of constant-velocity driving before the snapshot, so that
            scenes include agents close to the gap rather than only the spawn configuration.
        seed: base seed; scene ``k`` uses seed ``seed + k``.
    """

    def __init__(self, num_scenes: int = 100, num_agents: int = 10, num_rollouts: int = 100, horizon: int = 20,
                 warmup_steps: int = 0, seed: int = 0) -> None:
        self.cfg = CorridorConfig(num_agents=num_agents)
        self.pcfg = PredictionConfig(num_rollouts=num_rollouts, horizon=horizon)
        self.num_scenes = num_scenes
        self.warmup_steps = warmup_steps
        self.seed = seed
        self.fleet = HeterogeneousWorldModelFleet(self.cfg, self.pcfg, seed=seed)

    def __len__(self) -> int:
        return self.num_scenes

    def __getitem__(self, index: int) -> RolloutSample:
        if not 0 <= index < self.num_scenes:
            raise IndexError(index)
        scene_seed = self.seed + index
        env = BottleneckCorridorEnv(self.cfg, batch_size=1)
        env.reset([scene_seed])
        for _ in range(self.warmup_steps):
            env.step(env.v_des.clone(), torch.zeros_like(env.v_des))
        gen = torch.Generator().manual_seed(10_000 + scene_seed)
        joint = self.fleet.predict(scene_from_env(env), env.agent_type, gen)[0]
        num_agents = self.cfg.num_agents
        rollouts = joint.reshape(num_agents, self.pcfg.num_rollouts, self.pcfg.horizon, 2 * num_agents)
        latent = torch.tensor([WORLD_MODEL_SPECS[WORLD_MODEL_NAMES[t]].latent_dim for t in env.agent_type[0].tolist()])
        return RolloutSample(rollouts=rollouts, agent_type=env.agent_type[0].clone(), latent_dim=latent,
                             initial_pos=env.pos[0].clone(), direction=env.direction[0].clone())


def collate_rollouts(batch: list[RolloutSample]) -> RolloutSample:
    """Stack samples into a batch with a leading scene dimension."""
    return RolloutSample(
        rollouts=torch.stack([b.rollouts for b in batch]),
        agent_type=torch.stack([b.agent_type for b in batch]),
        latent_dim=torch.stack([b.latent_dim for b in batch]),
        initial_pos=torch.stack([b.initial_pos for b in batch]),
        direction=torch.stack([b.direction for b in batch]),
    )
