"""Tests for the simulator, dataset pipeline, plan selection and MPC tracker."""

from __future__ import annotations

import torch

from hetero_world.config import CorridorConfig, PredictionConfig
from hetero_world.controllers import advance_plan, free_path_reference
from hetero_world.dataset import HeteroTrajectoryDataset, collate_rollouts
from hetero_world.env import BottleneckCorridorEnv
from hetero_world.mpc import MPCTracker
from hetero_world.planner import select_joint_plan


def test_dataset_shape_and_types() -> None:
    ds = HeteroTrajectoryDataset(num_scenes=3, num_agents=10, num_rollouts=100, horizon=20)
    sample = ds[1]
    assert sample.rollouts.shape == (10, 100, 20, 20)
    assert set(sample.latent_dim.tolist()) <= {1024, 512, 128}
    batch = collate_rollouts([ds[0], ds[1]])
    assert batch.rollouts.shape == (2, 10, 100, 20, 20)
    assert torch.equal(ds[1].rollouts, sample.rollouts)


def test_rollouts_respect_single_file_gap_mostly() -> None:
    sample = HeteroTrajectoryDataset(num_scenes=1)[0]
    traj = sample.rollouts.view(10, 100, 20, 10, 2)
    cfg = CorridorConfig()
    inside = traj[..., 0].abs() < cfg.zone_half_length
    right = (inside & (sample.direction > 0)).any(-1)
    left = (inside & (sample.direction < 0)).any(-1)
    assert (right & left).float().mean() < 0.02


def test_env_deadlock_detection() -> None:
    cfg = CorridorConfig(num_agents=2)
    env = BottleneckCorridorEnv(cfg, 1)
    env.reset([0])
    zeros = torch.zeros(1, 2)
    for _ in range(cfg.freeze_steps + 5):
        env.step(zeros, zeros)
    assert bool(env.deadlocked[0]) and bool(env.over[0])


def test_env_collision_detection() -> None:
    cfg = CorridorConfig(num_agents=2)
    env = BottleneckCorridorEnv(cfg, 1)
    env.reset([0])
    env.pos[0, 0] = torch.tensor([0.0, 0.0])
    env.pos[0, 1] = torch.tensor([0.3, 0.0])
    env.step(torch.zeros(1, 2), torch.zeros(1, 2))
    assert bool(env.collided[0])


def test_mpc_tracks_free_path_when_alone() -> None:
    cfg = CorridorConfig(num_agents=1)
    env = BottleneckCorridorEnv(cfg, 1)
    env.reset([5])
    mpc = MPCTracker(cfg)
    while not bool(env.over.all()):
        v, w = mpc(env.pos, env.theta, env.speed, env.active, free_path_reference(env, mpc.horizon_steps))
        env.step(v, w)
    metrics = env.episode_metrics()
    assert bool(metrics["completed"][0]) and not bool(metrics["collision"][0])


def test_select_prefers_collision_free_plan() -> None:
    cfg = CorridorConfig(num_agents=2)
    horizon = 4
    safe = torch.zeros(horizon, 2, 2)
    safe[:, 0, 0] = torch.linspace(-4, -3, horizon)
    safe[:, 1, 0] = torch.linspace(4, 3, horizon)
    safe[:, 0, 1], safe[:, 1, 1] = 1.0, -1.0
    crash = safe.clone()
    crash[:, 1] = crash[:, 0] + 0.1
    plans = torch.stack([crash, safe]).unsqueeze(0)
    chosen = select_joint_plan(plans, torch.tensor([[-4.0, -4.0]]), torch.tensor([[1.0, -1.0]]),
                               torch.tensor([[True, True]]), cfg)
    assert torch.equal(chosen[0], safe)


def test_advance_plan_shifts_and_extrapolates() -> None:
    plan = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1, 1).expand(1, 1, 5, 1, 2).clone()
    advanced = advance_plan(plan, 2)
    assert torch.equal(advanced[0, 0, :, 0, 0], torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
    assert PredictionConfig().horizon > 2
