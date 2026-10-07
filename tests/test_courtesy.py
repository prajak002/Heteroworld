"""Benchmark v2 (doorway courtesy): pedestrians alone never deadlock, yields chain down a queue."""

import dataclasses

import torch

from hetero_world.benchmark import EVAL_SEED_BASE
from hetero_world.config import CorridorConfig
from hetero_world.env import BottleneckCorridorEnv, courtesy_yield


def _run_pedestrians(courtesy: bool, episodes: int = 40) -> dict:
    cfg = dataclasses.replace(CorridorConfig(num_agents=10, num_humans=10), courtesy=courtesy)
    env = BottleneckCorridorEnv(cfg, episodes)
    env.reset([EVAL_SEED_BASE + k for k in range(episodes)])
    zeros = torch.zeros(episodes, cfg.num_agents)
    while not bool(env.over.all()):
        env.step(zeros, zeros)
    return env.episode_metrics()


def test_pedestrians_alone_complete_without_deadlock():
    metrics = _run_pedestrians(courtesy=True)
    assert not bool(metrics["deadlock"].any())
    assert bool(metrics["completed"].all())
    assert not bool(metrics["collision"].any())


def test_shallower_agent_and_its_queue_yield():
    cfg = CorridorConfig(num_agents=3)
    # Agent 0 (+x) is inside the gap; agent 1 (-x) faces it from shallower depth; agent 2 queues behind 1.
    pos = torch.tensor([[[0.5, 0.0], [1.3, 0.0], [2.1, 0.0]]])
    direction = torch.tensor([[1.0, -1.0, -1.0]])
    active = torch.ones(1, 3, dtype=torch.bool)
    assert courtesy_yield(pos, direction, active, cfg).tolist() == [[False, True, True]]


def test_simulate_humans_shape_and_robots_follow_plan():
    cfg = CorridorConfig(num_agents=10, num_humans=4)
    env = BottleneckCorridorEnv(cfg, 2)
    env.reset([EVAL_SEED_BASE, EVAL_SEED_BASE + 1])
    plan = env.pos.unsqueeze(1).repeat(1, 20, 1, 1)
    out = env.simulate_humans(plan, plan_dt=0.25)
    assert out.shape == (2, 20, 10, 2)
    robots = ~env.is_human
    assert torch.allclose(out[:, -1][robots], plan[:, -1][robots])
