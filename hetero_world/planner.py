"""Deterministic selection of one joint plan from a set of candidate joint trajectories.

Every cooperative method ends with the same rule: score each candidate joint plan for safety and
progress and pick the minimiser. Because the rule is deterministic, agents that hold the same
candidate set (for example the same barycenter) select the same plan without further messages.
"""

from __future__ import annotations

import torch

from .config import CorridorConfig
from .geometry import wall_penetration


def score_joint_plans(plans: torch.Tensor, start_u: torch.Tensor, direction: torch.Tensor,
                      active: torch.Tensor, cfg: CorridorConfig, safety_margin: float = 0.2,
                      collision_weight: float = 1000.0, wall_weight: float = 1000.0) -> torch.Tensor:
    """Score candidate joint plans (lower is better).

    Args:
        plans: ``(..., K, H, M, 2)`` candidate joint trajectories.
        start_u: ``(..., M)`` current progress coordinate of each agent.
        direction: ``(..., M)`` travel direction of each agent.
        active: ``(..., M)`` agents that are still driving.
        cfg: corridor configuration.

    Returns:
        ``(..., K)`` scores: weighted squared intrusions into the safety distance and the walls,
        minus the total progress of active agents over the horizon.
    """
    lead = plans.shape[:-4]
    act = active.view(*lead, 1, 1, -1).to(plans.dtype)
    diff = plans.unsqueeze(-2) - plans.unsqueeze(-3)
    dist = diff.norm(dim=-1)
    pair = act.unsqueeze(-1) * act.unsqueeze(-2)
    num_agents = plans.shape[-2]
    eye = torch.eye(num_agents, dtype=plans.dtype, device=plans.device)
    pair = pair * (1.0 - eye)
    intrusion = (2.0 * cfg.agent_radius + safety_margin - dist).clamp_min(0.0)
    collision = 0.5 * (pair * intrusion.square()).sum(dim=(-1, -2, -3))
    wall = ((wall_penetration(plans, cfg) + 0.05).clamp_min(0.0).square() * act).sum(dim=(-1, -2))
    final_u = direction.view(*lead, 1, -1) * plans[..., -1, :, 0]
    progress = ((final_u - start_u.view(*lead, 1, -1)) * active.view(*lead, 1, -1).to(plans.dtype)).sum(-1)
    return collision_weight * collision + wall_weight * wall - progress


def select_joint_plan(plans: torch.Tensor, start_u: torch.Tensor, direction: torch.Tensor,
                      active: torch.Tensor, cfg: CorridorConfig) -> torch.Tensor:
    """Return the minimum-score plan ``(..., H, M, 2)`` from ``(..., K, H, M, 2)`` candidates.

    Scoring is chunked over the leading dimensions to bound the memory of the pairwise distances.
    """
    lead = plans.shape[:-4]
    num_cand, horizon, num_agents = plans.shape[-4:-1]
    flat_plans = plans.reshape(-1, num_cand, horizon, num_agents, 2)
    flat_u = start_u.reshape(-1, num_agents)
    flat_dir = direction.reshape(-1, num_agents)
    flat_act = active.reshape(-1, num_agents)
    per_row = num_cand * horizon * num_agents * num_agents
    chunk = max(1, int(2e7 // per_row))
    scores = torch.cat([
        score_joint_plans(flat_plans[i:i + chunk], flat_u[i:i + chunk], flat_dir[i:i + chunk],
                          flat_act[i:i + chunk], cfg)
        for i in range(0, flat_plans.shape[0], chunk)
    ]).view(*lead, num_cand)
    best = scores.argmin(dim=-1)
    index = best.view(*best.shape, 1, 1, 1, 1).expand(*best.shape, 1, *plans.shape[-3:])
    return plans.gather(-4, index).squeeze(-4)
