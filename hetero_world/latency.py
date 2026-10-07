"""Per-agent consensus computations used for latency measurements.

Each function returns a zero-argument closure that performs exactly the computation one receiving
agent runs after a broadcast round: fuse what it received and select a joint plan (including the
temporal-consistency term against a previous plan). World-model inference and MPC tracking are
excluded because they are identical across methods.
"""

from __future__ import annotations

from typing import Callable

import torch

from .config import ConsensusConfig, CorridorConfig, PredictionConfig
from .controllers import PlanConsensusController, consistency_cost
from .planner import score_joint_plans
from .sinkhorn import SinkhornBarycenterConsensus, pairwise_sinkhorn_costs


def _select(candidates: torch.Tensor, previous: torch.Tensor, start_u: torch.Tensor, direction: torch.Tensor,
            active: torch.Tensor, cfg: CorridorConfig, pcfg: PredictionConfig) -> torch.Tensor:
    """Score ``(K, H, M, 2)`` candidates and return the selected plan."""
    shift = int(round(pcfg.replan_every * cfg.dt / pcfg.plan_dt))
    scores = score_joint_plans(candidates.unsqueeze(0), start_u.unsqueeze(0), direction.unsqueeze(0),
                               active.unsqueeze(0), cfg)
    scores = scores + pcfg.consistency_weight * consistency_cost(
        candidates.view(1, 1, *candidates.shape), previous.view(1, 1, *previous.shape), active.unsqueeze(0), shift)[0]
    return candidates[scores.argmin()]


def consensus_closure(method: str, controller: PlanConsensusController, preds: torch.Tensor, agent_type: torch.Tensor,
                      start_u: torch.Tensor, direction: torch.Tensor, active: torch.Tensor,
                      observer: int = 0) -> Callable[[], torch.Tensor]:
    """Closure performing one agent's fusion + selection.

    Args:
        preds: ``(M, N, H, M, 2)`` rollouts of every agent in one scene.
        agent_type: ``(M,)`` model family of every agent.
        start_u, direction, active: ``(M,)`` scene state.
        observer: index of the receiving agent.
    """
    cfg, pcfg = controller.cfg, controller.pcfg
    num_agents, num_rollouts, horizon = preds.shape[:3]
    previous = preds[observer, 0]
    flat = preds.reshape(num_agents, num_rollouts, -1)

    if method == "hetero_world":
        module = controller.consensus

        def run() -> torch.Tensor:
            support = module(flat).support.view(num_rollouts, horizon, num_agents, 2)
            return _select(support, previous, start_u, direction, active, cfg, pcfg)
        return run

    if method == "union_select":
        union = preds.reshape(num_agents * num_rollouts, horizon, num_agents, 2)
        return lambda: _select(union, previous, start_u, direction, active, cfg, pcfg)

    if method == "independent":
        own = preds[observer]
        return lambda: _select(own, previous, start_u, direction, active, cfg, pcfg)

    snapshots = preds.reshape(num_agents, num_rollouts, horizon, -1)
    latents = []
    for j in range(num_agents):
        codec = controller.codecs[int(agent_type[j])]
        latents.append(codec.encode(snapshots[j]).half().float())

    if method == "feature_sharing":
        def run() -> torch.Tensor:
            decoded = torch.stack([
                controller.codecs[int(agent_type[j])].decode(latents[j]) for j in range(num_agents)
            ]).view(num_agents, num_rollouts, horizon, num_agents, 2)
            fused = decoded.mean(0)
            return _select(fused, previous, start_u, direction, active, cfg, pcfg)
        return run

    if method == "procrustes":
        receiver = int(agent_type[observer])
        target = controller.codecs[receiver]

        def run() -> torch.Tensor:
            sets = []
            for j in range(num_agents):
                if j == observer:
                    sets.append(preds[observer])
                    continue
                rot, mean_s, mean_t = controller.procrustes[(int(agent_type[j]), receiver)]
                mapped = (latents[j] - mean_s) @ rot + mean_t
                sets.append(target.decode(mapped).view(num_rollouts, horizon, num_agents, 2))
            union = torch.cat(sets, dim=0)
            return _select(union, previous, start_u, direction, active, cfg, pcfg)
        return run

    raise ValueError(f"no consensus computation for {method}")


def pairwise_closure(preds: torch.Tensor, ccfg: ConsensusConfig) -> Callable[[], torch.Tensor]:
    """Closure computing all ``M(M-1)/2`` pairwise entropic transport costs (the O(M^2) alternative)."""
    flat = preds.reshape(1, preds.shape[0], preds.shape[1], -1)
    return lambda: pairwise_sinkhorn_costs(flat, ccfg.epsilon, ccfg.max_iters)


def barycenter_closure(measures: torch.Tensor, ccfg: ConsensusConfig) -> Callable[[], torch.Tensor]:
    """Closure computing only the barycenter of ``(M, N, D)`` measures."""
    module = SinkhornBarycenterConsensus(ccfg.epsilon, ccfg.max_iters, ccfg.outer_iters)
    return lambda: module(measures).support
