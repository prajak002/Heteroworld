"""Sampling-based model predictive tracker that turns a reference trajectory into (v, omega) commands."""

from __future__ import annotations

import torch

from .config import CorridorConfig
from .geometry import lane_reference_y, wall_penetration


class MPCTracker:
    """Receding-horizon tracker over constant (speed, yaw-rate) motion primitives.

    At every step each agent evaluates ``num_speeds * num_turns`` primitives held for
    ``horizon_steps`` simulator steps under the true unicycle dynamics (including the acceleration
    limit), scores them, and applies the first command of the best one. The score combines

    * tracking: mean squared distance to the reference positions;
    * agent avoidance: squared intrusion into the safety distance around every other active agent,
      whose motion is extrapolated at constant velocity (reactive, non-cooperative), plus a large
      penalty for any predicted overlap;
    * wall avoidance: squared intrusion into a wall margin plus a large penalty for overlap;
    * a small yaw-rate effort term.

    Args:
        cfg: corridor configuration (dynamics limits, radius, time step).
        horizon_steps: prediction horizon in simulator steps.
        num_speeds, num_turns: grid resolution of the primitive library.
    """

    def __init__(self, cfg: CorridorConfig, horizon_steps: int = 10, num_speeds: int = 7, num_turns: int = 9,
                 track_weight: float = 1.0, collision_weight: float = 50.0, overlap_weight: float = 1000.0,
                 wall_weight: float = 200.0, effort_weight: float = 0.01, safety_margin: float = 0.15) -> None:
        self.cfg = cfg
        self.horizon_steps = horizon_steps
        speeds = torch.linspace(0.0, cfg.v_max, num_speeds)
        if cfg.courtesy:
            speeds = torch.cat([torch.tensor([-cfg.v_reverse]), speeds])
        self.v_min = -cfg.v_reverse if cfg.courtesy else 0.0
        turns = torch.linspace(-cfg.omega_max, cfg.omega_max, num_turns)
        grid_v, grid_w = torch.meshgrid(speeds, turns, indexing="ij")
        self.prim_v = grid_v.reshape(-1)
        self.prim_w = grid_w.reshape(-1)
        self.track_weight = track_weight
        self.collision_weight = collision_weight
        self.overlap_weight = overlap_weight
        self.wall_weight = wall_weight
        self.effort_weight = effort_weight
        self.safety_distance = 2.0 * cfg.agent_radius + safety_margin

    def rollout(self, pos: torch.Tensor, theta: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Simulate every primitive from the current state; returns ``(B, M, K, T, 2)`` positions."""
        cfg = self.cfg
        prim_v = self.prim_v.to(pos)
        prim_w = self.prim_w.to(pos)
        p = pos.unsqueeze(2).expand(-1, -1, prim_v.numel(), -1)
        th = theta.unsqueeze(2).expand_as(p[..., 0])
        v = speed.unsqueeze(2).expand_as(th)
        out = []
        for _ in range(self.horizon_steps):
            v = (v + (prim_v - v).clamp(-cfg.accel_max * cfg.dt, cfg.accel_max * cfg.dt)).clamp(self.v_min, cfg.v_max)
            th = th + prim_w * cfg.dt
            p = p + v.unsqueeze(-1) * torch.stack([torch.cos(th), torch.sin(th)], dim=-1) * cfg.dt
            out.append(p)
        return torch.stack(out, dim=3)

    def __call__(self, pos: torch.Tensor, theta: torch.Tensor, speed: torch.Tensor, active: torch.Tensor,
                 reference: torch.Tensor, direction: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute commands.

        Args:
            pos, theta, speed: ``(B, M, 2)``, ``(B, M)``, ``(B, M)`` current state.
            active: ``(B, M)`` agents that are still driving.
            reference: ``(B, M, T, 2)`` reference positions at the next ``T = horizon_steps`` steps.
            direction: ``(B, M)`` travel directions; with ``cfg.courtesy`` the doorway-courtesy reflex
                replaces the reference of yielding agents by a slow retreat along their own lane.

        Returns:
            ``v_cmd`` and ``omega`` of shape ``(B, M)``.
        """
        cfg = self.cfg
        num_agents = pos.shape[1]
        if cfg.courtesy and direction is not None:
            from .env import courtesy_yield
            yielding = courtesy_yield(pos, direction, active, cfg)
            k = torch.arange(1, self.horizon_steps + 1, dtype=pos.dtype, device=pos.device)
            back_x = pos[..., :1] - direction.unsqueeze(-1) * cfg.v_reverse * cfg.dt * k
            back = torch.stack([back_x, lane_reference_y(back_x, direction.unsqueeze(-1), cfg)], dim=-1)
            reference = torch.where(yielding.view(*yielding.shape, 1, 1), back, reference)
        traj = self.rollout(pos, theta, speed)
        track = (traj - reference.unsqueeze(2)).square().sum(-1).mean(-1)

        steps = cfg.dt * torch.arange(1, self.horizon_steps + 1, dtype=pos.dtype, device=pos.device)
        velocity = speed.unsqueeze(-1) * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        others = pos.unsqueeze(2) + velocity.unsqueeze(2) * steps.view(1, 1, -1, 1)
        diff = traj.unsqueeze(-2) - others.permute(0, 2, 1, 3).unsqueeze(1).unsqueeze(1)
        dist = diff.norm(dim=-1)
        eye = torch.eye(num_agents, dtype=torch.bool, device=pos.device)
        valid = active.unsqueeze(1) & ~eye.unsqueeze(0)
        valid = valid.view(valid.shape[0], num_agents, 1, 1, num_agents)
        intrusion = torch.where(valid, (self.safety_distance - dist).clamp_min(0.0), torch.zeros_like(dist))
        overlap = torch.where(valid, (dist < 2.0 * cfg.agent_radius + 0.02).to(dist.dtype), torch.zeros_like(dist))
        collision = self.collision_weight * intrusion.square().sum(dim=(-1, -2)) + \
            self.overlap_weight * overlap.sum(dim=(-1, -2))

        pen = wall_penetration(traj, cfg)
        wall = self.wall_weight * (pen + 0.05).clamp_min(0.0).square().sum(-1) + \
            self.overlap_weight * (pen > 0).to(pen.dtype).sum(-1)
        effort = self.effort_weight * self.prim_w.to(pos).square()

        cost = self.track_weight * track + collision + wall + effort
        best = cost.argmin(dim=-1)
        v_cmd = self.prim_v.to(pos)[best]
        omega = self.prim_w.to(pos)[best]
        v_cmd = torch.where(active, v_cmd, torch.zeros_like(v_cmd))
        omega = torch.where(active, omega, torch.zeros_like(omega))
        return v_cmd, omega
