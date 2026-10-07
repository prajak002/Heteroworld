"""Vectorised two-lane bottleneck corridor simulator with unicycle agents.

Every tensor carries a leading batch dimension ``B`` (independent episodes) followed by the agent
dimension ``M``. Agents ``0 .. num_right - 1`` start on the left and move in +x; the remaining agents
start on the right and move in -x. All opposing traffic must share a single-file gap at ``x = 0``.

Episode-level metrics tracked here:

* collision: any agent-agent disc overlap or agent-wall overlap deeper than ``collision_tolerance``;
* deadlock: every still-active agent has speed below ``freeze_speed`` for more than
  ``freeze_duration`` seconds (a global spatial freeze). A deadlocked episode is terminated;
* completion: every agent reached its goal before the time limit.

Mixed traffic: when ``cfg.num_humans > 0`` a random subset of agents are uncontrolled pedestrians.
Each pedestrian follows a ground-truth trajectory sampled at reset from the unbiased passing prior
(no ego bias, ``cfg.human_order_noise`` on the passing order). Pedestrians ignore the robots' plans
and never negotiate the gap; they only stop when another agent (same or oncoming direction) is directly
ahead of them, and catch up with their reference afterwards. Their commands passed to :meth:`step` are ignored.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from .config import CorridorConfig
from .geometry import lane_reference_y, wall_penetration

NUM_WORLD_MODEL_TYPES = 3


@dataclass
class StepInfo:
    """Per-step outputs that learning-based controllers need for reward shaping."""

    progress: torch.Tensor
    agent_collision: torch.Tensor
    wall_collision: torch.Tensor
    reached_goal: torch.Tensor


class BottleneckCorridorEnv:
    """Batched simulator of ``B`` independent bottleneck episodes with ``M`` agents each."""

    def __init__(self, cfg: CorridorConfig, batch_size: int, device: torch.device | str = "cpu",
                 terminate_on_deadlock: bool = True) -> None:
        self.cfg = cfg
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.terminate_on_deadlock = terminate_on_deadlock
        num_agents = cfg.num_agents
        direction = torch.ones(num_agents, device=self.device)
        direction[cfg.num_right:] = -1.0
        self.direction = direction.expand(batch_size, num_agents).clone()
        self.pos = torch.zeros(batch_size, num_agents, 2, device=self.device)
        self.theta = torch.zeros(batch_size, num_agents, device=self.device)
        self.speed = torch.zeros(batch_size, num_agents, device=self.device)
        self.v_des = torch.zeros(batch_size, num_agents, device=self.device)
        self.agent_type = torch.zeros(batch_size, num_agents, dtype=torch.long, device=self.device)
        self.done = torch.zeros(batch_size, num_agents, dtype=torch.bool, device=self.device)
        self.finish_step = torch.full((batch_size, num_agents), -1, dtype=torch.long, device=self.device)
        self.collided = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.collision_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.deadlocked = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.freeze_counter = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.over = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.min_separation = torch.full((batch_size,), float("inf"), device=self.device)
        self.hit_rr = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.hit_rh = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.hit_hh = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.hit_wall = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.is_human = torch.zeros(batch_size, num_agents, dtype=torch.bool, device=self.device)
        self.human_ref = None
        self.assertive = torch.zeros(batch_size, num_agents, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------------------------------
    def reset(self, seeds: list[int]) -> None:
        """Reset every episode; ``seeds[b]`` fully determines the initial condition of episode b."""
        if len(seeds) != self.batch_size:
            raise ValueError(f"expected {self.batch_size} seeds, got {len(seeds)}")
        self.reset_indices(torch.arange(self.batch_size, device=self.device), seeds)

    def reset_indices(self, indices: torch.Tensor, seeds: list[int]) -> None:
        """Reset only the episodes in ``indices`` using the matching ``seeds``."""
        cfg = self.cfg
        for b, seed in zip(indices.tolist(), seeds):
            gen = torch.Generator().manual_seed(int(seed))
            pos, theta, speed, v_des, agent_type = self._sample_initial_state(gen)
            self.pos[b] = pos.to(self.device)
            self.theta[b] = theta.to(self.device)
            self.speed[b] = speed.to(self.device)
            self.v_des[b] = v_des.to(self.device)
            self.agent_type[b] = agent_type.to(self.device)
            if cfg.num_humans > 0:
                humans = torch.randperm(cfg.num_agents, generator=gen)[:cfg.num_humans]
                self.is_human[b] = False
                self.is_human[b, humans] = True
                if self.human_ref is None:
                    self.human_ref = torch.zeros(self.batch_size, cfg.max_steps + 2, cfg.num_agents, 2,
                                                 device=self.device)
                self.human_ref[b] = self._human_reference(b, gen)
                # Hidden trait (benchmark v2): assertive pedestrians enter the gap without waiting.
                self.assertive[b] = torch.rand(cfg.num_agents, generator=gen) < cfg.assertive_prob
        self.done[indices] = False
        self.finish_step[indices] = -1
        self.collided[indices] = False
        self.hit_rr[indices] = False
        self.hit_rh[indices] = False
        self.hit_hh[indices] = False
        self.hit_wall[indices] = False
        self.collision_steps[indices] = 0
        self.deadlocked[indices] = False
        self.freeze_counter[indices] = 0
        self.over[indices] = False
        self.steps[indices] = 0
        self.min_separation[indices] = float("inf")
        if cfg.num_agents < 1:
            raise ValueError("the corridor needs at least one agent")

    def _human_reference(self, b: int, gen: torch.Generator) -> torch.Tensor:
        """Ground-truth pedestrian trajectories ``(T, M, 2)`` from the unbiased passing prior."""
        from .config import PredictionConfig
        from .world_models import (SceneState, WorldModelSpec, progress_to_positions, sample_passing_schedule,
                                   schedule_to_progress)
        cfg = self.cfg
        pcfg = PredictionConfig(num_rollouts=1, horizon=cfg.max_steps + 1, plan_dt=cfg.dt)
        spec = WorldModelSpec("truth", 0, order_noise=cfg.human_order_noise, ego_bias=0.0,
                              direction_bias=0.0, speed_noise=0.0)
        scene = SceneState(self.pos[b:b + 1].cpu(), self.direction[b:b + 1].cpu(), self.v_des[b:b + 1].cpu(),
                           torch.zeros(1, cfg.num_agents, dtype=torch.bool), self.speed[b:b + 1].cpu())
        speed = scene.v_des.unsqueeze(1)
        t_in, hold_u = sample_passing_schedule(scene, torch.zeros(1, dtype=torch.long), spec, speed, cfg, pcfg, gen)
        u = schedule_to_progress(scene, t_in, hold_u, speed, cfg, pcfg)
        pos = progress_to_positions(scene, u, cfg)[0, 0]
        return torch.cat([scene.pos, pos], dim=0).to(self.device)

    @torch.no_grad()
    def simulate_humans(self, robot_future: torch.Tensor, plan_dt: float) -> torch.Tensor:
        """Privileged forecast: pedestrian positions at plan times if robots follow ``robot_future``.

        ``robot_future`` is ``(B, H, M, 2)`` (robot entries are used) at times ``plan_dt * (1..H)``.
        Robots are moved along it with linear interpolation; pedestrians follow the true reactive
        rules of this simulator. Returns ``(B, H, M, 2)`` with robots copied from ``robot_future``.
        """
        cfg = self.cfg
        horizon = robot_future.shape[1]
        stride = int(round(plan_dt / cfg.dt))
        sim = copy.copy(self)
        sim.pos, sim.speed, sim.theta = self.pos.clone(), self.speed.clone(), self.theta.clone()
        waypoints = torch.cat([self.pos.unsqueeze(1), robot_future.to(self.pos)], dim=1)
        active = self.active
        out = []
        for t in range(1, horizon * stride + 1):
            k, w = divmod(t, stride)
            robot_pos = waypoints[:, k] if w == 0 else waypoints[:, k] + (w / stride) * (waypoints[:, k + 1] - waypoints[:, k])
            new_pos, new_speed, new_theta = sim._human_motion(robot_pos, sim.speed, sim.theta, active)
            human = (self.is_human & active).unsqueeze(-1)
            sim.pos = torch.where(human, new_pos, robot_pos)
            sim.speed, sim.theta = new_speed, new_theta
            if w == 0:
                out.append(sim.pos.clone())
        return torch.stack(out, dim=1)

    def _sample_initial_state(self, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
        """Sample queue positions, headings, speeds, desired speeds and world-model types."""
        cfg = self.cfg

        def uniform(n: int, low: float, high: float) -> torch.Tensor:
            return low + (high - low) * torch.rand(n, generator=gen)

        def queue(count: int) -> torch.Tensor:
            if count == 0:
                return torch.zeros(0)
            spacing = uniform(count, cfg.spawn_spacing_min, cfg.spawn_spacing_max)
            spacing[0] = 0.0
            first = cfg.spawn_clearance + uniform(1, 0.0, 0.5)
            return first + torch.cumsum(spacing, dim=0)

        right_u = queue(cfg.num_right)
        left_u = queue(cfg.num_left)
        x = torch.cat([-right_u, left_u])
        lane = torch.cat([torch.full((cfg.num_right,), cfg.lane_offset),
                          torch.full((cfg.num_left,), -cfg.lane_offset)])
        y = lane + uniform(cfg.num_agents, -0.1, 0.1)
        heading = torch.cat([torch.zeros(cfg.num_right), torch.full((cfg.num_left,), torch.pi)])
        theta = heading + uniform(cfg.num_agents, -0.05, 0.05)
        v_des = uniform(cfg.num_agents, cfg.v_des_min, cfg.v_des_max)
        speed = v_des * uniform(cfg.num_agents, 0.6, 1.0)
        types = torch.arange(cfg.num_agents) % NUM_WORLD_MODEL_TYPES
        agent_type = types[torch.randperm(cfg.num_agents, generator=gen)]
        return torch.stack([x, y], dim=-1), theta, speed, v_des, agent_type

    # ------------------------------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------------------------------
    @property
    def active(self) -> torch.Tensor:
        """Agents that are still driving in episodes that are still running."""
        return (~self.done) & (~self.over).unsqueeze(-1)

    @property
    def progress_coordinate(self) -> torch.Tensor:
        """Signed progress ``u = direction * x`` (increases towards each agent's goal)."""
        return self.direction * self.pos[..., 0]

    def step(self, v_cmd: torch.Tensor, omega: torch.Tensor) -> StepInfo:
        """Advance every running episode by one step of length ``dt``.

        Args:
            v_cmd: commanded forward speed, shape ``(B, M)``; clipped to ``[0, v_max]``.
            omega: commanded yaw rate, shape ``(B, M)``; clipped to ``[-omega_max, omega_max]``.
        """
        cfg = self.cfg
        active = self.active
        u_before = self.progress_coordinate
        v_min = -cfg.v_reverse if cfg.courtesy else 0.0
        v_cmd = v_cmd.clamp(v_min, cfg.v_max)
        omega = omega.clamp(-cfg.omega_max, cfg.omega_max)
        dv = (v_cmd - self.speed).clamp(-cfg.accel_max * cfg.dt, cfg.accel_max * cfg.dt)
        new_speed = (self.speed + dv).clamp(v_min, cfg.v_max)
        new_theta = self.theta + omega * cfg.dt
        heading = torch.stack([torch.cos(new_theta), torch.sin(new_theta)], dim=-1)
        new_pos = self.pos + new_speed.unsqueeze(-1) * heading * cfg.dt
        if self.cfg.num_humans > 0:
            new_pos, new_speed, new_theta = self._human_motion(new_pos, new_speed, new_theta, active)
        mask2 = active.unsqueeze(-1)
        self.speed = torch.where(active, new_speed, self.speed)
        self.theta = torch.where(active, new_theta, self.theta)
        self.pos = torch.where(mask2, new_pos, self.pos)
        progress = torch.where(active, self.progress_coordinate - u_before, torch.zeros_like(u_before))

        reached = active & (self.progress_coordinate >= cfg.goal_distance)
        self.finish_step = torch.where(reached, self.steps.unsqueeze(-1) + 1, self.finish_step)
        self.done = self.done | reached
        self._park(reached)

        still_active = self.active
        agent_hit, wall_hit, min_sep = self._collisions(still_active)
        running = ~self.over
        pair = self._last_overlap
        h = self.is_human
        self.hit_rr |= (pair & ~h.unsqueeze(1) & ~h.unsqueeze(2)).flatten(1).any(-1) & running
        self.hit_rh |= (pair & (h.unsqueeze(1) ^ h.unsqueeze(2))).flatten(1).any(-1) & running
        self.hit_hh |= (pair & h.unsqueeze(1) & h.unsqueeze(2)).flatten(1).any(-1) & running
        self.hit_wall |= wall_hit.any(-1) & running
        step_collision = (agent_hit | wall_hit).any(dim=-1) & running
        self.collided = self.collided | step_collision
        self.collision_steps = self.collision_steps + step_collision.long()
        self.min_separation = torch.where(running, torch.minimum(self.min_separation, min_sep),
                                          self.min_separation)

        frozen = (still_active & (self.speed.abs() >= cfg.freeze_speed)).sum(dim=-1) == 0
        frozen = frozen & still_active.any(dim=-1) & running
        self.freeze_counter = torch.where(frozen, self.freeze_counter + 1, torch.zeros_like(self.freeze_counter))
        newly_deadlocked = (self.freeze_counter > cfg.freeze_steps) & running
        self.deadlocked = self.deadlocked | newly_deadlocked

        self.steps = torch.where(running, self.steps + 1, self.steps)
        finished = ~(~self.done).any(dim=-1)
        timeout = self.steps >= cfg.max_steps
        ending = finished | timeout
        if self.terminate_on_deadlock:
            ending = ending | self.deadlocked
        self.over = self.over | (ending & running)
        return StepInfo(progress=progress, agent_collision=agent_hit, wall_collision=wall_hit,
                        reached_goal=reached)

    def _human_motion(self, new_pos: torch.Tensor, new_speed: torch.Tensor, new_theta: torch.Tensor,
                      active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Overwrite the motion of pedestrians: move towards the reference, stop if blocked ahead."""
        cfg = self.cfg
        arange = torch.arange(self.batch_size, device=self.device)
        if cfg.courtesy:
            # Benchmark v2: reactive path follower along the own lane at the desired speed.
            u_next = (self.progress_coordinate + self.v_des * cfg.dt).clamp(max=cfg.goal_distance + 1.0)
            x_next = self.direction * u_next
            target = torch.stack([x_next, lane_reference_y(x_next, self.direction, cfg)], dim=-1)
        else:
            index = (self.steps + 1).clamp(max=self.human_ref.shape[1] - 1)
            target = self.human_ref[arange, index]
        delta = target - self.pos
        norm = delta.norm(dim=-1, keepdim=True)
        delta = delta * (cfg.v_max * cfg.dt / norm.clamp_min(1e-9)).clamp(max=1.0)
        # Blocked: another active agent (same or oncoming direction) less than 2r + 0.35 m ahead in the
        # walking direction and within 2r + 0.1 m laterally. Pedestrians brake for what is in front of
        # them but never negotiate the gap.
        rel = self.pos.unsqueeze(1) - self.pos.unsqueeze(2)
        ahead = rel[..., 0] * self.direction.unsqueeze(-1)
        other = active.unsqueeze(1) & ~torch.eye(cfg.num_agents, dtype=torch.bool, device=self.device)
        blocked = (other & (ahead > 0) & (ahead < 2 * cfg.agent_radius + 0.35)
                           & (rel[..., 1].abs() < 2 * cfg.agent_radius + 0.1)).any(-1)
        human = self.is_human & active
        if cfg.courtesy:
            yielding = courtesy_yield(self.pos, self.direction, active, cfg)
            waiting = entry_wait(self.pos, target, self.direction, active, cfg) & ~self.assertive
            stop = blocked | waiting
            delta = torch.where(stop.unsqueeze(-1), torch.zeros_like(delta), delta)
            # Back off along the own lane, unless someone is right behind.
            behind = (other & (ahead < 0) & (ahead > -(2 * cfg.agent_radius + 0.1))
                      & (rel[..., 1].abs() < 2 * cfg.agent_radius + 0.1) & ~yielding.unsqueeze(1)).any(-1)
            back_x = self.pos[..., 0] - self.direction * cfg.v_reverse * cfg.dt
            back = torch.stack([back_x, lane_reference_y(back_x, self.direction, cfg)], dim=-1) - self.pos
            back = back * (cfg.v_reverse * cfg.dt / back.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
            retreat = yielding & ~behind
            delta = torch.where(retreat.unsqueeze(-1), back, delta)
        else:
            delta = torch.where(blocked.unsqueeze(-1), torch.zeros_like(delta), delta)
        step = delta.norm(dim=-1)
        theta_h = torch.where(step > 1e-6, torch.atan2(delta[..., 1], delta[..., 0]), self.theta)
        if cfg.courtesy:
            # A retreating pedestrian keeps facing forward and walks backwards.
            theta_h = torch.where(yielding & (step > 1e-6), torch.atan2(-delta[..., 1], -delta[..., 0]), theta_h)
            step = torch.where(yielding, -step, step)
        new_pos = torch.where(human.unsqueeze(-1), self.pos + delta, new_pos)
        new_speed = torch.where(human, step / cfg.dt, new_speed)
        new_theta = torch.where(human, theta_h, new_theta)
        return new_pos, new_speed, new_theta

    def _park(self, mask: torch.Tensor) -> None:
        """Move agents that reached their goal far away so they no longer interact."""
        if not bool(mask.any()):
            return
        num_agents = self.cfg.num_agents
        index = torch.arange(num_agents, device=self.device, dtype=self.pos.dtype).expand_as(self.speed)
        park_x = self.direction * (self.cfg.half_length + 50.0 + 2.0 * index)
        parked = torch.stack([park_x, torch.zeros_like(park_x)], dim=-1)
        self.pos = torch.where(mask.unsqueeze(-1), parked, self.pos)
        self.speed = torch.where(mask, torch.zeros_like(self.speed), self.speed)

    def _collisions(self, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-agent agent-agent hits, per-agent wall hits and per-episode min separation."""
        cfg = self.cfg
        diff = self.pos.unsqueeze(2) - self.pos.unsqueeze(1)
        dist = diff.norm(dim=-1)
        pair_valid = active.unsqueeze(2) & active.unsqueeze(1)
        eye = torch.eye(cfg.num_agents, dtype=torch.bool, device=self.device)
        pair_valid = pair_valid & ~eye
        dist = torch.where(pair_valid, dist, torch.full_like(dist, float("inf")))
        overlap = dist < (2.0 * cfg.agent_radius - cfg.collision_tolerance)
        self._last_overlap = overlap
        agent_hit = overlap.any(dim=-1)
        min_sep = dist.flatten(1).min(dim=-1).values
        wall_hit = (wall_penetration(self.pos, cfg) > cfg.collision_tolerance) & active
        return agent_hit, wall_hit, min_sep

    # ------------------------------------------------------------------------------------------
    # Episode statistics
    # ------------------------------------------------------------------------------------------
    def episode_metrics(self) -> dict[str, torch.Tensor]:
        """Per-episode outcome tensors (valid once every episode is over)."""
        completed = self.done.all(dim=-1)
        finish_time = torch.where(self.finish_step >= 0, self.finish_step.float() * self.cfg.dt,
                                  torch.full_like(self.finish_step, float("nan"), dtype=torch.float))
        makespan = torch.where(completed, finish_time.max(dim=-1).values,
                               torch.full_like(finish_time[:, 0], float("nan")))
        return {
            "collision": self.collided.clone(),
            "collision_robot_robot": self.hit_rr.clone(),
            "collision_robot_human": self.hit_rh.clone(),
            "collision_human_human": self.hit_hh.clone(),
            "collision_wall": self.hit_wall.clone(),
            "deadlock": self.deadlocked.clone(),
            "completed": completed,
            "makespan_s": makespan,
            "duration_s": self.steps.float() * self.cfg.dt,
            "collision_steps": self.collision_steps.clone(),
            "min_separation": self.min_separation.clone(),
        }


def courtesy_yield(pos: torch.Tensor, direction: torch.Tensor, active: torch.Tensor, cfg: CorridorConfig) -> torch.Tensor:
    """Agents that must back off, ``(B, M)``: blocked head-on by an oncoming agent deeper in the zone.

    An agent yields when an active agent of the opposite direction is less than ``2r + 0.6`` m ahead,
    within ``2r + 0.1`` m laterally, and closer to the centre of the narrow zone (smaller ``|x|``).
    Only blocks near the bottleneck count (``|x| < zone_half_length + 2r + 0.6``).
    """
    rel = pos.unsqueeze(1) - pos.unsqueeze(2)                      # [b, i, j] = pos_j - pos_i
    ahead = rel[..., 0] * direction.unsqueeze(-1)
    num_agents = pos.shape[1]
    oncoming = direction.unsqueeze(-1) != direction.unsqueeze(-2)
    other = active.unsqueeze(1) & ~torch.eye(num_agents, dtype=torch.bool, device=pos.device)
    reach = 2 * cfg.agent_radius + 0.6
    head_on = other & oncoming & (ahead > 0) & (ahead < reach) & (rel[..., 1].abs() < 2 * cfg.agent_radius + 0.1)
    depth = pos[..., 0].abs()
    deeper = depth.unsqueeze(1) < depth.unsqueeze(2)
    near = depth < cfg.zone_half_length + reach
    yielding = (head_on & deeper).any(-1) & near & active
    # A queue backs up together: whoever is right behind a yielding agent (same direction) yields too.
    follower = other & ~oncoming & (ahead > 0) & (ahead < reach) & (rel[..., 1].abs() < 2 * cfg.agent_radius + 0.1)
    for _ in range(num_agents):
        grown = yielding | (follower & yielding.unsqueeze(1)).any(-1)
        if bool((grown == yielding).all()):
            break
        yielding = grown
    return yielding


def entry_wait(pos: torch.Tensor, target: torch.Tensor, direction: torch.Tensor, active: torch.Tensor,
               cfg: CorridorConfig) -> torch.Tensor:
    """Agents outside the narrow zone whose next target is inside it while an oncoming agent is inside."""
    zone = cfg.zone_half_length
    outside = pos[..., 0].abs() >= zone
    entering = outside & (target[..., 0].abs() < zone)
    inside = active & (pos[..., 0].abs() < zone)
    for_plus = (inside & (direction < 0)).any(-1, keepdim=True)
    for_minus = (inside & (direction > 0)).any(-1, keepdim=True)
    oncoming_inside = torch.where(direction > 0, for_plus, for_minus)
    return entering & oncoming_inside & active
