"""Configuration objects shared by the simulator, world models, consensus and benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CorridorConfig:
    """Geometry, dynamics and metric thresholds of the two-lane bottleneck corridor.

    The corridor runs along the x axis. Lane centres sit at ``y = +lane_offset`` (agents moving
    in +x) and ``y = -lane_offset`` (agents moving in -x). Around ``x = 0`` the free space narrows
    to a single-file gap of half-width ``gap_half_width`` over ``|x| <= gap_half_length``, with a
    linear funnel of length ``taper_length`` on both sides. Two agents of radius ``agent_radius``
    cannot be side by side inside the gap, so opposing traffic must take turns.
    """

    num_agents: int = 10
    num_humans: int = 0
    human_order_noise: float = 1.0
    corridor_half_width: float = 2.0
    lane_offset: float = 1.0
    gap_half_length: float = 1.5
    gap_half_width: float = 0.55
    taper_length: float = 1.5
    gap_offset: float = 0.0
    agent_radius: float = 0.3
    spawn_clearance: float = 4.0
    spawn_spacing_min: float = 1.0
    spawn_spacing_max: float = 1.6
    goal_distance: float = 6.5
    v_max: float = 1.5
    v_des_min: float = 1.0
    v_des_max: float = 1.4
    omega_max: float = 1.5
    accel_max: float = 2.0
    dt: float = 0.1
    base_time: float = 40.0
    time_per_agent: float = 2.0
    collision_tolerance: float = 0.01
    freeze_speed: float = 0.1
    freeze_duration: float = 5.0
    # Doorway courtesy (benchmark v2): in a head-on block the agent deeper in the narrow zone keeps
    # the right of way and the other backs off at up to ``v_reverse``; pedestrians also wait at the
    # funnel mouth while an oncoming agent is inside. ``courtesy=False`` reproduces benchmark v1.
    courtesy: bool = True
    v_reverse: float = 0.3
    assertive_prob: float = 0.5

    @property
    def num_right(self) -> int:
        """Number of agents spawned on the left moving in +x."""
        return int(math.ceil(self.num_agents / 2))

    @property
    def num_left(self) -> int:
        """Number of agents spawned on the right moving in -x."""
        return self.num_agents - self.num_right

    @property
    def zone_half_length(self) -> float:
        """Half-length of the mutual-exclusion zone used by the passing schedule."""
        return self.gap_half_length + 0.5 * self.taper_length

    @property
    def half_length(self) -> float:
        """Half-length of the drawn corridor (covers every spawn position and goal)."""
        far_spawn = self.spawn_clearance + (max(self.num_right, 1) - 1) * self.spawn_spacing_max
        return max(far_spawn, self.goal_distance) + 2.0

    @property
    def max_steps(self) -> int:
        """Episode time limit in simulator steps (grows with the number of agents)."""
        return int(round((self.base_time + self.time_per_agent * self.num_agents) / self.dt))

    @property
    def freeze_steps(self) -> int:
        """Number of consecutive frozen steps that defines a deadlock."""
        return int(round(self.freeze_duration / self.dt))

    def with_agents(self, num_agents: int) -> "CorridorConfig":
        """Return a copy of this configuration with a different agent count."""
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values["num_agents"] = num_agents
        return CorridorConfig(**values)

    def with_humans(self, num_humans: int) -> "CorridorConfig":
        """Return a copy of this configuration in which ``num_humans`` agents are uncontrolled pedestrians."""
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values["num_humans"] = num_humans
        return CorridorConfig(**values)

    def with_sidewalk(self, wide: float, pinch: float, pinch_length: float, taper: float = 1.0) -> "CorridorConfig":
        """Return a copy whose geometry is a measured sidewalk pinch point (metres).

        ``wide`` is the typical clear width of the sidewalk, ``pinch`` the narrowest clear width and
        ``pinch_length`` the length over which it stays narrow. Lanes sit at a quarter of the width;
        spawn queues start past the funnel, goals lie 3 m beyond it, and the time limit grows with
        the extra walking distance.
        """
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        gap_half_length = max(0.25, pinch_length / 2)
        # Symmetric abstraction of the measured pinch (same widths and length). A one-sided pinch
        # (``gap_offset``) is supported by the geometry, but our reactive pedestrians have no lateral
        # avoidance, so the long diagonal merge it requires is not modelled faithfully.
        values.update(corridor_half_width=wide / 2, gap_half_width=pinch / 2, gap_half_length=gap_half_length,
                      taper_length=taper, lane_offset=min(wide / 4, wide / 2 - self.agent_radius - 0.05),
                      spawn_clearance=gap_half_length + taper + 1.0, goal_distance=gap_half_length + taper + 3.0)
        values["base_time"] = self.base_time + 2.0 * max(0.0, values["goal_distance"] - self.goal_distance) + 4.0 * max(0.0, pinch_length - 3.0)
        return CorridorConfig(**values)


@dataclass(frozen=True)
class PredictionConfig:
    """Rollout settings shared by all world models."""

    num_rollouts: int = 100
    horizon: int = 20
    plan_dt: float = 0.25
    replan_every: int = 5
    headway: float = 0.8
    hold_offset: float = 0.5
    queue_spacing: float = 0.9
    commit_margin: float = 0.1
    commit_speed: float = 0.3
    consistency_weight: float = 20.0

    def replan_hz(self, sim_dt: float) -> float:
        """Broadcast frequency in Hz for a simulator step of ``sim_dt`` seconds."""
        return 1.0 / (self.replan_every * sim_dt)


@dataclass(frozen=True)
class ConsensusConfig:
    """Hyper-parameters of the entropic free-support barycenter."""

    epsilon: float = 0.01
    max_iters: int = 50
    outer_iters: int = 3
