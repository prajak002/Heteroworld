"""Synthetic heterogeneous world models that predict joint futures of the bottleneck scene.

Each agent carries a private, black-box world model. Given the observed scene, the model emits
``N`` rollouts of the *joint* future (every agent's position over ``H`` plan steps). All three model
families share the same physical prior (a single-file passing schedule through the gap) but differ
in how they perturb it, mirroring the inductive biases of their real counterparts:

* Diffusion (1024-D latent): high-variance, strongly multimodal passing orders and smooth path noise.
* V-JEPA (512-D latent): the joint state is predicted in a nonlinear latent space and projected
  back to physical coordinates through a learned linear read-out, which adds projection artefacts.
* RSSM (128-D latent): recurrent kinematic rollouts whose speed and heading errors accumulate over
  the horizon.

The shared prior treats a moving agent as committed to the gap once it can no longer brake before
the zone entrance (stopping distance ``v^2 / (2 a_max)`` plus a margin).

Every model is egocentric: it believes that the observer and the observer's travel direction pass
the gap earlier than they really would. This is the source of disagreement that consensus resolves.

These generators are synthetic stand-ins with the stated latent dimensions; they are not trained
neural networks. The latent codecs are used by the latent-sharing baselines and by the JEPA read-out.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import CorridorConfig, PredictionConfig
from .geometry import lane_reference_y

WORLD_MODEL_NAMES = ("diffusion", "jepa", "rssm")


@dataclass(frozen=True)
class WorldModelSpec:
    """Behavioural parameters of one world-model family.

    Attributes:
        name: family name.
        latent_dim: dimension of the private latent state.
        order_noise: std (s) of the arrival-time noise that creates passing-order modes.
        ego_bias: time (s) by which the observer believes it reaches the gap earlier.
        direction_bias: time (s) by which the observer believes its direction has priority.
        speed_noise: relative std of the per-rollout cruise speed.
    """

    name: str
    latent_dim: int
    order_noise: float
    ego_bias: float
    direction_bias: float
    speed_noise: float


WORLD_MODEL_SPECS: dict[str, WorldModelSpec] = {
    "diffusion": WorldModelSpec("diffusion", 1024, order_noise=1.2, ego_bias=1.0, direction_bias=1.0, speed_noise=0.05),
    "jepa": WorldModelSpec("jepa", 512, order_noise=0.8, ego_bias=1.5, direction_bias=1.0, speed_noise=0.03),
    "rssm": WorldModelSpec("rssm", 128, order_noise=0.5, ego_bias=2.0, direction_bias=1.0, speed_noise=0.10),
}


@dataclass
class SceneState:
    """Observed state of a batch of scenes, shape ``(S, M)`` per field (``pos`` is ``(S, M, 2)``)."""

    pos: torch.Tensor
    direction: torch.Tensor
    v_des: torch.Tensor
    done: torch.Tensor
    speed: torch.Tensor

    def index(self, rows: torch.Tensor) -> "SceneState":
        """Select a subset of scenes."""
        return SceneState(self.pos[rows], self.direction[rows], self.v_des[rows], self.done[rows], self.speed[rows])


class LatentCodec:
    """Private nonlinear encoder ``z = tanh(W s / scale + b)`` with a ridge-regression read-out.

    ``s`` is a flattened scene snapshot of ``2 M`` coordinates. The read-out is fitted on uniformly
    random snapshots inside the corridor, so decoding is accurate but not exact.
    """

    def __init__(self, input_dim: int, latent_dim: int, cfg: CorridorConfig, seed: int,
                 gain: float = 1.5, ridge: float = 1e-3, num_calibration: int = 4096) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.scale = cfg.half_length
        self.x_limit = cfg.half_length
        self.weight = torch.randn(latent_dim, input_dim, generator=gen) * (gain / input_dim ** 0.5)
        self.bias = 0.1 * torch.randn(latent_dim, generator=gen)
        calib = self.random_snapshots(num_calibration, cfg, gen)
        z = self.encode(calib)
        z_aug = torch.cat([z, torch.ones(num_calibration, 1)], dim=1)
        gram = z_aug.T @ z_aug + ridge * num_calibration * torch.eye(latent_dim + 1)
        self.readout = torch.linalg.solve(gram, z_aug.T @ (calib / self.scale))

    def random_snapshots(self, count: int, cfg: CorridorConfig, gen: torch.Generator) -> torch.Tensor:
        """Uniformly random joint snapshots inside the corridor, shape ``(count, 2 M)``."""
        num_agents = self.input_dim // 2
        x = (torch.rand(count, num_agents, generator=gen) * 2.0 - 1.0) * cfg.half_length
        y = (torch.rand(count, num_agents, generator=gen) * 2.0 - 1.0) * cfg.corridor_half_width
        return torch.stack([x, y], dim=-1).reshape(count, -1)

    def encode(self, snapshots: torch.Tensor) -> torch.Tensor:
        """Map ``(..., 2 M)`` snapshots to ``(..., latent_dim)`` latents."""
        clipped = snapshots.clone()
        clipped[..., 0::2] = clipped[..., 0::2].clamp(-self.x_limit, self.x_limit)
        return torch.tanh(clipped / self.scale @ self.weight.T.to(snapshots) + self.bias.to(snapshots))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Map ``(..., latent_dim)`` latents back to ``(..., 2 M)`` snapshots."""
        ones = torch.ones(*latents.shape[:-1], 1, dtype=latents.dtype, device=latents.device)
        return torch.cat([latents, ones], dim=-1) @ self.readout.to(latents) * self.scale


# ----------------------------------------------------------------------------------------------
# Shared physical prior: single-file passing schedule
# ----------------------------------------------------------------------------------------------
def sample_passing_schedule(scene: SceneState, observer: torch.Tensor, spec: WorldModelSpec,
                            speed: torch.Tensor, cfg: CorridorConfig, pcfg: PredictionConfig,
                            gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample gap entry times and holding points for every rollout.

    Args:
        scene: ``S`` observed scenes.
        observer: index of the predicting agent in each scene, shape ``(S,)``.
        spec: world-model family parameters.
        speed: per-rollout cruise speeds, shape ``(S, N, M)``.
        cfg, pcfg: corridor and prediction configuration.
        gen: random generator.

    Returns:
        ``t_in`` (gap entry time, ``inf`` for agents that are not waiting) and ``hold_u`` (progress
        coordinate of the holding point), both of shape ``(S, N, M)``.
    """
    num_scenes, num_rollouts, num_agents = speed.shape
    u = (scene.direction * scene.pos[..., 0]).unsqueeze(1).expand(-1, num_rollouts, -1)
    direction = scene.direction.unsqueeze(1).expand(-1, num_rollouts, -1)
    active = (~scene.done).unsqueeze(1).expand(-1, num_rollouts, -1)
    u_in = -cfg.zone_half_length
    u_out = cfg.zone_half_length + cfg.agent_radius
    observed_speed = scene.speed.unsqueeze(1).expand(-1, num_rollouts, -1)
    stopping_point = u + observed_speed.square() / (2.0 * cfg.accel_max) + pcfg.commit_margin
    committed = active & (u < u_in) & (observed_speed > pcfg.commit_speed) & (stopping_point >= u_in)
    pending = active & (u < u_in) & ~committed
    inzone = active & (((u >= u_in) & (u < u_out)) | committed)
    zone_time = (u_out - u_in) / speed
    eta = (u_in - u).clamp(min=0.0) / speed
    is_right = direction > 0
    neg_inf = torch.full_like(u, -1e6)

    exit_inzone = torch.where(inzone, (u_out - u) / speed, torch.zeros_like(u))
    entry_inzone = torch.where(inzone, (u_in - u) / speed, neg_inf)
    last_exit_r = torch.where(is_right, exit_inzone, torch.zeros_like(u)).amax(dim=-1)
    last_exit_l = torch.where(~is_right, exit_inzone, torch.zeros_like(u)).amax(dim=-1)
    last_entry_r = torch.where(is_right, entry_inzone, neg_inf).amax(dim=-1)
    last_entry_l = torch.where(~is_right, entry_inzone, neg_inf).amax(dim=-1)

    agent_ids = torch.arange(num_agents, device=u.device)
    is_observer = agent_ids.view(1, 1, -1) == observer.view(-1, 1, 1)
    observer_dir = direction.gather(-1, observer.view(-1, 1, 1).expand(-1, num_rollouts, 1))
    same_dir = direction == observer_dir
    noise = spec.order_noise * torch.randn(speed.shape, generator=gen).to(u)
    score = eta + noise - spec.ego_bias * is_observer.float() - spec.direction_bias * same_dir.float()
    score = torch.where(pending, score, torch.full_like(score, float("inf")))
    order = torch.argsort(score, dim=-1)

    big = torch.full_like(u, float("inf"))
    queue_r = torch.argsort(torch.where(pending & is_right, -u, big), dim=-1)
    queue_l = torch.argsort(torch.where(pending & ~is_right, -u, big), dim=-1)
    count_r = torch.zeros(num_scenes, num_rollouts, dtype=torch.long, device=u.device)
    count_l = torch.zeros_like(count_r)
    t_in = torch.full_like(u, float("inf"))
    hold_u = torch.zeros_like(u)

    def take(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return values.gather(-1, index.unsqueeze(-1)).squeeze(-1)

    for k in range(num_agents):
        slot = order[..., k]
        slot_pending = take(pending, slot)
        slot_right = take(is_right, slot)
        agent = torch.where(slot_right, take(queue_r, count_r.clamp(max=num_agents - 1)),
                            take(queue_l, count_l.clamp(max=num_agents - 1)))
        rank = torch.where(slot_right, count_r, count_l).float()
        same_entry = torch.where(slot_right, last_entry_r, last_entry_l)
        opposite_exit = torch.where(slot_right, last_exit_l, last_exit_r)
        entry = torch.maximum(torch.maximum(take(eta, agent), opposite_exit), same_entry + pcfg.headway)
        exit_time = entry + take(zone_time, agent)
        old_entry = take(t_in, agent)
        old_hold = take(hold_u, agent)
        hold_value = u_in - pcfg.hold_offset - rank * pcfg.queue_spacing
        t_in.scatter_(-1, agent.unsqueeze(-1), torch.where(slot_pending, entry, old_entry).unsqueeze(-1))
        hold_u.scatter_(-1, agent.unsqueeze(-1), torch.where(slot_pending, hold_value, old_hold).unsqueeze(-1))
        update_r = slot_pending & slot_right
        update_l = slot_pending & ~slot_right
        last_entry_r = torch.where(update_r, entry, last_entry_r)
        last_entry_l = torch.where(update_l, entry, last_entry_l)
        last_exit_r = torch.where(update_r, torch.maximum(last_exit_r, exit_time), last_exit_r)
        last_exit_l = torch.where(update_l, torch.maximum(last_exit_l, exit_time), last_exit_l)
        count_r = count_r + update_r.long()
        count_l = count_l + update_l.long()
    return t_in, hold_u


def schedule_to_progress(scene: SceneState, t_in: torch.Tensor, hold_u: torch.Tensor, speed: torch.Tensor,
                         cfg: CorridorConfig, pcfg: PredictionConfig) -> torch.Tensor:
    """Turn a passing schedule into progress coordinates ``u`` of shape ``(S, N, H, M)``.

    A waiting agent drives at cruise speed until its holding point, waits, and leaves the holding
    point exactly in time to reach the zone entrance at ``t_in``.
    """
    num_rollouts = speed.shape[1]
    tau = pcfg.plan_dt * torch.arange(1, pcfg.horizon + 1, dtype=speed.dtype, device=speed.device)
    tau = tau.view(1, 1, -1, 1)
    u0 = (scene.direction * scene.pos[..., 0]).unsqueeze(1).expand(-1, num_rollouts, -1).unsqueeze(2)
    v = speed.unsqueeze(2)
    u_free = u0 + v * tau
    pending = torch.isfinite(t_in).unsqueeze(2)
    u_in = -cfg.zone_half_length
    start_gate = torch.maximum(hold_u.unsqueeze(2), u0)
    release = u_in + v * (tau - torch.where(pending, t_in.unsqueeze(2), torch.zeros_like(u_free)))
    u_wait = torch.minimum(u_free, torch.maximum(start_gate, release))
    u = torch.where(pending, u_wait, u_free)
    return u.clamp(max=cfg.goal_distance + 1.0)


def progress_to_positions(scene: SceneState, u: torch.Tensor, cfg: CorridorConfig) -> torch.Tensor:
    """Map progress coordinates ``(S, N, H, M)`` to positions ``(S, N, H, M, 2)`` on the lane reference.

    Agents that already reached their goal keep their parked position.
    """
    direction = scene.direction.view(scene.direction.shape[0], 1, 1, -1)
    x = direction * u
    y = lane_reference_y(x, direction, cfg)
    pos = torch.stack([x, y], dim=-1)
    parked = scene.pos.view(scene.pos.shape[0], 1, 1, -1, 2).expand_as(pos)
    done = scene.done.view(scene.done.shape[0], 1, 1, -1, 1)
    return torch.where(done, parked, pos)


def _base_rollouts(scene: SceneState, observer: torch.Tensor, spec: WorldModelSpec, cfg: CorridorConfig,
                   pcfg: PredictionConfig, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per-rollout speeds and the noiseless progress profile shared by all families."""
    num_scenes = scene.pos.shape[0]
    shape = (num_scenes, pcfg.num_rollouts, cfg.num_agents)
    rel = 1.0 + spec.speed_noise * torch.randn(shape, generator=gen).to(scene.pos)
    speed = (scene.v_des.unsqueeze(1) * rel).clamp(0.3 * cfg.v_des_min, cfg.v_max)
    t_in, hold_u = sample_passing_schedule(scene, observer, spec, speed, cfg, pcfg, gen)
    return schedule_to_progress(scene, t_in, hold_u, speed, cfg, pcfg), speed


def _active_mask(scene: SceneState) -> torch.Tensor:
    """Broadcastable ``(S, 1, 1, M)`` mask of agents that are still driving."""
    return (~scene.done).view(scene.done.shape[0], 1, 1, -1)


# ----------------------------------------------------------------------------------------------
# The three world-model families
# ----------------------------------------------------------------------------------------------
def diffusion_world_model_rollouts(scene: SceneState, observer: torch.Tensor, cfg: CorridorConfig,
                                   pcfg: PredictionConfig, gen: torch.Generator) -> torch.Tensor:
    """Agent type A: high-variance, multimodal joint rollouts with smooth Brownian path noise."""
    spec = WORLD_MODEL_SPECS["diffusion"]
    u, _ = _base_rollouts(scene, observer, spec, cfg, pcfg, gen)
    active = _active_mask(scene)
    along = torch.cumsum(0.05 * torch.randn(u.shape, generator=gen).to(u), dim=2)
    lateral = torch.cumsum(0.025 * torch.randn(u.shape, generator=gen).to(u), dim=2)
    u = torch.where(active, u + along, u)
    pos = progress_to_positions(scene, u, cfg)
    pos[..., 1] = torch.where(active, pos[..., 1] + lateral, pos[..., 1])
    return pos


def jepa_world_model_rollouts(scene: SceneState, observer: torch.Tensor, cfg: CorridorConfig,
                              pcfg: PredictionConfig, gen: torch.Generator, codec: LatentCodec) -> torch.Tensor:
    """Agent type B: joint rollouts predicted in a 512-D latent space and projected to physical space."""
    spec = WORLD_MODEL_SPECS["jepa"]
    u, _ = _base_rollouts(scene, observer, spec, cfg, pcfg, gen)
    pos = progress_to_positions(scene, u, cfg)
    snapshots = pos.flatten(-2)
    latents = codec.encode(snapshots)
    latents = latents + 0.05 * torch.randn(latents.shape, generator=gen).to(latents)
    projected = codec.decode(latents).view_as(pos)
    done = scene.done.view(scene.done.shape[0], 1, 1, -1, 1)
    return torch.where(done, pos, projected)


def rssm_world_model_rollouts(scene: SceneState, observer: torch.Tensor, cfg: CorridorConfig,
                              pcfg: PredictionConfig, gen: torch.Generator) -> torch.Tensor:
    """Agent type C: recurrent kinematic rollouts whose heading and speed errors accumulate."""
    spec = WORLD_MODEL_SPECS["rssm"]
    u, _ = _base_rollouts(scene, observer, spec, cfg, pcfg, gen)
    active = _active_mask(scene)
    heading_error = torch.cumsum(0.03 * torch.randn(u.shape, generator=gen).to(u), dim=2)
    step_length = torch.diff(u, dim=2, prepend=u[:, :, :1]).abs()
    lateral = torch.cumsum(step_length * torch.sin(heading_error), dim=2)
    along = torch.cumsum(0.02 * torch.randn(u.shape, generator=gen).to(u), dim=2)
    u = torch.where(active, u + along, u)
    pos = progress_to_positions(scene, u, cfg)
    pos[..., 1] = torch.where(active, pos[..., 1] + lateral, pos[..., 1])
    return pos


class HeterogeneousWorldModelFleet:
    """Runs every agent's private world model on a batch of scenes."""

    def __init__(self, cfg: CorridorConfig, pcfg: PredictionConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.pcfg = pcfg
        input_dim = 2 * cfg.num_agents
        self.codecs = {
            name: LatentCodec(input_dim, WORLD_MODEL_SPECS[name].latent_dim, cfg, seed=seed * 101 + k)
            for k, name in enumerate(WORLD_MODEL_NAMES)
        }

    def predict(self, scene: SceneState, agent_type: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Predict joint rollouts for every observer.

        Args:
            scene: ``B`` scenes with ``M`` agents.
            agent_type: world-model family index of every agent, shape ``(B, M)``.
            gen: random generator.

        Returns:
            Tensor of shape ``(B, M, N, H, M, 2)``: observer ``i`` of scene ``b`` predicts ``N``
            joint rollouts of all ``M`` agents over ``H`` plan steps.
        """
        cfg, pcfg = self.cfg, self.pcfg
        num_scenes, num_agents = agent_type.shape
        rows = torch.arange(num_scenes, device=agent_type.device).repeat_interleave(num_agents)
        observers = torch.arange(num_agents, device=agent_type.device).repeat(num_scenes)
        flat_type = agent_type.reshape(-1)
        out = torch.empty(num_scenes * num_agents, pcfg.num_rollouts, pcfg.horizon, num_agents, 2,
                          dtype=scene.pos.dtype, device=scene.pos.device)
        for k, name in enumerate(WORLD_MODEL_NAMES):
            sel = (flat_type == k).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            sub_scene = scene.index(rows[sel])
            sub_obs = observers[sel]
            if name == "diffusion":
                pred = diffusion_world_model_rollouts(sub_scene, sub_obs, cfg, pcfg, gen)
            elif name == "jepa":
                pred = jepa_world_model_rollouts(sub_scene, sub_obs, cfg, pcfg, gen, self.codecs[name])
            else:
                pred = rssm_world_model_rollouts(sub_scene, sub_obs, cfg, pcfg, gen)
            out[sel] = pred
        return out.view(num_scenes, num_agents, pcfg.num_rollouts, pcfg.horizon, num_agents, 2)


def scene_from_env(env) -> SceneState:
    """Snapshot the observable state of a :class:`~hetero_world.env.BottleneckCorridorEnv`."""
    return SceneState(pos=env.pos.clone(), direction=env.direction.clone(), v_des=env.v_des.clone(),
                      done=env.done.clone(), speed=env.speed.clone())
