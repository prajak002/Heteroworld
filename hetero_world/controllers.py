"""Closed-loop controllers: HETERO-WORLD, the four mandatory baselines and two ablations.

All plan-based controllers share the same world-model fleet, the same deterministic plan-selection
layer and the same MPC tracker. The selection layer scores candidates for safety and progress, adds a
temporal-consistency penalty against the previous plan, and also offers the previous plan advanced
by one replan period as a candidate (receding-horizon warm start). They differ only in what
is communicated and how the received information is fused:

==================  =====================================  ===========================================
method              broadcast payload                      fusion
==================  =====================================  ===========================================
hetero_world        N joint trajectories (physical space)  entropic W2 barycenter, select best atom
hw_robust           N joint trajectories (physical space)  mixed traffic: select a received rollout and score
                                                           its robot part against the pedestrian futures of
                                                           the barycenter (robust consensus forecast)
mixture_robust      N joint trajectories (physical space)  mixed traffic: as hw_robust, but against N pedestrian
                                                           futures drawn from the pooled rollouts (ablation)
hw_anchor           N joint trajectories (physical space)  select a received rollout, penalised by its
                                                           squared W2 distance to the barycenter support
union_select        N joint trajectories (physical space)  select best rollout from the union (ablation)
independent         nothing                                select best of own rollouts (ablation)
feature_sharing     N latent rollouts (N x H x d_i)        decode with sender decoders, index-wise mean
procrustes          N latent rollouts (N x H x d_i)        pairwise orthogonal Procrustes into own space
noncoop_mpc         nothing                                none: MPC tracks the free lane path
mappo               nothing                                learned decentralised policy
==================  =====================================  ===========================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import ConsensusConfig, CorridorConfig, PredictionConfig
from .geometry import lane_reference_y
from .mpc import MPCTracker
from .planner import score_joint_plans
from .sinkhorn import SinkhornBarycenterConsensus, sinkhorn_plan, squared_distances
from .learned_models import LearnedWorldModelFleet, observe
from .world_models import WORLD_MODEL_NAMES, WORLD_MODEL_SPECS, HeterogeneousWorldModelFleet, LatentCodec, scene_from_env

BYTES_PER_VALUE = 2  # every method transmits float16 values
PLAN_METHODS = ("hetero_world", "hw_robust", "mixture_robust", "oracle_robust", "cv_robust", "hw_anchor", "union_select", "independent", "feature_sharing", "procrustes")


def free_path_reference(env, horizon_steps: int) -> torch.Tensor:
    """Reference that drives along the lane centre at the desired speed, ``(B, M, T, 2)``."""
    cfg = env.cfg
    steps = cfg.dt * torch.arange(1, horizon_steps + 1, dtype=env.pos.dtype, device=env.pos.device)
    u = env.progress_coordinate.unsqueeze(-1) + env.v_des.unsqueeze(-1) * steps
    u = u.clamp(max=cfg.goal_distance + 1.0)
    direction = env.direction.unsqueeze(-1)
    x = direction * u
    return torch.stack([x, lane_reference_y(x, direction, cfg)], dim=-1)


def interpolate_plan(own_plan: torch.Tensor, origin: torch.Tensor, elapsed: float, plan_dt: float,
                     sim_dt: float, horizon_steps: int) -> torch.Tensor:
    """Sample an agent's own plan at the next ``horizon_steps`` simulator times.

    Args:
        own_plan: ``(B, M, H, 2)`` positions at plan times ``plan_dt * (1..H)``.
        origin: ``(B, M, 2)`` positions at plan creation (plan time 0).
        elapsed: seconds since plan creation.

    Returns:
        ``(B, M, T, 2)`` reference positions; times beyond the horizon hold the last waypoint.
    """
    waypoints = torch.cat([origin.unsqueeze(2), own_plan], dim=2)
    horizon = own_plan.shape[2]
    times = elapsed + sim_dt * torch.arange(1, horizon_steps + 1, dtype=own_plan.dtype, device=own_plan.device)
    idx = (times / plan_dt).clamp(0.0, float(horizon))
    lo = idx.floor().long().clamp(max=horizon - 1)
    frac = (idx - lo.to(idx.dtype)).view(1, 1, -1, 1)
    lo_pts = waypoints[:, :, lo]
    hi_pts = waypoints[:, :, (lo + 1).clamp(max=horizon)]
    return lo_pts + frac * (hi_pts - lo_pts)


@dataclass
class EpisodeCommStats:
    """Accumulators for communication and agreement statistics of a batch of episodes."""

    bytes_sent: torch.Tensor
    active_agent_steps: torch.Tensor
    plan_disagreement: list[torch.Tensor] = field(default_factory=list)
    belief_w2: list[torch.Tensor] = field(default_factory=list)
    alignment_w2: list[torch.Tensor] = field(default_factory=list)
    marginal_error: list[torch.Tensor] = field(default_factory=list)

    @staticmethod
    def stack_mean(values: list[torch.Tensor], batch: int) -> torch.Tensor:
        """Per-episode NaN-aware mean of a list of ``(B,)`` tensors."""
        if not values:
            return torch.full((batch,), float("nan"))
        data = torch.stack(values)
        return torch.nanmean(data, dim=0)


class Controller:
    """Common interface of every closed-loop method."""

    name = "controller"
    communicates = False

    def reset(self, env, seed: int) -> None:
        """Prepare for a new batch of episodes."""
        self.stats = EpisodeCommStats(bytes_sent=torch.zeros(env.batch_size),
                                      active_agent_steps=torch.zeros(env.batch_size))

    def act(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(v_cmd, omega)`` for the current state."""
        raise NotImplementedError

    def record_step(self, env) -> None:
        """Accumulate per-step bookkeeping (called once per simulator step before ``act``)."""
        self.stats.active_agent_steps += env.active.sum(-1).float().cpu()

    def episode_stats(self, env) -> dict[str, torch.Tensor]:
        """Per-episode communication and agreement statistics."""
        batch = env.batch_size
        seconds = (self.stats.active_agent_steps * env.cfg.dt).clamp_min(1e-9)
        return {
            "bandwidth_kBps": self.stats.bytes_sent / seconds / 1000.0,
            "plan_disagreement_m": EpisodeCommStats.stack_mean(self.stats.plan_disagreement, batch),
            "belief_w2_m": EpisodeCommStats.stack_mean(self.stats.belief_w2, batch),
            "alignment_w2_m": EpisodeCommStats.stack_mean(self.stats.alignment_w2, batch),
            "marginal_error": EpisodeCommStats.stack_mean(self.stats.marginal_error, batch),
        }


class NonCooperativeMPC(Controller):
    """Baseline 1: each agent tracks its own lane path with reactive MPC collision avoidance."""

    name = "noncoop_mpc"

    def __init__(self, cfg: CorridorConfig) -> None:
        self.mpc = MPCTracker(cfg)

    def act(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        ref = free_path_reference(env, self.mpc.horizon_steps)
        return self.mpc(env.pos, env.theta, env.speed, env.active, ref, env.direction)


def orthogonal_procrustes_map(source: LatentCodec, target: LatentCodec, anchors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Fit ``z_target ~ (z_source - m_s) R + m_t`` with orthogonal ``R`` on zero-padded latents.

    Both latent sets are zero-padded to ``D = max(d_s, d_t)`` and ``R`` is the orthogonal
    Procrustes solution ``U V^T`` of the cross-covariance. Because the padded source coordinates are
    zero and only the first ``d_t`` outputs are decoded, only the ``d_s x d_t`` block of ``R`` is
    needed at run time.

    Args:
        anchors: ``(A, 2 M)`` calibration snapshots encoded by both models (shared once offline).

    Returns:
        ``(R_block, mean_source, mean_target)`` with ``R_block`` of shape ``(d_s, d_t)``.
    """
    z_s = source.encode(anchors)
    z_t = target.encode(anchors)
    dim = max(z_s.shape[1], z_t.shape[1])
    mean_s, mean_t = z_s.mean(0), z_t.mean(0)
    pad_s = torch.nn.functional.pad(z_s - mean_s, (0, dim - z_s.shape[1]))
    pad_t = torch.nn.functional.pad(z_t - mean_t, (0, dim - z_t.shape[1]))
    u, _, vh = torch.linalg.svd(pad_s.T @ pad_t)
    rot = u @ vh
    return rot[:z_s.shape[1], :z_t.shape[1]].contiguous(), mean_s, mean_t


class PlanConsensusController(Controller):
    """World-model-based controllers that broadcast predictions and fuse them into a joint plan.

    Args:
        method: one of :data:`PLAN_METHODS`.
        cfg, pcfg, ccfg: corridor, prediction and consensus configuration.
        packet_loss: probability that a broadcast from one agent does not reach another.
        belief_every: compute the pairwise W2 belief discrepancy every this many replans
            (0 disables it).
        fleet_seed: seed of the world-model weights (latent codecs).
    """

    def __init__(self, method: str, cfg: CorridorConfig, pcfg: PredictionConfig, ccfg: ConsensusConfig,
                 packet_loss: float = 0.0, belief_every: int = 4, fleet_seed: int = 0,
                 num_anchors: int = 512, faulty_agents: int = 0, faulty_scale: float = 1.6,
                 anchor_weight: float = 20.0, learned: bool = False,
                 learned_scope: str = "all") -> None:
        if method not in PLAN_METHODS:
            raise ValueError(f"unknown method {method}")
        self.name = method
        self.method = method
        self.communicates = method != "independent"
        self.cfg, self.pcfg, self.ccfg = cfg, pcfg, ccfg
        self.packet_loss = packet_loss
        self.faulty_agents = faulty_agents
        self.faulty_scale = faulty_scale
        self.anchor_weight = anchor_weight
        self.belief_every = belief_every
        self.fleet = HeterogeneousWorldModelFleet(cfg, pcfg, seed=fleet_seed)
        # Trained networks replace the synthetic generators for prediction (codecs stay for baselines).
        ckpt_dir = "checkpoints" if fleet_seed == 0 else f"checkpoints/seed{fleet_seed}"
        self.learned_fleet = LearnedWorldModelFleet(pcfg.num_rollouts, ckpt_dir) if learned else None
        # "humans": the trained networks forecast only the pedestrians; robot rows (the candidate plans)
        # still come from the synthetic proposal generator. The networks were trained on pedestrian-only
        # episodes, so their robot rows imitate courteous waiting and contain no plan that clears a doorway.
        if learned_scope not in ("all", "humans"):
            raise ValueError(f"unknown learned_scope {learned_scope}")
        self.learned_scope = learned_scope
        self.mpc = MPCTracker(cfg)
        self.consensus = SinkhornBarycenterConsensus(ccfg.epsilon, ccfg.max_iters, ccfg.outer_iters)
        self.codecs = [self.fleet.codecs[name] for name in WORLD_MODEL_NAMES]
        self.latent_dims = torch.tensor([WORLD_MODEL_SPECS[name].latent_dim for name in WORLD_MODEL_NAMES])
        anchor_gen = torch.Generator().manual_seed(9_999 + fleet_seed)
        anchors = self.codecs[0].random_snapshots(num_anchors, cfg, anchor_gen)
        self.procrustes = {
            (s, t): orthogonal_procrustes_map(self.codecs[s], self.codecs[t], anchors)
            for s in range(len(self.codecs)) for t in range(len(self.codecs))
        }

    # ------------------------------------------------------------------------------------------
    def reset(self, env, seed: int) -> None:
        super().reset(env, seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.loss_gen = torch.Generator().manual_seed(seed + 7_777)
        self.step_count = 0
        self.replans = 0
        self.plan = None
        self.prev_plan = None
        self.origin = None
        self.plan_step = 0

    def payload_bytes(self, agent_type: torch.Tensor) -> torch.Tensor:
        """Bytes broadcast by each agent per replan, shape ``(B, M)``."""
        pcfg, cfg = self.pcfg, self.cfg
        if self.method in ("hetero_world", "hw_robust", "mixture_robust", "oracle_robust", "cv_robust", "hw_anchor", "union_select"):
            per_agent = pcfg.num_rollouts * pcfg.horizon * cfg.num_agents * 2 * BYTES_PER_VALUE
            return torch.full(agent_type.shape, float(per_agent))
        if self.method in ("feature_sharing", "procrustes"):
            dims = self.latent_dims[agent_type.cpu()].float()
            return pcfg.num_rollouts * pcfg.horizon * dims * BYTES_PER_VALUE
        return torch.zeros(agent_type.shape)

    def _receive_mask(self, env) -> torch.Tensor:
        """``(B, receiver, sender)`` mask of broadcasts that arrived (own data always available)."""
        batch, num_agents = env.batch_size, self.cfg.num_agents
        sender_active = env.active.cpu().unsqueeze(1).expand(batch, num_agents, num_agents)
        eye = torch.eye(num_agents, dtype=torch.bool).unsqueeze(0)
        if self.packet_loss > 0:
            arrived = torch.rand(batch, num_agents, num_agents, generator=self.loss_gen) >= self.packet_loss
        else:
            arrived = torch.ones(batch, num_agents, num_agents, dtype=torch.bool)
        return (arrived & sender_active) | eye

    # ------------------------------------------------------------------------------------------
    def _replan(self, env) -> None:
        cfg, pcfg = self.cfg, self.pcfg
        scene = scene_from_env(env)
        if self.learned_fleet is not None:
            preds = self.learned_fleet.predict(observe(env), env.pos, env.agent_type, self.gen).to(env.pos)
            if self.learned_scope == "humans":
                proposals = self.fleet.predict(scene, env.agent_type, self.gen)
                human_col = env.is_human.view(env.pos.shape[0], 1, 1, 1, -1, 1)
                preds = torch.where(human_col, preds, proposals)
        else:
            preds = self.fleet.predict(scene, env.agent_type, self.gen)
        if self.faulty_agents > 0:
            # Faulty world models hallucinate faster-than-feasible motion: every predicted
            # displacement from the current scene is stretched by ``faulty_scale``.
            k = self.faulty_agents
            start = env.pos.view(batch_size := env.pos.shape[0], 1, 1, 1, *env.pos.shape[1:])
            stretched = start + self.faulty_scale * (preds[:, :k] - start)
            preds = torch.cat([stretched, preds[:, k:]], dim=1)
        batch, num_agents, num_rollouts, horizon = preds.shape[:4]
        flat = preds.reshape(batch, num_agents, num_rollouts, -1)
        received = self._receive_mask(env)
        start_u = env.progress_coordinate
        direction = env.direction
        active = env.active
        human = env.is_human
        mixed = bool(human.any())
        if mixed:
            # Pedestrians carry no world model that anyone can hear: only robots broadcast.
            eye = torch.eye(num_agents, dtype=torch.bool).unsqueeze(0)
            received = (received & ~human.cpu().unsqueeze(1)) | eye
        robot_active = active & ~human
        ref_humans = None

        shift = int(round(pcfg.replan_every * cfg.dt / pcfg.plan_dt))
        prev = self.prev_plan
        lam = pcfg.consistency_weight

        def base_scores(candidates: torch.Tensor) -> torch.Tensor:
            """Safety/progress scores of ``(B, G, K, H, M, 2)`` candidates -> ``(B, G, K)``.

            In mixed traffic only robots count for progress and robot-robot safety; robot-pedestrian
            safety is scored against ``ref_humans`` when a robust method sets it, and otherwise against
            the pedestrian futures inside each candidate (the naive, optimistic reading).
            """
            groups = candidates.shape[1]
            exp = lambda t: t.unsqueeze(1).expand(batch, groups, *t.shape[1:])
            if not mixed:
                return _chunked_scores(candidates, exp(start_u), exp(direction), exp(active), cfg)
            scores = _chunked_scores(candidates, exp(start_u), exp(direction), exp(robot_active), cfg)
            return scores + human_cost(candidates, human, ref_humans, cfg)

        def shared_prev(groups: int) -> torch.Tensor | None:
            """Previous plans for ``groups`` candidate groups (1 means every observer agrees)."""
            if prev is None:
                return None
            return prev[:, :1] if groups == 1 else prev

        if self.method in ("hw_robust", "mixture_robust", "oracle_robust", "cv_robust"):
            union = preds.reshape(batch, 1, num_agents * num_rollouts, horizon, num_agents, 2)
            sender = (active & ~human).to(flat.dtype)
            sender = torch.where(sender.sum(-1, keepdim=True) > 0, sender, torch.ones_like(sender))
            # Under packet loss every robot builds its pedestrian reference only from the broadcasts it
            # received (one reference per receiver); on a clean channel all receivers share one reference.
            groups = 1 if self.packet_loss == 0 else num_agents
            if groups == 1:
                group_w = sender.unsqueeze(1)
            else:
                group_w = received.to(flat.dtype) * sender.unsqueeze(1)
                group_w = torch.where(group_w.sum(-1, keepdim=True) > 0, group_w, sender.unsqueeze(1).expand_as(group_w))
            if self.method == "hw_robust":
                expanded = flat.unsqueeze(1).expand(batch, groups, *flat.shape[1:]).reshape(batch * groups, *flat.shape[1:])
                ref_humans = self.consensus(expanded, group_w.reshape(batch * groups, -1)).support.view(
                    batch, groups, num_rollouts, horizon, num_agents, 2)
            elif self.method == "cv_robust":
                # Standard crowd-navigation baseline: pedestrians extrapolated at constant velocity.
                steps = pcfg.plan_dt * torch.arange(1, horizon + 1, dtype=env.pos.dtype).view(1, -1, 1, 1)
                vel = env.speed.unsqueeze(-1) * torch.stack([torch.cos(env.theta), torch.sin(env.theta)], dim=-1)
                ref_humans = (env.pos.unsqueeze(1) + steps * vel.unsqueeze(1)).view(batch, 1, 1, horizon, num_agents, 2)
            elif self.method == "oracle_robust" and cfg.courtesy:
                # Upper bound (benchmark v2): the simulator's true reactive pedestrians, rolled out
                # under the robots' previous joint plan (free-path motion before the first plan).
                if prev is None:
                    steps = pcfg.plan_dt * torch.arange(1, horizon + 1, dtype=env.pos.dtype)
                    u = start_u.unsqueeze(1) + env.v_des.unsqueeze(1) * steps.view(1, -1, 1)
                    x = direction.unsqueeze(1) * u.clamp(max=cfg.goal_distance + 1.0)
                    robot_future = torch.stack([x, lane_reference_y(x, direction.unsqueeze(1), cfg)], dim=-1)
                else:
                    robot_future = advance_plan(prev[:, :1], shift)[:, 0]
                ref_humans = env.simulate_humans(robot_future, pcfg.plan_dt).view(batch, 1, 1, horizon, num_agents, 2)
            elif self.method == "oracle_robust":
                # Upper bound (benchmark v1): the true pedestrian reference trajectories over the plan horizon.
                stride = int(round(pcfg.plan_dt / cfg.dt))
                steps = env.steps.view(-1, 1) + stride * torch.arange(1, horizon + 1).view(1, -1)
                steps = steps.clamp(max=env.human_ref.shape[1] - 1)
                ref_humans = env.human_ref[torch.arange(batch).unsqueeze(1), steps].view(batch, 1, 1, horizon, num_agents, 2)
            else:
                # N pedestrian futures drawn uniformly from the pooled rollouts each receiver heard.
                weights = group_w.repeat_interleave(num_rollouts, dim=-1).reshape(batch * groups, -1)
                pick = torch.multinomial(weights, num_rollouts, replacement=True, generator=self.gen).view(batch, groups, -1)
                ref_humans = union[torch.arange(batch).view(-1, 1, 1), 0, pick]
            self.last_ref_humans = ref_humans          # for visualisation (record_scenes.py); not used by the controller
            # Robot terms are shared by every receiver; only the pedestrian term depends on the reference.
            robot_scores = _chunked_scores(union, start_u.unsqueeze(1), direction.unsqueeze(1), robot_active.unsqueeze(1), cfg)
            ref_groups = ref_humans.shape[1]
            ped_scores = human_cost(union.expand(batch, ref_groups, *union.shape[2:]), human, ref_humans, cfg)
            scores = robot_scores + ped_scores + lam * consistency_cost(union, shared_prev(groups), active, shift)
            scores = scores.expand(batch, num_agents, -1).reshape(batch, num_agents, num_agents, num_rollouts)
            masked = torch.where(received.unsqueeze(-1), scores, torch.full_like(scores, float("inf")))
            best_scores, best = masked.flatten(2).min(-1)
            plans = _gather_candidates(union.expand(batch, num_agents, *union.shape[2:]), best)
            beliefs = None
        elif self.method == "independent":
            scores = base_scores(preds) + lam * consistency_cost(preds, prev, active, shift)
            best_scores, best = scores.min(-1)
            plans = _gather_candidates(preds, best)
            beliefs = flat
        elif self.method == "union_select":
            union = preds.reshape(batch, 1, num_agents * num_rollouts, horizon, num_agents, 2)
            groups = 1 if self.packet_loss == 0 else num_agents
            scores = base_scores(union) + lam * consistency_cost(union, shared_prev(groups), active, shift)
            scores = scores.expand(batch, num_agents, -1).reshape(batch, num_agents, num_agents, num_rollouts)
            masked = torch.where(received.unsqueeze(-1), scores, torch.full_like(scores, float("inf")))
            best_scores, best = masked.flatten(2).min(-1)
            plans = _gather_candidates(union.expand(batch, num_agents, *union.shape[2:]), best)
            beliefs = None
        elif self.method == "hetero_world":
            sender_alpha = (active & ~human).to(flat.dtype)
            sender_alpha = torch.where(sender_alpha.sum(-1, keepdim=True) > 0, sender_alpha, torch.ones_like(sender_alpha))
            if self.packet_loss > 0:
                alpha = received.to(flat.dtype) * sender_alpha.unsqueeze(1)
                alpha = torch.where(alpha.sum(-1, keepdim=True) > 0, alpha, torch.eye(num_agents).expand_as(alpha))
                measures = flat.unsqueeze(1).expand(batch, num_agents, *flat.shape[1:]).reshape(-1, *flat.shape[1:])
                out = self.consensus(measures, alpha.reshape(-1, num_agents))
                support = out.support.view(batch, num_agents, num_rollouts, horizon, num_agents, 2)
                w2 = out.w2_per_point.view(batch, num_agents, num_agents)
                w2 = (w2 * alpha).sum(-1) / alpha.sum(-1)
                marg = out.marginal_error.view(batch, num_agents, num_agents).mean(-1)
            else:
                out = self.consensus(flat, sender_alpha)
                support = out.support.view(batch, 1, num_rollouts, horizon, num_agents, 2)
                w2 = out.w2_per_point
                marg = out.marginal_error
            groups = support.shape[1]
            scores = base_scores(support) + lam * consistency_cost(support, shared_prev(groups), active, shift)
            best_scores, best = scores.min(-1)
            best_scores = best_scores.expand(batch, num_agents)
            plans = _gather_candidates(support.expand(batch, num_agents, *support.shape[2:]),
                                       best.expand(batch, num_agents) if groups == 1 else best)
            act_f = active.to(flat.dtype)
            denom = act_f.sum(-1).clamp_min(1.0)
            self.stats.alignment_w2.append(torch.where(active.any(-1), (w2 * act_f).sum(-1) / denom,
                                                       torch.full_like(denom, float("nan"))).cpu())
            self.stats.marginal_error.append(torch.where(active.any(-1), (marg * act_f).sum(-1) / denom,
                                                         torch.full_like(denom, float("nan"))).cpu())
            beliefs = support.reshape(batch, groups, num_rollouts, -1) if groups > 1 else None
        elif self.method == "hw_anchor":
            sender_alpha = (active & ~human).to(flat.dtype)
            sender_alpha = torch.where(sender_alpha.sum(-1, keepdim=True) > 0, sender_alpha, torch.ones_like(sender_alpha))
            if self.packet_loss > 0:
                alpha = received.to(flat.dtype) * sender_alpha.unsqueeze(1)
                alpha = torch.where(alpha.sum(-1, keepdim=True) > 0, alpha, torch.eye(num_agents).expand_as(alpha))
                measures = flat.unsqueeze(1).expand(batch, num_agents, *flat.shape[1:]).reshape(-1, *flat.shape[1:])
                out = self.consensus(measures, alpha.reshape(-1, num_agents))
                support = out.support.view(batch, num_agents, num_rollouts, -1)
            else:
                out = self.consensus(flat, sender_alpha)
                support = out.support.view(batch, 1, num_rollouts, -1)
            groups = support.shape[1]
            union = preds.reshape(batch, 1, num_agents * num_rollouts, horizon, num_agents, 2)
            # Consensus anchor: squared distance (m^2 per active point) from each received rollout to
            # its nearest barycenter atom, so rollouts that no other model supports are penalised.
            act_f = active.to(flat.dtype)
            pts = (act_f.sum(-1) * horizon).clamp_min(1.0).view(batch, 1, 1)
            mask = act_f.view(batch, 1, 1, num_agents, 1).expand(batch, 1, horizon, num_agents, 2).reshape(batch, 1, -1)
            cand = union.reshape(batch, 1, num_agents * num_rollouts, -1) * mask.unsqueeze(2)
            anchor = torch.stack([
                squared_distances(support[:, g:g + 1] * mask.unsqueeze(2), cand).squeeze(1).min(dim=-2).values
                for g in range(groups)], dim=1) / pts
            scores = base_scores(union) + lam * consistency_cost(union, shared_prev(groups), active, shift)
            scores = scores + self.anchor_weight * anchor
            scores = scores.expand(batch, num_agents, -1).reshape(batch, num_agents, num_agents, num_rollouts)
            masked = torch.where(received.unsqueeze(-1), scores, torch.full_like(scores, float("inf")))
            best_scores, best = masked.flatten(2).min(-1)
            plans = _gather_candidates(union.expand(batch, num_agents, *union.shape[2:]), best)
            beliefs = None
        elif self.method == "feature_sharing":
            decoded = self._latent_roundtrip(preds, env.agent_type, receiver_types=None)
            weight = received.to(flat.dtype)
            fused = torch.einsum("brs,bsnhmc->brnhmc", weight, decoded) / weight.sum(-1).view(batch, num_agents, 1, 1, 1, 1)
            if self.packet_loss == 0:
                fused = fused[:, :1]
            groups = fused.shape[1]
            scores = base_scores(fused) + lam * consistency_cost(fused, shared_prev(groups), active, shift)
            best_scores, best = scores.min(-1)
            best_scores = best_scores.expand(batch, num_agents)
            plans = _gather_candidates(fused.expand(batch, num_agents, *fused.shape[2:]),
                                       best.expand(batch, num_agents) if groups == 1 else best)
            beliefs = fused.reshape(batch, groups, num_rollouts, -1) if groups > 1 else None
        else:
            plans, beliefs, best_scores = self._procrustes_plans(preds, env, received, start_u, direction, active,
                                                                 prev, shift)

        if prev is not None:
            carry = advance_plan(prev, shift)
            carry_scores = base_scores(carry.unsqueeze(2))[:, :, 0]
            keep = carry_scores <= best_scores
            plans = torch.where(keep.view(batch, num_agents, 1, 1, 1), carry, plans)

        self.plan = plans
        self.prev_plan = plans
        self.origin = env.pos.clone()
        self.plan_step = 0
        self._record_agreement(plans, beliefs, active)
        sent = self.payload_bytes(env.agent_type) * env.active.cpu().float()
        if self.communicates:
            self.stats.bytes_sent += sent.sum(-1)
        self.replans += 1

    def _encode_broadcasts(self, preds: torch.Tensor, agent_type: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Encode every sender's rollouts with its private codec and quantise to float16.

        Returns:
            One ``(sender_mask, latents)`` pair per model family, ``latents`` of shape
            ``(count, N, H, d_family)``.
        """
        batch, num_agents, num_rollouts, horizon = preds.shape[:4]
        snapshots = preds.reshape(batch, num_agents, num_rollouts, horizon, -1)
        encoded = []
        for s, codec in enumerate(self.codecs):
            sel = agent_type == s
            z = codec.encode(snapshots[sel]).half().float() if bool(sel.any()) else None
            encoded.append((sel, z))
        return encoded

    def _decode_broadcasts(self, preds: torch.Tensor, encoded: list[tuple[torch.Tensor, torch.Tensor]],
                           receiver_type: int | None) -> torch.Tensor:
        """Decode broadcast latents into ``(B, M_sender, N, H, M, 2)`` joint rollouts.

        With ``receiver_type=None`` each sender's own decoder is used (oracle feature sharing).
        Otherwise the latents are aligned into the latent space of family ``receiver_type`` by
        orthogonal Procrustes and decoded with that receiver's decoder.
        """
        batch, num_agents, num_rollouts, horizon = preds.shape[:4]
        out = torch.empty(batch, num_agents, num_rollouts, horizon, preds.shape[-2] * 2, dtype=preds.dtype)
        for s, (sel, z) in enumerate(encoded):
            if z is None:
                continue
            if receiver_type is None:
                out[sel] = self.codecs[s].decode(z)
            else:
                rot, mean_s, mean_t = self.procrustes[(s, receiver_type)]
                out[sel] = self.codecs[receiver_type].decode((z - mean_s) @ rot + mean_t)
        return out.view_as(preds)

    def _latent_roundtrip(self, preds: torch.Tensor, agent_type: torch.Tensor,
                          receiver_types: int | None) -> torch.Tensor:
        """Encode, quantise and decode every broadcast (see :meth:`_decode_broadcasts`)."""
        return self._decode_broadcasts(preds, self._encode_broadcasts(preds, agent_type), receiver_types)

    def _procrustes_plans(self, preds, env, received, start_u, direction, active, prev, shift):
        """Baseline 2: each receiver aligns every sender's latents into its own space and decodes them.

        Decoded candidate sets depend only on the receiver's model family, so they are computed once
        per family; each receiver then selects among the candidates of the senders it heard from,
        always using its own native rollouts for itself.
        """
        cfg, lam = self.cfg, self.pcfg.consistency_weight
        batch, num_agents, num_rollouts, horizon = preds.shape[:4]
        receiver_type = env.agent_type
        rows = torch.arange(batch).unsqueeze(1).expand(batch, num_agents)
        receivers = torch.arange(num_agents).unsqueeze(0).expand(batch, num_agents)
        per_receiver = torch.full((batch, num_agents, num_agents * num_rollouts), float("inf"))
        family_sets = []
        encoded = self._encode_broadcasts(preds, env.agent_type)
        for t in range(len(self.codecs)):
            if not bool((receiver_type == t).any()):
                family_sets.append(torch.zeros(batch, 1, num_agents * num_rollouts, horizon, num_agents, 2))
                continue
            decoded = self._decode_broadcasts(preds, encoded, receiver_type=t)
            union = decoded.reshape(batch, 1, num_agents * num_rollouts, horizon, num_agents, 2)
            family_sets.append(union)
            scores = _chunked_scores(union, start_u.unsqueeze(1), direction.unsqueeze(1), active.unsqueeze(1), cfg)
            scores = scores + lam * consistency_cost(union, prev, active, shift)
            scores = scores.expand(batch, num_agents, -1)
            per_receiver = torch.where((receiver_type == t).unsqueeze(-1), scores, per_receiver)
        own_scores = _chunked_scores(preds, start_u.unsqueeze(1).expand(batch, num_agents, -1),
                                     direction.unsqueeze(1).expand(batch, num_agents, -1),
                                     active.unsqueeze(1).expand(batch, num_agents, -1), cfg)
        own_scores = own_scores + lam * consistency_cost(preds, prev, active, shift)
        sets = torch.cat(family_sets, dim=1)
        per_receiver = per_receiver.view(batch, num_agents, num_agents, num_rollouts)
        eye = torch.eye(num_agents, dtype=torch.bool).view(1, num_agents, num_agents, 1)
        per_receiver = torch.where(eye, own_scores.unsqueeze(2).expand_as(per_receiver), per_receiver)
        per_receiver = torch.where(received.unsqueeze(-1), per_receiver, torch.full_like(per_receiver, float("inf")))
        best_scores, best = per_receiver.flatten(2).min(-1)
        sender = best // num_rollouts
        rollout = best % num_rollouts
        decoded_choice = sets[rows, receiver_type, best]
        native_choice = preds[rows, receivers, rollout]
        plans = torch.where((sender == receivers).view(batch, num_agents, 1, 1, 1), native_choice, decoded_choice)
        subsampled = sets[:, :, ::num_agents]
        beliefs = subsampled[rows, receiver_type].reshape(batch, num_agents, num_rollouts, -1)
        return plans, beliefs, best_scores

    def _record_agreement(self, plans: torch.Tensor, beliefs: torch.Tensor | None, active: torch.Tensor) -> None:
        """Plan disagreement (every replan) and pairwise W2 belief discrepancy (every few replans)."""
        batch, num_agents = active.shape
        act = active.cpu()
        pair = act.unsqueeze(1) & act.unsqueeze(2) & ~torch.eye(num_agents, dtype=torch.bool)
        agents = act.view(batch, 1, 1, 1, num_agents).float()
        diff = (plans.unsqueeze(1) - plans.unsqueeze(2)).square().sum(-1)
        rms = ((diff * agents).sum(dim=(-1, -2)) / (agents.sum(dim=(-1, -2)) * plans.shape[2]).clamp_min(1.0)).sqrt()
        num_pairs = pair.sum(dim=(-1, -2))
        disagreement = (rms * pair).sum(dim=(-1, -2)) / num_pairs.clamp_min(1)
        disagreement = torch.where(num_pairs > 0, disagreement, torch.full_like(disagreement, float("nan")))
        self.stats.plan_disagreement.append(disagreement)
        if self.belief_every <= 0 or self.replans % self.belief_every != 0:
            return
        if beliefs is None:
            value = torch.where(num_pairs > 0, torch.zeros(batch), torch.full((batch,), float("nan")))
            self.stats.belief_w2.append(value)
            return
        rows, cols = torch.triu_indices(num_agents, num_agents, offset=1)
        cost = squared_distances(beliefs[:, rows], beliefs[:, cols])
        scale = cost.mean(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
        coupling = sinkhorn_plan(cost / scale, self.ccfg.epsilon, self.ccfg.max_iters)
        w2 = ((coupling * cost).sum(dim=(-1, -2)) / (beliefs.shape[-1] // 2)).sqrt()
        valid = pair[:, rows, cols]
        value = (w2 * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        self.stats.belief_w2.append(torch.where(valid.any(-1), value, torch.full_like(value, float("nan"))))

    # ------------------------------------------------------------------------------------------
    def act(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        if self.plan is None or self.plan_step >= self.pcfg.replan_every:
            self._replan(env)
        num_agents = self.cfg.num_agents
        idx = torch.arange(num_agents)
        own = self.plan[:, idx, :, idx].permute(1, 0, 2, 3)
        elapsed = self.plan_step * self.cfg.dt
        ref = interpolate_plan(own, self.origin, elapsed, self.pcfg.plan_dt, self.cfg.dt, self.mpc.horizon_steps)
        self.plan_step += 1
        return self.mpc(env.pos, env.theta, env.speed, env.active, ref, env.direction)


def _chunked_scores(candidates: torch.Tensor, start_u: torch.Tensor, direction: torch.Tensor,
                    active: torch.Tensor, cfg: CorridorConfig) -> torch.Tensor:
    """Scores of ``(B, G, K, H, M, 2)`` candidates, chunked over ``B * G`` to bound memory."""
    batch, groups, num_cand, horizon, num_agents, _ = candidates.shape
    flat = candidates.reshape(batch * groups, num_cand, horizon, num_agents, 2)
    fu = start_u.reshape(batch * groups, num_agents)
    fd = direction.reshape(batch * groups, num_agents)
    fa = active.reshape(batch * groups, num_agents)
    per_row = num_cand * horizon * num_agents * num_agents
    chunk = max(1, int(2e7 // per_row))
    rows = [score_joint_plans(flat[i:i + chunk], fu[i:i + chunk], fd[i:i + chunk], fa[i:i + chunk], cfg)
            for i in range(0, flat.shape[0], chunk)]
    return torch.cat(rows).view(batch, groups, num_cand)


def advance_plan(previous: torch.Tensor, shift: int) -> torch.Tensor:
    """Advance ``(B, G, H, M, 2)`` plans by ``shift`` plan steps, extrapolating the final segment."""
    horizon = previous.shape[2]
    if shift <= 0:
        return previous
    shift = min(shift, horizon - 1)
    kept = previous[:, :, shift:]
    last = previous[:, :, -1:]
    step = previous[:, :, -1:] - previous[:, :, -2:-1]
    k = torch.arange(1, shift + 1, dtype=previous.dtype, device=previous.device).view(1, 1, -1, 1, 1)
    return torch.cat([kept, last + k * step], dim=2)


def consistency_cost(candidates: torch.Tensor, previous: torch.Tensor | None, active: torch.Tensor,
                     shift: int) -> torch.Tensor:
    """Mean squared deviation of candidates from the previous plan advanced by one replan period.

    Args:
        candidates: ``(B, Gc, K, H, M, 2)`` candidate joint plans (``Gc`` is 1 or ``G``).
        previous: ``(B, G, H, M, 2)`` plans selected at the previous replan, or ``None``.
        active: ``(B, M)`` agents that are still driving.
        shift: number of plan steps elapsed since the previous replan.

    Returns:
        ``(B, max(Gc, G), K)`` costs in square metres (zeros when there is no previous plan).
    """
    batch, groups_c, num_cand, horizon, num_agents, _ = candidates.shape
    if previous is None or shift >= horizon:
        return torch.zeros(batch, groups_c, num_cand, dtype=candidates.dtype)
    groups = max(groups_c, previous.shape[1])
    overlap = horizon - shift
    tail = previous[:, :, shift:]
    act = active.to(candidates.dtype).view(batch, 1, 1, 1, num_agents)
    denom = (act.sum(-1).view(batch, 1, 1) * overlap).clamp_min(1.0)
    per_row = groups * num_cand * overlap * num_agents * 2
    chunk = max(1, int(4e7 // per_row))
    out = []
    for i in range(0, batch, chunk):
        diff = candidates[i:i + chunk, :, :, :overlap] - tail[i:i + chunk].unsqueeze(2)
        sq = (diff.square().sum(-1) * act[i:i + chunk]).sum(dim=(-1, -2))
        out.append(sq / denom[i:i + chunk])
    return torch.cat(out)


def human_cost(candidates: torch.Tensor, human: torch.Tensor, reference: torch.Tensor | None,
               cfg: CorridorConfig, safety_margin: float = 0.2, weight: float = 1000.0) -> torch.Tensor:
    """Robot-pedestrian intrusion cost of ``(B, G, K, H, M, 2)`` candidates -> ``(B, G, K)``.

    Robots are taken from each candidate. Pedestrians are taken from the same candidate when
    ``reference`` is ``None``; otherwise from every one of the ``S`` futures in ``reference``
    ``(B, Gr, S, H, M, 2)`` (``Gr`` is 1 or ``G``) and the cost is averaged over them (expected intrusion).
    Every scene has the same number of pedestrians, so they can be gathered into a dense tensor.
    """
    batch, groups, num_cand, horizon, num_agents, _ = candidates.shape
    num_h = int(human[0].sum())
    order = torch.argsort(human.long(), dim=-1, stable=True)
    robot_idx, human_idx = order[:, :num_agents - num_h], order[:, num_agents - num_h:]

    def pick(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        shape = [x.shape[0]] + [1] * (x.dim() - 3) + [idx.shape[1], 1]
        return x.gather(-2, idx.view(*shape).expand(*x.shape[:-2], idx.shape[1], 2))

    robots = pick(candidates, robot_idx)
    threshold = 2.0 * cfg.agent_radius + safety_margin
    if reference is None:
        humans = pick(candidates, human_idx)
        dist = (robots.unsqueeze(-2) - humans.unsqueeze(-3)).norm(dim=-1)
        return weight * (threshold - dist).clamp_min(0.0).square().sum(dim=(-1, -2, -3))
    humans = pick(reference, human_idx)                             # (B, Gr, S, H, Nh, 2), Gr in {1, G}
    num_s, num_r = humans.shape[2], robots.shape[-2]
    # Per plan step: all-pairs distances between candidate robot positions and reference
    # pedestrian positions with a batched cdist (same value as broadcasting, far less memory traffic).
    out = torch.zeros(batch, groups, num_cand, dtype=candidates.dtype)
    # Exact pruning: a (plan step, robot, pedestrian) block can only contribute if the bounding box of
    # the robot's candidate positions comes within ``threshold`` of the box of the pedestrian's reference
    # positions; all other pairs are farther apart than the threshold and cost exactly zero.
    for b in range(batch):
        for g in range(groups):
            rb = robots[b, g]                                          # (K, H, R, 2)
            hb = humans[b, g if humans.shape[1] > 1 else 0]            # (S, H, Nh, 2)
            r_lo, r_hi = rb.amin(0), rb.amax(0)                        # (H, R, 2)
            h_lo, h_hi = hb.amin(0), hb.amax(0)                        # (H, Nh, 2)
            gap = (torch.maximum(r_lo.unsqueeze(2) - h_hi.unsqueeze(1), h_lo.unsqueeze(1) - r_hi.unsqueeze(2))
                   .clamp_min(0.0).norm(dim=-1))                       # (H, R, Nh)
            near = gap < threshold
            for j, k in near.any(0).nonzero().tolist():
                steps = near[:, j, k].nonzero().flatten()
                rob = rb[:, steps, j].transpose(0, 1)                  # (T', K, 2)
                ped = hb[:, steps, k].transpose(0, 1)                  # (T', S, 2)
                dist = torch.cdist(rob, ped, compute_mode="use_mm_for_euclid_dist")
                out[b, g] += dist.neg_().add_(threshold).clamp_(min=0.0).square_().sum(dim=(0, 2))
    return weight * out / num_s


def _gather_candidates(candidates: torch.Tensor, best: torch.Tensor) -> torch.Tensor:
    """Select ``candidates[b, g, best[b, g]]`` from ``(B, G, K, H, M, 2)`` -> ``(B, G, H, M, 2)``."""
    batch, groups = best.shape
    rows = torch.arange(batch).unsqueeze(1).expand(batch, groups)
    cols = torch.arange(groups).unsqueeze(0).expand(batch, groups)
    return candidates[rows, cols, best]
