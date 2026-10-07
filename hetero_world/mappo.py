"""MAPPO baseline: shared decentralised actor, centralised critic, PPO with GAE.

The actor sees only local observations (own state, local free-space width, and the four nearest
agents within sensing range). The critic additionally receives a mean-pooled summary of every
agent's observation and global episode features (centralised training, decentralised execution).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .config import CorridorConfig
from .controllers import Controller
from .env import BottleneckCorridorEnv
from .geometry import free_half_width, lane_reference_y

NUM_NEIGHBOURS = 4
SENSING_RANGE = 6.0
OWN_FEATURES = 10
NEIGHBOUR_FEATURES = 6
OBS_DIM = OWN_FEATURES + NUM_NEIGHBOURS * NEIGHBOUR_FEATURES
CRITIC_DIM = 2 * OBS_DIM + 2


def build_observations(env: BottleneckCorridorEnv) -> torch.Tensor:
    """Local observation of every agent, shape ``(B, M, OBS_DIM)``, expressed in its travel frame."""
    cfg = env.cfg
    d = env.direction
    x, y = env.pos[..., 0], env.pos[..., 1]
    u = env.progress_coordinate
    heading = torch.where(d > 0, torch.zeros_like(d), torch.full_like(d, math.pi))
    rel_theta = env.theta - heading
    own = torch.stack([
        u / 10.0,
        y * d,
        torch.cos(rel_theta),
        torch.sin(rel_theta) * d,
        env.speed / cfg.v_max,
        (cfg.goal_distance - u) / 10.0,
        free_half_width(x, cfg),
        free_half_width(x + d * 1.0, cfg),
        free_half_width(x + d * 2.0, cfg),
        (y - lane_reference_y(x, d, cfg)) * d,
    ], dim=-1)

    active = env.active
    vel = env.speed.unsqueeze(-1) * torch.stack([torch.cos(env.theta), torch.sin(env.theta)], dim=-1)
    rel_pos = env.pos.unsqueeze(1) - env.pos.unsqueeze(2)
    rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)
    dist = rel_pos.norm(dim=-1)
    num_agents = cfg.num_agents
    eye = torch.eye(num_agents, dtype=torch.bool, device=env.pos.device)
    valid = active.unsqueeze(1) & ~eye & (dist < SENSING_RANGE)
    dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
    k = min(NUM_NEIGHBOURS, max(num_agents - 1, 1))
    nearest = dist.topk(k, dim=-1, largest=False)
    idx = nearest.indices
    nvalid = torch.isfinite(nearest.values)
    frame = d.unsqueeze(-1)

    def gather(t: torch.Tensor) -> torch.Tensor:
        return t.gather(2, idx.unsqueeze(-1).expand(*idx.shape, t.shape[-1]))

    npos = gather(rel_pos) * frame.unsqueeze(-1)
    nvel = gather(rel_vel) * frame.unsqueeze(-1)
    same = (d.unsqueeze(1).expand_as(dist).gather(2, idx) == d.unsqueeze(-1)).float()
    feats = torch.cat([npos / SENSING_RANGE, nvel / cfg.v_max, same.unsqueeze(-1), nvalid.float().unsqueeze(-1)], dim=-1)
    feats = torch.where(nvalid.unsqueeze(-1), feats, torch.zeros_like(feats))
    if k < NUM_NEIGHBOURS:
        pad = torch.zeros(*feats.shape[:2], NUM_NEIGHBOURS - k, NEIGHBOUR_FEATURES, device=feats.device)
        feats = torch.cat([feats, pad], dim=2)
    return torch.cat([own, feats.flatten(2)], dim=-1)


def build_critic_inputs(env: BottleneckCorridorEnv, obs: torch.Tensor) -> torch.Tensor:
    """Centralised critic input: own observation, mean of active agents' observations, globals."""
    active = env.active.float().unsqueeze(-1)
    pooled = (obs * active).sum(1, keepdim=True) / active.sum(1, keepdim=True).clamp_min(1.0)
    time_frac = (env.steps.float() / env.cfg.max_steps).view(-1, 1, 1).expand(-1, obs.shape[1], 1)
    done_frac = env.done.float().mean(-1).view(-1, 1, 1).expand(-1, obs.shape[1], 1)
    return torch.cat([obs, pooled.expand_as(obs), time_frac, done_frac], dim=-1)


def mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    """Two-hidden-layer tanh MLP."""
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))


class MAPPOAgent(nn.Module):
    """Gaussian actor over squashed (speed, yaw-rate) commands and a centralised value function."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.actor = mlp(OBS_DIM, hidden, 2)
        self.critic = mlp(CRITIC_DIM, 2 * hidden, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Action distribution over the pre-squash action."""
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp().expand(2))

    def value(self, critic_in: torch.Tensor) -> torch.Tensor:
        """State value."""
        return self.critic(critic_in).squeeze(-1)

    @staticmethod
    def squash(raw: torch.Tensor, cfg: CorridorConfig) -> tuple[torch.Tensor, torch.Tensor]:
        """Map raw actions to ``v_cmd in [0, v_max]`` and ``omega in [-omega_max, omega_max]``."""
        v_cmd = 0.5 * (torch.tanh(raw[..., 0]) + 1.0) * cfg.v_max
        omega = torch.tanh(raw[..., 1]) * cfg.omega_max
        return v_cmd, omega


@dataclass
class MAPPOConfig:
    """PPO hyper-parameters."""

    num_envs: int = 64
    rollout_steps: int = 64
    iterations: int = 400
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 8192
    lr: float = 3e-4
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    progress_reward: float = 1.0
    team_reward: float = 0.2
    collision_penalty: float = 2.0
    goal_reward: float = 5.0
    time_penalty: float = 0.01


def compute_rewards(env: BottleneckCorridorEnv, info, active_before: torch.Tensor, pcfg: MAPPOConfig) -> torch.Tensor:
    """Per-agent shaped reward for one step."""
    act = active_before.float()
    team = (info.progress * act).sum(-1, keepdim=True) / act.sum(-1, keepdim=True).clamp_min(1.0)
    collisions = (info.agent_collision | info.wall_collision).float()
    reward = (pcfg.progress_reward * info.progress + pcfg.team_reward * team
              - pcfg.collision_penalty * collisions + pcfg.goal_reward * info.reached_goal.float()
              - pcfg.time_penalty)
    return reward * act


def train_mappo(cfg: CorridorConfig, pcfg: MAPPOConfig, seed: int = 0, log_every: int = 50,
                checkpoint: Callable[[MAPPOAgent, list[dict], int], None] | None = None,
                checkpoint_every: int = 250) -> tuple[MAPPOAgent, list[dict]]:
    """Train MAPPO on the bottleneck corridor; returns the agent and a training log.

    ``checkpoint(agent, log, iteration)`` is called every ``checkpoint_every`` iterations.
    """
    torch.manual_seed(seed)
    agent = MAPPOAgent()
    optim = torch.optim.Adam(agent.parameters(), lr=pcfg.lr)
    env = BottleneckCorridorEnv(cfg, pcfg.num_envs, terminate_on_deadlock=False)
    next_seed = 1_000_000 + seed * 100_000
    env.reset(list(range(next_seed, next_seed + pcfg.num_envs)))
    next_seed += pcfg.num_envs
    log: list[dict] = []
    finished_eps = {"collision": [], "completed": [], "deadlock": []}
    start = time.time()
    for it in range(pcfg.iterations):
        obs_buf, cin_buf, raw_buf, logp_buf, val_buf, rew_buf, done_buf, mask_buf = ([] for _ in range(8))
        for _ in range(pcfg.rollout_steps):
            active = env.active.clone()
            obs = build_observations(env)
            cin = build_critic_inputs(env, obs)
            with torch.no_grad():
                dist = agent.distribution(obs)
                raw = dist.sample()
                logp = dist.log_prob(raw).sum(-1)
                value = agent.value(cin)
            v_cmd, omega = MAPPOAgent.squash(raw, cfg)
            info = env.step(v_cmd, omega)
            reward = compute_rewards(env, info, active, pcfg)
            done = info.reached_goal.clone()
            ended = env.over.clone()
            if bool(ended.any()):
                with torch.no_grad():
                    next_obs = build_observations(env)
                    bootstrap = agent.value(build_critic_inputs(env, next_obs))
                truncated = ended.unsqueeze(-1) & active & ~done
                reward = reward + pcfg.gamma * bootstrap * truncated.float()
                done = done | (ended.unsqueeze(-1) & active)
                metrics = env.episode_metrics()
                for key in finished_eps:
                    finished_eps[key].extend(metrics[key][ended].float().tolist())
                idx = ended.nonzero(as_tuple=True)[0]
                env.reset_indices(idx, list(range(next_seed, next_seed + idx.numel())))
                next_seed += idx.numel()
            done = done | ~active
            obs_buf.append(obs); cin_buf.append(cin); raw_buf.append(raw); logp_buf.append(logp)
            val_buf.append(torch.where(active, value, torch.zeros_like(value)))
            rew_buf.append(reward); done_buf.append(done.float()); mask_buf.append(active)
        with torch.no_grad():
            last_value = agent.value(build_critic_inputs(env, build_observations(env)))
            last_value = torch.where(env.active, last_value, torch.zeros_like(last_value))
        values = torch.stack(val_buf)
        rewards = torch.stack(rew_buf)
        dones = torch.stack(done_buf)
        adv = torch.zeros_like(rewards)
        gae = torch.zeros_like(last_value)
        next_value = last_value
        for t in reversed(range(pcfg.rollout_steps)):
            delta = rewards[t] + pcfg.gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + pcfg.gamma * pcfg.gae_lambda * (1.0 - dones[t]) * gae
            adv[t] = gae
            next_value = values[t]
        returns = adv + values
        mask = torch.stack(mask_buf)
        b_obs = torch.stack(obs_buf)[mask]
        b_cin = torch.stack(cin_buf)[mask]
        b_raw = torch.stack(raw_buf)[mask]
        b_logp = torch.stack(logp_buf)[mask]
        b_adv = adv[mask]
        b_ret = returns[mask]
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        num = b_obs.shape[0]
        for _ in range(pcfg.epochs):
            perm = torch.randperm(num)
            for s in range(0, num, pcfg.minibatch):
                mb = perm[s:s + pcfg.minibatch]
                dist = agent.distribution(b_obs[mb])
                logp = dist.log_prob(b_raw[mb]).sum(-1)
                ratio = (logp - b_logp[mb]).exp()
                pg = -torch.min(ratio * b_adv[mb], ratio.clamp(1 - pcfg.clip, 1 + pcfg.clip) * b_adv[mb]).mean()
                v_loss = (agent.value(b_cin[mb]) - b_ret[mb]).square().mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = pg + pcfg.value_coef * v_loss - pcfg.entropy_coef * entropy
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), pcfg.max_grad_norm)
                optim.step()
        if (it + 1) % log_every == 0 or it == 0:
            recent = {k: (sum(v[-200:]) / max(len(v[-200:]), 1)) for k, v in finished_eps.items()}
            entry = {"iteration": it + 1, "seconds": round(time.time() - start, 1),
                     "mean_step_reward": float((rewards.sum() / mask.sum()).item()),
                     "episodes": len(finished_eps["completed"]), **{f"recent_{k}": round(v, 3) for k, v in recent.items()}}
            log.append(entry)
            print(entry, flush=True)
        if checkpoint is not None and (it + 1) % checkpoint_every == 0:
            checkpoint(agent, log, it + 1)
    return agent, log


class MAPPOController(Controller):
    """Baseline 3: decentralised execution of the trained MAPPO actor (deterministic mean action)."""

    name = "mappo"

    def __init__(self, cfg: CorridorConfig, agent: MAPPOAgent) -> None:
        self.cfg = cfg
        self.agent = agent.eval()

    def act(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            raw = self.agent.actor(build_observations(env))
        v_cmd, omega = MAPPOAgent.squash(raw, self.cfg)
        active = env.active
        return torch.where(active, v_cmd, torch.zeros_like(v_cmd)), torch.where(active, omega, torch.zeros_like(omega))
