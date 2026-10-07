"""Learned heterogeneous world models of the corridor scene.

Three families with different architectures, objectives and private latent spaces. All of them map
the same observation ``obs (S, M, 7)`` (x, y, vx, vy, direction, v_des, done) to ``n`` sampled joint
futures ``(S, n, H, M, 2)`` in the shared physical trajectory space, which is the only interface the
consensus layer sees.

* :class:`DiffusionWM` (1024-wide denoiser): DDPM over the whole joint future, sampled with DDIM.
* :class:`JEPAWM` (512-D latent): joint-embedding predictive architecture. A context encoder and an
  EMA target encoder of the future; a multi-hypothesis predictor regresses the target embedding
  (winner-takes-all over ``K`` modes plus a mode classifier). Futures are read out with a probe
  trained on the frozen target embeddings, as is standard for JEPA models.
* :class:`RSSMWM` (128-D deterministic + 32-D stochastic state): recurrent state-space model trained
  with a sequential ELBO and rolled out with its learned prior.

Futures are predicted as displacements from the current positions. Agents that already reached
their goal are pinned to their current position.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

POS_SCALE = 10.0   # metres -> network units for positions
DISP_SCALE = 4.0   # metres -> network units for displacements


def encode_obs(obs: torch.Tensor) -> torch.Tensor:
    """Normalise ``(S, M, 7)`` observations and flatten them to ``(S, 7 M)``."""
    x = obs.clone()
    x[..., 0:2] = x[..., 0:2] / POS_SCALE
    x[..., 2:4] = x[..., 2:4] / 1.5
    x[..., 5] = x[..., 5] / 1.5
    return x.flatten(1)


def mlp(dims: list[int], act=nn.SiLU, last_act: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    """Common interface: ``loss(obs, future)`` for training and ``sample(obs, n)`` for planning."""

    family = "base"
    latent_dim = 0

    def __init__(self, num_agents: int, horizon: int) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.horizon = horizon
        self.obs_dim = 7 * num_agents
        self.out_dim = 2 * num_agents * horizon

    def to_target(self, obs: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """Absolute future ``(S, H, M, 2)`` -> normalised displacement vector ``(S, 2 H M)``."""
        return ((future - obs[:, None, :, 0:2]) / DISP_SCALE).flatten(1)

    def to_future(self, obs: torch.Tensor, flat: torch.Tensor, n: int) -> torch.Tensor:
        """Normalised displacements ``(S n, 2 H M)`` -> absolute futures ``(S, n, H, M, 2)``."""
        disp = flat.view(-1, n, self.horizon, self.num_agents, 2) * DISP_SCALE
        start = obs[:, None, None, :, 0:2]
        done = obs[:, None, None, :, 6:7] > 0.5
        return torch.where(done, start.expand_as(disp), start + disp)


# ----------------------------------------------------------------------------------------------
class DiffusionWM(WorldModel):
    """DDPM over joint futures with an MLP denoiser of width 1024 (epsilon prediction)."""

    family = "diffusion"
    latent_dim = 1024

    def __init__(self, num_agents: int, horizon: int, width: int = 1024, steps: int = 100,
                 sample_steps: int = 10) -> None:
        super().__init__(num_agents, horizon)
        self.steps = steps
        self.sample_steps = sample_steps
        self.ctx = mlp([self.obs_dim, 256, 256], last_act=True)
        self.temb = mlp([64, 128, 128], last_act=True)
        self.net = mlp([self.out_dim + 256 + 128, width, width, width, self.out_dim])
        s = 0.008
        t = torch.linspace(0, steps, steps + 1) / steps
        abar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        abar = (abar / abar[0]).clamp(1e-5, 1.0)
        self.register_buffer("abar", abar[1:])

    def _temb(self, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.exp(-math.log(1000.0) * torch.arange(32, device=t.device) / 32)
        ang = t.float().unsqueeze(-1) * freqs
        return self.temb(torch.cat([ang.sin(), ang.cos()], dim=-1))

    def _eps(self, x: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, ctx, self._temb(t)], dim=-1))

    def loss(self, obs: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        x0 = self.to_target(obs, future)
        t = torch.randint(0, self.steps, (x0.shape[0],), device=x0.device)
        a = self.abar[t].unsqueeze(-1)
        noise = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        return F.mse_loss(self._eps(xt, t, self.ctx(encode_obs(obs))), noise)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, n: int, gen: torch.Generator | None = None) -> torch.Tensor:
        ctx = self.ctx(encode_obs(obs)).repeat_interleave(n, dim=0)
        x = torch.randn(ctx.shape[0], self.out_dim, generator=gen, device="cpu").to(ctx)
        ts = torch.linspace(self.steps - 1, 0, self.sample_steps).round().long().tolist()
        for i, t in enumerate(ts):
            a = self.abar[t]
            tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            eps = self._eps(x, tt, ctx)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-5, 5)
            a_next = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
            x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps          # deterministic DDIM step
        return self.to_future(obs, x, n)


# ----------------------------------------------------------------------------------------------
class JEPAWM(WorldModel):
    """Joint-embedding predictive world model with a 512-D latent and a multi-hypothesis predictor."""

    family = "jepa"
    latent_dim = 512

    def __init__(self, num_agents: int, horizon: int, modes: int = 8, ema: float = 0.996,
                 latent_noise: float = 0.05) -> None:
        super().__init__(num_agents, horizon)
        d = self.latent_dim
        self.modes = modes
        self.ema = ema
        self.latent_noise = latent_noise
        self.context = mlp([self.obs_dim, 512, 512, d])
        self.future_enc = mlp([self.out_dim, 512, 512, d])
        self.target_enc = copy.deepcopy(self.future_enc)
        for p in self.target_enc.parameters():
            p.requires_grad_(False)
        self.predictor = mlp([d, 1024, 1024, modes * d])
        self.mode_head = mlp([d, 256, modes])
        self.probe = mlp([d, 512, 512, self.out_dim])

    @torch.no_grad()
    def update_target(self) -> None:
        for p_t, p_o in zip(self.target_enc.parameters(), self.future_enc.parameters()):
            p_t.mul_(self.ema).add_(p_o.detach(), alpha=1 - self.ema)

    def loss(self, obs: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        y = self.to_target(obs, future)
        c = self.context(encode_obs(obs))
        with torch.no_grad():
            target = self.target_enc(y)
        online = self.future_enc(y)
        pred = self.predictor(c).view(-1, self.modes, self.latent_dim)
        err = (pred - target.unsqueeze(1)).square().mean(-1)
        best = err.argmin(dim=1)
        jepa = err.gather(1, best.unsqueeze(1)).mean()
        mode_ce = F.cross_entropy(self.mode_head(c), best)
        # Anti-collapse: keep every online embedding dimension at unit-scale variance (VICReg term)
        # and align the online encoder with the predictor's winning hypothesis.
        std = online.std(dim=0)
        variance = F.relu(1.0 - std).mean()
        align = (online - pred.detach()[torch.arange(len(best)), best]).square().mean()
        # Probe: read the future out of the frozen target embedding.
        probe = F.mse_loss(self.probe(target), y)
        return jepa + 0.1 * mode_ce + variance + 0.1 * align + probe

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, n: int, gen: torch.Generator | None = None) -> torch.Tensor:
        c = self.context(encode_obs(obs))
        pred = self.predictor(c).view(-1, self.modes, self.latent_dim)
        probs = F.softmax(self.mode_head(c), dim=-1)
        idx = torch.multinomial(probs.cpu(), n, replacement=True, generator=gen).to(c.device)
        z = pred.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.latent_dim))
        noise = torch.randn(z.shape, generator=gen, device="cpu").to(z)
        z = z + self.latent_noise * z.std(dim=-1, keepdim=True) * noise
        return self.to_future(obs, self.probe(z.flatten(0, 1)), n)


# ----------------------------------------------------------------------------------------------
class RSSMWM(WorldModel):
    """Recurrent state-space model: GRU state (128) plus a 32-D Gaussian latent per plan step."""

    family = "rssm"
    latent_dim = 128

    def __init__(self, num_agents: int, horizon: int, stoch: int = 32, free_nats: float = 0.5) -> None:
        super().__init__(num_agents, horizon)
        h, m = self.latent_dim, 2 * num_agents
        self.stoch = stoch
        self.free_nats = free_nats
        self.init = mlp([self.obs_dim, 256, h], last_act=False)
        self.cell = nn.GRUCell(stoch + m, h)
        self.prior = mlp([h, 256, 2 * stoch])
        self.post = mlp([h + m, 256, 2 * stoch])
        self.decoder = mlp([h + stoch, 256, 256, m])

    @staticmethod
    def _dist(stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw = stats.chunk(2, dim=-1)
        return mean, F.softplus(raw) + 0.05

    def loss(self, obs: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        y = self.to_target(obs, future).view(-1, self.horizon, 2 * self.num_agents)
        steps = torch.diff(y, dim=1, prepend=torch.zeros_like(y[:, :1]))
        h = torch.tanh(self.init(encode_obs(obs)))
        prev = torch.zeros_like(steps[:, 0])
        recon, kl = 0.0, 0.0
        for t in range(self.horizon):
            pm, ps = self._dist(self.prior(h))
            qm, qs = self._dist(self.post(torch.cat([h, steps[:, t]], dim=-1)))
            z = qm + qs * torch.randn_like(qs)
            out = self.decoder(torch.cat([h, z], dim=-1))
            recon = recon + F.mse_loss(out, steps[:, t], reduction="none").sum(-1).mean()
            kl_t = (torch.log(ps / qs) + (qs.square() + (qm - pm).square()) / (2 * ps.square()) - 0.5).sum(-1)
            kl = kl + kl_t.mean().clamp_min(self.free_nats)
            h = self.cell(torch.cat([z, steps[:, t]], dim=-1), h)
            prev = steps[:, t]
        return (recon * 50.0 + kl) / self.horizon

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, n: int, gen: torch.Generator | None = None) -> torch.Tensor:
        h = torch.tanh(self.init(encode_obs(obs))).repeat_interleave(n, dim=0)
        pos = torch.zeros(h.shape[0], 2 * self.num_agents, device=h.device)
        outs = []
        for _ in range(self.horizon):
            pm, ps = self._dist(self.prior(h))
            z = pm + ps * torch.randn(pm.shape, generator=gen, device="cpu").to(pm)
            step = self.decoder(torch.cat([h, z], dim=-1))
            pos = pos + step
            outs.append(pos)
            h = self.cell(torch.cat([z, step], dim=-1), h)
        return self.to_future(obs, torch.stack(outs, dim=1).flatten(1), n)


FAMILIES = {"diffusion": DiffusionWM, "jepa": JEPAWM, "rssm": RSSMWM}


def build(family: str, num_agents: int, horizon: int) -> WorldModel:
    return FAMILIES[family](num_agents, horizon)


# ----------------------------------------------------------------------------------------------
def observe(env) -> torch.Tensor:
    """Observed scene features ``(B, M, 7)`` shared by every world model (also used for training data)."""
    cfg = env.cfg
    vel = env.speed.unsqueeze(-1) * torch.stack([torch.cos(env.theta), torch.sin(env.theta)], dim=-1)
    goal = torch.stack([env.direction * cfg.goal_distance, env.direction * cfg.lane_offset], dim=-1)
    pos = torch.where(env.done.unsqueeze(-1), goal, env.pos)
    vel = torch.where(env.done.unsqueeze(-1), torch.zeros_like(vel), vel)
    return torch.cat([pos, vel, env.direction.unsqueeze(-1), env.v_des.unsqueeze(-1),
                      env.done.float().unsqueeze(-1)], dim=-1)


class LearnedWorldModelFleet:
    """Drop-in replacement for the synthetic fleet that samples from the trained networks.

    Every observer runs the network of its family on the shared observation and draws its own
    ``num_rollouts`` samples. Agents that already finished keep their simulator (parked) position.
    """

    def __init__(self, num_rollouts: int, checkpoint_dir: str = "checkpoints",
                 names: tuple[str, ...] = ("diffusion", "jepa", "rssm")) -> None:
        self.num_rollouts = num_rollouts
        self.names = names
        self.models = []
        for name in names:
            ckpt = torch.load(f"{checkpoint_dir}/wm_{name}.pt", map_location="cpu")
            model = build(name, ckpt["num_agents"], ckpt["horizon"])
            model.load_state_dict(ckpt["state_dict"])
            self.models.append(model.eval())

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, parked: torch.Tensor, agent_type: torch.Tensor,
                gen: torch.Generator) -> torch.Tensor:
        """``obs (B, M, 7)``, ``parked (B, M, 2)`` -> rollouts ``(B, M, N, H, M, 2)`` per observer."""
        num_scenes, num_agents = agent_type.shape
        n = self.num_rollouts
        rows = torch.arange(num_scenes).repeat_interleave(num_agents)
        flat_type = agent_type.reshape(-1).cpu()
        horizon = self.models[0].horizon
        out = torch.empty(num_scenes * num_agents, n, horizon, num_agents, 2)
        for k, model in enumerate(self.models):
            sel = (flat_type == k).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            out[sel] = model.sample(obs[rows[sel]].float().cpu(), n, gen)
        done = obs[..., 6] > 0.5
        out = out.view(num_scenes, num_agents, n, horizon, num_agents, 2)
        mask = done.view(num_scenes, 1, 1, 1, num_agents, 1).cpu()
        return torch.where(mask, parked.view(num_scenes, 1, 1, 1, num_agents, 2).cpu().to(out), out)
