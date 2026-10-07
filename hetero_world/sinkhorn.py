"""Entropic optimal-transport consensus: batched log-domain Sinkhorn and free-support barycenters.

Notation (matches Section 3 of the paper):

* ``measures``: ``(B, M, N, D)`` atoms ``x_i^(n)`` of the empirical measures ``mu_i`` (uniform weights).
* ``support``: ``(B, N, D)`` atoms ``y^(a)`` of the barycenter ``nu`` (uniform weights).
* ``alpha``: ``(B, M)`` barycentric weights on the simplex (zero for missing measures).
* Ground cost ``c(x, y) = ||x - y||^2``. Costs are divided by a per-scene scale ``s`` (the
  alpha-weighted mean cost at initialisation) so that ``epsilon`` is dimensionless.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass
class BarycenterOutput:
    """Result of :class:`SinkhornBarycenterConsensus`.

    Attributes:
        support: ``(B, N, D)`` atoms of the consensus measure ``nu*``.
        alignment_cost: ``(B, M)`` transport cost ``<P_i, C_i>`` between ``nu*`` and each ``mu_i``
            (squared trajectory units, an upper bound on ``W_2^2`` up to the marginal error).
        w2_per_point: ``(B, M)`` the same quantity as an RMS distance per 2-D waypoint (metres).
        marginal_error: ``(B, M)`` L1 violation of the first marginal at the last Sinkhorn call.
        weights: ``(B, M)`` normalised barycentric weights that were used.
        plans: ``(B, M, N, N)`` final couplings when requested, otherwise ``None``.
    """

    support: torch.Tensor
    alignment_cost: torch.Tensor
    w2_per_point: torch.Tensor
    marginal_error: torch.Tensor
    weights: torch.Tensor
    plans: torch.Tensor | None


def squared_distances(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distances between ``y (..., N, D)`` and ``x (..., K, D)`` -> ``(..., N, K)``."""
    y_sq = (y * y).sum(-1, keepdim=True)
    x_sq = (x * x).sum(-1).unsqueeze(-2)
    return (y_sq + x_sq - 2.0 * (y @ x.transpose(-1, -2))).clamp_min(0.0)


def log_sinkhorn(cost: torch.Tensor, epsilon: float, iters: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched log-domain Sinkhorn with uniform marginals.

    Solves ``min_P <P, C> - epsilon H(P)`` subject to ``P 1 = 1/N`` and ``P^T 1 = 1/K`` through the
    dual updates ``f <- -eps LSE_k((g_k - C_nk)/eps) + eps log a`` and the symmetric update for ``g``.
    Potentials are returned divided by ``epsilon``.

    Args:
        cost: ``(..., N, K)`` non-negative cost matrices.
        epsilon: entropic regularisation (in the units of ``cost``).
        iters: number of full (f, g) updates.

    Returns:
        ``f / eps`` of shape ``(..., N)``, ``g / eps`` of shape ``(..., K)`` and ``log P`` ``(..., N, K)``.
    """
    n, k = cost.shape[-2], cost.shape[-1]
    log_a = -math.log(n)
    log_b = -math.log(k)
    scaled = cost / epsilon
    g = torch.zeros(cost.shape[:-2] + (k,), dtype=cost.dtype, device=cost.device)
    f = torch.zeros(cost.shape[:-1], dtype=cost.dtype, device=cost.device)
    for _ in range(iters):
        f = log_a - torch.logsumexp(g.unsqueeze(-2) - scaled, dim=-1)
        g = log_b - torch.logsumexp(f.unsqueeze(-1) - scaled, dim=-2)
    log_plan = f.unsqueeze(-1) + g.unsqueeze(-2) - scaled
    return f, g, log_plan


KERNEL_EXPONENT_LIMIT = 650.0


def kernel_sinkhorn(cost: torch.Tensor, epsilon: float, iters: int) -> torch.Tensor | None:
    """Kernel-domain Sinkhorn in float64 with row-minimum absorption.

    Computes the same iterates as :func:`log_sinkhorn` using matrix-vector products with the Gibbs
    kernel ``K = exp(-(C - min_k C)/eps)``, which is much cheaper than ``logsumexp`` on CPUs. Returns
    ``None`` when the exponent range would overflow float64 so that the caller can fall back to
    the log domain.

    Returns:
        The coupling ``P`` of shape ``(..., N, K)`` in the dtype of ``cost``, or ``None``.
    """
    n, k = cost.shape[-2], cost.shape[-1]
    c = cost.double()
    shifted = (c - c.amin(dim=-1, keepdim=True)) / epsilon
    if float(shifted.amax()) > KERNEL_EXPONENT_LIMIT:
        return None
    kernel = torch.exp(-shifted)
    kernel_t = kernel.transpose(-1, -2)
    v = torch.ones(c.shape[:-2] + (k, 1), dtype=c.dtype, device=c.device)
    u = torch.ones(c.shape[:-2] + (n, 1), dtype=c.dtype, device=c.device)
    for _ in range(iters):
        u = (1.0 / n) / (kernel @ v)
        v = (1.0 / k) / (kernel_t @ u)
    plan = u * kernel * v.transpose(-1, -2)
    if not bool(torch.isfinite(plan).all()):
        return None
    return plan.to(cost.dtype)


def sinkhorn_plan(cost: torch.Tensor, epsilon: float, iters: int, backend: str = "auto") -> torch.Tensor:
    """Entropic coupling with uniform marginals using the requested backend.

    Args:
        backend: ``"log"`` (any device and dtype), ``"kernel"`` (float64 matrix-vector products,
            falls back to ``"log"`` on overflow) or ``"auto"`` (kernel on CPU, log elsewhere).
    """
    use_kernel = backend == "kernel" or (backend == "auto" and cost.device.type == "cpu")
    if not use_kernel:
        _, _, log_plan = log_sinkhorn(cost, epsilon, iters)
        return log_plan.exp()
    batch_shape = cost.shape[:-2]
    flat = cost.reshape(-1, *cost.shape[-2:])
    exponent = (flat - flat.amin(dim=-1, keepdim=True)).amax(dim=(-1, -2)) / epsilon
    safe = exponent <= KERNEL_EXPONENT_LIMIT
    plan = torch.empty_like(flat)
    if bool(safe.any()):
        kernel_plan = kernel_sinkhorn(flat[safe], epsilon, iters)
        if kernel_plan is None:
            safe = torch.zeros_like(safe)
        else:
            plan[safe] = kernel_plan
    if bool((~safe).any()):
        _, _, log_plan = log_sinkhorn(flat[~safe], epsilon, iters)
        plan[~safe] = log_plan.exp()
    return plan.view(*batch_shape, *cost.shape[-2:])


class SinkhornBarycenterConsensus(nn.Module):
    """Free-support entropic Wasserstein barycenter of ``M`` empirical trajectory measures.

    The barycenter keeps ``N`` uniformly weighted atoms. Each outer iteration solves ``M``
    entropic transport problems in one batched call and moves the atoms to the alpha-weighted
    barycentric projections (Cuturi & Doucet, 2014, Alg. 2 with step size 1):

        y^(a) <- sum_i alpha_i (sum_n P_i[a, n] x_i^(n)) / (sum_n P_i[a, n]).

    Dividing by the realised row mass keeps the projection unbiased when a truncated Sinkhorn
    run has not fully converged.

    Args:
        epsilon: dimensionless entropic regularisation (applied to the normalised cost).
        max_iters: Sinkhorn iterations per transport solve.
        outer_iters: free-support fixed-point iterations.
        backend: Sinkhorn backend, see :func:`sinkhorn_plan`.
    """

    def __init__(self, epsilon: float = 0.01, max_iters: int = 50, outer_iters: int = 3,
                 backend: str = "auto") -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if max_iters < 1 or outer_iters < 1:
            raise ValueError("iteration counts must be positive")
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.outer_iters = outer_iters
        self.backend = backend

    def forward(self, measures: torch.Tensor, alpha: torch.Tensor | None = None,
                return_plans: bool = False) -> BarycenterOutput:
        """Compute the consensus measure.

        Args:
            measures: ``(B, M, N, D)`` or ``(M, N, D)`` atoms; ``D`` is the flattened trajectory
                dimension (``2 H M`` for joint-scene rollouts).
            alpha: ``(B, M)`` or ``(M,)`` non-negative weights; zero excludes a measure (for
                example a broadcast lost in transit). Defaults to uniform weights.
            return_plans: also return the final couplings.

        Returns:
            :class:`BarycenterOutput`; a leading batch dimension of 1 is added for 3-D inputs.
        """
        if measures.dim() == 3:
            measures = measures.unsqueeze(0)
            alpha = None if alpha is None else alpha.unsqueeze(0)
        batch, num_measures, num_atoms, dim = measures.shape
        if alpha is None:
            alpha = torch.ones(batch, num_measures, dtype=measures.dtype, device=measures.device)
        alpha = alpha.to(measures.dtype)
        if bool((alpha.sum(-1) <= 0).any()):
            raise ValueError("every scene needs at least one measure with positive weight")
        alpha = alpha / alpha.sum(-1, keepdim=True)

        first = alpha.argmax(dim=-1)
        support = measures[torch.arange(batch, device=measures.device), first]
        scale = None
        plan = None
        cost = None
        for _ in range(self.outer_iters):
            cost = squared_distances(support.unsqueeze(1), measures)
            if scale is None:
                scale = ((cost.mean(dim=(-1, -2)) * alpha).sum(-1)).clamp_min(1e-12)
            plan = sinkhorn_plan(cost / scale.view(-1, 1, 1, 1), self.epsilon, self.max_iters, self.backend)
            row_mass = plan.sum(-1, keepdim=True).clamp_min(1e-30)
            projection = (plan @ measures) / row_mass
            support = (alpha.view(batch, num_measures, 1, 1) * projection).sum(dim=1)

        final_cost = squared_distances(support.unsqueeze(1), measures)
        alignment = (plan * final_cost).sum(dim=(-1, -2))
        num_points = max(dim // 2, 1)
        marginal_error = (plan.sum(-1) - 1.0 / num_atoms).abs().sum(-1)
        return BarycenterOutput(
            support=support,
            alignment_cost=alignment,
            w2_per_point=(alignment / num_points).clamp_min(0.0).sqrt(),
            marginal_error=marginal_error,
            weights=alpha,
            plans=plan if return_plans else None,
        )


def pairwise_sinkhorn_costs(measures: torch.Tensor, epsilon: float = 0.01, max_iters: int = 50,
                            backend: str = "auto") -> torch.Tensor:
    """All-pairs entropic transport costs, the ``O(M^2)`` alternative to a barycenter.

    Args:
        measures: ``(B, M, N, D)`` atoms.

    Returns:
        ``(B, M, M)`` symmetric matrix of costs ``<P_ij, C_ij>`` (zero diagonal).
    """
    batch, num_measures, num_atoms, _ = measures.shape
    rows, cols = torch.triu_indices(num_measures, num_measures, offset=1, device=measures.device)
    out = torch.zeros(batch, num_measures, num_measures, dtype=measures.dtype, device=measures.device)
    if rows.numel() == 0:
        return out
    cost = squared_distances(measures[:, rows], measures[:, cols])
    scale = cost.mean(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    plan = sinkhorn_plan(cost / scale, epsilon, max_iters, backend)
    values = (plan * cost).sum(dim=(-1, -2))
    out[:, rows, cols] = values
    out[:, cols, rows] = values
    return out


def synchronize(device: torch.device) -> None:
    """Block until queued kernels on ``device`` have finished."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_latency_ms(fn: Callable[[], object], device: torch.device, warmup: int = 5,
                       repeats: int = 20) -> list[float]:
    """Wall-clock latency of ``fn`` in milliseconds.

    Uses ``torch.cuda.Event`` timing on CUDA devices and ``time.perf_counter`` with explicit
    synchronisation elsewhere.
    """
    for _ in range(warmup):
        fn()
    synchronize(device)
    timings: list[float] = []
    if device.type == "cuda":
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end))
        return timings
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        timings.append(1e3 * (time.perf_counter() - t0))
    return timings
