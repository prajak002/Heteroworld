"""Correctness and latency tests for the Sinkhorn barycenter consensus module."""

from __future__ import annotations

import os
import statistics

import pytest
import torch

from hetero_world.dataset import HeteroTrajectoryDataset
from hetero_world.sinkhorn import (SinkhornBarycenterConsensus, log_sinkhorn, measure_latency_ms,
                                   pairwise_sinkhorn_costs, sinkhorn_plan)

LATENCY_BUDGET_MS = 3.0
CPU_BUDGET_MS = float(os.environ.get("HETERO_CPU_LATENCY_BUDGET_MS", "15.0"))


def _benchmark_measures(device: torch.device) -> torch.Tensor:
    """Ten 100-atom joint-trajectory measures (M=10, N=100, D=400) from the dataset pipeline."""
    sample = HeteroTrajectoryDataset(num_scenes=1, seed=3)[0]
    return sample.rollouts.reshape(10, 100, -1).to(device)


def test_kernel_and_log_backends_agree() -> None:
    torch.manual_seed(0)
    cost = torch.rand(4, 30, 30, dtype=torch.float64)
    p_log = sinkhorn_plan(cost, 0.05, 200, backend="log")
    p_ker = sinkhorn_plan(cost, 0.05, 200, backend="kernel")
    assert torch.allclose(p_log, p_ker, atol=1e-10)


def test_plan_marginals_converge() -> None:
    torch.manual_seed(1)
    cost = torch.rand(2, 50, 50, dtype=torch.float64)
    _, _, log_plan = log_sinkhorn(cost, 0.1, 500)
    plan = log_plan.exp()
    assert torch.allclose(plan.sum(-1), torch.full((2, 50), 1 / 50, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(plan.sum(-2), torch.full((2, 50), 1 / 50, dtype=torch.float64), atol=1e-12)


def test_barycenter_of_translated_clouds_is_midpoint() -> None:
    torch.manual_seed(2)
    base = torch.randn(64, 2, dtype=torch.float64)
    shift = torch.tensor([4.0, -2.0], dtype=torch.float64)
    measures = torch.stack([base, base + shift])
    out = SinkhornBarycenterConsensus(epsilon=0.001, max_iters=3000, outer_iters=3)(measures)
    expected = base + 0.5 * shift
    assert torch.allclose(out.support[0].mean(0), expected.mean(0), atol=1e-4)
    assert (out.support[0] - expected).norm(dim=-1).max() < 0.08


def test_entropic_blur_grows_with_epsilon() -> None:
    torch.manual_seed(2)
    base = torch.randn(64, 2, dtype=torch.float64)
    measures = torch.stack([base, base + 3.0])
    errors = []
    for eps in (0.003, 0.03, 0.3):
        out = SinkhornBarycenterConsensus(epsilon=eps, max_iters=2000, outer_iters=3)(measures)
        errors.append(float((out.support[0] - (base + 1.5)).norm(dim=-1).mean()))
    assert errors[0] < errors[1] < errors[2]


def test_weights_select_measure() -> None:
    torch.manual_seed(3)
    a = torch.randn(32, 4, dtype=torch.float64)
    b = a + 10.0
    out = SinkhornBarycenterConsensus(epsilon=0.001, max_iters=300, outer_iters=2)(
        torch.stack([a, b]), torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert torch.allclose(out.support[0], a, atol=1e-6)


def test_pairwise_costs_symmetric() -> None:
    torch.manual_seed(4)
    measures = torch.randn(1, 4, 20, 6)
    costs = pairwise_sinkhorn_costs(measures, 0.05, 100)
    assert torch.allclose(costs, costs.transpose(-1, -2))
    assert torch.all(costs.diagonal(dim1=-2, dim2=-1) == 0)


def test_output_shapes() -> None:
    measures = _benchmark_measures(torch.device("cpu"))
    out = SinkhornBarycenterConsensus()(measures)
    assert out.support.shape == (1, 100, 400)
    assert out.alignment_cost.shape == (1, 10)
    assert torch.isfinite(out.support).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required for torch.cuda.Event timing")
def test_latency_cuda_event_under_3ms() -> None:
    """Median barycenter latency (M=10, N=100, eps=0.01, 50 iterations) measured with CUDA events."""
    device = torch.device("cuda")
    measures = _benchmark_measures(device)
    module = SinkhornBarycenterConsensus(epsilon=0.01, max_iters=50, outer_iters=2, backend="log")
    timings = measure_latency_ms(lambda: module(measures), device, warmup=10, repeats=50)
    median = statistics.median(timings)
    print(f"CUDA median latency {median:.3f} ms")
    assert median < LATENCY_BUDGET_MS


def test_latency_cpu_budget() -> None:
    """CPU latency check; the budget is set by HETERO_CPU_LATENCY_BUDGET_MS (default 15 ms)."""
    device = torch.device("cpu")
    measures = _benchmark_measures(device)
    module = SinkhornBarycenterConsensus(epsilon=0.01, max_iters=50, outer_iters=2)
    timings = measure_latency_ms(lambda: module(measures), device, warmup=5, repeats=30)
    median = statistics.median(timings)
    print(f"CPU median latency {median:.3f} ms (budget {CPU_BUDGET_MS} ms; 3 ms target applies to GPU)")
    assert median < CPU_BUDGET_MS
