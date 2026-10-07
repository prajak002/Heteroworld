"""Closed-form geometry of the bottleneck corridor (free-space width and lane reference)."""

from __future__ import annotations

import torch

from .config import CorridorConfig


def funnel_fraction(x: torch.Tensor, cfg: CorridorConfig) -> torch.Tensor:
    """Return 0 inside the gap, 1 outside the funnel, and a linear ramp in between."""
    return ((x.abs() - cfg.gap_half_length) / cfg.taper_length).clamp(0.0, 1.0)


def free_half_width(x: torch.Tensor, cfg: CorridorConfig) -> torch.Tensor:
    """Half-width of the free space at longitudinal position ``x``."""
    frac = funnel_fraction(x, cfg)
    return cfg.gap_half_width + (cfg.corridor_half_width - cfg.gap_half_width) * frac


def free_center(x: torch.Tensor, cfg: CorridorConfig) -> torch.Tensor:
    """Lateral centre of the free space: ``gap_offset`` inside the gap, 0 outside the funnel.

    A non-zero offset models a one-sided pinch (e.g. a tree pit at the curb, with the clear path
    along the building line).
    """
    return cfg.gap_offset * (1.0 - funnel_fraction(x, cfg))


def lane_reference_y(x: torch.Tensor, direction: torch.Tensor, cfg: CorridorConfig) -> torch.Tensor:
    """Lateral reference of an agent travelling in ``direction`` (+1 or -1) at position ``x``.

    Agents keep their lane centre (``direction * lane_offset``) away from the bottleneck and
    merge linearly to the centre of the gap through the funnel.
    """
    return free_center(x, cfg) + direction * cfg.lane_offset * funnel_fraction(x, cfg)


def wall_penetration(pos: torch.Tensor, cfg: CorridorConfig) -> torch.Tensor:
    """Signed penetration of an agent disc into the walls (positive means overlap)."""
    return (pos[..., 1] - free_center(pos[..., 0], cfg)).abs() + cfg.agent_radius - free_half_width(pos[..., 0], cfg)
