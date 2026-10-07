"""Render the three paper figures from simulator data and saved results.

Figure 1: joint-future beliefs of three heterogeneous world models and their W2 barycenter (space-time view).
Figure 2: V2X payload rate per agent versus agent count M.
Figure 3: per-agent consensus latency versus agent count M (up to 50).

Usage:
    python scripts/make_figures.py --results results --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hetero_world.benchmark import EVAL_SEED_BASE  # noqa: E402
from hetero_world.config import ConsensusConfig, CorridorConfig, PredictionConfig  # noqa: E402
from hetero_world.controllers import BYTES_PER_VALUE, PlanConsensusController  # noqa: E402
from hetero_world.env import BottleneckCorridorEnv  # noqa: E402
from hetero_world.planner import select_joint_plan  # noqa: E402
from hetero_world.world_models import WORLD_MODEL_NAMES, WORLD_MODEL_SPECS, scene_from_env  # noqa: E402

# Reference categorical palette (light mode), fixed slot order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                      "#e87ba4", "#008300", "#4a3aa7")
INK, INK_2, GRID, ZONE = "#0b0b0b", "#52514e", "#e4e3df", "#f1efe9"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def figure_consensus(out: Path, seed_offset: int = 3, warmup: int = 0) -> None:
    """Figure 1: x(t) of every agent under three world models' beliefs and under the barycenter."""
    cfg, pcfg, ccfg = CorridorConfig(), PredictionConfig(), ConsensusConfig(outer_iters=2)
    ctrl = PlanConsensusController("hetero_world", cfg, pcfg, ccfg, belief_every=0)
    env = BottleneckCorridorEnv(cfg, 1)
    env.reset([EVAL_SEED_BASE + seed_offset])
    for _ in range(warmup):
        env.step(env.v_des.clone(), torch.zeros_like(env.v_des))
    gen = torch.Generator().manual_seed(7)
    preds = ctrl.fleet.predict(scene_from_env(env), env.agent_type, gen)[0]
    flat = preds.reshape(cfg.num_agents, pcfg.num_rollouts, -1)
    out_b = ctrl.consensus(flat)
    support = out_b.support.view(pcfg.num_rollouts, pcfg.horizon, cfg.num_agents, 2)
    chosen = select_joint_plan(support.unsqueeze(0), env.progress_coordinate, env.direction, env.active, cfg)[0]
    types = env.agent_type[0].tolist()
    observers = [types.index(k) for k in range(3)]
    tau = pcfg.plan_dt * torch.arange(0, pcfg.horizon + 1)
    x0 = env.pos[0, :, 0]
    direction = env.direction[0]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True)
    panels = [(f"$\\mu_{{{i}}}$: {WORLD_MODEL_NAMES[types[i]]} ({WORLD_MODEL_SPECS[WORLD_MODEL_NAMES[types[i]]].latent_dim}-D)",
               preds[i], float(out_b.w2_per_point[0, i])) for i in observers]
    panels.append(("barycenter $\\nu^\\star$", support, None))
    for ax, (title, rollouts, w2) in zip(axes, panels):
        ax.axhspan(-cfg.zone_half_length, cfg.zone_half_length, color=ZONE, zorder=0, lw=0)
        for n in range(0, pcfg.num_rollouts, 2):
            for j in range(cfg.num_agents):
                xs = torch.cat([x0[j:j + 1], rollouts[n, :, j, 0]])
                ax.plot(tau, xs, color=BLUE if direction[j] > 0 else ORANGE, lw=0.4, alpha=0.12)
        if w2 is None:
            for j in range(cfg.num_agents):
                xs = torch.cat([x0[j:j + 1], chosen[:, j, 0]])
                ax.plot(tau, xs, color=BLUE if direction[j] > 0 else ORANGE, lw=1.6)
        ax.set_title(title, color=INK, fontsize=8)
        if w2 is not None:
            ax.text(0.97, 0.03, f"$W_2(\\mu,\\nu^\\star)$ = {w2:.2f} m", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=6.5, color=INK_2)
        ax.set_xlabel("plan time (s)")
        ax.set_ylim(-9, 9)
    axes[0].set_ylabel("longitudinal position x (m)")
    axes[0].text(0.1, cfg.zone_half_length - 0.9, "single-file zone", color=INK_2, fontsize=7)
    handles = [plt.Line2D([], [], color=BLUE, lw=1.6, label="agents moving +x"),
               plt.Line2D([], [], color=ORANGE, lw=1.6, label="agents moving -x"),
               plt.Line2D([], [], color=INK_2, lw=0.6, alpha=0.5, label="rollout / barycenter atom"),
               plt.Line2D([], [], color=INK_2, lw=1.6, label="selected consensus plan")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.16))
    fig.savefig(out / "fig1_consensus_distribution.pdf")
    fig.savefig(out / "fig1_consensus_distribution.png")
    plt.close(fig)


def figure_bandwidth(out: Path, results: Path) -> None:
    """Figure 2: analytic payload rate (Proposition 4) with measured points from the benchmark."""
    pcfg, cfg = PredictionConfig(), CorridorConfig()
    rate = pcfg.replan_hz(cfg.dt)
    agents = list(range(2, 51))
    hw = [pcfg.num_rollouts * pcfg.horizon * 2 * m * BYTES_PER_VALUE * rate / 1000 for m in agents]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    series = [("latent streaming, diffusion (1024-D)", 1024, ORANGE), ("latent streaming, V-JEPA (512-D)", 512, AQUA),
              ("latent streaming, RSSM (128-D)", 128, YELLOW)]
    for label, dim, color in series:
        value = pcfg.num_rollouts * pcfg.horizon * dim * BYTES_PER_VALUE * rate / 1000
        ax.plot([agents[0], agents[-1]], [value, value], color=color, lw=2, label=label)
    ax.plot(agents, hw, color=BLUE, lw=2, label="HETERO-WORLD (joint trajectories)")
    measured = []
    for path in sorted(results.glob("*/hetero_world*.json")) + sorted(results.glob("*/feature_sharing*.json")):
        payload = json.loads(path.read_text())
        bw = payload["summary"]["bandwidth_kBps"]["mean"]
        if bw:
            measured.append((payload["settings"]["agents"], bw, payload["method"]))
    for m, bw, method in measured:
        ax.plot([m], [bw], "o", ms=5, color=BLUE if method == "hetero_world" else MAGENTA,
                markeredgecolor="white", markeredgewidth=1, zorder=5)
    ax.plot([], [], "o", ms=5, color=BLUE, markeredgecolor="white", label="measured: HETERO-WORLD")
    ax.plot([], [], "o", ms=5, color=MAGENTA, markeredgecolor="white", label="measured: feature sharing (fleet mix)")
    ax.set_yscale("log")
    ax.set_xlabel("number of agents M")
    ax.set_ylabel("payload per agent (KB/s)")
    ax.legend(fontsize=6, loc="lower right")
    fig.savefig(out / "fig2_bandwidth_vs_agents.pdf")
    fig.savefig(out / "fig2_bandwidth_vs_agents.png")
    plt.close(fig)


def figure_latency(out: Path, latency_path: Path) -> None:
    """Figure 3: measured latency scaling with M for barycenter versus pairwise and latent baselines."""
    data = json.loads(latency_path.read_text())
    scaling = data["scaling"]
    agents = [row["agents"] for row in scaling]
    series = [
        ("barycenter_joint", "barycenter, joint beliefs ($D=2HM$)", BLUE, "-"),
        ("pairwise_joint", "all-pairs OT, joint beliefs", ORANGE, "-"),
        ("barycenter_fixed_dim", "barycenter, fixed $D=2H$", BLUE, "--"),
        ("pairwise_fixed_dim", "all-pairs OT, fixed $D=2H$", ORANGE, "--"),
        ("union_select", "union selection (no OT)", AQUA, "-"),
        ("procrustes", "pairwise Procrustes", VIOLET, "-"),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for key, label, color, style in series:
        mean = [row[key]["mean"] for row in scaling]
        std = [row[key]["std"] for row in scaling]
        ax.plot(agents, mean, style, color=color, lw=2, label=label, marker="o", ms=3)
        ax.fill_between(agents, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                        color=color, alpha=0.12, lw=0)
    ax.axhline(3.0, color=INK_2, lw=0.8, ls=":")
    ax.text(agents[0], 3.3, "3 ms", color=INK_2, fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(agents)
    ax.set_xticklabels([str(a) for a in agents])
    ax.set_xlabel("number of agents M")
    ax.set_ylabel("per-agent latency (ms)")
    hw = data["hardware"]
    ax.set_title(f"{hw['machine']} CPU, {hw['threads']} threads, N=100, L=50, T=2", fontsize=7, color=INK_2)
    ax.legend(fontsize=5.5, loc="upper left")
    fig.savefig(out / "fig3_latency_scaling.pdf")
    fig.savefig(out / "fig3_latency_scaling.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--only", default="1,2,3")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    which = set(args.only.split(","))
    if "1" in which:
        figure_consensus(args.out)
    if "2" in which:
        figure_bandwidth(args.out, args.results)
    if "3" in which:
        figure_latency(args.out, args.results / "latency.json")
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
