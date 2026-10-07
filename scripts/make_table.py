"""Generate booktabs LaTeX tables from benchmark and latency results.

Usage:
    python scripts/make_table.py --main results/main --latency results/latency.json --out paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAIN_ROWS = [
    ("noncoop_mpc", "Non-cooperative MPC"),
    ("procrustes", "Pairwise Procrustes alignment"),
    ("mappo", "MAPPO (CTDE)"),
    ("feature_sharing", "Feature-sharing V2X"),
    ("hetero_world", r"\textsc{Hetero-World} (ours)"),
]
ABLATION_ROWS = [
    ("independent", "Independent world models (no comm.)"),
    ("union_select", "Union of rollouts, no barycenter"),
    ("hetero_world", r"\textsc{Hetero-World} (barycenter)"),
]
COMMUNICATING = {"procrustes", "feature_sharing", "hetero_world", "union_select"}
HAS_CONSENSUS = {"procrustes", "feature_sharing", "hetero_world", "union_select", "independent"}
HAS_BELIEFS = {"procrustes", "feature_sharing", "hetero_world", "union_select", "independent"}

# (column header, summary key, decimals, applies-to set or None for all, lower_is_better)
COLUMNS = [
    (r"Deadlock (\%) $\downarrow$", "deadlock_rate_pct", 1, None, True),
    (r"Collision (\%) $\downarrow$", "collision_rate_pct", 1, None, True),
    (r"Completed (\%) $\uparrow$", "completed_rate_pct", 1, None, False),
    (r"Makespan (s) $\downarrow$", "makespan_s", 1, None, True),
    (r"V2X (KB/s/agent) $\downarrow$", "bandwidth_kBps", 1, COMMUNICATING, True),
    (r"Latency (ms) $\downarrow$", "latency", 1, HAS_CONSENSUS, True),
    (r"$\overline{W}_2$ (m) $\downarrow$", "belief_w2_m", 2, HAS_BELIEFS, True),
]


def load(main_dir: Path, latency_path: Path | None) -> dict[str, dict]:
    """Load summaries per method and attach per-agent latency."""
    data = {}
    for path in sorted(main_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        data[payload["method"]] = dict(payload["summary"])
    if latency_path is not None and latency_path.exists():
        lat = json.loads(latency_path.read_text())["per_agent_latency_ms_M10"]
        for method, values in lat.items():
            if method in data:
                data[method]["latency"] = values
    return data


def cell_value(summary: dict, key: str) -> tuple[float, float, int] | None:
    """(mean, std, n) for a metric, or None if undefined."""
    entry = summary.get(key)
    if entry is None or entry.get("mean") is None:
        return None
    return entry["mean"], entry["std"], entry.get("n", 0)


def format_table(rows: list[tuple[str, str]], data: dict[str, dict], caption: str, label: str) -> str:
    """Render one booktabs table*, bolding the best and underlining the second-best entry per column."""
    present = [(m, name) for m, name in rows if m in data]
    ranks: dict[tuple[str, str], str] = {}
    for _, key, dec, applies, lower in COLUMNS:
        vals = []
        for method, _ in present:
            if applies is not None and method not in applies:
                continue
            v = cell_value(data[method], key)
            if v is not None:
                vals.append((round(v[0], dec), method))
        distinct = sorted({v for v, _ in vals}, reverse=not lower)
        for value, method in vals:
            if distinct and value == distinct[0]:
                ranks[(method, key)] = "best"
            elif len(distinct) > 1 and value == distinct[1]:
                ranks[(method, key)] = "second"
    spec = "l" + "c" * len(COLUMNS)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             "Method & " + " & ".join(c[0] for c in COLUMNS) + r" \\", r"\midrule"]
    for method, name in present:
        cells = []
        for _, key, dec, applies, _ in COLUMNS:
            v = cell_value(data[method], key)
            if (applies is not None and method not in applies) or v is None:
                cells.append("--")
                continue
            mean, std, _ = v
            text = f"{mean:.{dec}f} $\\pm$ {std:.{dec}f}"
            rank = ranks.get((method, key))
            if rank == "best":
                text = rf"\textbf{{{mean:.{dec}f}}} $\pm$ {std:.{dec}f}"
            elif rank == "second":
                text = rf"\underline{{{mean:.{dec}f}}} $\pm$ {std:.{dec}f}"
            cells.append(text)
        if method == "hetero_world":
            lines.append(r"\midrule")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=Path("results/main"))
    parser.add_argument("--latency", type=Path, default=Path("results/latency.json"))
    parser.add_argument("--out", type=Path, default=Path("paper"))
    args = parser.parse_args()
    data = load(args.main, args.latency)
    n = next(iter(data.values()))["deadlock_rate_pct"]["n"] if data else 0
    common = (f"Rates are over {n} Monte Carlo episodes ($\\pm$ binomial standard error); continuous metrics are "
              "mean $\\pm$ standard deviation over episodes (makespan over completed episodes only). "
              "Latency is the per-agent fusion-and-selection time measured on an Apple M1 CPU. "
              "V2X and latency are ranked only among methods that communicate or fuse. "
              "Best in \\textbf{bold}, second \\underline{underlined}.")
    args.out.mkdir(parents=True, exist_ok=True)
    main_tex = format_table(MAIN_ROWS, data, "Two-lane bottleneck corridor, $M=10$ heterogeneous agents. " + common,
                            "tab:main")
    abl_tex = format_table(ABLATION_ROWS, data, "Ablation of the communication and fusion components ($M=10$). "
                           + common, "tab:ablation")
    (args.out / "table_main.tex").write_text(main_tex)
    (args.out / "table_ablation.tex").write_text(abl_tex)
    print(main_tex)
    print(abl_tex)


if __name__ == "__main__":
    main()
