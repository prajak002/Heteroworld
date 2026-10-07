"""Resumable job queue for long benchmark campaigns on one machine.

Each line of the jobs file is ``<output.json> <command ...>``. A job is skipped when its output file
already exists, so the queue can be restarted at any time and only unfinished shards run again.
Failed jobs are logged to ``<jobs>.failed`` and the queue carries on. Outputs are checked again right before a job
starts, so shards finished elsewhere (another queue, another machine) are skipped. With ``--respect-running`` the
queue also skips shards that another process is already running (any live ``run_benchmark.py`` command) and counts
those processes against ``--workers``.

Usage:
    python scripts/job_queue.py jobs/v2_main.txt --workers 8
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from pathlib import Path


def external_runs(own: set[int]) -> set[Path]:
    """Output paths of run_benchmark.py processes not started by this queue (from their command lines)."""
    ps = subprocess.run(["ps", "-Ao", "pid=,args="], capture_output=True, text=True, check=False).stdout
    outs = set()
    for line in ps.splitlines():
        pid, _, args = line.strip().partition(" ")
        if "scripts/run_benchmark.py" not in args or not pid.isdigit() or int(pid) in own:
            continue
        a = shlex.split(args)
        opt = lambda k: a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None
        if opt("--out") and opt("--methods") and "," not in opt("--methods"):
            outs.add(Path(opt("--out")) / f"{opt('--methods')}{opt('--tag') or ''}.json")
    return outs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--respect-running", action="store_true",
                        help="skip shards other processes are running, and count them against --workers")
    args = parser.parse_args()
    pending = []
    for line in args.jobs.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            out, cmd = line.split(" ", 1)
            pending.append((Path(out), cmd))
    todo = [(o, c) for o, c in pending if not o.exists()]
    print(f"{len(pending)} jobs, {len(pending) - len(todo)} already done, {len(todo)} to run", flush=True)
    running: list[tuple[subprocess.Popen, Path, str, float]] = []
    failed = args.jobs.with_suffix(".failed")
    while todo or running:
        busy = external_runs({p.pid for p, *_ in running}) if args.respect_running else set()
        while todo and len(running) + len(busy) < args.workers:
            out, cmd = todo.pop(0)
            if out.exists():
                continue
            if out in busy:
                todo.append((out, cmd))      # still running elsewhere: look again later
                if all(o in busy or o.exists() for o, _ in todo):
                    break
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            log = open(out.with_suffix(".log"), "w")
            proc = subprocess.Popen(shlex.split(cmd), stdout=log, stderr=subprocess.STDOUT)
            running.append((proc, out, cmd, time.time()))
        time.sleep(5)
        for item in list(running):
            proc, out, cmd, start = item
            if proc.poll() is None:
                continue
            running.remove(item)
            ok = proc.returncode == 0 and out.exists()
            if not ok:
                with open(failed, "a") as f:
                    f.write(f"{out} {cmd}\n")
            left = len(todo) + len(running)
            print(f"{'ok  ' if ok else 'FAIL'} {out} ({time.time() - start:.0f}s), {left} left", flush=True)


if __name__ == "__main__":
    main()
