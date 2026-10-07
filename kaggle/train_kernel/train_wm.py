"""Kaggle GPU job: train the three learned world models (diffusion, JEPA, RSSM) for run seeds 0, 1, 2.

Each (family, seed) trains for a fixed number of steps so every seed gets the same budget regardless of
machine load. Checkpoints and logs land in /kaggle/working/checkpoints (seed 0 at the top level,
seed k in seed<k>/), which is the layout run_benchmark.py --learned expects.
"""

import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time

import torch as _torch
if not _torch.cuda.is_available():
    raise SystemExit("No GPU in this session (torch.cuda.is_available() is False); refusing to run on CPU.")
print("GPU:", _torch.cuda.get_device_name(0), "| torch", _torch.__version__, flush=True)

SRC = "/kaggle/input"  # search every mounted input (mount paths vary)
WORK = "/tmp/hw"
OUT = "/kaggle/working"
STEPS = {"diffusion": 60000, "jepa": 40000, "rssm": 30000}
SEEDS = [0, 1, 2]

os.makedirs(WORK, exist_ok=True)
archive = [p for p in glob.glob(f"{SRC}/**/code.tar.gz", recursive=True) if os.path.isfile(p)]
if archive:
    with tarfile.open(archive[0]) as tar:
        tar.extractall(WORK)
else:  # Kaggle may have extracted the archive into a folder
    for name in ("hetero_world", "scripts", "tests"):
        hit = sorted(glob.glob(f"{SRC}/**/{name}", recursive=True), key=len)[0]
        shutil.copytree(hit, f"{WORK}/{name}")
data = glob.glob(f"{SRC}/**/wm_train.pt", recursive=True)[0]
os.makedirs(f"{WORK}/data", exist_ok=True)
os.symlink(data, f"{WORK}/data/wm_train.pt")
os.chdir(WORK)

import torch  # noqa: E402
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

os.makedirs(f"{OUT}/logs", exist_ok=True)
for seed in SEEDS:
    for family, steps in STEPS.items():
        t0 = time.time()
        log = f"{OUT}/logs/train_{family}_seed{seed}.log"
        cmd = [sys.executable, "scripts/train_world_model.py", "--family", family, "--seed", str(seed),
               "--max-steps", str(steps), "--minutes", "600"]
        with open(log, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        tail = open(log).read().strip().splitlines()[-1:]
        print(f"{family} seed {seed}: exit {proc.returncode} in {time.time() - t0:.0f}s {tail}", flush=True)
        shutil.copytree(f"{WORK}/checkpoints", f"{OUT}/checkpoints", dirs_exist_ok=True)
print("all done", flush=True)
