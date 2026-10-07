"""Kaggle GPU job: ETH/UCY leave-one-scene-out forecast study (3 seeds x 5 scenes x 3 model families)."""

import glob
import os
import shutil
import subprocess
import sys
import tarfile

import torch as _torch
if not _torch.cuda.is_available():
    raise SystemExit("No GPU in this session (torch.cuda.is_available() is False); refusing to run on CPU.")
print("GPU:", _torch.cuda.get_device_name(0), "| torch", _torch.__version__, flush=True)

SRC = "/kaggle/input"  # search every mounted input (mount paths vary)
WORK = "/tmp/hw"
OUT = "/kaggle/working"

os.makedirs(WORK, exist_ok=True)
for root, dirs, files in os.walk(SRC):
    depth = root[len(SRC):].count(os.sep)
    if depth <= 3:
        print(root, sorted(files)[:8], flush=True)
archive = [p for p in glob.glob(f"{SRC}/**/code.tar.gz", recursive=True) if os.path.isfile(p)]
if archive:
    with tarfile.open(archive[0]) as tar:
        tar.extractall(WORK)
else:  # Kaggle extracts archives into folders
    for name in ("hetero_world", "scripts"):
        hit = sorted(glob.glob(f"{SRC}/**/{name}", recursive=True), key=len)[0]
        shutil.copytree(hit, f"{WORK}/{name}")
os.makedirs(f"{WORK}/data/ethucy", exist_ok=True)
for f in glob.glob(f"{SRC}/**/*.pt", recursive=True):
    shutil.copy(f, f"{WORK}/data/ethucy/")
print(sorted(os.listdir(f"{WORK}/data/ethucy")), flush=True)
os.chdir(WORK)
cmd = [sys.executable, "scripts/ethucy_experiment.py", "--seeds", "0", "1", "2", "--steps", "8000",
       "--device", "cuda", "--out", f"{OUT}/ethucy_results.json"]
with open(f"{OUT}/ethucy.log", "w") as log:
    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
print(open(f"{OUT}/ethucy.log").read()[-6000:], flush=True)
print("exit", proc.returncode)
