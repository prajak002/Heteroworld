#!/usr/bin/env bash
# Train the three learned world models one after another (shares the GPU without contention).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/wm
.venv/bin/python scripts/train_world_model.py --family diffusion --minutes 60 > results/wm/train_diffusion.log 2>&1
.venv/bin/python scripts/train_world_model.py --family jepa --minutes 45 > results/wm/train_jepa.log 2>&1
.venv/bin/python scripts/train_world_model.py --family rssm --minutes 45 > results/wm/train_rssm.log 2>&1
echo "all done $(date)" > results/wm/DONE
