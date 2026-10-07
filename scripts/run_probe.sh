#!/usr/bin/env bash
# Stress probe on benchmark v2: where do the pedestrian references separate?
#   A: 6 pedestrians, all assertive.   B: 4 pedestrians, 2 robots with hallucinating world models.
set -uo pipefail
cd "$(dirname "$0")/.."
for m in union_select mixture_robust hw_robust cv_robust oracle_robust; do
  mkdir -p results/probeA/seed0 results/probeB/seed0
  .venv/bin/python scripts/run_benchmark.py --methods $m --episodes 20 --chunk 5 --humans 6 --assertive-prob 1.0 \
    --threads 1 --out results/probeA/seed0 > results/probeA/seed0/$m.log 2>&1 &
  .venv/bin/python scripts/run_benchmark.py --methods $m --episodes 20 --chunk 5 --humans 4 --faulty-agents 2 \
    --threads 1 --out results/probeB/seed0 > results/probeB/seed0/$m.log 2>&1 &
done
wait
echo "probe done $(date)" > results/probeA/DONE
