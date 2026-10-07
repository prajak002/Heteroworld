#!/usr/bin/env bash
# Ablations: entropic regularisation, packet loss and agent density.
# Usage: bash scripts/run_ablations.sh [eps|loss|density|all] [threads]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WHICH=${1:-all}
THREADS=${2:-4}
EPISODES=${EPISODES:-50}

if [[ "$WHICH" == "eps" || "$WHICH" == "all" ]]; then
  for eps in 0.001 0.003 0.01 0.03 0.1; do
    $PY scripts/run_benchmark.py --methods hetero_world --episodes "$EPISODES" --epsilon "$eps" \
      --threads "$THREADS" --tag "_eps${eps}" --out results/ablation_eps
  done
fi

if [[ "$WHICH" == "loss" || "$WHICH" == "all" ]]; then
  for p in 0.1 0.2 0.3 0.4; do
    $PY scripts/run_benchmark.py --methods hetero_world,union_select,feature_sharing --episodes "$EPISODES" \
      --packet-loss "$p" --threads "$THREADS" --tag "_loss${p}" --out results/ablation_loss
  done
fi

if [[ "$WHICH" == "density" || "$WHICH" == "all" ]]; then
  for m in 2 4 6 16 24; do
    $PY scripts/run_benchmark.py --methods hetero_world,union_select,noncoop_mpc --episodes "$EPISODES" \
      --agents "$m" --chunk 10 --threads "$THREADS" --tag "_M${m}" --out results/ablation_density
  done
fi
