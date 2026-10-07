#!/usr/bin/env bash
# Benchmark v2 pilot (20 episodes, 4 pedestrians), sharded over episodes, then world-model training.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/pilot6
mkdir -p $OUT results/wm
run() {  # method first_episode episodes
  .venv/bin/python scripts/run_benchmark.py --methods $1 --episodes $3 --first-episode $2 --chunk 5 \
    --humans 4 --threads 1 --tag _part$2 --out $OUT > $OUT/$1_part$2.log 2>&1
}
pids=()
for m in hw_robust mixture_robust; do for f in 0 5 10 15; do run $m $f 5 & pids+=($!); done; done
for m in union_select hetero_world oracle_robust independent; do for f in 0 10; do run $m $f 10 & pids+=($!); done; done
wait "${pids[@]}"
for m in hw_robust mixture_robust union_select hetero_world oracle_robust independent; do
  .venv/bin/python scripts/merge_results.py $OUT/${m}_part*.json --out $OUT/$m.json
done
echo "pilot6 done $(date)" > $OUT/DONE
.venv/bin/python scripts/train_world_model.py --family diffusion --minutes 60 > results/wm/train_diffusion.log 2>&1
.venv/bin/python scripts/train_world_model.py --family jepa --minutes 45 > results/wm/train_jepa.log 2>&1
.venv/bin/python scripts/train_world_model.py --family rssm --minutes 45 > results/wm/train_rssm.log 2>&1
echo "all done $(date)" > results/wm/DONE
