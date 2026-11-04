#!/usr/bin/env bash
# Held-out evaluation of BC, IFQL and flow BC from the exact first states of the 1000 validation episodes,
# then all figures in <repo>/plots/heldout/.
#
#   cd ~/il-experiment/ocbench/impls
#   conda activate temporal
#   bash run_heldout_analysis.sh 2>&1 | tee checkpoints/analysis/heldout_run.log
#
# Safe to re-run: runs whose results already exist in checkpoints/analysis/heldout/ are skipped.
# Takes ~7-8 min per run, 21 runs (~2.5 h). Needs the GPU to itself.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p checkpoints/analysis/heldout

BC=checkpoints/bc_torch_1793611331
IFQL=checkpoints/ifql_torch_1793629332

# 1) The three main models, with full per-step traces (for grasp analysis) and Q.
python agents/eval_heldout.py --traces \
    --run bc_30k:$BC/model_step_30000.pt \
    --run ifql_50k:$IFQL/model_step_50000.pt:32 \
    --run flowbc_50k:$IFQL/model_step_50000.pt:1

# 2) Every 10k checkpoint of all three (success + grasp stats, no full traces).
runs=()
for s in 10 20 30 40 50 60 70; do
    runs+=(--run "bc_${s}k:$BC/model_step_${s}000.pt")
    runs+=(--run "ifql_${s}k:$IFQL/model_step_${s}000.pt:32")
    runs+=(--run "flowbc_${s}k:$IFQL/model_step_${s}000.pt:1")
done
python agents/eval_heldout.py "${runs[@]}"

# 3) Demonstrator statistics on the same held-out episodes (only computed once).
if [ ! -f checkpoints/analysis/grasp/dataset_episodes.npz ]; then
    mkdir -p checkpoints/analysis/grasp
    python agents/dataset_grasp_stats.py
fi

# 4) Figures -> ../../plots/heldout/ and all numbers printed for the write-up.
python agents/plot_heldout.py | tee checkpoints/analysis/heldout/summary.txt

echo "Done. Figures in $(cd ../.. && pwd)/plots/heldout/, numbers in checkpoints/analysis/heldout/summary.txt"
