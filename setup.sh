#!/usr/bin/env bash
# BC vs IFQL on visual-block-double-task1-v0, end to end:
#   1. download the dataset (all 300 train shards if they fit, keeping RESERVE_GB free for checkpoints)
#      + the 100 validation shards that hold the 1000 held-out episodes
#   2. train MSE BC and IFQL (shard rotation: a group of shards in RAM at a time)
#   3. evaluate every 10k checkpoint from the 1000 held-out starting states (BC, IFQL best-of-32, flow BC)
#   4. action error on the held-out demos, demonstrator stats, and the same figures as for the single cube
#
# Run inside tmux, with the GPU free:
#   tmux new -s double
#   conda activate temporal
#   cd ~/il-experiment && bash setup.sh 2>&1 | tee setup_double.log
#
# Re-running is safe: finished downloads, trainings and evaluations are skipped; an interrupted BC run resumes from
# its last.pt (an interrupted IFQL run restarts). Rough time: BC ~9 h, IFQL ~17 h, held-out evals ~6 h.
# Knobs (environment variables): STEPS (default 70000), RESERVE_GB (default 30).
set -euo pipefail

ENV=visual-block-double-task1-v0
STEPS=${STEPS:-70000}
RESERVE_GB=${RESERVE_GB:-30}
NUM_VAL_SHARDS=100            # 10 episodes each -> 1000 held-out episodes
BC_RUN=bc_double              # -> ocbench/impls/checkpoints/bc_double
IFQL_RUN=ifql_double          # -> ocbench/impls/checkpoints/ifql_double
REPO=$(cd "$(dirname "$0")" && pwd)
cd "$REPO/ocbench/impls"
ANALYSIS=checkpoints/analysis_double
PLOTS="$REPO/plots/heldout_double"
mkdir -p "$ANALYSIS/heldout" "$ANALYSIS/grasp" "$PLOTS"

stage() { echo; echo "=== [$(date '+%F %T')] $* ==="; }
free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

stage "checks"
python -c "import torch, ocbench; assert torch.cuda.is_available()" || { echo "activate the 'temporal' conda env first"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 2000 ]; then echo "GPU already has ${used} MiB in use - stop other jobs first"; exit 1; fi
echo "free disk: $(free_gb) GB, free RAM: $(free -g | awk '/Mem:/ {print $7}') GB"

stage "1/6 download (keeping ${RESERVE_GB} GB free for checkpoints)"
python agents/download_dataset.py --env_name $ENV --num_val $NUM_VAL_SHARDS --reserve_gb "$RESERVE_GB"
if [ "$(free_gb)" -lt 20 ]; then echo "less than 20 GB free - not enough for checkpoints"; exit 1; fi

stage "2/6 train MSE BC ($STEPS steps)"
if [ -f "checkpoints/$BC_RUN/model_step_${STEPS}.pt" ]; then
    echo "already trained, skipping"
else
    resume=None
    [ -f "checkpoints/$BC_RUN/last.pt" ] && resume="checkpoints/$BC_RUN/last.pt" && echo "resuming from $resume"
    # 25 shards (~65 GB of images, successful episodes) in RAM at a time, next group every 5k steps
    python agents/bc_torch.py --set env_name=$ENV run_name=$BC_RUN resume="$resume" num_steps="$STEPS" \
        num_shards=None shard_group_size=25 rotate_every=5000 num_val_shards=10 \
        eval_interval=10000 eval_episodes=100 save_interval=10000
fi

stage "3/6 train IFQL ($STEPS steps)"
if [ -f "checkpoints/$IFQL_RUN/model_step_${STEPS}.pt" ]; then
    echo "already trained, skipping"
else
    rm -rf "checkpoints/$IFQL_RUN"  # IFQL can't resume: start clean
    # 20 shards (~66 GB, all episodes) in RAM at a time, next group every 4.5k steps
    python agents/ifql_torch.py --set env_name=$ENV run_name=$IFQL_RUN num_steps="$STEPS" \
        num_shards=None shard_group_size=20 rotate_every=4500 num_val_shards=10 \
        eval_interval=10000 eval_episodes=100 save_interval=10000
fi

stage "4/6 demonstrator stats on the held-out episodes"
[ -f "$ANALYSIS/grasp/dataset_episodes.npz" ] || python agents/dataset_grasp_stats.py --env_name $ENV --splits val \
    --num_val_shards $NUM_VAL_SHARDS --out "$ANALYSIS/grasp/dataset_episodes.npz"

stage "5/6 held-out evaluation of every 10k checkpoint (1000 starts each)"
runs=()
for ((s = 10000; s <= STEPS; s += 10000)); do
    k=$((s / 1000))
    runs+=(--run "bc_${k}k:checkpoints/$BC_RUN/model_step_${s}.pt")
    runs+=(--run "ifql_${k}k:checkpoints/$IFQL_RUN/model_step_${s}.pt:32")
    runs+=(--run "flowbc_${k}k:checkpoints/$IFQL_RUN/model_step_${s}.pt:1")
done
python agents/eval_heldout.py --env_name $ENV --num_val_shards $NUM_VAL_SHARDS --out_dir "$ANALYSIS/heldout" "${runs[@]}"

stage "6/6 action error + figures"
steps=$(seq 10000 10000 "$STEPS" | tr '\n' ' ')
[ -f "$ANALYSIS/heldout/action_error.csv" ] || python agents/eval_action_error.py --env_name $ENV \
    --num_val_shards $NUM_VAL_SHARDS --bc_run "checkpoints/$BC_RUN" --ifql_run "checkpoints/$IFQL_RUN" \
    --steps $steps --out "$ANALYSIS/heldout/action_error.csv"
python agents/plot_heldout.py --dir "$ANALYSIS/heldout" --dataset_stats "$ANALYSIS/grasp/dataset_episodes.npz" \
    --out "$PLOTS" --main auto --env_label $ENV | tee "$ANALYSIS/heldout/summary.txt"

stage "done"
echo "figures: $PLOTS"
echo "numbers: $(pwd)/$ANALYSIS/heldout/summary.txt"
echo "free disk: $(free_gb) GB"
