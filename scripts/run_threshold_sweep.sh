#!/bin/bash

THRESHOLDS=(0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89)
#THRESHOLDS=(0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 0.995)

# e2e
#THRESHOLDS=(0.9995 0.9999 0.99995 0.99999 0.999995 0.999999)
#THRESHOLDS=(1.0 1.0005 1.001 1.002 1.0025 1.005)  #1.0075 1.01 1.02 1.025)

# random skipping
#THRESHOLDS=(0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.12 0.14 0.15)
#THRESHOLDS=(0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15)

# run source .venv/bin/activate
source .venv/bin/activate

for THRESHOLD in "${THRESHOLDS[@]}"; do

  # manual thresholds - kv_strategy: full_compute, copy, project_only. decision_strategy: top1_strict, safe_knn, consensus_decay, semantic_boundary, softmax_expected_skip
  # wmt19 4 checkpoint
  #CMD="python -m main --target_prefix batch_20260507_154513 --use_ivfpq --subsample_fraction 1.0 --loc rds-cl --train_dataset wmt19 --train_samples 40000 --train_max_tokens 128 --train_batch_size 2048 --run_evaluation --eval_samples 100 --eval_max_tokens 128 --eval_dataset wmt19 --kv_strategy project_only --decision_strategy top1_strict --manual_thresholds $THRESHOLD"

  # wmt19 1 checkpoint
  CMD="python -m main --target_prefix batch_20260514_035404 --use_ivfpq --subsample_fraction 1.0 --loc rds-cl --train_dataset wmt19 --checkpoint_start 1  --checkpoint_end 28 --checkpoint_step 1 --train_samples 40000 --train_max_tokens 128 --train_batch_size 1024 --run_evaluation --eval_samples 100 --eval_max_tokens 128 --eval_dataset wmt19 --kv_strategy full_compute --decision_strategy top1_strict --manual_thresholds $THRESHOLD"


  #CMD="python -m main --target_prefix batch_20260507_152045 --use_ivfpq --subsample_fraction 1.0 --loc rds-cl --train_dataset e2e --train_samples 40000 --train_max_tokens 128 --train_batch_size 2048 --run_evaluation --eval_samples 100 --eval_max_tokens 128 --eval_dataset e2e --kv_strategy full_compute --decision_strategy top1_strict --manual_thresholds $THRESHOLD"
  #CMD="python -m main --target_prefix batch_20260309_042303 --use_ivfpq --subsample_fraction 1.0 --loc hpc-work --run_evaluation --eval_samples 100 --eval_max_tokens 128 --eval_dataset sharegpt --kv_strategy full_compute --decision_strategy top1_strict --manual_thresholds $THRESHOLD"

  # random skipping baseline
  #CMD="python -m main --target_prefix batch_20260507_154513 --loc rds-cl --train_dataset wmt19 --train_samples 40000 --train_max_tokens 128 --run_evaluation --eval_dataset wmt19 --eval_samples 100 --eval_max_tokens 128 --kv_strategy full_compute --eval_random_skip_probs $THRESHOLD"
  #CMD="python -m main --target_prefix batch_20260507_152045 --loc rds-cl --train_dataset e2e --train_samples 40000 --train_max_tokens 128 --run_evaluation --eval_dataset e2e --eval_samples 100 --eval_max_tokens 128 --kv_strategy full_compute --eval_random_skip_probs $THRESHOLD"
  #CMD="python -m main --target_prefix batch_20260309_042303 --loc hpc-work --run_evaluation --eval_samples 100 --eval_max_tokens 128 --kv_strategy full_compute --eval_random_skip_probs $THRESHOLD"

  TEMP_STR="${CMD#*--target_prefix }"
  PREFIX="${TEMP_STR%% *}"
  # replace the decimal point with an underscore for a clean session name (e.g., "run_0_86")
  SESSION_NAME="eval_${THRESHOLD/./_}_${PREFIX}"

  echo "Starting detached tmux session: $SESSION_NAME"

  # launch the command in a new detached tmux session
  # the '; bash' at the end keeps the window open after the Python script finishes
  tmux new-session -d -s "$SESSION_NAME" "$CMD; bash"
done

echo "All jobs launched! Use 'tmux ls' to view your running sessions."
echo "To attach to a session, use: tmux attach -t <session_name>"
