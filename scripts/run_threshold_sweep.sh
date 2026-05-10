#!/bin/bash

THRESHOLDS=(0.87 0.89 0.91 0.93 0.95 0.97)
#THRESHOLDS=(0.9995 0.9999 0.99995 0.99999 0.999995 0.999999)

for THRESHOLD in "${THRESHOLDS[@]}"; do

  echo "Starting detached tmux session: $SESSION_NAME"

  #CMD="python -m main --target_prefix batch_20260507_152045 --use_ivfpq --subsample_fraction 1.0 --loc rds-cl --train_dataset e2e --train_samples 40000 --train_max_tokens 128 --train_batch_size 2048 --run_evaluation --eval_samples 100 --eval_max_tokens 128 --eval_dataset e2e --kv_strategy full_compute --decision_strategy top1_strict --manual_thresholds $THRESHOLD"
  CMD="python -m main --target_prefix batch_20260507_154513 --use_ivfpq --subsample_fraction 1.0 --loc rds-cl --train_dataset wmt19 --train_samples 40000 --train_max_tokens 128 --train_batch_size 2048 --run_evaluation --eval_samples 100 --eval_max_tokens 128 --eval_dataset wmt19 --kv_strategy full_compute --decision_strategy top1_strict --manual_thresholds $THRESHOLD"

  TEMP_STR="${CMD#*--target_prefix }"
  PREFIX="${TEMP_STR%% *}"
  # replace the decimal point with an underscore for a clean session name (e.g., "run_0_86")
  SESSION_NAME="eval_${THRESHOLD/./_}_${PREFIX}"

  # launch the command in a new detached tmux session
  # the '; bash' at the end keeps the window open after the Python script finishes
  tmux new-session -d -s "$SESSION_NAME" "$CMD; bash"
done

echo "All jobs launched! Use 'tmux ls' to view your running sessions."
