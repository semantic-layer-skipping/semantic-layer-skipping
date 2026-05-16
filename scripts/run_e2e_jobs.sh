#!/bin/bash

# this script sends e2e optimisation jobs to tmux, each in its own window, with 5 second delay in between

# define the session name and the command to run
SESSION_NAME="e2e_batch"
COMMAND="python -m main --target_prefix batch_20260309_042303 --use_ivfpq --subsample_fraction 1.0 --loc hpc-work --run_e2e_optimisation --cal_dataset sharegpt --cal_max_tokens 128 --cal_samples 250 --e2e_trials 1"

echo "Creating tmux session: $SESSION_NAME"

# create a new detached tmux session and name the first window "run_1"
tmux new-session -d -s "$SESSION_NAME" -n "run_1"

# send the command to the first window and execute it (C-m simulates the Enter key)
echo "Launching run 1..."
tmux send-keys -t "$SESSION_NAME:run_1" "$COMMAND" C-m

NUM_RUNS=15
# loop to create windows 2 through NUM_RUNS
for (( i=2; i<=NUM_RUNS; i++ )); do
    echo "Waiting 5 seconds..."
    sleep 5

    echo "Launching run $i..."
    tmux new-window -t "$SESSION_NAME" -n "run_$i"

    tmux send-keys -t "$SESSION_NAME:run_$i" "$COMMAND" C-m
done

echo ""
echo "All $NUM_RUNS jobs launched successfully!"
echo "To view your jobs, attach to the session by running: tmux attach -t $SESSION_NAME"
echo "Once attached, use 'Ctrl+b' then 'n' to cycle through the windows."
