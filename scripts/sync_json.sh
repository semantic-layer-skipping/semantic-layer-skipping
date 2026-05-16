#!/bin/bash

# default paths to sync
DEFAULT_PATHS=(
    "/home/yff23/rds/rds-cl-acs-yff23-cjlENNKY3so/semantic-layer-skipping/experiments/batch_20260507_154513_Qwen2.5-1.5B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24"
    "/home/yff23/rds/rds-cl-acs-yff23-cjlENNKY3so/semantic-layer-skipping/experiments/batch_20260507_152045_Qwen2.5-1.5B-Instruct_e2e_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24"
    "/home/yff23/rds/hpc-work/semantic-layer-skipping/experiments/batch_20260309_042303_Qwen2.5-1.5B-Instruct_sharegpt_train_20000s_2048t_strict_strict_match_c4-8-12-16-20-24"
)

# function to handle the actual syncing logic for a single path
sync_path() {
    # 'local' ensures these variables don't bleed between loop iterations
    local HPC_PATH="$1"

    # extract the "batch_..." directory name dynamically
    if [[ "$HPC_PATH" =~ (batch_[^/]+) ]]; then
        local BATCH_DIR="${BASH_REMATCH[1]}"
    else
        echo "Error: The provided path must contain a 'batch_...' directory segment."
        echo "Path failed: $HPC_PATH"
        return 1
    fi

    # extract any sub-paths provided after the batch directory
    local SUB_PATH="${HPC_PATH#*${BATCH_DIR}}"
    SUB_PATH="${SUB_PATH%/}" # remove trailing slash if present

    # construct local target path
    local LOCAL_TARGET="hpc/experiments/${BATCH_DIR}${SUB_PATH}"

    # format remote path for rsync (must end with / to sync contents)
    local REMOTE_PATH="${HPC_PATH%/}/"

    echo "  Syncing from : wilkes:${REMOTE_PATH}"
    echo "  Syncing to   : ${LOCAL_TARGET}/"

    # create local directory tree if it doesn't exist
    mkdir -p "${LOCAL_TARGET}"

    # execute Rsync with updated exceptions - to avoid copying large files
    rsync -avzm \
        --include='**/calibration/db_ivfpq_*/' \
        --include='**/e2e_optimisation/db_ivfpq_*/' \
        --exclude='db_chunk_*/' \
        --exclude='db_merged_*/' \
        --exclude='db_ivfpq_*/' \
        --include='*/' \
        --include='*.json' \
        --exclude='*' \
        "wilkes:${REMOTE_PATH}" "${LOCAL_TARGET}/"

    echo " Sync complete for this path!"
    echo "--------------------------------------------------------"
}

# main script execution

if [ "$#" -eq 0 ]; then
    # not arguments provided: Loop through the defaults
    echo "No arguments provided. Syncing default batch paths..."
    for path in "${DEFAULT_PATHS[@]}"; do
        sync_path "$path"
    done
    echo " All default paths synced successfully!"

elif [ "$#" -eq 1 ]; then
    # 1 argument provided: Sync just that specific path
    sync_path "$1"

else
    # >1 arguments provided: Show help message
    echo "Usage: $0 [optional_hpc_source_path]"
    echo "If no path is provided, it will sync the default hardcoded paths."
    exit 1
fi
