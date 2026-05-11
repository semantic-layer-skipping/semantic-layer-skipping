#!/bin/bash

# validate input
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <hpc_source_path>"
    echo "Example: $0 /home/yff23/rds/hpc-work/semantic-layer-skipping/experiments/batch_2026..."
    exit 1
fi

HPC_PATH="$1"

# extract the "batch_..." prefix dynamically
if [[ "$HPC_PATH" =~ (batch_[^/]+) ]]; then
    BATCH_DIR="${BASH_REMATCH[1]}"
else
    echo "Error: The provided path must contain a 'batch_...' directory segment."
    exit 1
fi

# extract any sub-paths provided after the batch directory
SUB_PATH="${HPC_PATH#*${BATCH_DIR}}"
SUB_PATH="${SUB_PATH%/}" # remove trailing slash if present

# construct local target path
LOCAL_TARGET="hpc/experiments/${BATCH_DIR}${SUB_PATH}"

# format remote path for rsync (must end with / to sync contents)
REMOTE_PATH="${HPC_PATH%/}/"

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

echo "  Sync complete!"
