#!/bin/bash

# ensure we are in the project root
CURRENT_DIR=$(basename "$PWD")

if [ "$CURRENT_DIR" != "semantic-layer-skipping" ]; then
    echo "Error: Script must be run from the 'semantic-layer-skipping' directory."
    echo "Current directory is: $PWD"
    exit 1
fi

echo "Verified project root. Proceeding..."

# create the data directory
mkdir -p data

cd data || { echo "Failed to enter data directory"; exit 1; }

echo "Downloading datasets from GPTCache repository..."

# download the datasets
curl -O https://raw.githubusercontent.com/zilliztech/GPTCache/main/examples/benchmark/similiar_qqp.json.gz
curl -O https://raw.githubusercontent.com/zilliztech/GPTCache/main/examples/benchmark/similiar_qqp_full.json.gz

echo "Extracting datasets (tarball)..."

# -x: extract, -z: handle gzip, -f: file
tar -xzf similiar_qqp.json.gz
tar -xzf similiar_qqp_full.json.gz

# output dataset size
DIR_SIZE=$(du -sh . | awk '{print $1}')

echo "Total size of the 'data' directory: $DIR_SIZE"
