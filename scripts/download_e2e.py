import os

import pandas as pd
from datasets import Dataset

DATA_CACHE = os.path.expanduser("~/rds/hpc-work/data/semantic-layer-skipping")

print(f"Ensure raw CSVs are downloaded to: {DATA_CACHE}")
print("""
cd ~/rds/hpc-work/data/semantic-layer-skipping
wget https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/trainset.csv
wget https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/devset.csv
wget https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/testset_w_refs.csv
""")

local_files = {
    "train": os.path.join(DATA_CACHE, "trainset.csv"),
    "validation": os.path.join(DATA_CACHE, "devset.csv"),
    "test": os.path.join(DATA_CACHE, "testset_w_refs.csv"),
}

for split_name, file_path in local_files.items():
    cache_dir = os.path.join(DATA_CACHE, f".hf_cache_e2e/{split_name}")

    print(f"Reading local file: {file_path}")
    df = pd.read_csv(file_path)
    ds = Dataset.from_pandas(df)

    print(f"Saving to Hugging Face cache: {cache_dir}")
    ds.save_to_disk(cache_dir)
