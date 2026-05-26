# semantic-layer-skipping

This project is developed as part of a dissertation for Part III Computer Science at the University of Cambridge.
It presents a training-free retrieval-guided framework for adaptive layer skipping in Transformer models using dense semantic memory.
The framework operates by pre-populating vector indexes with hidden states mapped to optimal skipping trajectories which are calibrated to user-defined targets,
and used during online inference to dynamically bypass redundant computation.
We additionally extend this framework to an online serving environment by modifying the Mini-SGLang engine with a virtual pipelining architecture.

## Prerequisites

To set up the project, follow these steps:

1. Ensure `uv` is installed.
2. Synchronise the dependencies:
    ```bash
    uv sync
    ```
3. (Optional) Enable pre-commit git hooks if working on development:
    ```bash
    uv run pre-commit install
    ```

## Repository Overview
* **`src/semantic_layer_skipping/`**: Core framework source code.
  * `calibration/`: Offline threshold simulation and multi-objective evolutionary search (Optuna).
  * `data/`: Dataset loading and processing.
  * `eval/`: Automated plotting and metric aggregation.
  * `experiment/`: Experiment handlers, configurations, and lifecycle managers.
  * `inference/`: PyTorch runner wrappers, custom KV cache strategies, and generation loops.
  * `initial_analysis/`: Preliminary profiling scripts for layer-skipping behaviour.
  * `store.py`: Faiss index management (exact and IVFPQ indexes).
  * `main.py`: The primary entry point for orchestrating all pipeline stages.
* **`scripts/`**: SLURM submission scripts for HPC execution, dataset download scripts, and parallel job execution scripts.
* **`experiments/`, `plots/`, `slurm-results/`**: Output directories for generated data, visualisations, and HPC logs.
* **`semantic-layer-skipping/mini-sglang/`**: The modified online serving engine containing our virtual pipelining architecture.


## Running Experiments

Activate the virtual environment before running any scripts:
```bash
source .venv/bin/activate
```


### Main Pipeline (`main.py`)

The `main.py` script manages the entire lifecycle of the retrieval-guided framework. Because the pipeline involves large vector indexes and heavy simulation, execution is split into discrete sequential stages.

Experiment data and indexes are organised into specific folders governed by the `--target_prefix` argument. If this is not provided during the initial population stage, a timestamped folder name (e.g., `batch_20260526_120000`) is automatically generated. You should provide this exact `--target_prefix` in subsequent pipeline calls to access the correct experiment folder.

The execution stages include:
1. **Discovery (Optional):** Profiles hidden state distributions (mean, variance) for advanced state injection normalisation.
2. **Population:** Passes the training dataset through the model, extracting and evaluating optimal skipping trajectories. To prevent data loss, it saves the vectors to disk in safety chunks. Execution is resumable across a new run.
3. **Merging:** Combines the saved chunks into a single per-checkpoint indexes, applying optional fractional subsampling. **Note:** This is a memory-intensive operation as it loads all uncompressed vectors into RAM simultaneously.
4. **IVFPQ Conversion:** Converts the exact merged indexes into per-checkpoint IVFPQ indexes for accelerated retrieval. This is a relatively memory-efficient operation, as it constructs and trains the indexes sequentially, one checkpoint at a time.
5. **Calibration & E2E Optimisation:** Simulates routing decisions offline to establish optimal similarity thresholds based on bounded accuracy targets, or runs evolutionary search to find the Pareto frontier.
6. **Evaluation:** Performs end-to-end testing on hold-out datasets to compute quality (BERTScore, ROUGE-L, BLEU) and efficiency (theoretical acceleration ratio).
7. **Baselines:** Runs unified layer skipping or probability-based skipping (without requiring any vector indexes).


For full argument information, run
```bash
python -m main --help
```
The following section gives an example workflow for using the script.
Further examples can be found in `scripts/slurm_submit_population.wilkes`, `scripts/cpu_submit.sh`, `scripts/run_threshold_sweep.sh` and `run_e2e_jobs.sh`.

### Example Workflow

Here is a sample end-to-end workflow chaining these stages back-to-back using the WMT19 dataset.
Note how the foundational arguments (`--train_dataset`, `--train_samples`, `--train_max_tokens`) that specify an experiment (in conjunction with the target prefix) are consistently passed to ensure the experiment folder is accessed.
We can also specify the model and checkpoint structure to use with `--model_name`, `--checkpoint_start`, `--checkpoint_end`, `--checkpoint_step` arguments.
The offline population strategies are controlled by: `--early_exit_strategy`, `--skip_strategy`, `--kl_threshold` and `--injection_strategy`.

1. Populate the database in chunks:
   ```bash
   python -m main --target_prefix batch_demo --loc repo \
       --train_dataset wmt19 --train_samples 40000 --train_max_tokens 128 \
       --train_batch_size 2048 \
       --run_population
   ```

2. Merge the database chunks:
   ```bash
   python -m main --target_prefix batch_demo --loc repo \
       --train_dataset wmt19 --train_samples 40000 --train_max_tokens 128 \
       --subsample_fraction 1.0 \
       --run_merge
   ```

3. Build the IVFPQ index:
   ```bash
   python -m main --target_prefix batch_demo --loc repo \
       --train_dataset wmt19 --train_samples 40000 --train_max_tokens 128  \
       --subsample_fraction 1.0 \
       --run_ivfpq_conversion
   ```

4. (Optional) End-to-end optimisation/calibration:
   ```bash
   python -m main --target_prefix batch_demo --loc repo \
       --train_dataset wmt19 --train_samples 40000 --train_max_tokens 128 \
       --use_ivfpq --subsample_fraction 1.0 \
       --run_e2e_optimisation \
       --cal_dataset wmt19 --cal_samples 500 --cal_max_tokens 128 --e2e_trials 50
   ```
   This will save a group of sets of thresholds forming a pareto frontier, as well as all individual trial runs and their evaluation properties.

   Hint: If you have already run this once, and saved the baseline results, the script `scripts/run_e2e_jobs.sh` can be used to execute additional trials in parallel.

5. (Optional) Target-based next-token calibration:
   ```bash
   python -m main --target_prefix batch_demo --loc repo \
       --train_dataset wmt19 --train_samples 40000 --train_max_tokens 128 \
       --use_ivfpq --subsample_fraction 1.0 \
       --run_calibration \
       --cal_dataset wmt19 --cal_samples 500 --cal_max_tokens 128 --cal_target_precisions 0.62 0.64 0.65 0.66 0.68 --cal_batch_size 128
   ```
   This creates a `calibration/` sub-folder, and for the chosen db-calibration dataset, it creates a sub-folder (e.g. `db_ivfpq_subsampled_100pct_wmt19_validation_500s_128t_token_match`) to which it saves the precisions as subfolders, alongside the calibrated thresholds in `thresholds.json`.
   Either `--cal_target_precisions` or `--cal_hit_rates` can be specified to derive the thresholds. Note that the initial simulation and vector extraction can be slow, but running calibration with additional targets afterwards will be fast, using cached results.

6. Evaluate the framework (fixed thresholds):
   ```bash
   python -m main --target_prefix batch_demo --loc repo \
       --train_dataset wmt19 --train_samples 40000 --train_max_tokens 128 \
       --use_ivfpq --subsample_fraction 1.0 \
       --run_evaluation \
       --eval_dataset wmt19 --eval_samples 100 --eval_max_tokens 128 \
       --decision_strategy softmax_expected_skip --kv_strategy project_only \
       --manual_thresholds 0.90
   ```
   The online decision strategy can be configured with `--decision_strategy` and `--decision_strategy_k`, while the KV cache strategy can be configured with `--kv_strategy`.
   The `manual_thresholds` argument takes in a single value or a list of values to evaluate. If a list is provided, it evaluates each threshold separately.
   Since the evaluation stage typically does not 100% utilise the GPU (depending on model/GPU size), the script `scripts/run_threshold_sweep.sh` can be used to instantiate evaluations in parallel.

7. Evaluate calibration results (only if next-token calibration has run)
   ```
   python -m main --target_prefix batch_demo --loc repo \
      --train_dataset wmt19 --train_samples 40000 --train_max_tokens 128 \
      --use_ivfpq --subsample_fraction 1.0 \
      --run_calibration  --cal_dataset wmt19 --cal_samples 500 --cal_max_tokens 128 --cal_target_precisions 0.62 \
      --eval_dataset wmt19 --eval_samples 100 --eval_max_tokens 128 \
      --run_evaluation
   ```
   This will save the results in the specific `calibration/` sub-folder.
   Alternatively, you can provide the `--eval_calibration_run` argument with the calibration sub-folder path containing `thresholds.json` to evaluate.

8. Evaluate baselines:
   ```
   # unified layer skipping
   python -m main --target_prefix batch_demo --loc repo \
       --eval_dataset wmt19 --eval_samples 100 --eval_max_tokens 128 \
       --eval_unified_skip_ratios 1.05 1.10 1.15

   # probability-based random block skipping
   python -m main --target_prefix batch_demo --loc repo \
       --eval_dataset wmt19 --eval_samples 100 --eval_max_tokens 128 \
       --kv_strategy full_compute \
       --eval_random_skip_probs 0.05 0.10
   ```
   We provide only the experiment-identifying parameters, and the baseline-specific parameters. This saves in the `baselines/` sub-folder of the experiment folder.


### Initial analysis scripts

Preliminary analysis scripts used to profile the foundational layer redundancy of the models can be found in the initial_analysis folder. Run them using, e.g.,:
   ```
   python -m initial_analysis.runner
   ```


### Plotting

To aggregate metrics and generate standard plots (Pareto frontiers, threshold sensitivity curves, skip distribution histograms, etc.):
```bash
python -m eval.main_eval_plots
```
This file will need to be updated with your specific experiment folder paths and plotting settings.


## Virtual Pipelining

For an online serving environment, we extended the Mini-SGLang serving system to support our retrieval-guided skipping framework.
This implements on-device state ledgers and dual-mode CUDA graphs to translate theoretical reduction into measurable wall-clock acceleration.

Please see `semantic-layer-skipping/mini-sglang` for architecture details and serving benchmarks.


## Additional Information

### Running on HPC

To submit batch jobs to the GPU partition on Wilkes3, use the provided SLURM scripts:
```bash
sbatch scripts/slurm_submit_population.wilkes3
```
To submit to the CPU partition, use:
```bash
scripts/cpu_submit.sh
```

To monitor the full names of your currently queued or active jobs:
```bash
squeue -u <crsid> -O JobID,Partition,Name:70
```

### Result inspection

Evaluation metrics and outputs are saved as detailed JSON files in the results folder. To rapidly inspect the generated text outputs directly in the terminal, use jq:
```
jq '.' results/sharegpt_test_100s_2048t_top1_strict_full_generation.json | grep "text" | less -S
```
This command decodes the JSON and outputs the generated text onto a single horizontally scrollable line per entry, making qualitative analysis much simpler.


## Licensing

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
