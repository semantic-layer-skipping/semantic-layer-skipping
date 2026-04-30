import argparse
import datetime
import json
import logging
import os
from dataclasses import asdict

import torch
from calibration.calibrator import SkipCalibrator
from data.loader import DatasetFactory
from experiment.config import CalibrationConfig, EvalConfig, PopulationConfig
from experiment.evaluator import run_eval_loop
from experiment.manager import ExperimentManager
from inference.base_runner import SemanticSkipRunner
from inference.strategies import (
    EarlyExitStrategyMode,
    InjectionStrategyMode,
    OnlineStrategyType,
    SkipStrategyMode,
)
from inference.torch_runner import TorchSkipRunner
from store import SkippingVectorDB, verify_and_set_faiss_threads
from structures import DatasetName, DatasetSplit, EvalStrategy
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import get_experiment_output_dir, set_logging_config


def load_processed_ids(filepath: str) -> set[str]:
    if os.path.exists(filepath):
        with open(filepath) as f:
            return set(json.load(f))
    return set()


def save_processed_ids(filepath: str, processed_ids: set[str]):
    # write to a temp file first
    # this is so we don't interrupt during process
    tmp_filepath = filepath + ".tmp"
    with open(tmp_filepath, "w") as f:
        json.dump(list(processed_ids), f)
    os.replace(tmp_filepath, filepath)


# DISCOVERY
def get_discovery_cache_path(pop_cfg: PopulationConfig) -> str:
    """
    Generates a cache path tied to the model and dataset
    - not the experiment dir
    """
    model_clean = pop_cfg.model_name.split("/")[-1]
    # e.g., output_dir/discovery_Qwen2.5-1.5B_sharegpt_20000s/discovery_stats.pt
    cache_folder = (
        f"discovery_{model_clean}_{pop_cfg.train_dataset.value}"
        f"_{pop_cfg.train_samples}s"
    )
    return os.path.join(pop_cfg.output_dir, cache_folder, "discovery_stats.pt")


def load_discovery_stats(pop_cfg):
    discovery_path = get_discovery_cache_path(pop_cfg)
    if not os.path.exists(discovery_path):
        raise FileNotFoundError(
            f"Injection strategy '{pop_cfg.injection_strategy_mode.value}'"
            f"requested, but discovery file missing at {discovery_path}."
        )
    logging.info(f"Loading discovery stats from {discovery_path}")
    discovery_stats = torch.load(
        discovery_path, map_location=runner.device, weights_only=True
    )
    return discovery_stats


def run_discovery(
    runner: TorchSkipRunner,
    pop_cfg: PopulationConfig,
    tokenizer,
    batch_size: int,
):
    discovery_path = get_discovery_cache_path(pop_cfg)

    if os.path.exists(discovery_path):
        logging.info(f"Discovery stats already exist at {discovery_path}. Skipping.")
        return

    logging.info("STARTING DISCOVERY PHASE")

    dataset = DatasetFactory.get_dataset(
        pop_cfg.train_dataset,
        pop_cfg.train_split,
        pop_cfg.train_samples,
        tokenizer=tokenizer,
        max_total_tokens=pop_cfg.train_max_tokens,
    )

    batches = dataset.get_batches(batch_size=batch_size, strategy="sorted_length")
    total_batches = len(batches)
    aggregated = {}

    for i, batch in enumerate(batches):
        logging.info(f"Discovery batch {i + 1}/{total_batches}...")
        batch_prompts = [s.prompt for s in batch]

        batch_sums = runner.compute_batch_discovery_sums(
            batch_prompts, total_final_tokens=pop_cfg.train_max_tokens
        )

        for l_idx, stats in batch_sums.items():
            if l_idx not in aggregated:
                aggregated[l_idx] = {
                    "count": 0,
                    "sum_x": torch.zeros_like(stats["sum_x"]),
                    "sum_x_sq": torch.zeros_like(stats["sum_x_sq"]),
                    "sum_l2": torch.tensor(0.0, dtype=torch.float64),
                }

            aggregated[l_idx]["count"] += stats["count"]
            aggregated[l_idx]["sum_x"] += stats["sum_x"]
            aggregated[l_idx]["sum_x_sq"] += stats["sum_x_sq"]
            aggregated[l_idx]["sum_l2"] += stats["sum_l2"]

    logging.info("Calculating final distribution statistics...")
    discovery_stats = {}
    for l_idx, totals in aggregated.items():
        N = totals["count"]
        mean = totals["sum_x"] / N
        rms = torch.sqrt(totals["sum_x_sq"] / N)
        var = (totals["sum_x_sq"] / N) - (mean**2)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        global_norm = totals["sum_l2"] / N

        discovery_stats[l_idx] = {
            "mean": mean.to(torch.float32),
            "rms": rms.to(torch.float32),
            "std": std.to(torch.float32),
            "global_norm": global_norm.item(),
        }

    os.makedirs(os.path.dirname(discovery_path), exist_ok=True)
    torch.save(discovery_stats, discovery_path)
    logging.info(f"Successfully saved discovery stats to {discovery_path}")


# POPULATION
def run_population(
    runner: SemanticSkipRunner,
    manager: ExperimentManager,
    pop_cfg: PopulationConfig,
    tokenizer,
    batch_size,
    chunk_size_limit,  # we save every chunk_size_limit samples
):
    logging.info("STARTING POPULATION")
    db = manager._create_new_db()

    discovery_stats = None
    if pop_cfg.injection_strategy_mode is not None:
        discovery_stats = load_discovery_stats(pop_cfg)

    tracking_file = os.path.join(
        manager.population_config.base_path, "processed_ids.json"
    )
    processed_ids = load_processed_ids(tracking_file)
    logging.info(f"Loaded {len(processed_ids)} previously processed prompt IDs.")

    batched_dataset = DatasetFactory.get_dataset(
        pop_cfg.train_dataset,
        pop_cfg.train_split,
        pop_cfg.train_samples,
        tokenizer=tokenizer,
        max_total_tokens=pop_cfg.train_max_tokens,
    )

    current_chunk_samples = 0

    batches = batched_dataset.get_batches(
        batch_size=batch_size, strategy="sorted_length"
    )

    total_batches = len(batches)
    logging.info(f"Dataset split into {total_batches} sorted-length batches.")

    for i, batch in enumerate(batches):
        logging.info(
            f"Processing batch {i + 1}/{total_batches} with {len(batch)} samples..."
        )
        pending_samples = [s for s in batch if s.id not in processed_ids]
        runner.generate_and_populate_batched(
            pending_samples,
            db,
            early_exit_strategy_mode=pop_cfg.early_exit_strategy_mode,
            skip_strategy_mode=pop_cfg.skip_strategy_mode,
            kl_threshold=pop_cfg.kl_threshold,
            total_final_tokens=pop_cfg.train_max_tokens,
            log_prompts=True,
            discovery_stats=discovery_stats,
            injection_strategy=pop_cfg.injection_strategy_mode,
        )

        # track IDs
        for sample in pending_samples:
            processed_ids.add(sample.id)
            current_chunk_samples += 1

        # periodic saving
        if current_chunk_samples >= chunk_size_limit or (i == total_batches - 1):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_folder = os.path.join(
                manager.population_config.base_path, f"db_chunk_{timestamp}"
            )

            logging.info(
                f"Flushing chunk of {current_chunk_samples} samples to {chunk_folder}"
            )
            db.save(chunk_folder)

            # save tracking file atomically
            save_processed_ids(tracking_file, processed_ids)

            # save the experiment config if it doesn't exist yet
            config_path = os.path.join(
                manager.population_config.base_path, "population_config.json"
            )
            if not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    json.dump(asdict(manager.population_config), f, indent=4)

            # reset RAM and make new DB
            del db
            db = manager._create_new_db()
            current_chunk_samples = 0
            logging.info(
                f"Chunk with {len(pending_samples)} samples saved. Initialised new DB."
            )

    logging.info("Population complete.")


# MERGE WITH SUBSAMPLING
def run_merge(
    manager: ExperimentManager, pop_cfg: PopulationConfig, keep_fraction: float
):
    logging.info(f"STARTING MERGE WITH SUBSAMPLING ({keep_fraction * 100})")

    if manager.merged_db_exists(keep_fraction):
        logging.info(
            f"Merged DB with subsampling {keep_fraction} already exists. Skipping."
        )
        return

    output_dir = manager.get_merged_db_path(keep_fraction)

    SkippingVectorDB.create_merged_subsampled_db_from_chunks(
        base_dir=pop_cfg.base_path,
        output_dir=output_dir,
        n_checkpoints=len(pop_cfg.checkpoints),
        vector_dim=pop_cfg.vector_dim,
        keep_fraction=keep_fraction,
    )


def run_ivfpq_conversion(
    manager: ExperimentManager, pop_cfg: PopulationConfig, keep_fraction: float
):
    logging.info(f"STARTING IVFPQ CONVERSION FOR {keep_fraction * 100}% DB")

    if manager.ivfpq_db_exists(keep_fraction):
        logging.info(f"IVFPQ DB for {keep_fraction} already exists. Skipping.")
        return

    # build the IVFPQ db *from* the existing merged/subsampled exact DB
    source_dir = manager.get_merged_db_path(keep_fraction)
    output_dir = manager.get_ivfpq_db_path(keep_fraction)

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Cannot build IVFPQ. Source DB missing: {source_dir}")

    SkippingVectorDB.create_ivfpq_db_from_exact(
        source_dir=source_dir,
        output_dir=output_dir,
        n_checkpoints=len(pop_cfg.checkpoints),
        vector_dim=pop_cfg.vector_dim,
    )


# CALIBRATION
def run_calibration(
    runner: TorchSkipRunner,
    db: SkippingVectorDB,
    manager: ExperimentManager,
    calibration_configs: list[CalibrationConfig],
    tokenizer,
    batch_size: int = 128,
):
    logging.info("STARTING CALIBRATION")
    calibrator = SkipCalibrator(runner, db)

    for cal_cfg in calibration_configs:
        # check if the final threshold file already exists
        if manager.calibration_exists(cal_cfg.run_name):
            logging.info(
                f"Calibration thresholds for '{cal_cfg.run_name}' exist. Skipping."
            )
            continue

        calibrator.reset_results()

        # check if we need to run the heavy simulation loop
        if manager.raw_calibration_exists(cal_cfg.data_run_name):
            manager.load_raw_calibration_results(cal_cfg, calibrator)
        else:
            total_final_tokens = cal_cfg.max_gen_tokens
            logging.info(
                f"Running calibration simulation for data profile: "
                f"{cal_cfg.data_run_name}"
            )

            batched_dataset = DatasetFactory.get_dataset(
                cal_cfg.dataset, cal_cfg.split, cal_cfg.num_samples, tokenizer=tokenizer
            )
            batches = batched_dataset.get_batches(
                batch_size=batch_size, strategy="sorted_length"
            )

            for batch in tqdm(batches, desc="Calibrating Batches"):
                calibrator.run_calibration_batch(
                    prompts=batch, total_final_tokens=total_final_tokens
                )

            # cache the raw simulation results to disk
            manager.save_raw_calibration_results(cal_cfg, calibrator)

        # compute and save the specific precision thresholds
        thresholds = calibrator.find_optimal_thresholds(cal_cfg.target_precision)

        # pass calibrator=None because we already saved the raw results centrally
        manager.save_calibration_state(cal_cfg, thresholds, calibrator=None)


# EVALUATION
def run_evaluation(
    runner: TorchSkipRunner,
    db: SkippingVectorDB,
    manager: ExperimentManager,
    eval_configs: list[EvalConfig],
    tokenizer,
    db_path,
):
    for eval_cfg in eval_configs:
        logging.info(f"Running Evaluation: {eval_cfg.run_name}")
        active_thresholds = None
        try:
            # use manual thresholds if provided, else load from calibration
            if eval_cfg.thresholds is not None:
                logging.info(f"Using manual thresholds: {eval_cfg.thresholds}")
                active_thresholds = eval_cfg.thresholds
            else:
                active_thresholds = manager.load_thresholds(eval_cfg.calibration_run)
        except FileNotFoundError:
            logging.error(
                f"Could not load thresholds for calibration run: "
                f"{eval_cfg.calibration_run}"
            )
            continue
        logging.info(f"Running evaluation with thresholds: {active_thresholds}")
        dataset = DatasetFactory.get_dataset(
            eval_cfg.dataset,
            eval_cfg.split,
            eval_cfg.num_samples,
            tokenizer=tokenizer,
            max_total_tokens=eval_cfg.max_total_tokens,
        )
        discovery_stats = None
        if eval_cfg.injection_strategy_mode is not None:
            discovery_stats = load_discovery_stats(manager.population_config)
        metrics = run_eval_loop(
            runner, db, active_thresholds, eval_cfg, dataset, discovery_stats
        )
        manager.save_test_results(eval_cfg, metrics, db_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Layer Skipping Pipeline")

    # pipeline controls
    parser.add_argument(
        "--target_prefix",
        type=str,
        default=None,
        help="Prefix for the target run (if None, generates new)",
    )
    parser.add_argument(
        "--subsample_fraction",
        type=float,
        default=0.1,
        help="1.0 means merge all chunks",
    )
    parser.add_argument(
        "--loc",
        type=str,
        default=None,
        choices=["repo", "hpc-work", "rds-cl"],
        help="Determines the base path for storing experiment data. "
        "Repo is within the repo, hpc-work is the personal folder "
        "and rds-cl is the additional storage.",
    )

    # model and checkpoint setting
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint_start", type=int, default=4)
    parser.add_argument("--checkpoint_end", type=int, default=28)
    parser.add_argument("--checkpoint_step", type=int, default=4)

    # stage to run
    parser.add_argument("--run_discovery", action="store_true", default=False)
    parser.add_argument("--run_population", action="store_true", default=False)
    parser.add_argument("--run_merge", action="store_true", default=False)
    parser.add_argument("--run_ivfpq_conversion", action="store_true", default=False)
    parser.add_argument(
        "--use_ivfpq",
        action="store_true",
        default=False,
        help="Determines which DB is used for eval",
    )
    parser.add_argument("--run_calibration", action="store_true", default=False)
    parser.add_argument("--run_evaluation", action="store_true", default=False)

    # population/train settings
    parser.add_argument(
        "--train_dataset",
        type=str,
        default=DatasetName.SHAREGPT.value,
        choices=[e.value for e in DatasetName],
    )
    parser.add_argument("--train_samples", type=int, default=20_000)
    parser.add_argument("--train_max_tokens", type=int, default=2048)
    parser.add_argument(
        "--early_exit_strategy",
        type=str,
        default=EarlyExitStrategyMode.STRICT_MATCH,
        choices=[e.value for e in EarlyExitStrategyMode],
    )
    parser.add_argument(
        "--skip_strategy",
        type=str,
        default=SkipStrategyMode.STRICT,
        choices=[e.value for e in SkipStrategyMode],
    )
    parser.add_argument(
        "--kl_threshold",
        type=float,
        default=2.0,
        help="Only used if skip_strategy is KL_DIVERGENCE",
    )
    parser.add_argument(
        "--injection_strategy",
        type=str,
        default=None,
        choices=[e.value for e in InjectionStrategyMode],
        help="Strategy to use for state injection normalisation.",
    )
    parser.add_argument("--train_batch_size", type=int, default=160)
    parser.add_argument(
        "--train_chunk_size",
        type=int,
        default=2048,
        help="Number of samples to process before saving chunk to disk",
    )

    # calibration settings
    parser.add_argument(
        "--cal_dataset",
        type=str,
        default=DatasetName.SHAREGPT.value,
        choices=[e.value for e in DatasetName],
    )
    parser.add_argument("--cal_samples", type=int, default=4)
    parser.add_argument("--cal_max_tokens", type=int, default=2048)
    parser.add_argument("--cal_batch_size", type=int, default=128)
    parser.add_argument("--cal_target_precisions", type=float, nargs="+", default=[0.9])

    # evaluation settings
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default=DatasetName.SHAREGPT.value,
        choices=[e.value for e in DatasetName],
    )
    parser.add_argument("--eval_samples", type=int, default=50)
    parser.add_argument("--eval_max_tokens", type=int, default=256)
    parser.add_argument(
        "--decision_strategy",
        type=str,
        default=OnlineStrategyType.TOP1_STRICT.value,
        choices=[e.value for e in OnlineStrategyType],
        help="The k-NN decision strategy to use during evaluation.",
    )
    parser.add_argument(
        "--eval_calibration_run",
        type=str,
        default=None,
        help="Explicitly provide a calibration run name to pull thresholds from",
    )
    parser.add_argument(
        "--manual_thresholds",
        type=float,
        nargs="+",
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    set_logging_config()

    args = parse_args()

    experiment_output_dir = get_experiment_output_dir(loc=args.loc)
    logging.info(f"Using base experiment directory:  {experiment_output_dir}")

    # base setup
    population_cfg = PopulationConfig(
        model_name=args.model_name,
        run_prefix=args.target_prefix
        or f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        checkpoints=list(
            range(args.checkpoint_start, args.checkpoint_end, args.checkpoint_step)
        ),
        train_dataset=DatasetName(args.train_dataset),
        train_split=DatasetSplit.TRAIN,
        train_samples=args.train_samples,
        train_max_tokens=args.train_max_tokens,
        early_exit_strategy_mode=args.early_exit_strategy,
        skip_strategy_mode=args.skip_strategy,
        kl_threshold=args.kl_threshold,
        injection_strategy_mode=args.injection_strategy,
        output_dir=experiment_output_dir,
    )
    manager = ExperimentManager(population_cfg)
    logging.info(f"Experiment Name: {population_cfg.experiment_name}")

    runner = TorchSkipRunner(
        population_cfg.model_name, checkpoints=population_cfg.checkpoints
    )
    if population_cfg.vector_dim is None:
        population_cfg.vector_dim = runner.model.vector_dim

    tokenizer = AutoTokenizer.from_pretrained(population_cfg.model_name)

    # DISCOVERY
    needs_discovery = args.run_discovery or (
        args.run_population and population_cfg.injection_strategy_mode is not None
    )
    if needs_discovery:
        run_discovery(runner, population_cfg, tokenizer, args.train_batch_size)

    # POPULATION
    if args.run_population:
        run_population(
            runner,
            manager,
            population_cfg,
            tokenizer,
            args.train_batch_size,
            args.train_chunk_size,
        )

    # MERGE WITH SUBSAMPLING
    if args.run_merge:
        run_merge(manager, population_cfg, args.subsample_fraction)

    # IVFPQ CONVERSION
    if args.run_ivfpq_conversion:
        if args.subsample_fraction is None:
            logging.error(
                "Cannot convert unmerged chunks to IVFPQ directly. "
                "Set --subsample_fraction 1.0"
            )
        else:
            verify_and_set_faiss_threads()
            run_ivfpq_conversion(manager, population_cfg, args.subsample_fraction)

    # DB LOADING
    db = None
    active_db_path = None
    if args.run_calibration or args.run_evaluation:
        if args.subsample_fraction is None:
            # TODO: handle initialisation with unmerged chunks
            #  currently, this is not initialising a new one
            db = manager.initialise_db(ensure_exists=True)
            active_db_path = manager.population_config.db_path
        else:
            if args.use_ivfpq:
                logging.info(
                    f"Loading {args.subsample_fraction * 100}% IVFPQ DB into RAM..."
                )
                db = manager.load_ivfpq_db(args.subsample_fraction)
                active_db_path = manager.get_ivfpq_db_path(args.subsample_fraction)
            else:
                logging.info(
                    f"Loading {args.subsample_fraction * 100}% Exact Subsampled DB "
                    f"into RAM..."
                )
                db = manager.load_merged_db(args.subsample_fraction)
                active_db_path = manager.get_merged_db_path(args.subsample_fraction)

    # CALIBRATION
    cal_configs = []
    if args.run_calibration:
        # extract the folder name of the active vector db
        db_folder_name = (
            os.path.basename(os.path.normpath(active_db_path))
            if active_db_path
            else "default_db"
        )
        logging.info(f"Calibrating {db_folder_name}...")
        cal_configs = [
            CalibrationConfig(
                run_prefix=db_folder_name,
                target_precision=precision,
                dataset=DatasetName(args.cal_dataset),
                split=DatasetSplit.VALIDATION,
                num_samples=args.cal_samples,
                max_gen_tokens=args.cal_max_tokens,
            )
            for precision in args.cal_target_precisions
        ]
        run_calibration(
            runner, db, manager, cal_configs, tokenizer, args.cal_batch_size
        )

    # EVALUATION
    if args.run_evaluation:
        eval_configs = []

        # branch 1: evaluate using fixed manual thresholds (if provided)
        if args.manual_thresholds is not None:
            for threshold in args.manual_thresholds:
                eval_configs.append(
                    EvalConfig(
                        calibration_run="manual_thresholds",
                        dataset=DatasetName(args.eval_dataset),
                        split=DatasetSplit.TEST,
                        num_samples=args.eval_samples,
                        strategy=EvalStrategy.FULL_GENERATION,
                        max_total_tokens=args.eval_max_tokens,
                        thresholds={
                            ckpt_idx: threshold
                            for ckpt_idx in range(len(population_cfg.checkpoints))
                        },
                        online_decision_strategy_type=args.decision_strategy,
                        injection_strategy_mode=population_cfg.injection_strategy_mode,
                    )
                )

        # branch 2: evaluate using calibration stats
        else:
            # option 1: user explicitly provides a single path
            if args.eval_calibration_run:
                eval_configs.append(
                    EvalConfig(
                        calibration_run=args.eval_calibration_run,
                        dataset=DatasetName(args.eval_dataset),
                        split=DatasetSplit.TEST,
                        num_samples=args.eval_samples,
                        strategy=EvalStrategy.FULL_GENERATION,
                        max_total_tokens=args.eval_max_tokens,
                        online_decision_strategy_type=args.decision_strategy,
                        injection_strategy_mode=population_cfg.injection_strategy_mode,
                    )
                )
            else:
                # dynamically reconstruct the expected calibration paths based on args
                db_folder_name = (
                    os.path.basename(os.path.normpath(active_db_path))
                    if active_db_path
                    else "default_db"
                )

                for precision in args.cal_target_precisions:
                    # create a mock config to compute the deterministic run_name
                    target_cal_cfg = CalibrationConfig(
                        run_prefix=db_folder_name,
                        target_precision=precision,
                        dataset=DatasetName(args.cal_dataset),
                        split=DatasetSplit.VALIDATION,
                        num_samples=args.cal_samples,
                        max_gen_tokens=args.cal_max_tokens,
                    )

                    # verify the threshold JSON actually exists on disk before queueing
                    if manager.calibration_exists(target_cal_cfg.run_name):
                        eval_configs.append(
                            EvalConfig(
                                calibration_run=target_cal_cfg.run_name,
                                dataset=DatasetName(args.eval_dataset),
                                split=DatasetSplit.TEST,
                                num_samples=args.eval_samples,
                                strategy=EvalStrategy.FULL_GENERATION,
                                max_total_tokens=args.eval_max_tokens,
                                online_decision_strategy_type=args.decision_strategy,
                                injection_strategy_mode=population_cfg.injection_strategy_mode,
                            )
                        )
                    else:
                        logging.warning(
                            f"Skipping evaluation for {precision} precision: "
                            f"Calibration folder '{target_cal_cfg.run_name}' not found."
                        )

                if not eval_configs:
                    logging.error(
                        "No valid calibration runs found to evaluate. "
                        "Ensure your --cal_* arguments match an existing run, "
                        "or run calibration first."
                    )

        if eval_configs:
            run_evaluation(
                runner, db, manager, eval_configs, tokenizer, db_path=active_db_path
            )

    logging.info("Pipeline Execution Complete.")
