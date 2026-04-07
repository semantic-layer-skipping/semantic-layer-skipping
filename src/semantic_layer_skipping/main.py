import argparse
import datetime
import json
import logging
import os
from dataclasses import asdict

from calibration.calibrator import SkipCalibrator
from data.loader import DatasetFactory
from experiment.config import CalibrationConfig, EvalConfig, PopulationConfig
from experiment.evaluator import run_eval_loop
from experiment.manager import ExperimentManager
from inference.base_runner import SemanticSkipRunner
from inference.torch_runner import TorchSkipRunner
from inference.transformer_lens_runner import LensSkipRunner
from store import SkippingVectorDB, verify_and_set_faiss_threads
from structures import DatasetName, DatasetSplit, EvalStrategy
from transformers import AutoTokenizer
from utils import set_logging_config


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
            total_final_tokens=pop_cfg.train_max_tokens,
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
    runner: SemanticSkipRunner,
    db: SkippingVectorDB,
    manager: ExperimentManager,
    calibration_configs: list[CalibrationConfig],
    tokenizer,
):
    logging.info("STARTING CALIBRATION")
    calibrator = SkipCalibrator(runner, db)

    for cal_cfg in calibration_configs:
        if manager.calibration_exists(cal_cfg.run_name):
            logging.info(f"Calibration '{cal_cfg.run_name}' exists. Skipping.")
            continue

        logging.info(f"Running Calibration: {cal_cfg.run_name}")
        samples = DatasetFactory.get_dataset(
            cal_cfg.dataset, cal_cfg.split, cal_cfg.num_samples, tokenizer=tokenizer
        )

        calibrator.reset_results()
        calibrator.run_calibration_pass(
            samples,
            max_new_tokens=cal_cfg.max_gen_tokens,
            success_strategy=cal_cfg.success_strategy,
        )
        thresholds = calibrator.find_optimal_thresholds(cal_cfg.target_precision)
        manager.save_calibration_state(cal_cfg, thresholds)


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

        dataset = DatasetFactory.get_dataset(
            eval_cfg.dataset,
            eval_cfg.split,
            eval_cfg.num_samples,
            tokenizer=tokenizer,
            max_total_tokens=eval_cfg.max_total_tokens,
        )
        metrics = run_eval_loop(runner, db, active_thresholds, eval_cfg, dataset)
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

    # model and checkpoint setting
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint_start", type=int, default=4)
    parser.add_argument("--checkpoint_end", type=int, default=28)
    parser.add_argument("--checkpoint_step", type=int, default=4)

    # stage to run
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
    parser.add_argument("--train_batch_size", type=int, default=160)
    parser.add_argument(
        "--train_chunk_size",
        type=int,
        default=2048,
        help="Number of samples to process before saving chunk to disk",
    )

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
        "--manual_thresholds",
        type=float,
        nargs="+",
        default=[0.992, 0.994, 0.996, 0.998, 0.9995],
    )

    return parser.parse_args()


if __name__ == "__main__":
    set_logging_config()

    args = parse_args()

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
    )
    manager = ExperimentManager(population_cfg)
    logging.info(f"Experiment Name: {population_cfg.experiment_name}")

    runner = TorchSkipRunner(
        population_cfg.model_name, checkpoints=population_cfg.checkpoints
    )
    if population_cfg.vector_dim is None:
        population_cfg.vector_dim = runner.model.vector_dim

    tokenizer = AutoTokenizer.from_pretrained(population_cfg.model_name)

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
    if args.run_calibration:
        # TODO: replace with torch runner
        lens_runner: SemanticSkipRunner = LensSkipRunner(
            population_cfg.model_name, checkpoints=population_cfg.checkpoints
        )
        cal_configs = [
            CalibrationConfig(
                target_precision=0.95,
                dataset=DatasetName.NEWTON,
                split=DatasetSplit.VALIDATION,
                num_samples=2,
            ),
            # CalibrationConfig(
            #     target_precision=0.80,
            #     dataset=DatasetName.NEWTON,
            #     split=DatasetSplit.VALIDATION,
            #     num_samples=4,
            # ),
        ]
        run_calibration(lens_runner, db, manager, cal_configs, tokenizer)

    # EVALUATION
    if args.run_evaluation:
        # example 1: standard full generation eval
        # (loads thresholds from calibration run)
        # EvalConfig(
        #     calibration_run=cal_configs[0].run_name,
        #     dataset=DatasetName.NEWTON,
        #     split=DatasetSplit.TEST,
        #     num_samples=3,
        #     strategy=EvalStrategy.FULL_GENERATION,
        # )
        # example 2 - manual threshold evaluation

        eval_configs = []
        thresholds = args.manual_thresholds

        for threshold in thresholds:
            eval_config = EvalConfig(
                calibration_run="manual_thresholds",
                dataset=DatasetName(args.eval_dataset),
                split=DatasetSplit.TEST,
                num_samples=args.eval_samples,
                strategy=EvalStrategy.FULL_GENERATION,
                max_total_tokens=args.eval_max_tokens,
                # provide manual thresholds
                thresholds={
                    ckpt_idx: threshold
                    for ckpt_idx in range(len(population_cfg.checkpoints))
                },
            )
            eval_configs.append(eval_config)

        run_evaluation(
            runner, db, manager, eval_configs, tokenizer, db_path=active_db_path
        )

    logging.info("Pipeline Execution Complete.")
