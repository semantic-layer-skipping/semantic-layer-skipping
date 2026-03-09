import datetime
import json
import logging
import os
import sys
from dataclasses import asdict

from calibration.calibrator import SkipCalibrator
from data.loader import DatasetFactory
from experiment.config import CalibrationConfig, EvalConfig, PopulationConfig
from experiment.evaluator import run_eval_loop
from experiment.manager import ExperimentManager
from inference.base_runner import SemanticSkipRunner
from inference.torch_runner import TorchSkipRunner
from inference.transformer_lens_runner import LensSkipRunner
from structures import DatasetName, DatasetSplit, EvalStrategy
from transformers import AutoTokenizer


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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    POPULATE_ONLY = True

    # base setup
    population_cfg = PopulationConfig(
        run_prefix=f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        # run_prefix=f"batch-torch-train-skip-lenscalib_20260228_213543",
        checkpoints=list(range(4, 28, 4)),
        train_dataset=DatasetName.SHAREGPT,
        train_split=DatasetSplit.TRAIN,
        train_samples=20_000,
        train_max_tokens=2048,  # max number of tokens, after generation
        # skip_strategy_mode=SkipStrategyMode.COSINE,
    )
    manager = ExperimentManager(population_cfg)
    logging.info(f"Experiment Name: {population_cfg.experiment_name}")
    runner = TorchSkipRunner(
        population_cfg.model_name, checkpoints=population_cfg.checkpoints
    )
    if population_cfg.vector_dim is None:
        population_cfg.vector_dim = runner.model.vector_dim

    tokenizer = AutoTokenizer.from_pretrained(population_cfg.model_name)

    tracking_file = os.path.join(
        manager.population_config.base_path, "processed_ids.json"
    )
    processed_ids = load_processed_ids(tracking_file)
    logging.info(f"Loaded {len(processed_ids)} previously processed prompt IDs.")

    # POPULATION
    # populate = not manager.db_exists()
    # db = manager.initialise_db(force_new=False)  # load or create new

    populate = True
    # first creation is fresh
    db = manager._create_new_db()

    if populate:
        logging.info(f"Getting population dataset {population_cfg.train_dataset}...")
        batched_dataset = DatasetFactory.get_dataset(
            population_cfg.train_dataset,
            population_cfg.train_split,
            population_cfg.train_samples,
            tokenizer=tokenizer,
        )

        CHUNK_SIZE_LIMIT = 2048  # save every 2048 samples
        current_chunk_samples = 0

        batch_size = 160
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
                early_exit_strategy_mode=population_cfg.early_exit_strategy_mode,
                skip_strategy_mode=population_cfg.skip_strategy_mode,
                total_final_tokens=population_cfg.train_max_tokens,
            )

            # track IDs
            for sample in pending_samples:
                processed_ids.add(sample.id)
                current_chunk_samples += 1

            # periodic saving
            if current_chunk_samples >= CHUNK_SIZE_LIMIT or (i == total_batches - 1):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                chunk_folder = os.path.join(
                    manager.population_config.base_path, f"db_chunk_{timestamp}"
                )

                logging.info(
                    f"Flushing chunk of {current_chunk_samples} "
                    f"samples to {chunk_folder}"
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
                    f"Chunk with {len(pending_samples)} samples saved. "
                    f"Initialised new DB."
                )

        # manager.save_population_state(db)

    if POPULATE_ONLY:
        logging.info("Population complete. Exiting as POPULATE_ONLY is set to True.")
        sys.exit(0)

    lens_runner: SemanticSkipRunner = LensSkipRunner(
        population_cfg.model_name, checkpoints=population_cfg.checkpoints
    )

    # CALIBRATION
    calibrator = SkipCalibrator(lens_runner, db)

    calibration_configs = [
        CalibrationConfig(
            target_precision=0.95,
            dataset=DatasetName.NEWTON,
            split=DatasetSplit.VALIDATION,
            num_samples=4,
        ),
        # CalibrationConfig(
        #     target_precision=0.80,
        #     dataset=DatasetName.NEWTON,
        #     split=DatasetSplit.VALIDATION,
        #     num_samples=4,
        # ),
    ]

    for cal_cfg in calibration_configs:
        if manager.calibration_exists(cal_cfg.run_name):
            logging.info(f"Calibration '{cal_cfg.run_name}' exists. Skipping.")
            continue

        logging.info(f"Running Calibration: {cal_cfg.run_name}")

        samples = DatasetFactory.get_dataset(
            cal_cfg.dataset,
            cal_cfg.split,
            cal_cfg.num_samples,
            tokenizer=tokenizer,
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
    eval_configs = []
    for cal_cfg in calibration_configs:
        # full generation evaluation
        eval_configs.append(
            EvalConfig(
                calibration_run=cal_cfg.run_name,
                dataset=DatasetName.NEWTON,
                split=DatasetSplit.TEST,
                num_samples=3,
                strategy=EvalStrategy.FULL_GENERATION,
            )
        )
        # add incremental match strategy for comparison
        eval_configs.append(
            EvalConfig(
                calibration_run=cal_cfg.run_name,
                dataset=DatasetName.NEWTON,
                split=DatasetSplit.TEST,
                num_samples=3,
                strategy=EvalStrategy.INCREMENTAL_MATCH,
            )
        )

    for eval_cfg in eval_configs:
        logging.info(f"Running Evaluation: {eval_cfg.run_name}")

        # load the thresholds specific to the calibration run
        try:
            thresholds = manager.load_thresholds(eval_cfg.calibration_run)
            metrics = run_eval_loop(
                runner, db, thresholds, eval_cfg, tokenizer=tokenizer
            )
            manager.save_test_results(eval_cfg, metrics)
        except FileNotFoundError:
            logging.error(f"Could not load thresholds for {eval_cfg.calibration_run}")
