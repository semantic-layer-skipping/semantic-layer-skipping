import datetime
import logging
import sys

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    POPULATE_ONLY = True

    # base setup
    population_cfg = PopulationConfig(
        run_prefix=f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        # run_prefix=f"batch-torch-train-skip-lenscalib_20260228_213543",
        checkpoints=list(range(4, 28, 4)),
        train_dataset=DatasetName.SHAREGPT,
        train_split=DatasetSplit.TRAIN,
        train_samples=128,
        train_max_tokens=1536,
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

    # POPULATION
    populate = not manager.db_exists()
    db = manager.initialise_db(force_new=False)  # load or create new
    if populate:
        logging.info("Populating DB...")
        dataset = DatasetFactory.get_dataset(
            population_cfg.train_dataset,
            population_cfg.train_split,
            population_cfg.train_samples,
            tokenizer=tokenizer,
        )
        runner.generate_and_populate_batched(
            dataset,
            db,
            early_exit_strategy_mode=population_cfg.early_exit_strategy_mode,
            skip_strategy_mode=population_cfg.skip_strategy_mode,
            total_final_tokens=population_cfg.train_max_tokens,
        )
        manager.save_population_state(db)

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
