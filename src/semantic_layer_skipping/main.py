import logging

from calibration.calibrator import SkipCalibrator
from data.loader import DatasetFactory
from experiment.config import CalibrationConfig, EvalConfig, PopulationConfig
from experiment.evaluator import run_eval_loop
from experiment.manager import ExperimentManager
from inference.runner import SemanticSkipRunner
from structures import DatasetName, DatasetSplit, EvalStrategy

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # base setup
    population_cfg = PopulationConfig(
        checkpoints=list(range(4, 28, 4)),
        train_dataset=DatasetName.NEWTON,
        train_split=DatasetSplit.TRAIN,
        train_samples=3,
        train_max_tokens=50,
    )
    manager = ExperimentManager(population_cfg)
    logging.info(f"Experiment Name: {population_cfg.experiment_name}")

    runner = SemanticSkipRunner(
        population_cfg.model_name, checkpoints=population_cfg.checkpoints
    )
    if population_cfg.vector_dim is None:
        population_cfg.vector_dim = runner.model.cfg.d_model

    # POPULATION
    populate = not manager.db_exists()
    db = manager.initialise_db(force_new=False)  # load or create new
    if populate:
        logging.info("Populating DB...")
        dataset = DatasetFactory.get_dataset(
            population_cfg.train_dataset,
            population_cfg.train_split,
            population_cfg.train_samples,
        )
        for sample in dataset:
            runner.generate_and_populate(
                sample,
                db,
                max_new_tokens=population_cfg.train_max_tokens,
                skip_strategy_mode=population_cfg.skip_strategy_mode,
                early_exit_strategy_mode=population_cfg.early_exit_strategy_mode,
            )
        manager.save_population_state(db)

    # CALIBRATION
    calibrator = SkipCalibrator(runner, db)

    calibration_configs = [
        CalibrationConfig(
            target_precision=0.95,
            dataset=DatasetName.NEWTON,
            split=DatasetSplit.VALIDATION,
            num_samples=4,
        ),
        CalibrationConfig(
            target_precision=0.80,
            dataset=DatasetName.NEWTON,
            split=DatasetSplit.VALIDATION,
            num_samples=4,
        ),
    ]

    for cal_cfg in calibration_configs:
        if manager.calibration_exists(cal_cfg.run_name):
            logging.info(f"Calibration '{cal_cfg.run_name}' exists. Skipping.")
            continue

        logging.info(f"Running Calibration: {cal_cfg.run_name}")

        samples = DatasetFactory.get_dataset(
            cal_cfg.dataset, cal_cfg.split, cal_cfg.num_samples
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
            metrics = run_eval_loop(runner, db, thresholds, eval_cfg)
            manager.save_test_results(eval_cfg, metrics)
        except FileNotFoundError:
            logging.error(f"Could not load thresholds for {eval_cfg.calibration_run}")
