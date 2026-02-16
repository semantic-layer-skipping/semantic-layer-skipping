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
        experiment_name="newton_second",
        checkpoints=list(range(4, 28, 4)),
        train_dataset=DatasetName.NEWTON,
        train_split=DatasetSplit.TRAIN,
        train_samples=3,
    )
    manager = ExperimentManager(population_cfg)
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
    run_calibration = False
    # run A: high precision
    cal_cfg_strict = CalibrationConfig(
        run_name="exact_95",
        target_precision=0.95,
        dataset=DatasetName.NEWTON,
        split=DatasetSplit.VALIDATION,
        num_samples=4,
    )

    calibrator = SkipCalibrator(runner, db)
    samples = DatasetFactory.get_dataset(
        cal_cfg_strict.dataset, cal_cfg_strict.split, cal_cfg_strict.num_samples
    )
    if run_calibration:
        logging.info("Running Calibration: 0.95")
        calibrator.run_calibration_pass(
            samples,
            max_new_tokens=cal_cfg_strict.max_gen_tokens,
            success_strategy=cal_cfg_strict.success_strategy,
        )
        thresholds_strict = calibrator.find_optimal_thresholds(
            cal_cfg_strict.target_precision
        )
        manager.save_calibration_state(cal_cfg_strict, thresholds_strict)

    # 80 precision run for comparison
    cal_cfg_loose = CalibrationConfig(
        run_name="exact_80",
        target_precision=0.80,
        dataset=DatasetName.NEWTON,
        split=DatasetSplit.VALIDATION,
        num_samples=4,
    )
    logging.info("Running Calibration: 0.80")

    if run_calibration:
        calibrator.reset_results()
        calibrator.run_calibration_pass(samples)
        thresholds_loose = calibrator.find_optimal_thresholds(
            cal_cfg_loose.target_precision
        )
        manager.save_calibration_state(cal_cfg_loose, thresholds_loose)

    # EVALUATION
    logging.info("Evaluating...")
    configs = [
        EvalConfig(
            run_name="test_incremental_match_95_on_newton",
            calibration_run="exact_95",
            dataset=DatasetName.NEWTON,
            split=DatasetSplit.TEST,
            num_samples=3,
            strategy=EvalStrategy.INCREMENTAL_MATCH,
        ),
        EvalConfig(
            run_name="test_incremental_match_80_on_newton",
            calibration_run="exact_80",
            dataset=DatasetName.NEWTON,
            split=DatasetSplit.TEST,
            num_samples=3,
            strategy=EvalStrategy.INCREMENTAL_MATCH,
        ),
    ]

    for eval_cfg in configs:
        thresholds = manager.load_thresholds(eval_cfg.calibration_run)
        metrics = run_eval_loop(runner, db, thresholds, eval_cfg)
        manager.save_test_results(eval_cfg, metrics)
