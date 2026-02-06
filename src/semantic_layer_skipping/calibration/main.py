import logging

from calibration.calibrator import SkipCalibrator
from inference.runner import SemanticSkipRunner
from inference.strategies import (
    EarlyExitStrategyMode,
    SkipStrategyMode,
)
from store import SkippingVectorDB
from utils import ISAAC_NEWTON_QUESTIONS, question_to_prompt

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. select model
    model = "Qwen/Qwen2.5-1.5B-Instruct"  # has 28 layers

    # 2. determine checkpoints and initialise components
    checkpoints = list(range(4, 28, 4))  # TODO: can use hidden state analysis
    runner = SemanticSkipRunner(model_name=model, checkpoints=checkpoints)
    db = SkippingVectorDB(
        n_checkpoints=len(checkpoints), vector_dim=runner.model.cfg.d_model
    )

    # 3. define datasets
    train_prompts = [question_to_prompt(q) for q in ISAAC_NEWTON_QUESTIONS[:3]]
    calib_prompts = [question_to_prompt(q) for q in ISAAC_NEWTON_QUESTIONS[3:7]]
    test_prompts = [question_to_prompt(q) for q in ISAAC_NEWTON_QUESTIONS[7:]]

    # 4. populate DB with training prompts
    early_exit_strategy = EarlyExitStrategyMode.STRICT_MATCH
    skip_strategy_mode = SkipStrategyMode.STRICT
    for train_prompt in train_prompts:
        runner.generate_and_populate(
            train_prompt,
            db,
            early_exit_strategy_mode=early_exit_strategy,
            skip_strategy_mode=skip_strategy_mode,
        )

    # 5. calibrate
    calibrator = SkipCalibrator(runner, db)
    calibrator.run_calibration_pass(calib_prompts)
    optimal_thresholds = calibrator.find_optimal_thresholds(min_precision=0.90)

    # 6. test inference with thresholds
    for test_prompt in test_prompts:
        runner.generate_with_skipping(test_prompt, db, threshold=optimal_thresholds)
