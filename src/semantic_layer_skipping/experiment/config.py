import os
from dataclasses import dataclass, field

from inference.strategies import EarlyExitStrategyMode, SkipStrategyMode
from structures import CalibrationSuccessStrategy, EvalStrategy


@dataclass
class PopulationConfig:
    """Defines the Base state to populate a store for a specific experiment."""

    experiment_name: str
    output_dir: str = "experiments"

    # model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    checkpoints: list[int] = field(default_factory=lambda: list(range(4, 28, 4)))
    vector_dim: int = 1536

    # vector db
    index_type: str = "flat"  # TODO: make actual enum and add more options (e.g. IVF)

    # dataset
    train_dataset: str = "newton"
    train_split: str = "train"
    train_samples: int = 3

    # generation params
    train_max_tokens: int = 25
    skip_strategy_mode: SkipStrategyMode = SkipStrategyMode.STRICT
    early_exit_strategy_mode: EarlyExitStrategyMode = EarlyExitStrategyMode.STRICT_MATCH

    @property
    def base_path(self):
        return os.path.join(self.output_dir, self.experiment_name)

    @property
    def db_path(self):
        return os.path.join(self.base_path, "vector_db")


@dataclass
class CalibrationConfig:
    """Defines a specific calibration threshold derivation run."""

    run_name: str

    # dataset
    dataset: str = "newton"
    split: str = "validation"
    num_samples: int = 3

    # strategy
    target_precision: float = 0.9
    success_strategy: CalibrationSuccessStrategy = (
        CalibrationSuccessStrategy.TOKEN_MATCH
    )
    max_gen_tokens: int = 25


@dataclass
class TestConfig:
    """Defines a final evaluation run."""

    run_name: str  # e.g. "test_run_01"

    # allows to use a specific calibration run's thresholds for testing
    calibration_run: str

    # dataset
    dataset: str = "newton"
    split: str = "test"
    num_samples: int = 2

    # evaluation
    max_gen_tokens: int = 25
    strategy: EvalStrategy = EvalStrategy.FULL_GENERATION
