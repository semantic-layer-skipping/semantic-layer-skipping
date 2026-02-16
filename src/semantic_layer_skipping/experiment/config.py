import logging
import os
from dataclasses import dataclass, field

from inference.strategies import EarlyExitStrategyMode, SkipStrategyMode
from structures import (
    CalibrationSuccessStrategy,
    DatasetName,
    DatasetSplit,
    EvalStrategy,
)


@dataclass
class PopulationConfig:
    """Defines the Base state to populate a store for a specific experiment."""

    # optional prefix for the folder name
    run_prefix: str | None = None

    # if None, will be auto-generated in __post_init__
    experiment_name: str | None = None

    output_dir: str = "experiments"

    # model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    checkpoints: list[int] = field(default_factory=lambda: list(range(4, 28, 4)))
    vector_dim: int | None = None  # if None, inferred from model

    # vector db
    index_type: str = "flat"  # TODO: make actual enum and add more options (e.g. IVF)

    # dataset
    train_dataset: DatasetName = DatasetName.NEWTON
    train_split: DatasetSplit = DatasetSplit.TRAIN
    train_samples: int = 3

    # generation params
    train_max_tokens: int = 25
    skip_strategy_mode: SkipStrategyMode = SkipStrategyMode.STRICT
    early_exit_strategy_mode: EarlyExitStrategyMode = EarlyExitStrategyMode.STRICT_MATCH

    def __post_init__(self):
        if self.experiment_name is None:
            # construct automatic name
            parts = []
            if self.run_prefix:
                parts.append(sanitise_prefix(self.run_prefix))

            parts.append(_shorten_model_name(self.model_name))
            parts.append(self.train_dataset.value)
            parts.append(self.train_split.value)
            parts.append(f"{self.train_samples}s")
            parts.append(f"{self.train_max_tokens}t")

            # include strategies to differentiate population methods
            # using .value assuming StrEnum or similar string representation
            parts.append(self.skip_strategy_mode.value)
            parts.append(self.early_exit_strategy_mode.value)

            self.experiment_name = "_".join(parts)

    @property
    def base_path(self):
        return os.path.join(self.output_dir, self.experiment_name)

    @property
    def db_path(self):
        return os.path.join(self.base_path, "vector_db")


@dataclass
class CalibrationConfig:
    """Defines a specific calibration threshold derivation run."""

    # optional prefix
    run_prefix: str | None = None

    # if None, auto-generated
    run_name: str | None = None

    # dataset
    dataset: DatasetName = DatasetName.NEWTON
    split: DatasetSplit = DatasetSplit.VALIDATION
    num_samples: int = 3

    # strategy
    target_precision: float = 0.9
    success_strategy: CalibrationSuccessStrategy = (
        CalibrationSuccessStrategy.TOKEN_MATCH
    )
    max_gen_tokens: int = 25

    def __post_init__(self):
        if self.run_name is None:
            parts = []
            if self.run_prefix:
                parts.append(sanitise_prefix(self.run_prefix))

            parts.append(self.dataset.value)
            parts.append(self.split.value)
            parts.append(f"{self.num_samples}s")
            parts.append(f"{self.target_precision}p")
            parts.append(f"{self.max_gen_tokens}t")
            parts.append(self.success_strategy.value)

            self.run_name = "_".join(parts)


@dataclass
class EvalConfig:
    """Defines a final evaluation run."""

    # dependency
    calibration_run: str

    # optional prefix
    run_prefix: str | None = None

    # if None, auto-generated
    run_name: str | None = None

    # dataset
    dataset: DatasetName = DatasetName.NEWTON
    split: DatasetSplit = DatasetSplit.TEST
    num_samples: int = 2

    # evaluation
    max_gen_tokens: int = 25
    strategy: EvalStrategy = EvalStrategy.FULL_GENERATION

    def __post_init__(self):
        if self.run_name is None:
            parts = []
            if self.run_prefix:
                parts.append(sanitise_prefix(self.run_prefix))

            parts.append(self.dataset.value)
            parts.append(self.split.value)
            parts.append(f"{self.num_samples}s")
            parts.append(f"{self.max_gen_tokens}t")
            parts.append(self.strategy.value)

            self.run_name = "_".join(parts)


def _shorten_model_name(full_name: str) -> str:
    """Removes the organization prefix, e.g. 'Qwen/Qwen2.5' -> 'Qwen2.5'"""
    if "/" in full_name:
        return full_name.split("/")[-1]
    return full_name


def sanitise_prefix(prefix: str) -> str:
    """Removes or replaces characters that are not suitable for file paths."""
    new_prefix = prefix.replace(" ", "_")
    if "/" in prefix:
        new_prefix = prefix.replace("/", "_")
        logging.warning(
            f"Replaced '/' with '_' in run prefix ({prefix}. New prefix: '{new_prefix}'"
        )

    return new_prefix
