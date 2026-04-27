import logging
import os
from dataclasses import dataclass, field

from inference.strategies import (
    EarlyExitStrategyMode,
    OnlineStrategyType,
    SkipStrategyMode,
)
from structures import (
    CalibrationSuccessStrategy,
    DatasetName,
    DatasetSplit,
    EvalStrategy,
)
from utils import get_experiment_output_dir


@dataclass
class PopulationConfig:
    """Defines the Base state to populate a store for a specific experiment."""

    # optional prefix for the folder name
    run_prefix: str | None = None

    # if None, will be auto-generated in __post_init__
    experiment_name: str | None = None

    output_dir: str | None = None

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
    kl_threshold: float = 2

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = get_experiment_output_dir()

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
            # StrEnum results in string representation in name
            parts.append(self.skip_strategy_mode)
            if self.skip_strategy_mode == SkipStrategyMode.KL_DIVERGENCE:
                parts.append(f"thresh{self.kl_threshold}")
            parts.append(self.early_exit_strategy_mode)
            if self.early_exit_strategy_mode == SkipStrategyMode.KL_DIVERGENCE:
                parts.append(f"thresh{self.kl_threshold}")

            # add checkpoints to the end
            checkpoint_str = "c" + "-".join(map(str, self.checkpoints))
            parts.append(checkpoint_str)

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
    run_name: str | None = None  # stores precision run
    data_run_name: str | None = None  # stores the raw vectors

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
        if self.data_run_name is None:
            parts = []
            if self.run_prefix:
                parts.append(sanitise_prefix(self.run_prefix))

            parts.append(self.dataset.value)
            parts.append(self.split.value)
            parts.append(f"{self.num_samples}s")
            parts.append(f"{self.max_gen_tokens}t")
            parts.append(self.success_strategy.value)

            self.data_run_name = "_".join(parts)

        if self.run_name is None:
            # nest the precision under the raw data directory
            self.run_name = f"{self.data_run_name}/precisions/{self.target_precision}p"


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

    # online inference
    online_decision_strategy_type: OnlineStrategyType = OnlineStrategyType.TOP1_STRICT

    # evaluation
    max_total_tokens: int = 25
    strategy: EvalStrategy = EvalStrategy.FULL_GENERATION

    # thresholds
    thresholds: dict[int, float] | None = None  # if None, loaded from calibration_run

    def __post_init__(self):
        if self.thresholds is None and self.calibration_run == "manual_thresholds":
            raise ValueError("Manual evaluation requires explicit thresholds.")

        if self.run_name is None:
            parts = []
            if self.run_prefix:
                parts.append(sanitise_prefix(self.run_prefix))

            parts.append(self.dataset.value)
            parts.append(self.split.value)
            parts.append(f"{self.num_samples}s")
            parts.append(f"{self.max_total_tokens}t")
            parts.append(self.online_decision_strategy_type)
            parts.append(self.strategy.value)

            if self.thresholds is not None:
                sorted_ckpts = sorted(self.thresholds.keys())
                # convert float to string ("85")
                thresh_strings = [
                    str(self.thresholds[k]).lstrip("0.") for k in sorted_ckpts
                ]
                parts.append("thresh-" + "-".join(thresh_strings))

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
