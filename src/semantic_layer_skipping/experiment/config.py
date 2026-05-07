import logging
import os
from dataclasses import dataclass, field

from calibration.calibrator import CalibrationStrategyMode
from inference.strategies import (
    EarlyExitStrategyMode,
    InjectionStrategyMode,
    KVStrategyMode,
    OnlineStrategyMode,
    SkipStrategyMode,
)
from structures import (
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
    injection_strategy_mode: InjectionStrategyMode | None = None

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
            if self.early_exit_strategy_mode == EarlyExitStrategyMode.KL_DIVERGENCE:
                parts.append(f"thresh{self.kl_threshold}")
            if self.injection_strategy_mode:
                parts.append(f"inj_{self.injection_strategy_mode}")

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
    max_gen_tokens: int = 25

    # strategy
    strategy: CalibrationStrategyMode = CalibrationStrategyMode.TOKEN_MATCH

    # targets - can be None depending on target
    target_precision: float | None = None
    target_hit_rate: float | dict[int, float] | None = None
    kl_success_threshold: float | None = None

    def __post_init__(self):
        if self.data_run_name is None:
            parts = []
            if self.run_prefix:
                parts.append(sanitise_prefix(self.run_prefix))

            parts.append(self.dataset.value)
            parts.append(self.split.value)
            parts.append(f"{self.num_samples}s")
            parts.append(f"{self.max_gen_tokens}t")
            # raw data is strategy-independent: we add this to ensure
            # legacy cache hits
            parts.append("token_match")

            self.data_run_name = "_".join(parts)

        if self.run_name is None:
            if self.strategy == CalibrationStrategyMode.HIT_RATE:
                if isinstance(self.target_hit_rate, dict):
                    hr_str = "-".join([str(v) for v in self.target_hit_rate.values()])
                    self.run_name = f"{self.data_run_name}/hit_rates/staggered_{hr_str}"
                else:
                    self.run_name = (
                        f"{self.data_run_name}/hit_rates/{self.target_hit_rate}hr"
                    )
            elif self.strategy == CalibrationStrategyMode.KL_DIVERGENCE:
                assert self.kl_success_threshold is not None, (
                    "Expected kl threshold to be provided"
                )
                self.run_name = (
                    f"{self.data_run_name}/kl_divergence/"
                    f"{self.target_precision}p_"
                    f"{self.kl_success_threshold}kl"
                )
            else:
                # legacy token match structure
                self.run_name = (
                    f"{self.data_run_name}/precisions/{self.target_precision}p"
                )


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
    online_decision_strategy_mode: OnlineStrategyMode = OnlineStrategyMode.TOP1_STRICT
    injection_strategy_mode: InjectionStrategyMode | None = None
    kv_strategy_mode: KVStrategyMode | None = None

    # evaluation
    max_total_tokens: int = 25
    strategy: EvalStrategy = EvalStrategy.FULL_GENERATION

    # thresholds
    thresholds: dict[int, float] | None = None  # if None, loaded from calibration_run

    # baseline configs
    random_skip_prob: float | None = None

    def __post_init__(self):
        if (
            self.thresholds is None and self.calibration_run == "manual_thresholds"
        ) and self.random_skip_prob is None:
            raise ValueError("Manual evaluation requires explicit thresholds.")

        if self.run_name is None:
            parts = []
            if self.run_prefix:
                parts.append(sanitise_prefix(self.run_prefix))

            parts.append(self.dataset.value)
            parts.append(self.split.value)
            parts.append(f"{self.num_samples}s")
            parts.append(f"{self.max_total_tokens}t")

            if self.random_skip_prob is not None:
                parts.append(f"baseline_random_prob_{self.random_skip_prob}")
            else:
                parts.append(self.online_decision_strategy_mode)
                parts.append(self.strategy.value)

            if self.injection_strategy_mode:
                parts.append(f"inj_{self.injection_strategy_mode}")

            if self.kv_strategy_mode:
                parts.append(f"kv_{self.kv_strategy_mode}")

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
