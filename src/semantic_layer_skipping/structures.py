from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any


# --- Skipping Decision ---
class Action(StrEnum):
    CONTINUE = auto()  # TODO: do we want to store this case in DBs?
    EXIT = auto()
    SKIP = auto()


@dataclass
class SkipDecision:
    action: Action
    skip_count: int = 0  # only used if action is SKIP

    def __str__(self):
        if self.action == Action.SKIP:
            return f"SKIP-{self.skip_count}"
        return self.action.name


# --- DB Search Result ---
@dataclass
class SearchResult:
    similarity: float
    decision: SkipDecision
    neighbour_id: int

    def __str__(self):
        return (
            f"SearchResult(similarity={self.similarity:.2f}, "
            f"decision={self.decision}, neighbour_id={self.neighbour_id})"
        )


# -- Dataset Sample --
@dataclass
class DatasetSample:
    id: str
    prompt: str | list[dict[str, str]]

    label: str | None = None

    # classification / multiple-choice options (if applicable)
    choices: list[str] | None = None

    # length and tokenizer info
    prompt_length: int | None = None
    tokenizer_name: str | None = None

    # metadata for routing or analysis (e.g., "math", "coding", "complexity_score")
    metadata: dict[str, Any] = field(default_factory=dict)


# -- Datasets --
class DatasetName(StrEnum):
    NEWTON = auto()
    GSM8K = auto()
    SHAREGPT = auto()
    BOOLQ = auto()
    MMLU = auto()
    QQP = auto()
    E2E = auto()
    WMT19 = auto()


class DatasetSplit(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


# -- Calibration Result ---
class CalibrationSuccessStrategy(StrEnum):
    TOKEN_MATCH = auto()
    # future extensibility can consider task success:
    # TASK_SUCCESS = auto()


# -- Generation Result --
@dataclass
class SkipGenerationResult:
    full_text: str
    generated_text: str
    # raw token IDs of the new part
    generated_tokens: list[int]
    prompt_tokens: list[int]

    generated_token_count: int
    skipped_layers: int

    # tracks, for each checkpoint, number of blocks skipped from there
    # e.g., {0: {0: 15, 2: 5}, 1: {0: 20, 1: 3, 'exit': 1}}
    checkpoint_skip_counts: dict[int, dict[Any, int]] = field(default_factory=dict)

    # tracks the frequency of each returned neighbour ID per checkpoint
    db_hit_counts: dict[int, dict[int, int]] = field(default_factory=dict)
    # tracks the total number of items in each checkpoint index
    db_index_sizes: dict[int, int] = field(default_factory=dict)

    # tracks how many tokens skipped how many layers
    token_skip_distribution: dict[int, int] = field(default_factory=dict)


# -- Evaluation Strategy --
class EvalStrategy(StrEnum):
    # run both fully, compare strings (Levenshtein/BLEU)
    FULL_GENERATION = auto()
    # run skipping first, then check if Baseline agrees step-by-step
    INCREMENTAL_MATCH = auto()
