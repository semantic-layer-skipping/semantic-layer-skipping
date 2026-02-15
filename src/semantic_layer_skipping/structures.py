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

    def __str__(self):
        return (
            f"SearchResult(similarity={self.similarity:.2f}, decision={self.decision})"
        )


# -- Dataset Sample --
@dataclass
class DatasetSample:
    id: str
    prompt: str | list[dict[str, str]]

    label: str | None = None

    # classification / multiple-choice options (if applicable)
    choices: list[str] | None = None

    # metadata for routing or analysis (e.g., "math", "coding", "complexity_score")
    metadata: dict[str, Any] = field(default_factory=dict)


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


# -- Evaluation Strategy --
class EvalStrategy(StrEnum):
    # run both fully, compare strings (Levenshtein/BLEU)
    FULL_GENERATION = auto()
    # run skipping first, then check if Baseline agrees step-by-step
    INCREMENTAL_MATCH = auto()
    # TODO: compare output against a gold label answer
    TASK_ACCURACY = auto()
