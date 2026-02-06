from dataclasses import dataclass
from enum import StrEnum, auto


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


# -- Calibration Result ---
class CalibrationSuccessStrategy(StrEnum):
    TOKEN_MATCH = auto()
    # future extensibility can consider task success:
    # TASK_SUCCESS = auto()
