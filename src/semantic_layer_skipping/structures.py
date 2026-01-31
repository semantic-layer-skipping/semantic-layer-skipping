from dataclasses import dataclass
from enum import Enum, auto


# --- Skipping Decision ---
class Action(Enum):
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
