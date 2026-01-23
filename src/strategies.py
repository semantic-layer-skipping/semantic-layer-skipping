import torch
import torch.nn.functional as F
from enum import Enum, auto
from dataclasses import dataclass


# --- Skipping Decision ---

class Action(Enum):
    CONTINUE = auto() # TODO: do we want to store this case in DBs?
    EXIT = auto()
    SKIP = auto()

@dataclass
class SkipDecision:
    action: Action
    skip_count: int = 0  # only used if action is SKIP

    def __str__(self):
        if self.action == Action.SKIP:
            return f"SKIP {self.skip_count} layers"
        return self.action.name


# --- Skip Strategy Modes ---
# note: these are used to select which strategy to use in the runner
class SkipStrategyMode(Enum):
    COSINE = auto()
    STRICT = auto()


# --- Early Exit Strategies ---

class EarlyExitStrategy:
    """Base class for deciding if a token is 'finished'."""

    def should_exit(self, early_logits: torch.Tensor, final_logits: torch.Tensor) -> bool:
        raise NotImplementedError


class StrictMatchStrategy(EarlyExitStrategy):
    """Exits only if the predicted token is EXACTLY the same."""

    def should_exit(self, early_logits: torch.Tensor, final_logits: torch.Tensor) -> bool:
        early_token = torch.argmax(early_logits).item()
        final_token = torch.argmax(final_logits).item()
        return early_token == final_token


class KLDivergenceStrategy(EarlyExitStrategy):
    """Exits if the probability distribution is close enough (soft match)."""

    def __init__(self, threshold: float = 2):
        self.threshold = threshold

    def should_exit(self, early_logits: torch.Tensor, final_logits: torch.Tensor) -> bool:
        # KL expects log_probs as input, and standard probs as target
        log_early = F.log_softmax(early_logits, dim=-1)
        probs_final = F.softmax(final_logits, dim=-1)

        kl = F.kl_div(log_early, probs_final, reduction='sum').item()
        return kl < self.threshold
