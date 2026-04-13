from enum import StrEnum, auto

import torch
import torch.nn.functional as functional


# --- Skip Strategy Modes ---
# note: these are used to select which strategy to use in the runner
class SkipStrategyMode(StrEnum):
    COSINE = auto()
    STRICT = auto()


# --- Early Exit Strategies ---


class EarlyExitStrategyMode(StrEnum):
    STRICT_MATCH = auto()
    KL_DIVERGENCE = auto()


class EarlyExitStrategy:
    """Base class for deciding if a token is 'finished'."""

    def should_exit(
        self, early_logits: torch.Tensor, final_logits: torch.Tensor
    ) -> bool:
        raise NotImplementedError

    def should_exit_batched(
        self, early_logits: torch.Tensor, final_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            early_logits: Tensor of shape [num_checkpoints, batch_size, vocab_size]
            final_logits: Tensor of shape [batch_size, vocab_size]
        Returns:
            Boolean Tensor of shape [num_checkpoints, batch_size]
        """
        raise NotImplementedError


class StrictMatchStrategy(EarlyExitStrategy):
    """Exits only if the predicted token is EXACTLY the same."""

    def should_exit(
        self, early_logits: torch.Tensor, final_logits: torch.Tensor
    ) -> bool:
        early_token = torch.argmax(early_logits).item()
        final_token = torch.argmax(final_logits).item()
        return early_token == final_token

    def should_exit_batched(
        self, early_logits: torch.Tensor, final_logits: torch.Tensor
    ) -> torch.Tensor:
        # argmax across the vocab dimension
        early_tokens = torch.argmax(
            early_logits, dim=-1
        )  # [num_checkpoints, batch_size]
        final_tokens = torch.argmax(final_logits, dim=-1)  # [batch_size]

        # broadcast final_tokens to match early_tokens shape and compare
        return early_tokens == final_tokens.unsqueeze(0)


class KLDivergenceStrategy(EarlyExitStrategy):
    """Exits if the probability distribution is close enough (soft match)."""

    def __init__(self, threshold: float = 2):
        self.threshold = threshold

    def should_exit(
        self, early_logits: torch.Tensor, final_logits: torch.Tensor
    ) -> bool:
        # KL expects log_probs as input, and standard probs as target
        log_early = functional.log_softmax(early_logits, dim=-1)
        probs_final = functional.softmax(final_logits, dim=-1)

        kl = functional.kl_div(log_early, probs_final, reduction="sum").item()
        return kl < self.threshold

    def should_exit_batched(
        self, early_logits: torch.Tensor, final_logits: torch.Tensor
    ) -> torch.Tensor:
        log_early = functional.log_softmax(early_logits, dim=-1)
        probs_final = functional.softmax(final_logits, dim=-1)

        # broadcast probs_final to match
        # log_early shape [num_checkpoints, batch_size, vocab]
        probs_final_expanded = probs_final.unsqueeze(0).expand_as(log_early)

        # calculate KL div. reduction="none" keeps all dimensions.
        # then we sum across the vocab dimension (dim=-1) to get the KL per sequence.
        kl = functional.kl_div(log_early, probs_final_expanded, reduction="none").sum(
            dim=-1
        )

        return kl < self.threshold


def get_early_exit_strategy(mode: EarlyExitStrategyMode) -> EarlyExitStrategy:
    if mode == EarlyExitStrategyMode.STRICT_MATCH:
        return StrictMatchStrategy()
    elif mode == EarlyExitStrategyMode.KL_DIVERGENCE:
        return KLDivergenceStrategy()
    else:
        raise ValueError(f"Unsupported Early Exit Strategy Mode: {mode}")


# repetition and frequency penalties
def apply_repetition_penalty(
    logits: torch.Tensor, past_tokens: torch.Tensor, penalty: float
) -> None:
    """
    Helper function to apply repetition penalty in-place.
    Uses the same method as HF: https://huggingface.co/docs/transformers/main_classes/text_generation
    Handles both 1D and batched 2D past_tokens safely.
    """
    if penalty <= 1.0:
        return

    # ensure logits is 2D (Batch, Vocab) to handle potential 3D inputs (Batch, 1, Vocab)
    if logits.dim() == 3:
        logits = logits.squeeze(1)

    # apply penalty per sequence in the batch
    for i in range(logits.shape[0]):
        # use the corresponding history for each sequence if past_tokens is batched
        history = past_tokens[i] if past_tokens.dim() > 1 else past_tokens
        seen_tokens = torch.unique(history)

        score = logits[i, seen_tokens]
        # apply the CTRL penalty formula
        penalised_score = torch.where(score < 0, score * penalty, score / penalty)
        logits[i, seen_tokens] = penalised_score


def apply_frequency_penalty(
    logits: torch.Tensor, past_tokens: torch.Tensor, penalty: float
) -> None:
    """
    Helper function to apply OpenAI-style frequency penalty in-place.
    Formula: new_logit = old_logit - (frequency * penalty)
    Handles both 1D and batched 2D past_tokens safely.
    """
    if penalty <= 0.0:
        return

    # ensure logits is 2D (Batch, Vocab) to handle potential 3D inputs (Batch, 1, Vocab)
    if logits.dim() == 3:
        logits = logits.squeeze(1)

    # apply penalty per sequence in the batch
    for i in range(logits.shape[0]):
        # use the corresponding history for each sequence if past_tokens is batched
        history = past_tokens[i] if past_tokens.dim() > 1 else past_tokens
        unique_tokens, counts = torch.unique(history, return_counts=True)

        # apply the additive penalty:
        # subtract (count * penalty) from the specific logits
        logits[i, unique_tokens] -= (counts * penalty).to(logits.dtype)
