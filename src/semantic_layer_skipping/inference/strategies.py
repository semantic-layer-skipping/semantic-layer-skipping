import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum, auto

import torch
import torch.nn.functional as functional
from structures import Action, SearchResult, SkipDecision


# --- Skip Strategy Modes ---
# note: these are used to select which strategy to use in the runner
class SkipStrategyMode(StrEnum):
    COSINE = auto()
    STRICT = auto()
    KL_DIVERGENCE = auto()


# --- Early Exit Strategies ---


class EarlyExitStrategyMode(StrEnum):
    STRICT_MATCH = auto()
    KL_DIVERGENCE = auto()


class EarlyExitStrategy(ABC):
    """Base class for deciding if a token is 'finished'."""

    @abstractmethod
    def should_exit(
        self, early_logits: torch.Tensor, final_logits: torch.Tensor
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
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

    def __init__(self, threshold: float = 2.0, top_k: int = 50):
        self.threshold = threshold
        self.top_k = top_k

    def should_exit(
        self, early_logits: torch.Tensor, final_logits: torch.Tensor
    ) -> bool:
        # calculate full distributions
        probs_final_full = functional.softmax(final_logits, dim=-1)
        log_early_full = functional.log_softmax(early_logits, dim=-1)

        # isolate the top-K absolute probabilities and indices
        # from the true distribution
        true_top_k_probs, top_k_indices = torch.topk(
            probs_final_full, k=self.top_k, dim=-1
        )

        # gather the absolute log-probabilities from the early distribution
        early_top_k_log_probs = torch.gather(
            log_early_full, dim=-1, index=top_k_indices
        )

        # calculate truncated forward KL divergence
        kl = (
            (true_top_k_probs * (true_top_k_probs.log() - early_top_k_log_probs))
            .sum(dim=-1)
            .item()
        )

        return kl < self.threshold

    def should_exit_batched(
        self, early_logits: torch.Tensor, final_logits: torch.Tensor
    ) -> torch.Tensor:
        # calculate full distributions
        probs_final_full = functional.softmax(final_logits, dim=-1)
        log_early_full = functional.log_softmax(early_logits, dim=-1)

        # isolate the top-K absolute probabilities and indices
        # from the true distribution
        true_top_k_probs, top_k_indices = torch.topk(
            probs_final_full, k=self.top_k, dim=-1
        )

        # early_logits shape: [num_checkpoints, batch_size, vocab]
        # top_k_indices shape: [batch_size, k]
        # expand indices to match checkpoints dimension for gather
        num_checkpoints = early_logits.shape[0]
        expanded_indices = top_k_indices.unsqueeze(0).expand(num_checkpoints, -1, -1)

        # gather the absolute log-probabilities from the early distribution
        early_top_k_log_probs = torch.gather(
            log_early_full, dim=-1, index=expanded_indices
        )

        # expand true probs for broadcasting during KL math
        true_top_k_probs_expanded = true_top_k_probs.unsqueeze(0)

        # calculate truncated forward KL divergence
        kl = (
            true_top_k_probs_expanded
            * (true_top_k_probs_expanded.log() - early_top_k_log_probs)
        ).sum(dim=-1)

        return kl < self.threshold


def get_early_exit_strategy(
    mode: EarlyExitStrategyMode, kl_threshold=2, kl_top_k=50
) -> EarlyExitStrategy:
    if mode == EarlyExitStrategyMode.STRICT_MATCH:
        return StrictMatchStrategy()
    elif mode == EarlyExitStrategyMode.KL_DIVERGENCE:
        return KLDivergenceStrategy(threshold=kl_threshold, top_k=kl_top_k)
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


# Inference-time online decision strategies


@dataclass
class FinalOnlineDecision:
    """The concrete action the Runner must take."""

    skip_decision: SkipDecision  # action and skip count
    # metadata for tracking/logging
    similarities: list[float] = field(
        default_factory=list
    )  # for all retrieved neighbours
    neighbour_ids: list[int] = field(default_factory=list)


class OnlineDecisionStrategy(ABC):
    @property
    @abstractmethod
    def required_k(self) -> int:
        """Tells the DB how many neighbours this strategy needs."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        checkpoint_idx: int,
        retrieved_neighbours: list[SearchResult],
        base_threshold: float,
    ) -> FinalOnlineDecision:
        """
        Evaluates the neighbourhood and decides the action.
        """
        raise NotImplementedError


class Top1StrictStrategy(OnlineDecisionStrategy):
    @property
    def required_k(self) -> int:
        return 1

    def evaluate(
        self, checkpoint_idx, retrieved_neighbours, base_threshold
    ) -> FinalOnlineDecision:
        if not retrieved_neighbours:
            return FinalOnlineDecision(SkipDecision(Action.CONTINUE))

        top_hit = retrieved_neighbours[0]
        if top_hit.similarity < base_threshold:
            return FinalOnlineDecision(
                SkipDecision(Action.CONTINUE),
                similarities=[top_hit.similarity],
                neighbour_ids=[top_hit.neighbour_id],
            )

        return FinalOnlineDecision(
            skip_decision=SkipDecision(
                top_hit.decision.action, top_hit.decision.skip_count
            ),
            similarities=[top_hit.similarity],
            neighbour_ids=[top_hit.neighbour_id],
        )


class SafeKNNStrategy(OnlineDecisionStrategy):
    """
    Retrieves k neighbours. Requires all to meet the base threshold.
    - If ANY are CONTINUE, final action is CONTINUE.
    - If ALL are EXIT, final action is EXIT.
    - Otherwise, executes a SKIP with the minimum jump among the SKIP actions.
    """

    def __init__(self, k: int = 3):
        self._k = k

    @property
    def required_k(self) -> int:
        return self._k

    def evaluate(
        self, checkpoint_idx, retrieved_neighbours, base_threshold
    ) -> FinalOnlineDecision:
        if len(retrieved_neighbours) < self._k:
            return FinalOnlineDecision(SkipDecision(Action.CONTINUE))

        # filter strictly above threshold
        valid_hits = [n for n in retrieved_neighbours if n.similarity >= base_threshold]

        if len(valid_hits) < self._k:
            # we don't have enough neighbours to form a safe consensus
            return FinalOnlineDecision(SkipDecision(Action.CONTINUE))

        neighbour_ids = [n.neighbour_id for n in valid_hits]
        similarities = [n.similarity for n in valid_hits]

        # Rule 1: if ANY are CONTINUE, we must continue (safest)
        if any(n.decision.action == Action.CONTINUE for n in valid_hits):
            return FinalOnlineDecision(
                SkipDecision(Action.CONTINUE), similarities, neighbour_ids
            )

        # Rule 2: if ALL are EXIT, we can safely exit
        if all(n.decision.action == Action.EXIT for n in valid_hits):
            return FinalOnlineDecision(
                SkipDecision(Action.EXIT), similarities, neighbour_ids
            )

        # Rule 3: otherwise, find the minimum safe skip among the SKIP actions.
        # (we ignore EXITs here, as EXIT implies skipping to the end,
        # which is > any skip_count)
        skip_hits = [n for n in valid_hits if n.decision.action == Action.SKIP]
        min_skip = min(n.decision.skip_count for n in skip_hits)

        return FinalOnlineDecision(
            skip_decision=SkipDecision(Action.SKIP, skip_count=min_skip),
            similarities=similarities,
            neighbour_ids=neighbour_ids,
        )


class ConsensusDecayStrategy(OnlineDecisionStrategy):
    """
    Lowers the required threshold if all k neighbours agree on the exact same action.
    Otherwise, applies the default strict threshold to the top neighbour.
    """

    def __init__(self, k: int = 3, decay_bonus: float = 0.02):
        self._k = k
        self.decay_bonus = decay_bonus

    @property
    def required_k(self) -> int:
        return self._k

    def evaluate(
        self, checkpoint_idx, retrieved_neighbours, base_threshold
    ) -> FinalOnlineDecision:
        if not retrieved_neighbours:
            return FinalOnlineDecision(SkipDecision(Action.CONTINUE))

        top_hit = retrieved_neighbours[0]
        active_threshold = base_threshold

        # check for unanimous consensus
        if len(retrieved_neighbours) == self._k:
            first_dec = top_hit.decision
            is_unanimous = all(
                n.decision.action == first_dec.action
                and n.decision.skip_count == first_dec.skip_count
                for n in retrieved_neighbours
            )
            if is_unanimous:
                active_threshold -= self.decay_bonus
        neighbour_ids = [n.neighbour_id for n in retrieved_neighbours]
        similarities = [n.similarity for n in retrieved_neighbours]
        # apply the resolved threshold to the top hit
        if top_hit.similarity < active_threshold:
            return FinalOnlineDecision(
                SkipDecision(Action.CONTINUE),
                similarities=similarities,
                neighbour_ids=neighbour_ids,
            )

        return FinalOnlineDecision(
            skip_decision=SkipDecision(
                top_hit.decision.action, top_hit.decision.skip_count
            ),
            similarities=similarities,
            neighbour_ids=neighbour_ids,
        )


class SemanticBoundaryStrategy(OnlineDecisionStrategy):
    """
    Analyses the similarity delta between the 1st and 2nd nearest neighbours.
    If the delta is very small AND the neighbours disagree on the action,
    it raises the threshold to prevent jumping on ambiguous boundaries.
    """

    def __init__(self, boundary_delta: float = 0.005, variance_penalty: float = 0.02):
        self.boundary_delta = boundary_delta
        self.variance_penalty = variance_penalty

    @property
    def required_k(self) -> int:
        return 2

    def evaluate(
        self, checkpoint_idx, retrieved_neighbours, base_threshold
    ) -> FinalOnlineDecision:
        if not retrieved_neighbours:
            return FinalOnlineDecision(SkipDecision(Action.CONTINUE))

        top_hit = retrieved_neighbours[0]
        active_threshold = base_threshold

        if len(retrieved_neighbours) >= 2:
            second_hit = retrieved_neighbours[1]
            sim_delta = top_hit.similarity - second_hit.similarity

            # checks if we are on a boundary between two conflicting decisions
            if sim_delta < self.boundary_delta:
                dec_1 = top_hit.decision
                dec_2 = second_hit.decision

                disagreement = (dec_1.action != dec_2.action) or (
                    dec_1.skip_count != dec_2.skip_count
                )

                if disagreement:
                    # we are in a high-variance/risky boundary zone,
                    # so make the threshold stricter.
                    active_threshold += self.variance_penalty

        if top_hit.similarity < active_threshold:
            return FinalOnlineDecision(
                SkipDecision(Action.CONTINUE),
                similarities=[top_hit.similarity],
                neighbour_ids=[top_hit.neighbour_id],
            )

        return FinalOnlineDecision(
            skip_decision=SkipDecision(
                top_hit.decision.action, top_hit.decision.skip_count
            ),
            similarities=[top_hit.similarity],
            neighbour_ids=[top_hit.neighbour_id],
        )


class SoftmaxExpectedSkipStrategy(OnlineDecisionStrategy):
    """
    Calculates the expected continuous skip hop using a temperature-scaled Softmax
    over the similarities. EXIT actions are treated mathematically as a skip count
    equal to 1 + max_skip_present_in_neighbours.
    """

    def __init__(self, k: int = 5, temperature: float = 0.005):
        self._k = k
        self.temperature = temperature

    @property
    def required_k(self) -> int:
        return self._k

    def evaluate(
        self, checkpoint_idx, retrieved_neighbours, base_threshold
    ) -> FinalOnlineDecision:
        valid_hits = [n for n in retrieved_neighbours if n.similarity >= base_threshold]

        if not valid_hits:
            return FinalOnlineDecision(SkipDecision(Action.CONTINUE))

        similarities = [n.similarity for n in valid_hits]
        neighbour_ids = [n.neighbour_id for n in valid_hits]

        # if any hits are CONTINUE, abort the expected value calculation to be safe
        if any(n.decision.action == Action.CONTINUE for n in valid_hits):
            return FinalOnlineDecision(
                SkipDecision(Action.CONTINUE),
                similarities=similarities,
                neighbour_ids=neighbour_ids,
            )

        # determine the maximum skip count present among SKIP actions
        skip_hits = [n for n in valid_hits if n.decision.action == Action.SKIP]
        max_skip_present = max((n.decision.skip_count for n in skip_hits), default=0)

        # exit is mathematically treated as one layer past the furthest known skip
        effective_exit_skip = 1 + max_skip_present

        # subtract max similarity to prevent math.exp OverflowError
        max_sim = max(similarities)
        exp_weights = [
            math.exp((sim - max_sim) / self.temperature) for sim in similarities
        ]
        sum_exp = sum(exp_weights)

        # calculate expected value across all valid hits
        expected_skip = 0.0
        for weight, hit in zip(exp_weights, valid_hits, strict=True):
            normalised_weight = weight / sum_exp

            # resolve the numerical value of the hit
            if hit.decision.action == Action.EXIT:
                hit_val = effective_exit_skip
            else:
                hit_val = hit.decision.skip_count
            expected_skip += normalised_weight * hit_val

        # floor to ensure we don't overshoot the safe continuous margin
        final_skip = math.floor(expected_skip)

        if final_skip == 0:
            return FinalOnlineDecision(
                SkipDecision(Action.CONTINUE),
                similarities=similarities,
                neighbour_ids=neighbour_ids,
            )
        elif final_skip == effective_exit_skip:
            return FinalOnlineDecision(
                SkipDecision(Action.EXIT),
                similarities=similarities,
                neighbour_ids=neighbour_ids,
            )
        else:
            return FinalOnlineDecision(
                skip_decision=SkipDecision(Action.SKIP, skip_count=final_skip),
                similarities=similarities,
                neighbour_ids=neighbour_ids,
            )


class OnlineStrategyType(StrEnum):
    """String representations for command-line mapping of inference strategies."""

    TOP1_STRICT = auto()
    SAFE_KNN = auto()
    CONSENSUS_DECAY = auto()
    SEMANTIC_BOUNDARY = auto()
    SOFTMAX_EXPECTED_SKIP = auto()


def get_decision_strategy(
    strategy_type: OnlineStrategyType | str, **kwargs
) -> OnlineDecisionStrategy:
    """
    Factory function to instantiate the correct decision strategy.
    Pass kwargs to override default parameters (e.g., k=5, temperature=0.01).
    """
    # normalise string if passed directly from argparse
    if isinstance(strategy_type, str):
        strategy_type = OnlineStrategyType(strategy_type.lower())

    if strategy_type == OnlineStrategyType.TOP1_STRICT:
        return Top1StrictStrategy()

    elif strategy_type == OnlineStrategyType.SAFE_KNN:
        return SafeKNNStrategy(**kwargs)

    elif strategy_type == OnlineStrategyType.CONSENSUS_DECAY:
        return ConsensusDecayStrategy(**kwargs)

    elif strategy_type == OnlineStrategyType.SEMANTIC_BOUNDARY:
        return SemanticBoundaryStrategy(**kwargs)

    elif strategy_type == OnlineStrategyType.SOFTMAX_EXPECTED_SKIP:
        return SoftmaxExpectedSkipStrategy(**kwargs)

    else:
        raise ValueError(f"Unknown OnlineStrategyType: {strategy_type}")
