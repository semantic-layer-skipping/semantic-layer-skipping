import logging
from abc import ABC, abstractmethod

import torch
from inference.model import Model
from inference.strategies import (
    EarlyExitStrategyMode,
    SkipStrategyMode,
)
from store import SkippingVectorDB
from structures import DatasetSample, SkipDecision, SkipGenerationResult
from utils import get_device

PromptType = str | list[dict] | DatasetSample
DEFAULT_THRESH = 0.7


class EarlyExitSignal(Exception):  # noqa: N818
    def __init__(self, final_logits):
        self.final_logits = final_logits


# context for shared state during inference
class SkipCtx:
    def __init__(self):
        self.skipping_active = False
        self.landing_layer = -1
        self.teleport_vector = None
        self.skipped_layers_count = 0


class SemanticSkipRunner(ABC):
    """Abstract base class for running and populating Skip logic."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        device: str | None = None,
        checkpoints: list[int] = None,
    ):
        self.device = get_device() if device is None else device
        logging.info(f"Loading model '{model_name}' on {self.device}...")

        self.model_name = model_name
        self.model: Model = self._load_model()

        # Enforce batching pad semantics universally
        if self.model.tokenizer.pad_token_id is None:
            self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id
        self.model.tokenizer.padding_side = "left"

        n_layers = self.model.n_layers
        if checkpoints is None:
            self.checkpoints = list(range(0, n_layers, 4))
        else:
            self.checkpoints = sorted([c for c in checkpoints if c < n_layers])

        logging.info(
            f"Initialised with {len(self.checkpoints)} checkpoints: {self.checkpoints}"
        )

    @abstractmethod
    def _load_model(self) -> Model:
        pass

    @abstractmethod
    def _get_early_exit_logits(self, state: torch.Tensor) -> torch.Tensor:
        pass

    def format_prompt(self, prompt: PromptType) -> str:
        """
        Helper to handle:
         Chat Template formatting (adding <|im_start|>, system prompts, etc.)

        Returns:
            The formatted prompt string ready for tokenisation.
        """
        tokenizer = self.model.tokenizer
        if isinstance(prompt, DatasetSample):
            prompt = prompt.prompt

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

        formatted_prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted_prompt_str

    @abstractmethod
    def generate_and_populate(
        self,
        prompt: PromptType,
        vector_db: SkippingVectorDB,
        *,
        early_exit_strategy_mode: EarlyExitStrategyMode | None = None,
        skip_strategy_mode: SkipStrategyMode | None = None,
        similarity_threshold: float = 0.95,
        max_new_tokens: int = 20,
    ) -> str:
        """
        Runs full inference to find ground truth.
        Then checks every checkpoint to see if we could have Exited or Skipped.
        Populates the DB with positive examples.

        - DB should have matching number of checkpoints.
        - Uses the early_exit_strategy_mode when deciding whether we should
            populate the vector db with an early exit decision.
        - Uses the skip_strategy_mode to determine whether to use cosine similarity
        or strict match for skip decisions, which this method implements.
        - Returns the final completed text after generation (prompt + generated).
        """
        pass

    @abstractmethod
    def generate_and_populate_batched(
        self,
        prompts: list[PromptType],
        vector_db: SkippingVectorDB,
        *,
        early_exit_strategy_mode: EarlyExitStrategyMode | None = None,
        skip_strategy_mode: SkipStrategyMode | None = None,
        similarity_threshold: float = 0.95,
        total_final_tokens: int = 512,
    ) -> list[str]:
        """
        Batched version of generate_and_populate.
        """
        pass

    @abstractmethod
    def generate_with_skipping(
        self,
        prompt: PromptType,
        vector_db: SkippingVectorDB | None = None,
        threshold: float | dict[int, float] = DEFAULT_THRESH,
        max_total_tokens: int = 20,
        format_prompt: bool = True,
        **kwargs,
    ) -> SkipGenerationResult:
        """
        Runs inference with skipping enabled.
        If a decision is to skip or early-exit, this method simulates this skipping.
        Args:
            - prompt: The input prompt to generate from.
                    Can be a string, list of messages, or DatasetSample.
            - vector_db: The DB to query for skip decisions.
                    If None, runs with no skipping.
            - threshold: Similarity threshold(s) for deciding whether to apply a skip.
                    Threshold can be a single float or a dict {checkpoint_idx: float}.
            - max_new_tokens: The maximum number of tokens to generate.
            - format_prompt: Whether to apply chat template formatting to the prompt.

        Returns the final completed text after generation (prompt + generated).
        """
        pass

    @abstractmethod
    def simulate_decision(
        self,
        tokens: torch.Tensor,
        checkpoint_idx: int,
        current_state: torch.Tensor,
        decision: SkipDecision,
    ) -> int:
        """
        Simulates the outcome of a SkipDecision (EXIT or SKIP) given the current state
            and tokens up to this point.
        Returns the predicted token ID resulting from this decision.
        Args:
            - tokens: The input tokens up to the current point
            - checkpoint_idx: The index of the current checkpoint in self.checkpoints
            - current_state: The hidden state at the checkpoint
            - decision: The SkipDecision to simulate
        """
        pass
