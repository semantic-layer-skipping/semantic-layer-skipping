import logging

import torch
import torch.nn.functional as functional
from inference.strategies import (
    EarlyExitStrategyMode,
    SkipStrategyMode,
    get_early_exit_strategy,
)
from store import SkippingVectorDB
from structures import Action, DatasetSample, SkipDecision, SkipGenerationResult
from transformer_lens import HookedTransformer
from utils import ISAAC_NEWTON_QUESTIONS, get_device

# defines promptlike type
PromptType = str | list[dict] | DatasetSample

# default cosine similarity threshold for whether a skip is valid
DEFAULT_THRESH = 0.7


# class to signal early exit during inference with skipping
class EarlyExitSignal(Exception):  # noqa: N818
    def __init__(self, final_logits):
        self.final_logits = final_logits


class SemanticSkipRunner:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        device: str | None = None,
        checkpoints: list[int] = None,
    ):
        """
        Initialises the model and puts it in evaluation mode.
        Args:
            checkpoints (list[int]): Points (layer indices)
                                    where skip decisions are evaluated.
                                     Default is every 4th layer.
        """
        self.device = get_device() if device is None else device
        logging.info(f"Loading model '{model_name}' on {self.device}...")

        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        self.model.eval()

        n_layers = self.model.cfg.n_layers

        if checkpoints is None:
            # default: every 4 layers
            self.checkpoints = list(range(0, n_layers, 4))
        else:
            self.checkpoints = sorted([c for c in checkpoints if c < n_layers])

        logging.info(
            f"Initialised with {len(self.checkpoints)} checkpoints: {self.checkpoints}"
        )

    def _prepare_input(self, prompt: PromptType) -> tuple[torch.Tensor, int]:
        """
        Internal helper to handle:
        1. Chat Template formatting (adding <|im_start|>, system prompts, etc.)
        2. Tokenisation
        3. Device placement

        Returns:
            input_tokens (torch.Tensor): The tokenised input on the correct device.
        """
        tokenizer = self.model.tokenizer

        # normalise prompt into list of messages format expected by chat template
        if isinstance(prompt, str):
            # map single string prompt to chat format with one user message
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        elif isinstance(prompt, DatasetSample):
            if isinstance(prompt.prompt, str):
                messages = [{"role": "user", "content": prompt.prompt}]
            else:
                messages = prompt.prompt
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

        # apply chat template (this adds the <|im_start|> tags)
        formatted_prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        logging.debug(f"Formatted Prompt: {formatted_prompt_str}")

        # tokenise
        input_tokens = tokenizer(
            formatted_prompt_str,
            return_tensors="pt",
            add_special_tokens=False,  # template should've already added them
        ).input_ids

        input_tokens = input_tokens.to(self.device)
        return input_tokens

    def _get_early_exit_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Helper to project a hidden state to vocab logits using the model head."""
        # apply final norm and unembedding to get logits
        normalised = self.model.ln_final(state)
        return self.model.unembed(normalised)

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
        if decision.action == Action.EXIT:
            early_logits = self._get_early_exit_logits(current_state)
            # TODO: we could also return the full distribution here
            return torch.argmax(early_logits).item()

        elif decision.action == Action.SKIP:
            # calculate target layer
            target_ckpt_idx = checkpoint_idx + decision.skip_count
            assert target_ckpt_idx < len(self.checkpoints), (
                f"Invalid Skip Decision: checkpoint_idx={checkpoint_idx}, "
                f"skip_count={decision.skip_count}, "
                f"which leads to target_ckpt_idx={target_ckpt_idx} "
                f"exceeding available checkpoints={len(self.checkpoints)}"
            )
            target_layer_idx = self.checkpoints[target_ckpt_idx]

            def injection_hook(resid_pre, hook):
                resid_pre[:, -1, :] = current_state
                return resid_pre

            # TODO: this simulation only considers 1 skip event - we could extend
            #  this to simulate multiple sequential skips by stacking hooks.
            with torch.no_grad():
                # run with hooks to simulate the skip
                sim_logits = self.model.run_with_hooks(
                    tokens,
                    fwd_hooks=[
                        (f"blocks.{target_layer_idx}.hook_resid_pre", injection_hook)
                    ],
                )
            return torch.argmax(sim_logits[0, -1, :]).item()
        else:
            raise ValueError(f"Unknown action in SkipDecision: {decision.action}")

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
        logging.info(f"Generating & Populating for input prompt: '{prompt}'")

        tokens = self._prepare_input(prompt)

        for _ in range(max_new_tokens):
            # run a single step to get next token and populate DB
            next_token_id = self._populate_step(
                tokens,
                vector_db,
                early_exit_strategy_mode,
                skip_strategy_mode,
                similarity_threshold,
            )
            if next_token_id == self.model.tokenizer.eos_token_id:
                logging.info("  [Generation] Reached EOS token, stopping generation.")
                break

            # append new token to input for next iteration
            next_token_tensor = torch.tensor(data=[[next_token_id]], device=self.device)
            tokens = torch.cat([tokens, next_token_tensor], dim=1)

        full_text = self.model.to_string(tokens[0])
        logging.info(f"Full generated text: '{full_text}'")
        return full_text

    def _populate_step(
        self,
        tokens: torch.Tensor,
        vector_db: SkippingVectorDB,
        early_exit_strategy_mode: EarlyExitStrategyMode | None,
        skip_strategy_mode: SkipStrategyMode,
        similarity_threshold: float,
    ) -> int:
        """
        Runs a single forward pass, checks for skips/exits, populates DB.
        Returns the next predicted token ID.
        """
        # run full model to get ground truth
        original_logits, full_cache = self.model.run_with_cache(
            tokens, return_type="logits"
        )
        target_final_logits = original_logits[0, -1, :]
        target_token_id = torch.argmax(target_final_logits).item()

        # iterate over checkpoints
        for checkpoint_idx, layer_idx in enumerate(self.checkpoints):
            # get the hidden state at this checkpoint's input
            current_state = full_cache[f"blocks.{layer_idx}.hook_resid_pre"][0, -1, :]

            # early exit
            if early_exit_strategy_mode:
                early_logits = self._get_early_exit_logits(current_state)
                early_exit_strategy = get_early_exit_strategy(early_exit_strategy_mode)
                if early_exit_strategy.should_exit(early_logits, target_final_logits):
                    decision = SkipDecision(action=Action.EXIT)
                    vector_db.add_vector(
                        checkpoint_idx,
                        # move to cpu and detach for numpy
                        current_state.detach().cpu().numpy().reshape(1, -1),
                        decision,
                    )
                    continue  # if we exit, don't consider skipping

            # layer skipping
            if skip_strategy_mode is None:
                continue

            # goal is to find the furthest valid skip - breaks once found
            for future_checkpoint_idx in range(
                len(self.checkpoints) - 1, checkpoint_idx, -1
            ):
                target_layer_idx = self.checkpoints[future_checkpoint_idx]
                is_valid_skip = False
                decision_to_test = SkipDecision(
                    action=Action.SKIP,
                    skip_count=future_checkpoint_idx - checkpoint_idx,
                )
                if skip_strategy_mode == SkipStrategyMode.COSINE:
                    # compare Input(Current) vs Input(Target)
                    # this means we can skip to Input(Target)
                    target_state = full_cache[
                        f"blocks.{target_layer_idx}.hook_resid_pre"
                    ][0, -1, :]
                    sim = functional.cosine_similarity(
                        current_state, target_state, dim=0
                    ).item()
                    if sim >= similarity_threshold:
                        is_valid_skip = True

                elif skip_strategy_mode == SkipStrategyMode.STRICT:
                    sim_token_id = self.simulate_decision(
                        tokens, checkpoint_idx, current_state, decision_to_test
                    )
                    if sim_token_id == target_token_id:
                        is_valid_skip = True

                if is_valid_skip:
                    # the decision is valid, so add to DB
                    # and break - we only want to add the furthest valid skip
                    vector_db.add_vector(
                        checkpoint_idx,
                        current_state.detach().cpu().numpy().reshape(1, -1),
                        decision_to_test,
                    )
                    break

        return target_token_id

    def generate_with_skipping(
        self,
        prompt: PromptType,
        vector_db: SkippingVectorDB | None = None,
        threshold: float | dict[int, float] = DEFAULT_THRESH,
        max_new_tokens: int = 20,
    ) -> SkipGenerationResult:
        """
        Runs inference with skipping enabled.
        If a decision is to skip or early-exit, this method simulates this skipping.
        Threshold can be a single float or a dict {checkpoint_idx: float}.
        If vector_db is None, will run without any skipping (for ablation).
        Returns the final completed text after generation (prompt + generated).
        """
        logging.info(f"Generating with Skipping for input prompt: '{prompt}'")
        input_tokens_tensor = self._prepare_input(prompt)
        input_length = input_tokens_tensor.shape[1]

        # we will keep appending to this tensor as we generate new tokens
        tokens = input_tokens_tensor.clone()

        # context for shared state during inference
        class SkipCtx:
            def __init__(self):
                self.skipping_active = False
                self.landing_layer = -1
                self.teleport_vector = None
                self.skipped_layers_count = 0

        ctx = SkipCtx()

        # single hook, added at each checkpoint (layer input)
        def checkpoint_hook(resid_pre, hook):
            layer_idx = hook.layer()

            # 1. landing logic: check if we have arrived at the target layer
            if ctx.skipping_active:
                if layer_idx == ctx.landing_layer:
                    logging.info(f"  [L{layer_idx}] Skip LANDED - Injecting state.")
                    resid_pre[:, -1, :] = ctx.teleport_vector
                    ctx.skipping_active = False
                    ctx.landing_layer = -1
                    ctx.teleport_vector = None
                else:
                    pass  # still skipping, do nothing
                return resid_pre

            # 2. decision logic, run if we are not skipping
            checkpoint_idx = self.checkpoints.index(layer_idx)
            query_vec = resid_pre[0, -1, :].detach().cpu().numpy().reshape(1, -1)
            result = vector_db.search(checkpoint_idx, query_vec)

            if result:
                # get threshold for this checkpoint
                if isinstance(threshold, dict):
                    local_thresh = threshold.get(checkpoint_idx, DEFAULT_THRESH)
                else:
                    local_thresh = threshold

                if result.similarity < local_thresh:
                    return resid_pre

                logging.info(
                    f"  [L{layer_idx}] Retrieved decision from DB: {result}, "
                    f"(threshold: {local_thresh:.4f})"
                )

                if result.decision.action == Action.EXIT:
                    logging.info(f"  [L{layer_idx}] EARLY EXIT triggered.")
                    final_logits = self._get_early_exit_logits(resid_pre[0, -1, :])
                    raise EarlyExitSignal(final_logits)

                elif result.decision.action == Action.SKIP:
                    current_ckpt_idx = checkpoint_idx
                    target_ckpt_idx = current_ckpt_idx + result.decision.skip_count

                    if target_ckpt_idx < len(self.checkpoints):
                        target_layer = self.checkpoints[target_ckpt_idx]

                        logging.info(
                            f"  [L{layer_idx}]  SKIPPING to L{target_layer}"
                            f" (Checkpoint {target_ckpt_idx})."
                        )

                        ctx.skipping_active = True
                        ctx.landing_layer = target_layer
                        # current input state is the teleported vector
                        ctx.teleport_vector = resid_pre[:, -1, :].clone()

                        # count number of skipped layers
                        ctx.skipped_layers_count += target_layer - layer_idx

            return resid_pre

        # add hooks at all checkpoints
        fwd_hooks = []
        if vector_db is not None:
            for layer_idx in self.checkpoints:
                fwd_hooks.append(
                    (f"blocks.{layer_idx}.hook_resid_pre", checkpoint_hook)
                )

        # generation loop
        for i in range(max_new_tokens):
            # reset context flags for the new token pass
            ctx.skipping_active = False
            ctx.landing_layer = -1
            ctx.teleport_vector = None

            try:
                # if hooks list is empty (vector_db=None), this runs normally
                logits = self.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
                final_logits = logits[0, -1, :]
            except EarlyExitSignal as e:
                final_logits = e.final_logits

            # greedy decode
            next_token_id = torch.argmax(final_logits).item()
            if next_token_id == self.model.tokenizer.eos_token_id:
                logging.info("  [Generation] Reached EOS token, stopping generation.")
                break

            next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
            tokens = torch.cat([tokens, next_token_tensor], dim=1)
            logging.info(
                f"Generated token ({i}): '{self.model.to_string(next_token_id)}'"
            )

        full_text = self.model.to_string(tokens[0])

        # extract just the new tokens
        generated_tokens_tensor = tokens[0, input_length:]
        generated_tokens = generated_tokens_tensor.tolist()
        generated_text = self.model.to_string(generated_tokens_tensor)
        prompt_tokens = input_tokens_tensor[0].tolist()

        logging.info(
            f"Final Generated String: '{full_text}'\n"
            f"Total Skipped Layers: {ctx.skipped_layers_count}. "
            f"Total Generated Tokens: {len(generated_tokens)}."
        )
        return SkipGenerationResult(
            full_text=full_text,
            generated_text=generated_text,
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_tokens,
            generated_token_count=len(generated_tokens),
            skipped_layers=ctx.skipped_layers_count,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = "Qwen/Qwen2.5-1.5B-Instruct"  # has 28 layers
    # every 4 layers - last layer is not a checkpoint because we have early exit at 34
    checkpoints = list(range(4, 28, 4))
    runner = SemanticSkipRunner(model_name=model, checkpoints=checkpoints)
    vector_db = SkippingVectorDB(
        n_checkpoints=len(checkpoints), vector_dim=runner.model.cfg.d_model
    )

    num_populate = 7
    num_test = 3
    assert num_test + num_populate <= len(ISAAC_NEWTON_QUESTIONS)
    population_questions = ISAAC_NEWTON_QUESTIONS[:num_populate]
    test_questions = ISAAC_NEWTON_QUESTIONS[-num_test:]
    # test_questions = [PROMPTS[-1]]

    for question in population_questions:
        early_exit_strategy_mode = EarlyExitStrategyMode.STRICT_MATCH
        final_token = runner.generate_and_populate(
            question,
            vector_db,
            early_exit_strategy_mode=early_exit_strategy_mode,
            skip_strategy_mode=SkipStrategyMode.STRICT,
            # similarity_threshold=0.95 # only used for COSINE mode
        )

    # now run inference with skipping
    for question in test_questions:
        runner.generate_with_skipping(question, vector_db, threshold=0.9)
