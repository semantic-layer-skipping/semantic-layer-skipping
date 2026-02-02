import logging

import torch
import torch.nn.functional as functional
from inference.strategies import (
    EarlyExitStrategy,
    SkipStrategyMode,
    StrictMatchStrategy,
)
from store import SkippingVectorDB
from structures import Action, SkipDecision
from transformer_lens import HookedTransformer
from utils import ISAAC_NEWTON_QUESTIONS, get_device, question_to_prompt


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

        # default: every 4 layers
        if checkpoints is None:
            self.checkpoints = list(range(0, n_layers, 4))
        else:
            self.checkpoints = sorted([c for c in checkpoints if c < n_layers])

        logging.info(
            f"Initialised with {len(self.checkpoints)} checkpoints: {self.checkpoints}"
        )

    def _get_early_exit_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Helper to project a hidden state to vocab logits using the model head."""
        # apply final norm
        normalised = self.model.ln_final(state)
        # apply unembedding
        return self.model.unembed(normalised)

    def generate_and_populate(
        self,
        prompt: str,
        vector_db: SkippingVectorDB,
        *,
        early_exit_strategy: EarlyExitStrategy | None = None,
        skip_strategy_mode: SkipStrategyMode | None = SkipStrategyMode.COSINE,
        similarity_threshold: float = 0.95,
        max_new_tokens: int = 20,
    ) -> str:
        """
        Runs full inference to find ground truth.
        Then checks every checkpoint to see if we could have Exited or Skipped.
        Populates the DB with positive examples.

        DB should have number of layers equal to the number of decision points.
        Uses the early_exit_strategy when deciding whether we should
            populate the vector db with an early exit decision.
        Uses the skip_strategy_mode to determine whether to use cosine similarity
        or strict match for skip decisions, which this method implements.
        Returns the final prediction.
        """

        logging.info(f"Generating & Populating for input prompt: '{prompt}'")

        tokens = self.model.to_tokens(prompt)

        for _ in range(max_new_tokens):
            # run a single step to get next token and populate DB
            next_token_id = self._populate_step(
                tokens,
                vector_db,
                early_exit_strategy,
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
        early_exit_strategy: EarlyExitStrategy | None,
        skip_strategy_mode: SkipStrategyMode,
        similarity_threshold: float,
    ) -> int:
        """
        Runs a single forward pass on 'tokens', checks for skips/exits, populates DB,
        and returns the next predicted token ID.
        """
        # run full model to get ground truth
        original_logits, full_cache = self.model.run_with_cache(
            tokens, return_type="logits"
        )

        # final token logits for early exit
        target_final_logits = original_logits[0, -1, :]
        target_token_id = torch.argmax(target_final_logits).item()

        # iterate over checkpoints
        for checkpoint_idx, layer_idx in enumerate(self.checkpoints):
            # get the hidden state at this checkpoint's input
            # (Hidden_Dim)
            current_state = full_cache[f"blocks.{layer_idx}.hook_resid_pre"][0, -1, :]

            # early exit
            if early_exit_strategy:
                early_logits = self._get_early_exit_logits(current_state)
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
                continue  # no skipping strategy defined

            # goal is to find the furthest valid skip - breaks once found
            for future_checkpoint_idx in range(
                len(self.checkpoints) - 1, checkpoint_idx, -1
            ):
                target_layer_idx = self.checkpoints[future_checkpoint_idx]
                is_valid_skip = False
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
                    # run simulation: inject current_state into input target_layer_idx
                    def injection_hook(resid_pre, hook, state=current_state):
                        resid_pre[:, -1, :] = state
                        return resid_pre

                    with torch.no_grad():
                        # re-run from the skip point
                        # note: this re-runs the whole sequence with the hook
                        sim_logits = self.model.run_with_hooks(
                            tokens,
                            fwd_hooks=[
                                (
                                    f"blocks.{target_layer_idx}.hook_resid_pre",
                                    injection_hook,
                                )
                            ],
                        )
                    sim_token_id = torch.argmax(sim_logits[0, -1, :]).item()
                    if sim_token_id == target_token_id:
                        is_valid_skip = True

                if is_valid_skip:
                    # determine how many checkpoints we are skipping
                    skip_n_checkpoints = future_checkpoint_idx - checkpoint_idx
                    decision = SkipDecision(
                        action=Action.SKIP, skip_count=skip_n_checkpoints
                    )
                    vector_db.add_vector(
                        checkpoint_idx,
                        current_state.detach().cpu().numpy().reshape(1, -1),
                        decision,
                    )
                    break  # stop after finding max skip for this checkpoint

        return target_token_id

    def generate_with_skipping(
        self,
        prompt: str,
        vector_db: SkippingVectorDB,
        threshold: float = 0.7,
        max_new_tokens: int = 20,
    ) -> str:
        """
        Runs inference, querying the DB at every checkpoint to perform actions.
        If a decision is to skip or early-exit, this method simulates this skipping.
        Threshold is the cosine similarity threshold for deciding whether to skip:
            if the nearest neighbor's similarity is above this threshold,
            we take its decision.
        Returns the final prediction.
        """
        logging.info(f"Generating with Skipping for input prompt: '{prompt}'")
        tokens = self.model.to_tokens(prompt)

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

            # 1. landing logic: we have arrived at the target layer after skipping
            if ctx.skipping_active:
                if layer_idx == ctx.landing_layer:
                    logging.info(f"  [L{layer_idx}] Skip LANDED - Injecting state.")
                    resid_pre[:, -1, :] = ctx.teleport_vector
                    ctx.skipping_active = False
                    ctx.landing_layer = -1
                    ctx.teleport_vector = None
                else:
                    # still skipping, do nothing
                    pass
                return resid_pre

            # 2. decision logic, run if we are not skipping

            # query db at this layer
            checkpoint_idx = self.checkpoints.index(layer_idx)
            query_vec = resid_pre[0, -1, :].detach().cpu().numpy().reshape(1, -1)
            result = vector_db.search(checkpoint_idx, query_vec)

            if result:
                logging.info(
                    f"  [L{layer_idx}] Retrieved from DB {result}, "
                    f"Threshold: {threshold:.2f}]"
                )

                if result.similarity < threshold:
                    return resid_pre

                elif result.decision.action == Action.EXIT:
                    logging.info(f"  [L{layer_idx}]  EARLY EXIT.")
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
        for layer_idx in self.checkpoints:
            fwd_hooks.append((f"blocks.{layer_idx}.hook_resid_pre", checkpoint_hook))

        # generation loop
        for i in range(max_new_tokens):
            # reset context flags for the new token pass
            ctx.skipping_active = False
            ctx.landing_layer = -1
            ctx.teleport_vector = None

            try:
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

        pred_str = self.model.to_string(tokens[0])
        logging.info(
            f"Final Generated String: '{pred_str}'\n"
            f"Total Skipped Layers: {ctx.skipped_layers_count}. "
            f"Total Generated Tokens: {i + 1}."
        )
        return pred_str


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = "Qwen/Qwen2.5-1.5B-Instruct"  # has 28 layers
    # every 4 layers - last layer is not a checkpoint because we have early exit at 34
    checkpoints = list(range(4, 28, 4))
    runner = SemanticSkipRunner(model_name=model, checkpoints=checkpoints)
    vector_db = SkippingVectorDB(
        n_checkpoints=len(checkpoints), vector_dim=runner.model.cfg.d_model
    )

    num_test = 1
    num_calibrate = 1
    assert num_calibrate + num_test <= len(ISAAC_NEWTON_QUESTIONS)
    calibration_questions = ISAAC_NEWTON_QUESTIONS[:num_test]
    test_questions = ISAAC_NEWTON_QUESTIONS[-num_test:]
    # test_questions = [PROMPTS[-1]]

    for question in calibration_questions:
        prompt = question_to_prompt(question)
        # exit_strategy = KLDivergenceStrategy(threshold=2.0)
        exit_strategy = StrictMatchStrategy()
        final_token = runner.generate_and_populate(
            prompt,
            vector_db,
            early_exit_strategy=exit_strategy,
            skip_strategy_mode=SkipStrategyMode.STRICT,
            # similarity_threshold=0.95 # only used for COSINE mode
        )

    # now run inference with skipping
    for question in test_questions:
        prompt = question_to_prompt(question)
        # prompt += "the"
        predicted_token = runner.generate_with_skipping(
            prompt, vector_db, threshold=0.9
        )
