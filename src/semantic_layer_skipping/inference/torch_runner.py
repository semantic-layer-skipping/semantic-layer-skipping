import logging

import torch
import torch.nn.functional as functional
from inference.base_runner import (
    DEFAULT_THRESH,
    EarlyExitSignal,
    PromptType,
    SemanticSkipRunner,
    SkipCtx,
)
from inference.model import Model, TorchModel
from inference.strategies import (
    EarlyExitStrategyMode,
    SkipStrategyMode,
    get_early_exit_strategy,
)
from store import SkippingVectorDB
from structures import Action, SkipDecision, SkipGenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


class ReadOnlyCache(DynamicCache):
    """
    Inherits from HF's DynamicCache to perfectly preserve all metadata.
    Overrides update() during the forward pass to prevent cache mutation.
    """

    def initial_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # use standard update logic for cache initialisation
        super().update(key_states, value_states, layer_idx, cache_kwargs=cache_kwargs)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # get past states
        past_k, past_v = self[layer_idx]

        # perform concatenation for the new states
        # but do NOT update the internal cache, just return the concatenated result
        k_out = torch.cat([past_k, key_states], dim=-2)
        v_out = torch.cat([past_v, value_states], dim=-2)

        return k_out, v_out


class TorchSkipRunner(SemanticSkipRunner):
    def _load_model(self) -> Model:
        inner = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=self.device, torch_dtype=torch.float16
        )
        inner.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return TorchModel(inner, tokenizer)

    def _get_early_exit_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Projects intermediate hidden states directly to vocab space."""
        normed_state = self.model.inner.model.norm(hidden_state)
        return self.model.inner.lm_head(normed_state)

    def simulate_decision(
        self,
        tokens: torch.Tensor,
        checkpoint_idx: int,
        current_state: torch.Tensor,
        decision: SkipDecision,
    ) -> int:
        raise NotImplementedError("Not yet implemented!")

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
        raise NotImplementedError(
            "Not implemented! Use generate_and_populate_batched instead."
        )

    def generate_and_populate_batched(
        self,
        prompts: list[PromptType],
        vector_db: SkippingVectorDB,
        *,
        early_exit_strategy_mode: EarlyExitStrategyMode | None = None,
        skip_strategy_mode: SkipStrategyMode | None = None,
        similarity_threshold: float = 0.95,
        total_final_tokens: int = 1024,
    ) -> list[str]:
        logging.info(
            f"Batched populating {len(prompts)} prompts. "
            f"Target total: {total_final_tokens} tokens."
        )

        formatted_prompts = [self.format_prompt(p) for p in prompts]

        # log inputs
        for idx, p in enumerate(formatted_prompts):
            logging.info(
                f"\n{'=' * 20} FORMATTED INPUT PROMPT {idx} {'=' * 20}\n{p}\n{'=' * 54}"
            )

        inputs = self.model.tokenizer(
            formatted_prompts, return_tensors="pt", padding=True
        ).to(self.device)

        prompt_tokens = inputs.input_ids
        attention_mask = inputs.attention_mask
        batch_size = prompt_tokens.shape[0]

        prompt_lengths = attention_mask.sum(dim=-1)
        logging.info(
            f"  Initial token lengths for prompts in batch: {prompt_lengths.tolist()}"
        )

        if (prompt_lengths >= total_final_tokens).all():
            logging.warning("All prompts meet or exceed total_final_tokens. Skipping.")
            return [
                self.model.tokenizer.decode(t[m == 1])
                for t, m in zip(prompt_tokens, attention_mask, strict=True)
            ]

        # phase 1: generate tokens for all inputs in the batch
        with torch.no_grad():
            full_sequence_tokens = self.model.inner.generate(
                prompt_tokens,
                attention_mask=attention_mask,
                max_length=total_final_tokens,
                do_sample=False,  # greedy decoding
                pad_token_id=self.model.tokenizer.pad_token_id,
            )

        # log full generated outputs
        for idx in range(batch_size):
            p_len = prompt_lengths[idx]
            gen_only_tokens = full_sequence_tokens[idx, p_len:]
            gen_text = self.model.tokenizer.decode(
                gen_only_tokens, skip_special_tokens=True
            )
            logging.info(
                f"\n{'=' * 20} FULL GENERATED TEXT {idx} {'=' * 20}\n{gen_text}\n"
                f"{'=' * 63}"
            )

        # phase 2: get the cache and hidden states for all tokens (except the last one)
        # this is a parallelised pass to extract what we need
        tokens_to_process = full_sequence_tokens[:, :-1]
        full_attention_mask = (
            tokens_to_process != self.model.tokenizer.pad_token_id
        ).long()

        with torch.no_grad():
            gt_outputs = self.model.inner(
                tokens_to_process,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )

        hidden_states = gt_outputs.hidden_states
        original_logits = gt_outputs.logits
        past_key_values = gt_outputs.past_key_values

        logging.info("  [Generation] Phase 1 and 2 complete: hidden states extracted.")

        # phase 3: simulation loop
        prompt_len = prompt_tokens.shape[1]
        seq_len = full_sequence_tokens.shape[1]
        total_gen_steps = seq_len - prompt_len

        early_exit_strategy = None
        if early_exit_strategy_mode:
            early_exit_strategy = get_early_exit_strategy(early_exit_strategy_mode)

        # dummy forward function to skip redundant layers
        def get_dummy_forward():
            def dummy(hidden_states, *args, **kwargs):
                return (hidden_states,)

            return dummy

        # we define a factory function to create hooks for injecting state
        # PyTorch's garbage collection can be tricky with closures,
        # so we use this factory to ensure the correct state is captured and cleaned
        def make_injection_hook(states_to_inject):
            def injection_hook(module, args, kwargs):
                new_args = (states_to_inject.unsqueeze(1),) + args[1:]
                return new_args, kwargs

            return injection_hook

        for step in range(prompt_len - 1, seq_len - 1):
            target_tokens = full_sequence_tokens[:, step + 1]
            target_final_logits = original_logits[:, step, :]

            # identify which sequences in the batch are still active (still generating)
            active_batch_mask = target_tokens != self.model.tokenizer.pad_token_id
            # set active batch mask to be true for all
            if not active_batch_mask.any():
                # technically, this should never execute: phase 1 finds max length
                continue

            # log phase 3 step
            step_num = step - (prompt_len - 1) + 1
            # decode target tokens for logging
            target_tokens_str = self.model.tokenizer.batch_decode(target_tokens)
            logging.info(
                f"  [Phase 3] Step {step_num}/{total_gen_steps} | Target tokens: "
                f"{target_tokens_str}"
            )

            # form the KV cache for this simulated generation
            # which contains all KVs up to current step
            # we use our custom class, but initialise it using native HF logic
            # the custom class ensures later updates don't mutate the cache
            sim_cache = ReadOnlyCache()
            for l_idx in range(len(self.model.inner.model.layers)):
                k, v = past_key_values[l_idx]
                sim_cache.initial_update(k[:, :, :step, :], v[:, :, :step, :], l_idx)

            # mask covers cache (length `step`) + the new token (length 1) = `step + 1`
            sim_attn_mask = full_attention_mask[:, : step + 1]

            # iterate through checkpoints and simulate decisions
            for i, layer_idx in enumerate(self.checkpoints):
                current_states = hidden_states[layer_idx][:, step, :]

                # early exit
                if early_exit_strategy:
                    early_logits = self._get_early_exit_logits(current_states)
                    # TODO: the strategy could be batched as well
                    for b in range(batch_size):
                        if active_batch_mask[b]:
                            if early_exit_strategy.should_exit(
                                early_logits[b], target_final_logits[b]
                            ):
                                vector_db.add_vector(
                                    i,
                                    current_states[b]
                                    .to(torch.float32)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .reshape(1, -1),
                                    SkipDecision(action=Action.EXIT),
                                )

                # layer skipping
                if skip_strategy_mode is None:
                    continue

                # indicates whether we already found the longest possible skip for this
                found_skip_mask = torch.zeros(
                    batch_size, dtype=torch.bool, device=self.device
                )

                # start evaluating the largest skips first
                for j in range(len(self.checkpoints) - 1, i, -1):
                    # we will evaluate whether
                    # we can skip from checkpoint i to checkpoint j

                    target_layer_idx = self.checkpoints[j]
                    skip_count = j - i

                    # only evaluate those still active and haven't found a skip yet
                    eval_mask = (~found_skip_mask) & active_batch_mask
                    if not eval_mask.any():
                        break

                    valid_skips = torch.zeros(
                        batch_size, dtype=torch.bool, device=self.device
                    )

                    if skip_strategy_mode == SkipStrategyMode.COSINE:
                        target_states = hidden_states[target_layer_idx][:, step, :]
                        sims = functional.cosine_similarity(
                            current_states, target_states, dim=-1
                        )
                        valid_skips = (sims >= similarity_threshold) & eval_mask

                    elif skip_strategy_mode == SkipStrategyMode.STRICT:
                        # we run strict simulation from target layer with injected state
                        # up to the end.
                        # 1. physically bypass all layers up to the target
                        original_forwards = {}
                        for layer_index in range(target_layer_idx):
                            layer_module = self.model.inner.model.layers[layer_index]
                            original_forwards[layer_index] = layer_module.forward
                            layer_module.forward = get_dummy_forward()

                        # 2. register a hook at the target layer to inject our state
                        handle = self.model.inner.model.layers[
                            target_layer_idx
                        ].register_forward_pre_hook(
                            make_injection_hook(current_states), with_kwargs=True
                        )
                        # dummy tokens, which will be overwritten by the injected state
                        dummy_tokens = full_sequence_tokens[:, step : step + 1]

                        # 3. run forward pass with dummy tokens and injected state,
                        # using the pre-populated KV cache
                        try:
                            with torch.no_grad():
                                sim_outputs = self.model.inner(
                                    dummy_tokens,
                                    attention_mask=sim_attn_mask,
                                    # pass the previously-computed KV cache
                                    past_key_values=sim_cache,
                                    # ensures huggingface doesnt' return cache
                                    # however, it still internally updates the cache,
                                    # which is why we use our custom ReadOnlyCache
                                    use_cache=False,
                                )
                            sim_preds = torch.argmax(
                                sim_outputs.logits[:, -1, :], dim=-1
                            )
                            valid_skips = (sim_preds == target_tokens) & eval_mask
                        finally:
                            handle.remove()
                            # restore original forwards
                            for layer_index, original_fwd in original_forwards.items():
                                self.model.inner.model.layers[
                                    layer_index
                                ].forward = original_fwd

                    # add successful furthest skips to the DB
                    for b in range(batch_size):
                        # TODO: could batch this DB insertion
                        if valid_skips[b]:
                            vector_db.add_vector(
                                i,
                                current_states[b]
                                .to(torch.float32)
                                .detach()
                                .cpu()
                                .numpy()
                                .reshape(1, -1),
                                SkipDecision(action=Action.SKIP, skip_count=skip_count),
                            )

                    found_skip_mask |= valid_skips

        full_texts = [
            self.model.tokenizer.decode(
                t[t != self.model.tokenizer.pad_token_id], skip_special_tokens=True
            )
            for t in full_sequence_tokens
        ]

        del hidden_states, past_key_values, original_logits
        torch.cuda.empty_cache()

        logging.info("  [Generation] Phase 3 Complete. Finished batched population.")
        return full_texts

    def generate_with_skipping(
        self,
        prompt: PromptType,
        vector_db: SkippingVectorDB | None = None,
        threshold: float | dict[int, float] = DEFAULT_THRESH,
        max_new_tokens: int = 20,
        format_prompt: bool = True,
    ) -> SkipGenerationResult:
        logging.info(f"Generating with Skipping for input prompt: '{prompt}'")
        if format_prompt:
            prompt = self.format_prompt(prompt)

        inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        input_length = input_ids.shape[1]

        ctx = SkipCtx()
        handles = []

        # hook added at each checkpoint
        def make_eval_hook(layer_idx):
            def hook(module, args, kwargs):
                hidden_state = args[0]

                # 1. landing logic: check if we have arrived at the target layer
                if ctx.skipping_active:
                    if layer_idx == ctx.landing_layer:
                        logging.info(f"  [L{layer_idx}] Skip LANDED - Injecting state.")
                        # overwrite the latest token (-1:)
                        hidden_state[:, -1:, :] = ctx.teleport_vector
                        ctx.skipping_active = False
                        ctx.landing_layer = -1
                        ctx.teleport_vector = None
                    else:
                        pass  # still skipping, do nothing
                    new_args = (hidden_state,) + args[1:]
                    return new_args, kwargs

                # 2. decision logic, run if we are not skipping
                checkpoint_idx = self.checkpoints.index(layer_idx)
                query_vec = (
                    hidden_state[0, -1, :]
                    .to(torch.float32)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(1, -1)
                )

                if vector_db is not None:
                    result = vector_db.search(checkpoint_idx, query_vec)

                    if result:
                        # get threshold for this checkpoint
                        if isinstance(threshold, dict):
                            local_thresh = threshold.get(checkpoint_idx, DEFAULT_THRESH)
                        else:
                            local_thresh = threshold

                        if result.similarity < local_thresh:
                            new_args = (hidden_state,) + args[1:]
                            return new_args, kwargs

                        logging.info(
                            f"  [L{layer_idx}] Retrieved decision from DB: {result}, "
                            f"(threshold: {local_thresh:.4f})"
                        )

                        if result.decision.action == Action.EXIT:
                            logging.info(f"  [L{layer_idx}] EARLY EXIT triggered.")
                            final_logits = self._get_early_exit_logits(
                                hidden_state[:, -1:, :]
                            )
                            raise EarlyExitSignal(final_logits)

                        elif result.decision.action == Action.SKIP:
                            current_ckpt_idx = checkpoint_idx
                            target_ckpt_idx = (
                                current_ckpt_idx + result.decision.skip_count
                            )

                            if target_ckpt_idx < len(self.checkpoints):
                                target_layer = self.checkpoints[target_ckpt_idx]

                                logging.info(
                                    f"  [L{layer_idx}]  SKIPPING to L{target_layer}"
                                    f" (Checkpoint {target_ckpt_idx})."
                                )

                                ctx.skipping_active = True
                                ctx.landing_layer = target_layer
                                # clone the state so intermediate layers
                                # don't mutate our teleport vector
                                ctx.teleport_vector = hidden_state[:, -1:, :].clone()
                                ctx.skipped_layers_count += target_layer - layer_idx

                new_args = (hidden_state,) + args[1:]
                return new_args, kwargs

            return hook

        # add hooks at all checkpoints
        if vector_db is not None:
            for layer_idx in self.checkpoints:
                layer = self.model.inner.model.layers[layer_idx]
                handles.append(
                    layer.register_forward_pre_hook(
                        make_eval_hook(layer_idx), with_kwargs=True
                    )
                )

        all_tokens = input_ids.clone()
        current_tokens = input_ids
        past_key_values = DynamicCache()

        try:
            # generation loop
            for i in range(max_new_tokens):
                # reset context flags for the new token pass
                ctx.skipping_active = False
                ctx.landing_layer = -1
                ctx.teleport_vector = None

                try:
                    with torch.no_grad():
                        # the model updates past_key_values in-place during this call
                        outputs = self.model.inner(
                            current_tokens,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    final_logits = outputs.logits[:, -1, :]

                except EarlyExitSignal as e:
                    final_logits = e.final_logits

                # greedy decode
                next_token_id = torch.argmax(final_logits, dim=-1).item()

                if next_token_id == self.model.tokenizer.eos_token_id:
                    logging.info(
                        "  [Generation] Reached EOS token, stopping generation."
                    )
                    break

                logging.info(
                    f"Generated token ({i}): "
                    f"'{self.model.tokenizer.decode(next_token_id)}'"
                )

                next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
                all_tokens = torch.cat([all_tokens, next_token_tensor], dim=1)

                # for the next step, input is just the new token
                # this is essentially the decode step using past_key_values
                current_tokens = next_token_tensor
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=self.device)], dim=1
                )

        finally:
            # clean up hooks
            for h in handles:
                h.remove()

        generated_tokens_tensor = all_tokens[0, input_length:]
        full_text = self.model.to_string(all_tokens[0])
        generated_text = self.model.to_string(generated_tokens_tensor)

        return SkipGenerationResult(
            full_text=full_text,
            generated_text=generated_text,
            generated_tokens=generated_tokens_tensor.tolist(),
            prompt_tokens=input_ids[0].tolist(),
            generated_token_count=len(generated_tokens_tensor),
            skipped_layers=ctx.skipped_layers_count,
        )
