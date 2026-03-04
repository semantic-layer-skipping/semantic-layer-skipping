import contextlib
import logging
import time
from collections import defaultdict

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

# global dictionary to accumulate times across all steps
phase3_timings = defaultdict(float)


@contextlib.contextmanager
def sync_timer(name: str, device: torch.device):
    """Accurately measures hardware execution time by forcing synchronization."""
    # sync before starting the timer to clear any pending queue
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    yield

    # sync after the block so the CPU waits for the GPU/MPS to finish math
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

    phase3_timings[name] += time.perf_counter() - start


class ReadOnlyCache(DynamicCache):
    """
    Inherits from HF's DynamicCache to perfectly preserve all metadata.
    Overrides update() during the forward pass to prevent cache mutation
    and dynamically re-maps batch sizes for cross-checkpoint parallelization.
    """

    def __init__(self):
        super().__init__()
        self.batch_mapping = None

    def initial_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # use standard update logic for cache initialisation
        super().update(key_states, value_states, layer_idx, cache_kwargs=cache_kwargs)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # get past states
        past_k, past_v = self[layer_idx]

        # dynamically duplicate/reorder the history
        # to match the expanded simulation batch
        if self.batch_mapping is not None:
            past_k = past_k[self.batch_mapping]
            past_v = past_v[self.batch_mapping]

        k_out = torch.cat([past_k, key_states], dim=-2)
        v_out = torch.cat([past_v, value_states], dim=-2)

        return k_out, v_out

    def set_batch_mapping(self, batch_mapping):
        """
        Sets the batch mapping for dynamic reordering/duplication of cache states.
        This is used to align the cache with the active sequences in the batch
        during parallel simulation.
        """
        self.batch_mapping = batch_mapping


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

            # track which (start_checkpoint_idx, batch_idx) has already found a skip
            furthest_skip_found = torch.zeros(
                (len(self.checkpoints), batch_size),
                dtype=torch.bool,
                device=self.device,
            )

            if early_exit_strategy:
                with sync_timer("2. Early Exit Logic", self.device):
                    for i, layer_idx in enumerate(self.checkpoints):
                        current_states = hidden_states[layer_idx][:, step, :]
                        early_logits = self._get_early_exit_logits(current_states)
                        for b in range(batch_size):
                            if active_batch_mask[b]:
                                if early_exit_strategy.should_exit(
                                    early_logits[b], target_final_logits[b]
                                ):
                                    # TODO: should we prevent skipping
                                    #  if early exit is triggered?
                                    # furthest_skip_found[i, b] = True
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

            if skip_strategy_mode is None:
                continue

            # form the KV cache for this simulated generation
            # which contains all KVs up to current step
            # we use our custom class, but initialise it using native HF logic
            # the custom class ensures later updates don't mutate the cache
            with sync_timer("1. Cache Wrapper Setup", self.device):
                sim_cache = ReadOnlyCache()
                for l_idx in range(len(self.model.inner.model.layers)):
                    k, v = past_key_values[l_idx]
                    sim_cache.initial_update(
                        k[:, :, :step, :], v[:, :, :step, :], l_idx
                    )

            # mask covers cache (length `step`) + the new token (length 1) = `step + 1`
            sim_attn_mask = full_attention_mask[:, : step + 1]

            # iterate backwards through target checkpoints (j)
            for j_idx in range(len(self.checkpoints) - 1, 0, -1):
                target_layer_idx = self.checkpoints[j_idx]

                # identify all source checkpoints (i) that want to skip here
                # also track which sequences want to skip from which source checkpoint
                active_i = []
                active_b = []
                for i_idx in range(j_idx):
                    for b in range(batch_size):
                        if not furthest_skip_found[i_idx, b] and active_batch_mask[b]:
                            active_i.append(i_idx)
                            active_b.append(b)

                if not active_i:
                    continue

                num_active = len(active_i)

                if skip_strategy_mode == SkipStrategyMode.COSINE:
                    for idx in range(num_active):
                        i_idx = active_i[idx]
                        b = active_b[idx]
                        source_layer = self.checkpoints[i_idx]

                        sim = functional.cosine_similarity(
                            hidden_states[source_layer][b : b + 1, step, :],
                            hidden_states[target_layer_idx][b : b + 1, step, :],
                            dim=-1,
                        )
                        if sim.item() >= similarity_threshold:
                            furthest_skip_found[i_idx, b] = True
                            vector_db.add_vector(
                                i_idx,
                                hidden_states[source_layer][b]
                                .to(torch.float32)
                                .detach()
                                .cpu()
                                .numpy()
                                .reshape(1, -1),
                                SkipDecision(
                                    action=Action.SKIP, skip_count=(j_idx - i_idx)
                                ),
                            )

                elif skip_strategy_mode == SkipStrategyMode.STRICT:
                    with sync_timer("3. Strict Forward Pass Setup", self.device):
                        # dynamically stack states to form 1 batch
                        states_to_inject = torch.stack(
                            [
                                hidden_states[self.checkpoints[i_idx]][b, step, :]
                                for i_idx, b in zip(active_i, active_b, strict=True)
                            ]
                        )

                        dummy_tokens = torch.stack(
                            [full_sequence_tokens[b, step] for b in active_b]
                        ).unsqueeze(1)

                        sim_attn_mask_batched = torch.stack(
                            [sim_attn_mask[b] for b in active_b]
                        )

                        # tell the cache how to map ground truth sequences
                        # to this expanded batch
                        sim_cache.batch_mapping = torch.tensor(
                            active_b, dtype=torch.long, device=self.device
                        )

                        original_forwards = {}
                        for layer_index in range(target_layer_idx):
                            layer_module = self.model.inner.model.layers[layer_index]
                            original_forwards[layer_index] = layer_module.forward
                            layer_module.forward = get_dummy_forward()

                        handle = self.model.inner.model.layers[
                            target_layer_idx
                        ].register_forward_pre_hook(
                            make_injection_hook(states_to_inject), with_kwargs=True
                        )

                    try:
                        with sync_timer(
                            "4. Strict Forward Pass Execution", self.device
                        ):
                            with torch.no_grad():
                                sim_outputs = self.model.inner(
                                    dummy_tokens,
                                    attention_mask=sim_attn_mask_batched,
                                    # pass the previously-computed KV cache
                                    past_key_values=sim_cache,
                                    # ensures huggingface doesnt' return cache
                                    # however, it still internally updates it,
                                    # which is why we use our custom ReadOnlyCache
                                    use_cache=False,
                                )
                            sim_preds = torch.argmax(
                                sim_outputs.logits[:, -1, :], dim=-1
                            )

                        with sync_timer("5. Vector DB Insertion", self.device):
                            for idx in range(num_active):
                                i_idx = active_i[idx]
                                b = active_b[idx]

                                if sim_preds[idx] == target_tokens[b]:
                                    furthest_skip_found[i_idx, b] = True
                                    skip_count = j_idx - i_idx
                                    vector_db.add_vector(
                                        i_idx,
                                        hidden_states[self.checkpoints[i_idx]][
                                            b, step, :
                                        ]
                                        .to(torch.float32)
                                        .detach()
                                        .cpu()
                                        .numpy()
                                        .reshape(1, -1),
                                        SkipDecision(
                                            action=Action.SKIP, skip_count=skip_count
                                        ),
                                    )
                    finally:
                        with sync_timer("6. Finally cleanup", self.device):
                            handle.remove()
                            for layer_index, original_fwd in original_forwards.items():
                                self.model.inner.model.layers[
                                    layer_index
                                ].forward = original_fwd

                            # clean up the mapping so it doesn't pollute the next run
                            sim_cache.batch_mapping = None

        full_texts = [
            self.model.tokenizer.decode(
                t[t != self.model.tokenizer.pad_token_id], skip_special_tokens=True
            )
            for t in full_sequence_tokens
        ]

        del hidden_states, past_key_values, original_logits
        torch.cuda.empty_cache()

        logging.info("  [Generation] Phase 3 Complete. Finished batched population.")

        logging.info("\n=== Phase 3 Profiling Results ===")
        for name, duration in phase3_timings.items():
            logging.info(f"{name}: {duration:.4f} seconds")
        logging.info("=================================")

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
