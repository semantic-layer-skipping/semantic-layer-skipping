import contextlib
import logging
import time
from collections import defaultdict
from typing import Any

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
    apply_frequency_penalty,
    apply_repetition_penalty,
    get_early_exit_strategy,
)
from store import SkippingVectorDB
from structures import Action, SkipDecision, SkipGenerationResult
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# global dictionary to accumulate times across all steps
phase3_timings = defaultdict(float)
REPETITION_PENALTY = 1.2


@contextlib.contextmanager
def timer(name: str, device: torch.device, sync: bool = False):
    """Measures async execution time"""
    if sync:
        # sync before starting the timer to clear any pending queue
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    start = time.perf_counter()

    yield

    if sync:
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
    def _load_model(self, compile_model=False) -> Model:
        inner = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=self.device, torch_dtype=torch.float16
        )
        if compile_model and self.device.type == "cuda":
            logging.info("Optimising model with torch.compile...")
            # TODO: compilation might be effective if we subclass model with a
            #  skipping forward that takes additional args (e.g. checkpoint_idx)
            inner = torch.compile(inner)
        inner.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return TorchModel(inner, tokenizer)

    def get_early_exit_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
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
        log_prompts: bool = False,
        log_profiling: bool = False,
        repetition_penalty: float = REPETITION_PENALTY,
    ) -> list[str]:
        # reset timings for this batch
        phase3_timings.clear()
        if len(prompts) == 0:
            logging.warning("No prompts provided to generate_and_populate_batched.")
            return []

        start = time.perf_counter()
        if log_profiling:
            logging.info(
                f"Batched populating {len(prompts)} prompts. "
                f"Target total: {total_final_tokens} tokens."
            )

        formatted_prompts = [self.format_prompt(p) for p in prompts]

        # log inputs
        if log_prompts:
            for idx, p in enumerate(formatted_prompts):
                logging.info(
                    f"\n{'=' * 20} "
                    f"FORMATTED INPUT PROMPT {idx} "
                    f"{'=' * 20}\n{p}\n{'=' * 54}"
                )

        inputs = self.model.tokenizer(
            formatted_prompts, return_tensors="pt", padding=True
        ).to(self.device)

        prompt_tokens = inputs.input_ids
        attention_mask = inputs.attention_mask
        batch_size = prompt_tokens.shape[0]

        prompt_lengths = attention_mask.sum(dim=-1)
        if log_prompts:
            logging.info(
                f"  Initial token lengths for prompts in batch: "
                f"{prompt_lengths.tolist()}"
            )

        if (prompt_lengths >= total_final_tokens).all():
            logging.warning("All prompts meet or exceed total_final_tokens. Skipping.")
            return [
                self.model.tokenizer.decode(t[m == 1])
                for t, m in zip(prompt_tokens, attention_mask, strict=True)
            ]

        # phase 1: generate tokens for all inputs in the batch
        with timer("Phase 1: Generation", self.device):
            with torch.no_grad():
                full_sequence_tokens = self.model.inner.generate(
                    prompt_tokens,
                    attention_mask=attention_mask,
                    max_length=total_final_tokens,
                    do_sample=False,  # greedy decoding
                    pad_token_id=self.model.tokenizer.pad_token_id,
                    repetition_penalty=repetition_penalty,
                )

        if log_profiling:
            logging.info("  [Generation] Phase 1 complete: generated full sequences.")

        # # log full generated outputs
        if log_prompts:
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
        with timer("Phase 2: Hidden State Extraction", self.device):
            tokens_to_process = full_sequence_tokens[:, :-1]
            full_attention_mask = (
                tokens_to_process != self.model.tokenizer.pad_token_id
            ).long()

            with torch.no_grad():
                # call .model to bypass the lm_head and only return hidden states
                gt_outputs = self.model.inner.model(
                    tokens_to_process,
                    attention_mask=full_attention_mask,
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True,
                )

            hidden_states = gt_outputs.hidden_states
            # calling .model returns hidden states for all layers, but no logits
            # original_logits = gt_outputs.logits # this is massive: takes too much VRAM
            past_key_values = gt_outputs.past_key_values

        if log_profiling:
            logging.info(
                "  [Generation] Phase 2 complete: extracted hidden states/cache."
            )

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

        # only use tqdm if log_prompts is False
        step_iterator = range(prompt_len - 1, seq_len - 1)
        if not log_prompts:
            step_iterator = tqdm(
                step_iterator,
                desc="[Phase 3] Simulating generation and populating DB",
            )
        for step in step_iterator:
            target_tokens = full_sequence_tokens[:, step + 1]
            # identify which sequences in the batch are still active (still generating)
            active_batch_mask = target_tokens != self.model.tokenizer.pad_token_id
            # set active batch mask to be true for all
            if not active_batch_mask.any():
                # technically, this should never execute: phase 1 finds max length
                continue

            with timer("[Phase 3] 0. Compute Target Logits", self.device):
                # compute target final logits, on the fly, to save memory
                # these will be used in early exit strategy
                final_hidden_state = hidden_states[-1][:, step, :]
                with torch.no_grad():
                    # project through the norm and lm_head to get logits for this step
                    normed_state = self.model.inner.model.norm(final_hidden_state)
                    target_final_logits = self.model.inner.lm_head(normed_state)

            # pre-stack all checkpoints for this step once so we can index them
            # shape: [num_checkpoints, batch_size, hidden_dim]
            step_states = torch.stack(
                [hidden_states[l_idx][:, step, :] for l_idx in self.checkpoints]
            )

            # log phase 3 step
            step_num = step - (prompt_len - 1) + 1
            # decode target tokens for logging
            if log_prompts:
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
                with timer("1. Early Exit Logic", self.device):
                    with torch.no_grad():
                        batched_early_logits = self.get_early_exit_logits(step_states)
                        exit_mask = early_exit_strategy.should_exit_batched(
                            batched_early_logits, target_final_logits
                        )

                    # mask out sequences that are just padding tokens
                    valid_exits = exit_mask & active_batch_mask.unsqueeze(0)
                    # get the coordinates of all True values
                    checkpoint_indices, batch_indices = valid_exits.nonzero(
                        as_tuple=True
                    )

                    for i, b in zip(
                        checkpoint_indices.tolist(), batch_indices.tolist(), strict=True
                    ):
                        # TODO: we could also not prevent skipping
                        #  if early exit is triggered?
                        furthest_skip_found[i, b] = True
                        vector_db.add_vector(
                            i,
                            step_states[i, b]
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
            with timer("2. Cache Wrapper Setup", self.device):
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
                # vectorised discovery: find all active i < j that haven't skipped yet
                # shape: [j_idx, batch_size]
                valid_sources = (
                    ~furthest_skip_found[:j_idx, :]
                ) & active_batch_mask.unsqueeze(0)

                # get the tensor indices of all True values
                active_i_tensor, active_b_tensor = valid_sources.nonzero(as_tuple=True)
                if active_i_tensor.numel() == 0:
                    continue

                if skip_strategy_mode == SkipStrategyMode.COSINE:
                    # vectorised cosine simulation
                    source_states = step_states[active_i_tensor, active_b_tensor]
                    target_states = hidden_states[target_layer_idx][
                        active_b_tensor, step, :
                    ]

                    sims = functional.cosine_similarity(
                        source_states, target_states, dim=-1
                    )
                    success_indices = (sims >= similarity_threshold).nonzero(
                        as_tuple=True
                    )[0]

                    for idx in success_indices.tolist():
                        i_idx = active_i_tensor[idx].item()
                        b = active_b_tensor[idx].item()
                        furthest_skip_found[i_idx, b] = True
                        vector_db.add_vector(
                            i_idx,
                            source_states[idx]
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
                    with timer("3. Strict Forward Pass Setup", self.device):
                        # index into the states to get states to inject
                        states_to_inject = step_states[active_i_tensor, active_b_tensor]

                        dummy_tokens = full_sequence_tokens[
                            active_b_tensor, step
                        ].unsqueeze(1)
                        sim_attn_mask_batched = sim_attn_mask[active_b_tensor]

                    with timer(
                        "4. Strict Forward Pass Execution (Batched)", self.device
                    ):
                        # execute the strict forward pass
                        sim_preds = self.execute_strict_batched_simulation(
                            target_layer_idx=target_layer_idx,
                            states_to_inject=states_to_inject,
                            active_b_tensor=active_b_tensor,
                            dummy_tokens=dummy_tokens,
                            sim_attn_mask_batched=sim_attn_mask_batched,
                            sim_cache=sim_cache,
                        )

                    with timer("5. Vector DB Insertion", self.device):
                        # vectorised check for successful skips
                        target_tokens_subset = target_tokens[active_b_tensor]
                        success_mask = sim_preds == target_tokens_subset

                        success_indices = success_mask.nonzero(as_tuple=True)[0]

                        for idx in success_indices.tolist():
                            i_idx = active_i_tensor[idx].item()
                            b = active_b_tensor[idx].item()

                            furthest_skip_found[i_idx, b] = True
                            skip_count = j_idx - i_idx
                            vector_db.add_vector(
                                i_idx,
                                states_to_inject[idx]
                                .to(torch.float32)
                                .detach()
                                .cpu()
                                .numpy()
                                .reshape(1, -1),
                                SkipDecision(action=Action.SKIP, skip_count=skip_count),
                            )

        full_texts = [
            self.model.tokenizer.decode(
                t[t != self.model.tokenizer.pad_token_id], skip_special_tokens=True
            )
            for t in full_sequence_tokens
        ]

        del hidden_states, past_key_values
        torch.cuda.empty_cache()

        if log_profiling:
            end = time.perf_counter()

            logging.info(
                "  [Generation] Phase 3 Complete. Finished batched population."
            )

            logging.info("\n=== Phase 3 Profiling Results ===")
            for name, duration in phase3_timings.items():
                logging.info(f"{name}: {duration:.4f} seconds")

            logging.info(
                f"Total time for generate_and_populate_batched: "
                f"{end - start:.4f} seconds"
            )

        return full_texts

    def execute_strict_batched_simulation(
        self,
        target_layer_idx: int,
        states_to_inject: torch.Tensor,
        active_b_tensor: torch.Tensor,
        dummy_tokens: torch.Tensor,
        sim_attn_mask_batched: torch.Tensor,
        sim_cache: ReadOnlyCache,
    ) -> torch.Tensor:
        """
        Logic for executing a batched strict matching forward pass.
        Injects states at a target layer and returns predicted next tokens.
        """

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

        # tell the cache how to map ground truth sequences
        # to this expanded batch
        sim_cache.batch_mapping = active_b_tensor

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
            with torch.no_grad():
                sim_outputs = self.model.inner(
                    dummy_tokens,
                    attention_mask=sim_attn_mask_batched,
                    # pass the previously-computed KV cache
                    past_key_values=sim_cache,
                    # ensures huggingface doesn't return cache
                    # however, it still internally updates it,
                    # which is why we use our custom ReadOnlyCache
                    use_cache=False,
                )
            sim_preds = torch.argmax(sim_outputs.logits[:, -1, :], dim=-1)
            return sim_preds
        finally:
            handle.remove()
            for layer_index, original_fwd in original_forwards.items():
                self.model.inner.model.layers[layer_index].forward = original_fwd

            # clean up the mapping so it doesn't pollute the next run
            sim_cache.batch_mapping = None

    def generate_with_skipping(
        self,
        prompt: PromptType,
        vector_db: SkippingVectorDB | None = None,
        threshold: float | dict[int, float] = DEFAULT_THRESH,
        max_total_tokens: int = 2048,
        format_prompt: bool = True,
        log_skips: bool = True,
        repetition_penalty: float = REPETITION_PENALTY,
        frequency_penalty: float = 0,
    ) -> SkipGenerationResult:
        if log_skips:
            logging.info(f"Generating with Skipping for input prompt: '{prompt}'")

        if format_prompt:
            prompt = self.format_prompt(prompt)

        inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        input_length = input_ids.shape[1]

        if input_length >= max_total_tokens:
            if log_skips:
                logging.warning(
                    f"Prompt length ({input_length}) is "
                    f">= max_total_tokens ({max_total_tokens}). "
                    "Returning prompt without generating new tokens."
                )
            full_text = self.model.to_string(input_ids[0])
            return SkipGenerationResult(
                full_text=full_text,
                generated_text="",
                generated_tokens=[],
                prompt_tokens=input_ids[0].tolist(),
                generated_token_count=0,
                skipped_layers=0,
            )

        ctx = SkipCtx()
        handles = []

        # metrics to track
        checkpoint_skip_counts: dict[int, dict[Any, int]] = {
            i: defaultdict(int) for i in range(len(self.checkpoints))
        }
        db_hit_counts: dict[int, dict[int, int]] = {
            i: defaultdict(int) for i in range(len(self.checkpoints))
        }

        # pre-allocate tensors for the full generation process to avoid dynamic resizing
        # has shape (1=batch_size, max_total_tokens)
        all_tokens = torch.zeros(
            (1, max_total_tokens), dtype=torch.long, device=self.device
        )
        all_tokens[0, :input_length] = input_ids[0]
        attention_mask = torch.zeros(
            (1, max_total_tokens), dtype=torch.long, device=self.device
        )
        attention_mask[0, :input_length] = inputs.attention_mask[0]

        # hook added at each checkpoint
        def make_eval_hook(layer_idx):
            checkpoint_idx = self.checkpoints.index(layer_idx)

            def hook(module, args, kwargs):
                hidden_state = args[0]

                # 1. landing logic: check if we have arrived at the target layer
                if ctx.skipping_active:
                    if layer_idx == ctx.landing_layer:
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
                query_vec = (
                    hidden_state[0, -1, :]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                    .reshape(1, -1)
                )

                if vector_db is not None:
                    results = vector_db.search(checkpoint_idx, query_vec)

                    if results:
                        # TODO: runner supports only top-1
                        result = results[0]
                        # log the id returned by FAISS
                        db_hit_counts[checkpoint_idx][result.neighbour_id] += 1

                        # get threshold for this checkpoint
                        if isinstance(threshold, dict):
                            local_thresh = threshold.get(checkpoint_idx, DEFAULT_THRESH)
                        else:
                            local_thresh = threshold

                        if result.similarity < local_thresh:
                            checkpoint_skip_counts[checkpoint_idx][0] += 1
                            new_args = (hidden_state,) + args[1:]
                            return new_args, kwargs

                        if log_skips:
                            logging.info(
                                f"  [L{layer_idx}] Decision to execute: "
                                f"{result.decision}"
                                f" ({local_thresh=:.4f}, {result.similarity=:.4f})"
                            )

                        if result.decision.action == Action.EXIT:
                            checkpoint_skip_counts[checkpoint_idx]["exit"] += 1
                            if log_skips:
                                logging.info(f"  [L{layer_idx}] EARLY EXIT triggered.")
                            final_logits = self.get_early_exit_logits(
                                hidden_state[:, -1:, :]
                            )
                            raise EarlyExitSignal(final_logits)

                        elif result.decision.action == Action.SKIP:
                            skip_amount = result.decision.skip_count
                            checkpoint_skip_counts[checkpoint_idx][skip_amount] += 1

                            target_ckpt_idx = (
                                checkpoint_idx + result.decision.skip_count
                            )

                            if target_ckpt_idx < len(self.checkpoints):
                                target_layer = self.checkpoints[target_ckpt_idx]

                                if log_skips:
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
                    else:
                        # no result found in vector db
                        checkpoint_skip_counts[checkpoint_idx][0] += 1

                new_args = (hidden_state,) + args[1:]
                return new_args, kwargs

            return hook

        num_generated = 0
        token_skip_distribution = defaultdict(int)
        try:
            # prefill: run model on prompt to get past_key_values
            with torch.no_grad():
                outputs = self.model.inner(
                    input_ids,
                    attention_mask=attention_mask[:, :input_length],
                    use_cache=True,
                )
            final_logits = outputs.logits[:, -1, :]
            apply_repetition_penalty(
                final_logits, all_tokens[0, :input_length], repetition_penalty
            )
            apply_frequency_penalty(
                final_logits, all_tokens[0, :input_length], frequency_penalty
            )

            # greedy decode
            next_token_id = torch.argmax(final_logits, dim=-1).item()
            past_key_values = outputs.past_key_values

            # record the first generated token
            all_tokens[0, input_length] = next_token_id
            attention_mask[0, input_length] = 1
            current_tokens = torch.tensor([[next_token_id]], device=self.device)
            num_generated += 1
            # we do not update token_skip_distribution as it only considers decode ones

            if log_skips:
                logging.info(
                    f"Generated token (0) [PREFILL]: "
                    f"'{self.model.tokenizer.decode(next_token_id)}'"
                )

            # DECODE
            if next_token_id != self.model.tokenizer.eos_token_id:
                # add hooks at all checkpoints
                if vector_db is not None:
                    for layer_idx in self.checkpoints:
                        layer = self.model.inner.model.layers[layer_idx]
                        handles.append(
                            layer.register_forward_pre_hook(
                                make_eval_hook(layer_idx), with_kwargs=True
                            )
                        )

                # generation loop
                for i in range(input_length + 1, max_total_tokens):
                    # reset context flags for the new token pass
                    ctx.skipping_active = False
                    ctx.landing_layer = -1
                    ctx.teleport_vector = None
                    skips_before_token = ctx.skipped_layers_count

                    try:
                        with torch.no_grad():
                            # the model updates past_key_values in-place during this
                            outputs = self.model.inner(
                                current_tokens,
                                attention_mask=attention_mask[:, :i],
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                        final_logits = outputs.logits[:, -1, :]

                    except EarlyExitSignal as e:
                        final_logits = e.final_logits

                    # apply penalty
                    apply_repetition_penalty(
                        final_logits, all_tokens[0, :i], repetition_penalty
                    )
                    apply_frequency_penalty(
                        final_logits, all_tokens[0, :i], frequency_penalty
                    )

                    # greedy decode
                    next_token_id = torch.argmax(final_logits, dim=-1).item()

                    if next_token_id == self.model.tokenizer.eos_token_id:
                        if log_skips:
                            logging.info(
                                "  [Generation] Reached EOS token, stopping generation."
                            )
                        break

                    if log_skips:
                        logging.info(
                            f"Generated token ({num_generated}): "
                            f"'{self.model.tokenizer.decode(next_token_id)}'"
                        )

                    all_tokens[0, i] = next_token_id
                    attention_mask[0, i] = 1
                    current_tokens = torch.tensor([[next_token_id]], device=self.device)
                    num_generated += 1

                    # update token skipping distribution
                    skips_this_token = ctx.skipped_layers_count - skips_before_token
                    token_skip_distribution[skips_this_token] += 1

        finally:
            # clean up hooks
            for h in handles:
                h.remove()

        # slice the pre-allocated tensors down to the actual generated length
        total_length = input_length + num_generated
        generated_tokens_tensor = all_tokens[0, input_length:total_length]

        full_text = self.model.to_string(all_tokens[0, :total_length])
        generated_text = self.model.to_string(generated_tokens_tensor)

        # convert default dicts to normal dicts before returning
        clean_skip_counts = {k: dict(v) for k, v in checkpoint_skip_counts.items()}
        clean_hit_counts = {k: dict(v) for k, v in db_hit_counts.items()}
        db_index_sizes = vector_db.get_index_sizes() if vector_db is not None else {}

        return SkipGenerationResult(
            full_text=full_text,
            generated_text=generated_text,
            generated_tokens=generated_tokens_tensor.tolist(),
            prompt_tokens=input_ids[0].tolist(),
            generated_token_count=len(generated_tokens_tensor),
            skipped_layers=ctx.skipped_layers_count,
            checkpoint_skip_counts=clean_skip_counts,
            db_hit_counts=clean_hit_counts,
            db_index_sizes=db_index_sizes,
            token_skip_distribution=dict(token_skip_distribution),
        )
