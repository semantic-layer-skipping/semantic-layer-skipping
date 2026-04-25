import logging
from collections import defaultdict

import pandas as pd
import torch
from inference.torch_runner import ReadOnlyCache, TorchSkipRunner
from pydantic import BaseModel
from store import SkippingVectorDB
from structures import Action, DatasetSample


class CalibrationResult(BaseModel):
    checkpoint_idx: int
    similarity: float
    decision_type: Action
    success: bool


class SkipCalibrator:
    def __init__(self, runner: TorchSkipRunner, vector_db: SkippingVectorDB):
        self.runner = runner
        self.db = vector_db
        # store results per checkpoint
        self.results: dict[int, list[CalibrationResult]] = defaultdict(list)

    def run_calibration_batch(
        self,
        prompts: list[str | DatasetSample],
        total_final_tokens: int = 2048,
    ):
        """
        Batched calibration pass to determine similarity thresholds.
        Extracts ground truth context in one pass, then dynamically groups
        DB hits into sub-batches for parallel simulation of skips and early exits.
        """
        if not prompts:
            return

        batch_size = len(prompts)
        logging.info(f"Starting batched calibration on {batch_size} prompts...")

        formatted_prompts = [self.runner.format_prompt(p) for p in prompts]
        inputs = self.runner.model.tokenizer(
            formatted_prompts, return_tensors="pt", padding=True
        ).to(self.runner.device)

        prompt_tokens = inputs.input_ids
        attention_mask = inputs.attention_mask
        prompt_len = prompt_tokens.shape[1]

        if (attention_mask.sum(dim=-1) >= total_final_tokens).all():
            logging.warning("All prompts meet or exceed total_final_tokens. Skipping.")
            return

        # Phase 1: generate full sequence (ground truth)
        with torch.no_grad():
            full_sequence_tokens = self.runner.model.inner.generate(
                prompt_tokens,
                attention_mask=attention_mask,
                max_length=total_final_tokens,
                do_sample=False,
                pad_token_id=self.runner.model.tokenizer.pad_token_id,
            )

        # Phase 2: extract hidden states and cache for all tokens simultaneously
        tokens_to_process = full_sequence_tokens[:, :-1]
        full_attention_mask = (
            tokens_to_process != self.runner.model.tokenizer.pad_token_id
        ).long()

        with torch.no_grad():
            gt_outputs = self.runner.model.inner.model(
                tokens_to_process,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )

        hidden_states = gt_outputs.hidden_states
        past_key_values = gt_outputs.past_key_values
        seq_len = full_sequence_tokens.shape[1]

        # Phase 3: simulation loop and batch regrouping
        for step in range(prompt_len - 1, seq_len - 1):
            target_tokens = full_sequence_tokens[:, step + 1]
            active_batch_mask = (
                target_tokens != self.runner.model.tokenizer.pad_token_id
            )

            if not active_batch_mask.any():
                continue

            # pre-stack all checkpoints for this exact generation step
            # shape: [num_checkpoints, batch_size, hidden_dim]
            step_states = torch.stack(
                [hidden_states[l_idx][:, step, :] for l_idx in self.runner.checkpoints]
            )

            # map target_layer_idx list of skip requests
            sim_requests = defaultdict(list)
            # collect early exits for batched processing
            exit_requests = []

            # query db sequentially and group requests by their action type
            for b in range(batch_size):
                if not active_batch_mask[b]:
                    continue

                for i_idx, _layer_idx in enumerate(self.runner.checkpoints):
                    query_vec = (
                        step_states[i_idx, b]
                        .to(torch.float32)
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(1, -1)
                    )
                    results = self.db.search(i_idx, query_vec)

                    if not results:
                        continue

                    # TODO: calibration supports only top-1 neighbour and similarity
                    result = results[0]

                    if result.decision.action == Action.SKIP:
                        j_idx = i_idx + result.decision.skip_count
                        if j_idx < len(self.runner.checkpoints):
                            target_layer_idx = self.runner.checkpoints[j_idx]
                            sim_requests[target_layer_idx].append(
                                {
                                    "b_idx": b,
                                    "i_idx": i_idx,
                                    "state": step_states[i_idx, b],
                                    "sim": result.similarity,
                                }
                            )

                    elif result.decision.action == Action.EXIT:
                        exit_requests.append(
                            {
                                "b_idx": b,
                                "i_idx": i_idx,
                                "state": step_states[i_idx, b],
                                "sim": result.similarity,
                            }
                        )

            # evaluate batched early exits
            if exit_requests:
                b_indices = [req["b_idx"] for req in exit_requests]
                active_b_tensor = torch.tensor(b_indices, device=self.runner.device)
                states_to_eval = torch.stack([req["state"] for req in exit_requests])

                # project states directly to vocab space in one pass
                with torch.no_grad():
                    early_logits = self.runner.get_early_exit_logits(states_to_eval)
                    sim_preds = torch.argmax(early_logits, dim=-1)

                target_tokens_subset = target_tokens[active_b_tensor]
                self._add_results(
                    exit_requests, sim_preds, target_tokens_subset, Action.EXIT
                )

            # evaluate batched skips
            if sim_requests:
                # setup global readonly cache for this step
                sim_cache = ReadOnlyCache()
                for l_idx in range(len(self.runner.model.inner.model.layers)):
                    k, v = past_key_values[l_idx]
                    sim_cache.initial_update(
                        k[:, :, :step, :], v[:, :, :step, :], l_idx
                    )

                sim_attn_mask = full_attention_mask[:, : step + 1]

                # execute batched simulations grouped by target layer
                for target_layer_idx, requests in sim_requests.items():
                    b_indices = [req["b_idx"] for req in requests]
                    active_b_tensor = torch.tensor(b_indices, device=self.runner.device)
                    states_to_inject = torch.stack([req["state"] for req in requests])

                    dummy_tokens = full_sequence_tokens[
                        active_b_tensor, step
                    ].unsqueeze(1)
                    sim_attn_mask_batched = sim_attn_mask[active_b_tensor]

                    # run strict simulation logic extracted from skip runner
                    sim_preds = self.runner.execute_strict_batched_simulation(
                        target_layer_idx=target_layer_idx,
                        states_to_inject=states_to_inject,
                        active_b_tensor=active_b_tensor,
                        dummy_tokens=dummy_tokens,
                        sim_attn_mask_batched=sim_attn_mask_batched,
                        sim_cache=sim_cache,
                    )

                    target_tokens_subset = target_tokens[active_b_tensor]
                    self._add_results(
                        requests, sim_preds, target_tokens_subset, Action.SKIP
                    )

        # clear variables to prevent gradual memory leaks across multiple batch calls
        del hidden_states, past_key_values
        torch.cuda.empty_cache()
        logging.info("Batched calibration pass complete.")

    def _add_results(
        self,
        requests: list[dict],
        predictions: torch.Tensor,
        target_tokens_subset: torch.Tensor,
        decision_type: Action,
    ):
        success_mask = (predictions == target_tokens_subset).cpu().tolist()

        for req, success in zip(requests, success_mask, strict=True):
            self.results[req["i_idx"]].append(
                CalibrationResult(
                    checkpoint_idx=req["i_idx"],
                    similarity=req["sim"],
                    decision_type=decision_type,
                    success=success,
                )
            )

    def find_optimal_thresholds(self, min_precision: float = 0.98) -> dict[int, float]:
        """
        Analyses calibration results to find the lowest similarity threshold
        that maintains 'min_precision'.

        Returns: Dict {checkpoint_idx: threshold}
        """
        thresholds = {}
        logging.info(
            f"Computing Thresholds (Target Precision: {min_precision * 100:.1f}%)"
        )

        for checkpoint_idx, results_list in self.results.items():
            if not results_list:
                continue

            data_dicts = [result.model_dump() for result in results_list]
            df = pd.DataFrame(data_dicts)
            df = df.sort_values(by="similarity", ascending=False)
            best_threshold = 1.0  # start with safest

            # iterate through candidate similarities - high to low,
            # stopping when precision drops below target
            similarity_candidates = sorted(df["similarity"].unique(), reverse=True)
            for t in similarity_candidates:
                subset = df[df["similarity"] >= t]

                accuracy = subset["success"].mean()
                if accuracy >= min_precision:
                    best_threshold = t
                else:
                    # if precision drops below target, we stop expanding
                    # we assume monotonicity: lower similarity = higher risk of error
                    break

            thresholds[checkpoint_idx] = float(best_threshold)
            kept = len(df[df["similarity"] >= best_threshold])
            total = len(df)
            layer_num = self.runner.checkpoints[checkpoint_idx]
            logging.info(
                f"Checkpoint {checkpoint_idx} (L{layer_num}): "
                f"Threshold {best_threshold:.4f} | "
                f"Keeps {kept}/{total} ({kept / total:.1%})"
            )

        return thresholds

    def reset_results(self):
        """Clears stored calibration results."""
        self.results.clear()
