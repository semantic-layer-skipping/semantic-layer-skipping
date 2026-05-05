import logging
from collections import defaultdict
from enum import StrEnum, auto

import pandas as pd
import torch
from inference.torch_runner import ReadOnlyCache, TorchSkipRunner
from pydantic import BaseModel
from store import SkippingVectorDB
from structures import Action, DatasetSample
from tqdm import tqdm
from utils import compute_truncated_kl_divergence


class CalibrationStrategyMode(StrEnum):
    TOKEN_MATCH = auto()
    KL_DIVERGENCE = auto()
    HIT_RATE = auto()


class CalibrationResult(BaseModel):
    checkpoint_idx: int
    similarity: float
    decision_type: Action
    success: bool  # legacy/fallback success (defaults to strict match)
    is_strict_match: bool | None = None
    kl_div: float | None = None


class SkipCalibrator:
    def __init__(self, runner: TorchSkipRunner, vector_db: SkippingVectorDB):
        self.runner = runner
        self.db = vector_db
        # store results per checkpoint
        self.results: dict[int, list[CalibrationResult]] = defaultdict(list)
        # track total queries for per-checkpoint hit-rate calculation
        self.total_queries: dict[int, int] = defaultdict(int)

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
        logging.info(f"Finished Phase 1 sequence generation on {batch_size} prompts.")

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
        logging.info(
            f"Finished Phase 2 hidden state extraction on {batch_size} prompts."
        )

        hidden_states = gt_outputs.hidden_states
        past_key_values = gt_outputs.past_key_values
        seq_len = full_sequence_tokens.shape[1]

        # Phase 3: simulation loop and batch regrouping
        for step in tqdm(prompt_len - 1, seq_len - 1, desc="Phase 3"):
            target_tokens = full_sequence_tokens[:, step + 1]
            active_batch_mask = (
                target_tokens != self.runner.model.tokenizer.pad_token_id
            )

            if not active_batch_mask.any():
                continue

            # compute target final logits for KL divergence
            final_hidden_state = hidden_states[-1][:, step, :]
            with torch.no_grad():
                target_final_logits = self.runner.get_early_exit_logits(
                    final_hidden_state
                )
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
                    self.total_queries[i_idx] += 1

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
                    sim_logits = self.runner.get_early_exit_logits(states_to_eval)

                target_logits_subset = target_final_logits[active_b_tensor]
                target_tokens_subset = target_tokens[active_b_tensor]
                self._add_results(
                    exit_requests,
                    sim_logits,
                    target_logits_subset,
                    target_tokens_subset,
                    Action.EXIT,
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
                    sim_logits = self.runner.execute_strict_batched_simulation(
                        target_layer_idx=target_layer_idx,
                        states_to_inject=states_to_inject,
                        active_b_tensor=active_b_tensor,
                        dummy_tokens=dummy_tokens,
                        sim_attn_mask_batched=sim_attn_mask_batched,
                        sim_cache=sim_cache,
                    )

                    target_logits_subset = target_final_logits[active_b_tensor]
                    target_tokens_subset = target_tokens[active_b_tensor]
                    self._add_results(
                        requests,
                        sim_logits,
                        target_logits_subset,
                        target_tokens_subset,
                        Action.SKIP,
                    )

        # clear variables to prevent gradual memory leaks across multiple batch calls
        del hidden_states, past_key_values
        torch.cuda.empty_cache()
        logging.info("Batched calibration pass complete.")

    def _add_results(
        self,
        requests: list[dict],
        sim_logits: torch.Tensor,
        target_logits_subset: torch.Tensor,
        target_tokens_subset: torch.Tensor,
        decision_type: Action,
    ):
        """Calculates all metrics (Strict and KL) and appends to results."""
        sim_preds = torch.argmax(sim_logits, dim=-1)
        strict_match_mask = (sim_preds == target_tokens_subset).cpu().tolist()

        # calculate exact KL Divergence over the vocab distribution
        with torch.no_grad():
            kl_divs = (
                compute_truncated_kl_divergence(
                    true_logits=target_logits_subset, sim_logits=sim_logits, top_k=50
                )
                .cpu()
                .tolist()
            )

        for req, is_strict, kl in zip(
            requests, strict_match_mask, kl_divs, strict=True
        ):
            self.results[req["i_idx"]].append(
                CalibrationResult(
                    checkpoint_idx=req["i_idx"],
                    similarity=req["sim"],
                    decision_type=decision_type,
                    success=is_strict,  # keep backwards compatibility default
                    is_strict_match=is_strict,
                    kl_div=kl,
                )
            )

    def find_optimal_thresholds(
        self,
        *,
        strategy: CalibrationStrategyMode,
        min_precision: float | dict[int, float],
        min_hit_rate: float | dict[int, float],
        kl_success_threshold: float = 2.0,  # only used for kl divergence strategy
    ) -> dict[int, float]:
        """
        Dynamically calculates optimal thresholds based on the selected strategy.
        - STRICT: uses min_precision (e.g., 0.98)
        - KL_DIVERGENCE: uses min_precision (e.g., 0.95),
            where "success" is kl_div < kl_success_threshold
        - HIT_RATE: min_hit_rate = ratio of queries to skip (e.g., 0.10 for 10%)
        """
        thresholds = {}
        if (
            strategy == CalibrationStrategyMode.TOKEN_MATCH
            or strategy == CalibrationStrategyMode.KL_DIVERGENCE
        ):
            target_value = min_precision
        elif strategy == CalibrationStrategyMode.HIT_RATE:
            target_value = min_hit_rate
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        logging.info(
            f"Computing Thresholds | Strategy: {strategy.name} | Target: {target_value}"
        )
        for checkpoint_idx, results_list in self.results.items():
            if not results_list:
                continue

            local_target = (
                target_value.get(checkpoint_idx, target_value)
                if isinstance(target_value, dict)
                else target_value
            )
            if local_target is None:
                raise ValueError(
                    f"Target value for strategy {strategy} is missing "
                    f"for checkpoint {checkpoint_idx}."
                )

            data_dicts = [result.model_dump() for result in results_list]
            df = pd.DataFrame(data_dicts)

            # sort highest similarity to lowest
            df = df.sort_values(by="similarity", ascending=False).reset_index(drop=True)

            best_threshold = 1.0
            if strategy in (
                CalibrationStrategyMode.TOKEN_MATCH,
                CalibrationStrategyMode.KL_DIVERGENCE,
            ):
                # define what "success" is based on the strategy
                if strategy == CalibrationStrategyMode.TOKEN_MATCH:
                    # fallback to success if is_strict_match is None
                    df["current_success"] = df["is_strict_match"].fillna(df["success"])
                else:
                    if "kl_div" not in df.columns or df["kl_div"].isnull().all():
                        raise ValueError(
                            "The column 'kl_div' is missing or contains null values "
                            "in the results. "
                            "Please re-run calibration to compute KL divergence values."
                        )
                    df["current_success"] = df["kl_div"] < kl_success_threshold

                # vectorised cumulative precision
                df["cum_successes"] = df["current_success"].cumsum()
                df["cum_total"] = df.index + 1
                df["cum_precision"] = df["cum_successes"] / df["cum_total"]

                # only evaluate the precision at the end of each similarity group
                df_thresholds = df.drop_duplicates(subset=["similarity"], keep="last")

                # find all points where the true cumulative precision meets our target
                valid_thresholds = df_thresholds[
                    df_thresholds["cum_precision"] >= local_target
                ]

                if not valid_thresholds.empty:
                    # take the lowest similarity that still maintained
                    # the target precision
                    best_threshold = valid_thresholds["similarity"].min()
                # otherwise, it never met the target, so best_threshold stays at 1.0

            elif strategy == CalibrationStrategyMode.HIT_RATE:
                total_possible_queries = self.total_queries[checkpoint_idx]
                target_hit_count = int(total_possible_queries * local_target)

                if target_hit_count == 0:
                    best_threshold = 1.0
                elif target_hit_count > len(df):
                    logging.warning(
                        f"L{checkpoint_idx}: Target hit rate ({local_target:.1%}) is "
                        f"impossible. DB is too sparse."
                    )
                    best_threshold = df["similarity"].min() if not df.empty else 1.0
                else:
                    # threshold is the similarity of the Nth item
                    best_threshold = df.iloc[target_hit_count - 1]["similarity"]

            thresholds[checkpoint_idx] = float(best_threshold)

            # logging stats
            kept = len(df[df["similarity"] >= best_threshold])
            total = len(df)
            layer_num = self.runner.checkpoints[checkpoint_idx]
            logging.info(
                f"Checkpoint {checkpoint_idx} (L{layer_num}): "
                f"Threshold {best_threshold:.4f} | "
                f"Keeps {kept}/{total} ({kept / total:.1%})"
            )

        return thresholds

    def get_serialised_results(self) -> dict:
        """Converts internal results and query counts to a json-safe dictionary."""
        return {
            "results": {
                k: [v.model_dump() for v in v_list]
                for k, v_list in self.results.items()
            },
            "total_queries": {k: v for k, v in self.total_queries.items()},
        }

    def load_serialised_results(self, data: dict):
        """Loads pre-computed simulation results from disk, with legacy fallback."""
        self.results.clear()
        self.total_queries.clear()

        # check if new format (contains explicitly separated dicts)
        if "results" in data and "total_queries" in data:
            for ckpt_idx_str, res_list in data["results"].items():
                ckpt_idx = int(ckpt_idx_str)
                self.results[ckpt_idx] = [CalibrationResult(**res) for res in res_list]

            for ckpt_idx_str, count in data["total_queries"].items():
                self.total_queries[int(ckpt_idx_str)] = count
        else:
            # legacy format fallback
            logging.warning(
                "Legacy calibration data detected ('total_queries' missing). "
                "Assuming the DB returned a hit for EVERY query. "
                "Hit-Rate calibration may be slightly skewed if the DB was sparse."
            )
            for ckpt_idx_str, res_list in data.items():
                ckpt_idx = int(ckpt_idx_str)
                self.results[ckpt_idx] = [CalibrationResult(**res) for res in res_list]
                # fallback: Assume total queries equals the number of hits recorded
                self.total_queries[ckpt_idx] = len(res_list)

    def reset_results(self):
        """Clears stored calibration results."""
        self.results.clear()
        self.total_queries.clear()
