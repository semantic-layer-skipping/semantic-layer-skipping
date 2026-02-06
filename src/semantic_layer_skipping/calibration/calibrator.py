import logging
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import torch
from inference.runner import SemanticSkipRunner
from store import SkippingVectorDB
from structures import Action, CalibrationSuccessStrategy


@dataclass
class CalibrationResult:
    checkpoint_idx: int
    similarity: float
    decision_type: Action
    success: bool


class SkipCalibrator:
    def __init__(self, runner: SemanticSkipRunner, vector_db: SkippingVectorDB):
        self.runner = runner
        self.db = vector_db
        # store results per checkpoint
        self.results: dict[int, list[CalibrationResult]] = defaultdict(list)

    def run_calibration_pass(
        self,
        prompts: list[str],
        success_strategy: CalibrationSuccessStrategy = (
            CalibrationSuccessStrategy.TOKEN_MATCH
        ),
    ):
        """
        Runs a calibration pass on already-populated DB using the provided prompts.
        For each prompt, it checks DB at every checkpoint, simulates the retrieved skip,
        and records if it would have worked according to the success_strategy.
        """
        logging.info(f"Starting Calibration on {len(prompts)} prompts...")

        for prompt in prompts:
            tokens = self.runner.model.to_tokens(prompt)

            # get ground truth
            # TODO: we can batch process this
            logits, cache = self.runner.model.run_with_cache(
                tokens, return_type="logits"
            )
            ground_truth_token = torch.argmax(logits[0, -1, :]).item()

            # iterate over checkpoints
            for checkpoint_idx, layer_idx in enumerate(self.runner.checkpoints):
                hook_name = f"blocks.{layer_idx}.hook_resid_pre"
                current_state = cache[hook_name][0, -1, :]

                # query db for nearest neighbour
                query_vec = current_state.detach().cpu().numpy().reshape(1, -1)
                result = self.db.search(checkpoint_idx, query_vec)
                if not result:
                    continue

                # simulate decision
                predicted_token = self.runner.simulate_decision(
                    tokens, checkpoint_idx, current_state, result.decision
                )

                # check success
                is_success = False
                if success_strategy == CalibrationSuccessStrategy.TOKEN_MATCH:
                    is_success = predicted_token == ground_truth_token
                    # TODO: for task success, we can run generate_with_skipping
                    #  and check if final output is correct

                # store calibration result
                self.results[checkpoint_idx].append(
                    CalibrationResult(
                        checkpoint_idx=checkpoint_idx,
                        similarity=result.similarity,
                        decision_type=result.decision.action,
                        success=is_success,
                    )
                )

        logging.info(f"Calibration Pass Complete: ran {len(prompts)} prompts.")

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

            df = pd.DataFrame(results_list)
            df = df.sort_values(by="similarity", ascending=False)
            best_threshold = 1.0  # start with safest

            # iterate through candidate similarities - high to low,
            # stopping when precision drops below target
            similarity_candidates = df["similarity"].unique()
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
