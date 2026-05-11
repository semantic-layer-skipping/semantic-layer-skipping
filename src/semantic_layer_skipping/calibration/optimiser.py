import json
import logging
import os

import numpy as np
import optuna
from data.loader import BatchedDataset
from experiment.config import EvalConfig
from experiment.evaluator import run_eval_loop
from experiment.manager import ExperimentManager
from inference.torch_runner import TorchSkipRunner
from store import SkippingVectorDB


class E2EOptimiser:
    def __init__(
        self,
        runner: TorchSkipRunner,
        db: SkippingVectorDB,
        eval_config: EvalConfig,
        dataset: BatchedDataset,
        manager: ExperimentManager,
        run_name: str,
        acc_metric: str = "bert_label_ratio",
        eff_metric: str = "theoretical_speedup",
        threshold_lower_bound: float = 0.70,
        threshold_upper_bound: float = 1.00,
    ):
        self.runner = runner
        self.db = db
        self.config = eval_config
        self.dataset = dataset
        self.manager = manager
        self.run_name = run_name

        self.acc_metric = acc_metric
        self.eff_metric = eff_metric
        self.threshold_lower_bound = threshold_lower_bound
        self.threshold_upper_bound = threshold_upper_bound

        self.num_checkpoints = len(self.runner.checkpoints)
        logging.info(
            f"E2E Optimiser initialised for {self.num_checkpoints} checkpoints."
        )

        self.opt_path = self.manager.get_e2e_optimisation_path(run_name)
        os.makedirs(self.opt_path, exist_ok=True)
        self.db_url = f"sqlite:///{os.path.join(self.opt_path, 'optuna_study.db')}"

        # initialise and handle caching immediately
        self.precomputed_baselines = self._load_or_compute_baselines()

    def _load_or_compute_baselines(self) -> dict:
        """Loads baselines from cache, or runs a full baseline pass to generate them."""
        cache_path = os.path.join(self.opt_path, "precomputed_baselines.json")

        if os.path.exists(cache_path):
            logging.info(f"Loading precomputed baselines from {cache_path}")
            with open(cache_path) as f:
                return json.load(f)

        logging.info(
            "Precomputed baselines not found. "
            "Running initial baseline generation pass..."
        )

        # run eval_loop with db=None to force pure baseline generation
        # for the whole dataset
        baseline_summary = run_eval_loop(
            self.runner,
            db=None,
            thresholds=None,
            config=self.config,
            dataset=self.dataset,
            eval_bert=True,
        )

        # reformat the samples into a lookup dictionary
        baselines = {}
        for sample_data in baseline_summary["samples"]:
            baselines[sample_data["id"]] = {
                "baseline_text": sample_data.get(
                    "baseline_text", sample_data.get("generated_text")
                ),
                "baseline_tokens": sample_data.get(
                    "baseline_tokens", sample_data.get("generated_tokens", [])
                ),
                "baseline_extracted_answer": sample_data.get(
                    "baseline_extracted_answer", sample_data.get("extracted_answer")
                ),
                "baseline_is_correct": sample_data.get(
                    "baseline_is_correct", sample_data.get("is_correct", False)
                ),
                "baseline_label_bleu": sample_data.get(
                    "baseline_label_bleu", sample_data.get("label_bleu", 0.0)
                ),
                "baseline_label_rouge": sample_data.get(
                    "baseline_label_rouge", sample_data.get("label_rouge", 0.0)
                ),
                "baseline_label_bert": sample_data.get(
                    "baseline_label_bert", sample_data.get("label_bert", 0.0)
                ),
                "baseline_label_token_accuracy": sample_data.get(
                    "baseline_label_token_accuracy",
                    sample_data.get("label_token_accuracy", 0.0),
                ),
            }

        # save to disk
        with open(cache_path, "w") as f:
            json.dump(baselines, f, indent=4)

        logging.info(f"Successfully cached {len(baselines)} baselines to {cache_path}")
        return baselines

    def _seed_anchor_trials(self, study: optuna.Study):
        # seed with uniform thresholds
        anchors = np.arange(0.86, 1.00, 0.02)
        logging.info(f"Ensuring {len(anchors)} uniform anchor trials are queued...")
        for val in anchors:
            val = round(val, 2)
            # ensure we don't accidentally seed below the lower bound
            val = max(self.threshold_lower_bound, min(self.threshold_upper_bound, val))
            trial_params = {f"T{i}": val for i in range(self.num_checkpoints)}
            study.enqueue_trial(trial_params, skip_if_exists=True)

    def objective(self, trial: optuna.Trial):
        thresholds = {}
        for i in range(self.num_checkpoints):
            thresholds[i] = trial.suggest_float(
                f"T{i}", self.threshold_lower_bound, self.threshold_upper_bound
            )
        logging.info(f"Trial {trial.number} testing thresholds: {thresholds}")

        summary = run_eval_loop(
            self.runner,
            self.db,
            thresholds,
            self.config,
            self.dataset,
            precomputed_baselines=self.precomputed_baselines,
        )

        base_bert = summary["accuracy"].get("avg_baseline_label_bert_score", 0.0)
        skip_bert = summary["accuracy"].get("avg_label_bert_score", 0.0)
        bert_ratio = (skip_bert / base_bert) if base_bert != 0 else 0.0
        summary["accuracy"]["bert_label_ratio"] = bert_ratio

        # target metrics
        accuracy_score = summary["accuracy"][self.acc_metric]
        efficiency_score = summary["efficiency"][self.eff_metric]

        trial_data = {
            "trial_id": trial.number,
            "thresholds": thresholds,
            "accuracy_score": accuracy_score,
            "efficiency_score": efficiency_score,
            "full_summary": summary,
        }
        self.manager.save_e2e_trial_result(self.run_name, trial.number, trial_data)

        return accuracy_score, efficiency_score

    def optimise(self, n_trials: int = 50):
        # NSGAIISampler handles Multi-Objective optimisation with Pareto front
        study = optuna.create_study(
            study_name=self.run_name,
            storage=self.db_url,
            load_if_exists=True,
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler(),
        )

        self._seed_anchor_trials(study)

        logging.info(f"Executing E2E Optimisation for {n_trials} trials.")
        study.optimize(self.objective, n_trials=n_trials)

        self._export_pareto_front(study)

    def _export_pareto_front(self, study):
        try:
            pareto_trials = study.best_trials
        except ValueError:
            logging.warning("No Pareto front found yet.")
            return

        raw_results = []
        for t in pareto_trials:
            raw_results.append(
                {
                    "trial_id": t.number,
                    "thresholds": t.params,
                    "accuracy_score": t.values[0],
                    "efficiency_score": t.values[1],
                }
            )

        raw_results = sorted(
            raw_results, key=lambda x: x["efficiency_score"], reverse=True
        )

        lookup_table = {}
        # the ratio might exceed 1.0, so upper bound is 1.06
        # TODO: this range assumes we are optimising the ratio
        for target_acc in np.arange(0.10, 1.06, 0.05):
            target_acc = round(target_acc, 2)

            valid_configs = [
                res for res in raw_results if res["accuracy_score"] >= target_acc
            ]

            if valid_configs:
                best_config = valid_configs[0]
                lookup_table[str(target_acc)] = {
                    "achieved_accuracy": best_config["accuracy_score"],
                    "achieved_efficiency": best_config["efficiency_score"],
                    "optimal_thresholds": best_config["thresholds"],
                }
            else:
                lookup_table[str(target_acc)] = "UNACHIEVABLE"

        self.manager.save_e2e_pareto_front(self.run_name, raw_results, lookup_table)
