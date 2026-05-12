import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# noqa for unused import - we actually need it to use style "science"
import scienceplots  # noqa: F401
import scipy.stats as stats

# plotting constants
FIG_SIZE_STANDARD = (10, 6)
FIG_SIZE_SMALL = (8, 5)
FIG_SIZE_PARETO = (9, 6)

SAMPLE_METRIC_MAPPING = {
    # standard metrics
    "avg_bleu": "bleu",
    "avg_rouge_l": "rouge",
    "avg_bert_score": "bert_score",
    "avg_token_accuracy": "token_accuracy",
    "avg_label_bleu": "label_bleu",
    "avg_label_rouge_l": "label_rouge",
    "avg_label_bert_score": "label_bert",
    "avg_label_token_accuracy": "label_token_accuracy",
    # map relative metrics to their raw NUMERATOR column
    "avg_relative_bleu": "label_bleu",
    "avg_relative_rouge_l": "label_rouge",
    "avg_relative_bert_score": "label_bert",
    "avg_relative_token_accuracy": "label_token_accuracy",
    # efficiency metrics
    "theoretical_speedup": "theoretical_speedup",
    "skipped_layer_percentage": "skipped_layer_percentage",
}

QUALITY_DISPLAY_NAMES = {
    "avg_bleu": "BLEU",
    "avg_rouge_l": "ROUGE-L",
    "avg_bert_score": "BERTScore",
    "avg_token_accuracy": "Token Accuracy",
    "avg_label_bleu": "Label BLEU",
    "avg_label_rouge_l": "Label ROUGE-L",
    "avg_label_bert_score": "Label BERTScore",
    "avg_label_token_accuracy": "Label Token Accuracy",
    "avg_relative_bleu": "Relative BLEU",
    "avg_relative_rouge_l": "Relative ROUGE-L",
    "avg_relative_bert_score": "Relative BERTScore",
    "avg_relative_token_accuracy": "Relative Token Acc",
}

EFFICIENCY_DISPLAY_NAMES = {
    "theoretical_speedup": "Acceleration Ratio",
    "skipped_layer_percentage": "Skipped Layer Percentage",
    "avg_skipped_per_token": "Avg Skipped Per Token",
}


def use_science_style():
    plt.style.use("science")

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            # fonts
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
            # global grid properties
            "axes.grid": True,
            "grid.alpha": 0.6,
            "grid.linestyle": "--",
        }
    )


def load_eval_results(
    results_dir: str, file_prefix: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scans a directory for JSON files matching the prefix.
    Returns:
      - df_agg: Aggregated metrics per threshold.
      - df_samples: Raw sample-level data across all thresholds.
    """
    records = []
    sample_records = []

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    for filename in os.listdir(results_dir):
        if filename.startswith(file_prefix) and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)

            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            config = data.get("config", {})
            metrics = data.get("metrics", {})
            thresholds = config.get("thresholds")

            if not thresholds:
                continue

            # check if all thresholds are the same (uniform)
            thresh_values = list(thresholds.values())
            if len(set(thresh_values)) != 1:
                logging.info(f"Skipping {filename} - Mixed thresholds detected.")
                continue

            uniform_threshold = float(thresh_values[0])

            # extract metrics
            acc_metrics = metrics.get("accuracy", {})
            eff_metrics = metrics.get("efficiency", {})
            avg_skipped_per_token = eff_metrics.get("avg_skipped_per_token", 0)

            # extract baseline metrics safely for ratio calculation
            b_bleu = acc_metrics["avg_baseline_label_bleu"]
            b_rouge = acc_metrics.get("avg_baseline_label_rouge_l", 0)
            b_bert = acc_metrics.get("avg_baseline_label_bert_score", 0)
            b_tok = acc_metrics.get("avg_baseline_label_token_accuracy", 0)

            # aggregate records
            records.append(
                {
                    "threshold": uniform_threshold,
                    # skipped vs baseline metrics
                    "avg_token_accuracy": acc_metrics.get("avg_token_accuracy", 0),
                    "avg_bleu": acc_metrics.get("avg_bleu", 0),
                    "avg_rouge_l": acc_metrics.get("avg_rouge_l", 0),
                    "avg_bert_score": acc_metrics.get("avg_bert_score", 0),
                    # skipped vs label metrics
                    "avg_label_bleu": acc_metrics.get("avg_label_bleu", 0),
                    "avg_label_rouge_l": acc_metrics.get("avg_label_rouge_l", 0),
                    "avg_label_bert_score": acc_metrics.get("avg_label_bert_score", 0),
                    "avg_label_token_accuracy": acc_metrics.get(
                        "avg_label_token_accuracy", 0
                    ),
                    # baseline vs label metrics
                    "avg_baseline_label_bleu": b_bleu,
                    "avg_baseline_label_rouge_l": b_rouge,
                    "avg_baseline_label_bert_score": b_bert,
                    "avg_baseline_label_token_accuracy": b_tok,
                    # relative quality metrics
                    "avg_relative_bleu": acc_metrics["avg_label_bleu"] / b_bleu,
                    "avg_relative_rouge_l": acc_metrics["avg_label_rouge_l"],
                    "avg_relative_bert_score": acc_metrics["avg_label_bert_score"],
                    "avg_relative_token_accuracy": acc_metrics[
                        "avg_label_token_accuracy"
                    ],
                    # efficiency metrics
                    "avg_skipped_per_token": avg_skipped_per_token,
                    "skipped_layer_percentage": avg_skipped_per_token * 100,
                    "theoretical_speedup": eff_metrics.get("theoretical_speedup", 1.0),
                    # tracking metrics
                    "checkpoint_skip_stats": eff_metrics.get(
                        "global_checkpoint_skip_counts", {}
                    ),
                    "db_hit_counts": eff_metrics.get("global_db_hit_counts", {}),
                    "db_index_sizes": eff_metrics.get("db_index_sizes", {}),
                    "token_skip_distribution": eff_metrics.get(
                        "global_token_skip_distribution", {}
                    ),
                    "request_skip_distribution": eff_metrics.get(
                        "global_request_skip_distribution", {}
                    ),
                }
            )

            samples_list = metrics.get("samples", [])
            total_skipped = sum(s["skipped_count"] for s in samples_list)
            total_tokens = sum(s["generated_token_count"] for s in samples_list)

            # avg_skipped_per_token = total_skipped / (n_layers * total_tokens)
            n_layers = round(total_skipped / (avg_skipped_per_token * total_tokens))

            # sample-level records (for scatter plots and CIs)
            for sample in samples_list:
                gen_tokens = sample.get("generated_token_count", 0)
                skipped_layers = sample.get("skipped_count", 0)

                # compute sample-specific efficiencies
                assert gen_tokens > 0 and n_layers > 0
                sample_skip_fraction = skipped_layers / (n_layers * gen_tokens)
                sample_speedup = 1.0 / (1.0 - sample_skip_fraction)
                sample_skip_pct = sample_skip_fraction * 100.0

                b_lbl_bleu = sample["baseline_label_bleu"]
                b_lbl_rouge = sample.get("baseline_label_rouge", 0)
                b_lbl_bert = sample.get("baseline_label_bert", 0)
                b_lbl_tok = sample.get("baseline_label_token_accuracy", 0)

                sample_records.append(
                    {
                        "threshold": uniform_threshold,
                        "sample_id": sample.get("id"),
                        "generated_token_count": gen_tokens,
                        "skipped_layers": skipped_layers,
                        # raw quality metrics
                        "bleu": sample.get("bleu", 0),
                        "rouge": sample.get("rouge", 0),
                        "bert_score": sample.get("bert_score", 0),
                        "token_accuracy": sample.get("token_accuracy", 0),
                        "label_bleu": sample.get("label_bleu", 0),
                        "label_rouge": sample.get("label_rouge", 0),
                        "label_bert": sample.get("label_bert", 0),
                        "label_token_accuracy": sample.get("label_token_accuracy", 0),
                        # raw relative metrics
                        "baseline_label_bleu": b_lbl_bleu,
                        "baseline_label_rouge": b_lbl_rouge,
                        "baseline_label_bert": b_lbl_bert,
                        "baseline_label_token_accuracy": b_lbl_tok,
                        # raw efficiency metrics
                        "theoretical_speedup": sample_speedup,
                        "skipped_layer_percentage": sample_skip_pct,
                        # per-sample skip acceptances
                        "checkpoint_skip_stats": sample.get(
                            "checkpoint_skip_stats", {}
                        ),
                    }
                )

    df_agg = (
        pd.DataFrame(records).sort_values(by="threshold").reset_index(drop=True)
        if records
        else pd.DataFrame()
    )
    df_samples = pd.DataFrame(sample_records) if sample_records else pd.DataFrame()

    return df_agg, df_samples


def calculate_grouped_ci(
    data: np.ndarray,
    baseline_data: np.ndarray = None,  # used in relative ratio calculations
    *,
    group_size: int = 10,
    ci_method: str = "t_dist",
    confidence: float = 0.95,
    n_sigma: int = 2,
):
    """Helper to compute grouped means and confidence intervals.
    If baseline_data is provided, computes the ratio of the grouped means."""

    # group the data to stabilise variance
    if group_size > 1:
        n_groups = len(data) // group_size
        if n_groups == 0:
            # fallback if we have fewer samples than the group size
            group_means = data
        else:
            # truncate to make perfectly sized groups
            truncated_data = data[: n_groups * group_size]
            groups = truncated_data.reshape(n_groups, group_size)
            group_means = np.mean(groups, axis=1)
    else:
        group_means = data

    # group the baseline data (if calculating a ratio)
    if baseline_data is not None:
        if group_size > 1:
            n_groups = len(baseline_data) // group_size
            if n_groups == 0:
                base_group_means = baseline_data
            else:
                truncated_base = baseline_data[: n_groups * group_size]
                base_groups = truncated_base.reshape(n_groups, group_size)
                base_group_means = np.mean(base_groups, axis=1)
        else:
            base_group_means = baseline_data

        # divide the grouped means, using a tiny epsilon to prevent strict zero division
        group_means = group_means / np.maximum(base_group_means, 1e-9)

    # calculate mean
    mean_val = np.mean(group_means)

    # cannot compute variance on less than 2 groups
    if len(group_means) < 2:
        return mean_val, 0.0

    if ci_method == "t_dist":
        sem = stats.sem(group_means)
        conf_bound = (1.0 + confidence) / 2.0
        ci = sem * stats.t.ppf(conf_bound, len(group_means) - 1)
    elif ci_method == "std":
        ci = n_sigma * np.std(group_means, ddof=1)
    else:
        ci = 0.0

    return mean_val, ci
