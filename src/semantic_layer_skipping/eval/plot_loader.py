import json
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

# noqa for unused import - we actually need it to use style "science"
import scienceplots  # noqa: F401

# plotting constants
FIG_SIZE_STANDARD = (10, 6)
FIG_SIZE_SMALL = (8, 5)
FIG_SIZE_PARETO = (9, 6)


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
                    "avg_baseline_label_bleu": acc_metrics.get(
                        "avg_baseline_label_bleu", 0
                    ),
                    "avg_baseline_label_rouge_l": acc_metrics.get(
                        "avg_baseline_label_rouge_l", 0
                    ),
                    "avg_baseline_label_bert_score": acc_metrics.get(
                        "avg_baseline_label_bert_score", 0
                    ),
                    "avg_baseline_label_token_accuracy": acc_metrics.get(
                        "avg_baseline_label_token_accuracy", 0
                    ),
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

            # sample-level records (for scatter plots)
            for sample in metrics["samples"]:
                sample_records.append(
                    {
                        "threshold": uniform_threshold,
                        "sample_id": sample.get("id"),
                        "generated_token_count": sample["generated_token_count"],
                        "skipped_layers": sample.get("skipped_count", 0),
                    }
                )

    df_agg = (
        pd.DataFrame(records).sort_values(by="threshold").reset_index(drop=True)
        if records
        else pd.DataFrame()
    )
    df_samples = pd.DataFrame(sample_records) if sample_records else pd.DataFrame()

    return df_agg, df_samples
