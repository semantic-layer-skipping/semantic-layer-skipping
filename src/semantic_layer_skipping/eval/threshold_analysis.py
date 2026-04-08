import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import PLOTS_DIR, set_logging_config


def load_uniform_threshold_results(results_dir: str, file_prefix: str) -> pd.DataFrame:
    """
    Scans a directory for JSON files matching the prefix, extracts metrics for
    uniform threshold experiments, and returns a sorted Pandas DataFrame.
    """
    records = []

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    for filename in os.listdir(results_dir):
        if filename.startswith(file_prefix) and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)

            with open(filepath) as f:
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
                }
            )

    df = pd.DataFrame(records)
    if not df.empty:
        # sort by threshold ascending threshold
        df = df.sort_values(by="threshold").reset_index(drop=True)

    return df


def plot_threshold_sensitivity(
    df: pd.DataFrame,
    quality_metric="avg_token_accuracy",
    efficiency_metric="skipped_layer_percentage",
):
    """
    Plots Threshold (X) vs Quality (Left Y) and Efficiency (Right Y).
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.grid(True, linestyle="--", alpha=0.6)

    # primary y-axis for quality
    color1 = "tab:blue"
    ax1.set_xlabel("Cosine Similarity Threshold", fontsize=12, fontweight="bold")
    ax1.set_ylabel(
        quality_metric.replace("_", " ").title(),
        color=color1,
        fontsize=12,
        fontweight="bold",
    )
    (line1,) = ax1.plot(
        df["threshold"],
        df[quality_metric],
        color=color1,
        marker="o",
        linewidth=2,
        label=quality_metric,
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    # secondary y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel(
        efficiency_metric.replace("_", " ").title(),
        color=color2,
        fontsize=12,
        fontweight="bold",
    )
    (line2,) = ax2.plot(
        df["threshold"],
        df[efficiency_metric],
        color=color2,
        marker="s",
        linewidth=2,
        linestyle="--",
        label=efficiency_metric,
    )
    ax2.tick_params(axis="y", labelcolor=color2)
    # don't show grid lines for the secondary y-axis to avoid clutter
    ax2.grid(False)

    plt.title(
        "Uniform Thresholding Impact: Quality vs. Efficiency",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    # combine legends
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)

    plot_dir = os.path.join(PLOTS_DIR, "threshold_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(
        plot_dir, f"threshold_sensitivity_{quality_metric}_{efficiency_metric}.png"
    )
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved threshold sensitivity plot to {plot_path}")


def plot_pareto_front(
    df: pd.DataFrame,
    quality_metric="avg_token_accuracy",
    efficiency_metric="avg_skipped_per_token",
):
    """
    Plots Efficiency (X) vs Quality (Y) to show the Pareto Front.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.grid(True, linestyle="--", alpha=0.6)

    # scatter plot
    ax.scatter(
        df[efficiency_metric], df[quality_metric], color="purple", s=100, zorder=5
    )

    # connect dots
    ax.plot(
        df[efficiency_metric],
        df[quality_metric],
        color="gray",
        linestyle="-",
        alpha=0.6,
        zorder=4,
    )

    # annotate each point with its threshold value
    for _, row in df.iterrows():
        ax.annotate(
            f"T: {row['threshold']}",
            (row[efficiency_metric], row[quality_metric]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    ax.set_title(
        "Uniform Thresholding Pareto Front: Efficiency vs. Quality",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(efficiency_metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(quality_metric.replace("_", " ").title(), fontsize=12)

    fig.tight_layout()
    plot_dir = os.path.join(PLOTS_DIR, "threshold_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(
        plot_dir, f"pareto_front_{quality_metric}_{efficiency_metric}.png"
    )
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved Pareto front plot to {plot_path}")


def plot_baseline_vs_skipped_quality(
    df: pd.DataFrame,
    skipped_metric: str,
    baseline_metric: str,
    metric_display_name: str,
    x_axis_metric: str = "skipped_layer_percentage",
):
    """
    Plots Baseline vs Skipped Model quality against an efficiency axis
    to visualise degradation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, linestyle="--", alpha=0.6)

    # plot skipped model performance
    ax.plot(
        df[x_axis_metric],
        df[skipped_metric],
        marker="o",
        linewidth=2.5,
        color="tab:blue",
        label="Skipped Model",
    )

    # plot baseline model performance
    ax.plot(
        df[x_axis_metric],
        df[baseline_metric],
        marker="x",
        linewidth=2.5,
        linestyle="--",
        color="tab:orange",
        label="Baseline Model",
    )

    ax.set_xlabel("Skipped Layer Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_display_name, fontsize=12, fontweight="bold")
    ax.set_title(
        f"Baseline vs. Skipped Generation: {metric_display_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)

    fig.tight_layout()
    plot_dir = os.path.join(PLOTS_DIR, "label_comparisons")
    os.makedirs(plot_dir, exist_ok=True)

    clean_name = metric_display_name.replace(" ", "_").replace("-", "").lower()
    plot_path = os.path.join(plot_dir, f"baseline_vs_skipped_{clean_name}.png")

    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved Baseline vs Skipped comparison plot to {plot_path}")


def plot_quality_scale_factor(
    df: pd.DataFrame,
    skipped_metric: str,
    baseline_metric: str,
    metric_display_name: str,
    x_axis_metric: str = "skipped_layer_percentage",
):
    """
    Plots the scale factor (Skipped / Baseline) against an efficiency axis
    to visualise relative degradation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, linestyle="--", alpha=0.6)

    # calculate the scale factor (avoiding division by zero)
    # if baseline is 0, we set the scale factor to NaN so it doesn't plot a broken point
    safe_baseline = df[baseline_metric].replace(0, np.nan)
    scale_factor = df[skipped_metric] / safe_baseline

    # plot the skipped model's relative performance
    ax.plot(
        df[x_axis_metric],
        scale_factor,
        marker="o",
        linewidth=2.5,
        color="tab:blue",
        label=f"{metric_display_name} (Relative)",
    )

    # plot the baseline reference line at 1.0
    ax.axhline(
        y=1.0,
        color="tab:orange",
        linestyle="--",
        linewidth=2.5,
        label="Baseline Reference (1.0x)",
    )

    ax.set_xlabel("Skipped Layer Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Scale Factor (Skipped / Baseline)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Proportion of Baseline Quality Retained: {metric_display_name}\n",
        fontsize=14,
        fontweight="bold",
    )

    ax.legend(fontsize=11)

    # set y-axis bottom limit to 0
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    plot_dir = os.path.join(PLOTS_DIR, "label_comparisons")
    os.makedirs(plot_dir, exist_ok=True)

    clean_name = metric_display_name.replace(" ", "_").replace("-", "").lower()
    plot_path = os.path.join(plot_dir, f"scale_factor_{clean_name}.png")

    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved Scale Factor plot for {metric_display_name} to {plot_path}")


if __name__ == "__main__":
    set_logging_config()

    # results dir
    # RESULTS_DIR = get_experiment_output_dir() + "/batch_20260310_155736_Qwen2.5-1.5B-Instruct_newton_train_3s_50t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results" # noqa: E501
    # RESULTS_DIR = get_experiment_output_dir() + "/batch_20260309_042303_Qwen2.5-1.5B-Instruct_sharegpt_train_20000s_2048t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results"  # noqa: E501
    # RESULTS_DIR = "hpc/experiments/batch_20260309_042303_Qwen2.5-1.5B-Instruct_sharegpt_train_20000s_2048t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results_256tokens_50samples_fixed_thresh"  # noqa: E501
    RESULTS_DIR = "hpc/experiments/batch_20260309_042303_Qwen2.5-1.5B-Instruct_sharegpt_train_20000s_2048t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results_ivfpq1"  # noqa: E501

    # prefix of files to analyse
    # PREFIX = "sharegpt_test_10s_25t_full_generation_thresh-"
    # PREFIX = "sharegpt_test_100s_256t_full_generation_thresh-"
    PREFIX = "sharegpt_test"

    logging.info("Loading data... from %s with prefix %s", RESULTS_DIR, PREFIX)
    df = load_uniform_threshold_results(RESULTS_DIR, PREFIX)

    if df.empty:
        logging.warning("No valid data found. Check your directory path and prefix.")
    else:
        logging.info(f"Loaded {len(df)} experiment configurations.")

        # standard generation plots (evaluate skipped against baseline)
        standard_metrics = [
            "avg_token_accuracy",
            "avg_bleu",
            "avg_rouge_l",
            "avg_bert_score",
        ]
        for metric in standard_metrics:
            plot_threshold_sensitivity(
                df, quality_metric=metric, efficiency_metric="skipped_layer_percentage"
            )
            plot_pareto_front(
                df, quality_metric=metric, efficiency_metric="skipped_layer_percentage"
            )

        # evaluate against ground-truth
        comparison_pairs = [
            ("avg_label_bleu", "avg_baseline_label_bleu", "BLEU Score vs Label"),
            (
                "avg_label_rouge_l",
                "avg_baseline_label_rouge_l",
                "ROUGE-L Score vs Label",
            ),
            (
                "avg_label_bert_score",
                "avg_baseline_label_bert_score",
                "BERTScore F1 vs Label",
            ),
            (
                "avg_label_token_accuracy",
                "avg_baseline_label_token_accuracy",
                "Token Accuracy vs Label",
            ),
        ]

        for skipped_metric, baseline_metric, display_name in comparison_pairs:
            plot_baseline_vs_skipped_quality(
                df,
                skipped_metric=skipped_metric,
                baseline_metric=baseline_metric,
                metric_display_name=display_name,
            )
            plot_quality_scale_factor(
                df,
                skipped_metric=skipped_metric,
                baseline_metric=baseline_metric,
                metric_display_name=display_name,
            )
