import json
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from utils import PLOTS_DIR, get_experiment_output_dir, set_logging_config


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

            records.append(
                {
                    "threshold": uniform_threshold,
                    "avg_token_accuracy": acc_metrics.get("avg_token_accuracy", 0),
                    "avg_bleu": acc_metrics.get("avg_bleu", 0),
                    "avg_rouge_l": acc_metrics.get("avg_rouge_l", 0),
                    "avg_skipped_per_token": eff_metrics.get(
                        "avg_skipped_per_token", 0
                    ),
                    "skipped_layer_percentage": eff_metrics.get(
                        "avg_skipped_per_token", 0
                    )
                    * 100,
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

    plot_dir = PLOTS_DIR + "/threshold_analysis/"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(
        plot_dir + f"threshold_sensitivity_{quality_metric}_{efficiency_metric}.png",
        dpi=300,
    )


def plot_pareto_front(
    df: pd.DataFrame,
    quality_metric="avg_token_accuracy",
    efficiency_metric="avg_skipped_per_token",
):
    """
    Plots Efficiency (X) vs Quality (Y) to show the Pareto Front.
    """
    plt.figure(figsize=(9, 6))
    plt.grid(True, linestyle="--", alpha=0.6)

    # scatter plot
    plt.scatter(
        df[efficiency_metric], df[quality_metric], color="purple", s=100, zorder=5
    )

    # connect dots
    plt.plot(
        df[efficiency_metric],
        df[quality_metric],
        color="gray",
        linestyle="-",
        alpha=0.6,
        zorder=4,
    )

    # annotate each point with its threshold value
    for _, row in df.iterrows():
        plt.annotate(
            f"T: {row['threshold']:.2f}",
            (row[efficiency_metric], row[quality_metric]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    plt.title(
        "Uniform Thresholding Pareto Front: Efficiency vs. Quality",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel(efficiency_metric, fontsize=12)
    plt.ylabel(quality_metric, fontsize=12)

    plt.tight_layout()
    plot_dir = PLOTS_DIR + "/threshold_analysis/"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(
        plot_dir + f"pareto_front_{quality_metric}_{efficiency_metric}.png", dpi=300
    )


if __name__ == "__main__":
    set_logging_config()

    # results dir
    # RESULTS_DIR = get_experiment_output_dir() + "/batch_20260310_155736_Qwen2.5-1.5B-Instruct_newton_train_3s_50t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results" # noqa: E501
    RESULTS_DIR = (
        get_experiment_output_dir()
        + "/batch_20260309_042303_Qwen2.5-1.5B-Instruct_sharegpt_train_20000s_2048t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results"  # noqa: E501
    )

    # prefix of files to analyse
    # PREFIX = "sharegpt_test_10s_25t_full_generation_thresh-"
    PREFIX = "sharegpt_test_100s_256t_full_generation_thresh-"

    logging.info("Loading data...")
    df = load_uniform_threshold_results(RESULTS_DIR, PREFIX)

    if df.empty:
        logging.warning("No valid data found. Check your directory path and prefix.")
    else:
        logging.info(f"Loaded {len(df)} experiment configurations.")
        logging.info(f"df\n{df}")

        for quality_metric in ["avg_token_accuracy", "avg_bleu", "avg_rouge_l"]:
            logging.info("Plotting Threshold Sensitivity...")
            plot_threshold_sensitivity(
                df,
                quality_metric=quality_metric,
                efficiency_metric="skipped_layer_percentage",
            )

            logging.info("Plotting Pareto Front...")
            plot_pareto_front(
                df,
                quality_metric=quality_metric,
                efficiency_metric="skipped_layer_percentage",
            )
