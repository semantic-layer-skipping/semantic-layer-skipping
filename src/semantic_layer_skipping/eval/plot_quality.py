import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eval.plot_loader import FIG_SIZE_PARETO, FIG_SIZE_STANDARD
from utils import PLOTS_DIR


def plot_threshold_sensitivity(
    df: pd.DataFrame,
    quality_metric="avg_token_accuracy",
    efficiency_metric="skipped_layer_percentage",
    root_plot_dir: str = PLOTS_DIR,
):
    """
    Plots Threshold (X) vs Quality (Left Y) and Efficiency (Right Y).
    """
    fig, ax1 = plt.subplots(figsize=FIG_SIZE_STANDARD)

    # primary y-axis for quality
    color1 = "tab:blue"
    ax1.set_xlabel(r"\textbf{Cosine Similarity Threshold}")
    ax1.set_ylabel(
        rf"\textbf{{{quality_metric.replace('_', ' ').title()}}}", color=color1
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
        rf"\textbf{{{efficiency_metric.replace('_', ' ').title()}}}", color=color2
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

    plt.title(r"\textbf{Uniform Thresholding Impact: Quality vs. Efficiency}")
    fig.tight_layout()

    # combine legends
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center left")

    plot_dir = os.path.join(root_plot_dir, "threshold_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(
        plot_dir, f"threshold_sensitivity_{quality_metric}_{efficiency_metric}.png"
    )
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved threshold sensitivity plot to {plot_path}")


def plot_pareto_front(
    df: pd.DataFrame,
    quality_metric="avg_token_accuracy",
    efficiency_metric="avg_skipped_per_token",
    root_plot_dir: str = PLOTS_DIR,
):
    """
    Plots Efficiency (X) vs Quality (Y) to show the Pareto Front.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_PARETO)

    ax.scatter(
        df[efficiency_metric], df[quality_metric], color="purple", s=100, zorder=5
    )
    ax.plot(
        df[efficiency_metric],
        df[quality_metric],
        color="gray",
        linestyle="-",
        alpha=0.6,
        zorder=4,
    )

    for _, row in df.iterrows():
        ax.annotate(
            f"T: {row['threshold']}",
            (row[efficiency_metric], row[quality_metric]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=11,
        )

    ax.set_title(r"\textbf{Uniform Thresholding Pareto Front: Efficiency vs. Quality}")
    ax.set_xlabel(rf"\textbf{{{efficiency_metric.replace('_', ' ').title()}}}")
    ax.set_ylabel(rf"\textbf{{{quality_metric.replace('_', ' ').title()}}}")

    fig.tight_layout()
    plot_dir = os.path.join(root_plot_dir, "threshold_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(
        plot_dir, f"pareto_front_{quality_metric}_{efficiency_metric}.png"
    )
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved Pareto front plot to {plot_path}")


def plot_baseline_vs_skipped_quality(
    df: pd.DataFrame,
    skipped_metric: str,
    baseline_metric: str,
    metric_display_name: str,
    x_axis_metric: str = "skipped_layer_percentage",
    root_plot_dir: str = PLOTS_DIR,
):
    """
    Plots Baseline vs Skipped Model quality against an efficiency axis
    to visualise degradation.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)

    ax.plot(
        df[x_axis_metric],
        df[skipped_metric],
        marker="o",
        linewidth=2.5,
        color="tab:blue",
        label="Skipped Model",
    )
    ax.plot(
        df[x_axis_metric],
        df[baseline_metric],
        marker="x",
        linewidth=2.5,
        linestyle="--",
        color="tab:orange",
        label="Baseline Model",
    )

    ax.set_xlabel(r"\textbf{Skipped Layer Percentage (\%)}")
    ax.set_ylabel(rf"\textbf{{{metric_display_name}}}")
    ax.set_title(rf"\textbf{{Baseline vs. Skipped Generation: {metric_display_name}}}")

    fig.tight_layout()
    plot_dir = os.path.join(root_plot_dir, "label_comparisons")
    os.makedirs(plot_dir, exist_ok=True)

    clean_name = metric_display_name.replace(" ", "_").replace("-", "").lower()
    plot_path = os.path.join(plot_dir, f"baseline_vs_skipped_{clean_name}.png")

    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved Baseline vs Skipped comparison plot to {plot_path}")


def plot_quality_scale_factor(
    df: pd.DataFrame,
    skipped_metric: str,
    baseline_metric: str,
    metric_display_name: str,
    x_axis_metric: str = "skipped_layer_percentage",
    root_plot_dir: str = PLOTS_DIR,
):
    """
    Plots the scale factor (Skipped / Baseline) against an efficiency axis
    to visualise relative degradation.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)

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

    ax.set_xlabel(r"\textbf{Skipped Layer Percentage (\%)}")
    ax.set_ylabel(r"\textbf{Scale Factor (Skipped / Baseline)}")
    ax.set_title(
        rf"\textbf{{Proportion of Baseline Quality Retained: {metric_display_name}}}"
    )
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    plot_dir = os.path.join(root_plot_dir, "label_comparisons")
    os.makedirs(plot_dir, exist_ok=True)

    clean_name = metric_display_name.replace(" ", "_").replace("-", "").lower()
    plot_path = os.path.join(plot_dir, f"scale_factor_{clean_name}.png")

    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved Scale Factor plot for {metric_display_name} to {plot_path}")
