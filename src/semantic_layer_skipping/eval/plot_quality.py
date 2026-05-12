import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eval.utils import (
    EFFICIENCY_DISPLAY_NAMES,
    FIG_SIZE_PARETO,
    FIG_SIZE_STANDARD,
    QUALITY_DISPLAY_NAMES,
    SAMPLE_METRIC_MAPPING,
    calculate_grouped_ci,
)
from utils import PLOTS_DIR


def _compute_plot_metrics(
    df_samples: pd.DataFrame,
    thresholds: list,
    metric_name: str,
    sample_col: str,
    group_size: int,
    ci_method: str,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shared helper to compute means and CIs for a given metric across thresholds.
    Safely handles the macro-averaging logic for relative metrics.
    """
    means, cis = [], []
    for t in thresholds:
        raw_data = df_samples[df_samples["threshold"] == t][sample_col].values

        # if it's a relative metric, fetch baseline data for safe grouped division
        if metric_name.startswith("avg_relative_"):
            # get the baseline column (e.g. "label_bleu" -> "baseline_label_bleu")
            base_col = f"baseline_{sample_col}"
            raw_base = df_samples[df_samples["threshold"] == t][base_col].values
            mean_val, ci_val = calculate_grouped_ci(
                raw_data,
                baseline_data=raw_base,
                group_size=group_size,
                ci_method=ci_method,
                confidence=confidence,
            )
        else:
            # otherwise, get CI data directly
            mean_val, ci_val = calculate_grouped_ci(
                raw_data,
                group_size=group_size,
                ci_method=ci_method,
                confidence=confidence,
            )

        means.append(mean_val)
        cis.append(ci_val)

    return np.array(means), np.array(cis)


def plot_threshold_sensitivity(
    df_agg: pd.DataFrame,
    df_samples: pd.DataFrame,
    *,
    quality_metric="avg_token_accuracy",
    efficiency_metric="theoretical_speedup",
    root_plot_dir: str = PLOTS_DIR,
    group_size: int = 1,
    ci_method: str = "t_dist",
    confidence: float = 0.95,
):
    """
    Plots Threshold (X) vs Quality (Left Y) with CIs, and Efficiency (Right Y) with CIs.
    """
    fig, ax1 = plt.subplots(figsize=FIG_SIZE_STANDARD)

    sample_col_q = SAMPLE_METRIC_MAPPING.get(quality_metric, quality_metric)
    sample_col_e = SAMPLE_METRIC_MAPPING.get(efficiency_metric, efficiency_metric)

    display_qual_metric = QUALITY_DISPLAY_NAMES.get(
        quality_metric, quality_metric.replace("avg_", "").replace("_", " ").title()
    )
    display_eff_metric = EFFICIENCY_DISPLAY_NAMES.get(
        efficiency_metric, efficiency_metric.replace("_", " ").title()
    )

    thresholds = sorted(df_agg["threshold"].unique())

    means_q, cis_q = _compute_plot_metrics(
        df_samples,
        thresholds,
        quality_metric,
        sample_col_q,
        group_size,
        ci_method,
        confidence,
    )
    means_e, cis_e = _compute_plot_metrics(
        df_samples,
        thresholds,
        efficiency_metric,
        sample_col_e,
        group_size,
        ci_method,
        confidence,
    )

    # primary y-axis for quality
    color1 = "tab:blue"
    ax1.set_xlabel(r"\textbf{Checkpoint Similarity Threshold}")
    ax1.set_ylabel(rf"\textbf{{{display_qual_metric}}}", color=color1)

    ax1.plot(
        thresholds,
        means_q,
        color=color1,
        marker="o",
        linewidth=2,
        label=display_qual_metric,
    )

    if ci_method != "none" and np.any(cis_q > 0):
        ax1.fill_between(
            thresholds, means_q - cis_q, means_q + cis_q, color=color1, alpha=0.2
        )

    ax1.tick_params(axis="y", labelcolor=color1)

    # secondary y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel(rf"\textbf{{{display_eff_metric}}}", color=color2)

    ax2.plot(
        thresholds,
        means_e,
        color=color2,
        marker="s",
        linewidth=2,
        linestyle="--",
        label=display_eff_metric,
    )

    if ci_method != "none" and np.any(cis_e > 0):
        ax2.fill_between(
            thresholds, means_e - cis_e, means_e + cis_e, color=color2, alpha=0.2
        )

    ax2.tick_params(axis="y", labelcolor=color2)

    # don't show grid lines for the secondary y-axis to avoid clutter
    ax2.grid(False)

    plt.title(r"\textbf{Thresholding Impact: Quality vs. Efficiency}")
    fig.tight_layout()

    # combine legends cleanly
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center left")

    # save
    plot_dir = os.path.join(root_plot_dir, "threshold_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(
        plot_dir, f"threshold_sensitivity_{quality_metric}_{efficiency_metric}.pdf"
    )
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved threshold sensitivity plot with CIs to {plot_path}")


def plot_pareto_frontier(
    df_agg: pd.DataFrame,
    df_samples: pd.DataFrame,
    *,
    quality_metric="avg_token_accuracy",
    efficiency_metric="theoretical_speedup",
    root_plot_dir: str = PLOTS_DIR,
    group_size: int = 1,
    ci_method: str = "t_dist",
    confidence: float = 0.95,
    label_interval: int = 2,
):
    """
    Plots Efficiency (X) vs Quality (Y) to show the Pareto frontier with error bars.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_PARETO)

    sample_col_q = SAMPLE_METRIC_MAPPING.get(quality_metric, quality_metric)
    sample_col_e = SAMPLE_METRIC_MAPPING.get(efficiency_metric, efficiency_metric)

    display_qual_metric = QUALITY_DISPLAY_NAMES.get(
        quality_metric, quality_metric.replace("avg_", "").replace("_", " ").title()
    )
    display_eff_metric = EFFICIENCY_DISPLAY_NAMES.get(
        efficiency_metric, efficiency_metric.replace("_", " ").title()
    )

    thresholds = sorted(df_agg["threshold"].unique())

    y_means, y_errs = _compute_plot_metrics(
        df_samples,
        thresholds,
        quality_metric,
        sample_col_q,
        group_size,
        ci_method,
        confidence,
    )
    x_means, x_errs = _compute_plot_metrics(
        df_samples,
        thresholds,
        efficiency_metric,
        sample_col_e,
        group_size,
        ci_method,
        confidence,
    )

    # plot styling constants
    pareto_line_color = "purple"
    errorbar_color = "mediumorchid"
    errorbar_alpha = 0.6
    line_alpha = 0.9
    point_size = 60
    label_xytext_offset = (2.3, 2.3)

    # plot error bars
    ax.errorbar(
        x_means,
        y_means,
        xerr=x_errs,
        yerr=y_errs,
        fmt="none",
        color=errorbar_color,
        capsize=4,
        elinewidth=1.5,
        alpha=errorbar_alpha,
        zorder=3,
    )

    # plot the main connecting line
    ax.plot(
        x_means,
        y_means,
        color=pareto_line_color,
        linestyle="-",
        linewidth=2.0,
        alpha=line_alpha,
        zorder=4,
    )

    # plot centroids
    ax.scatter(
        x_means,
        y_means,
        color=pareto_line_color,
        s=point_size,
        alpha=line_alpha,
        zorder=5,
    )

    # annotations
    for i, t in enumerate(thresholds):
        if i % label_interval == 0:
            ax.annotate(
                rf"\textbf{{T: {t}}}",
                (x_means[i], y_means[i]),
                textcoords="offset points",
                xytext=label_xytext_offset,
                ha="left",
                va="bottom",
                fontsize=10,
                zorder=6,
            )

    ax.set_title(r"\textbf{Pareto Frontier: Efficiency vs. Quality}")
    ax.set_xlabel(rf"\textbf{{{display_eff_metric}}}")
    ax.set_ylabel(rf"\textbf{{{display_qual_metric}}}")

    fig.tight_layout()
    plot_dir = os.path.join(root_plot_dir, "threshold_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(
        plot_dir, f"pareto_front_errorbars_{quality_metric}_{efficiency_metric}.pdf"
    )
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved Pareto frontier with error bars plot to {plot_path}")


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
    plot_path = os.path.join(plot_dir, f"baseline_vs_skipped_{clean_name}.pdf")

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
    plot_path = os.path.join(plot_dir, f"scale_factor_{clean_name}.pdf")

    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved Scale Factor plot for {metric_display_name} to {plot_path}")
