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
    param_vals: list,
    param_col: str,
    metric_name: str,
    sample_col: str,
    group_size: int,
    ci_method: str,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shared helper to compute means and CIs for a given metric across a parameter axis.
    Safely handles the macro-averaging logic for relative metrics.
    """
    means, cis = [], []
    for val in param_vals:
        raw_data = df_samples[df_samples[param_col] == val][sample_col].values

        # if it's a relative metric, fetch baseline data for safe grouped division
        if metric_name.startswith("avg_relative_"):
            # get the baseline column (e.g. "label_bleu" -> "baseline_label_bleu")
            base_col = f"baseline_{sample_col}"
            raw_base = df_samples[df_samples[param_col] == val][base_col].values
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
        "threshold",
        quality_metric,
        sample_col_q,
        group_size,
        ci_method,
        confidence,
    )
    means_e, cis_e = _compute_plot_metrics(
        df_samples,
        thresholds,
        "threshold",
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
    baselines: list = None,
    main_method_name: str = "Retrieval-Guided Skip",
    *,
    quality_metric="avg_token_accuracy",
    efficiency_metric="theoretical_speedup",
    root_plot_dir: str = PLOTS_DIR,
    group_size: int = 1,
    ci_method: str = "t_dist",
    confidence: float = 0.95,
    label_interval: int = 2,
    include_no_skip_point: bool = True,
):
    """
    Plots Efficiency (X) vs Quality (Y) to show the Pareto frontier with error bars,
    supporting baselines and a no-skip point.
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

    datasets_to_plot = []

    # plot baselines first
    baseline_colors = [
        ("tab:orange", "navajowhite"),
        ("tab:green", "lightgreen"),
        ("tab:red", "lightcoral"),
    ]
    if baselines:
        for idx, b in enumerate(baselines):
            c_main, c_err = baseline_colors[idx % len(baseline_colors)]
            datasets_to_plot.append(
                {
                    "display_name": b["display_name"],
                    "df_agg": b["df_agg"],
                    "df_samples": b["df_samples"],
                    "param_key": b["param_key"],
                    "label_prefix": b["label_prefix"],
                    "color": c_main,
                    "error_color": c_err,
                    "marker": "s",
                    "show_labels": b.get("show_labels", False),
                    "z_base": 2,
                }
            )

    # add the main method
    datasets_to_plot.append(
        {
            "display_name": main_method_name,
            "df_agg": df_agg,
            "df_samples": df_samples,
            "param_key": "threshold",
            "label_prefix": "T",
            "color": "purple",
            "error_color": "plum",
            "marker": "o",
            "show_labels": True,
            "z_base": 10,
        }
    )

    # track these for tight x-axis bounds after plotting all datasets
    global_max_x = -np.inf
    global_min_x = np.inf
    for ds in datasets_to_plot:
        param_col = ds["param_key"]
        param_vals = sorted(ds["df_agg"][param_col].unique())

        y_means, y_errs = _compute_plot_metrics(
            ds["df_samples"],
            param_vals,
            param_col,
            quality_metric,
            sample_col_q,
            group_size,
            ci_method,
            confidence,
        )
        x_means, x_errs = _compute_plot_metrics(
            ds["df_samples"],
            param_vals,
            param_col,
            efficiency_metric,
            sample_col_e,
            group_size,
            ci_method,
            confidence,
        )

        # inject no-skip point
        if include_no_skip_point and ds["display_name"] == main_method_name:
            # skipped vs baseline and relative metrics both are just 1.0
            if quality_metric.startswith("avg_relative_") or quality_metric in [
                "avg_bleu",
                "avg_rouge_l",
                "avg_bert_score",
                "avg_token_accuracy",
            ]:
                anchor_y, anchor_y_err = 1.0, 0.0

            # skipped vs label metrics anchor at the baseline's score
            # against the human label
            else:
                # dynamically construct the baseline column name
                # (e.g. "label_bleu" -> "baseline_label_bleu")
                base_col_q = f"baseline_{sample_col_q}"
                if base_col_q in ds["df_samples"].columns:
                    raw_base_q = ds["df_samples"][base_col_q].values
                    anchor_y, anchor_y_err = calculate_grouped_ci(
                        raw_base_q,
                        group_size=group_size,
                        ci_method=ci_method,
                        confidence=confidence,
                    )
                else:
                    anchor_y, anchor_y_err = 0.0, 0.0

            # efficiency part
            if "speedup" in efficiency_metric:
                anchor_x, anchor_x_err = 1.0, 0.0
            else:
                anchor_x, anchor_x_err = 0.0, 0.0
            x_means = np.insert(x_means, 0, anchor_x)
            y_means = np.insert(y_means, 0, anchor_y)
            x_errs = np.insert(x_errs, 0, anchor_x_err)
            y_errs = np.insert(y_errs, 0, anchor_y_err)
            param_vals = ["No Skip"] + list(param_vals)

        # sort the data by x-axis (efficiency)
        sort_indices = np.argsort(x_means)
        x_means = x_means[sort_indices]
        y_means = y_means[sort_indices]
        x_errs = x_errs[sort_indices]
        y_errs = y_errs[sort_indices]
        param_vals = [param_vals[i] for i in sort_indices]

        # track global limits for tight bounding
        global_max_x = max(global_max_x, np.max(x_means + x_errs))
        global_min_x = min(global_min_x, np.min(x_means - x_errs))

        # z-order layers
        z_err = ds["z_base"]
        z_line = ds["z_base"] + 1
        z_scat = ds["z_base"] + 2

        # error bars
        error_bar_alpha = 1.0
        ax.errorbar(
            x_means,
            y_means,
            xerr=x_errs,
            yerr=y_errs,
            fmt="none",
            color=ds["error_color"],
            capsize=4,
            elinewidth=1.5,
            alpha=error_bar_alpha,
            zorder=z_err,
        )

        # plot main line
        ax.plot(
            x_means,
            y_means,
            color=ds["color"],
            linestyle="-",
            linewidth=2.0,
            alpha=0.9,
            zorder=z_line,
            label=ds["display_name"],
        )
        ax.scatter(
            x_means,
            y_means,
            color=ds["color"],
            marker=ds["marker"],
            s=60,
            alpha=0.9,
            zorder=z_scat,
        )

        # annotations
        if ds.get("show_labels", True):
            for i, val in enumerate(param_vals):
                # Silently skip drawing a label for the anchor point
                if str(val) == "No Skip":
                    continue
                if label_interval <= len(param_vals) and i % label_interval == 0:
                    ax.annotate(
                        rf"\textbf{{{ds['label_prefix']}: {val}}}",
                        (x_means[i], y_means[i]),
                        textcoords="offset points",
                        xytext=(2.3, 2.3),
                        ha="left",
                        va="bottom",
                        fontsize=10,
                        zorder=z_scat + 1,
                    )

    # tighten the x-axis so it ends immediately after the furthest error bar
    x_range = global_max_x - global_min_x
    ax.set_xlim(
        left=global_min_x - (x_range * 0.02), right=global_max_x + (x_range * 0.02)
    )

    ax.set_title(r"\textbf{Pareto Frontier: Efficiency vs. Quality}")
    ax.set_xlabel(rf"\textbf{{{display_eff_metric}}}")
    ax.set_ylabel(rf"\textbf{{{display_qual_metric}}}")

    # legend box
    ax.legend(
        loc="best",
        frameon=True,
        facecolor="#f5f5f5",  # lightgray background
        edgecolor="darkgray",
        framealpha=0.95,
    )

    fig.tight_layout()
    # get final level folder name of root_plot_dir
    folder_prefix = os.path.basename(root_plot_dir.rstrip("/"))
    plot_dir = os.path.join(root_plot_dir, "threshold_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(
        plot_dir,
        f"pareto_front_errorbars_{quality_metric}_{efficiency_metric}_{folder_prefix}.pdf",
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
