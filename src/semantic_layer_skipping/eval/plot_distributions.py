import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eval.plot_loader import FIG_SIZE_SMALL, FIG_SIZE_STANDARD
from utils import PLOTS_DIR


def plot_token_skip_histogram(row: pd.Series, root_plot_dir: str = PLOTS_DIR):
    """Plots the distribution of layers skipped per token."""
    dist = row["token_skip_distribution"]
    if not dist:
        return

    x = sorted([int(k) for k in dist.keys()])
    y = [dist[str(k)] for k in x]

    fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
    # remove grid
    ax.grid(False)

    ax.bar([str(val) for val in x], y, color="tab:purple", edgecolor="black", zorder=3)

    # use log scale for y axis
    ax.set_yscale("log")

    ax.set_title(rf"\textbf{{Token Skip Distribution (Threshold: {row['threshold']})}}")
    ax.set_xlabel(r"\textbf{Layers Skipped by a Single Token}")
    ax.set_ylabel(r"\textbf{Number of Tokens}")

    fig.tight_layout()
    plot_dir = os.path.join(root_plot_dir, "distributions")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"token_dist_t{row['threshold']}.png")

    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved Token Distribution plot to {plot_path}")


def plot_grouped_token_skip_histogram(df: pd.DataFrame, root_plot_dir: str = PLOTS_DIR):
    """
    Plots a grouped bar chart comparing token skip distributions across all thresholds.
    """
    # find all unique skip amounts across all thresholds
    all_skips = set()
    for dist in df["token_skip_distribution"]:
        all_skips.update(dist.keys())

    if not all_skips:
        return
    skip_amounts = sorted([int(k) for k in all_skips])

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)
    ax.grid(axis="x")

    # grouping bars
    x_positions = np.arange(len(skip_amounts))
    num_thresholds = len(df)
    total_width = 0.8  # total width of a cluster of bars
    bar_width = total_width / num_thresholds

    # automatically colour each threshold nicely
    cmap = plt.get_cmap("viridis_r")
    colours = cmap(np.linspace(0.1, 0.9, num_thresholds))

    # plot a set of bars for each threshold
    for i, (_, row) in enumerate(df.iterrows()):
        threshold = row["threshold"]
        dist = row["token_skip_distribution"]

        total_tokens = sum(dist.values()) if dist else 1
        # convert raw counts to percentages
        y_values = [
            (dist.get(str(skip), 0) / total_tokens) * 100 if dist else 0
            for skip in skip_amounts
        ]

        # calculate the exact offset to place this bar
        offset = (i - num_thresholds / 2) * bar_width + bar_width / 2

        ax.bar(
            x_positions + offset,
            y_values,
            width=bar_width,
            label=f"{threshold}",
            color=colours[i],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_yscale("log")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(skip) for skip in skip_amounts])

    ax.set_title(r"\textbf{Token Skip Distribution by Threshold}")
    ax.set_xlabel(r"\textbf{Layers Skipped by a Single Token}")
    ax.set_ylabel(r"\textbf{Percentage of Tokens (\%)}")

    ax.legend(title=r"\textbf{Threshold}", loc="upper right")

    fig.tight_layout()
    plot_dir = os.path.join(root_plot_dir, "distributions")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "grouped_token_distribution.png")

    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved Grouped Token Distribution plot to {plot_path}")


def plot_generated_length_vs_skipped(
    df_samples: pd.DataFrame, target_threshold: float, root_plot_dir: str = PLOTS_DIR
):
    """Scatter plot of Generated Token Count vs Total Skipped Layers."""
    df_filtered = df_samples[df_samples["threshold"] == target_threshold].copy()
    if df_filtered.empty:
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)

    x = df_filtered["generated_token_count"]
    y = df_filtered["skipped_layers"]

    ax.scatter(x, y, alpha=0.6, color="tab:green", zorder=3, label=r"\textbf{Requests}")

    # add a linear trendline to visualise the average skip rate
    if len(x) > 1 and x.std() > 0:
        m, b = np.polyfit(x, y, 1)
        ax.plot(
            x,
            m * x + b,
            color="tab:red",
            linestyle="--",
            linewidth=2,
            zorder=4,
            label=rf"\textbf{{Trend (Avg {m:.1f} skips/token)}}",
        )
        ax.legend()

    ax.set_title(
        rf"\textbf{{Generated Length vs Total Skipped (Threshold: {target_threshold})}}"
    )
    ax.set_xlabel(r"\textbf{Generated Tokens}")
    ax.set_ylabel(r"\textbf{Total Skipped Layers}")

    fig.tight_layout()
    plot_dir = os.path.join(root_plot_dir, "distributions")
    os.makedirs(plot_dir, exist_ok=True)

    plot_path = os.path.join(plot_dir, f"generated_vs_skips_t{target_threshold}.png")

    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved Generated Length vs Skips Scatter plot to {plot_path}")
