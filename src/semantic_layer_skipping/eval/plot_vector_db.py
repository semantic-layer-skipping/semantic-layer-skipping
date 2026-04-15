import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eval.plot_loader import FIG_SIZE_STANDARD
from utils import PLOTS_DIR


def plot_db_active_usage(row: pd.Series, root_plot_dir: str = PLOTS_DIR):
    """
    Generates a logarithmic bar chart comparing the total database capacity
    against the active number of unique vectors retrieved.
    """
    sizes = row["db_index_sizes"]
    hits = row["db_hit_counts"]
    if not sizes or not hits:
        return

    checkpoints = sorted([int(k) for k in sizes.keys()])
    total_sizes = [sizes[str(ckpt)] for ckpt in checkpoints]
    unique_hits = [len(hits.get(str(ckpt), {})) for ckpt in checkpoints]

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)
    ax.grid(axis="y")

    x = np.arange(len(checkpoints))
    width = 0.35

    bars_total = ax.bar(
        x - width / 2,
        total_sizes,
        width,
        label=r"\textbf{Total DB Size}",
        color="lightgray",
        edgecolor="black",
        zorder=3,
    )
    bars_hits = ax.bar(
        x + width / 2,
        unique_hits,
        width,
        label=r"\textbf{Unique Vectors Hit}",
        color="tab:red",
        edgecolor="black",
        zorder=3,
    )

    # add exact numerical labels on top of the bars
    for bar in bars_total:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval * 1.1,
            f"{int(yval):,}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )

    for bar in bars_hits:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval * 1.1,
            f"{int(yval):,}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )

    # use log scale, but force the bottom limit so the major ticks render correctly
    ax.set_yscale("log")
    ax.set_ylim(bottom=1000, top=max(total_sizes) * 3)

    ax.set_title(
        rf"\textbf{{Database Capacity vs. Active Usage (Threshold: {row['threshold']})}}"  # noqa: E501
    )
    ax.set_xlabel(r"\textbf{Checkpoint Index}")
    ax.set_ylabel(r"\textbf{Number of Vectors}")

    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints)
    ax.legend(loc="upper right")

    fig.tight_layout()
    plot_dir = os.path.join(root_plot_dir, "vector_db")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"db_active_usage_t{row['threshold']}.png")

    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved DB Active Usage plot to {plot_path}")


def _plot_rank_frequency(
    row: pd.Series,
    root_plot_dir: str,
    plot_type: str,
    use_log_scale: bool,
    title: str,
    xlabel: str,
    ylabel: str,
    filename_prefix: str,
    top_n: int = None,
):
    """
    Internal core function to plot rank-frequency distributions.
    Handles data extraction, sorting, axis scaling, and file saving to ensure DRY code.
    """
    hits = row.get("db_hit_counts")
    if not hits:
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)
    checkpoints = sorted([int(k) for k in hits.keys()])
    has_data = False

    for ckpt in checkpoints:
        ckpt_hits = hits.get(str(ckpt), {})
        if not ckpt_hits:
            continue

        # extract frequencies and sort descending
        frequencies = sorted(list(ckpt_hits.values()), reverse=True)

        # apply slice if top_n is specified
        if top_n is not None:
            frequencies = frequencies[:top_n]

        ranks = np.arange(1, len(frequencies) + 1)

        # render dynamically based on the requested plot style
        if plot_type == "line":
            ax.plot(
                ranks, frequencies, linewidth=2, label=f"Checkpoint {ckpt}", zorder=3
            )
        elif plot_type == "scatter":
            ax.scatter(
                ranks,
                frequencies,
                s=15,
                alpha=0.6,
                label=f"Checkpoint {ckpt}",
                zorder=3,
            )

        has_data = True

    if not has_data:
        plt.close(fig)
        return

    # apply logarithmic scaling if requested
    if use_log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # apply text
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=r"\textbf{Checkpoints}")

    fig.tight_layout()
    plot_dir = os.path.join(root_plot_dir, "vector_db")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{filename_prefix}_t{row['threshold']}.png")

    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(
        f"Saved {filename_prefix.replace('_', ' ').title()} plot to {plot_path}"
    )


def plot_vector_zipf_curve(
    row: pd.Series, root_plot_dir: str = PLOTS_DIR, top_n: int = 200
):
    """
    Visualises the Zipfian (power-law) curve by plotting the hit frequencies
    of the most accessed vectors on standard linear axes.
    """
    _plot_rank_frequency(
        row=row,
        root_plot_dir=root_plot_dir,
        plot_type="line",
        use_log_scale=False,
        title=rf"\textbf{{Top {top_n} Vector Hits (Threshold: {row['threshold']})}}",  # noqa: E501
        xlabel=r"\textbf{Vector Rank}",
        ylabel=r"\textbf{Hit Count}",
        filename_prefix="vector_zipf_curve",
        top_n=top_n,
    )


def plot_vector_hit_distribution(row: pd.Series, root_plot_dir: str = PLOTS_DIR):
    """
    Visualises the rank-frequency distribution of vector hits to identify
    power-law (Zipfian) behaviour. Plots on a log-log scale using a scatter plot.
    """
    _plot_rank_frequency(
        row=row,
        root_plot_dir=root_plot_dir,
        plot_type="scatter",
        use_log_scale=True,
        title=rf"\textbf{{Vector Hit Frequency Distribution (Threshold: {row['threshold']})}}",  # noqa: E501
        xlabel=r"\textbf{Vector Rank (Log Scale)}",
        ylabel=r"\textbf{Hit Count (Log Scale)}",
        filename_prefix="vector_hit_distribution",
        top_n=None,
    )
