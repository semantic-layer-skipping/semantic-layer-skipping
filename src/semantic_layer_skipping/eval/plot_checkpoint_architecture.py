import logging
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eval.utils import FIG_SIZE_STANDARD, calculate_grouped_ci
from utils import PLOTS_DIR


def plot_checkpoint_skip_heatmap(
    row: pd.Series,
    root_plot_dir: str = PLOTS_DIR,
    show_colorbar: bool = True,
):
    """Plots a heatmap of the distribution of size of skips per checkpoint."""
    stats = row.get("checkpoint_skip_stats", {})
    if not stats:
        return

    if "trial_id" in row.index:
        title_str = f"(Trial {int(row['trial_id'])})"
        file_suffix = f"trial_{int(row['trial_id'])}"
    else:
        val = row.get("threshold", "Mixed")
        title_str = f"(Threshold: {val})"
        file_suffix = str(val)

    checkpoints = sorted([int(k) for k in stats.keys()])
    max_ckpt = max(checkpoints) if checkpoints else 0
    all_decisions = set()
    for ckpt in checkpoints:
        for decision in stats[str(ckpt)].keys():
            all_decisions.add(str(decision))

    # remove "0" (no skip) and "exit" for the sorting of block skips
    decision_cols = sorted(
        [d for d in all_decisions if d not in ("exit", "0")], key=int
    )
    if "exit" in all_decisions:
        decision_cols.append("exit")  # add exit to the end

    # initialise with np.nan for N/A handling
    matrix = np.full((len(checkpoints), len(decision_cols)), np.nan)
    for i, ckpt in enumerate(checkpoints):
        # only count actual skips and exits (ignore no-skips)
        total_skips = sum(
            val for dec, val in stats[str(ckpt)].items() if str(dec) != "0"
        )

        for j, dec in enumerate(decision_cols):
            # determine if this cell is structurally impossible (N/A)
            is_na = False
            if dec != "exit":
                # if checkpoint + skip size exceeds the final checkpoint, it's N/A
                if int(ckpt) + int(dec) > max_ckpt:
                    is_na = True
            if is_na:
                matrix[i, j] = np.nan
            else:
                if total_skips == 0:
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = (
                        stats[str(ckpt)].get(str(dec), 0) / total_skips
                    ) * 100

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)

    # extract 0.2 to 0.7 colour map
    base_cmap = plt.get_cmap("viridis")
    cmap = colors.ListedColormap(base_cmap(np.linspace(0.2, 0.7, 256)))
    cmap.set_bad(color="#f0f0f0")

    im = ax.imshow(
        matrix,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=100,
    )
    if show_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r"\textbf{Frequency (\%)}", rotation=-90, va="bottom")

    for i in range(len(checkpoints)):
        for j in range(len(decision_cols)):
            val = matrix[i, j]
            if np.isnan(val):
                # render N/A text
                ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=9)
            else:
                # use black text
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black")

    ax.set_xticks(np.arange(len(decision_cols)))
    ax.set_yticks(np.arange(len(checkpoints)))
    ax.set_xticklabels(decision_cols)
    ax.set_yticklabels(checkpoints)

    ax.set_title(rf"\textbf{{Skip Decision Heatmap {title_str}}}")
    ax.set_xlabel(r"\textbf{Skip Amount (Blocks / Exit)}")
    ax.set_ylabel(r"\textbf{Checkpoint Index}")

    ax.grid(False)

    # set minor ticks exactly halfway between the cells
    ax.set_xticks(np.arange(-0.5, len(decision_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(checkpoints), 1), minor=True)

    # draw a solid white line on the minor ticks to frame the cells
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plot_dir = os.path.join(root_plot_dir, "architecture")
    os.makedirs(plot_dir, exist_ok=True)

    suffix_modifier = "" if show_colorbar else "_noscale"
    plot_path = os.path.join(plot_dir, f"heatmap_t{file_suffix}{suffix_modifier}.pdf")

    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved pure Matplotlib Heatmap to {plot_path}")


def plot_skip_acceptance_rate(
    df_agg: pd.DataFrame,
    df_samples: pd.DataFrame,
    root_plot_dir: str = PLOTS_DIR,
    group_size: int = 10,
    ci_method: str = "t_dist",
    confidence: float = 0.95,
):
    """
    Plots the percentage of visits that resulted in a skip (>0) per checkpoint,
    with Confidence Intervals and distinct markers.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)

    all_ckpts = set()
    for stats in df_agg["checkpoint_skip_stats"]:
        if isinstance(stats, dict):
            all_ckpts.update(stats.keys())
    logging.info(f"Detected checkpoints in data: {all_ckpts}")

    if not all_ckpts:
        logging.warning("No checkpoint data found in df['checkpoint_skip_stats']!")
        plt.close(fig)
        return

    thresholds = sorted(df_agg["threshold"].unique())

    # colour map and distinct markers (supports up to 12 distinct shapes cleanly)
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "p", "*", "X", "<", ">", "h", "H"]

    for idx, ckpt in enumerate(sorted([int(k) for k in all_ckpts])):
        means = []
        cis = []

        for t in thresholds:
            # isolate samples for this threshold
            subset = df_samples[df_samples["threshold"] == t]

            raw_rates = []
            for _, row in subset.iterrows():
                # extract skip stats for this specific sample
                stats = row.get("checkpoint_skip_stats", {})
                ckpt_stats = stats.get(str(ckpt), {})

                total_visits = sum(ckpt_stats.values())
                skips = sum(v for k, v in ckpt_stats.items() if str(k) != "0")
                rate = (skips / total_visits) * 100
                raw_rates.append(rate)

            raw_rates = np.array(raw_rates, dtype=float)

            # calculate CIs
            mean_val, ci_val = calculate_grouped_ci(
                raw_rates,
                group_size=group_size,
                ci_method=ci_method,
                confidence=confidence,
            )

            means.append(mean_val)
            cis.append(ci_val)

        means = np.array(means)
        cis = np.array(cis)
        color = cmap(idx)

        # select marker based on index and plot
        marker = markers[idx % len(markers)]
        ax.plot(
            thresholds,
            means,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=7,
            label=f"Checkpoint {ckpt}",
        )

        # plot the confidence band
        if ci_method != "none" and np.any(cis > 0):
            ax.fill_between(
                thresholds, means - cis, means + cis, color=color, alpha=0.15
            )

    ax.set_title(r"\textbf{Skip Acceptance Rate by Checkpoint}")
    ax.set_xlabel(r"\textbf{Cosine Similarity Threshold}")
    ax.set_ylabel(r"\textbf{Skip Probability (\%)}")
    ax.legend(loc="best")

    fig.tight_layout()
    # get final level folder name of root_plot_dir
    folder_prefix = os.path.basename(root_plot_dir.rstrip("/"))
    plot_dir = os.path.join(root_plot_dir, "architecture")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"skip_acceptance_rates_{folder_prefix}.pdf")

    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved Skip Acceptance Rates plot to {plot_path}")
