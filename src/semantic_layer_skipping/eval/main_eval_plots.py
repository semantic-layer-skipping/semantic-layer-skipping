import logging
import os
import sys

from eval.plot_checkpoint_architecture import (
    plot_checkpoint_skip_heatmap,
    plot_skip_acceptance_rate,
)
from eval.plot_distributions import (
    plot_generated_length_vs_skipped,
    plot_grouped_token_skip_histogram,
    plot_token_skip_histogram,
)
from eval.plot_quality import (
    plot_pareto_frontier,
    plot_threshold_sensitivity,
)
from eval.plot_vector_db import (
    plot_db_active_usage,
    plot_vector_hit_distribution,
    plot_vector_zipf_curve,
)
from eval.utils import load_eval_results, use_science_style
from utils import set_logging_config

# fmt: off
# results dir
# RESULTS_DIR = "hpc/experiments/batch_20260309_042303_Qwen2.5-1.5B-Instruct_sharegpt_train_20000s_2048t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results_db_ivfpq_subsampled_100pct"  # noqa: E501
# RESULTS_DIR = "hpc/experiments/batch_20260407_021109_Qwen2.5-1.5B-Instruct_sharegpt_train_10000s_2048t_strict_strict_match_c2-4-6-8-10-12-14-16-18-20-22-24-26/manual_eval_results_db_ivfpq_subsampled_10pct"  # noqa: E501
# RESULTS_DIR = "hpc/experiments/batch_20260407_025540_Qwen2.5-3B-Instruct_sharegpt_train_10000s_2048t_strict_strict_match_c4-8-12-16-20-24-28-32/manual_eval_results_db_ivfpq_subsampled_10pct"  # noqa: E501

RESULTS_DIR_WMT = "hpc/experiments/batch_20260507_154513_Qwen2.5-1.5B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results_db_ivfpq_subsampled_100pct" # noqa: E501
RESULTS_DIR_SHAREGPT = "hpc/experiments/batch_20260309_042303_Qwen2.5-1.5B-Instruct_sharegpt_train_20000s_2048t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results_db_ivfpq_subsampled_100pct" # noqa: E501
RESULTS_DIR_E2E = "hpc/experiments/batch_20260507_152045_Qwen2.5-1.5B-Instruct_e2e_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results_db_ivfpq_subsampled_100pct" # noqa: E501


ACTIVE_MAIN_EXPERIMENTS = ["wmt-kv-full", "wmt-kv-project-only", "wmt-kv-copy"] # ["sharegpt-standard"] # noqa: E501
ACTIVE_BASELINES = [] #["prob-skip"]
#figures: "Pareto Frontier: Efficiency vs. Quality", "KV Computation Pareto Frontiers", "Search Strategy Pareto Frontiers" # noqa: E501
experiment_title = "KV Computation Pareto Frontiers"

main_experiments_config = {
    "sharegpt-standard": {
        "results_dir": RESULTS_DIR_SHAREGPT,
        "prefix": "sharegpt_test_100s_128t",
        "exact_vals": [0.84, 0.85, 0.86, 0.87, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.98, 0.99], # noqa: E501
        "baseline_exact_vals": [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15], # noqa: E501
        "display_name": "Retrieval-Guided Skip",
        "color": "purple",
        "error_color": "plum",
        "marker": "o",
        "inject_no_skip": False,
        "show_labels": True,
    },
    "e2e-standard": {
        "results_dir": RESULTS_DIR_E2E,
        "prefix": "e2e_test_100s_128t",
        "exact_vals": [0.94, 0.96, 0.98, 0.99, 0.994, 0.996, 1.0, 1.0005, 1.001, 1.002, 1.005], # noqa: E501
        "baseline_exact_vals": [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15], # noqa: E501
        "display_name": "Retrieval-Guided Skip",
        "color": "purple",
        "error_color": "plum",
        "marker": "o",
        "inject_no_skip": True,
        "show_labels": True,
    },
    "wmt-standard": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_top1_strict_full_generation_kv_full_compute",
        "exact_vals": [0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], # noqa: E501
        "display_name": "Full Compute KV",
        "color": "purple",
        "error_color": "plum",
        "marker": "o",
        "inject_no_skip": True,
        "show_labels": True,
        "baseline_exact_vals": [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15],
    },
    # kv ablations
    "wmt-kv-full": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_top1_strict_full_generation_kv_full_compute",
        "exact_vals": [0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], # noqa: E501
        "display_name": "Async Full Compute",
        "color": "purple",
        "error_color": "plum",
        "marker": "o",
        "inject_no_skip": True,
        "show_labels": False,
        "baseline_exact_vals": [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15], # noqa: E501
    },
    "wmt-kv-project-only": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_top1_strict_full_generation_kv_project_only",
        "exact_vals": [0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], # noqa: E501
        "display_name": "Project Only",
        "color": "tab:red",
        "error_color": "pink",
        "marker": "^",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-kv-copy": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_top1_strict_full_generation_kv_copy",
        "exact_vals": [0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], # noqa: E501
        "display_name": "Copy",
        "color": "tab:blue",
        "error_color": "lightsteelblue",
        "marker": "^",
        "inject_no_skip": True,
        "show_labels": False,
    },
}


baseline_configs = {
    "prob-skip": {
        "dir_name": "random_skip",
        "file_prefix": "random_baseline",
        "param_key": "random_skip_prob",
        "display_name": "Prob-Skip",
        "exact_vals": [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15],
        "label_prefix": "P",
        "show_labels": False,
    }
}

# fmt: on
if __name__ == "__main__":
    set_logging_config()
    use_science_style()

    # quality flags
    PLOT_PARETO_FRONTIER = True
    PLOT_THRESHOLD_SENSITIVITY = False

    PLOT_SKIP_ACCEPTANCE_RATE = False
    PLOT_GROUPED_TOKEN_DISTRIBUTION = False

    # single threshold plots
    TARGET_THRESHOLD = 0.9

    PLOT_CHECKPOINT_SKIP_HEATMAP = False
    PLOT_TOKEN_SKIP_HISTOGRAM = False
    PLOT_PROMPT_LENGTH_VS_SKIPPED = False
    PLOT_DB_UTILISATION = False

    # load experiments data
    loaded_main_experiments = []
    for exp_key in ACTIVE_MAIN_EXPERIMENTS:
        if exp_key not in main_experiments_config:
            continue

        conf = main_experiments_config[exp_key]
        logging.info(
            "Loading main experiment '%s' from %s with prefix %s",
            exp_key,
            conf["results_dir"],
            conf["prefix"],
        )
        df_agg, df_samples = load_eval_results(
            conf["results_dir"], conf["prefix"], exact_vals=conf["exact_vals"]
        )
        if not df_agg.empty:
            conf_copy = conf.copy()
            conf_copy["df_agg"] = df_agg
            conf_copy["df_samples"] = df_samples
            loaded_main_experiments.append(conf_copy)

    if not loaded_main_experiments:
        logging.warning("No valid main experiment data found. Exiting.")
        sys.exit(0)

    unique_results_dirs = set(exp["results_dir"] for exp in loaded_main_experiments)
    experiment_prefix = experiment_title.lower().replace(" ", "-")
    if len(unique_results_dirs) == 1:
        # all data originates from a single HPC batch folder - so we store here
        first_exp = loaded_main_experiments[0]
        experiment_name = experiment_prefix or first_exp["prefix"]
        OUTPUT_PLOTS_DIR = os.path.join(
            first_exp["results_dir"], f"plots-prefix-{experiment_name}"
        )
    else:
        # cross-directory:
        assert experiment_prefix is not None
        OUTPUT_PLOTS_DIR = os.path.join("hpc", "plots", experiment_prefix)

    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    logging.info("Output plots directory set to: %s", OUTPUT_PLOTS_DIR)

    # load baselines (using the base dir of the first active main experiment)
    loaded_baselines = []
    first_exp = loaded_main_experiments[0]
    base_experiments_dir = os.path.dirname(first_exp["results_dir"])
    #  the baseline override from the primary experiment (if it exists)
    primary_baseline_override = first_exp.get("baseline_exact_vals")
    for b_key in ACTIVE_BASELINES:
        if b_key not in baseline_configs:
            continue

        b_conf = baseline_configs[b_key]
        b_dir = os.path.join(base_experiments_dir, "baselines", b_conf["dir_name"])
        logging.info("Loading baseline '%s' from %s", b_key, b_dir)
        try:
            b_agg, b_samp = load_eval_results(
                b_dir, b_conf["file_prefix"], param_key=b_conf["param_key"]
            )

            # determine which exact_vals to use: the override from the main config,
            # or the default from the baseline config
            exact = (
                primary_baseline_override
                if primary_baseline_override is not None
                else b_conf.get("exact_vals")
            )
            if not b_agg.empty and exact is not None:
                b_agg = b_agg[b_agg[b_conf["param_key"]].isin(exact)]
                b_samp = b_samp[b_samp[b_conf["param_key"]].isin(exact)]

            if not b_agg.empty:
                loaded_baselines.append(
                    {
                        "display_name": b_conf["display_name"],
                        "df_agg": b_agg,
                        "df_samples": b_samp,
                        "param_key": b_conf["param_key"],
                        "label_prefix": b_conf["label_prefix"],
                        "show_labels": b_conf.get("show_labels", False),
                    }
                )
        except FileNotFoundError:
            logging.warning("Baseline directory not found: %s", b_dir)

    file_suffix = "_vs_".join(ACTIVE_MAIN_EXPERIMENTS)

    standard_metrics = [
        # "avg_token_accuracy",
        "avg_bleu",
        "avg_rouge_l",
        "avg_bert_score",
        "avg_label_bert_score",
        "avg_label_rouge_l",
        "avg_label_bleu",
        "avg_relative_bleu",
        "avg_relative_rouge_l",
        "avg_relative_bert_score",
    ]

    if PLOT_PARETO_FRONTIER:
        logging.info("Generating Standard Quality Plots...")
        for metric in standard_metrics:
            group_size = 10 if "relative" in metric else 1

            plot_pareto_frontier(
                main_experiments=loaded_main_experiments,
                baselines=loaded_baselines,
                quality_metric=metric,
                efficiency_metric="theoretical_speedup",
                root_plot_dir=OUTPUT_PLOTS_DIR,
                group_size=group_size,
                label_interval=2,
                plot_filename_suffix=file_suffix,
                experiment_title=experiment_title,
            )

    experiment_plots_dir = OUTPUT_PLOTS_DIR

    if PLOT_THRESHOLD_SENSITIVITY:
        for metric in standard_metrics:
            group_size = 10 if "relative" in metric else 1

            plot_threshold_sensitivity(
                loaded_main_experiments[0]["df_agg"],
                df_samples=loaded_main_experiments[0]["df_samples"],
                quality_metric=metric,
                efficiency_metric="theoretical_speedup",
                root_plot_dir=experiment_plots_dir,
                group_size=group_size,
                ci_method="t_dist",
            )

    df_agg = loaded_main_experiments[0]["df_agg"]
    df_samples = loaded_main_experiments[0]["df_samples"]

    if PLOT_SKIP_ACCEPTANCE_RATE:
        logging.info("Generating Skip Acceptance Rate Plot...")
        plot_skip_acceptance_rate(
            df_agg,
            df_samples,
            root_plot_dir=experiment_plots_dir,
            group_size=1,
        )
    if PLOT_GROUPED_TOKEN_DISTRIBUTION:
        logging.info("Generating Stacked Token Distribution Plot...")
        plot_grouped_token_skip_histogram(df_agg, root_plot_dir=experiment_plots_dir)

    # retrieve the specific row matching the target threshold
    target_row_df = df_agg[df_agg["threshold"] == TARGET_THRESHOLD]

    if not target_row_df.empty:
        target_row = target_row_df.iloc[0]

        if PLOT_CHECKPOINT_SKIP_HEATMAP:
            logging.info(f"Plotting Heatmap for Threshold {TARGET_THRESHOLD}...")
            plot_checkpoint_skip_heatmap(target_row, root_plot_dir=experiment_plots_dir)

        if PLOT_TOKEN_SKIP_HISTOGRAM:
            logging.info(
                f"Plotting Token Skip Histogram for Threshold {TARGET_THRESHOLD}..."
            )
            plot_token_skip_histogram(target_row, root_plot_dir=experiment_plots_dir)

        if PLOT_PROMPT_LENGTH_VS_SKIPPED and not df_samples.empty:
            logging.info(
                f"Plotting Length vs Skipped for Threshold {TARGET_THRESHOLD}..."
            )
            plot_generated_length_vs_skipped(
                df_samples,
                target_threshold=TARGET_THRESHOLD,
                root_plot_dir=experiment_plots_dir,
            )

        if PLOT_DB_UTILISATION:
            logging.info(
                f"Plotting DB Utilisation Plots for Threshold {TARGET_THRESHOLD}..."
            )
            plot_db_active_usage(target_row, root_plot_dir=experiment_plots_dir)
            plot_vector_zipf_curve(target_row, root_plot_dir=experiment_plots_dir)
            plot_vector_hit_distribution(target_row, root_plot_dir=experiment_plots_dir)
    else:
        logging.warning(f"Target threshold {TARGET_THRESHOLD} not found in the data!")
