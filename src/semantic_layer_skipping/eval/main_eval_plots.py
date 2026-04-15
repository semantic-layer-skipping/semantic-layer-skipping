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
from eval.plot_loader import load_eval_results, use_science_style
from eval.plot_quality import (
    plot_baseline_vs_skipped_quality,
    plot_pareto_front,
    plot_quality_scale_factor,
    plot_threshold_sensitivity,
)
from eval.plot_vector_db import (
    plot_db_active_usage,
    plot_vector_hit_distribution,
    plot_vector_zipf_curve,
)
from utils import set_logging_config

if __name__ == "__main__":
    set_logging_config()
    use_science_style()

    # quality flags
    PLOT_STANDARD_QUALITY = False
    PLOT_LABEL_COMPARISONS = False

    # architecture and single-threshold flags
    PLOT_SKIP_ACCEPTANCE_RATE = False
    PLOT_CHECKPOINT_SKIP_HEATMAP = False
    PLOT_GROUPED_TOKEN_DISTRIBUTION = False

    # single threshold
    PLOT_TOKEN_SKIP_HISTOGRAM = False
    PLOT_PROMPT_LENGTH_VS_SKIPPED = False

    # db eval - single threshold
    PLOT_DB_UTILISATION = True

    TARGET_THRESHOLD = 0.86

    # results dir
    RESULTS_DIR = "hpc/experiments/batch_20260309_042303_Qwen2.5-1.5B-Instruct_sharegpt_train_20000s_2048t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results_db_ivfpq_subsampled_10pct/"  # noqa: E501

    # prefix of files to analyse
    PREFIX = "sharegpt_test_100s_2048t"

    experiment_plots_dir = os.path.join(RESULTS_DIR, f"plots-prefix-{PREFIX}")

    logging.info("Loading data... from %s with prefix %s", RESULTS_DIR, PREFIX)
    df_agg, df_samples = load_eval_results(RESULTS_DIR, PREFIX)

    if df_agg.empty:
        logging.warning("No valid data found. Check your directory path and prefix.")
        sys.exit(0)

    if PLOT_STANDARD_QUALITY:
        logging.info("Generating Standard Quality Plots...")
        standard_metrics = [
            "avg_token_accuracy",
            "avg_bleu",
            "avg_rouge_l",
            "avg_bert_score",
        ]
        for metric in standard_metrics:
            plot_threshold_sensitivity(
                df_agg,
                quality_metric=metric,
                efficiency_metric="skipped_layer_percentage",
                root_plot_dir=experiment_plots_dir,
            )
            plot_pareto_front(
                df_agg,
                quality_metric=metric,
                efficiency_metric="skipped_layer_percentage",
                root_plot_dir=experiment_plots_dir,
            )

    if PLOT_LABEL_COMPARISONS:
        logging.info("Generating Label Comparison Plots...")
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
                df_agg,
                skipped_metric,
                baseline_metric,
                display_name,
                root_plot_dir=experiment_plots_dir,
            )
            plot_quality_scale_factor(
                df_agg,
                skipped_metric,
                baseline_metric,
                display_name,
                root_plot_dir=experiment_plots_dir,
            )

    if PLOT_SKIP_ACCEPTANCE_RATE:
        logging.info("Generating Skip Acceptance Rate Plot...")
        plot_skip_acceptance_rate(df_agg, root_plot_dir=experiment_plots_dir)

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
