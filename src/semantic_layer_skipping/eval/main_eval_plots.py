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
RESULTS_DIR_WMT = "hpc/experiments/batch_20260507_154513_Qwen2.5-1.5B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results_db_ivfpq_subsampled_100pct"  # noqa: E501
RESULTS_DIR_SHAREGPT = "hpc/experiments/batch_20260309_042303_Qwen2.5-1.5B-Instruct_sharegpt_train_20000s_2048t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results_db_ivfpq_subsampled_100pct"  # noqa: E501
RESULTS_DIR_E2E = "hpc/experiments/batch_20260507_152045_Qwen2.5-1.5B-Instruct_e2e_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results_db_ivfpq_subsampled_100pct"  # noqa: E501
RESULTS_DIR_WMT_C1 = "hpc/experiments/batch_20260514_035404_Qwen2.5-1.5B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27/manual_eval_results_db_ivfpq_subsampled_100pct" # noqa: E501
RESULTS_DIR_WMT_3B = "hpc/experiments/batch_20260516_232926_Qwen2.5-3B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24-28-32/manual_eval_results_db_ivfpq_subsampled_100pct" # noqa: E501

RESULTS_DIR_SHAREGPT_CAL = "hpc/experiments/batch_20260309_042303_Qwen2.5-1.5B-Instruct_sharegpt_train_20000s_2048t_strict_strict_match_c4-8-12-16-20-24/e2e_optimisation/db_ivfpq_subsampled_100pct_sharegpt_validation_250s_128t_top1_strict_full_generation_bert_label_ratio_theoretical_speedup/trials"

# figure configuration
ACTIVE_FIGURE = "sharegpt_checkpoint" #wmt_checkpoint sharegpt_checkpoint e2e_checkpoint

FIGURES_CONFIG = {
    "sharegpt_pareto": {
        "main_experiments": ["sharegpt-standard"],
        "baselines": ["prob-skip"],
        "experiment_title": "Pareto Frontier: Efficiency vs. Quality",
        "plot_types": [
            "pareto_frontier",
            "threshold_sensitivity",
            # "skip_acceptance_rate",
            # "grouped_token_distribution",
            # "checkpoint_skip_heatmap",
            # "token_skip_histogram",
            # "prompt_length_vs_skipped",
            # "db_utilisation"
        ],
        "target_threshold": 0.9,
    },
    "e2e_pareto": {
        "main_experiments": ["e2e-standard"],
        "baselines": ["prob-skip"],
        "experiment_title": "Pareto Frontier: Efficiency vs. Quality",
        "plot_types": ["pareto_frontier", "threshold_sensitivity"],
    },
    "wmt_pareto": {
        "main_experiments": ["wmt-standard"],
        "baselines": ["prob-skip"],
        "experiment_title": "Pareto Frontier: Efficiency vs. Quality",
        "plot_types": ["pareto_frontier", "threshold_sensitivity"],
    },
    "wmt_kv_ablation_pareto": {
        "main_experiments": ["wmt-kv-full", "wmt-kv-project-only", "wmt-kv-copy"],
        "baselines": [],
        "experiment_title": "KV Computation Pareto Frontiers",
        "plot_types": ["pareto_frontier"],
        "metrics": ["avg_label_bert_score"],
        "y_bounds": (0.0, 0.65)
    },
    "wmt_3B_kv_ablation_pareto": {
        "main_experiments": ["wmt-3B-kv-full", "wmt-3B-kv-project-only", "wmt-3B-kv-copy"],
        "baselines": [],
        "experiment_title": "KV Computation Pareto Frontiers",
        "plot_types": ["pareto_frontier"],
        "metrics": ["avg_label_bert_score"],
        "y_bounds": (0.0, 0.65)
    },
    # search strategy ablations
    "wmt_safe_knn_ablation_pareto": {
        "main_experiments": [
            "wmt-top1_strict-200",
            # "wmt-safe-knn",
            #"wmt-safe-knn-5",
            #"wmt-safe-knn-10",
            "wmt-200-safe-knn-5",
            "wmt-200-safe-knn-20",
            "wmt-200-safe-knn-50",
            #"wmt-safe-knn-50",
            #"wmt-consensus-decay",
            #"wmt-semantic-boundary",
        ],
        "baselines": [],
        "experiment_title": "Safe $k$-NN Online Decision Strategy Pareto Frontiers",
        "plot_types": ["pareto_frontier"],
        "target_threshold": 0.9,
        "confidence": 0.8,
        "metrics": ["avg_label_bert_score"],
        "force_pareto": True,
        "y_bounds": (0.37, 0.60)
    },
    "wmt_expected_skip_ablation_pareto": {
        "main_experiments": [
            "wmt-top1_strict-200",
            "wmt-200-softmax-expected-skip-5",
            # "wmt-softmax-expected-skip-10",
            "wmt-200-softmax-expected-skip-20",
            "wmt-200-softmax-expected-skip-50",
            # "wmt-consensus-decay",
            #"wmt-semantic-boundary",
        ],
        "baselines": [],
        "experiment_title": "Softmax Expected Online Decision Strategy Pareto Frontiers",
        "plot_types": ["pareto_frontier"],
        "target_threshold": 0.9,
        "confidence": 0.8,
        "metrics": ["avg_label_bert_score"],
        "force_pareto": True,
        "y_bounds": (0.37, 0.60)
    },
    # safe knn threshold sensitivity
    "wmt_safe_knn_overall": {
        "main_experiments": [
            "wmt-safe-knn-thresholds",
        ],
        "experiment_title": "Safe-knn thresholds",
        "baselines": [],
        "plot_types": [
            "threshold_sensitivity",
            "skip_acceptance_rate",
            "grouped_token_distribution",
            "checkpoint_skip_heatmap",
            #"token_skip_histogram",
            "prompt_length_vs_skipped",
            "db_utilisation"
        ],
        "confidence": 0.95,
    },
    # threshold sensitivity
    "wmt_c1_overall": {
        "main_experiments": ["wmt-c1-thresholds"],
        "experiment_title": "C1 thresholds",
        "plot_types": ["threshold_sensitivity"],
    },
    "wmt_c4_overall": {
        "main_experiments": ["wmt-c4-thresholds"],
        "experiment_title": "C4 thresholds",
        "plot_types": ["threshold_sensitivity"],
    },
    # cal
    "sharegpt_trial_55": {
        "main_experiments": ["sharegpt-trial-55"],
        "plot_types": [
            # "skip_acceptance_rate",
            # "grouped_token_distribution",
            "checkpoint_skip_heatmap",
            "prompt_length_vs_skipped",
            "db_utilisation"
        ],
        "experiment_title": "trial-55",
    },
    # cal
    "sharegpt_trial_73": {
        "main_experiments": ["sharegpt-trial-73"],
        "plot_types": [
            "checkpoint_skip_heatmap",
            "prompt_length_vs_skipped",
            "db_utilisation"
        ],
        "experiment_title": "trial-73",
    },
    "sharegpt_trial_56": {
        "main_experiments": ["sharegpt-trial-56"],
        "plot_types": [
            "checkpoint_skip_heatmap",
            "prompt_length_vs_skipped",
            "db_utilisation"
        ],
        "experiment_title": "trial-56",
    },
    "wmt_checkpoint": {
        "main_experiments": ["wmt-standard"],
        "experiment_title": "wmt_checkpoint",
        "plot_types": [
            "skip_acceptance_rate",
            "grouped_token_distribution",
            "checkpoint_skip_heatmap",
            #"token_skip_histogram",
            "prompt_length_vs_skipped",
            "db_utilisation"
        ],
        "target_threshold": 0.93, #0.86
    },
    "wmt_3B_checkpoint": {
        "main_experiments": ["wmt-3B-kv-full"],
        "experiment_title": "wmt_3B_checkpoint",
        "plot_types": [
            "skip_acceptance_rate",
            "grouped_token_distribution",
            "checkpoint_skip_heatmap",
            # "token_skip_histogram",
            "prompt_length_vs_skipped",
            "db_utilisation"
        ],
        "target_threshold": 0.86,  # 0.86
    },
    "e2e_checkpoint": {
        "main_experiments": ["e2e-checkpoint"],
        "experiment_title": "e2e_checkpoint",
        "plot_types": [
            "skip_acceptance_rate",
            #"grouped_token_distribution",
            "checkpoint_skip_heatmap",
            #"token_skip_histogram",
            #"prompt_length_vs_skipped",
            #"db_utilisation"
        ],
        "target_threshold": 0.94, #0.94
    },
    "sharegpt_checkpoint": {
        "main_experiments": ["sharegpt-standard"],
        "experiment_title": "sharegpt_checkpoint",
        "plot_types": [
            "skip_acceptance_rate",
            #"grouped_token_distribution",
            "checkpoint_skip_heatmap",
            #"token_skip_histogram",
            #"prompt_length_vs_skipped",
            #"db_utilisation"
        ],
        "target_threshold": 0.84, # 0.86
        "show_colorbar": False,
    },
}

main_experiments_config = {
    "sharegpt-standard": {
        "results_dir": RESULTS_DIR_SHAREGPT,
        "prefix": "sharegpt_test_100s_128t",
        "exact_vals": [0.84, 0.85, 0.86, 0.87, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.98, 0.99],
        "baseline_exact_vals": [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15],
        "display_name": "Retrieval-Guided Skip",
        "color": "purple",
        "error_color": "plum",
        "marker": "o",
        "inject_no_skip": False,
        "show_labels": False,
    },
    "e2e-standard": {
        "results_dir": RESULTS_DIR_E2E,
        "prefix": "e2e_test_100s_128t",
        "exact_vals": [0.94, 0.96, 0.98, 0.99, 0.994, 0.996, 1.0, 1.0005, 1.001, 1.002, 1.005],
        "baseline_exact_vals": [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15],
        "display_name": "Retrieval-Guided Skip",
        "color": "purple",
        "error_color": "plum",
        "marker": "o",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-standard": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_top1_strict_full_generation_kv_full_compute",
        "exact_vals": [0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
        "display_name": "Full Compute KV",
        "color": "purple",
        "error_color": "plum",
        "marker": "o",
        "inject_no_skip": True,
        "show_labels": False,
        "baseline_exact_vals": [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15],
    },
    # kv ablations
    "wmt-kv-full": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_top1_strict_full_generation_kv_full_compute",
        "exact_vals": [0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
        "display_name": "Async Full Compute",
        "color": "purple",
        "error_color": "plum",
        "marker": "o",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-kv-project-only": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_top1_strict_full_generation_kv_project_only",
        "exact_vals": [0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
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
        "exact_vals": [0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
        "display_name": "Copy",
        "color": "tab:blue",
        "error_color": "lightsteelblue",
        "marker": "s",
        "inject_no_skip": True,
        "show_labels": False,
    },
    # 3B kv ablation
    "wmt-3B-kv-full": {
        "results_dir": RESULTS_DIR_WMT_3B,
        "prefix": "wmt19_test_100s_128t_top1_strict_k_5_full_generation_kv_full_compute",
        "exact_vals": [0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
        "display_name": "Async Full Compute",
        "color": "purple",
        "error_color": "plum",
        "marker": "o",
        "inject_no_skip": True,
        "show_labels": False,

    },
    "wmt-3B-kv-project-only": {
        "results_dir": RESULTS_DIR_WMT_3B,
        "prefix": "wmt19_test_100s_128t_top1_strict_k_5_full_generation_kv_project_only",
        "exact_vals": [0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
        "display_name": "Project Only",
        "color": "tab:red",
        "error_color": "pink",
        "marker": "^",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-3B-kv-copy": {
        "results_dir": RESULTS_DIR_WMT_3B,
        "prefix": "wmt19_test_100s_128t_top1_strict_k_5_full_generation_kv_copy",
        "exact_vals": [0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
        # noqa: E501
        "display_name": "Copy",
        "color": "tab:blue",
        "error_color": "lightsteelblue",
        "marker": "s",
        "inject_no_skip": True,
        "show_labels": False,
    },
    # search strategy ablations
    "wmt-top1_strict": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_top1_strict_full_generation_kv_full_compute",
        "exact_vals": [0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.98, 0.99],
        "display_name": "Top-1 Strict",
        "color": "purple",
        "error_color": "plum",
        "marker": "o",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-top1_strict-200": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_200s_128t_top1_strict_k_5_full_generation_kv_full_compute",
        "exact_vals": [0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
        "display_name": "Top-1 Strict",
        "color": "purple",
        "error_color": "plum",
        "marker": "o",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-safe-knn": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_safe_knn_full_generation_kv_full_compute",
        "exact_vals": [0.83, 0.84, 0.86, 0.87, 0.89, 0.9, 0.92, 0.93, 0.94, 0.95, 0.97, 0.99],
        "display_name": "Safe KNN ($k$=3)",
        "color": "tab:blue",
        "error_color": "lightsteelblue",
        "marker": "^",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-safe-knn-5": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_safe_knn_k_5_full_generation",
        "exact_vals": [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
        "display_name": "Safe KNN ($k$=5)",
        "color": "tab:orange",
        "error_color": "navajowhite",
        "marker": "s",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-safe-knn-10": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_safe_knn_k_10_full_generation",  # noqa: E501
        "exact_vals": [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], # noqa: E501
        "display_name": "Safe KNN ($k$=10)",
        "color": "tab:green",
        "error_color": "lightgreen",
        "marker": "D",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-safe-knn-50": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_safe_knn_k_50_full_generation",  # noqa: E501
        "exact_vals": [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95,
                       0.96, 0.97, 0.98, 0.99],  # noqa: E501
        "display_name": "Safe KNN ($k$=50)",
        "color": "tab:blue",
        "error_color": "lightsteelblue",
        "marker": "D",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-200-safe-knn-5": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_200s_128t_safe_knn_k_5_full_generation",  # noqa: E501
        "exact_vals": [0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94,
                       0.96, 0.97, 0.98, 0.99],  # noqa: E501
        "display_name": "Safe KNN ($k$=5)",
        "color": "tab:orange",
        "error_color": "navajowhite",
        "marker": "D",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-200-safe-knn-20": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_200s_128t_safe_knn_k_20_full_generation",  # noqa: E501
        "exact_vals": [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95,
                       0.96, 0.97, 0.98, 0.99],  # noqa: E501
        "display_name": "Safe KNN ($k$=20)",
        "color": "tab:green",
        "error_color": "lightgreen",
        "marker": "D",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-200-safe-knn-50": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_200s_128t_safe_knn_k_50_full_generation",  # noqa: E501
        "exact_vals": [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.95,
                     0.98, 0.99],  # noqa: E501
        "display_name": "Safe KNN ($k$=50)",
        "color": "tab:blue",
        "error_color": "lightsteelblue",
        "marker": "D",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-softmax-expected-skip-5": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_softmax_expected_skip_full_generation_kv_full_compute",  # noqa: E501
        "exact_vals": [0.84, 0.85, 0.86, 0.87, 0.88, 0.9, 0.91, 0.92, 0.94, 0.95, 0.97, 0.99],  # noqa: E501
        "display_name": "Softmax Expected ($k$=5)",
        "color": "tab:orange",
        "error_color": "navajowhite",
        "marker": "v",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-softmax-expected-skip-10": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_softmax_expected_skip_k_10_full_generation_kv_full_compute",  # noqa: E501
        "exact_vals": [0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], # noqa: E501
        # noqa: E501
        "display_name": "Softmax Expected ($k$=10)",
        "color": "tab:green",
        "error_color": "lightgreen",
        "marker": "D",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-softmax-expected-skip-20": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_softmax_expected_skip_k_20_full_generation_kv_full_compute",  # noqa: E501
        "exact_vals": [0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95,
                       0.96, 0.97, 0.98, 0.99],  # noqa: E501
        # noqa: E501
        "display_name": "Softmax Expected ($k$=20)",
        "color": "tab:green",
        "error_color": "lightgreen",
        "marker": "D",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-softmax-expected-skip-50": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_softmax_expected_skip_k_50_full_generation_kv_full_compute",  # noqa: E501
        "exact_vals": [0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95,
                       0.96, 0.97, 0.98, 0.99],  # noqa: E501
        # noqa: E501
        "display_name": "Softmax Expected ($k$=50)",
        "color": "tab:blue",
        "error_color": "lightsteelblue",
        "marker": "D",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-200-softmax-expected-skip-5": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_200s_128t_softmax_expected_skip_k_5_full_generation_kv_full_compute",  # noqa: E501
        "exact_vals": [0.85, 0.86, 0.87, 0.88, 0.9, 0.91, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],  # noqa: E501
        "display_name": "Softmax Expected ($k$=5)",
        "color": "tab:orange",
        "error_color": "navajowhite",
        "marker": "v",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-200-softmax-expected-skip-20": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_200s_128t_softmax_expected_skip_k_20_full_generation_kv_full_compute",  # noqa: E501
        "exact_vals": [0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95,
                       0.96, 0.97, 0.98, 0.99],  # noqa: E501
        # noqa: E501
        "display_name": "Softmax Expected ($k$=20)",
        "color": "tab:green",
        "error_color": "lightgreen",
        "marker": "D",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-200-softmax-expected-skip-50": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_200s_128t_softmax_expected_skip_k_50_full_generation_kv_full_compute",  # noqa: E501
        "exact_vals": [0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95,
                       0.96, 0.97, 0.98, 0.99],  # noqa: E501
        # noqa: E501
        "display_name": "Softmax Expected ($k$=50)",
        "color": "tab:blue",
        "error_color": "lightsteelblue",
        "marker": "D",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-consensus-decay": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_consensus_decay_full_generation_kv_full_compute", # noqa: E501
        "exact_vals": [0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], # noqa: E501
        "display_name": "Consensus Decay",
        "color": "tab:red",
        "error_color": "lightcoral",
        "marker": "s",
        "inject_no_skip": True,
        "show_labels": False,
    },
    "wmt-semantic-boundary": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_semantic_boundary_full_generation_kv_full_compute", # noqa: E501
        "exact_vals": [0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], # noqa: E501
        "display_name": "Semantic Boundary",
        "color": "tab:green",
        "error_color": "lightgreen",
        "marker": "D",
        "inject_no_skip": True,
        "show_labels": False,
    },
    # safe-knn thresholds
    "wmt-safe-knn-thresholds": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_safe_knn_full_generation_kv_full_compute",
        "exact_vals": [0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
    },
    # wmt c1 thresholds
    "wmt-c1-thresholds":{
        "results_dir": RESULTS_DIR_WMT_C1,
        "prefix": "wmt19_test_100s_128t_top1_strict_full_generation_kv_full_compute",
        "exact_vals": [0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0],
    },
    "wmt-c4-thresholds": {
        "results_dir": RESULTS_DIR_WMT,
        "prefix": "wmt19_test_100s_128t_top1_strict_full_generation_kv_full_compute",
        "exact_vals": [0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0],
    },
    # cal runs
    "sharegpt-trial-55": {
        "results_dir": RESULTS_DIR_SHAREGPT_CAL,
        "prefix": "trial_55",
        "exact_vals": None,
    },
    "sharegpt-trial-56": {
        "results_dir": RESULTS_DIR_SHAREGPT_CAL,
        "prefix": "trial_56",
        "exact_vals": None,
    },
    # checkpoint
    "e2e-checkpoint": {
        "results_dir": RESULTS_DIR_E2E,
        "prefix": "e2e_test_100s_128t",
        "exact_vals": [0.94, 0.96, 0.98, 0.99, 0.995, 1.0, 1.002, 1.005],
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

    if ACTIVE_FIGURE not in FIGURES_CONFIG:
        logging.error("Active figure '%s' not found in FIGURES_CONFIG.", ACTIVE_FIGURE)
        sys.exit(1)

    active_conf = FIGURES_CONFIG[ACTIVE_FIGURE]
    active_main_experiments = active_conf.get("main_experiments", [])
    active_baselines = active_conf.get("baselines", [])
    experiment_title = active_conf.get("experiment_title", "Experiment Plots")
    confidence = active_conf.get("confidence", 0.95)
    target_threshold = active_conf.get("target_threshold", None)
    plot_types = active_conf.get("plot_types", [])
    force_pareto = active_conf.get("force_pareto", False)
    y_bounds = active_conf.get("y_bounds", None)
    show_colorbar = active_conf.get("show_colorbar", True)

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
    experiment_metrics = active_conf.get("metrics", None) or standard_metrics

    # dynamically set quality flags based on the config
    PLOT_PARETO_FRONTIER = "pareto_frontier" in plot_types
    PLOT_THRESHOLD_SENSITIVITY = "threshold_sensitivity" in plot_types
    PLOT_SKIP_ACCEPTANCE_RATE = "skip_acceptance_rate" in plot_types
    PLOT_GROUPED_TOKEN_DISTRIBUTION = "grouped_token_distribution" in plot_types
    PLOT_CHECKPOINT_SKIP_HEATMAP = "checkpoint_skip_heatmap" in plot_types
    PLOT_TOKEN_SKIP_HISTOGRAM = "token_skip_histogram" in plot_types
    PLOT_PROMPT_LENGTH_VS_SKIPPED = "prompt_length_vs_skipped" in plot_types
    PLOT_DB_UTILISATION = "db_utilisation" in plot_types

    # load experiments data
    loaded_main_experiments = []
    for exp_key in active_main_experiments:
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
    for b_key in active_baselines:
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

    file_suffix = "_vs_".join(active_main_experiments)

    if PLOT_PARETO_FRONTIER:
        logging.info("Generating Standard Quality Plots...")
        for metric in experiment_metrics:
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
                confidence=confidence,
                force_pareto=force_pareto,
                y_bounds=y_bounds,
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

    # retrieve the specific row matching the target threshold, or take the single-threshodl file
    if target_threshold is not None and "threshold" in df_agg.columns:
        target_row_df = df_agg[df_agg["threshold"] == target_threshold]
    else:
        # if target_threshold is None or if this is a trial run
        target_row_df = df_agg

    if not target_row_df.empty:
        target_row = target_row_df.iloc[0]

        if PLOT_CHECKPOINT_SKIP_HEATMAP:
            logging.info(f"Plotting Heatmap for Threshold {target_threshold}...")
            plot_checkpoint_skip_heatmap(
                target_row,
                root_plot_dir=experiment_plots_dir,
                show_colorbar=show_colorbar,
            )

        if PLOT_TOKEN_SKIP_HISTOGRAM:
            logging.info(
                f"Plotting Token Skip Histogram for Threshold {target_threshold}..."
            )
            plot_token_skip_histogram(target_row, root_plot_dir=experiment_plots_dir)

        if PLOT_PROMPT_LENGTH_VS_SKIPPED and not df_samples.empty:
            logging.info(
                f"Plotting Length vs Skipped for Threshold {target_threshold}..."
            )
            plot_generated_length_vs_skipped(
                df_samples,
                target_threshold=target_threshold,
                root_plot_dir=experiment_plots_dir,
            )

        if PLOT_DB_UTILISATION:
            logging.info(
                f"Plotting DB Utilisation Plots for Threshold {target_threshold}..."
            )
            plot_db_active_usage(target_row, root_plot_dir=experiment_plots_dir)
            plot_vector_zipf_curve(target_row, root_plot_dir=experiment_plots_dir)
            plot_vector_hit_distribution(target_row, root_plot_dir=experiment_plots_dir)
    else:
        logging.warning(f"Target threshold {target_threshold} not found in the data!")
