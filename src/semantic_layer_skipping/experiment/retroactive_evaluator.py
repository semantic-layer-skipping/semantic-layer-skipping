import json
import logging
import os
import shutil
from pathlib import Path

import evaluate
from data.loader import DatasetFactory
from experiment.evaluator import (
    _calc_bertscore,
    _calc_bleu,
    _calc_rouge,
    _calc_token_accuracy,
)
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
from structures import DatasetName, DatasetSplit
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import set_logging_config


def _safe_avg(lst: list) -> float:
    return sum(lst) / len(lst) if lst and len(lst) > 0 else 0.0


def retroactively_add_metrics(
    file_paths: list[str], model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
):
    """
    Reads old evaluation JSONs, computes missing BERT and label-based metrics,
    and overwrites the files with the updated data.
    """
    rouge_calc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    chen_cherry = SmoothingFunction()
    bert_calc = evaluate.load("bertscore")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for file_path in file_paths:
        if not os.path.exists(file_path):
            logging.info(f"File not found: {file_path}. Skipping.")
            continue

        logging.info(f"\nProcessing: {file_path}")
        with open(file_path) as f:
            data = json.load(f)

        # extract config
        config = data["config"]
        dataset_name = DatasetName(config["dataset"])
        split = DatasetSplit(config["split"])
        max_total_tokens = config["max_total_tokens"]

        # load the full dataset split to build an ID -> Label lookup map
        logging.info(
            f"Loading dataset '{dataset_name}' to extract ground-truth labels..."
        )
        full_dataset = DatasetFactory.get_dataset(
            dataset_name,
            split,
            # we use all num_samples to load the entire split,
            # ensuring we find all IDs regardless of shuffling
            n_samples=999_999,
            tokenizer=tokenizer,
            max_total_tokens=max_total_tokens,
        )

        id_to_label = {
            sample.id: sample.label
            for sample in full_dataset.samples
            if sample.label is not None
        }

        # setup metric trackers for aggregation
        new_metrics = {
            "bert_scores": [],
            "label_bleu_scores": [],
            "label_rouge_l_scores": [],
            "label_bert_scores": [],
            "label_token_accuracies": [],
            "baseline_label_bleu_scores": [],
            "baseline_label_rouge_l_scores": [],
            "baseline_label_bert_scores": [],
            "baseline_label_token_accuracies": [],
        }

        # retroactively compute missing scores for each sample
        for sample_data in tqdm(data["metrics"]["samples"], desc="Updating Samples"):
            gen_text = sample_data["generated_text"]
            base_text = sample_data["baseline_text"]
            sample_id = sample_data["id"]

            # calculate missing baseline vs skipped semantic metric
            if base_text and gen_text and "bert_score" not in sample_data:
                bert_score = _calc_bertscore(base_text, gen_text, bert_calc)
                sample_data["bert_score"] = bert_score
                new_metrics["bert_scores"].append(bert_score)

            # calculate label-based metrics
            label = id_to_label.get(sample_id)
            if label is not None:
                # tokenise text on-the-fly since old JSONs don't store token arrays
                label_tokens = tokenizer.encode(label, add_special_tokens=False)
                gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)
                base_tokens = tokenizer.encode(base_text, add_special_tokens=False)

                # skipped vs label
                if "label_bleu" not in sample_data:
                    l_bleu = _calc_bleu(label, gen_text, chen_cherry.method1)
                    l_rouge = _calc_rouge(label, gen_text, rouge_calc)
                    l_bert = _calc_bertscore(label, gen_text, bert_calc)
                    l_tok_acc = _calc_token_accuracy(label_tokens, gen_tokens)

                    sample_data.update(
                        {
                            "label": label,
                            "label_bleu": l_bleu,
                            "label_rouge": l_rouge,
                            "label_bert": l_bert,
                            "label_token_accuracy": l_tok_acc,
                        }
                    )
                    new_metrics["label_bleu_scores"].append(l_bleu)
                    new_metrics["label_rouge_l_scores"].append(l_rouge)
                    new_metrics["label_bert_scores"].append(l_bert)
                    new_metrics["label_token_accuracies"].append(l_tok_acc)

                # baseline vs label
                if base_text and "baseline_label_bleu" not in sample_data:
                    b_l_bleu = _calc_bleu(label, base_text, chen_cherry.method1)
                    b_l_rouge = _calc_rouge(label, base_text, rouge_calc)
                    b_l_bert = _calc_bertscore(label, base_text, bert_calc)
                    b_l_tok_acc = _calc_token_accuracy(label_tokens, base_tokens)

                    sample_data.update(
                        {
                            "baseline_label_bleu": b_l_bleu,
                            "baseline_label_rouge": b_l_rouge,
                            "baseline_label_bert": b_l_bert,
                            "baseline_label_token_accuracy": b_l_tok_acc,
                        }
                    )
                    new_metrics["baseline_label_bleu_scores"].append(b_l_bleu)
                    new_metrics["baseline_label_rouge_l_scores"].append(b_l_rouge)
                    new_metrics["baseline_label_bert_scores"].append(b_l_bert)
                    new_metrics["baseline_label_token_accuracies"].append(b_l_tok_acc)

        # update top-level averages
        accuracy_dict = data["metrics"]["accuracy"]

        if new_metrics["bert_scores"]:
            accuracy_dict["avg_bert_score"] = _safe_avg(new_metrics["bert_scores"])

        if new_metrics["label_bleu_scores"]:
            accuracy_dict["avg_label_bleu"] = _safe_avg(
                new_metrics["label_bleu_scores"]
            )
            accuracy_dict["avg_label_rouge_l"] = _safe_avg(
                new_metrics["label_rouge_l_scores"]
            )
            accuracy_dict["avg_label_bert_score"] = _safe_avg(
                new_metrics["label_bert_scores"]
            )
            accuracy_dict["avg_label_token_accuracy"] = _safe_avg(
                new_metrics["label_token_accuracies"]
            )

        if new_metrics["baseline_label_bleu_scores"]:
            accuracy_dict["avg_baseline_label_bleu"] = _safe_avg(
                new_metrics["baseline_label_bleu_scores"]
            )
            accuracy_dict["avg_baseline_label_rouge_l"] = _safe_avg(
                new_metrics["baseline_label_rouge_l_scores"]
            )
            accuracy_dict["avg_baseline_label_bert_score"] = _safe_avg(
                new_metrics["baseline_label_bert_scores"]
            )
            accuracy_dict["avg_baseline_label_token_accuracy"] = _safe_avg(
                new_metrics["baseline_label_token_accuracies"]
            )

        file_p = Path(file_path)
        backup_path = file_p.with_name(f"ORIGINAL_{file_p.name}")
        shutil.copy2(file_path, backup_path)

        # save the updated JSON
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        logging.info(f"Backup created at: {backup_path}")
        logging.info(f"Successfully updated and saved: {file_path}")


if __name__ == "__main__":
    set_logging_config()

    old_files = [
        "hpc/experiments/batch_20260309_042303_Qwen2.5-1.5B-Instruct_sharegpt_train_20000s_2048t_strict_strict_match_c4-8-12-16-20-24/manual_eval_results_256tokens_50samples_fixed_thresh/sharegpt_test_50s_256t_full_generation_thresh-95-95-95-95-95-95.json",
    ]

    retroactively_add_metrics(
        file_paths=old_files, model_name="Qwen/Qwen2.5-1.5B-Instruct"
    )
