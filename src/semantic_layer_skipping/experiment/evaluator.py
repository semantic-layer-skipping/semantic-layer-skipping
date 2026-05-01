from collections import defaultdict

import evaluate
import Levenshtein
from data.extractor import AnswerExtractor
from data.loader import BatchedDataset
from experiment.config import EvalConfig
from inference.base_runner import SemanticSkipRunner
from inference.strategies import get_decision_strategy
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from store import SkippingVectorDB
from structures import EvalStrategy
from tqdm import tqdm


def _calc_bleu(ref_text: str, hyp_text: str, smoothing_method) -> float:
    """Calculates BLEU score even with empty strings."""
    ref_tokens = str(ref_text).split() if ref_text else [""]
    hyp_tokens = str(hyp_text).split() if hyp_text else [""]
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing_method)


def _calc_rouge(
    ref_text: str, hyp_text: str, scorer: rouge_scorer.RougeScorer
) -> float:
    """Calculates ROUGE-L f-measure."""
    ref_str = str(ref_text) if ref_text is not None else ""
    hyp_str = str(hyp_text) if hyp_text is not None else ""
    return scorer.score(ref_str, hyp_str)["rougeL"].fmeasure


def _calc_bertscore(ref_text: str, hyp_text: str, scorer, eval_bert=True) -> float:
    """Calculates BERTScore F1."""
    if not eval_bert:
        return -1

    ref_str = str(ref_text).strip() if ref_text else ""
    hyp_str = str(hyp_text).strip() if hyp_text else ""

    # BERTScore fails on completely empty strings.
    # if both are empty, perfect match. if only one is empty, 0.
    if not ref_str or not hyp_str:
        return 1.0 if ref_str == hyp_str else 0.0

    res = scorer.compute(
        predictions=[hyp_str],
        references=[ref_str],
        lang="en",
        rescale_with_baseline=True,
        model_type="roberta-large",
    )
    return res["f1"][0]


def _calc_token_accuracy(ref_tokens: list[int], hyp_tokens: list[int]) -> float:
    """Calculates token accuracy."""
    # token accuracy - how many tokens in the reference match the baseline

    # if they have different number of tokens,
    # we compare up to the length of the shorter one
    min_len = min(len(ref_tokens), len(hyp_tokens))

    if min_len == 0:
        # if both are empty, perfect match. if only one is empty, 0 accuracy.
        return 1.0 if len(ref_tokens) == len(hyp_tokens) else 0.0

    matches = sum(1 for i in range(min_len) if ref_tokens[i] == hyp_tokens[i])
    token_accuracy = matches / min_len if min_len > 0 else 1.0
    return token_accuracy


def run_eval_loop(
    runner: SemanticSkipRunner,
    db: SkippingVectorDB,
    thresholds: dict[int, float],
    config: EvalConfig,
    dataset: BatchedDataset,
    eval_bert: bool = True,
    discovery_stats=None,
) -> dict:
    metrics = {
        "exact_matches": 0,
        "fuzzy_matches": 0,
        "task_correctness": 0,
        "baseline_task_correctness": 0,
        "incremental_agreement_rate": 0.0,
        "total_skipped_layers": 0,
        "total_possible_layers": 0,
        # scores computed from skipped against baseline
        "bleu_scores": [],
        "rouge_l_scores": [],
        "bert_scores": [],
        "token_accuracies": [],
        # scores computed from skipped and baseline against the label
        "label_bleu_scores": [],
        "label_rouge_l_scores": [],
        "label_bert_scores": [],
        "label_token_accuracies": [],
        "baseline_label_bleu_scores": [],
        "baseline_label_rouge_l_scores": [],
        "baseline_label_bert_scores": [],
        "baseline_label_token_accuracies": [],
        # raw samples
        "samples": [],
    }

    global_skip_stats = defaultdict(lambda: defaultdict(int))
    global_hit_counts = defaultdict(lambda: defaultdict(int))
    global_index_sizes = {}
    global_token_skip_distribution = defaultdict(int)
    global_request_skip_distribution = defaultdict(int)

    # setup scorers
    rouge_calc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    chen_cherry = SmoothingFunction()
    bert_calc = evaluate.load("bertscore") if eval_bert else None

    for sample in tqdm(dataset, desc="evaluating"):
        # run generation with skipping enabled
        skip_res = runner.generate_with_skipping(
            sample,
            db,
            threshold=thresholds,
            max_total_tokens=config.max_total_tokens,
            log_skips=False,
            decision_strategy=get_decision_strategy(
                config.online_decision_strategy_mode
            ),
            discovery_stats=discovery_stats,
            injection_strategy=config.injection_strategy_mode,
            kv_strategy=config.kv_strategy_mode,
        )

        sample_data = {
            "id": sample.id,
            "prompt": sample.prompt,
            "generated_text": skip_res.generated_text,
            "skipped_count": skip_res.skipped_layers,
            "generated_token_count": skip_res.generated_token_count,
            # track per-sample skipping and DB metrics
            "checkpoint_skip_stats": skip_res.checkpoint_skip_counts,
            "db_hit_counts": skip_res.db_hit_counts,
        }

        # aggregate stats globally across the entire dataset
        for ckpt, stats in skip_res.checkpoint_skip_counts.items():
            for skip_amount, count in stats.items():
                global_skip_stats[ckpt][skip_amount] += count

        for ckpt, hits in skip_res.db_hit_counts.items():
            for neighbour_id, count in hits.items():
                global_hit_counts[ckpt][neighbour_id] += count

        if skip_res.db_index_sizes:
            global_index_sizes = skip_res.db_index_sizes

        for skip_amt, count in skip_res.token_skip_distribution.items():
            global_token_skip_distribution[skip_amt] += count

        global_request_skip_distribution[skip_res.skipped_layers] += 1

        # task accuracy check
        if sample.label is not None:
            extracted_answer = AnswerExtractor.extract(
                config.dataset, skip_res.full_text
            )

            is_correct = _normalise_answer_string(
                extracted_answer
            ) == _normalise_answer_string(sample.label)
            if is_correct:
                metrics["task_correctness"] += 1

            # evaluate skipped-generation against the ground-truth label
            l_bleu = _calc_bleu(
                sample.label, skip_res.generated_text, chen_cherry.method1
            )
            l_rouge = _calc_rouge(sample.label, skip_res.generated_text, rouge_calc)
            l_bert = _calc_bertscore(
                sample.label, skip_res.generated_text, bert_calc, eval_bert
            )

            sample_label_tokens = runner.model.tokenizer.encode(
                sample.label, add_special_tokens=False
            )
            l_token_accuracy = _calc_token_accuracy(
                sample_label_tokens, skip_res.generated_tokens
            )

            metrics["label_bleu_scores"].append(l_bleu)
            metrics["label_rouge_l_scores"].append(l_rouge)
            metrics["label_bert_scores"].append(l_bert)
            metrics["label_token_accuracies"].append(l_token_accuracy)

            sample_data.update(
                {
                    "label": sample.label,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "label_bleu": l_bleu,
                    "label_rouge": l_rouge,
                    "label_bert": l_bert,
                    "label_token_accuracy": l_token_accuracy,
                }
            )

        # strategy: full generation comparison
        # generates the full baseline text to compare quality metrics
        if config.strategy == EvalStrategy.FULL_GENERATION:
            # run baseline without skipping (vector_db=None)
            base_res = runner.generate_with_skipping(
                sample,
                vector_db=None,
                max_total_tokens=config.max_total_tokens,
                log_skips=False,
            )

            # baseline task accuracy check
            if sample.label is not None:
                base_extracted = AnswerExtractor.extract(
                    config.dataset, base_res.full_text
                )
                base_is_correct = _normalise_answer_string(
                    base_extracted
                ) == _normalise_answer_string(sample.label)

                # evaluate baseline-generation against the ground-truth label
                b_l_bleu = _calc_bleu(
                    sample.label, base_res.generated_text, chen_cherry.method1
                )
                b_l_rouge = _calc_rouge(
                    sample.label, base_res.generated_text, rouge_calc
                )
                b_l_bert = _calc_bertscore(
                    sample.label, base_res.generated_text, bert_calc, eval_bert
                )
                b_l_token_accuracy = _calc_token_accuracy(
                    sample_label_tokens, base_res.generated_tokens
                )

                metrics["baseline_label_bleu_scores"].append(b_l_bleu)
                metrics["baseline_label_rouge_l_scores"].append(b_l_rouge)
                metrics["baseline_label_bert_scores"].append(b_l_bert)
                metrics["baseline_label_token_accuracies"].append(b_l_token_accuracy)

                if base_is_correct:
                    metrics["baseline_task_correctness"] += 1

                sample_data.update(
                    {
                        "baseline_extracted_answer": base_extracted,
                        "baseline_is_correct": base_is_correct,
                        "baseline_label_bleu": b_l_bleu,
                        "baseline_label_rouge": b_l_rouge,
                        "baseline_label_bert": b_l_bert,
                        "baseline_label_token_accuracy": b_l_token_accuracy,
                    }
                )

            # calculate metrics (skipped vs baseline)
            is_exact = base_res.generated_text == skip_res.generated_text
            similarity = Levenshtein.ratio(
                base_res.generated_text, skip_res.generated_text
            )
            # update counters
            if is_exact:
                metrics["exact_matches"] += 1
            if similarity > 0.9:
                metrics["fuzzy_matches"] += 1

            # n-gram and semantic metrics (baseline vs skipped)
            bleu = _calc_bleu(
                base_res.generated_text, skip_res.generated_text, chen_cherry.method1
            )
            rouge = _calc_rouge(
                base_res.generated_text, skip_res.generated_text, rouge_calc
            )
            bert_score = _calc_bertscore(
                base_res.generated_text, skip_res.generated_text, bert_calc, eval_bert
            )

            metrics["bleu_scores"].append(bleu)
            metrics["rouge_l_scores"].append(rouge)
            metrics["bert_scores"].append(bert_score)

            token_accuracy = _calc_token_accuracy(
                base_res.generated_tokens, skip_res.generated_tokens
            )
            metrics["token_accuracies"].append(token_accuracy)

            sample_data.update(
                {
                    "baseline_text": base_res.generated_text,
                    "similarity": similarity,
                    "bleu": bleu,
                    "rouge": rouge,
                    "bert_score": bert_score,
                    "num_baseline_generated_tokens": len(base_res.generated_tokens),
                    "token_accuracy": token_accuracy,
                }
            )

        # strategy: incremental token match
        # checks if skipped model matches baseline step-by-step
        elif config.strategy == EvalStrategy.INCREMENTAL_MATCH:
            # extract the full sequence of tokens generated by the skipping model
            full_gen_tokens = skip_res.generated_tokens
            base_prompt_text = sample

            matches = 0
            mismatch_indices = []

            # iterate through the sequence generated by the skipping model
            for i, actual_token_id in enumerate(full_gen_tokens):
                # construct the prompt for the baseline model at this step
                if i > 0:
                    current_token_ids = skip_res.prompt_tokens + full_gen_tokens[:i]
                    current_prompt = runner.model.to_string(current_token_ids)
                    format_prompt = False
                else:
                    current_prompt = base_prompt_text
                    format_prompt = True

                # run baseline (vector_db=None) for 1 generated token
                baseline_res = runner.generate_with_skipping(
                    current_prompt,
                    vector_db=None,
                    max_total_tokens=1,
                    format_prompt=format_prompt,
                    log_skips=False,
                )
                # handle edge case where baseline generates nothing - EOS
                if not baseline_res.generated_tokens:
                    mismatch_indices.append(i)
                    continue

                # extract the single token it predicted
                baseline_token_id = baseline_res.generated_tokens[0]

                # compare
                if baseline_token_id == actual_token_id:
                    matches += 1
                else:
                    mismatch_indices.append(i)

            # calculate agreement
            total_gen = len(full_gen_tokens)
            agreement = matches / total_gen if total_gen > 0 else 1.0

            metrics["incremental_agreement_rate"] += agreement

            sample_data.update(
                {
                    "agreement_rate": agreement,
                    "mismatch_indices": mismatch_indices,
                    "total_tokens_checked": total_gen,
                }
            )

        # calculate efficiency metrics
        # total possible layers = model depth * number of new tokens generated
        n_layers = runner.model.n_layers

        possible = n_layers * skip_res.generated_token_count

        metrics["total_possible_layers"] += possible
        metrics["total_skipped_layers"] += skip_res.skipped_layers

        metrics["samples"].append(sample_data)

    # aggregation
    total = len(dataset)

    # calculates average agreement if strategy is incremental
    def _safe_avg(lst):
        return sum(lst) / len(lst) if lst and len(lst) > 0 else 0.0

    avg_agreement = 0.0
    if config.strategy == EvalStrategy.INCREMENTAL_MATCH and total > 0:
        avg_agreement = metrics["incremental_agreement_rate"] / total

    # efficiency calculations
    total_possible = metrics["total_possible_layers"] + 1e-9
    skip_fraction = metrics["total_skipped_layers"] / total_possible
    theoretical_speedup = 1 / (1 - skip_fraction + 1e-9)
    avg_skipped_per_token = metrics["total_skipped_layers"] / total_possible

    summary = {
        "strategy": str(config.strategy),
        "accuracy": {
            "task_accuracy": metrics["task_correctness"] / total if total > 0 else 0,
            "baseline_task_accuracy": metrics["baseline_task_correctness"] / total
            if total > 0
            else 0,
            "exact_match_pct": metrics["exact_matches"] / total if total > 0 else 0,
            "fuzzy_match_pct": metrics["fuzzy_matches"] / total if total > 0 else 0,
            # averages for baseline vs skipped
            "avg_bleu": _safe_avg(metrics["bleu_scores"]),
            "avg_rouge_l": _safe_avg(metrics["rouge_l_scores"]),
            "avg_bert_score": _safe_avg(metrics["bert_scores"]),
            "avg_token_accuracy": _safe_avg(metrics["token_accuracies"]),
            # averages for generated texts against ground-truth labels
            "avg_label_bleu": _safe_avg(metrics["label_bleu_scores"]),
            "avg_baseline_label_bleu": _safe_avg(metrics["baseline_label_bleu_scores"]),
            "avg_label_rouge_l": _safe_avg(metrics["label_rouge_l_scores"]),
            "avg_baseline_label_rouge_l": _safe_avg(
                metrics["baseline_label_rouge_l_scores"]
            ),
            "avg_label_bert_score": _safe_avg(metrics["label_bert_scores"]),
            "avg_baseline_label_bert_score": _safe_avg(
                metrics["baseline_label_bert_scores"]
            ),
            "avg_label_token_accuracy": _safe_avg(metrics["label_token_accuracies"]),
            "avg_baseline_label_token_accuracy": _safe_avg(
                metrics["baseline_label_token_accuracies"]
            ),
            # incremental evaluation
            "incremental_token_accuracy": avg_agreement,
        },
        "efficiency": {
            "avg_skipped_per_token": avg_skipped_per_token,
            "theoretical_speedup": theoretical_speedup,
            # global counts
            "global_checkpoint_skip_counts": {
                k: dict(v) for k, v in global_skip_stats.items()
            },
            "global_db_hit_counts": {k: dict(v) for k, v in global_hit_counts.items()},
            "db_index_sizes": global_index_sizes,
            "global_token_skip_distribution": dict(global_token_skip_distribution),
            "global_request_skip_distribution": dict(global_request_skip_distribution),
        },
        "samples": metrics["samples"],
    }

    return summary


def _normalise_answer_string(s):
    if s is None:
        return None
    try:
        # if the answer is a number, convert to float for comparison
        return float(s)
    except ValueError:
        return s.strip().lower()
