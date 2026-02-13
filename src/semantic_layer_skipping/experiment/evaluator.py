import Levenshtein
from data.data_loader import DatasetFactory
from experiment.config import TestConfig
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from structures import EvalStrategy
from tqdm import tqdm


def run_test_loop(runner, db, thresholds: dict[int, float], config: TestConfig) -> dict:
    # load prompts using the factory
    prompts = DatasetFactory.get_prompts(
        config.dataset, config.split, config.num_samples
    )

    metrics = {
        "exact_matches": 0,
        "fuzzy_matches": 0,
        "total_skipped_layers": 0,
        "total_possible_layers": 0,
        "bleu_scores": [],
        "rouge_l_scores": [],
        "samples": [],
    }

    # setup scorers
    rouge_calc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    chen_cherry = SmoothingFunction()

    for prompt in tqdm(prompts, desc="testing"):
        # run generation with skipping enabled
        skip_res = runner.generate_with_skipping(
            prompt, db, threshold=thresholds, max_new_tokens=config.max_gen_tokens
        )

        sample_data = {
            "prompt": prompt,
            "skipped_text": skip_res.generated_text,
            "skipped_count": skip_res.skipped_layers,
        }

        # strategy: full generation comparison
        # generates the full baseline text to compare quality metrics
        if config.strategy == EvalStrategy.FULL_GENERATION:
            # run baseline without skipping (vector_db=None)
            base_res = runner.generate_with_skipping(
                prompt, vector_db=None, max_new_tokens=config.max_gen_tokens
            )

            # 1. string distance metrics
            is_exact = base_res.generated_text == skip_res.generated_text
            similarity = Levenshtein.ratio(
                base_res.generated_text, skip_res.generated_text
            )

            # 2. n-gram metrics (bleu/rouge)
            # approximate tokenization with split() for metric calculation
            ref_tokens = base_res.generated_text.split()
            hyp_tokens = skip_res.generated_text.split()

            # handle empty generation edge cases
            if not ref_tokens:
                ref_tokens = [""]
            if not hyp_tokens:
                hyp_tokens = [""]

            bleu = sentence_bleu(
                [ref_tokens], hyp_tokens, smoothing_function=chen_cherry.method1
            )
            rouge = rouge_calc.score(base_res.generated_text, skip_res.generated_text)[
                "rougeL"
            ].fmeasure

            # update counters
            if is_exact:
                metrics["exact_matches"] += 1
            if similarity > 0.9:
                metrics["fuzzy_matches"] += 1

            metrics["bleu_scores"].append(bleu)
            metrics["rouge_l_scores"].append(rouge)

            sample_data.update(
                {
                    "baseline_text": base_res.generated_text,
                    "similarity": similarity,
                    "bleu": bleu,
                    "rouge": rouge,
                }
            )

        # strategy: incremental token match
        # checks if skipped model matches baseline step-by-step
        elif config.strategy == EvalStrategy.INCREMENTAL_MATCH:
            # TODO: implement incremental logic if required
            pass

        # calculate efficiency metrics
        # total possible layers = model depth * number of new tokens generated
        n_layers = runner.model.cfg.n_layers
        possible = n_layers * skip_res.generated_token_count

        metrics["total_possible_layers"] += possible
        metrics["total_skipped_layers"] += skip_res.skipped_layers

        metrics["samples"].append(sample_data)

    # aggregation
    total = len(prompts)
    if total > 0:
        avg_bleu = sum(metrics["bleu_scores"]) / total
        avg_rouge = sum(metrics["rouge_l_scores"]) / total

        # theoretical speedup = 1 / (1 - skip_fraction)
        # add epsilon to prevent division by zero
        skip_fraction = metrics["total_skipped_layers"] / (
            metrics["total_possible_layers"] + 1e-9
        )
        theoretical_speedup = 1 / (1 - skip_fraction + 1e-9)

        avg_skipped_per_token = metrics["total_skipped_layers"] / (
            metrics["total_possible_layers"] + 1e-9
        )
    else:
        avg_bleu = 0
        avg_rouge = 0
        theoretical_speedup = 1.0
        avg_skipped_per_token = 0

    summary = {
        "strategy": str(config.strategy),
        "accuracy": {
            "exact_match_pct": metrics["exact_matches"] / total if total > 0 else 0,
            "fuzzy_match_pct": metrics["fuzzy_matches"] / total if total > 0 else 0,
            "avg_bleu": avg_bleu,
            "avg_rouge_l": avg_rouge,
        },
        "efficiency": {
            "avg_skipped_per_token": avg_skipped_per_token,
            "theoretical_speedup": theoretical_speedup,
        },
        "samples": metrics["samples"],
    }

    return summary
