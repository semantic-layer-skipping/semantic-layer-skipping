from datasets import load_dataset
from utils import (
    ISAAC_NEWTON_QUESTIONS_CALIBRATION,
    ISAAC_NEWTON_QUESTIONS_TEST,
    ISAAC_NEWTON_QUESTIONS_TRAIN,
    question_to_prompt,
)

MIN_WIKITEXT_LEN = 50
PROMPT_WIKITEXT_CHAR_LIMIT = 100


class DatasetFactory:
    @staticmethod
    def get_prompts(dataset_name: str, split: str, n_samples: int) -> list[str]:
        if dataset_name == "newton":
            train_dataset = [
                question_to_prompt(q) for q in ISAAC_NEWTON_QUESTIONS_TRAIN
            ]
            calibration_dataset = [
                question_to_prompt(q) for q in ISAAC_NEWTON_QUESTIONS_CALIBRATION
            ]
            test_dataset = [question_to_prompt(q) for q in ISAAC_NEWTON_QUESTIONS_TEST]
            if split == "train":
                return train_dataset[:n_samples]
            elif split == "validation":
                return calibration_dataset[:n_samples]
            elif split == "test":
                return test_dataset[:n_samples]
            else:
                raise ValueError(f"Unknown split: {split}")

        elif dataset_name == "wikitext":
            # unlabelled text completion
            ds = load_dataset(
                "wikitext", "wikitext-2-raw-v1", split=split, streaming=True
            )
            prompts = []
            for item in ds.take(n_samples):
                if len(item["text"]) > MIN_WIKITEXT_LEN:  # filter empty lines
                    # take first 100 chars as prompt
                    prompts.append(item["text"][:PROMPT_WIKITEXT_CHAR_LIMIT])
            return prompts

        elif dataset_name == "gsm8k":
            # labelled maths problems
            ds = load_dataset("gsm8k", "main", split=split, streaming=True)
            prompts = []
            for item in ds.take(n_samples):
                prompt = question_to_prompt(item["question"])
                prompts.append(prompt)
            return prompts
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
