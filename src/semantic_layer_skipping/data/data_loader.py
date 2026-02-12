from datasets import load_dataset
from utils import ISAAC_NEWTON_QUESTIONS, question_to_prompt


class DatasetFactory:
    @staticmethod
    def get_prompts(dataset_name: str, split: str, n_samples: int):
        if dataset_name == "newton":
            base_prompts = [question_to_prompt(q) for q in ISAAC_NEWTON_QUESTIONS]
            train_dataset = base_prompts[:3]
            calibration_dataset = base_prompts[3:7]
            test_dataset = base_prompts[7:]  # 3 prompts
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
                if len(item["text"]) > 50:  # filter empty lines
                    # take first 100 chars as prompt
                    prompts.append(item["text"][:100])
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
