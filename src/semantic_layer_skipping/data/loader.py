import gzip
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator

from datasets import load_dataset
from structures import DatasetName, DatasetSample, DatasetSplit
from utils import (
    ISAAC_NEWTON_QUESTIONS_CALIBRATION,
    ISAAC_NEWTON_QUESTIONS_TEST,
    ISAAC_NEWTON_QUESTIONS_TRAIN,
)

DEFAULT_MAX_TOKENS = 2048


class BaseDataset(ABC):
    def __init__(
        self, split: DatasetSplit, n_samples: int, seed: int = 42, tokenizer=None
    ):
        self.split = split
        self.n_samples = n_samples
        self.seed = seed
        self.tokenizer = tokenizer

    @abstractmethod
    def load(self) -> list[DatasetSample]:
        pass


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = DEFAULT_MAX_TOKENS,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths
    (This is adapted from vLLM benchmarks)
    These are lengths in terms of number of tokens.
    Note: there is around 35k samples in ShareGPT that are valid (<2048 total)
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (
        prompt_too_short or output_too_short or prompt_too_long or combined_too_long
    )


class NewtonDataset(BaseDataset):
    def load(self) -> list[DatasetSample]:
        match self.split:
            case DatasetSplit.TRAIN:
                source = ISAAC_NEWTON_QUESTIONS_TRAIN
            case DatasetSplit.VALIDATION:
                source = ISAAC_NEWTON_QUESTIONS_CALIBRATION
            case DatasetSplit.TEST:
                source = ISAAC_NEWTON_QUESTIONS_TEST
            case _:
                raise ValueError(f"Unknown split: {self.split}")

        samples = []
        for idx, q in enumerate(source[: self.n_samples]):
            chat_messages = [{"role": "user", "content": q}]

            samples.append(
                DatasetSample(
                    id=f"newton-{self.split.value}-{idx}",
                    prompt=chat_messages,
                    label=None,
                    metadata={"source": "newton"},
                )
            )
        return samples


class GSM8KDataset(BaseDataset):
    SYSTEM_PROMPT = (
        "You are a helpful and math expert assistant. "
        "Please put the final answer within \\boxed{}."
    )

    # train/validation split ratio
    TRAIN_RATIO = 0.7

    def load(self) -> list[DatasetSample]:
        if self.split == DatasetSplit.TEST:
            ds = load_dataset("gsm8k", "main", split="test")
        else:
            # train and validation will be split from original train split
            ds = load_dataset("gsm8k", "main", split="train")

        # shuffle
        all_indices = list(range(len(ds)))
        rng = random.Random(self.seed)
        rng.shuffle(all_indices)

        # now split and limit
        if self.split == DatasetSplit.TEST:
            selected_indices = all_indices
        else:
            split_point = int(len(all_indices) * self.TRAIN_RATIO)
            if self.split == DatasetSplit.TRAIN:
                selected_indices = all_indices[:split_point]
            else:
                selected_indices = all_indices[split_point:]
        selected_indices = selected_indices[: self.n_samples]

        # build samples
        samples = []
        for i, idx in enumerate(selected_indices):
            item = ds[int(idx)]  # load by index

            chat_messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ]

            # extract clean label (just the final answer, not the full solution)
            clean_label = item["answer"].split("#### ")[-1].strip()

            samples.append(
                DatasetSample(
                    id=f"gsm8k-{self.split.value}-{i}",
                    prompt=chat_messages,
                    label=clean_label,
                    metadata={"source": "gsm8k"},
                )
            )
        return samples


class ShareGPTDataset(BaseDataset):
    """
    Loads ShareGPT style conversations from Hugging Face.
    Splits: Train (80%), Validation (10%), Test (10%).
    """

    SPLIT_RATIOS = {
        # 0.6, 0.3, 0.1 is a split of 22522, 14965, 3712
        DatasetSplit.TRAIN: (0.0, 0.6),
        DatasetSplit.VALIDATION: (0.6, 0.9),
        DatasetSplit.TEST: (0.9, 1.0),
    }

    def __init__(
        self,
        split: DatasetSplit,
        n_samples: int,
        dataset_path: str = "anon8231489123/ShareGPT_Vicuna_unfiltered",
        tokenizer=None,
        max_total_tokens: int = 2048,
    ):
        super().__init__(split, n_samples, tokenizer=tokenizer)
        self.dataset_path = dataset_path
        assert self.tokenizer is not None, (
            "ShareGPTDataset requires a tokenizer for sequence length validation"
        )
        self.max_total_tokens = max_total_tokens

    def load(self) -> list[DatasetSample]:
        # we load full dataset - 94_145 conversations
        ds = load_dataset(
            self.dataset_path,
            data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
            split="train",
        )

        # convert to list and shuffle deterministically
        all_data = [item for item in ds]
        rng = random.Random(self.seed)
        rng.shuffle(all_data)

        # split according to defined ratios
        start_ratio, end_ratio = self.SPLIT_RATIOS[self.split]
        start_idx = int(len(all_data) * start_ratio)
        end_idx = int(len(all_data) * end_ratio)
        split_data = all_data[start_idx:end_idx]

        # process and filter dynamically
        samples = []
        for idx, item in enumerate(split_data):
            if len(samples) >= self.n_samples:
                break

            # load conversations
            convs = item.get("conversations", item.get("conversation", []))
            if len(convs) < 2:
                continue

            # extract history and label
            formatted_conv = []
            for msg in convs[:-1]:
                role = "user" if msg["from"] == "human" else "assistant"
                formatted_conv.append({"role": role, "content": msg["value"]})

            output_msg = convs[-1]["value"]
            # validate sequence length
            try:
                # apply chat template to simulate the exact prompt tokens
                prompt_str = self.tokenizer.apply_chat_template(
                    formatted_conv, tokenize=False, add_generation_prompt=True
                )
                prompt_len = len(self.tokenizer.encode(prompt_str))
                output_len = len(self.tokenizer.encode(output_msg))

                if not is_valid_sequence(
                    prompt_len, output_len, max_total_len=self.max_total_tokens
                ):
                    continue

            except ValueError:
                # skip if tokenizer fails to apply chat template (e.g. malformed roles)
                continue

            samples.append(
                DatasetSample(
                    # use original id
                    id=item.get("id", f"sharegpt-{self.split.value}-{idx}"),
                    prompt=formatted_conv,
                    # set label from sharegpt
                    label=output_msg,
                    prompt_length=prompt_len,
                    tokenizer_name=self.tokenizer.name_or_path,
                    metadata={"source": "sharegpt"},
                )
            )
        return samples


class BoolQDataset(BaseDataset):
    """
    Binary True/False Questions.
    System Prompt enforces Chain of Thought + Boxed Answer.
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant. "
        "You will be given a passage and a question. "
        "Answer the question with either 'True' or 'False'. "
        "Output the final answer within \\boxed{}. "
        "Example: \\boxed{True}"
    )
    SEED = 42

    def load(self) -> list[DatasetSample]:
        if self.split == DatasetSplit.VALIDATION:
            ds = load_dataset("google/boolq", split="validation")
            start_idx, end_idx = 0, len(ds)
        else:
            ds = load_dataset("google/boolq", split="train")
            split_point = int(len(ds) * 0.8)
            if self.split == DatasetSplit.TRAIN:
                start_idx, end_idx = 0, split_point
            elif self.split == DatasetSplit.TEST:
                start_idx, end_idx = split_point, len(ds)
            else:
                raise ValueError(f"Unknown split: {self.split}")

        # deterministically shuffle and select subset
        data_slice = [ds[i] for i in range(start_idx, end_idx)]
        rng = random.Random(self.SEED)
        rng.shuffle(data_slice)
        selected_data = data_slice[: self.n_samples]

        samples = []
        for idx, item in enumerate(selected_data):
            user_content = f"Passage: {item['passage']}\nQuestion: {item['question']}?"

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            samples.append(
                DatasetSample(
                    id=f"boolq-{self.split.value}-{idx}",
                    prompt=messages,
                    label="True" if item["answer"] else "False",
                    choices=["True", "False"],
                    metadata={"source": "boolq"},
                )
            )
        return samples


# TODO: look into MMLU-pro
class MMLUDataset(BaseDataset):
    """
    Multiple Choice Questions.
    System Prompt enforces Chain of Thought + Boxed Answer (Letter).
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant. "
        "Answer the multiple choice question by selecting the best option "
        "(A, B, C, or D). "
        "Output the final answer letter within \\boxed{}. "
        "Example: \\boxed{C}"
    )

    def __init__(self, split: DatasetSplit, n_samples: int, subset: str = "all"):
        super().__init__(split, n_samples)
        self.subset = subset

    def load(self) -> list[DatasetSample]:
        if self.split == DatasetSplit.TRAIN:
            hf_split = "auxiliary_train"
        elif self.split == DatasetSplit.VALIDATION:
            hf_split = "validation"
        else:  # TEST
            hf_split = "test"

        ds = load_dataset("cais/mmlu", self.subset, split=hf_split)

        all_data = [x for x in ds]
        rng = random.Random(42)
        rng.shuffle(all_data)
        selected_data = all_data[: self.n_samples]

        samples = []
        options = ["A", "B", "C", "D"]

        for idx, item in enumerate(selected_data):
            question_text = f"{item['question']}\n"
            for i, choice in enumerate(item["choices"]):
                question_text += f"{options[i]}. {choice}\n"

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": question_text},
            ]
            samples.append(
                DatasetSample(
                    id=f"mmlu-{self.split.value}-{self.subset}-{idx}",
                    prompt=messages,
                    label=options[item["answer"]],
                    choices=options,
                    metadata={"source": "mmlu", "subset": self.subset},
                )
            )
        return samples


class QQPDataset(BaseDataset):
    """
    Loads QQP (Quora Question Pairs) from a local JSON or GZ archive.

    Workload design:
    - Train uses the "origin" question.
    - Validation (Calibration) uses the "similar" question from pairs 5k-15k.
    - Test (Evaluation) uses the "similar" question from the first 5k pairs.
    - Guarantees no exact sentence overlap between Train and Val/Test,
      and no duplicate origins anywhere.
    """

    # dataset has 30141 entries, but removing duplicates, only 20419 suitable.
    MAX_PAIRS = 30141
    VAL_OFFSET = 5000  # 5-15k is calibration
    TEST_OFFSET = 0  # 0-5k is evaluation

    SYSTEM_PROMPT = "You are a helpful assistant who answers the user's question."

    def __init__(
        self,
        split: DatasetSplit,
        n_samples: int,
        dataset_path: str = "data/similiar_qqp.json",
        seed: int = 42,
        tokenizer=None,
        max_total_tokens: int = 2048,
    ):
        super().__init__(split, n_samples, seed, tokenizer)
        self.dataset_path = dataset_path
        self.max_total_tokens = max_total_tokens

    def _load_raw_data(self) -> list[dict]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Could not find QQP dataset at {self.dataset_path}"
            )

        if self.dataset_path.endswith(".gz"):
            with gzip.open(self.dataset_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(self.dataset_path, encoding="utf-8") as f:
                return json.load(f)

    def _get_clean_pairs(self) -> list[dict]:
        """
        Deterministically builds a master list of 30k unique pairs.
        Prevents any sentence leakage or duplication.
        """
        raw_data = self._load_raw_data()
        logging.info(f"Loaded {len(raw_data)} pairs of QQP")

        # shuffle deterministically to ensure reproducible splits
        rng = random.Random(self.seed)
        rng.shuffle(raw_data)

        clean_pairs = []
        seen_texts = set()

        for item in raw_data:
            origin = item["origin"].strip()
            similar = item["similar"].strip()

            # skip malformed pairs
            if not origin or not similar:
                continue

            # skip pairs that are identical
            if origin == similar:
                continue

            # skip if sentence has been seen before
            if origin in seen_texts or similar in seen_texts:
                continue

            seen_texts.add(origin)
            seen_texts.add(similar)
            clean_pairs.append({"origin": origin, "similar": similar})

            if len(clean_pairs) >= self.MAX_PAIRS:
                break

        return clean_pairs

    def load(self) -> list[DatasetSample]:
        clean_pairs = self._get_clean_pairs()

        # determine starting offset based on the split
        if self.split == DatasetSplit.TRAIN:
            start_offset = 0
            text_key = "origin"

        elif self.split == DatasetSplit.VALIDATION:
            start_offset = self.VAL_OFFSET
            text_key = "similar"

        elif self.split == DatasetSplit.TEST:
            start_offset = self.TEST_OFFSET
            text_key = "similar"
        else:
            raise ValueError(f"Unsupported split: {self.split}")
        logging.info(f"Start offset: {start_offset}")
        samples = []
        # iterate dynamically from the offset so we can skip invalid lengths
        for idx in range(start_offset, len(clean_pairs)):
            if len(samples) >= self.n_samples:
                break

            pair = clean_pairs[idx]
            text = pair[text_key]

            # formatting as a chat sequence, including the system prompt
            formatted_conv = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
            prompt_len = None

            if self.tokenizer is not None:
                try:
                    prompt_str = self.tokenizer.apply_chat_template(
                        formatted_conv, tokenize=False, add_generation_prompt=True
                    )
                    prompt_len = len(self.tokenizer.encode(prompt_str))

                    # check if prompt length is below 1024
                    if not is_valid_sequence(
                        prompt_len,
                        output_len=0,
                        max_total_len=self.max_total_tokens,
                        skip_min_output_len_check=True,
                    ):
                        continue
                except ValueError:
                    continue

            samples.append(
                DatasetSample(
                    id=f"qqp-{self.split.value}-{idx}",
                    prompt=formatted_conv,
                    label=None,
                    prompt_length=prompt_len,
                    tokenizer_name=self.tokenizer.name_or_path
                    if self.tokenizer
                    else None,
                    metadata={
                        "source": "qqp",
                        "pair_index": idx,  # keeps the original mapping transparent
                        "text_type": text_key,
                    },
                )
            )

        # warn if validation dropped too many samples
        if len(samples) < self.n_samples:
            logging.warning(
                f"Dataset shortage for {self.split.value}: "
                f"Requested {self.n_samples} samples, "
                f"but only found {len(samples)} valid ones "
                f"after prompt length validation (<1024). "
            )

        return samples


class BatchedDataset:
    def __init__(self, samples: list[DatasetSample], tokenizer=None):
        self.samples = samples
        self.tokenizer = tokenizer
        self._lengths_computed = False

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[DatasetSample]:
        return iter(self.samples)

    def _compute_lengths(self):
        """Computes and caches the token length for all samples."""
        if self._lengths_computed or not self.tokenizer:
            return
        for sample in self.samples:
            if sample.prompt_length is None:
                if isinstance(sample.prompt, list):
                    prompt_str = self.tokenizer.apply_chat_template(
                        sample.prompt, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt_str = sample.prompt

                # tokenize to get exact length
                sample.prompt_length = len(self.tokenizer.encode(prompt_str))
        self._lengths_computed = True

    def get_batches(
        self, batch_size: int, strategy: str = "sorted_length", seed: int = 42
    ) -> list[list[DatasetSample]]:
        """
        Yields batches of DatasetSamples based on the requested strategy.
        Seed is used for reproducible shuffling when strategy is "random".

        Strategies:
        - "sequential": Original order.
        - "random": Shuffled order.
        - "sorted_length": Sorts all by length, then chunks.
        """
        if strategy == "sorted_length":
            if any(s.prompt_length is None for s in self.samples):
                assert self.tokenizer is not None, (
                    f"Strategy '{strategy}' requires a tokenizer "
                    f"to compute missing prompt lengths."
                )
            self._compute_lengths()

        if strategy == "sequential":
            working_list = self.samples[:]
        elif strategy == "random":
            working_list = self.samples[:]
            rng = random.Random(seed)
            rng.shuffle(working_list)
        elif strategy == "sorted_length":
            # sort ascending by length
            working_list = sorted(self.samples, key=lambda sample: sample.prompt_length)
        else:
            raise ValueError(f"Unknown batching strategy: {strategy}")

        # chunk into batches
        return [
            working_list[i : i + batch_size]
            for i in range(0, len(working_list), batch_size)
        ]


class DatasetFactory:
    @staticmethod
    def get_dataset(
        name: DatasetName, split: DatasetSplit, n_samples: int, **kwargs
    ) -> BatchedDataset:
        # pass tokenizer through if it exists in kwargs
        tokenizer = kwargs.get("tokenizer", None)

        if name == DatasetName.NEWTON:
            samples = NewtonDataset(split, n_samples).load()
        elif name == DatasetName.GSM8K:
            samples = GSM8KDataset(split, n_samples).load()
        elif name == DatasetName.SHAREGPT:
            repo = kwargs.get(
                "dataset_path", "anon8231489123/ShareGPT_Vicuna_unfiltered"
            )
            max_total_tokens = kwargs.get("max_total_tokens", DEFAULT_MAX_TOKENS)
            samples = ShareGPTDataset(
                split=split,
                n_samples=n_samples,
                dataset_path=repo,
                tokenizer=tokenizer,
                max_total_tokens=max_total_tokens,
            ).load()
        elif name == DatasetName.BOOLQ:
            samples = BoolQDataset(split, n_samples).load()
        elif name == DatasetName.MMLU:
            subset = kwargs.get("subset", "all")
            samples = MMLUDataset(split, n_samples, subset=subset).load()
        elif name == DatasetName.QQP:
            dataset_path = kwargs.get("dataset_path", "data/similiar_qqp.json")
            max_total_tokens = kwargs.get("max_total_tokens", DEFAULT_MAX_TOKENS)
            samples = QQPDataset(
                split=split,
                n_samples=n_samples,
                dataset_path=dataset_path,
                tokenizer=tokenizer,
                max_total_tokens=max_total_tokens,
            ).load()
        else:
            raise ValueError(f"Unknown dataset: {name}")

        logging.info(f"Loaded {len(samples)} samples for {name.value} {split.value}")

        return BatchedDataset(samples, tokenizer=tokenizer)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    logging.basicConfig(level=logging.INFO)

    # initialise the tokenizer using the same model we use for inference
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    logging.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load 100 samples to get a good distribution
    logging.info("Loading ShareGPT samples...")
    dataset = DatasetFactory.get_dataset(
        DatasetName.QQP, DatasetSplit.TEST, n_samples=20_000, tokenizer=tokenizer
    )
    logging.info(f"Loaded {len(dataset)} samples with prompt length statistics:")

    BATCH_SIZE = 256
    batches = dataset.get_batches(batch_size=BATCH_SIZE, strategy="sorted_length")

    for i, batch in enumerate(batches):
        lengths = [s.prompt_length for s in batch]
        min_len, max_len = min(lengths), max(lengths)
        logging.info(f"Batch {i + 1:02d}: {len(batch)} items")
        logging.info(f"  Length Range: {min_len:4d} -> {max_len:4d} tokens")
