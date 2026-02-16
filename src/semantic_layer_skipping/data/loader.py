import random
from abc import ABC, abstractmethod

from datasets import load_dataset
from structures import DatasetName, DatasetSample, DatasetSplit
from utils import (
    ISAAC_NEWTON_QUESTIONS_CALIBRATION,
    ISAAC_NEWTON_QUESTIONS_TEST,
    ISAAC_NEWTON_QUESTIONS_TRAIN,
)


class BaseDataset(ABC):
    def __init__(self, split: DatasetSplit, n_samples: int):
        self.split = split
        self.n_samples = n_samples

    @abstractmethod
    def load(self) -> list[DatasetSample]:
        pass


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
        "Think step by step. "
        "Please put the final answer within \\boxed{}."
    )

    # train/validation split ratio
    TRAIN_RATIO = 0.7
    SEED = 42

    def load(self) -> list[DatasetSample]:
        if self.split == DatasetSplit.TEST:
            ds = load_dataset("gsm8k", "main", split="test")
        else:
            # train and validation will be split from original train split
            ds = load_dataset("gsm8k", "main", split="train")

        # shuffle
        all_indices = list(range(len(ds)))
        rng = random.Random(self.SEED)
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
        DatasetSplit.TRAIN: (0.0, 0.5),
        DatasetSplit.VALIDATION: (0.5, 0.9),
        DatasetSplit.TEST: (0.9, 1.0),
    }

    def __init__(
        self,
        split: DatasetSplit,
        n_samples: int,
        dataset_path: str = "anon8231489123/ShareGPT_Vicuna_unfiltered",
        seed: int = 42,
    ):
        super().__init__(split, n_samples)
        self.dataset_path = dataset_path
        self.seed = seed

    def load(self) -> list[DatasetSample]:
        # we load full dataset
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
        selected_data = split_data[: self.n_samples]

        # process
        samples = []
        for idx, item in enumerate(selected_data):
            # load conversations
            convs = item.get("conversations", item.get("conversation", []))
            if len(convs) < 2:
                continue

            # extract history and label
            formatted_conv = []
            for msg in convs[:-1]:
                role = "user" if msg["from"] == "human" else "assistant"
                formatted_conv.append({"role": role, "content": msg["value"]})

            samples.append(
                DatasetSample(
                    id=f"sharegpt-{self.split.value}-{idx}",
                    prompt=formatted_conv,
                    # we can optionally set the label as the last assistant message,
                    # but for now we leave it None since it's more of a conversation
                    label=None,
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
        "First, think step by step and explain your reasoning. "
        "Then, output the final answer within \\boxed{}. "
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


class MMLUDataset(BaseDataset):
    """
    Multiple Choice Questions.
    System Prompt enforces Chain of Thought + Boxed Answer (Letter).
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant. "
        "Answer the multiple choice question by selecting the best option "
        "(A, B, C, or D). "
        "First, think step by step and explain your reasoning. "
        "Then, output the final answer letter within \\boxed{}. "
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


class DatasetFactory:
    @staticmethod
    def get_dataset(
        name: DatasetName, split: DatasetSplit, n_samples: int, **kwargs
    ) -> list[DatasetSample]:
        if name == DatasetName.NEWTON:
            return NewtonDataset(split, n_samples).load()
        elif name == DatasetName.GSM8K:
            return GSM8KDataset(split, n_samples).load()
        elif name == DatasetName.SHAREGPT:
            repo = kwargs.get(
                "dataset_path", "anon8231489123/ShareGPT_Vicuna_unfiltered"
            )
            return ShareGPTDataset(
                split=split, n_samples=n_samples, dataset_path=repo
            ).load()
        elif name == DatasetName.BOOLQ:
            return BoolQDataset(split, n_samples).load()
        elif name == DatasetName.MMLU:
            # subsets: e.g., "high_school_computer_science"
            subset = kwargs.get("subset", "all")
            return MMLUDataset(split, n_samples, subset=subset).load()
        else:
            raise ValueError(f"Unknown dataset: {name}")


if __name__ == "__main__":

    def run_demo(name, split, **kwargs):
        print(f"--- {name} ({split.value}) ---")
        data = DatasetFactory.get_dataset(name, split, n_samples=2, **kwargs)
        for sample in data:
            print(f"ID: {sample.id}")
            if isinstance(sample.prompt, list):
                last_msg = sample.prompt[-1]
                content = last_msg["content"]
                role = last_msg["role"]
                print(f"Last Input ({role}): {content[:100]}...")
            else:
                print(f"Last Input: {str(sample.prompt)[:100]}...")
            print(f"Label: {sample.label}")
            print("-" * 20, "\n")

    run_demo(DatasetName.NEWTON, DatasetSplit.TRAIN)
    run_demo(DatasetName.GSM8K, DatasetSplit.TRAIN)
    run_demo(DatasetName.BOOLQ, DatasetSplit.VALIDATION)
    run_demo(DatasetName.MMLU, DatasetSplit.TEST, subset="high_school_computer_science")
    run_demo(DatasetName.SHAREGPT, DatasetSplit.TRAIN)
