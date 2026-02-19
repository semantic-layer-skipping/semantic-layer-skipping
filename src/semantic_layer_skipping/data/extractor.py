import re

from structures import DatasetName


class AnswerExtractor:
    """
    Static methods to extract answers based on the dataset type.
    """

    @staticmethod
    def extract(dataset_name: DatasetName, model_output: str) -> str | None:
        output = model_output.strip()

        match dataset_name:
            case DatasetName.NEWTON | DatasetName.SHAREGPT:
                # these datasets have no strict answer
                return output
            case _:
                return AnswerExtractor._extract_boxed(output)

    @staticmethod
    def _extract_boxed(text: str) -> str | None:
        # find the last occurence of \boxed{...} and extract
        matches = re.findall(r"\\boxed\{(.*?)\}", text)
        if matches:
            return matches[-1].strip()
        return None
