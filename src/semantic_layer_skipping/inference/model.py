from abc import ABC, abstractmethod

import torch


class Model(ABC):
    """Encapsulates differences between TransformerLens and native PyTorch models."""

    def __init__(self, inner_model, tokenizer):
        self.inner = inner_model
        self.tokenizer = tokenizer

    @property
    @abstractmethod
    def vector_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def n_layers(self) -> int:
        pass

    @abstractmethod
    def to_string(self, token_ids) -> str:
        pass

    @abstractmethod
    def to_tokens(self, text: str | list[str]) -> torch.Tensor:
        pass

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id


class LensModel(Model):
    @property
    def vector_dim(self) -> int:
        return self.inner.cfg.d_model

    @property
    def n_layers(self) -> int:
        return self.inner.cfg.n_layers

    def to_string(self, token_ids) -> str:
        return self.inner.to_string(token_ids)

    def to_tokens(self, text: str | list[str]) -> torch.Tensor:
        return self.inner.to_tokens(text)


class TorchModel(Model):
    @property
    def vector_dim(self) -> int:
        return self.inner.config.hidden_size

    @property
    def n_layers(self) -> int:
        return self.inner.config.num_hidden_layers

    def to_string(self, token_ids) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def to_tokens(self, text: str | list[str]) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt", padding=True)
