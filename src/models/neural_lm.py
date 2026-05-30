from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from src.models.bigram_lm import language_modeling_loss, set_language_modeling_seed


def causal_mean(hidden: Tensor, attention_mask: Tensor | None = None) -> Tensor:
    """A fixed causal context mixer for tensors shaped (batch, time, hidden)."""

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape (batch, time, hidden_dim).")
    batch_size, time_steps, _ = hidden.shape
    causal = torch.tril(
        torch.ones(time_steps, time_steps, dtype=hidden.dtype, device=hidden.device),
    )

    if attention_mask is not None:
        if attention_mask.shape != hidden.shape[:2]:
            raise ValueError("attention_mask must have shape (batch, time).")
        key_mask = attention_mask.to(dtype=hidden.dtype, device=hidden.device)[:, None, :]
        weights = causal[None, :, :] * key_mask
    else:
        weights = causal[None, :, :].expand(batch_size, -1, -1)

    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return weights @ hidden


def noncausal_mean(hidden: Tensor) -> Tensor:
    """Intentionally wrong full-sequence mean used by tests and failure experiments."""

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape (batch, time, hidden_dim).")
    return hidden.mean(dim=1, keepdim=True).expand_as(hidden)


class CurrentTokenLanguageModel(nn.Module):
    """A minimal embedding LM whose positions do not exchange information."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=padding_idx,
        )
        self.mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.token_embedding(input_ids)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        _ = attention_mask
        hidden = self.embed(input_ids)
        hidden = self.mixer(hidden)
        logits = self.lm_head(hidden)
        loss = language_modeling_loss(logits, labels) if labels is not None else None
        return logits, loss


class CausalMeanLanguageModel(CurrentTokenLanguageModel):
    """An embedding LM with a fixed causal mean context mixer."""

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        hidden = self.embed(input_ids)
        context = causal_mean(hidden, attention_mask)
        context = self.mixer(context)
        logits = self.lm_head(context)
        loss = language_modeling_loss(logits, labels) if labels is not None else None
        return logits, loss


NeuralLanguageModel = CausalMeanLanguageModel


@dataclass(frozen=True)
class NeuralLMConfig:
    seed: int = 0
    hidden_dim: int = 32
    batch_size: int = 16
    block_size: int = 8
    lr: float = 0.01
    steps: int = 100
    padding_idx: int | None = None
    device: str = "cpu"


@dataclass
class NeuralLMHistoryItem:
    step: int
    loss: float


@dataclass
class NeuralLMTrainingHistory:
    items: list[NeuralLMHistoryItem] = field(default_factory=list)

    @property
    def losses(self) -> list[float]:
        return [item.loss for item in self.items]


def embedding_rows_with_grad(model: NeuralLanguageModel) -> set[int]:
    grad = model.token_embedding.weight.grad
    if grad is None:
        return set()
    row_has_grad = grad.detach().abs().sum(dim=-1) > 0
    return set(row_has_grad.nonzero(as_tuple=False).flatten().tolist())


def cosine_similarity_matrix(embedding_weight: Tensor, token_ids: list[int]) -> Tensor:
    selected = embedding_weight.detach()[token_ids]
    normalized = nn.functional.normalize(selected, dim=-1)
    return normalized @ normalized.T


def train_neural_language_model(
    token_ids: Tensor,
    vocab_size: int,
    config: NeuralLMConfig,
) -> tuple[NeuralLanguageModel, NeuralLMTrainingHistory]:
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a 1D tensor.")
    if token_ids.dtype != torch.long:
        raise TypeError("token_ids must use dtype torch.long.")
    if len(token_ids) <= config.block_size:
        raise ValueError("token_ids must contain at least block_size + 1 tokens.")

    set_language_modeling_seed(config.seed)
    model = NeuralLanguageModel(
        vocab_size=vocab_size,
        hidden_dim=config.hidden_dim,
        padding_idx=config.padding_idx,
    ).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    generator = torch.Generator().manual_seed(config.seed + 1)
    history = NeuralLMTrainingHistory()
    token_ids = token_ids.to(config.device)

    for step in range(1, config.steps + 1):
        max_start = len(token_ids) - config.block_size - 1
        starts = torch.randint(
            low=0,
            high=max_start + 1,
            size=(config.batch_size,),
            generator=generator,
            device=config.device,
        )
        inputs = torch.stack(
            [token_ids[start : start + config.block_size] for start in starts],
        )
        labels = torch.stack(
            [token_ids[start + 1 : start + config.block_size + 1] for start in starts],
        )
        logits, loss = model(inputs, labels)
        assert loss is not None
        _ = logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.items.append(NeuralLMHistoryItem(step=step, loss=loss.item()))

    return model, history
