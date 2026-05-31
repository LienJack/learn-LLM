from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """A minimal causal LM where each token predicts the next token directly."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        logits = self.token_logits(input_ids)
        loss = language_modeling_loss(logits, labels) if labels is not None else None
        return logits, loss


class CountBigramLM:
    """A count-based bigram baseline with add-k smoothing."""

    def __init__(self, vocab_size: int, smoothing: float = 1.0) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive.")
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative.")
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.counts = torch.full((vocab_size, vocab_size), smoothing, dtype=torch.float32)

    def fit(self, token_ids: Tensor) -> CountBigramLM:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a 1D tensor.")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use dtype torch.long.")
        if token_ids.numel() < 2:
            raise ValueError("token_ids must contain at least two tokens.")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.vocab_size:
            raise ValueError("token_ids contain ids outside the vocabulary.")

        self.counts = torch.full_like(self.counts, self.smoothing)
        for current_id, next_id in zip(token_ids[:-1], token_ids[1:], strict=True):
            self.counts[current_id.item(), next_id.item()] += 1.0
        return self

    def next_probs(self, prev_id: int) -> Tensor:
        if prev_id < 0 or prev_id >= self.vocab_size:
            raise ValueError("prev_id must be inside the vocabulary.")
        row = self.counts[prev_id]
        total = row.sum()
        if total.item() == 0:
            return torch.full((self.vocab_size,), 1.0 / self.vocab_size)
        return row / total

    def nll(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a 1D tensor.")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use dtype torch.long.")
        if token_ids.numel() < 2:
            raise ValueError("token_ids must contain at least two tokens.")

        losses = []
        for current_id, next_id in zip(token_ids[:-1], token_ids[1:], strict=True):
            probability = self.next_probs(current_id.item())[next_id.item()].clamp_min(1e-12)
            losses.append(-probability.log())
        return torch.stack(losses).mean()

    def generate(
        self,
        start_id: int,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        generated = [start_id]
        current_id = start_id
        for _ in range(max_new_tokens):
            next_id = torch.multinomial(
                self.next_probs(current_id),
                num_samples=1,
                generator=generator,
            ).item()
            generated.append(next_id)
            current_id = next_id
            if eos_token_id is not None and next_id == eos_token_id:
                break
        return torch.tensor(generated, dtype=torch.long)


@dataclass(frozen=True)
class LanguageModelingConfig:
    seed: int = 0
    block_size: int = 8
    batch_size: int = 16
    lr: float = 0.05
    steps: int = 100
    device: str = "cpu"


@dataclass
class LMHistoryItem:
    step: int
    loss: float


@dataclass
class LMTrainingHistory:
    items: list[LMHistoryItem] = field(default_factory=list)

    @property
    def losses(self) -> list[float]:
        return [item.loss for item in self.items]


def set_language_modeling_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_lm_batch(
    token_ids: Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a 1D tensor.")
    if token_ids.dtype != torch.long:
        raise TypeError("token_ids must use dtype torch.long.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    if len(token_ids) <= block_size:
        raise ValueError("token_ids must contain at least block_size + 1 tokens.")

    max_start = len(token_ids) - block_size - 1
    starts = torch.randint(
        low=0,
        high=max_start + 1,
        size=(batch_size,),
        generator=generator,
    )
    inputs = torch.stack([token_ids[start : start + block_size] for start in starts])
    labels = torch.stack([token_ids[start + 1 : start + block_size + 1] for start in starts])
    return inputs, labels


def language_modeling_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, time, vocab_size).")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape (batch, time).")

    batch_size, time_steps, vocab_size = logits.shape
    return F.cross_entropy(
        logits.reshape(batch_size * time_steps, vocab_size),
        labels.reshape(batch_size * time_steps),
        ignore_index=-100,
    )


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    if logits.ndim != 1:
        raise ValueError("logits must be a 1D tensor for one position.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive when provided.")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1] when provided.")

    if temperature == 0:
        return logits.argmax(dim=-1)

    scaled_logits = logits / temperature
    if top_k is not None:
        k = min(top_k, scaled_logits.numel())
        values, _ = torch.topk(scaled_logits, k=k)
        threshold = values[-1]
        scaled_logits = scaled_logits.masked_fill(scaled_logits < threshold, float("-inf"))

    if top_p is not None and top_p < 1:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
        remove_sorted = cumulative_probabilities > top_p
        remove_sorted[1:] = remove_sorted[:-1].clone()
        remove_sorted[0] = False
        remove_indices = sorted_indices[remove_sorted]
        scaled_logits = scaled_logits.scatter(
            dim=0,
            index=remove_indices,
            src=torch.full_like(remove_indices, float("-inf"), dtype=scaled_logits.dtype),
        )

    probabilities = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(0)


@torch.no_grad()
def generate(
    model: BigramLanguageModel,
    prompt_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be a 1D tensor.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    model.eval()
    generated = prompt_ids.clone()
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        input_ids = generated[-1:].to(device).unsqueeze(0)
        logits, _ = model(input_ids)
        next_id = sample_next_token(
            logits[0, -1].cpu(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        generated = torch.cat([generated.cpu(), next_id.view(1)])
        if eos_token_id is not None and next_id.item() == eos_token_id:
            break

    return generated


def train_bigram_language_model(
    token_ids: Tensor,
    vocab_size: int,
    config: LanguageModelingConfig,
) -> tuple[BigramLanguageModel, LMTrainingHistory]:
    set_language_modeling_seed(config.seed)
    model = BigramLanguageModel(vocab_size=vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    generator = torch.Generator().manual_seed(config.seed + 1)
    history = LMTrainingHistory()

    token_ids = token_ids.to(config.device)
    for step in range(1, config.steps + 1):
        inputs, labels = make_lm_batch(token_ids, config.block_size, config.batch_size, generator)
        logits, loss = model(inputs, labels)
        assert loss is not None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.items.append(LMHistoryItem(step=step, loss=loss.item()))

    return model, history
