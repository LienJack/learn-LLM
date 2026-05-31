from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.models.bigram_lm import (
    language_modeling_loss,
    sample_next_token,
    set_language_modeling_seed,
)
from src.models.transformer_block import TransformerBlock
from src.tokenizer.simple_tokenizer import CharacterTokenizer


@dataclass(frozen=True)
class MiniGPTConfig:
    vocab_size: int
    block_size: int
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive.")
        if self.block_size < 1:
            raise ValueError("block_size must be positive.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")


@dataclass(frozen=True)
class MiniGPTTrainingConfig:
    seed: int = 0
    batch_size: int = 16
    lr: float = 0.003
    steps: int = 100
    device: str = "cpu"


@dataclass
class MiniGPTHistoryItem:
    step: int
    loss: float


@dataclass
class MiniGPTTrainingHistory:
    items: list[MiniGPTHistoryItem] = field(default_factory=list)

    @property
    def losses(self) -> list[float]:
        return [item.loss for item in self.items]


class MiniGPT(nn.Module):
    """A compact decoder-only GPT assembled from earlier lesson components."""

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ],
        )
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, time).")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have shape (batch, time).")

        _batch_size, time_steps = input_ids.shape
        if time_steps > self.config.block_size:
            raise ValueError(
                f"Sequence length {time_steps} exceeds block_size {self.config.block_size}.",
            )

        positions = torch.arange(time_steps, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask, causal=True)
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)
        loss = language_modeling_loss(logits, labels) if labels is not None else None
        return logits, loss


@torch.no_grad()
def generate(
    model: MiniGPT,
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
    device = next(model.parameters()).device
    generated = prompt_ids.to(device).clone()

    for _ in range(max_new_tokens):
        context = generated[-model.config.block_size :].unsqueeze(0)
        logits, _ = model(context)
        next_id = sample_next_token(
            logits[0, -1].detach().cpu(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        ).to(device)
        generated = torch.cat([generated, next_id.view(1)])
        if eos_token_id is not None and next_id.item() == eos_token_id:
            break

    return generated.cpu()


def make_random_lm_batch(
    token_ids: Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    if len(token_ids) <= block_size:
        raise ValueError("token_ids must contain at least block_size + 1 tokens.")
    max_start = len(token_ids) - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
    inputs = torch.stack([token_ids[start : start + block_size] for start in starts])
    labels = torch.stack([token_ids[start + 1 : start + block_size + 1] for start in starts])
    return inputs, labels


def train_mini_gpt(
    token_ids: Tensor,
    model_config: MiniGPTConfig,
    train_config: MiniGPTTrainingConfig,
) -> tuple[MiniGPT, MiniGPTTrainingHistory]:
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a 1D tensor.")
    if token_ids.dtype != torch.long:
        raise TypeError("token_ids must use dtype torch.long.")

    set_language_modeling_seed(train_config.seed)
    model = MiniGPT(model_config).to(train_config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    generator = torch.Generator().manual_seed(train_config.seed + 1)
    history = MiniGPTTrainingHistory()
    token_ids = token_ids.to(train_config.device)

    for step in range(1, train_config.steps + 1):
        inputs, labels = make_random_lm_batch(
            token_ids,
            model_config.block_size,
            train_config.batch_size,
            generator,
        )
        inputs = inputs.to(train_config.device)
        labels = labels.to(train_config.device)
        logits, loss = model(inputs, labels)
        assert loss is not None
        _ = logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.items.append(MiniGPTHistoryItem(step=step, loss=loss.item()))

    return model, history


def save_checkpoint(
    path: str | Path,
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "token_to_id": tokenizer.token_to_id,
        "extra": extra or {},
    }
    torch.save(checkpoint, Path(path))


def load_checkpoint(path: str | Path) -> tuple[MiniGPT, CharacterTokenizer, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    config = MiniGPTConfig(**checkpoint["config"])
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer = CharacterTokenizer(checkpoint["token_to_id"])
    return model, tokenizer, checkpoint.get("extra", {})
