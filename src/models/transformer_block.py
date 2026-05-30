from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.models.attention import causal_mask


@dataclass(frozen=True)
class MultiHeadAttentionOutput:
    values: Tensor
    weights: Tensor


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention for one Transformer block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if num_heads < 1:
            raise ValueError("num_heads must be positive.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, return_weights: bool = False) -> Tensor | MultiHeadAttentionOutput:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, time, hidden_dim).")

        batch_size, time_steps, hidden_dim = x.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"expected hidden_dim={self.hidden_dim}, got {hidden_dim}.")

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = q @ k.transpose(-2, -1) / self.head_dim**0.5
        mask = causal_mask(time_steps, device=x.device)
        scores = scores.masked_fill(
            ~mask.view(1, 1, time_steps, time_steps),
            torch.finfo(x.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        values = weights @ v
        values = self._merge_heads(values, batch_size, time_steps)
        values = self.resid_dropout(self.out_proj(values))

        if return_weights:
            return MultiHeadAttentionOutput(values=values, weights=weights)
        return values

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, time_steps, _ = x.shape
        return (
            x.view(batch_size, time_steps, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _merge_heads(self, x: Tensor, batch_size: int, time_steps: int) -> Tensor:
        return x.transpose(1, 2).contiguous().view(batch_size, time_steps, self.hidden_dim)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, expansion_factor: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = hidden_dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """A pre-norm Transformer block with residual attention and FFN paths."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        expansion_factor: int = 4,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, expansion_factor, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x
