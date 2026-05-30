from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AttentionOutput:
    values: Tensor
    weights: Tensor
    scores: Tensor


def causal_mask(time_steps: int, device: torch.device | str | None = None) -> Tensor:
    if time_steps < 1:
        raise ValueError("time_steps must be positive.")
    return torch.tril(torch.ones(time_steps, time_steps, dtype=torch.bool, device=device))


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = True,
    attention_mask: Tensor | None = None,
) -> AttentionOutput:
    """Computes scaled dot-product attention for tensors shaped (B, T, H)."""

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must all have shape (batch, time, hidden).")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes in this lesson.")

    head_dim = q.size(-1)
    scores = q @ k.transpose(-2, -1) / head_dim**0.5
    allowed: Tensor | None = None
    if causal:
        time_steps = q.size(-2)
        allowed = causal_mask(time_steps, device=q.device).unsqueeze(0)

    if attention_mask is not None:
        if attention_mask.shape != q.shape[:2]:
            raise ValueError("attention_mask must have shape (batch, time).")
        key_mask = attention_mask.to(dtype=torch.bool, device=q.device)[:, None, :]
        allowed = key_mask if allowed is None else allowed & key_mask

    if allowed is not None:
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1)
    if allowed is not None:
        weights = weights.masked_fill(~allowed, 0.0)
        normalizer = weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps,
        )
        weights = weights / normalizer
    values = weights @ v
    return AttentionOutput(values=values, weights=weights, scores=scores)


class SingleHeadSelfAttention(nn.Module):
    """A tiny self-attention module before introducing multi-head attention."""

    def __init__(self, input_dim: int, head_dim: int, causal: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.causal = causal
        self.q_proj = nn.Linear(input_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, head_dim, bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> AttentionOutput:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, time, input_dim).")
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return scaled_dot_product_attention(
            q,
            k,
            v,
            causal=self.causal,
            attention_mask=attention_mask,
        )


def future_attention_mass(weights: Tensor) -> Tensor:
    if weights.ndim != 3:
        raise ValueError("weights must have shape (batch, time, time).")
    time_steps = weights.size(-1)
    future_mask = ~causal_mask(time_steps, device=weights.device)
    return weights.masked_select(future_mask.unsqueeze(0)).sum()
