from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by LLaMA-style decoder blocks."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if x.size(-1) != self.weight.numel():
            raise ValueError("last dimension must match hidden_dim.")
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        if hidden_dim < 1 or intermediate_dim < 1:
            raise ValueError("hidden_dim and intermediate_dim must be positive.")
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def rotary_frequencies(head_dim: int, seq_len: int, base: float = 10_000.0) -> Tensor:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE.")
    positions = torch.arange(seq_len, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    return positions[:, None] * inv_freq[None, :]


def rotary_cache(head_dim: int, seq_len: int, base: float = 10_000.0) -> tuple[Tensor, Tensor]:
    freqs = rotary_frequencies(head_dim=head_dim, seq_len=seq_len, base=base)
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply RoPE to tensors shaped (..., seq_len, head_dim)."""

    if x.size(-1) % 2 != 0:
        raise ValueError("last dimension must be even.")
    if freqs.shape != (x.size(-2), x.size(-1) // 2):
        raise ValueError("freqs must have shape (seq_len, head_dim / 2).")
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = freqs.cos().to(device=x.device, dtype=x.dtype)
    sin = freqs.sin().to(device=x.device, dtype=x.dtype)
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def apply_rope_with_cache(
    q: Tensor,
    k: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
    position_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply cached RoPE to q/k shaped (B, heads, T, D)."""

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape (batch, heads, time, head_dim).")
    if q.size(0) != k.size(0) or q.size(2) != k.size(2) or q.size(3) != k.size(3):
        raise ValueError("q and k must share batch, time, and head_dim.")
    if q.size(-1) % 2 != 0:
        raise ValueError("RoPE requires even head_dim.")
    if cos_cache.shape != sin_cache.shape:
        raise ValueError("cos_cache and sin_cache must have identical shapes.")
    if cos_cache.shape[1] != q.size(-1) // 2:
        raise ValueError("RoPE cache must have shape (max_seq_len, head_dim / 2).")

    if position_ids.ndim == 1:
        if position_ids.numel() != q.size(2):
            raise ValueError("1D position_ids must have length time.")
        position_ids = position_ids.unsqueeze(0).expand(q.size(0), -1)
    if position_ids.shape != (q.size(0), q.size(2)):
        raise ValueError("position_ids must have shape (batch, time) or (time,).")

    cos = cos_cache.to(device=q.device, dtype=q.dtype)[position_ids][:, None, :, :]
    sin = sin_cache.to(device=q.device, dtype=q.dtype)[position_ids][:, None, :, :]
    return _apply_rope_cached(q, cos, sin), _apply_rope_cached(k, cos, sin)


def _apply_rope_cached(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def repeat_kv(
    x: Tensor,
    repeats: int | None = None,
    num_query_heads: int | None = None,
    num_kv_heads: int | None = None,
) -> Tensor:
    if repeats is None:
        if num_query_heads is None or num_kv_heads is None:
            raise ValueError("provide repeats or num_query_heads and num_kv_heads.")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads.")
        repeats = num_query_heads // num_kv_heads
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    if repeats == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(-1, -1, repeats, -1, -1)
        .reshape(x.size(0), x.size(1) * repeats, x.size(2), x.size(3))
    )


@dataclass(frozen=True)
class KVCache:
    keys: Tensor
    values: Tensor


class GroupedQueryAttention(nn.Module):
    """Tiny GQA module for teaching query heads vs key/value heads."""

    def __init__(
        self,
        hidden_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        if hidden_dim % num_query_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_query_heads.")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads.")
        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_query_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor, cache: KVCache | None = None) -> tuple[Tensor, KVCache]:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, time, hidden_dim).")
        batch, time_steps, _hidden = x.shape
        q = self._shape(self.q_proj(x), self.num_query_heads)
        k = self._shape(self.k_proj(x), self.num_kv_heads)
        v = self._shape(self.v_proj(x), self.num_kv_heads)
        if cache is not None:
            k = torch.cat([cache.keys, k], dim=2)
            v = torch.cat([cache.values, v], dim=2)
        new_cache = KVCache(keys=k, values=v)

        repeats = self.num_query_heads // self.num_kv_heads
        k_for_attention = repeat_kv(k, repeats)
        v_for_attention = repeat_kv(v, repeats)
        scores = q @ k_for_attention.transpose(-2, -1) / math.sqrt(self.head_dim)
        key_steps = k_for_attention.size(-2)
        query_positions = torch.arange(key_steps - time_steps, key_steps, device=x.device)
        key_positions = torch.arange(key_steps, device=x.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(
            ~causal.view(1, 1, time_steps, key_steps),
            torch.finfo(x.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        values = weights @ v_for_attention
        merged = values.transpose(1, 2).contiguous().view(batch, time_steps, self.hidden_dim)
        return self.out_proj(merged), new_cache

    def _shape(self, x: Tensor, heads: int) -> Tensor:
        batch, time_steps, _ = x.shape
        return x.view(batch, time_steps, heads, self.head_dim).transpose(1, 2).contiguous()
