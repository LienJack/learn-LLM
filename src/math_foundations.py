from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ShapeTrace:
    name: str
    shape: tuple[int, ...]


def linear_transform(points: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Apply the lesson convention x @ W + b to a batch of vectors."""

    if points.ndim != 2 or weight.ndim != 2:
        raise ValueError("points and weight must both be rank-2 tensors.")
    if points.size(-1) != weight.size(0):
        raise ValueError("points last dimension must match weight first dimension.")
    output = points @ weight
    if bias is not None:
        if bias.shape != (weight.size(1),):
            raise ValueError("bias must have shape (output_dim,).")
        output = output + bias
    return output


def cosine_similarity_matrix(a: Tensor, b: Tensor) -> Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must both have shape (items, dim).")
    if a.size(-1) != b.size(-1):
        raise ValueError("a and b must share the same embedding dimension.")
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    return a_norm @ b_norm.T


def stable_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def naive_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    exp = logits.exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def cross_entropy_from_logits(logits: Tensor, target: Tensor) -> Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must include a class dimension.")
    return F.cross_entropy(logits, target)


def brier_score(probabilities: Tensor, target: Tensor) -> Tensor:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (items, classes).")
    if target.shape != (probabilities.size(0),):
        raise ValueError("target must have shape (items,).")
    one_hot = F.one_hot(target, num_classes=probabilities.size(1)).to(probabilities.dtype)
    return (probabilities - one_hot).pow(2).sum(dim=-1).mean()


def expected_calibration_error(
    confidences: Tensor,
    correct: Tensor,
    *,
    n_bins: int = 10,
) -> Tensor:
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same shape.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    if ((confidences < 0) | (confidences > 1)).any():
        raise ValueError("confidences must be in [0, 1].")

    correct_float = correct.to(dtype=confidences.dtype)
    ece = torch.zeros((), dtype=confidences.dtype, device=confidences.device)
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        if not in_bin.any():
            continue
        bin_confidence = confidences[in_bin].mean()
        bin_accuracy = correct_float[in_bin].mean()
        ece = ece + in_bin.float().mean() * (bin_confidence - bin_accuracy).abs()
    return ece


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive.")
    if successes < 0 or successes > total:
        raise ValueError("successes must be in [0, total].")
    if z <= 0:
        raise ValueError("z must be positive.")

    phat = successes / total
    denominator = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denominator
    margin = z * ((phat * (1 - phat) + z**2 / (4 * total)) / total) ** 0.5 / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def finite_difference_gradient(
    fn,
    x: Tensor,
    *,
    eps: float = 1e-4,
) -> Tensor:
    """Approximate d fn(x) / dx for scalar-output functions."""

    if not x.is_floating_point():
        raise TypeError("x must be a floating point tensor.")
    grad = torch.zeros_like(x)
    flat_x = x.detach().clone().reshape(-1)
    flat_grad = grad.reshape(-1)
    for index in range(flat_x.numel()):
        plus = flat_x.clone()
        minus = flat_x.clone()
        plus[index] += eps
        minus[index] -= eps
        plus_value = fn(plus.reshape_as(x))
        minus_value = fn(minus.reshape_as(x))
        flat_grad[index] = (plus_value - minus_value) / (2 * eps)
    return grad


def one_step_gradient_descent(
    parameter: Tensor,
    grad: Tensor,
    *,
    learning_rate: float,
) -> Tensor:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if parameter.shape != grad.shape:
        raise ValueError("parameter and grad must have the same shape.")
    return parameter - learning_rate * grad


def trace_language_model_shapes(
    *,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    vocab_size: int,
) -> list[ShapeTrace]:
    if hidden_dim % num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads.")
    head_dim = hidden_dim // num_heads
    return [
        ShapeTrace("input_ids", (batch_size, seq_len)),
        ShapeTrace("embedding", (batch_size, seq_len, hidden_dim)),
        ShapeTrace("qkv", (batch_size, num_heads, seq_len, head_dim)),
        ShapeTrace("attention_scores", (batch_size, num_heads, seq_len, seq_len)),
        ShapeTrace("logits", (batch_size, seq_len, vocab_size)),
        ShapeTrace("labels", (batch_size, seq_len)),
    ]
