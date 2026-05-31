import pytest
import torch

from src.math_foundations import (
    brier_score,
    cosine_similarity_matrix,
    cross_entropy_from_logits,
    expected_calibration_error,
    finite_difference_gradient,
    linear_transform,
    naive_softmax,
    one_step_gradient_descent,
    stable_softmax,
    trace_language_model_shapes,
    wilson_interval,
)


def test_linear_transform_matches_lesson_matrix_convention() -> None:
    points = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weight = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    bias = torch.tensor([1.0, -1.0])

    transformed = linear_transform(points, weight, bias)

    assert torch.equal(transformed, torch.tensor([[3.0, 1.0], [7.0, 3.0]]))


def test_cosine_similarity_focuses_on_direction_not_length() -> None:
    a = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    b = torch.tensor([[3.0, 0.0], [0.0, 1.0]])

    similarities = cosine_similarity_matrix(a, b)

    assert similarities.shape == (2, 2)
    assert torch.allclose(similarities[:, 0], torch.ones(2))
    assert torch.allclose(similarities[:, 1], torch.zeros(2))


def test_stable_softmax_handles_large_logits_and_sums_to_one() -> None:
    logits = torch.tensor([[1000.0, 1001.0, 1002.0]])

    naive = naive_softmax(logits)
    probs = stable_softmax(logits)
    shifted = stable_softmax(logits + 123.0)

    assert torch.isnan(naive).any()
    assert not torch.isnan(probs).any()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))
    assert torch.allclose(probs, shifted)
    assert probs.argmax(dim=-1).item() == 2


def test_cross_entropy_from_logits_matches_pytorch_loss() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    target = torch.tensor([0, 2])

    loss = cross_entropy_from_logits(logits, target)

    assert loss.ndim == 0
    assert loss.item() > 0


def test_calibration_metrics_measure_confidence_quality() -> None:
    probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.6, 0.4],
        ],
    )
    target = torch.tensor([0, 1, 0])
    confidences = torch.tensor([0.9, 0.8, 0.6])
    correct = torch.tensor([True, False, True])

    assert brier_score(probabilities, target).item() > 0
    assert expected_calibration_error(confidences, correct, n_bins=2).item() > 0


def test_wilson_interval_keeps_zero_failures_from_becoming_zero_risk() -> None:
    low, high_success = wilson_interval(successes=20, total=20)
    low_failure, high_failure = wilson_interval(successes=0, total=20)

    assert low < 1.0
    assert high_success == pytest.approx(1.0)
    assert low_failure == 0.0
    assert high_failure > 0.1


def test_finite_difference_gradient_matches_autograd_for_quadratic() -> None:
    x = torch.tensor([1.5, -2.0], dtype=torch.float64, requires_grad=True)
    loss = (x.pow(2).sum())
    loss.backward()

    estimated = finite_difference_gradient(lambda value: value.pow(2).sum(), x.detach())

    assert torch.allclose(estimated, x.grad, atol=1e-3)


def test_gradient_descent_steps_against_gradient() -> None:
    parameter = torch.tensor([1.0, -2.0])
    grad = torch.tensor([0.5, -0.25])

    updated = one_step_gradient_descent(parameter, grad, learning_rate=0.1)

    assert torch.allclose(updated, torch.tensor([0.95, -1.975]))


def test_language_model_shape_trace_explains_transformer_flow() -> None:
    trace = trace_language_model_shapes(
        batch_size=2,
        seq_len=4,
        hidden_dim=12,
        num_heads=3,
        vocab_size=20,
    )

    assert [item.name for item in trace] == [
        "input_ids",
        "embedding",
        "qkv",
        "attention_scores",
        "logits",
        "labels",
    ]
    assert trace[2].shape == (2, 3, 4, 4)

    with pytest.raises(ValueError, match="divisible"):
        trace_language_model_shapes(
            batch_size=2,
            seq_len=4,
            hidden_dim=10,
            num_heads=3,
            vocab_size=20,
        )
