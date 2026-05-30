import torch

from src.models.attention import (
    SingleHeadSelfAttention,
    causal_mask,
    future_attention_mass,
    scaled_dot_product_attention,
)


def test_causal_mask_is_lower_triangular() -> None:
    mask = causal_mask(4)

    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ],
    )
    assert torch.equal(mask, expected)


def test_attention_output_shapes_are_correct() -> None:
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)

    output = scaled_dot_product_attention(q, k, v)

    assert output.values.shape == (2, 4, 8)
    assert output.weights.shape == (2, 4, 4)
    assert output.scores.shape == (2, 4, 4)


def test_attention_weights_sum_to_one_on_last_dimension() -> None:
    q = torch.randn(2, 5, 3)
    k = torch.randn(2, 5, 3)
    v = torch.randn(2, 5, 3)

    output = scaled_dot_product_attention(q, k, v)

    assert torch.allclose(output.weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_causal_attention_has_zero_future_weights() -> None:
    q = torch.randn(2, 5, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 4)

    output = scaled_dot_product_attention(q, k, v, causal=True)

    assert future_attention_mass(output.weights).item() == 0.0


def test_non_causal_attention_can_see_future_tokens() -> None:
    q = torch.ones(1, 4, 3)
    k = torch.ones(1, 4, 3)
    v = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    output = scaled_dot_product_attention(q, k, v, causal=False)

    assert future_attention_mass(output.weights).item() > 0.0


def test_attention_float32_has_no_nan() -> None:
    q = torch.randn(2, 6, 16, dtype=torch.float32)
    k = torch.randn(2, 6, 16, dtype=torch.float32)
    v = torch.randn(2, 6, 16, dtype=torch.float32)

    output = scaled_dot_product_attention(q, k, v)

    assert not torch.isnan(output.values).any()
    assert not torch.isnan(output.weights).any()


def test_future_token_intervention_does_not_change_past_output() -> None:
    q = torch.randn(1, 5, 4)
    k = torch.randn(1, 5, 4)
    v = torch.randn(1, 5, 4)
    changed_q = q.clone()
    changed_k = k.clone()
    changed_v = v.clone()
    changed_q[:, 3:] = torch.randn_like(changed_q[:, 3:]) * 100.0
    changed_k[:, 3:] = torch.randn_like(changed_k[:, 3:]) * 100.0
    changed_v[:, 3:] = torch.randn_like(changed_v[:, 3:]) * 100.0

    output = scaled_dot_product_attention(q, k, v, causal=True)
    changed_output = scaled_dot_product_attention(changed_q, changed_k, changed_v, causal=True)

    assert torch.allclose(output.values[:, :3], changed_output.values[:, :3], atol=1e-6)


def test_wrong_softmax_dimension_fails_row_normalization() -> None:
    q = torch.randn(1, 4, 3)
    k = torch.randn(1, 4, 3)
    scores = q @ k.transpose(-2, -1) / 3**0.5
    wrong_weights = torch.softmax(scores, dim=-2)

    assert not torch.allclose(wrong_weights.sum(dim=-1), torch.ones(1, 4), atol=1e-6)


def test_zero_mask_instead_of_negative_infinity_leaks_future_attention() -> None:
    q = torch.ones(1, 4, 3)
    k = torch.ones(1, 4, 3)
    scores = q @ k.transpose(-2, -1) / 3**0.5
    mask = causal_mask(4)
    wrong_scores = scores.masked_fill(~mask, 0.0)
    wrong_weights = torch.softmax(wrong_scores, dim=-1)

    assert future_attention_mass(wrong_weights).item() > 0.0


def test_padding_mask_removes_key_positions() -> None:
    q = torch.randn(1, 4, 3)
    k = torch.randn(1, 4, 3)
    v = torch.randn(1, 4, 3)
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    output = scaled_dot_product_attention(q, k, v, causal=False, attention_mask=attention_mask)

    assert torch.allclose(output.weights[:, :, 2:], torch.zeros(1, 4, 2))


def test_single_head_self_attention_projects_inputs() -> None:
    module = SingleHeadSelfAttention(input_dim=6, head_dim=4)
    x = torch.randn(2, 5, 6)

    output = module(x)

    assert output.values.shape == (2, 5, 4)
    assert output.weights.shape == (2, 5, 5)
