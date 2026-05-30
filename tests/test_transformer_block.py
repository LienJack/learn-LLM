import pytest
import torch

from src.models.transformer_block import CausalSelfAttention, FeedForward, TransformerBlock


def test_hidden_dim_must_be_divisible_by_num_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(hidden_dim=10, num_heads=3)


def test_causal_self_attention_shape_is_stable_across_head_counts() -> None:
    x = torch.randn(2, 5, 12)

    for num_heads in [1, 2, 3, 4, 6]:
        attention = CausalSelfAttention(hidden_dim=12, num_heads=num_heads)
        output = attention(x)
        assert isinstance(output, torch.Tensor)
        assert output.shape == x.shape


def test_causal_mask_applies_to_all_heads() -> None:
    x = torch.randn(2, 5, 12)
    attention = CausalSelfAttention(hidden_dim=12, num_heads=3)

    output = attention(x, return_weights=True)
    weights = output.weights
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)

    assert weights.shape == (2, 3, 5, 5)
    assert torch.all(weights.masked_select(future.view(1, 1, 5, 5)) == 0)


def test_feed_forward_preserves_input_shape() -> None:
    ffn = FeedForward(hidden_dim=8, expansion_factor=4)
    x = torch.randn(2, 5, 8)

    output = ffn(x)

    assert output.shape == x.shape


def test_transformer_block_input_output_shape_match() -> None:
    block = TransformerBlock(hidden_dim=12, num_heads=3)
    x = torch.randn(2, 5, 12)

    output = block(x)

    assert output.shape == x.shape


def test_dropout_differs_between_train_and_eval() -> None:
    torch.manual_seed(0)
    block = TransformerBlock(hidden_dim=12, num_heads=3, dropout=0.5)
    x = torch.randn(2, 5, 12)

    block.train()
    train_a = block(x)
    train_b = block(x)

    block.eval()
    eval_a = block(x)
    eval_b = block(x)

    assert not torch.allclose(train_a, train_b)
    assert torch.allclose(eval_a, eval_b)


def test_stacked_blocks_backpropagate_without_nan() -> None:
    torch.manual_seed(0)
    stack = torch.nn.Sequential(
        TransformerBlock(hidden_dim=12, num_heads=3),
        TransformerBlock(hidden_dim=12, num_heads=3),
        TransformerBlock(hidden_dim=12, num_heads=3),
    )
    x = torch.randn(2, 5, 12, requires_grad=True)

    output = stack(x)
    loss = output.pow(2).mean()
    loss.backward()

    assert not torch.isnan(output).any()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0
    for parameter in stack.parameters():
        assert parameter.grad is not None
        assert not torch.isnan(parameter.grad).any()
