import torch

from src.models.llama_components import (
    GroupedQueryAttention,
    KVCache,
    RMSNorm,
    SwiGLU,
    apply_rope,
    apply_rope_with_cache,
    repeat_kv,
    rotary_cache,
    rotary_frequencies,
)


def test_rmsnorm_preserves_shape_and_normalizes_rms() -> None:
    norm = RMSNorm(hidden_dim=4)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    output = norm(x)
    rms = output.pow(2).mean(dim=-1).sqrt()

    assert output.shape == x.shape
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rope_preserves_vector_norm_and_shape() -> None:
    x = torch.randn(2, 3, 4)
    freqs = rotary_frequencies(head_dim=4, seq_len=3)

    rotated = apply_rope(x, freqs)

    assert rotated.shape == x.shape
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-6)


def test_rope_with_position_ids_supports_cache_offsets() -> None:
    q = torch.randn(1, 4, 2, 6)
    k = torch.randn(1, 2, 2, 6)
    cos, sin = rotary_cache(head_dim=6, seq_len=8)
    position_ids = torch.tensor([[3, 4]], dtype=torch.long)

    q_rot, k_rot = apply_rope_with_cache(q, k, cos, sin, position_ids)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-6)
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-6)


def test_rope_rejects_odd_head_dim() -> None:
    q = torch.randn(1, 2, 3, 5)
    k = torch.randn(1, 2, 3, 5)
    cos, sin = rotary_cache(head_dim=6, seq_len=3)

    try:
        apply_rope_with_cache(q, k, cos, sin, torch.arange(3))
    except ValueError as exc:
        assert "even" in str(exc)
    else:
        raise AssertionError("Expected RoPE to reject odd head_dim.")


def test_swiglu_maps_back_to_hidden_dimension() -> None:
    ffn = SwiGLU(hidden_dim=6, intermediate_dim=16)
    x = torch.randn(2, 5, 6)

    output = ffn(x)

    assert output.shape == x.shape


def test_repeat_kv_expands_key_value_heads_for_gqa() -> None:
    x = torch.randn(2, 2, 4, 3)

    repeated = repeat_kv(x, num_query_heads=6, num_kv_heads=2)

    assert repeated.shape == (2, 6, 4, 3)
    assert torch.equal(repeated[:, 0], x[:, 0])
    assert torch.equal(repeated[:, 2], x[:, 0])
    assert torch.equal(repeated[:, 3], x[:, 1])


def test_repeat_kv_rejects_non_divisible_heads() -> None:
    x = torch.randn(2, 2, 4, 3)

    try:
        repeat_kv(x, num_query_heads=5, num_kv_heads=2)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected repeat_kv to reject non-divisible heads.")


def test_grouped_query_attention_returns_cache_and_respects_shapes() -> None:
    torch.manual_seed(0)
    attention = GroupedQueryAttention(hidden_dim=12, num_query_heads=4, num_kv_heads=2)
    x = torch.randn(2, 3, 12)

    output, cache = attention(x)

    assert output.shape == x.shape
    assert isinstance(cache, KVCache)
    assert cache.keys.shape == (2, 2, 3, 3)
    assert cache.values.shape == (2, 2, 3, 3)


def test_grouped_query_attention_appends_kv_cache_for_incremental_decode() -> None:
    torch.manual_seed(0)
    attention = GroupedQueryAttention(hidden_dim=12, num_query_heads=4, num_kv_heads=2)
    prefix = torch.randn(1, 3, 12)
    next_token = torch.randn(1, 1, 12)

    _, cache = attention(prefix)
    output, next_cache = attention(next_token, cache=cache)

    assert output.shape == (1, 1, 12)
    assert next_cache.keys.shape[-2] == 4
    assert next_cache.values.shape[-2] == 4
