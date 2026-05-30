import torch

from src.models.mini_gpt import (
    MiniGPT,
    MiniGPTConfig,
    MiniGPTTrainingConfig,
    generate,
    load_checkpoint,
    save_checkpoint,
    train_mini_gpt,
)
from src.tokenizer.simple_tokenizer import CharacterTokenizer


def test_mini_gpt_forward_returns_logits_and_loss() -> None:
    config = MiniGPTConfig(vocab_size=12, block_size=6, hidden_dim=12, num_layers=2, num_heads=3)
    model = MiniGPT(config)
    input_ids = torch.randint(0, 12, (2, 6))
    labels = torch.randint(0, 12, (2, 6))

    logits, loss = model(input_ids, labels)

    assert logits.shape == (2, 6, 12)
    assert loss is not None
    assert loss.ndim == 0


def test_mini_gpt_rejects_sequences_longer_than_block_size() -> None:
    model = MiniGPT(MiniGPTConfig(vocab_size=12, block_size=4, hidden_dim=12, num_heads=3))
    input_ids = torch.randint(0, 12, (2, 5))

    try:
        model(input_ids)
    except ValueError as exc:
        assert "block_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError for overlong input.")


def test_causal_mask_prevents_future_token_leakage() -> None:
    torch.manual_seed(0)
    model = MiniGPT(MiniGPTConfig(vocab_size=20, block_size=6, hidden_dim=12, num_heads=3))
    model.eval()
    original = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    changed_future = original.clone()
    changed_future[0, 4:] = torch.tensor([10, 11])

    logits_a, _ = model(original)
    logits_b, _ = model(changed_future)

    assert torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-6)
    assert not torch.allclose(logits_a[:, 4:], logits_b[:, 4:])


def test_checkpoint_round_trip_preserves_parameters_and_tokenizer(tmp_path) -> None:
    tokenizer = CharacterTokenizer.from_texts(["hello"])
    config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=4,
        hidden_dim=12,
        num_heads=3,
    )
    model = MiniGPT(config)
    path = tmp_path / "mini_gpt.pt"

    save_checkpoint(path, model, tokenizer, extra={"step": 3})
    loaded_model, loaded_tokenizer, extra = load_checkpoint(path)

    assert loaded_tokenizer.token_to_id == tokenizer.token_to_id
    assert extra == {"step": 3}
    for original, loaded in zip(model.parameters(), loaded_model.parameters(), strict=True):
        assert torch.allclose(original, loaded)


def test_generate_crops_context_and_respects_max_new_tokens() -> None:
    torch.manual_seed(0)
    model = MiniGPT(MiniGPTConfig(vocab_size=8, block_size=4, hidden_dim=12, num_heads=3))
    prompt = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)

    output = generate(
        model,
        prompt,
        max_new_tokens=3,
        top_k=2,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.shape == (9,)
    assert torch.equal(output[:6], prompt)
    assert output.max().item() < 8


def test_generate_stops_at_eos_token() -> None:
    model = MiniGPT(MiniGPTConfig(vocab_size=5, block_size=4, hidden_dim=12, num_heads=3))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.lm_head.bias[2] = 10.0

    output = generate(
        model,
        torch.tensor([1], dtype=torch.long),
        max_new_tokens=5,
        eos_token_id=2,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.tolist() == [1, 2]


def test_tiny_corpus_training_reduces_loss() -> None:
    pattern = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    model_config = MiniGPTConfig(
        vocab_size=5,
        block_size=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
    )
    train_config = MiniGPTTrainingConfig(seed=0, batch_size=16, lr=0.01, steps=50)

    _, history = train_mini_gpt(pattern.repeat(64), model_config, train_config)

    assert history.losses[-1] < history.losses[0] * 0.5
