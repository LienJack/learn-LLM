import torch
from torch import nn

from src.models.bigram_lm import (
    BigramLanguageModel,
    LanguageModelingConfig,
    generate,
    language_modeling_loss,
    make_lm_batch,
    sample_next_token,
    train_bigram_language_model,
)


def test_make_lm_batch_returns_inputs_and_right_shifted_labels() -> None:
    token_ids = torch.arange(12, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)

    inputs, labels = make_lm_batch(token_ids, block_size=4, batch_size=3, generator=generator)

    assert inputs.shape == (3, 4)
    assert labels.shape == (3, 4)
    assert torch.equal(labels[:, :-1], inputs[:, 1:])


def test_bigram_language_model_outputs_logits_and_loss() -> None:
    model = BigramLanguageModel(vocab_size=5)
    input_ids = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
    labels = torch.tensor([[1, 2, 3], [3, 4, 0]], dtype=torch.long)

    logits, loss = model(input_ids, labels)

    assert logits.shape == (2, 3, 5)
    assert loss is not None
    assert loss.ndim == 0


def test_language_modeling_loss_matches_cross_entropy_flattening() -> None:
    logits = torch.randn(2, 3, 4)
    labels = torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.long)

    loss = language_modeling_loss(logits, labels)
    expected = nn.functional.cross_entropy(logits.reshape(6, 4), labels.reshape(6))

    assert torch.allclose(loss, expected)


def test_train_bigram_language_model_reduces_tiny_corpus_loss() -> None:
    pattern = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    token_ids = pattern.repeat(64)
    config = LanguageModelingConfig(seed=0, block_size=6, batch_size=16, lr=0.1, steps=80)

    _, history = train_bigram_language_model(token_ids, vocab_size=4, config=config)

    assert history.losses[-1] < history.losses[0] * 0.25


def test_generate_appends_tokens_inside_vocabulary() -> None:
    pattern = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    token_ids = pattern.repeat(32)
    config = LanguageModelingConfig(seed=0, block_size=4, batch_size=8, lr=0.1, steps=50)
    model, _ = train_bigram_language_model(token_ids, vocab_size=4, config=config)

    output = generate(
        model,
        prompt_ids=torch.tensor([0], dtype=torch.long),
        max_new_tokens=6,
        temperature=0.8,
        top_k=2,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.shape == (7,)
    assert output.min().item() >= 0
    assert output.max().item() < 4


def test_sample_next_token_respects_top_k() -> None:
    logits = torch.tensor([0.0, 1.0, 10.0, 9.0])

    samples = [
        sample_next_token(logits, top_k=2, generator=torch.Generator().manual_seed(seed)).item()
        for seed in range(10)
    ]

    assert set(samples).issubset({2, 3})


def test_sample_next_token_respects_top_p() -> None:
    logits = torch.tensor([10.0, 1.0, 0.5, 0.0])

    samples = [
        sample_next_token(logits, top_p=0.5, generator=torch.Generator().manual_seed(seed)).item()
        for seed in range(10)
    ]

    assert set(samples) == {0}


def test_perplexity_matches_exp_loss() -> None:
    logits = torch.randn(2, 3, 4)
    labels = torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.long)

    loss = language_modeling_loss(logits, labels)
    perplexity = torch.exp(loss)

    assert torch.allclose(perplexity, torch.tensor(loss.item()).exp())


def test_generate_stops_at_eos_token() -> None:
    model = BigramLanguageModel(vocab_size=3)
    with torch.no_grad():
        model.token_logits.weight.fill_(-10.0)
        model.token_logits.weight[:, 2] = 10.0

    output = generate(
        model,
        prompt_ids=torch.tensor([0], dtype=torch.long),
        max_new_tokens=5,
        eos_token_id=2,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.tolist() == [0, 2]
