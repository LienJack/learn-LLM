import torch

from src.models.neural_lm import (
    CurrentTokenLanguageModel,
    NeuralLanguageModel,
    NeuralLMConfig,
    causal_mean,
    cosine_similarity_matrix,
    embedding_rows_with_grad,
    noncausal_mean,
    train_neural_language_model,
)


def test_embedding_outputs_expected_shape() -> None:
    model = NeuralLanguageModel(vocab_size=10, hidden_dim=6)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    hidden = model.embed(input_ids)

    assert hidden.shape == (2, 3, 6)


def test_current_token_model_outputs_expected_shape() -> None:
    model = CurrentTokenLanguageModel(vocab_size=10, hidden_dim=6)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    logits, loss = model(input_ids)

    assert logits.shape == (2, 3, 10)
    assert loss is None


def test_forward_outputs_logits_and_loss() -> None:
    model = NeuralLanguageModel(vocab_size=10, hidden_dim=6)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    labels = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long)

    logits, loss = model(input_ids, labels)

    assert logits.shape == (2, 3, 10)
    assert loss is not None
    assert loss.ndim == 0


def test_only_seen_embedding_rows_receive_gradient() -> None:
    model = NeuralLanguageModel(vocab_size=8, hidden_dim=4)
    input_ids = torch.tensor([[1, 2, 1], [2, 3, 3]], dtype=torch.long)
    labels = torch.tensor([[2, 3, 2], [3, 4, 4]], dtype=torch.long)

    _, loss = model(input_ids, labels)
    assert loss is not None
    loss.backward()

    assert embedding_rows_with_grad(model) == {1, 2, 3}


def test_padding_embedding_row_is_not_updated() -> None:
    model = NeuralLanguageModel(vocab_size=6, hidden_dim=4, padding_idx=0)
    input_ids = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    labels = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
    before = model.token_embedding.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    _, loss = model(input_ids, labels)
    assert loss is not None
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = model.token_embedding.weight.detach()
    assert torch.allclose(before[0], after[0])
    assert not torch.allclose(before[1], after[1])


def test_causal_mean_output_shape() -> None:
    hidden = torch.randn(2, 4, 6)

    context = causal_mean(hidden)

    assert context.shape == (2, 4, 6)


def test_causal_mean_ignores_future_tokens() -> None:
    hidden = torch.randn(1, 5, 4)
    changed = hidden.clone()
    changed[:, 3:, :] = torch.randn_like(changed[:, 3:, :]) * 100.0

    original_context = causal_mean(hidden)
    changed_context = causal_mean(changed)

    assert torch.allclose(original_context[:, :3], changed_context[:, :3])


def test_noncausal_mean_fails_no_future_dependency() -> None:
    hidden = torch.randn(1, 5, 4)
    changed = hidden.clone()
    changed[:, 3:, :] = torch.randn_like(changed[:, 3:, :]) * 100.0

    original_context = noncausal_mean(hidden)
    changed_context = noncausal_mean(changed)

    assert not torch.allclose(original_context[:, :3], changed_context[:, :3])


def test_causal_mean_model_logits_ignore_future_tokens() -> None:
    model = NeuralLanguageModel(vocab_size=20, hidden_dim=8)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    changed_input_ids = input_ids.clone()
    changed_input_ids[:, 3:] = torch.tensor([[18, 19]], dtype=torch.long)

    logits, _ = model(input_ids)
    changed_logits, _ = model(changed_input_ids)

    assert torch.allclose(logits[:, :3], changed_logits[:, :3])


def test_tiny_corpus_loss_decreases() -> None:
    pattern = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    token_ids = pattern.repeat(64)
    config = NeuralLMConfig(seed=0, hidden_dim=16, batch_size=16, block_size=6, lr=0.03, steps=80)

    _, history = train_neural_language_model(token_ids, vocab_size=5, config=config)

    assert history.losses[-1] < history.losses[0] * 0.35


def test_cosine_similarity_matrix_has_expected_shape_and_diagonal() -> None:
    model = NeuralLanguageModel(vocab_size=6, hidden_dim=4)

    similarity = cosine_similarity_matrix(model.token_embedding.weight, [1, 2, 3])

    assert similarity.shape == (3, 3)
    assert torch.allclose(torch.diag(similarity), torch.ones(3), atol=1e-6)
