import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.simple_mlp import (
    SimpleMLP,
    ToyClassificationDataset,
    TrainingConfig,
    compute_grad_norm,
    compute_update_norm,
    evaluate,
    run_overfit_tiny_experiment,
    run_training,
    split_dataset,
    train_one_epoch,
)


def test_dataset_returns_feature_and_label_with_expected_shapes() -> None:
    dataset = ToyClassificationDataset(num_samples=8, seed=0)

    x, y = dataset[0]

    assert x.shape == (2,)
    assert x.dtype == torch.float32
    assert y.shape == ()
    assert y.dtype == torch.long
    assert y.item() in {0, 1}


def test_model_forward_outputs_class_logits() -> None:
    model = SimpleMLP(input_dim=2, hidden_dim=8, num_classes=2)
    x = torch.randn(4, 2)

    logits = model(x)

    assert logits.shape == (4, 2)


def test_train_val_split_has_no_overlap() -> None:
    dataset = ToyClassificationDataset(num_samples=100, seed=0)
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8, seed=0)

    assert val_dataset is not None
    assert set(train_dataset.indices).isdisjoint(set(val_dataset.indices))
    assert len(train_dataset) == 80
    assert len(val_dataset) == 20


def test_train_one_epoch_reports_grad_and_update_norms() -> None:
    dataset = ToyClassificationDataset(num_samples=64, seed=0)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = SimpleMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    metrics = train_one_epoch(model, dataloader, optimizer, loss_fn)

    assert metrics.loss > 0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.grad_norm is not None
    assert metrics.update_norm is not None
    assert metrics.grad_norm > 0
    assert metrics.update_norm > 0


def test_training_loss_decreases_over_epochs() -> None:
    config = TrainingConfig(seed=0, epochs=20, lr=0.1)

    _, history = run_training(config)

    first_loss = history.items[0].train.loss
    last_loss = history.items[-1].train.loss
    assert len(history.items) == 20
    assert last_loss < first_loss * 0.8


def test_parameters_are_updated_after_one_epoch() -> None:
    dataset = ToyClassificationDataset(num_samples=64, seed=0)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = SimpleMLP()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    train_one_epoch(model, dataloader, optimizer, loss_fn)

    after = [parameter.detach().clone() for parameter in model.parameters()]
    parameter_changed = [
        not torch.allclose(before_param, after_param)
        for before_param, after_param in zip(before, after, strict=True)
    ]
    assert any(parameter_changed)


def test_evaluate_does_not_create_gradients() -> None:
    dataset = ToyClassificationDataset(num_samples=32, seed=0)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = SimpleMLP()
    loss_fn = nn.CrossEntropyLoss()

    for parameter in model.parameters():
        parameter.grad = None

    metrics = evaluate(model, dataloader, loss_fn)

    assert metrics.loss > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fixed_seed_makes_training_reproducible() -> None:
    config = TrainingConfig(seed=123, epochs=5)

    model_a, history_a = run_training(config)
    model_b, history_b = run_training(config)

    losses_a = [item.train.loss for item in history_a.items]
    losses_b = [item.train.loss for item in history_b.items]
    assert losses_a == pytest.approx(losses_b)

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(param_a, param_b)


def test_model_can_overfit_tiny_dataset() -> None:
    _, history = run_overfit_tiny_experiment()

    final_train = history.items[-1].train

    assert final_train.accuracy >= 0.95
    assert final_train.loss < history.items[0].train.loss


def test_train_and_eval_modes_change_dropout_behavior() -> None:
    torch.manual_seed(0)
    model = SimpleMLP(dropout=0.5)
    x = torch.randn(8, 2)

    model.train()
    out_train_a = model(x)
    out_train_b = model(x)

    model.eval()
    out_eval_a = model(x)
    out_eval_b = model(x)

    assert not torch.allclose(out_train_a, out_train_b)
    assert torch.allclose(out_eval_a, out_eval_b)


def test_compute_norm_helpers_report_positive_values_after_backward_and_step() -> None:
    dataset = ToyClassificationDataset(num_samples=16, seed=0)
    x, y = next(iter(DataLoader(dataset, batch_size=16)))
    model = SimpleMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    params_before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = loss_fn(model(x), y)
    loss.backward()
    grad_norm = compute_grad_norm(model)
    optimizer.step()
    update_norm = compute_update_norm(params_before, model)

    assert grad_norm > 0
    assert update_norm > 0
