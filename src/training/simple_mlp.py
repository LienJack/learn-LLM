from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset


class ToyClassificationDataset(Dataset[tuple[Tensor, Tensor]]):
    """A two-class dataset made from 2D points separated by a noisy line."""

    def __init__(
        self,
        num_samples: int = 512,
        seed: int = 42,
        noise_std: float = 0.15,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(num_samples, 2, generator=generator)
        noise = noise_std * torch.randn(num_samples, generator=generator)
        boundary_score = self.features[:, 0] + 0.7 * self.features[:, 1] + noise
        self.labels = (boundary_score > 0).long()

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.features[index], self.labels[index]


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    num_samples: int = 512
    train_ratio: float = 0.8
    batch_size: int = 32
    hidden_dim: int = 32
    dropout: float = 0.0
    noise_std: float = 0.15
    lr: float = 0.1
    epochs: int = 20
    device: str = "cpu"


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    grad_norm: float | None = None
    update_norm: float | None = None


@dataclass
class HistoryItem:
    epoch: int
    train: EpochMetrics
    val: EpochMetrics | None


@dataclass
class TrainingHistory:
    items: list[HistoryItem] = field(default_factory=list)

    @property
    def train_losses(self) -> list[float]:
        return [item.train.loss for item in self.items]

    @property
    def val_losses(self) -> list[float]:
        return [item.val.loss for item in self.items if item.val is not None]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_dataset(
    dataset: Dataset[tuple[Tensor, Tensor]],
    train_ratio: float,
    seed: int,
) -> tuple[Subset[tuple[Tensor, Tensor]], Subset[tuple[Tensor, Tensor]] | None]:
    if not 0 < train_ratio <= 1:
        raise ValueError("train_ratio must be in (0, 1].")

    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    train_size = int(len(indices) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    return train_dataset, val_dataset


def make_dataloaders(
    config: TrainingConfig,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]] | None]:
    dataset = ToyClassificationDataset(
        num_samples=config.num_samples,
        seed=config.seed,
        noise_std=config.noise_std,
    )
    train_dataset, val_dataset = split_dataset(dataset, config.train_ratio, config.seed + 1)
    train_generator = torch.Generator().manual_seed(config.seed + 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader


def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().pow(2).sum().item()
    return total**0.5


def compute_update_norm(params_before: list[Tensor], model: nn.Module) -> float:
    total = 0.0
    params_after = [p.detach() for p in model.parameters() if p.requires_grad]
    for before, after in zip(params_before, params_after, strict=True):
        total += (after - before).pow(2).sum().item()
    return total**0.5


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str = "cpu",
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_grad_norm = 0.0
    total_update_norm = 0.0
    num_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        params_before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        grad_norm = compute_grad_norm(model)
        optimizer.step()
        update_norm = compute_update_norm(params_before, model)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += batch_size
        total_grad_norm += grad_norm
        total_update_norm += update_norm
        num_batches += 1

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        grad_norm=total_grad_norm / num_batches,
        update_norm=total_update_norm / num_batches,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    device: str = "cpu",
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += batch_size

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )


def run_training(config: TrainingConfig) -> tuple[SimpleMLP, TrainingHistory]:
    set_seed(config.seed)
    train_loader, val_loader = make_dataloaders(config)
    model = SimpleMLP(hidden_dim=config.hidden_dim, dropout=config.dropout).to(config.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    history = TrainingHistory()

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, config.device)
        val_metrics = evaluate(model, val_loader, loss_fn, config.device) if val_loader else None
        history.items.append(HistoryItem(epoch=epoch, train=train_metrics, val=val_metrics))

    return model, history


def run_overfit_tiny_experiment() -> tuple[SimpleMLP, TrainingHistory]:
    config = TrainingConfig(
        seed=0,
        num_samples=16,
        train_ratio=1.0,
        batch_size=16,
        hidden_dim=64,
        dropout=0.0,
        noise_std=0.0,
        lr=0.1,
        epochs=200,
    )
    return run_training(config)


def print_history(history: TrainingHistory) -> None:
    for item in history.items:
        val_text = (
            f" val_loss={item.val.loss:.4f} val_acc={item.val.accuracy:.3f}"
            if item.val is not None
            else ""
        )
        print(
            f"epoch={item.epoch:03d} "
            f"train_loss={item.train.loss:.4f} "
            f"train_acc={item.train.accuracy:.3f} "
            f"grad_norm={item.train.grad_norm:.4f} "
            f"update_norm={item.train.update_norm:.4f}"
            f"{val_text}"
        )


def build_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    if args.experiment == "overfit_tiny":
        return TrainingConfig(
            seed=args.seed,
            num_samples=16,
            train_ratio=1.0,
            batch_size=16,
            hidden_dim=64,
            dropout=0.0,
            noise_std=0.0,
            lr=args.lr,
            epochs=args.epochs or 200,
        )

    return TrainingConfig(
        seed=args.seed,
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        noise_std=args.noise_std,
        lr=args.lr,
        epochs=args.epochs or 20,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chapter 1 MLP training experiments.")
    parser.add_argument(
        "--experiment",
        choices=["baseline", "overfit_tiny"],
        default="baseline",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = build_config_from_args(args)
    _, history = run_training(config)
    print_history(history)


if __name__ == "__main__":
    main()
