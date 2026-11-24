"""Train a TruthTableNet on a logic gate and report its truth table.

Example:
    python -m truthtable.train --gate xor --epochs 2000
    python -m truthtable.train --gate and --save models/and.pt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from truthtable.dataset import GATES, TruthTableDataset, make_truth_table
from truthtable.model import DEFAULT_HIDDEN_SIZE, TruthTableNet


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    gate: str,
    epochs: int = 2000,
    lr: float = 0.05,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    batch_size: int = 4,
    seed: int = 0,
) -> tuple[TruthTableNet, list[float]]:
    """Train a network on ``gate``.

    Returns:
        The trained model and the mean training loss recorded for each epoch.
    """
    set_seed(seed)
    dataset = TruthTableDataset(gate)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TruthTableNet(hidden_size=hidden_size)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        history.append(epoch_loss / len(dataset))

    model.eval()
    return model, history


def accuracy(model: TruthTableNet, X: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of rows where the thresholded prediction matches the target."""
    preds = model.predict(X)
    return (preds == y).float().mean().item()


def format_truth_table(model: TruthTableNet, gate: str) -> str:
    """Render inputs, expected output and predicted output as a text table."""
    X, y = make_truth_table(gate)
    preds = model.predict(X)
    lines = [f"{'A':>2} {'B':>2} | {'expected':>8} {'predicted':>9}", "-" * 27]
    for row, target, pred in zip(X, y, preds):
        a, b = int(row[0]), int(row[1])
        lines.append(
            f"{a:>2} {b:>2} | {int(target.item()):>8} {int(pred.item()):>9}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a logic-gate network.")
    parser.add_argument("--gate", default="xor", choices=sorted(GATES))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path, default=None,
                        help="optional path to write a state_dict checkpoint")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    model, history = train(
        gate=args.gate,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    X, y = make_truth_table(args.gate)
    acc = accuracy(model, X, y)
    print(f"gate={args.gate} hidden={args.hidden} epochs={args.epochs} seed={args.seed}")
    print(f"final loss={history[-1]:.6f} accuracy={acc:.2%}")
    print(format_truth_table(model, args.gate))

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"saved checkpoint to {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
