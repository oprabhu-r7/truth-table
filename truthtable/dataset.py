"""Truth-table datasets for the AND, OR and XOR logic gates.

Each gate is a 2-input boolean function. The full truth table has four rows,
so the whole dataset is tiny and can be held in memory as two tensors:

    X: (4, 2) float32 - every combination of two bits
    y: (4, 1) float32 - the gate's output for each row
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

# Gate name -> boolean function over integer (0/1) tensors.
GATES = {
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
}


def truth_table_inputs() -> torch.Tensor:
    """Return all four 2-bit input combinations, shape (4, 2), float32."""
    return torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )


def make_truth_table(gate: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the inputs and targets for ``gate``.

    Args:
        gate: One of ``"and"``, ``"or"`` or ``"xor"`` (case-insensitive).

    Returns:
        ``(X, y)`` where ``X`` has shape (4, 2) and ``y`` has shape (4, 1).

    Raises:
        ValueError: If the gate name is unknown.
    """
    key = gate.lower()
    if key not in GATES:
        raise ValueError(f"unknown gate {gate!r}; expected one of {sorted(GATES)}")

    X = truth_table_inputs()
    a = X[:, 0].long()
    b = X[:, 1].long()
    y = GATES[key](a, b).float().unsqueeze(1)
    return X, y


class TruthTableDataset(Dataset):
    """A PyTorch ``Dataset`` yielding ``(inputs, target)`` pairs for one gate."""

    def __init__(self, gate: str) -> None:
        self.gate = gate.lower()
        self.X, self.y = make_truth_table(self.gate)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
