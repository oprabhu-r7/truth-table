"""A small multilayer perceptron that learns 2-input logic gates.

XOR is not linearly separable, so a purely linear model cannot learn it.
A single hidden layer with a non-linear activation is enough for all three
gates (AND, OR, XOR).
"""
from __future__ import annotations

import torch
from torch import nn

# Number of hidden units used when none is given.
DEFAULT_HIDDEN_SIZE = 1


class TruthTableNet(nn.Module):
    """2 -> ``hidden_size`` -> 1 network.

    ``forward`` returns raw logits. Apply ``torch.sigmoid`` to get probabilities,
    or use :meth:`predict` to get hard 0/1 labels.
    """

    def __init__(self, hidden_size: int = DEFAULT_HIDDEN_SIZE) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, 1)."""
        return self.net(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return hard 0/1 predictions of shape (batch, 1)."""
        probs = torch.sigmoid(self.forward(x))
        return (probs >= threshold).float()
