import pytest

from truthtable.dataset import make_truth_table
from truthtable.train import accuracy, train


@pytest.mark.parametrize("gate", ["and", "or", "xor"])
def test_model_learns_gate_perfectly(gate):
    model, _ = train(gate, epochs=2000, seed=0)
    X, y = make_truth_table(gate)
    assert accuracy(model, X, y) == 1.0


def test_loss_decreases_during_training():
    _, history = train("and", epochs=200, seed=0)
    assert history[-1] < history[0]


def test_training_is_reproducible():
    _, h1 = train("or", epochs=50, seed=123)
    _, h2 = train("or", epochs=50, seed=123)
    assert h1 == h2
