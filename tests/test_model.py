import pytest
import torch

from truthtable.dataset import truth_table_inputs
from truthtable.model import DEFAULT_HIDDEN_SIZE, TruthTableNet


def test_forward_returns_one_logit_per_row():
    model = TruthTableNet(hidden_size=3)
    out = model(truth_table_inputs())
    assert out.shape == (4, 1)


def test_predict_is_binary():
    model = TruthTableNet()
    preds = model.predict(truth_table_inputs())
    assert preds.shape == (4, 1)
    assert set(preds.unique().tolist()) <= {0.0, 1.0}


def test_default_hidden_size_is_used():
    model = TruthTableNet()
    assert model.hidden_size == DEFAULT_HIDDEN_SIZE
    assert model.net[0].out_features == DEFAULT_HIDDEN_SIZE


def test_invalid_hidden_size_raises():
    with pytest.raises(ValueError):
        TruthTableNet(hidden_size=0)


def test_predict_respects_threshold():
    torch.manual_seed(0)
    model = TruthTableNet(hidden_size=2)
    X = truth_table_inputs()
    always_one = model.predict(X, threshold=0.0)
    always_zero = model.predict(X, threshold=1.1)
    assert torch.all(always_one == 1)
    assert torch.all(always_zero == 0)
