import pytest
import torch

from truthtable.dataset import GATES, TruthTableDataset, make_truth_table, truth_table_inputs

EXPECTED = {
    "and": [0, 0, 0, 1],
    "or": [0, 1, 1, 1],
    "xor": [0, 1, 1, 0],
}


def test_inputs_cover_all_four_combinations():
    X = truth_table_inputs()
    assert X.shape == (4, 2)
    assert X.dtype == torch.float32
    assert X.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


@pytest.mark.parametrize("gate", ["and", "or", "xor"])
def test_targets_match_boolean_truth_table(gate):
    X, y = make_truth_table(gate)
    assert X.shape == (4, 2)
    assert y.shape == (4, 1)
    assert y.squeeze(1).tolist() == EXPECTED[gate]


def test_gate_name_is_case_insensitive():
    _, y_lower = make_truth_table("xor")
    _, y_upper = make_truth_table("XOR")
    assert torch.equal(y_lower, y_upper)


def test_unknown_gate_raises():
    with pytest.raises(ValueError):
        make_truth_table("nand")


def test_gate_registry_is_complete():
    assert set(GATES) == {"and", "or", "xor"}


def test_dataset_len_and_items():
    ds = TruthTableDataset("or")
    assert len(ds) == 4
    x, y = ds[3]
    assert x.tolist() == [1.0, 1.0]
    assert y.tolist() == [1.0]
