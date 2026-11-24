# Architecture

## Data

`truthtable/dataset.py` holds the whole problem. A gate's truth table is four
rows, so it is built in memory on demand:

- `truth_table_inputs()` returns `X` with shape `(4, 2)`, containing every
  combination of two bits in the order `00, 01, 10, 11`.
- `make_truth_table(gate)` applies the gate's boolean function to `X` and
  returns `(X, y)`, where `y` has shape `(4, 1)`. Gates are defined in the
  `GATES` dict and implemented with integer bitwise operators (`&`, `|`, `^`).
- `TruthTableDataset(gate)` wraps the same data as a `torch.utils.data.Dataset`
  so it can be used with a `DataLoader`.

## Model

`truthtable/model.py` defines `TruthTableNet`:

```
Linear(2 -> H) -> Tanh -> Linear(H -> 1)
```

- `forward` returns logits. They are not squashed, so the loss can use
  `BCEWithLogitsLoss`, which is numerically more stable than sigmoid followed by
  BCE.
- `predict` applies a sigmoid and thresholds at 0.5 to produce hard 0/1 labels.
- `H` (the hidden width) is a parameter, defaulting to `DEFAULT_HIDDEN_SIZE`.

Why a hidden layer: a single linear unit draws one straight decision boundary.
AND and OR can be separated by one line, but XOR cannot. A hidden layer with a
non-linear activation can carve out the two diagonal regions XOR requires.

## Training

`truthtable/train.py` contains `train()`:

1. Seed Python, NumPy and PyTorch, so runs are reproducible.
2. Build the dataset and a shuffling `DataLoader`.
3. Optimise with Adam and `BCEWithLogitsLoss`.
4. Return the model (in eval mode) and the mean loss for each epoch.

`accuracy()` compares thresholded predictions with the targets over the full
table. `main()` is the CLI wrapper around these functions.

## Tests

| File | What it guards |
|------|----------------|
| `tests/test_dataset.py` | Truth-table values for each gate; input ordering; error on unknown gate |
| `tests/test_model.py` | Output shapes, binary predictions, hidden-size configuration |
| `tests/test_train.py` | Every gate reaches 100% accuracy; loss decreases; training is reproducible |

The end-to-end test in `test_train.py` is the one that matters most. Unit tests
can pass while the model as a whole still fails to learn a gate.
