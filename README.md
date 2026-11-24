# Truth Table Learner

A small PyTorch project that trains a neural network to reproduce the truth
tables of three 2-input logic gates:

| A | B | AND | OR | XOR |
|---|---|-----|----|-----|
| 0 | 0 |  0  | 0  |  0  |
| 0 | 1 |  0  | 1  |  1  |
| 1 | 0 |  0  | 1  |  1  |
| 1 | 1 |  1  | 1  |  0  |

AND and OR are linearly separable, but XOR is not. That is why the model is a
small multilayer perceptron with a hidden layer rather than a single linear
unit. Learning XOR is the interesting part of the project.

## Setup

Requires Python 3.9+ and PyTorch.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m truthtable.train --gate xor
python -m truthtable.train --gate and --epochs 1000 --lr 0.05
python -m truthtable.train --gate or --save models/or.pt
```

Each run prints the final loss, the accuracy, and the predicted truth table.

| Flag | Default | Meaning |
|------|---------|---------|
| `--gate` | `xor` | `and`, `or` or `xor` |
| `--epochs` | `2000` | Passes over the 4-row dataset |
| `--lr` | `0.05` | Adam learning rate |
| `--hidden` | `DEFAULT_HIDDEN_SIZE` (in `truthtable/model.py`) | Hidden units |
| `--batch-size` | `4` | Mini-batch size (4 = full batch) |
| `--seed` | `0` | Random seed for reproducibility |
| `--save` | none | Write a `state_dict` checkpoint to this path |

## Test

```bash
python -m pytest -q
```

## Project layout

```
truthtable/
  dataset.py   Truth tables as tensors, plus a torch Dataset class
  model.py     TruthTableNet: 2 -> hidden (tanh) -> 1
  train.py     Training loop, accuracy, truth-table printing, CLI
tests/
  test_dataset.py  Truth-table correctness for each gate
  test_model.py    Model shapes, outputs and configuration
  test_train.py    End-to-end: each gate must reach 100% accuracy
docs/
  ARCHITECTURE.md  Design notes and how the pieces fit together
```

For the design, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
