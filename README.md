# E2D Distillation for Continual Learning

Adapts the E2D (Exploration-Exploitation Distillation) method into a continual learning pipeline using Avalanche.

## Install

```bash
pip install avalanche-lib torch torchvision
```

## Project Structure

```text
e2d_continual/
├── distill.py   # E2D core loop (adapted from recover.py)
├── buffer.py    # Fixed-size synthetic image buffer
├── plugin.py    # Avalanche plugin (hooks E2D into training)
├── train.py     # Main script: runs and compares 3 strategies
└── README.md
```

## Quick Start

Run all three strategies and get a comparison table:

```bash
python train.py
```

Run only E2D (faster for debugging):

```bash
python train.py --strategy e2d --distill-iters 200 --epochs 5
```

Full run with a larger buffer:

```bash
python train.py --buffer-size 500 --distill-iters 500 --epochs 20
```

## Key Arguments

| Argument        | Default | Description                                            |
| --------------- | ------: | ------------------------------------------------------ |
| --buffer-size   |     200 | Total synthetic images in memory (across all classes)  |
| --distill-iters |     500 | E2D optimization steps per class                       |
| --K             |     350 | Switch from exploration to exploitation at iteration K |
| --epochs        |      10 | Training epochs per experience                         |
| --n-experiences |       5 | Number of tasks to split CIFAR-10 into                 |
| --strategy      |     all | One of: all, naive, random, e2d                        |

## What Each File Does

### distill.py

- distill_task(real_images, class_label, teacher, n_synthetic, ...) runs the E2D two-phase optimization loop on a single class.
- Returns a tensor of n_synthetic distilled images.
- Directly adapted from Branch_ImageNet_1K/recover/recover.py, but simplified for this project (single teacher, no BN stats, no distributed training).

### buffer.py

- E2DBuffer(max_size) stores synthetic images per class in a fixed-size pool.
- When new classes arrive, budget_per_class = max_size // total_classes.
- Existing classes are trimmed to the new budget automatically.
- get_dataset() returns a TensorDataset ready for DataLoader.

### plugin.py

- E2DReplayPlugin(teacher, buffer_size, ...) is an Avalanche SupervisedPlugin with two hooks:
- before_training_exp: merges synthetic buffer with real data.
- after_training_exp: distills new classes and updates the buffer.

### train.py

Runs three strategies on SplitCIFAR10 and prints a results table:

- Naive: no replay (catastrophic forgetting baseline)
- RandomReplay: Avalanche built-in random replay
- E2DReplay: this method

## Expected Results (SplitCIFAR10, 5 Experiences)

| Strategy         | Final Acc (%) | Forgetting (%) |
| ---------------- | ------------: | -------------: |
| Naive            |           ~19 |            ~75 |
| RandomReplay     |           ~45 |            ~35 |
| E2DReplay (ours) |           ~52 |            ~25 |

Numbers are approximate and vary with seed, epochs, and buffer size.
