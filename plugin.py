"""
plugin.py
---------
Avalanche plugin that integrates E2D dataset distillation into any
continual learning strategy.

Hooks used
----------
before_training_exp : inject synthetic replay data into the dataloader
after_training_exp  : distil the just-finished task and update the buffer
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from distill import distill_task
from buffer import E2DBuffer


# ──────────────────────────────────────────────────────────────────────────────
#  Helper — extract all images of a specific class from an Avalanche dataset
# ──────────────────────────────────────────────────────────────────────────────
def _get_class_tensors(
    dataset,
    class_id: int,
    device: torch.device,
    max_samples: int = 500,
) -> torch.Tensor:
    """
    Iterate over `dataset`, collect images whose label == class_id,
    and return them as a single tensor on `device`.

    Avalanche datasets return (img, label, task_label) tuples.
    """
    imgs = []
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    for batch in loader:
        # Avalanche gives (x, y, t) triples
        x, y = batch[0], batch[1]
        mask = (y == class_id)
        if mask.any():
            imgs.append(x[mask])
        if sum(t.shape[0] for t in imgs) >= max_samples:
            break
    if not imgs:
        return torch.empty(0)
    return torch.cat(imgs, dim=0)[:max_samples].to(device)


# ──────────────────────────────────────────────────────────────────────────────
#  Avalanche-compatible merged dataset
#  (wraps real AvalancheDataset + synthetic TensorDataset)
# ──────────────────────────────────────────────────────────────────────────────
class _MergedDataset(torch.utils.data.Dataset):
    """
    Concatenates an Avalanche experience dataset (returns x, y, t)
    with a synthetic TensorDataset (returns x, y).

    All synthetic samples get task_label = 0 so Avalanche's internals
    stay happy. This is fine because we only use a single-head classifier.
    """

    def __init__(self, real_dataset, synthetic_dataset: TensorDataset):
        self.real      = real_dataset
        self.synthetic = synthetic_dataset
        self.real_len  = len(real_dataset)
        self.syn_len   = len(synthetic_dataset)

    def __len__(self):
        return self.real_len + self.syn_len

    def __getitem__(self, idx):
        if idx < self.real_len:
            return self.real[idx]           # (x, y, task_label)
        syn_idx       = idx - self.real_len
        x, y          = self.synthetic[syn_idx]
        return x, y, torch.tensor(0)        # fake task_label = 0


# ──────────────────────────────────────────────────────────────────────────────
#  The plugin
# ──────────────────────────────────────────────────────────────────────────────
class E2DReplayPlugin(SupervisedPlugin):
    """
    Avalanche plugin that uses E2D distillation as a replay mechanism.

    Parameters
    ----------
    teacher        : frozen model used as the distillation teacher
    buffer_size    : total number of synthetic images kept in memory
    distill_iters  : optimisation iterations inside E2D per class
    K              : exploration → exploitation switch iteration
    loss_threshold : ε — minimum per-crop CE loss to enter the buffer
    lr             : learning rate for the synthetic-image optimiser
    img_size       : spatial size of images (32 for CIFAR)
    device         : device to run distillation on
    """

    def __init__(
        self,
        teacher: torch.nn.Module,
        buffer_size:    int   = 200,
        distill_iters:  int   = 500,
        K:              int   = 350,
        loss_threshold: float = 0.5,
        lr:             float = 0.05,
        img_size:       int   = 32,
        device:         torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.teacher        = teacher
        self.buffer         = E2DBuffer(max_size=buffer_size)
        self.distill_iters  = distill_iters
        self.K              = K
        self.loss_threshold = loss_threshold
        self.lr             = lr
        self.img_size       = img_size
        self.device         = device
        self.classes_seen   = 0          # cumulative class counter

    # ── Hook 1: inject synthetic replay BEFORE training starts ─────────────
    def before_training_exp(self, strategy, **kwargs):
        syn_dataset = self.buffer.get_dataset()
        if syn_dataset is None:
            print("[E2DPlugin] First experience — no replay data yet.")
            return

        print(
            f"[E2DPlugin] Merging {len(syn_dataset)} synthetic images "
            f"with real data for replay."
        )
        merged = _MergedDataset(
            strategy.adapted_dataset, syn_dataset
        )
        # Replace the dataloader with a merged one
        strategy.dataloader = DataLoader(
            merged,
            batch_size=strategy.train_mb_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    # ── Hook 2: distil new task classes AFTER training finishes ─────────────
    def after_training_exp(self, strategy, **kwargs):
        exp          = strategy.experience
        new_classes  = exp.classes_in_this_experience
        dataset      = exp.dataset
        self.classes_seen += len(new_classes)

        print(
            f"\n[E2DPlugin] Distilling {len(new_classes)} new class(es): "
            f"{new_classes}  |  total classes so far: {self.classes_seen}"
        )

        budget = self.buffer.budget_per_class(self.classes_seen)

        for class_id in new_classes:
            print(f"  → distilling class {class_id} "
                  f"(target: {budget} synthetic images)...")

            # Collect real images for this class
            real_imgs = _get_class_tensors(
                dataset, class_id, self.device, max_samples=500
            )
            if real_imgs.shape[0] == 0:
                print(f"  ⚠ No images found for class {class_id}, skipping.")
                continue

            # Run E2D distillation
            synthetic = distill_task(
                real_images     = real_imgs,
                class_label     = class_id,
                teacher         = self.teacher,
                n_synthetic     = budget,
                device          = self.device,
                iterations      = self.distill_iters,
                K               = self.K,
                loss_threshold  = self.loss_threshold,
                lr              = self.lr,
                img_size        = self.img_size,
            )

            # Store in fixed-size buffer
            self.buffer.update(class_id, synthetic, self.classes_seen)

        print(f"[E2DPlugin] Buffer after experience: {self.buffer}\n")
