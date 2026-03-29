"""Synthetic replay buffer for E2D continual learning.

Two budget modes
----------------
fixed_per_class=True  (recommended):
    Every class always gets exactly `max_size` images.
    Total buffer grows as: max_size × n_classes_seen.
    For CIFAR-10 with max_size=50: 50×10 = 500 images max — still tiny.

fixed_per_class=False (original, not recommended for small buffers):
    Total images capped at max_size, split equally across classes.
    Each class gets max_size // n_classes images.
    With max_size=200 and 10 classes → only 20 images/class → too few.
"""

from __future__ import annotations
from typing import Dict, Optional
import torch
from torch.utils.data import TensorDataset


class E2DBuffer:
    def __init__(self, max_size: int, fixed_per_class: bool = True):
        """
        Parameters
        ----------
        max_size : int
            If fixed_per_class=True  → images per class (e.g. 50).
            If fixed_per_class=False → total images across all classes.
        fixed_per_class : bool
            True  = each class always gets exactly max_size images (recommended).
            False = divide max_size equally among all classes seen so far.
        """
        self.max_size        = max_size
        self.fixed_per_class = fixed_per_class
        self.data:     Dict[int, torch.Tensor] = {}
        self.task_ids: Dict[int, int]          = {}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_classes(self) -> int:
        return len(self.data)

    @property
    def total_images(self) -> int:
        return sum(t.shape[0] for t in self.data.values())

    def budget_per_class(self, total_classes: int) -> int:
        if self.fixed_per_class:
            return self.max_size                          # always the same
        else:
            return max(1, self.max_size // max(1, total_classes))

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        class_id:      int,
        new_images:    torch.Tensor,
        total_classes: int,
        task_id:       int = 0,
    ) -> None:
        budget = self.budget_per_class(total_classes)

        if not self.fixed_per_class:
            # Trim existing classes to new (possibly smaller) budget
            for cid in list(self.data.keys()):
                if self.data[cid].shape[0] > budget:
                    self.data[cid] = self.data[cid][:budget].contiguous()

        self.data[class_id]     = new_images[:budget].detach().cpu().contiguous()
        self.task_ids[class_id] = int(task_id)

        mode_str = "fixed" if self.fixed_per_class else "shared"
        print(
            f"  [Buffer] class {class_id} → {self.data[class_id].shape[0]} imgs "
            f"| total {self.total_images} imgs "
            f"| {budget}/class ({mode_str} budget)"
        )

    # ── Access ────────────────────────────────────────────────────────────────

    def get_dataset(self) -> Optional[TensorDataset]:
        """Return all synthetic images as a TensorDataset, or None if empty."""
        if not self.data:
            return None
        imgs, labels, tasks = [], [], []
        for class_id in sorted(self.data.keys()):
            n = self.data[class_id].shape[0]
            imgs.append(self.data[class_id])
            labels.append(torch.full((n,), class_id, dtype=torch.long))
            tasks.append(
                torch.full((n,), self.task_ids.get(class_id, 0), dtype=torch.long)
            )
        return TensorDataset(
            torch.cat(imgs,   dim=0),
            torch.cat(labels, dim=0),
            torch.cat(tasks,  dim=0),
        )

    def get_class_images(self, class_id: int) -> Optional[torch.Tensor]:
        return self.data.get(class_id)

    def __repr__(self) -> str:
        counts = {cid: imgs.shape[0] for cid, imgs in self.data.items()}
        mode   = "fixed" if self.fixed_per_class else "shared"
        return (
            f"E2DBuffer(max_size={self.max_size}, mode={mode}, "
            f"total={self.total_images}, counts={counts})"
        )