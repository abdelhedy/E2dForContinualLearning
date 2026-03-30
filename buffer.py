from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import TensorDataset


class E2DBuffer:
    """Synthetic replay buffer with optional per-class fixed budget.

    The buffer stores distilled images together with hard labels, task ids, and
    teacher relabeling logits used for replay KD.
    """

    def __init__(self, max_size: int, fixed_per_class: bool = True):
        self.max_size = max_size
        self.fixed_per_class = fixed_per_class
        self.images: Dict[int, torch.Tensor] = {}
        self.logits: Dict[int, torch.Tensor] = {}
        self.task_ids: Dict[int, int] = {}

    @property
    def n_classes(self) -> int:
        return len(self.images)

    @property
    def total_images(self) -> int:
        return sum(t.shape[0] for t in self.images.values())

    def budget_per_class(self, total_classes: int) -> int:
        if self.fixed_per_class:
            return self.max_size
        return max(1, self.max_size // max(1, total_classes))

    def update(
        self,
        class_id: int,
        new_images: torch.Tensor,
        new_logits: torch.Tensor,
        total_classes: int,
        task_id: int,
    ) -> None:
        budget = self.budget_per_class(total_classes)

        if not self.fixed_per_class:
            for cid in list(self.images.keys()):
                if self.images[cid].shape[0] > budget:
                    self.images[cid] = self.images[cid][:budget].contiguous()
                    self.logits[cid] = self.logits[cid][:budget].contiguous()

        self.images[class_id] = new_images[:budget].detach().cpu().contiguous()
        self.logits[class_id] = new_logits[:budget].detach().cpu().contiguous()
        self.task_ids[class_id] = int(task_id)

        mode_str = "fixed" if self.fixed_per_class else "shared"
        print(
            f"  [Buffer] class {class_id} → {self.images[class_id].shape[0]} imgs "
            f"| total {self.total_images} imgs | {budget}/class ({mode_str} budget)"
        )

    def get_dataset(self) -> Optional[TensorDataset]:
        if not self.images:
            return None

        all_imgs = []
        all_labels = []
        all_tasks = []
        all_logits = []
        for class_id, imgs in self.images.items():
            n = imgs.shape[0]
            all_imgs.append(imgs)
            all_labels.append(torch.full((n,), class_id, dtype=torch.long))
            all_tasks.append(torch.full((n,), -1, dtype=torch.long))
            all_logits.append(self.logits[class_id])

        return TensorDataset(
            torch.cat(all_imgs, dim=0),
            torch.cat(all_labels, dim=0),
            torch.cat(all_tasks, dim=0),
            torch.cat(all_logits, dim=0),
        )

    def __repr__(self) -> str:
        counts = {cid: t.shape[0] for cid, t in self.images.items()}
        mode = "fixed" if self.fixed_per_class else "shared"
        return f"E2DBuffer(max_size={self.max_size}, mode={mode}, total={self.total_images}, counts={counts})"
